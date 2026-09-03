#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Build and exclusively stage one canonical live acceptance contract.

All runtime and run-identity pins are derived from the Pair and its sealed
receipts. Unverified lineage metadata that cannot be observed from those
artifacts is carried only for compatibility with the frozen collector-v2
schema; it is not acceptance authority. The resulting contract is validated
by the live evaluator before it is made visible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

BUILDER_INPUT_SCHEMA = "nemo-rl-strict-live-contract-unverified-lineage-metadata-v1"
TOPOLOGY_RE = re.compile(
    r"TP(?P<tp>[1-9][0-9]*)/CP(?P<cp>[1-9][0-9]*)/"
    r"PP(?P<pp>[1-9][0-9]*)/EP(?P<ep>[1-9][0-9]*)/"
    r"ETP(?P<etp>[1-9][0-9]*)/(?P<sp>SP|NO-SP)\Z"
)
HSG_SHARED_PROJECT_ANCESTOR = {
    "path": "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text",
    "uid": 0,
    "gid": 20330,
    "mode": 0o775,
}


class ContractBuildError(RuntimeError):
    """The supplied evidence cannot produce one valid live contract."""


def _trusted_sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractBuildError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _retained_source(path: Path, label: str) -> bytes:
    """Retain one stable regular source file without following its final link."""
    if not path.is_absolute():
        raise ContractBuildError(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
            raise ContractBuildError(f"{label} must be one nonempty regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContractBuildError(f"{label} ended while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContractBuildError(f"{label} grew while it was read")
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after):
            raise ContractBuildError(f"{label} changed while it was read")
        return b"".join(chunks)
    except OSError as error:
        raise ContractBuildError(f"cannot read {label}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_live_evaluator(expected_sha256: str) -> ModuleType:
    path = Path(__file__).resolve(strict=True).with_name("evaluate_strict_single_env_live.py")
    raw = _retained_source(path, "live evaluator")
    expected = _trusted_sha256(expected_sha256, "trusted evaluator source SHA-256")
    if _digest_bytes(raw) != expected:
        raise ContractBuildError("live evaluator differs from the trusted OOB SHA-256")
    name = "nemo_rl_strict_single_env_live_evaluator_for_builder"
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except Exception as error:
        raise ContractBuildError(f"cannot load retained live evaluator: {error}") from error
    return module


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ContractBuildError(f"{label} must be an exact JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractBuildError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, " f"extra={sorted(set(value) - expected)}"
        )


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_digest(document: Any) -> str:
    digest = getattr(document, "sha256", None)
    if type(digest) is not str:
        raise ContractBuildError("evidence document lacks an authenticated digest")
    return digest


def _unverified_lineage_metadata(live: ModuleType, document: Any) -> dict[str, Any]:
    live._require_canonical_document(document, "unverified lineage metadata", trailing_lf=False)
    value = _mapping(document.value, "unverified lineage metadata input")
    _exact_keys(value, {"schema", "unverified_lineage_metadata"}, "lineage metadata input")
    if value["schema"] != BUILDER_INPUT_SCHEMA:
        raise ContractBuildError("unexpected unverified lineage metadata input schema")
    metadata = copy.deepcopy(_mapping(value["unverified_lineage_metadata"], "unverified lineage metadata"))
    live._validate_unverified_lineage_metadata(metadata)
    return metadata


def _source_provenance(pair: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    source = pair["source"]
    commits = {
        "nemo_rl": source["head"],
        "megatron_bridge": source["bridge"]["head"],
        "megatron_lm": source["mcore"]["head"],
        "nemo_gym": source["gym"]["gitlink_commit"],
    }
    trees = {
        "nemo_rl": source["tree"],
        "megatron_bridge": source["bridge"]["tree"],
        "megatron_lm": source["mcore"]["tree"],
        "nemo_gym": source["gym"]["tree"],
    }
    return commits, trees


def _topology(pair: Mapping[str, Any], jobs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    match = TOPOLOGY_RE.fullmatch(pair["campaign"]["training_topology"])
    if match is None:
        raise ContractBuildError("Pair training topology has an unsupported shape")
    if jobs["off"]["hardware"] != jobs["on"]["hardware"]:
        raise ContractBuildError("OFF/ON hardware observations differ")
    hardware = jobs["off"]["hardware"]
    if jobs["off"]["gpus_per_node"] != jobs["on"]["gpus_per_node"]:
        raise ContractBuildError("OFF/ON GPU counts differ")
    if jobs["off"]["job_num_nodes"] != jobs["on"]["job_num_nodes"]:
        raise ContractBuildError("OFF/ON node counts differ")
    values = {name: int(match.group(name)) for name in ("tp", "cp", "pp", "ep", "etp")}
    nodes = jobs["off"]["job_num_nodes"]
    gpus = jobs["off"]["gpus_per_node"]
    return {
        "cluster_name": "HSG",
        "slurm_partition": pair["campaign"]["slurm"]["partition"],
        "gpu_model": hardware["gpu_model"],
        "nvidia_driver_version": hardware["driver_version"],
        "allocated_nodes": nodes,
        "gpus_per_node": gpus,
        "trainer_nodes": nodes,
        "trainer_gpus_per_node": gpus,
        "generation_nodes": nodes,
        "generation_gpus_per_node": pair["campaign"]["vllm_tp"],
        "tensor_parallel_size": values["tp"],
        "context_parallel_size": values["cp"],
        "sequence_parallel": match.group("sp") == "SP",
        "pipeline_parallel_size": values["pp"],
        "expert_parallel_size": values["ep"],
        "expert_tensor_parallel_size": values["etp"],
        "mtp_num_layers": pair["campaign"]["training_mtp"]["layers"],
    }


def _config(pair: Mapping[str, Any], topology: Mapping[str, Any], arm: str) -> dict[str, Any]:
    campaign = pair["campaign"]
    identity = pair["wandb"]
    pair_id = pair["pair_id"]
    environment = pair["selection"]["environment"]
    steps = campaign["steps"]
    epochs = campaign["epochs"]
    if type(steps) is not int or type(epochs) is not int or steps % epochs != 0:
        raise ContractBuildError("Pair steps do not form complete epochs")
    reward_policy = campaign["reward_and_advantage"]
    generation = campaign["generation"]
    return {
        "max_num_steps": steps,
        "epochs": epochs,
        "steps_per_epoch": steps // epochs,
        "fixture_rows": pair["artifacts"]["fixture"]["rows"],
        "num_prompts_per_step": campaign["prompts_per_step"],
        "num_generations_per_prompt": campaign["generations_per_prompt"],
        "seed": campaign["generation_seed_base"],
        "generation_seed_base": campaign["generation_seed_base"],
        "data_shuffle": campaign["data_shuffle"],
        "reward_scaling_enabled": reward_policy["reward_scaling"]["enabled"],
        "reward_shaping_enabled": reward_policy["reward_shaping"]["enabled"],
        "shared_prefix_mode": "observe" if arm == "off" else "train",
        "wandb_enabled": campaign["logging"]["wandb_enabled"],
        "tensorboard_enabled": campaign["logging"]["tensorboard_enabled"],
        "wandb_entity": identity["entity"],
        "wandb_project": identity["project"],
        "wandb_group": identity["group"]["value"],
        "wandb_run_name": identity["arms"][arm]["name"],
        "tensor_parallel_size": topology["tensor_parallel_size"],
        "context_parallel_size": topology["context_parallel_size"],
        "sequence_parallel": topology["sequence_parallel"],
        "pipeline_parallel_size": topology["pipeline_parallel_size"],
        "expert_parallel_size": topology["expert_parallel_size"],
        "expert_tensor_parallel_size": topology["expert_tensor_parallel_size"],
        "mtp_num_layers": topology["mtp_num_layers"],
        "mtp_use_repeated_layer": campaign["training_mtp"]["repeated_layer"],
        "mtp_detach_heads": campaign["training_mtp"]["detach_heads"],
        "mtp_loss_scaling_factor": campaign["training_mtp"]["loss_scale"],
        "slurm_partition": campaign["slurm"]["partition"],
        "slurm_account": campaign["slurm"]["account"],
        "max_new_tokens": generation["max_new_tokens"],
        "temperature": generation["temperature"],
        "top_k": generation["top_k"],
        "top_p": generation["top_p"],
    }


def _receipt_pin(document: Any, semantic_pins: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sha256": _document_digest(document),
        "semantic_pins": copy.deepcopy(dict(semantic_pins)),
    }


def _common_provenance(
    live: ModuleType,
    pair_document: Any,
    collector_path: Path,
    holdout: Mapping[str, Any],
    expected_terminal_collector_sha256: str,
    expected_wandb_collector_sha256: str,
) -> dict[str, str]:
    pair = pair_document.value
    selection = pair["selection"]
    deployment = pair["deployment"]
    source = pair["source"]
    terminal_collector_raw = _retained_source(
        Path(live.__file__).resolve(strict=True).with_name("collect_strict_single_env_terminal_jobs.py"),
        "terminal scheduler collector",
    )
    expected_terminal_collector = _trusted_sha256(
        expected_terminal_collector_sha256,
        "trusted terminal scheduler collector SHA-256",
    )
    if _digest_bytes(terminal_collector_raw) != expected_terminal_collector:
        raise ContractBuildError("terminal scheduler collector differs from the trusted OOB SHA-256")
    wandb_collector_raw = _retained_source(collector_path, "W&B collector")
    expected_wandb_collector = _trusted_sha256(
        expected_wandb_collector_sha256,
        "trusted W&B collector SHA-256",
    )
    if _digest_bytes(wandb_collector_raw) != expected_wandb_collector:
        raise ContractBuildError("W&B collector differs from the trusted OOB SHA-256")
    common = {
        "pair_manifest_sha256": _document_digest(pair_document),
        "acceptance_contract_sha256": "0" * 64,
        "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
        "model_tree_sha256": pair["artifacts"]["model"]["tree_sha256_v1"],
        "training_container_sha256": pair["artifacts"]["container"]["sha256"],
        "sandbox_container_sha256": pair["artifacts"]["sandbox_container"]["sha256"],
        "verifier_source_sha256": selection["gym_resources"]["verifier_source"]["sha256"],
        "reward_liveness_contract_sha256": holdout["contract_sha256"],
        "gym_config_sha256": selection["gym_resources"]["config"]["sha256"],
        "environment_recipe_sha256": selection["config"]["sha256"],
        "launcher_sha256": source["launcher_sha256"],
        "reward_semantics_contract_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        "nemo_runnable_manifest_sha256": deployment["nemo_runnable_manifest_sha256"],
        "bridge_runnable_manifest_sha256": deployment["bridge_runnable_manifest_sha256"],
        "mcore_runnable_manifest_sha256": deployment["mcore_runnable_manifest_sha256"],
        "deployment_ready_file_sha256": deployment["ready_file_sha256"],
        "deployment_ready_sha256": deployment["ready"],
        "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        "pair_campaign_sha256": pair["pair_campaign_sha256"],
        "runtime_tool_manifest_sha256": pair["runtime_tools"]["manifest"]["sha256"],
        "strict_pair_arm_wrapper_sha256": source["arm_wrapper_sha256"],
        "strict_pair_contract_sha256": source["contract_sha256"],
        "strict_pair_parent_wrapper_sha256": source["parent_wrapper_sha256"],
        "submission_contract_sha256": pair["scheduler_submission"]["contract"]["sha256"],
        "terminal_scheduler_collector_sha256": expected_terminal_collector,
        "wandb_exporter_sha256": expected_wandb_collector,
    }
    _exact_keys(common, set(live.COMMON_PROVENANCE_KEYS), "derived common provenance")
    return common


def build_contract(
    *,
    live: ModuleType,
    pair_document: Any,
    submission_document: Any,
    holdout_document: Any,
    execution_documents: Mapping[str, Any],
    job_documents: Mapping[str, Any],
    terminal_document: Any,
    unverified_lineage_document: Any,
    collector_path: Path,
    expected_terminal_collector_sha256: str,
    expected_wandb_collector_sha256: str,
) -> dict[str, Any]:
    pair = pair_document.value
    jobs = {arm: job_documents[arm].value for arm in ("off", "on")}
    holdout = holdout_document.value
    unverified_lineage = _unverified_lineage_metadata(live, unverified_lineage_document)
    topology = _topology(pair, jobs)
    commits, trees = _source_provenance(pair)
    common = _common_provenance(
        live,
        pair_document,
        collector_path,
        holdout,
        expected_terminal_collector_sha256,
        expected_wandb_collector_sha256,
    )
    arms = {}
    for arm in ("off", "on"):
        marker = execution_documents[arm].value
        job = jobs[arm]
        arms[arm] = {
            "runtime_environment_sha256": pair["slurm_export_boundary"]["arms"][arm]["sha256"],
            "shared_prefix_runtime_trace_sha256": marker["shared_prefix_runtime_trace_sha256"],
            "runtime_direction_receipt_sha256": _document_digest(execution_documents[arm]),
            "snapshot_manifest_sha256": pair["source"]["snapshots"][arm]["manifest_sha256"],
            "entrypoint_sha256": pair["source"]["entrypoint_sha256"],
            "wrapper_sha256": pair["source"]["job_wrapper"]["sha256"],
            "inner_ray_sha256": job["inner_ray_sha256"],
            "command_sha256": job["command_sha256"],
            "mounts_sha256": job["mounts_sha256"],
        }
    environment = pair["selection"]["environment"]
    pair_id = pair["pair_id"]
    terminal_value = _mapping(terminal_document.value, "terminal scheduler Pair receipt")
    terminal_semantic_keys = {
        "schema",
        "capture_method",
        "collector_sha256",
        "pair_id",
        "environment",
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "job_exit_receipt_sha256s",
        "submission_contract_sha256",
        "runtime_tool_manifest_sha256",
        "capture_sha256s",
        "composition_sha256",
    }
    _exact_keys(
        terminal_value,
        terminal_semantic_keys | {"captures"},
        "terminal scheduler Pair receipt",
    )
    contract = {
        "schema": live.CONTRACT_SCHEMA,
        "pair": {
            "pair_id": pair_id,
            "environment": environment,
            "entity": pair["wandb"]["entity"],
            "project": pair["wandb"]["project"],
            "group": pair["wandb"]["group"]["value"],
            "run_ids": {arm: pair["wandb"]["arms"][arm]["run_id"] for arm in ("off", "on")},
        },
        "campaign": copy.deepcopy(pair["campaign"]),
        "acceptance": copy.deepcopy(live.ACCEPTANCE),
        "provenance": {
            "common": common,
            "source_commits": commits,
            "source_git_trees": trees,
            # Frozen collector-v2 compatibility name. The assurance string and
            # evaluator guarantee that these opaque values are lineage-only;
            # no acceptance gate consumes them.
            "trusted_oob_declarations": unverified_lineage,
            "topology": topology,
            "arms": arms,
        },
        "configs": {arm: _config(pair, topology, arm) for arm in ("off", "on")},
        "holdout": {
            "receipt_sha256": _document_digest(holdout_document),
            "primary_reward_mean_min": holdout["frozen_reward_primary_mean_min"],
            "tail_reward_mean_min": holdout["frozen_reward_tail_mean_min"],
        },
        "receipts": {
            "shared_prefix_execution_marker_receipts": {
                arm: _receipt_pin(
                    execution_documents[arm],
                    {
                        "schema": live.EXECUTION_MARKER_RECEIPT_SCHEMA,
                        "scope": "shared_prefix_physical_execution",
                        "marker_semantics": live.EXECUTION_MARKER_SEMANTICS,
                        "status": "PASS",
                        "pair_id": pair_id,
                        "environment": environment,
                        "arm": arm,
                        "shared_prefix_mode": ("observe" if arm == "off" else "train"),
                    },
                )
                for arm in ("off", "on")
            },
            "strict_job_exit_receipts": {
                arm: _receipt_pin(
                    job_documents[arm],
                    {
                        "schema": live.JOB_RECEIPT_SCHEMA,
                        "phase": "EXIT",
                        "post_verified": True,
                        "driver_exit_code": 0,
                        "pair_id": pair_id,
                        "environment": environment,
                        "arm": arm,
                    },
                )
                for arm in ("off", "on")
            },
            "terminal_scheduler_pair_receipt": _receipt_pin(
                terminal_document,
                {key: copy.deepcopy(terminal_value[key]) for key in terminal_semantic_keys},
            ),
        },
        "verifier_metric": live.VERIFIER_METRIC_BY_ENVIRONMENT[environment],
    }
    common["acceptance_contract_sha256"] = live._acceptance_contract_payload_sha256(contract)
    live.validate_contract(contract)
    manifest = live._validate_pair_manifest(pair_document, contract)
    artifacts = {
        "pair_manifest": pair_document,
        "submission_receipt": submission_document,
        "holdout": holdout_document,
        "off_execution": execution_documents["off"],
        "on_execution": execution_documents["on"],
        "off_job_exit": job_documents["off"],
        "on_job_exit": job_documents["on"],
        "terminal_scheduler": terminal_document,
    }
    submitted = live._authenticate_submission_receipt_bytes(
        artifacts,
        _document_digest(submission_document),
        manifest,
        contract,
    )
    exited = live._validate_execution_receipts(contract, artifacts, manifest)
    if submitted != exited:
        raise ContractBuildError("submission and EXIT receipts bind different scheduler jobs")
    live._validate_terminal_scheduler_receipt(
        contract,
        artifacts,
        manifest,
        submitted,
        exited,
    )
    live._validate_holdout(holdout_document, contract)
    return contract


def _canonical_contract_bytes(live: ModuleType, contract: Mapping[str, Any]) -> bytes:
    return live._canonical_json_bytes(contract, "live acceptance contract")


def _trusted_ancestor_metadata(path: Path, metadata: os.stat_result) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    shared_exception = (
        str(path) == HSG_SHARED_PROJECT_ANCESTOR["path"]
        and metadata.st_uid == HSG_SHARED_PROJECT_ANCESTOR["uid"]
        and metadata.st_gid == HSG_SHARED_PROJECT_ANCESTOR["gid"]
        and mode == HSG_SHARED_PROJECT_ANCESTOR["mode"]
    )
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid in {0, os.geteuid()}
        and (mode & 0o022 == 0 or shared_exception)
    )


def _output_parent(path: Path) -> tuple[int, str]:
    if not path.is_absolute() or ".." in path.parts:
        raise ContractBuildError("output path must be lexical-canonical absolute")
    absolute = Path(os.path.normpath(str(path)))
    if absolute != path or absolute.name in {"", ".", ".."}:
        raise ContractBuildError("output path must be lexical-canonical absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        if not _trusted_ancestor_metadata(Path("/"), os.fstat(descriptor)):
            raise ContractBuildError("output root ancestry differs from the closed path policy")
        traversed = Path("/")
        components = absolute.parts[1:-1]
        for index, component in enumerate(components):
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            metadata = os.fstat(descriptor)
            traversed /= component
            if not _trusted_ancestor_metadata(traversed, metadata):
                raise ContractBuildError("output ancestry differs from the closed path policy")
            if index == len(components) - 1 and (
                metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ContractBuildError("output immediate parent must be current-user mode 0700")
        return descriptor, absolute.name
    except OSError as error:
        os.close(descriptor)
        raise ContractBuildError(f"cannot authenticate output ancestry: {error}") from error
    except BaseException:
        os.close(descriptor)
        raise


def _stage_exclusive(path: Path, raw: bytes) -> None:
    if not raw:
        raise ContractBuildError("acceptance contract cannot be empty")
    parent_fd, target = _output_parent(path)
    temporary = f".{target}.candidate-{uuid.uuid4().hex}"
    descriptor: int | None = None
    published = False
    complete = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o400,
            dir_fd=parent_fd,
        )
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise ContractBuildError("short write while staging acceptance contract")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or metadata.st_size != len(raw)
        ):
            raise ContractBuildError("staged acceptance contract metadata differs")
        os.link(
            temporary,
            target,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        if os.fstat(descriptor).st_nlink != 2:
            raise ContractBuildError("published acceptance contract link count differs")
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        if os.fstat(descriptor).st_nlink != 1:
            raise ContractBuildError("sealed acceptance contract link count differs")
        os.fsync(parent_fd)
        complete = True
    except FileExistsError as error:
        raise ContractBuildError(f"output already exists: {path}") from error
    except OSError as error:
        raise ContractBuildError(f"cannot stage acceptance contract: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except OSError:
            pass
        if published and not complete:
            try:
                os.unlink(target, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-terminal-collector-sha256", required=True)
    parser.add_argument("--expected-wandb-collector-sha256", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--holdout-receipt", type=Path, required=True)
    parser.add_argument("--off-execution-receipt", type=Path, required=True)
    parser.add_argument("--on-execution-receipt", type=Path, required=True)
    parser.add_argument("--off-job-exit-receipt", type=Path, required=True)
    parser.add_argument("--on-job-exit-receipt", type=Path, required=True)
    parser.add_argument("--terminal-scheduler-receipt", type=Path, required=True)
    parser.add_argument("--unverified-lineage-metadata", type=Path, required=True)
    parser.add_argument(
        "--collector",
        type=Path,
        default=Path(__file__).with_name("collect_strict_single_env_wandb.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        live = _load_live_evaluator(args.expected_evaluator_sha256)
    except (ContractBuildError, OSError, ImportError) as error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 2
    try:
        pair = live.load_document(args.pair_manifest, "Pair manifest")
        submission = live.load_document(args.submission_receipt, "pair submission receipt")
        holdout = live.load_document(args.holdout_receipt, "holdout receipt")
        executions = {
            "off": live.load_document(args.off_execution_receipt, "OFF execution receipt"),
            "on": live.load_document(args.on_execution_receipt, "ON execution receipt"),
        }
        jobs = {
            "off": live.load_document(args.off_job_exit_receipt, "OFF job EXIT receipt"),
            "on": live.load_document(args.on_job_exit_receipt, "ON job EXIT receipt"),
        }
        terminal = live.load_document(
            args.terminal_scheduler_receipt,
            "terminal scheduler Pair receipt",
        )
        unverified_lineage = live.load_document(
            args.unverified_lineage_metadata,
            "unverified lineage metadata input",
        )
        contract = build_contract(
            live=live,
            pair_document=pair,
            submission_document=submission,
            holdout_document=holdout,
            execution_documents=executions,
            job_documents=jobs,
            terminal_document=terminal,
            unverified_lineage_document=unverified_lineage,
            collector_path=args.collector.resolve(strict=True),
            expected_terminal_collector_sha256=args.expected_terminal_collector_sha256,
            expected_wandb_collector_sha256=args.expected_wandb_collector_sha256,
        )
        raw = _canonical_contract_bytes(live, contract)
        _stage_exclusive(args.output, raw)
    except (
        ContractBuildError,
        live.EvidenceError,
        OSError,
        KeyError,
        TypeError,
    ) as error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "schema": live.CONTRACT_SCHEMA,
                "path": str(args.output.resolve(strict=True)),
                "sha256": _digest_bytes(raw),
                "payload_sha256": contract["provenance"]["common"]["acceptance_contract_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
