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

"""Replay one captured citation K=4 cohort through dense and shared-prefix Nano.

This is a fail-closed, opt-in GPU diagnostic.  ``run-arm`` is launched under
``torch.distributed.run`` with four ranks.  The launcher starts a fresh process
group for each arm so every arm imports the same checkpoint and begins with an
empty optimizer state.

The two authoritative arms are:

* ``off``: conventional dense execution (shared-prefix ``observe`` mode); and
* ``on``: production shared-prefix state-fork execution (``train`` mode).

Both perform a streamed NeMo-RL optimizer step with TP2/CP2/SP/EP4/ETP1/PP1 and
five repeated, detached MTP heads.  Evidence includes selected-token logprobs,
RL and MTP losses, finalized-gradient sketches grouped by physical model
family, sampled optimizer updates, and post-update selected-token logprobs.

``on-mamba-replay`` is explicitly diagnostic: it replaces the CP Mamba
state-fork helper with the existing decomposed replay-prefix implementation in
that one process. ``on-mamba-packed-replay`` instead reconstructs dense branches
and invokes Mamba's unchanged packed fused kernel. Both record only the
pre-training selected-token logprobs; neither changes production source or
performs an optimizer step.

``compare`` consumes the sealed per-rank JSON files after all three processes
exit.  It never treats the captured rewards as fresh reward-distribution
evidence; they are immutable inputs for a numerical parity check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import ray
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nemo_rl.models.policy import PolicyConfig, SequencePackingConfig

SCHEMA = "nemorl-shared-prefix-captured-citation-parity-v4"
WORLD_SIZE = 4
TP_SIZE = 2
CP_SIZE = 2
EP_SIZE = 4
ETP_SIZE = 1
PP_SIZE = 1
MTP_LAYERS = 5
BATCH_SIZE = 4
MAX_TOTAL_SEQUENCE_LENGTH = 4096
PACKING_TOKENS = 16384
LOGPROB_CHUNK_SIZE = 512
GRADIENT_SAMPLE_COUNT = 33
MASTER_SAMPLE_COUNT = 257
HASH_CHUNK_ELEMENTS = 4 * 1024 * 1024
LOGPROB_FILTER_LIMIT = 2.0
LOGPROB_STRICT_LIMIT = 1.05
REQUIRED_GRADIENT_FAMILIES = ("attention", "mamba", "moe", "mtp")
EXPECTED_MTP_LOSS_KEYS = tuple(f"mtp_{depth}_loss" for depth in range(1, 6))
EXPECTED_MTP_ACCEPTANCE_KEYS = tuple(f"mtp_{depth}_acceptance_rate" for depth in range(1, 6))
DIAGNOSTIC_ARMS = frozenset(("on-mamba-replay", "on-mamba-packed-replay"))


class ParityError(RuntimeError):
    """The captured input, runtime, or numerical evidence is invalid."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParityError(f"captured JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ParityError(f"captured JSON contains non-finite constant {value!r}")


def _singleton(row: dict[str, Any], key: str, *, row_index: int) -> Any:
    if key not in row:
        raise ParityError(f"captured row {row_index} is missing {key!r}")
    value = row[key]
    if not isinstance(value, list) or len(value) != 1:
        raise ParityError(f"captured row {row_index} field {key!r} must be a singleton list")
    return value[0]


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ParityError(f"{label} must be finite, got {value!r}")
    return result


def _int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParityError(f"{label} must be an integer, got {value!r}")
    return value


def _number_list(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list):
        raise ParityError(f"{label} must be a list")
    return [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ParityError(f"{label} must be a list")
    return [_int(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _require_regular_readonly_file(path: Path, *, expected_sha256: str) -> str:
    if not path.is_absolute():
        raise ParityError(f"captured batch path must be absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ParityError(f"cannot resolve captured batch {path}: {error}") from error
    if resolved != path or path.is_symlink():
        raise ParityError(f"captured batch must be a canonical non-symlink path: {path}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ParityError(f"captured batch is not a regular file: {path}")
    if mode & 0o222:
        raise ParityError(f"captured batch must have no write bits: {path}")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ParityError(
            "captured batch digest mismatch: " f"expected={expected_sha256} actual={actual_sha256} path={path}"
        )
    return actual_sha256


def _read_captured_rows(path: Path, *, expected_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    digest = _require_regular_readonly_file(path, expected_sha256=expected_sha256)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_no_duplicate_keys,
                    parse_constant=_reject_constant,
                )
            except json.JSONDecodeError as error:
                raise ParityError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ParityError(f"captured row at {path}:{line_number} is not an object")
            rows.append(value)
    if len(rows) != BATCH_SIZE:
        raise ParityError(f"captured cohort must contain exactly K=4 rows, got {len(rows)}")

    rows.sort(key=lambda row: _int(row.get("idx"), label="captured row idx"))
    if [row["idx"] for row in rows] != list(range(BATCH_SIZE)):
        raise ParityError("captured row indices must be exactly 0,1,2,3")

    required = {
        "advantages",
        "generation_logprobs",
        "input_lengths",
        "prev_logprobs",
        "prompt_ids",
        "rewards",
        "sample_loss_mask",
        "shared_prefix_group_id",
        "shared_prefix_prompt_lengths",
        "token_ids",
        "token_loss_mask",
    }
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ParityError(f"captured row {index} is missing {sorted(missing)}")
        token_ids = _int_list(_singleton(row, "token_ids", row_index=index), label=f"row {index} token_ids")
        prompt_ids = _int_list(
            _singleton(row, "prompt_ids", row_index=index),
            label=f"row {index} prompt_ids",
        )
        token_mask = _number_list(
            _singleton(row, "token_loss_mask", row_index=index),
            label=f"row {index} token_loss_mask",
        )
        advantages = _number_list(
            _singleton(row, "advantages", row_index=index),
            label=f"row {index} advantages",
        )
        prev_logprobs = _number_list(
            _singleton(row, "prev_logprobs", row_index=index),
            label=f"row {index} prev_logprobs",
        )
        generation_logprobs = _number_list(
            _singleton(row, "generation_logprobs", row_index=index),
            label=f"row {index} generation_logprobs",
        )
        input_length = _int(
            _singleton(row, "input_lengths", row_index=index),
            label=f"row {index} input_length",
        )
        prompt_length = _int(
            _singleton(row, "shared_prefix_prompt_lengths", row_index=index),
            label=f"row {index} prompt_length",
        )
        vectors = {
            "token_mask": token_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
        }
        for name, vector in vectors.items():
            if len(vector) != len(token_ids):
                raise ParityError(f"row {index} {name} width {len(vector)} != token width {len(token_ids)}")
        if not 0 < prompt_length < input_length <= len(token_ids):
            raise ParityError(
                f"row {index} has invalid prompt/input/token lengths "
                f"{prompt_length}/{input_length}/{len(token_ids)}"
            )
        if prompt_ids != token_ids[:prompt_length]:
            raise ParityError(f"row {index} prompt_ids do not match the token prefix")
        if any(value != 0.0 for value in token_mask[:prompt_length]):
            raise ParityError(f"row {index} applies policy loss inside the prompt")
        if not any(value > 0.0 for value in token_mask[prompt_length:input_length]):
            raise ParityError(f"row {index} has no selected completion token")
        if any(value != 0.0 for value in token_mask[input_length:]):
            raise ParityError(f"row {index} applies policy loss after input_length")
        group_id = _singleton(row, "shared_prefix_group_id", row_index=index)
        if not isinstance(group_id, str) or not group_id:
            raise ParityError(f"row {index} has an invalid shared-prefix group id")
        parsed.append(
            {
                "token_ids": token_ids,
                "prompt_ids": prompt_ids,
                "token_mask": token_mask,
                "advantages": advantages,
                "prev_logprobs": prev_logprobs,
                "generation_logprobs": generation_logprobs,
                "input_length": input_length,
                "prompt_length": prompt_length,
                "sample_mask": _finite_float(
                    _singleton(row, "sample_loss_mask", row_index=index),
                    label=f"row {index} sample_loss_mask",
                ),
                "reward": _finite_float(
                    _singleton(row, "rewards", row_index=index),
                    label=f"row {index} reward",
                ),
                "group_id": group_id,
            }
        )

    widths = {len(row["token_ids"]) for row in parsed}
    prompt_lengths = {row["prompt_length"] for row in parsed}
    prompt_tokens = {tuple(row["prompt_ids"]) for row in parsed}
    group_ids = {row["group_id"] for row in parsed}
    if len(widths) != 1 or len(prompt_lengths) != 1 or len(prompt_tokens) != 1 or len(group_ids) != 1:
        raise ParityError("captured K=4 rows must have one width, exact prompt, prompt length, and group id")
    rewards = [row["reward"] for row in parsed]
    if len(set(rewards)) < 2:
        raise ParityError(f"captured citation rewards are degenerate: {rewards}")
    if any(row["sample_mask"] <= 0.0 for row in parsed):
        raise ParityError("captured citation cohort contains a masked sample")
    selected_advantages = [
        advantage
        for row in parsed
        for advantage, mask in zip(row["advantages"], row["token_mask"], strict=True)
        if mask > 0.0
    ]
    if not selected_advantages or max(map(abs, selected_advantages)) == 0.0:
        raise ParityError("captured citation cohort has no nonzero selected-token advantage")

    summary = {
        "sha256": digest,
        "rows": len(parsed),
        "width": next(iter(widths)),
        "input_lengths": [row["input_length"] for row in parsed],
        "prompt_length": next(iter(prompt_lengths)),
        "rewards": rewards,
        "selected_tokens": int(sum(sum(row["token_mask"]) for row in parsed)),
    }
    return parsed, summary


def _build_batch(rows: Sequence[dict[str, Any]]) -> Any:
    from nemo_rl.data.packing.shared_prefix_metadata import (
        SHARED_PREFIX_GROUP_ID,
        SHARED_PREFIX_PROMPT_LENGTHS,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    width = len(rows[0]["token_ids"])
    input_lengths = torch.tensor([row["input_length"] for row in rows], dtype=torch.int32)
    input_ids = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long)
    if input_ids.shape != (BATCH_SIZE, width):
        raise ParityError(f"unexpected captured input tensor shape {tuple(input_ids.shape)}")
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": torch.arange(width).unsqueeze(0) < input_lengths.unsqueeze(1),
            "token_mask": torch.tensor([row["token_mask"] for row in rows], dtype=torch.float32),
            "sample_mask": torch.tensor([row["sample_mask"] for row in rows], dtype=torch.float32),
            "advantages": torch.tensor([row["advantages"] for row in rows], dtype=torch.float32),
            "prev_logprobs": torch.tensor([row["prev_logprobs"] for row in rows], dtype=torch.float32),
            "generation_logprobs": torch.tensor([row["generation_logprobs"] for row in rows], dtype=torch.float32),
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor([row["prompt_length"] for row in rows], dtype=torch.int32),
            SHARED_PREFIX_GROUP_ID: [row["group_id"] for row in rows],
        }
    )


def _require_dense_packing_metadata(batch: Any, *, stage: str) -> None:
    """Fail before model construction if driver-side sequence packing was skipped."""
    micro_batch_indices = batch.micro_batch_indices
    micro_batch_lengths = batch.micro_batch_lengths
    elem_counts_per_gb = batch.elem_counts_per_gb
    if (
        not isinstance(micro_batch_indices, list)
        or len(micro_batch_indices) != 1
        or not isinstance(micro_batch_lengths, list)
        or len(micro_batch_lengths) != 1
        or not isinstance(elem_counts_per_gb, list)
        or len(elem_counts_per_gb) != 1
    ):
        raise ParityError(f"dense {stage} preparation did not materialize one DP=1 " "microbatch-metadata chunk")
    ranges = micro_batch_indices[0]
    lengths = micro_batch_lengths[0]
    if not isinstance(ranges, list) or not ranges or len(ranges) != len(lengths):
        raise ParityError(f"dense {stage} preparation produced invalid microbatch metadata")
    cursor = 0
    for microbatch_index, bounds in enumerate(ranges):
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or bounds[0] != cursor
            or not isinstance(bounds[1], int)
            or bounds[1] <= cursor
        ):
            raise ParityError(f"dense {stage} microbatch {microbatch_index} has invalid bounds {bounds!r}")
        length = lengths[microbatch_index]
        if not isinstance(length, int) or length <= 0:
            raise ParityError(f"dense {stage} microbatch {microbatch_index} has invalid length {length!r}")
        cursor = bounds[1]
    if cursor != batch.size or elem_counts_per_gb != [batch.size]:
        raise ParityError(
            f"dense {stage} metadata covers {cursor}/{batch.size} rows with "
            f"elem_counts_per_gb={elem_counts_per_gb!r}"
        )


def _sequence_packing_config(policy_config: "PolicyConfig") -> "SequencePackingConfig":
    raw_config = policy_config.get("sequence_packing")
    if raw_config is None or raw_config["enabled"] is not True:
        raise ParityError("captured parity preparation requires enabled sequence packing")
    return cast("SequencePackingConfig", raw_config)


def _stage_bin_capacity(
    sequence_packing: "SequencePackingConfig",
    *,
    stage: Literal["logprob", "train"],
) -> int:
    if stage == "logprob":
        if "logprob_mb_tokens" not in sequence_packing:
            raise ParityError("policy.sequence_packing.logprob_mb_tokens is required")
        bin_capacity = int(sequence_packing["logprob_mb_tokens"])
    else:
        bin_capacity = int(sequence_packing["train_mb_tokens"])
    if bin_capacity <= 0:
        raise ParityError(f"policy.sequence_packing.{stage}_mb_tokens must be positive")
    return bin_capacity


def _prepare_batch_for_worker(
    source_batch: Any,
    *,
    policy_config: "PolicyConfig",
    shared_prefix_mode: Literal["observe", "train"],
    stage: Literal["logprob", "train"],
) -> tuple[Any, list[int] | None]:
    """Mirror ``LMPolicy`` DP=1 sharding and preparation for one worker call.

    The worker API consumes an already-sharded batch.  Dense sequence packing
    therefore needs the metadata normally attached by
    ``Policy._shard_for_logprob`` / ``Policy._shard_for_train``.  Shared-prefix
    train mode instead mirrors ``Policy._shard_shared_prefix_data`` and stamps
    the conservative, stage-capacity-specific execution-slot plan.

    Returns:
        The sole DP shard and, when packing changed row order, the original row
        index for every row in worker-output order.
    """
    from nemo_rl.data.packing.shared_prefix_metadata import (
        SHARED_PREFIX_EXECUTION_SLOT,
        SHARED_PREFIX_GROUP_ID,
        plan_fixed_execution_slots,
        plan_group_coherent_shards,
    )
    from nemo_rl.distributed.batched_data_dict import SlicedDataDict

    sequence_packing = _sequence_packing_config(policy_config)
    bin_capacity = _stage_bin_capacity(sequence_packing, stage=stage)
    batch_size = None if stage == "logprob" else BATCH_SIZE

    if shared_prefix_mode == "train":
        if SHARED_PREFIX_EXECUTION_SLOT in source_batch:
            raise ParityError("source batch contains reserved shared-prefix execution slots")
        group_ids = list(source_batch[SHARED_PREFIX_GROUP_ID])
        sequence_lengths = [int(length) for length in source_batch["input_lengths"].detach().cpu().tolist()]
        slot_plan = plan_fixed_execution_slots(
            group_ids=group_ids,
            sequence_lengths=sequence_lengths,
            bin_capacity=bin_capacity,
            batch_size=batch_size,
            sequence_length_pad_multiple=policy_config["make_sequence_length_divisible_by"],
        )
        shard_plan = plan_group_coherent_shards(
            group_ids=group_ids,
            sequence_lengths=sequence_lengths,
            num_shards=1,
            batch_size=batch_size,
        )
        indices = list(shard_plan.shard_indices[0])
        shard = source_batch.select_indices(indices)
        shard[SHARED_PREFIX_EXECUTION_SLOT] = torch.tensor(
            [slot_plan.row_slot_ids[index] for index in indices],
            dtype=torch.long,
        )
        prepared = SlicedDataDict(shard.get_dict())
        slots = prepared[SHARED_PREFIX_EXECUTION_SLOT]
        if not isinstance(slots, torch.Tensor) or slots.shape != (source_batch.size,):
            raise ParityError(f"shared-prefix {stage} preparation produced invalid execution slots")
        expected_slots = [slot_plan.row_slot_ids[index] for index in indices]
        actual_slots = [int(value) for value in slots.detach().cpu().tolist()]
        if actual_slots != expected_slots:
            raise ParityError(
                f"shared-prefix {stage} execution-slot values differ from the "
                f"driver plan: expected={expected_slots} actual={actual_slots}"
            )
        execution_units = set(zip(prepared[SHARED_PREFIX_GROUP_ID], actual_slots, strict=True))
        if (
            slot_plan.units_per_group_by_chunk != (1,)
            or tuple(actual_slots) != (0, 0, 0, 0)
            or len(execution_units) != 1
        ):
            raise ParityError(
                f"captured K=4 {stage} batch must form one real shared-prefix "
                f"execution unit, got units_per_group_by_chunk="
                f"{slot_plan.units_per_group_by_chunk!r} execution_units="
                f"{sorted(execution_units)!r}"
            )
        source_order = (
            list(shard_plan.rank_order_permutation) if shard_plan.rank_order_permutation is not None else None
        )
        return prepared, source_order

    sequence_packing_args: dict[str, Any] = {
        "algorithm": sequence_packing["algorithm"],
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "max_tokens_per_microbatch": bin_capacity,
        "sequence_length_pad_multiple": policy_config["make_sequence_length_divisible_by"],
    }
    microbatch_order = sequence_packing.get("microbatch_order")
    if microbatch_order is not None:
        sequence_packing_args["microbatch_order"] = microbatch_order
    sharded, source_order = source_batch.shard_by_batch_size(
        1,
        batch_size=batch_size,
        # pyrefly: ignore  # bad-argument-type
        sequence_packing_args=sequence_packing_args,
    )
    if len(sharded) != 1:
        raise ParityError(f"DP=1 preparation returned {len(sharded)} shards")
    prepared = sharded[0]
    _require_dense_packing_metadata(prepared, stage=stage)
    return prepared, list(source_order)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _require_dense_fixture_plan(
    source: Any,
    prepared: Any,
    source_order: list[int] | None,
    *,
    policy_config: "PolicyConfig",
    stage: Literal["logprob", "train"],
) -> dict[str, Any]:
    """Prove this fixture took the exact one-bin dense LMPolicy preparation."""
    if source_order is None:
        raise ParityError(f"dense {stage} sequence packing did not return source order")
    source_index = torch.tensor(source_order, dtype=torch.long)
    expected_input_ids = source["input_ids"].index_select(0, source_index)
    if not torch.equal(prepared["input_ids"], expected_input_ids):
        raise ParityError(f"dense {stage} prepared input IDs do not follow the returned source order")
    padding_multiple = int(policy_config["make_sequence_length_divisible_by"])
    padded_tokens = sum(
        _round_up(int(length), padding_multiple) for length in source["input_lengths"].detach().cpu().tolist()
    )
    capacity = _stage_bin_capacity(
        _sequence_packing_config(policy_config),
        stage=stage,
    )
    expected_indices = [[[0, BATCH_SIZE]]]
    expected_lengths = [[padded_tokens]]
    if (
        prepared.micro_batch_indices != expected_indices
        or prepared.micro_batch_lengths != expected_lengths
        or prepared.elem_counts_per_gb != [BATCH_SIZE]
        or padded_tokens > capacity
    ):
        raise ParityError(
            f"dense {stage} fixture must form one real packed microbatch: "
            f"indices={prepared.micro_batch_indices!r} "
            f"lengths={prepared.micro_batch_lengths!r} "
            f"padded_tokens={padded_tokens} capacity={capacity}"
        )
    return {
        "source_order": source_order,
        "micro_batch_indices": prepared.micro_batch_indices,
        "micro_batch_lengths": prepared.micro_batch_lengths,
        "padded_tokens": padded_tokens,
        "capacity": capacity,
    }


def _require_shared_fixture_plan(
    prepared: Any,
    *,
    policy_config: "PolicyConfig",
    stage: Literal["logprob", "train"],
) -> dict[str, Any]:
    """Run the worker-side planner and reject dense fallback for captured K=4."""
    from nemo_rl.data.packing.shared_prefix_metadata import SHARED_PREFIX_EXECUTION_SLOT
    from nemo_rl.models.megatron.data import (
        _plan_prescribed_shared_prefix_execution_units,
    )

    padding_multiple = int(policy_config["make_sequence_length_divisible_by"])
    capacity = _stage_bin_capacity(
        _sequence_packing_config(policy_config),
        stage=stage,
    )
    units = _plan_prescribed_shared_prefix_execution_units(
        prepared,
        cfg=policy_config,
        bin_capacity=capacity,
        padding_multiple=padding_multiple,
    )
    if len(units) != 1:
        raise ParityError(f"captured K=4 {stage} plan produced {len(units)} execution units")
    unit = units[0]
    input_lengths = [int(length) for length in prepared["input_lengths"].detach().cpu().tolist()]
    prompt_lengths = [int(length) for length in prepared["shared_prefix_prompt_lengths"].detach().cpu().tolist()]
    if len(set(prompt_lengths)) != 1:
        raise ParityError(f"captured K=4 {stage} plan has unequal prompt lengths")
    prompt_length = prompt_lengths[0]
    expected_physical_length = prompt_length + sum(
        _round_up(input_length, padding_multiple) - prompt_length for input_length in input_lengths
    )
    expected_layout_order = tuple(
        sorted(
            range(BATCH_SIZE),
            key=lambda row: (
                -(_round_up(input_lengths[row], padding_multiple) - prompt_length),
                -(input_lengths[row] - prompt_length),
                row,
            ),
        )
    )
    slots = tuple(int(value) for value in prepared[SHARED_PREFIX_EXECUTION_SLOT].detach().cpu().tolist())
    layout = unit.shared_layout
    if (
        tuple(sorted(unit.row_indices)) != tuple(range(BATCH_SIZE))
        or unit.row_indices != expected_layout_order
        or layout is None
        or layout.row_indices != expected_layout_order
        or slots != (0, 0, 0, 0)
        or unit.physical_length != expected_physical_length
        or layout.physical_total_length != expected_physical_length
        or expected_physical_length > capacity
    ):
        raise ParityError(
            f"captured K=4 {stage} must produce one non-fallback shared star: "
            f"rows={unit.row_indices!r} expected_rows={expected_layout_order!r} "
            f"shared={layout is not None} slots={slots!r} "
            f"physical_length={unit.physical_length} "
            f"expected_physical_length={expected_physical_length} capacity={capacity}"
        )
    return {
        "execution_units": len(units),
        "row_indices": list(unit.row_indices),
        "slot_ids": list(slots),
        "shared_layout": True,
        "physical_length": unit.physical_length,
        "capacity": capacity,
    }


def _require_worker_microbatch_plan(
    prepared: Any,
    *,
    policy_config: "PolicyConfig",
    shared_prefix_mode: Literal["observe", "train"],
    stage: Literal["logprob", "train"],
) -> None:
    """Execute the worker function that rejected job 6780368, without iteration."""
    from nemo_rl.models.megatron.data import get_microbatch_iterator

    capacity = _stage_bin_capacity(
        _sequence_packing_config(policy_config),
        stage=stage,
    )
    _iterator, num_microbatches, _mbs, _seq_len, _padded_seq_len = get_microbatch_iterator(
        prepared,
        policy_config,
        1,
        straggler_timer=cast(Any, None),
        shared_prefix_bin_capacity=(capacity if shared_prefix_mode == "train" else None),
    )
    if num_microbatches != 1:
        raise ParityError(
            f"{shared_prefix_mode} {stage} worker planned {num_microbatches} " "microbatches instead of one"
        )


def _preflight_batch_preparation(
    rows: Sequence[dict[str, Any]],
    *,
    policy_config: "PolicyConfig",
    shared_prefix_mode: Literal["observe", "train"],
) -> dict[str, Any]:
    """Exercise both stage-specific preparation paths without constructing a model."""
    stages: dict[str, Any] = {}
    shared = shared_prefix_mode == "train"
    for stage in ("logprob", "train"):
        source = _build_batch(rows)
        if shared:
            prepared, source_order = _prepare_batch_for_worker(
                source,
                policy_config=policy_config,
                shared_prefix_mode="train",
                stage=stage,
            )
        else:
            prepared, source_order = _prepare_batch_for_worker(
                source,
                policy_config=policy_config,
                shared_prefix_mode="observe",
                stage=stage,
            )
        if source.size != BATCH_SIZE or prepared.size != BATCH_SIZE:
            raise ParityError(f"{shared_prefix_mode} {stage} preparation changed K=4 batch size")
        if source_order is not None and sorted(source_order) != list(range(BATCH_SIZE)):
            raise ParityError(f"{shared_prefix_mode} {stage} preparation returned invalid row order {source_order}")
        plan = (
            _require_shared_fixture_plan(
                prepared,
                policy_config=policy_config,
                stage=stage,
            )
            if shared
            else _require_dense_fixture_plan(
                source,
                prepared,
                source_order,
                policy_config=policy_config,
                stage=stage,
            )
        )
        if shared:
            _require_worker_microbatch_plan(
                prepared,
                policy_config=policy_config,
                shared_prefix_mode="train",
                stage=stage,
            )
        else:
            _require_worker_microbatch_plan(
                prepared,
                policy_config=policy_config,
                shared_prefix_mode="observe",
                stage=stage,
            )
        stages[stage] = {
            "capacity": _stage_bin_capacity(
                _sequence_packing_config(policy_config),
                stage=stage,
            ),
            "microbatches": (
                len(prepared.micro_batch_indices[0]) if prepared.micro_batch_indices is not None else None
            ),
            "source_order": source_order,
            "plan": plan,
            "worker_num_microbatches": 1,
        }
    return stages


def _build_policy_and_loss_config(
    model_path: str, *, repo_root: Path, shared_prefix_mode: str
) -> tuple["PolicyConfig", Any, str]:
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.loss import ClippedPGLossConfig
    from nemo_rl.models.policy import PolicyConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    recipe = repo_root / "examples/nemo_gym/nemotron-3.5-nano/rlvr.yaml"
    if not recipe.is_file() or recipe.is_symlink() or recipe.resolve(strict=True) != recipe:
        raise ParityError(f"Nano recipe must be an exact canonical file: {recipe}")
    register_omegaconf_resolvers()
    config = load_config(recipe)
    overrides: dict[str, Any] = {
        "policy.model_name": model_path,
        "policy.tokenizer.name": model_path,
        "policy.train_global_batch_size": BATCH_SIZE,
        "policy.train_micro_batch_size": 1,
        "policy.logprob_batch_size": 1,
        "policy.max_total_sequence_length": MAX_TOTAL_SEQUENCE_LENGTH,
        "policy.logprob_chunk_size": LOGPROB_CHUNK_SIZE,
        "policy.shared_prefix_training.mode": shared_prefix_mode,
        "policy.sequence_packing.train_mb_tokens": PACKING_TOKENS,
        "policy.sequence_packing.logprob_mb_tokens": PACKING_TOKENS,
        "policy.megatron_cfg.tensor_model_parallel_size": TP_SIZE,
        "policy.megatron_cfg.context_parallel_size": CP_SIZE,
        "policy.megatron_cfg.sequence_parallel": True,
        "policy.megatron_cfg.expert_model_parallel_size": EP_SIZE,
        "policy.megatron_cfg.expert_tensor_parallel_size": ETP_SIZE,
        "policy.megatron_cfg.pipeline_model_parallel_size": PP_SIZE,
        "policy.megatron_cfg.train_iters": 1,
        "policy.megatron_cfg.activation_checkpointing": True,
        "policy.megatron_cfg.recompute_granularity": "full",
        "policy.megatron_cfg.recompute_method": "uniform",
        "policy.megatron_cfg.recompute_num_layers": 1,
        "policy.megatron_cfg.moe_aux_loss_coeff": 0.0,
        "policy.megatron_cfg.moe_z_loss_coeff": 0.0,
        "policy.megatron_cfg.moe_input_jitter_eps": None,
        "policy.megatron_cfg.moe_router_load_balancing_type": "none",
        "policy.megatron_cfg.moe_shared_expert_overlap": False,
        "policy.megatron_cfg.mtp_num_layers": MTP_LAYERS,
        "policy.megatron_cfg.mtp_loss_scaling_factor": 0.3,
        "policy.megatron_cfg.mtp_use_repeated_layer": True,
        "policy.megatron_cfg.mtp_detach_heads": True,
        "policy.megatron_cfg.cuda_graph_impl": "none",
        "policy.megatron_cfg.fp8_cfg.enabled": False,
        "policy.generation.temperature": 1.0,
        "policy.generation.top_p": 1.0,
        "policy.generation.top_k": None,
    }
    for key, value in overrides.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)

    policy = OmegaConf.to_container(config.policy, resolve=True)
    loss = OmegaConf.to_container(config.loss_fn, resolve=True)
    if not isinstance(policy, dict) or not isinstance(loss, dict):
        raise ParityError("resolved policy and loss configurations must be dictionaries")
    if not all(isinstance(key, str) for key in loss):
        raise ParityError("resolved loss configuration keys must be strings")
    loss_dict = cast(dict[str, Any], loss)
    loss_config = ClippedPGLossConfig(**loss_dict)
    required_loss = {
        "force_on_policy_ratio": True,
        "use_importance_sampling_correction": True,
        "reference_policy_kl_penalty": 0.0,
        "token_level_loss": True,
    }
    for name, expected in required_loss.items():
        actual = getattr(loss_config, name)
        if actual != expected:
            raise ParityError(f"loss config {name} must be {expected!r}, got {actual!r}")
    fingerprint_policy = copy.deepcopy(policy)
    fingerprint_policy["shared_prefix_training"]["mode"] = "<arm>"
    fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "policy": fingerprint_policy,
                "loss": loss_dict,
                "topology": _topology(),
            }
        )
    )
    return cast(PolicyConfig, policy), loss_config, fingerprint


def _topology() -> dict[str, Any]:
    return {
        "world_size": WORLD_SIZE,
        "tensor_parallel_size": TP_SIZE,
        "context_parallel_size": CP_SIZE,
        "sequence_parallel": True,
        "expert_parallel_size": EP_SIZE,
        "expert_tensor_parallel_size": ETP_SIZE,
        "pipeline_parallel_size": PP_SIZE,
        "mtp_num_layers": MTP_LAYERS,
        "mtp_use_repeated_layer": True,
        "mtp_detach_heads": True,
    }


def _worker_sharding() -> Any:
    from nemo_rl.distributed.named_sharding import NamedSharding

    return NamedSharding(
        layout=np.arange(WORLD_SIZE).reshape(PP_SIZE, 1, CP_SIZE, TP_SIZE),
        names=[
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ],
    )


def _family(name: str) -> str:
    lowered = name.lower()
    if "mtp" in lowered or "multi_token" in lowered:
        return "mtp"
    if ".experts." in lowered or "router" in lowered or "shared_expert" in lowered:
        return "moe"
    if "self_attention" in lowered or ".attention." in lowered:
        return "attention"
    if ".mixer." in lowered or "mamba" in lowered:
        return "mamba"
    if "embedding" in lowered or "output_layer" in lowered:
        return "embedding_output"
    return "other"


def _named_parameters(model: torch.nn.Module) -> Iterable[tuple[str, torch.nn.Parameter]]:
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    chunks = unwrapped if isinstance(unwrapped, (list, tuple)) else [unwrapped]
    for chunk_index, chunk in enumerate(chunks):
        for name, parameter in chunk.named_parameters():
            yield f"chunk{chunk_index}.{name}", parameter


def _sample_indices(numel: int, *, device: torch.device, sample_count: int) -> torch.Tensor:
    count = min(sample_count, numel)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.linspace(0, numel - 1, count, dtype=torch.float64, device=device).to(torch.long)


def _optimizer_parts(optimizer: Any) -> list[Any]:
    parts = getattr(optimizer, "chained_optimizers", None)
    if parts is None:
        return [optimizer]
    return [part for optimizer_part in parts for part in _optimizer_parts(optimizer_part)]


def _optimizer_model_parameter_coverage(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer: Any,
) -> dict[str, Any]:
    """Audit full model ownership separately from this rank's optimizer shards.

    ``DistributedOptimizer.model_param_group_index_map`` intentionally contains
    only parameters intersecting this rank's DP/CP-owned gradient-buffer shard.
    With CP>1, a parameter wholly owned by a peer is therefore absent locally.
    The full ownership contract lives in each optimizer part's DDP buffers:
    ``param_index_map`` covers every parameter before byte-range sharding.

    This audit requires exact full-model ownership across those buffers and an
    exact bijection between the local shard map and raw optimizer parameters.
    It never treats local shard absence as proof that a model parameter is
    unoptimized.
    """
    named = list(named_parameters)
    required_by_id: dict[int, tuple[str, torch.nn.Parameter]] = {}
    required_names: set[str] = set()
    for name, parameter in named:
        if not parameter.requires_grad:
            continue
        if name in required_names:
            raise ParityError(f"duplicate trainable model parameter name: {name}")
        identity = id(parameter)
        if identity in required_by_id:
            raise ParityError(
                "trainable model parameter is exposed under more than one name: "
                f"{required_by_id[identity][0]!r} and {name!r}"
            )
        required_names.add(name)
        required_by_id[identity] = (name, parameter)
    if not required_by_id:
        raise ParityError("model exposes no trainable parameters")

    ownership_counts: dict[int, int] = {}
    local_shard_ids: set[int] = set()
    part_reports: list[dict[str, Any]] = []
    for part_index, part in enumerate(_optimizer_parts(optimizer)):
        parameter_map = getattr(part, "model_param_group_index_map", None)
        if not isinstance(parameter_map, dict):
            raise ParityError(f"optimizer part {part_index} lacks a distributed local parameter map")
        raw_optimizer = getattr(part, "optimizer", None)
        groups = getattr(raw_optimizer, "param_groups", None)
        if not isinstance(groups, list):
            raise ParityError(f"optimizer part {part_index} has invalid raw parameter groups")
        expected_locations: set[tuple[int, int]] = set()
        for group_index, group in enumerate(groups):
            raw_parameters = group.get("params") if isinstance(group, dict) else None
            if not isinstance(raw_parameters, list):
                raise ParityError(f"optimizer part {part_index} raw group {group_index} lacks a parameter list")
            expected_locations.update((group_index, parameter_index) for parameter_index in range(len(raw_parameters)))

        observed_locations: set[tuple[int, int]] = set()
        part_local_ids: set[int] = set()
        for parameter, location in parameter_map.items():
            if not isinstance(parameter, torch.Tensor):
                raise ParityError(f"optimizer part {part_index} maps a non-tensor model parameter")
            if (
                not isinstance(location, tuple)
                or len(location) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in location)
            ):
                raise ParityError(f"optimizer part {part_index} has invalid local parameter location: {location!r}")
            if location in observed_locations:
                raise ParityError(f"optimizer part {part_index} maps more than one model parameter to {location}")
            observed_locations.add(location)
            part_local_ids.add(id(parameter))
        if observed_locations != expected_locations:
            raise ParityError(
                f"optimizer part {part_index} local model/raw shard mapping is not bijective: "
                f"mapped={len(observed_locations)} raw={len(expected_locations)}"
            )

        model_param_gbuf_map = getattr(part, "model_param_gbuf_map", None)
        gbuf_ranges = getattr(part, "gbuf_ranges", None)
        if not isinstance(model_param_gbuf_map, dict) or not isinstance(gbuf_ranges, list):
            raise ParityError(f"optimizer part {part_index} lacks its local gradient-buffer mapping")
        gbuf_range_parameter_ids: set[int] = set()
        for gbuf_range_map in gbuf_ranges:
            if not isinstance(gbuf_range_map, dict):
                raise ParityError(f"optimizer part {part_index} has invalid gradient-buffer ranges")
            for bucket_maps in gbuf_range_map.values():
                if not isinstance(bucket_maps, list):
                    raise ParityError(f"optimizer part {part_index} has invalid gradient-buffer buckets")
                for bucket_map in bucket_maps:
                    param_map = bucket_map.get("param_map") if isinstance(bucket_map, dict) else None
                    if not isinstance(param_map, dict):
                        raise ParityError(f"optimizer part {part_index} bucket lacks a local parameter map")
                    for parameter in param_map:
                        identity = id(parameter)
                        if identity in gbuf_range_parameter_ids:
                            raise ParityError(f"optimizer part {part_index} locally shards one parameter twice")
                        gbuf_range_parameter_ids.add(identity)
        if (
            part_local_ids != {id(parameter) for parameter in model_param_gbuf_map}
            or part_local_ids != gbuf_range_parameter_ids
        ):
            raise ParityError(
                f"optimizer part {part_index} local parameter maps disagree across optimizer, "
                "gradient-buffer index, and gradient-buffer ranges"
            )

        get_model_param_range_map = getattr(part, "_get_model_param_range_map", None)
        if not callable(get_model_param_range_map):
            raise ParityError(f"optimizer part {part_index} lacks its model shard-range accessor")
        for model_parameter, location in parameter_map.items():
            group_index, parameter_index = location
            raw_parameter = groups[group_index]["params"][parameter_index]
            if not isinstance(raw_parameter, torch.Tensor):
                raise ParityError(f"optimizer part {part_index} raw optimizer shard is not a tensor")
            range_map = get_model_param_range_map(model_parameter)
            model_range = range_map.get("param") if isinstance(range_map, dict) else None
            start = getattr(model_range, "start", None)
            end = getattr(model_range, "end", None)
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not 0 <= start < end <= model_parameter.numel()
            ):
                raise ParityError(f"optimizer part {part_index} has an invalid model shard range")
            expected_view = model_parameter.detach().view(-1)[start:end]
            if (
                raw_parameter.dtype != expected_view.dtype
                or raw_parameter.device != expected_view.device
                or raw_parameter.numel() != expected_view.numel()
                or raw_parameter.data_ptr() != expected_view.data_ptr()
            ):
                raise ParityError(
                    f"optimizer part {part_index} model/raw shard binding is not the exact "
                    f"recorded parameter view at group={group_index} index={parameter_index}"
                )

        per_model_buffers = getattr(part, "per_model_buffers", None)
        if not isinstance(per_model_buffers, dict):
            raise ParityError(f"optimizer part {part_index} lacks full DDP-buffer parameter ownership")
        part_owned_ids: set[int] = set()
        for model_index, buffers in per_model_buffers.items():
            if not isinstance(buffers, (list, tuple)):
                raise ParityError(f"optimizer part {part_index} model {model_index!r} has invalid buffers")
            for buffer_index, buffer in enumerate(buffers):
                param_index_map = getattr(buffer, "param_index_map", None)
                if not isinstance(param_index_map, dict):
                    raise ParityError(
                        f"optimizer part {part_index} model {model_index!r} buffer "
                        f"{buffer_index} lacks a full parameter index map"
                    )
                for parameter in param_index_map:
                    if not isinstance(parameter, torch.Tensor) or not parameter.requires_grad:
                        raise ParityError(f"optimizer part {part_index} full ownership contains an invalid parameter")
                    identity = id(parameter)
                    if identity in part_owned_ids:
                        raise ParityError(f"optimizer part {part_index} owns one model parameter more than once")
                    part_owned_ids.add(identity)
        if not part_local_ids.issubset(part_owned_ids):
            raise ParityError(f"optimizer part {part_index} locally shards a parameter outside its full buffers")
        for identity in part_owned_ids:
            ownership_counts[identity] = ownership_counts.get(identity, 0) + 1
        local_shard_ids.update(part_local_ids)
        part_reports.append(
            {
                "part": part_index,
                "full_model_parameter_count": len(part_owned_ids),
                "local_shard_model_parameter_count": len(part_local_ids),
                "raw_optimizer_shard_count": len(expected_locations),
            }
        )

    duplicate_ownership = sorted(
        required_by_id[identity][0]
        for identity, count in ownership_counts.items()
        if count != 1 and identity in required_by_id
    )
    missing_ownership = sorted(
        name for identity, (name, _parameter) in required_by_id.items() if ownership_counts.get(identity, 0) == 0
    )
    unexpected_ownership = sorted(identity for identity in ownership_counts if identity not in required_by_id)
    if duplicate_ownership or missing_ownership or unexpected_ownership:
        raise ParityError(
            "distributed optimizer full-buffer ownership does not exactly cover trainable model "
            f"parameters: missing={len(missing_ownership)} duplicate={len(duplicate_ownership)} "
            f"unexpected={len(unexpected_ownership)} missing_sample={missing_ownership[:5]} "
            f"duplicate_sample={duplicate_ownership[:5]}"
        )
    if not local_shard_ids:
        raise ParityError("distributed optimizer exposes no local model-parameter shards")

    full_family_counts: dict[str, int] = {}
    local_family_counts: dict[str, int] = {}
    for identity, (name, _parameter) in required_by_id.items():
        family = _family(name)
        full_family_counts[family] = full_family_counts.get(family, 0) + 1
        if identity in local_shard_ids:
            local_family_counts[family] = local_family_counts.get(family, 0) + 1
    return {
        "method": "ddp-full-buffer-ownership-plus-local-shard-bijection",
        "trainable_model_parameter_count": len(required_by_id),
        "full_owned_model_parameter_count": len(ownership_counts),
        "local_shard_model_parameter_count": len(local_shard_ids),
        "peer_owned_model_parameter_count": len(required_by_id) - len(local_shard_ids),
        "full_family_counts": dict(sorted(full_family_counts.items())),
        "local_shard_family_counts": dict(sorted(local_family_counts.items())),
        "parts": part_reports,
    }


def _optimizer_model_parameter_coverage_self_test() -> dict[str, Any]:
    """Pin the distinction between full ownership and local CP/DP shards."""

    class FakeBuffer:
        def __init__(self, parameters: Sequence[torch.nn.Parameter]):
            self.param_index_map = {
                parameter: (index, index + parameter.numel(), 0) for index, parameter in enumerate(parameters)
            }

    class FakeRange:
        def __init__(self, start: int, end: int):
            self.start = start
            self.end = end

    class FakeRawOptimizer:
        pass

    class FakePart:
        def __init__(
            self,
            owned: Sequence[torch.nn.Parameter],
            local: Sequence[torch.nn.Parameter],
        ):
            self.per_model_buffers = {0: [FakeBuffer(owned)]}
            raw_parameters = [parameter.detach().view(-1) for parameter in local]
            self.optimizer = FakeRawOptimizer()
            self.optimizer.param_groups = [{"params": raw_parameters}]
            self.model_param_group_index_map = {parameter: (0, index) for index, parameter in enumerate(local)}
            self.model_param_gbuf_map = {parameter: (0, (parameter.dtype, parameter.dtype), 0) for parameter in local}
            self.gbuf_ranges = [
                {(torch.float32, torch.float32): [{"param_map": {parameter: {} for parameter in local}}]}
            ]

        def _get_model_param_range_map(self, parameter: torch.nn.Parameter) -> dict[str, Any]:
            return {"param": FakeRange(0, parameter.numel())}

    class FakeChain:
        def __init__(self, parts: Sequence[FakePart]):
            self.chained_optimizers = list(parts)

    parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in range(4)]
    named = [(f"layers.{index}.weight", parameter) for index, parameter in enumerate(parameters)]
    result = _optimizer_model_parameter_coverage(
        named,
        FakeChain(
            [
                FakePart(parameters[:2], parameters[:1]),
                FakePart(parameters[2:], parameters[2:3]),
            ]
        ),
    )
    if result["trainable_model_parameter_count"] != 4 or result["peer_owned_model_parameter_count"] != 2:
        raise ParityError("optimizer ownership self-test did not preserve peer-owned parameters")

    rejected = 0
    cases = (
        FakeChain([FakePart(parameters[:1], parameters[:1]), FakePart(parameters[2:], parameters[2:3])]),
        FakeChain([FakePart(parameters[:3], parameters[:1]), FakePart(parameters[2:], parameters[2:3])]),
    )
    for optimizer in cases:
        try:
            _optimizer_model_parameter_coverage(named, optimizer)
        except ParityError:
            rejected += 1
        else:
            raise ParityError("optimizer ownership self-test accepted invalid full ownership")

    swapped = FakePart(parameters[:2], parameters[:2])
    swapped.optimizer.param_groups[0]["params"].reverse()
    try:
        _optimizer_model_parameter_coverage(named[:2], FakeChain([swapped]))
    except ParityError:
        rejected += 1
    else:
        raise ParityError("optimizer ownership self-test accepted swapped model/raw bindings")
    return {"green_cases": 1, "rejected_cases": rejected, "result": result}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's exact persisted bytes, including BF16 and integer state."""
    digest = hashlib.sha256()
    flattened_bytes = tensor.detach().contiguous().view(torch.uint8).view(-1)
    for start in range(0, flattened_bytes.numel(), HASH_CHUNK_ELEMENTS):
        chunk = flattened_bytes[start : start + HASH_CHUNK_ELEMENTS].cpu().numpy()
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _reconstruct_fp32_master_from_bf16_remainder(
    primary: torch.Tensor,
    remainder: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct TE's logical FP32 master from BF16 upper and INT16 lower bits.

    TransformerEngine rounds the persisted BF16 upper half up when the signed
    lower half is negative.  Its Adam remainder kernel first subtracts one from
    that upper half in this case, then joins the two INT16 halves as FP32.  This
    implements that exact pinned-TE operation with tensor bit views.
    """
    if sys.byteorder != "little":
        raise ParityError("TE BF16/INT16 master reconstruction requires little-endian byte order")
    if primary.dtype != torch.bfloat16 or remainder.dtype != torch.int16:
        raise ParityError(
            "TE remainder reconstruction requires BF16 primary and INT16 remainder, "
            f"got {primary.dtype}/{remainder.dtype}"
        )
    if primary.shape != remainder.shape or primary.device != remainder.device:
        raise ParityError(
            "TE remainder reconstruction shape/device mismatch: "
            f"primary={tuple(primary.shape)}/{primary.device} "
            f"remainder={tuple(remainder.shape)}/{remainder.device}"
        )

    lower = remainder.detach().contiguous().view(-1)
    rounded_upper = primary.detach().contiguous().view(torch.int16).view(-1)
    corrected_upper = (rounded_upper.to(torch.int32) - (lower < 0).to(torch.int32)).to(torch.int16)
    halves = torch.stack((lower, corrected_upper), dim=1).contiguous()
    return halves.view(torch.float32).reshape(primary.shape)


def _master_remainder_reconstruction_self_test() -> dict[str, Any]:
    """Bit-exact deterministic regression test for the pinned TE split format."""
    master = torch.tensor(
        (0.0, 1.0, -1.0, 0.1, -3.1415927, 1.0e-30, 1.0e30, -0.33333334),
        dtype=torch.float32,
    )
    halves = master.contiguous().view(torch.int16).reshape(-1, 2)
    lower = halves[:, 0].clone()
    upper = halves[:, 1].clone()
    if not bool(torch.any(lower < 0)) or not bool(torch.any(lower >= 0)):
        raise ParityError("TE remainder reconstruction self-test lacks both rounding branches")
    rounded_upper = (upper.to(torch.int32) + (lower < 0).to(torch.int32)).to(torch.int16)
    primary = rounded_upper.contiguous().view(torch.bfloat16)
    reconstructed = _reconstruct_fp32_master_from_bf16_remainder(primary, lower)
    if not torch.equal(reconstructed.view(torch.int32), master.view(torch.int32)):
        raise ParityError("TE BF16/INT16 reconstruction self-test is not bit exact")

    zero_remainder = torch.zeros_like(lower)
    zero_reconstructed = _reconstruct_fp32_master_from_bf16_remainder(primary, zero_remainder)
    primary_fp32 = primary.float()
    if not torch.equal(zero_reconstructed.view(torch.int32), primary_fp32.view(torch.int32)):
        raise ParityError("zero TE remainder does not reconstruct the exact BF16 primary")
    return {
        "algorithm": "te-bf16-rounded-upper-int16-lower-v1",
        "cases": master.numel(),
        "negative_remainder_cases": int(torch.count_nonzero(lower < 0).item()),
        "master_sha256": _tensor_sha256(master),
        "reconstructed_sha256": _tensor_sha256(reconstructed),
        "zero_remainder_sha256": _tensor_sha256(zero_reconstructed),
    }


def _optimizer_binding(
    optimizer: Any,
    parameter: torch.nn.Parameter,
) -> tuple[Any, Any, torch.Tensor] | None:
    """Resolve one model parameter to its unique distributed-optimizer shard."""
    matches: list[tuple[Any, Any, torch.Tensor]] = []
    for part in _optimizer_parts(optimizer):
        parameter_map = getattr(part, "model_param_group_index_map", None)
        if not isinstance(parameter_map, dict) or parameter not in parameter_map:
            continue
        location = parameter_map[parameter]
        if (
            not isinstance(location, tuple)
            or len(location) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in location)
        ):
            raise ParityError(f"invalid distributed-optimizer parameter location: {location!r}")
        group_index, group_order = location
        raw_optimizer = getattr(part, "optimizer", None)
        groups = getattr(raw_optimizer, "param_groups", None)
        if not isinstance(groups, list) or not 0 <= group_index < len(groups):
            raise ParityError("distributed optimizer has invalid raw parameter groups")
        raw_group = groups[group_index]
        raw_parameters = raw_group.get("params") if isinstance(raw_group, dict) else None
        if not isinstance(raw_parameters, list) or not 0 <= group_order < len(raw_parameters):
            raise ParityError("distributed optimizer has invalid raw parameter ordering")
        raw_parameter = raw_parameters[group_order]
        if not isinstance(raw_parameter, torch.Tensor):
            raise ParityError("distributed optimizer raw parameter is not a tensor")
        matches.append((part, raw_optimizer, raw_parameter))
    if len(matches) > 1:
        raise ParityError("model parameter is owned by more than one distributed optimizer")
    return matches[0] if matches else None


def _optimizer_master(
    optimizer: Any,
    parameter: torch.nn.Parameter,
) -> tuple[torch.Tensor | None, str, dict[str, Any]]:
    """Return one logical FP32 master and its exact persisted representation."""
    binding = _optimizer_binding(optimizer, parameter)
    if binding is not None:
        part, raw_optimizer, raw_parameter = binding
        config = getattr(part, "config", None)
        precision_aware = getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", None)
        if precision_aware is True:
            raw_state = getattr(raw_optimizer, "state", None)
            state = raw_state.get(raw_parameter) if hasattr(raw_state, "get") else None
            if not isinstance(state, dict) or "master_param" not in state:
                raise ParityError("precision-aware optimizer state was not officially initialized")
            stored_master = state["master_param"]
            if not isinstance(stored_master, torch.Tensor):
                raise ParityError("precision-aware optimizer master state is not a tensor")

            stores_remainders = getattr(raw_optimizer, "store_param_remainders", None)
            if raw_parameter.dtype == torch.bfloat16 and stores_remainders is True:
                if stored_master.dtype != torch.int16:
                    raise ParityError("TE BF16 remainder storage must be INT16, " f"got {stored_master.dtype}")
                master = _reconstruct_fp32_master_from_bf16_remainder(
                    raw_parameter,
                    stored_master,
                )
                return (
                    master,
                    "te-bf16-int16-remainder",
                    {
                        "kind": "bf16-primary-int16-remainder",
                        "primary_dtype": str(raw_parameter.dtype),
                        "primary_sha256": _tensor_sha256(raw_parameter),
                        "remainder_dtype": str(stored_master.dtype),
                        "remainder_sha256": _tensor_sha256(stored_master),
                    },
                )

            if raw_parameter.dtype == torch.float32:
                return (
                    raw_parameter,
                    "precision-aware-native-fp32-primary",
                    {
                        "kind": "native-fp32-primary",
                        "primary_dtype": str(raw_parameter.dtype),
                        "primary_sha256": _tensor_sha256(raw_parameter),
                    },
                )

            get_unscaled_state = getattr(raw_optimizer, "get_unscaled_state", None)
            if not callable(get_unscaled_state):
                raise ParityError("precision-aware optimizer lacks get_unscaled_state")
            master = get_unscaled_state(raw_parameter, "master_param")
            if not isinstance(master, torch.Tensor):
                raise ParityError("precision-aware optimizer returned a non-tensor master")
            return (
                master,
                "precision-aware-full-master",
                {
                    "kind": "full-master",
                    "master_dtype": str(stored_master.dtype),
                    "master_sha256": _tensor_sha256(stored_master),
                },
            )

        get_state = getattr(part, "_get_main_param_and_optimizer_states", None)
        if not callable(get_state):
            raise ParityError("distributed optimizer lacks its parameter-state accessor")
        tensors = get_state(parameter)
        if not isinstance(tensors, dict):
            raise ParityError("distributed optimizer returned invalid parameter state")
        master = tensors.get("param")
        if master is not None and not isinstance(master, torch.Tensor):
            raise ParityError("distributed optimizer master parameter is not a tensor")
        storage = (
            {
                "kind": "full-master",
                "master_dtype": str(master.dtype),
                "master_sha256": _tensor_sha256(master),
            }
            if isinstance(master, torch.Tensor)
            else {"kind": "missing"}
        )
        return master, "distributed-optimizer-state", storage

    master = getattr(parameter, "main_param", None)
    if master is not None:
        if not isinstance(master, torch.Tensor):
            raise ParityError("parameter main_param is not a tensor")
        return (
            master,
            "parameter-main-param",
            {
                "kind": "full-master",
                "master_dtype": str(master.dtype),
                "master_sha256": _tensor_sha256(master),
            },
        )
    if parameter.dtype == torch.float32:
        return (
            parameter,
            "native-fp32-parameter",
            {
                "kind": "native-fp32-primary",
                "primary_dtype": str(parameter.dtype),
                "primary_sha256": _tensor_sha256(parameter),
            },
        )
    return None, "no-local-master-shard", {"kind": "missing"}


def _optimizer_raw_parameter_fingerprint(optimizer: Any) -> dict[str, Any]:
    """Hash every raw optimizer primary shard in stable group order."""
    digest = hashlib.sha256()
    tensor_count = 0
    numel = 0
    seen: set[int] = set()
    for part_index, part in enumerate(_optimizer_parts(optimizer)):
        raw_optimizer = getattr(part, "optimizer", None)
        groups = getattr(raw_optimizer, "param_groups", None)
        if not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups):
            raw_parameters = group.get("params") if isinstance(group, dict) else None
            if not isinstance(raw_parameters, list):
                raise ParityError("raw optimizer parameter group lacks a parameter list")
            for parameter_index, raw_parameter in enumerate(raw_parameters):
                if not isinstance(raw_parameter, torch.Tensor):
                    raise ParityError("raw optimizer parameter is not a tensor")
                identity = id(raw_parameter)
                if identity in seen:
                    raise ParityError("raw optimizer parameter occurs more than once")
                seen.add(identity)
                tensor_sha256 = _tensor_sha256(raw_parameter)
                digest.update(
                    _canonical_json_bytes(
                        {
                            "part": part_index,
                            "group": group_index,
                            "parameter": parameter_index,
                            "dtype": str(raw_parameter.dtype),
                            "shape": list(raw_parameter.shape),
                            "sha256": tensor_sha256,
                        }
                    )
                )
                tensor_count += 1
                numel += raw_parameter.numel()
    if tensor_count == 0:
        raise ParityError("optimizer exposes no raw primary shards")
    return {"sha256": digest.hexdigest(), "tensor_count": tensor_count, "numel": numel}


def _finalized_gradient_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """Hash every finalized gradient, recording absent gradients explicitly."""
    digest = hashlib.sha256()
    tensor_count = 0
    absent_count = 0
    numel = 0
    for name, parameter in sorted(_named_parameters(model), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if gradient is None:
            digest.update(_canonical_json_bytes({"name": name, "present": False}))
            absent_count += 1
            continue
        if not isinstance(gradient, torch.Tensor):
            raise ParityError(f"finalized gradient for {name} is not a tensor")
        tensor_sha256 = _tensor_sha256(gradient)
        digest.update(
            _canonical_json_bytes(
                {
                    "name": name,
                    "present": True,
                    "dtype": str(gradient.dtype),
                    "shape": list(gradient.shape),
                    "sha256": tensor_sha256,
                }
            )
        )
        tensor_count += 1
        numel += gradient.numel()
    if tensor_count == 0:
        raise ParityError("no finalized gradients are available before optimizer state initialization")
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "absent_count": absent_count,
        "numel": numel,
    }


def _optimizer_group_step_metadata(optimizer: Any) -> list[dict[str, Any]]:
    """Capture group-level optimizer step counters without creating state."""
    evidence: list[dict[str, Any]] = []
    for part_index, part in enumerate(_optimizer_parts(optimizer)):
        raw_optimizer = getattr(part, "optimizer", None)
        groups = getattr(raw_optimizer, "param_groups", None)
        if not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ParityError("raw optimizer parameter group is not a dictionary")
            item: dict[str, Any] = {
                "part": part_index,
                "group": group_index,
                "present": "step" in group,
            }
            if "step" in group:
                step = group["step"]
                if isinstance(step, bool):
                    raise ParityError("optimizer group step must not be boolean")
                if isinstance(step, int):
                    item.update({"kind": "int", "value": step})
                elif isinstance(step, torch.Tensor) and step.numel() == 1:
                    item.update(
                        {
                            "kind": "tensor",
                            "dtype": str(step.dtype),
                            "sha256": _tensor_sha256(step),
                            "value": int(step.detach().cpu().item()),
                        }
                    )
                else:
                    raise ParityError(f"unsupported optimizer group step metadata: {step!r}")
            evidence.append(item)
    if not evidence:
        raise ParityError("optimizer exposes no parameter-group step metadata")
    return evidence


def _initialize_optimizer_state_for_probe(
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    """Materialize lazy state with the exact initializer used by pinned TE.step.

    The paired MCore closure currently calls ``initialize_state(p)`` without
    TE 2.15's required remainder flag, so this diagnostic invokes TE's public
    initializer directly with the same flag expression as its lazy step path.
    """
    primary_before = _optimizer_raw_parameter_fingerprint(optimizer)
    gradients_before = _finalized_gradient_fingerprint(model)
    group_steps_before = _optimizer_group_step_metadata(optimizer)
    scheduler_steps_before = getattr(scheduler, "num_steps", None)
    if isinstance(scheduler_steps_before, bool) or not isinstance(scheduler_steps_before, int):
        raise ParityError(f"scheduler num_steps must be an integer, got {scheduler_steps_before!r}")
    parameter_coverage = _optimizer_model_parameter_coverage(
        _named_parameters(model),
        optimizer,
    )

    initialized_parts = 0
    initialized_parameters = 0
    bf16_remainder_parameters = 0
    for part in _optimizer_parts(optimizer):
        parameter_map = getattr(part, "model_param_group_index_map", None)
        if not isinstance(parameter_map, dict):
            continue
        config = getattr(part, "config", None)
        if getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", None) is not True:
            raise ParityError("captured parity requires the recipe's precision-aware distributed optimizer")
        if getattr(config, "store_param_remainders", None) is not True:
            raise ParityError("captured parity must retain the recipe's BF16/INT16 master representation")
        raw_optimizer = getattr(part, "optimizer", None)
        if getattr(raw_optimizer, "store_param_remainders", None) is not True:
            raise ParityError("TE optimizer did not enable the recipe's parameter-remainder storage")
        groups = getattr(raw_optimizer, "param_groups", None)
        state = getattr(raw_optimizer, "state", None)
        initialize_state = getattr(raw_optimizer, "initialize_state", None)
        if not isinstance(groups, list) or not hasattr(state, "get"):
            raise ParityError("distributed optimizer has invalid raw optimizer state")
        if not callable(initialize_state):
            raise ParityError("pinned TE optimizer lacks its public initialize_state method")
        raw_parameters = [
            parameter for group in groups if isinstance(group, dict) for parameter in group.get("params", [])
        ]
        if not raw_parameters or any(not isinstance(parameter, torch.Tensor) for parameter in raw_parameters):
            raise ParityError("distributed optimizer has invalid raw parameters")
        nonempty = [parameter for parameter in raw_parameters if state.get(parameter)]
        if nonempty:
            raise ParityError(
                "optimizer state must be empty before the official diagnostic initializer, "
                f"got {len(nonempty)} initialized parameters"
            )

        for raw_parameter in raw_parameters:
            initialize_state(
                raw_parameter,
                raw_optimizer.store_param_remainders and raw_parameter.dtype == torch.bfloat16,
            )
        initialized_parts += 1
        for raw_parameter in raw_parameters:
            parameter_state = state.get(raw_parameter)
            if not isinstance(parameter_state, dict) or set(parameter_state) != {
                "exp_avg",
                "exp_avg_sq",
                "master_param",
            }:
                raise ParityError(
                    "official optimizer initializer produced unexpected state keys: "
                    f"{sorted(parameter_state) if isinstance(parameter_state, dict) else parameter_state!r}"
                )
            exp_avg = parameter_state["exp_avg"]
            exp_avg_sq = parameter_state["exp_avg_sq"]
            stored_master = parameter_state["master_param"]
            if not all(isinstance(value, torch.Tensor) for value in (exp_avg, exp_avg_sq, stored_master)):
                raise ParityError("official optimizer initializer produced non-tensor state")
            if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
                raise ParityError(
                    "captured recipe requires FP32 Adam moments, got " f"{exp_avg.dtype}/{exp_avg_sq.dtype}"
                )
            if torch.count_nonzero(exp_avg).item() != 0 or torch.count_nonzero(exp_avg_sq).item() != 0:
                raise ParityError("official optimizer initializer produced nonzero Adam moments")
            if raw_parameter.dtype == torch.bfloat16:
                if stored_master.dtype != torch.int16 or torch.count_nonzero(stored_master).item() != 0:
                    raise ParityError("official TE BF16 master remainder must initialize to INT16 zeros")
                reconstructed = _reconstruct_fp32_master_from_bf16_remainder(
                    raw_parameter,
                    stored_master,
                )
                if not torch.equal(reconstructed.view(torch.int32), raw_parameter.float().view(torch.int32)):
                    raise ParityError("zero TE remainder does not reconstruct the exact BF16 primary shard")
                bf16_remainder_parameters += 1
            elif stored_master.dtype != torch.float32:
                raise ParityError(f"non-BF16 precision-aware master must be FP32, got {stored_master.dtype}")
            initialized_parameters += 1

    if initialized_parts == 0 or initialized_parameters == 0 or bf16_remainder_parameters == 0:
        raise ParityError("official optimizer initialization did not cover precision-aware BF16 remainder state")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    primary_after = _optimizer_raw_parameter_fingerprint(optimizer)
    gradients_after = _finalized_gradient_fingerprint(model)
    group_steps_after = _optimizer_group_step_metadata(optimizer)
    scheduler_steps_after = getattr(scheduler, "num_steps", None)
    if primary_after != primary_before:
        raise ParityError("official optimizer state initialization mutated a primary parameter shard")
    if gradients_after != gradients_before:
        raise ParityError("official optimizer state initialization mutated a finalized gradient")
    if group_steps_after != group_steps_before:
        raise ParityError("official optimizer state initialization advanced optimizer group steps")
    if scheduler_steps_after != scheduler_steps_before:
        raise ParityError("official optimizer state initialization advanced the scheduler")
    return {
        "method": "transformer-engine-initialize_state",
        "mcore_init_state_fn_used": False,
        "mcore_init_state_fn_incompatibility": ("calls-te-initialize_state-without-required-remainder-flag"),
        "store_param_remainders": True,
        "initialized_parts": initialized_parts,
        "initialized_parameters": initialized_parameters,
        "bf16_remainder_parameters": bf16_remainder_parameters,
        "parameter_coverage": parameter_coverage,
        "primary_before": primary_before,
        "primary_after": primary_after,
        "gradients_before": gradients_before,
        "gradients_after": gradients_after,
        "group_steps_before": group_steps_before,
        "group_steps_after": group_steps_after,
        "scheduler_steps_before": scheduler_steps_before,
        "scheduler_steps_after": scheduler_steps_after,
        "unchanged": True,
    }


def _snapshot_optimizer_masters(model: torch.nn.Module, optimizer: Any) -> dict[str, Any]:
    """Fingerprint every logical FP32 master and its exact persisted state."""
    parameters: dict[str, dict[str, Any]] = {}
    family_hashers: dict[str, Any] = {}
    family_storage_hashers: dict[str, Any] = {}
    missing_by_family: dict[str, int] = {}
    local_shard_parameter_ids = {
        id(parameter)
        for part in _optimizer_parts(optimizer)
        for parameter in getattr(part, "model_param_group_index_map", {})
    }
    if not local_shard_parameter_ids:
        raise ParityError("optimizer exposes no local model-parameter shards to snapshot")
    named_local_shard_parameter_ids: set[int] = set()
    with torch.no_grad():
        for name, parameter in sorted(_named_parameters(model), key=lambda item: item[0]):
            if not parameter.requires_grad:
                continue
            family = _family(name)
            if id(parameter) not in local_shard_parameter_ids:
                missing_by_family[family] = missing_by_family.get(family, 0) + 1
                continue
            named_local_shard_parameter_ids.add(id(parameter))
            master, source, storage = _optimizer_master(optimizer, parameter)
            if master is None:
                raise ParityError(f"locally optimizer-bound parameter {name} has no master shard")
            if master.dtype != torch.float32:
                raise ParityError(f"optimizer master for {name} must be FP32, got {master.dtype} ({source})")
            flattened = master.detach().reshape(-1)
            if flattened.numel() == 0:
                continue
            if not torch.isfinite(flattened).all():
                raise ParityError(f"optimizer master for {name} contains non-finite values")
            indices = _sample_indices(
                flattened.numel(),
                device=flattened.device,
                sample_count=MASTER_SAMPLE_COUNT,
            )
            tensor_sha256 = _tensor_sha256(flattened)
            tensor_sum = float(torch.sum(flattened, dtype=torch.float64).item())
            tensor_l2 = float(torch.linalg.vector_norm(flattened, ord=2, dtype=torch.float64).item())
            if not math.isfinite(tensor_sum) or not math.isfinite(tensor_l2):
                raise ParityError(f"optimizer master summary for {name} is non-finite")
            parameters[name] = {
                "family": family,
                "source": source,
                "dtype": str(master.dtype),
                "sharded": bool(getattr(parameter, "main_param_sharded", False)),
                "numel": flattened.numel(),
                "sha256": tensor_sha256,
                "sum": tensor_sum,
                "l2": tensor_l2,
                "samples": flattened.index_select(0, indices).cpu().tolist(),
                "storage": storage,
            }
            family_hasher = family_hashers.setdefault(family, hashlib.sha256())
            family_hasher.update(
                _canonical_json_bytes(
                    {
                        "name": name,
                        "numel": flattened.numel(),
                        "sha256": tensor_sha256,
                    }
                )
            )
            family_storage_hasher = family_storage_hashers.setdefault(family, hashlib.sha256())
            family_storage_hasher.update(
                _canonical_json_bytes(
                    {
                        "name": name,
                        "numel": flattened.numel(),
                        "storage": storage,
                    }
                )
            )

    if named_local_shard_parameter_ids != local_shard_parameter_ids:
        raise ParityError(
            "optimizer local shard map contains parameters absent from the named trainable model: "
            f"mapped={len(local_shard_parameter_ids)} named={len(named_local_shard_parameter_ids)}"
        )
    if len(parameters) != len(local_shard_parameter_ids):
        raise ParityError(
            "optimizer master snapshot did not cover every local shard: "
            f"snapshots={len(parameters)} local_shards={len(local_shard_parameter_ids)}"
        )

    families: dict[str, Any] = {}
    for family, family_hasher in family_hashers.items():
        names = sorted(name for name, item in parameters.items() if item["family"] == family)
        families[family] = {
            "sha256": family_hasher.hexdigest(),
            "storage_sha256": family_storage_hashers[family].hexdigest(),
            "tensor_count": len(names),
            "numel": sum(parameters[name]["numel"] for name in names),
            "missing_model_parameter_count": missing_by_family.get(family, 0),
        }
    return {
        "parameters": parameters,
        "families": families,
        "local_shard_model_parameter_count": len(local_shard_parameter_ids),
        "peer_owned_model_parameter_count": sum(missing_by_family.values()),
    }


def _snapshot_finalized_gradients(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for name, parameter in _named_parameters(model):
            if not parameter.requires_grad:
                continue
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            item: dict[str, Any] = {
                "family": _family(name),
                "numel": parameter.numel(),
                "present": gradient is not None,
            }
            if gradient is not None:
                flattened = gradient.detach().reshape(-1)
                indices = _sample_indices(
                    flattened.numel(),
                    device=flattened.device,
                    sample_count=GRADIENT_SAMPLE_COUNT,
                )
                l2 = torch.linalg.vector_norm(flattened, ord=2, dtype=torch.float32)
                linf = torch.linalg.vector_norm(flattened, ord=float("inf"), dtype=torch.float32)
                total = torch.sum(flattened, dtype=torch.float32)
                scalars = (float(l2.item()), float(linf.item()), float(total.item()))
                if not all(math.isfinite(value) for value in scalars):
                    raise ParityError(f"non-finite finalized gradient for {name}: {scalars}")
                item.update(
                    {
                        "l2": scalars[0],
                        "linf": scalars[1],
                        "sum": scalars[2],
                        "samples": flattened.index_select(0, indices).float().cpu().tolist(),
                    }
                )
            evidence[name] = item
    return evidence


def _selected_logprobs(
    worker: Any,
    rows: Sequence[dict[str, Any]],
    *,
    policy_config: "PolicyConfig",
    shared_prefix_mode: Literal["observe", "train"],
) -> list[list[float]]:
    source_batch = _build_batch(rows)
    mask = (source_batch["token_mask"].to(dtype=torch.bool) & source_batch["attention_mask"].to(dtype=torch.bool)).cpu()
    prepared, source_order = _prepare_batch_for_worker(
        source_batch,
        policy_config=policy_config,
        shared_prefix_mode=shared_prefix_mode,
        stage="logprob",
    )
    result = worker.get_logprobs(
        data=prepared,
        micro_batch_size=1,
        require_router_replay=False,
    )
    if source_order is not None:
        # Match LMPolicy.get_logprobs: worker outputs follow the packer's
        # execution order, while parity evidence is always captured-row order.
        result.reorder_data(source_order)
    logprobs = result["logprobs"].float().cpu()
    if logprobs.shape != mask.shape:
        raise ParityError(f"logprob/mask shape mismatch: {tuple(logprobs.shape)}/{tuple(mask.shape)}")
    selected_by_sequence: list[list[float]] = []
    for row in range(BATCH_SIZE):
        selected = logprobs[row][mask[row]]
        if selected.numel() == 0 or not torch.isfinite(selected).all():
            raise ParityError(f"selected policy logprobs for captured sequence {row} are empty or non-finite")
        selected_by_sequence.append(selected.tolist())
    return selected_by_sequence


def _scalar(value: Any, *, label: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ParityError(f"{label} must be scalar, got shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    return _finite_float(value, label=label)


def _metric_evidence(metrics: dict[str, Any]) -> dict[str, Any]:
    if "global_loss" not in metrics:
        raise ParityError("training metrics are missing global_loss")
    mtp = metrics.get("mtp_metrics")
    if not isinstance(mtp, dict):
        raise ParityError("training metrics are missing mtp_metrics")
    expected = {*EXPECTED_MTP_LOSS_KEYS, *EXPECTED_MTP_ACCEPTANCE_KEYS, "grad_norm"}
    if set(mtp) != expected:
        raise ParityError(
            f"MTP metric coverage mismatch: missing={sorted(expected - set(mtp))} "
            f"extra={sorted(set(mtp) - expected)}"
        )
    return {
        "global_loss": _scalar(metrics["global_loss"], label="global_loss"),
        "grad_norm": _scalar(metrics["grad_norm"], label="grad_norm"),
        "mtp": {name: _scalar(mtp[name], label=name) for name in sorted(mtp)},
    }


def _install_mamba_replay_diagnostic() -> None:
    from megatron.core.models.hybrid import shared_prefix

    production = shared_prefix._forward_mamba_layer_shared_prefix_cp
    diagnostic = shared_prefix._forward_mamba_layer_shared_prefix_cp_replay
    if production is diagnostic:
        raise ParityError("Mamba replay diagnostic was already installed")
    if production.__name__ != "_forward_mamba_layer_shared_prefix_cp":
        raise ParityError(f"unexpected production Mamba CP helper: {production!r}")
    shared_prefix._forward_mamba_layer_shared_prefix_cp = diagnostic
    if shared_prefix._forward_mamba_layer_shared_prefix_cp is not diagnostic:
        raise ParityError("failed to install process-local Mamba replay diagnostic")


def _install_mamba_packed_replay_diagnostic() -> None:
    """Install the forward-only dense packed-fused Mamba oracle process-locally."""
    from megatron.core.models.hybrid import shared_prefix

    production = shared_prefix._forward_mamba_layer_shared_prefix_cp
    diagnostic = shared_prefix._forward_mamba_layer_shared_prefix_cp_packed_fused_oracle
    if production is diagnostic:
        raise ParityError("packed-fused Mamba replay diagnostic was already installed")
    if production.__name__ != "_forward_mamba_layer_shared_prefix_cp":
        raise ParityError(f"unexpected production Mamba CP helper: {production!r}")
    shared_prefix._forward_mamba_layer_shared_prefix_cp = diagnostic
    if shared_prefix._forward_mamba_layer_shared_prefix_cp is not diagnostic:
        raise ParityError("failed to install process-local packed-fused Mamba replay diagnostic")


MAMBA_RUNTIME_VARIANTS = (
    "mamba_state_fork",
    "mamba_decomposed_replay",
    "mamba_packed_fused",
)
RUNTIME_REQUIRED_COUNTERS = ("hybrid_stack", "mamba_cp", "attention_cp")
RUNTIME_PATH_COUNTERS = (*RUNTIME_REQUIRED_COUNTERS, *MAMBA_RUNTIME_VARIANTS)


def _install_shared_prefix_runtime_counters() -> tuple[
    dict[str, int],
    dict[str, str],
    Callable[[], None],
]:
    """Count the three production shared-prefix dispatch boundaries."""
    from megatron.core.models.hybrid import (
        hybrid_model,
        shared_prefix,
        shared_prefix_fused,
    )

    counts = {name: 0 for name in RUNTIME_PATH_COUNTERS}
    original_hybrid = hybrid_model.forward_hybrid_stack_shared_prefix
    original_mamba = shared_prefix._forward_mamba_layer_shared_prefix_cp
    original_attention = shared_prefix_fused.flash_composed_forest_attention_cp
    mamba_variant_by_implementation = {
        "_forward_mamba_layer_shared_prefix_cp": "mamba_state_fork",
        "_forward_mamba_layer_shared_prefix_cp_replay": "mamba_decomposed_replay",
        "_forward_mamba_layer_shared_prefix_cp_packed_fused_oracle": "mamba_packed_fused",
    }
    try:
        mamba_variant = mamba_variant_by_implementation[original_mamba.__name__]
    except KeyError as error:
        raise ParityError(f"cannot attest unknown shared-prefix Mamba implementation {original_mamba!r}") from error

    @wraps(original_hybrid)
    def counted_hybrid(*args: Any, **kwargs: Any) -> Any:
        counts["hybrid_stack"] += 1
        return original_hybrid(*args, **kwargs)

    @wraps(original_mamba)
    def counted_mamba(*args: Any, **kwargs: Any) -> Any:
        counts["mamba_cp"] += 1
        counts[mamba_variant] += 1
        return original_mamba(*args, **kwargs)

    @wraps(original_attention)
    def counted_attention(*args: Any, **kwargs: Any) -> Any:
        counts["attention_cp"] += 1
        return original_attention(*args, **kwargs)

    hybrid_model.forward_hybrid_stack_shared_prefix = counted_hybrid
    shared_prefix._forward_mamba_layer_shared_prefix_cp = counted_mamba
    shared_prefix_fused.flash_composed_forest_attention_cp = counted_attention

    def restore() -> None:
        hybrid_model.forward_hybrid_stack_shared_prefix = original_hybrid
        shared_prefix._forward_mamba_layer_shared_prefix_cp = original_mamba
        shared_prefix_fused.flash_composed_forest_attention_cp = original_attention

    implementations = {
        "hybrid_stack": original_hybrid.__name__,
        "mamba_cp": original_mamba.__name__,
        "mamba_variant": mamba_variant,
        "attention_cp": original_attention.__name__,
    }
    return counts, implementations, restore


def _runtime_path_snapshot(counts: dict[str, int]) -> dict[str, int]:
    if set(counts) != set(RUNTIME_PATH_COUNTERS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()
    ):
        raise ParityError(f"invalid shared-prefix runtime counters: {counts!r}")
    return {name: counts[name] for name in RUNTIME_PATH_COUNTERS}


def _require_runtime_path_counts(
    current: dict[str, int],
    *,
    shared_prefix_enabled: bool,
    phase: str,
    previous: dict[str, int] | None = None,
) -> None:
    baseline = previous or {name: 0 for name in RUNTIME_PATH_COUNTERS}
    if shared_prefix_enabled:
        failed = [name for name in RUNTIME_REQUIRED_COUNTERS if current[name] <= baseline[name]]
        if failed:
            raise ParityError(
                f"shared-prefix {phase} did not increase runtime counters "
                f"{failed}: previous={baseline!r} current={current!r}"
            )
        active_variants = [name for name in MAMBA_RUNTIME_VARIANTS if current[name] > baseline[name]]
        mamba_delta = current["mamba_cp"] - baseline["mamba_cp"]
        if len(active_variants) != 1 or current[active_variants[0]] - baseline[active_variants[0]] != mamba_delta:
            raise ParityError(
                "shared-prefix Mamba runtime attestation is ambiguous: "
                f"active={active_variants!r} previous={baseline!r} current={current!r}"
            )
    elif any(current.values()):
        raise ParityError(f"dense OFF {phase} unexpectedly called shared-prefix runtime paths: {current!r}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = _canonical_json_bytes(value) + b"\n"
    with temporary.open("xb", buffering=0) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def _run_arm(arguments: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise ParityError(f"run-arm requires torchrun WORLD_SIZE={WORLD_SIZE}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= local_rank < WORLD_SIZE:
        raise ParityError(f"invalid LOCAL_RANK={local_rank}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise ParityError(f"run-arm requires exactly four visible CUDA devices, got {torch.cuda.device_count()}")
    if not torch.cuda.is_bf16_supported():
        raise ParityError("run-arm requires CUDA BF16 support")
    master_reconstruction_self_test = _master_remainder_reconstruction_self_test()
    model_path = Path(arguments.model_path)
    if not model_path.is_absolute() or not (model_path / "config.json").is_file():
        raise ParityError(f"model path is not an absolute HF checkpoint: {model_path}")
    repo_root = Path(arguments.repo_root)
    try:
        resolved_repo_root = repo_root.resolve(strict=True)
    except OSError as error:
        raise ParityError(f"cannot resolve NeMo repo root {repo_root}: {error}") from error
    if (
        not repo_root.is_absolute()
        or not repo_root.is_dir()
        or repo_root.is_symlink()
        or resolved_repo_root != repo_root
    ):
        raise ParityError(f"NeMo repo root must be an absolute canonical non-symlink directory: {repo_root}")
    output_dir = Path(arguments.output_dir)
    if not output_dir.is_absolute() or not output_dir.is_dir():
        raise ParityError(f"output directory must already exist and be absolute: {output_dir}")
    rows, batch_summary = _read_captured_rows(Path(arguments.batch), expected_sha256=arguments.expected_batch_sha256)

    shared_prefix_mode: Literal["observe", "train"] = "observe" if arguments.arm == "off" else "train"
    policy_config, loss_config, config_sha256 = _build_policy_and_loss_config(
        str(model_path),
        repo_root=repo_root,
        shared_prefix_mode=shared_prefix_mode,
    )
    batch_preparation = _preflight_batch_preparation(
        rows,
        policy_config=policy_config,
        shared_prefix_mode=shared_prefix_mode,
    )
    from nemo_rl.algorithms.loss import ClippedPGLossFn
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.models.megatron.setup import destroy_parallel_state
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    ray.get_gpu_ids = lambda: [local_rank]
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    worker = None
    restore_runtime_counters: Callable[[], None] | None = None
    evidence: dict[str, Any] | None = None
    optimizer_probe: dict[str, Any] = {"calls": 0}
    try:
        worker = MegatronPolicyWorkerImpl(
            config=policy_config,
            tokenizer=tokenizer,
            init_optimizer=True,
            init_reference_model=False,
            worker_sharding_annotations=_worker_sharding(),
        )
        model_config = worker._get_model_config()
        observed_topology = {
            "world_size": torch.distributed.get_world_size(),
            "tensor_parallel_size": model_config.tensor_model_parallel_size,
            "context_parallel_size": model_config.context_parallel_size,
            "sequence_parallel": model_config.sequence_parallel,
            "expert_parallel_size": model_config.expert_model_parallel_size,
            "expert_tensor_parallel_size": model_config.expert_tensor_parallel_size,
            "pipeline_parallel_size": model_config.pipeline_model_parallel_size,
            "mtp_num_layers": model_config.mtp_num_layers,
            "mtp_use_repeated_layer": model_config.mtp_use_repeated_layer,
            "mtp_detach_heads": model_config.mtp_detach_heads,
        }
        if observed_topology != _topology():
            raise ParityError(f"runtime topology mismatch: {observed_topology}")
        if model_config.num_moe_experts != 128:
            raise ParityError(f"Nano must construct 128 experts, got {model_config.num_moe_experts}")
        worker_shared_prefix_enabled = worker._shared_prefix_training_enabled
        expected_shared_prefix_enabled = shared_prefix_mode == "train"
        if worker_shared_prefix_enabled is not expected_shared_prefix_enabled:
            raise ParityError(
                "worker shared-prefix activation differs from the arm contract: "
                f"expected={expected_shared_prefix_enabled} "
                f"actual={worker_shared_prefix_enabled}"
            )
        if arguments.arm == "on-mamba-replay":
            _install_mamba_replay_diagnostic()
        elif arguments.arm == "on-mamba-packed-replay":
            _install_mamba_packed_replay_diagnostic()
        runtime_counts, runtime_implementations, restore_runtime_counters = _install_shared_prefix_runtime_counters()

        pre_logprobs = _selected_logprobs(
            worker,
            rows,
            policy_config=policy_config,
            shared_prefix_mode=shared_prefix_mode,
        )
        calls_after_pre_logprob = _runtime_path_snapshot(runtime_counts)
        _require_runtime_path_counts(
            calls_after_pre_logprob,
            shared_prefix_enabled=expected_shared_prefix_enabled,
            phase="pre-logprob",
        )
        evidence = {
            "schema": SCHEMA,
            "arm": arguments.arm,
            "rank": rank,
            "local_rank": local_rank,
            "batch": batch_summary,
            "config_sha256": config_sha256,
            "shared_prefix_mode": shared_prefix_mode,
            "topology": observed_topology,
            "repo_root": str(repo_root),
            "model_path": str(model_path),
            "seed": arguments.seed,
            "batch_preparation": batch_preparation,
            "worker_shared_prefix_training_enabled": worker_shared_prefix_enabled,
            "runtime_path_implementations": runtime_implementations,
            "runtime_path_calls": {
                "after_pre_logprob": calls_after_pre_logprob,
            },
            "optimizer_master_reconstruction_self_test": master_reconstruction_self_test,
            "selected_logprobs_before": pre_logprobs,
            "diagnostic_only": arguments.arm in DIAGNOSTIC_ARMS,
        }

        if arguments.arm not in DIAGNOSTIC_ARMS:
            original_step = worker.optimizer.step

            def step_with_evidence(*args: Any, **kwargs: Any) -> Any:
                optimizer_probe["calls"] += 1
                if optimizer_probe["calls"] != 1:
                    raise ParityError("optimizer.step was called more than once")
                torch.cuda.synchronize()
                optimizer_probe["gradients"] = _snapshot_finalized_gradients(worker.model)
                optimizer_probe["optimizer_state_initialization"] = _initialize_optimizer_state_for_probe(
                    worker.model,
                    worker.optimizer,
                    worker.scheduler,
                )
                optimizer_probe["optimizer_masters_before"] = _snapshot_optimizer_masters(
                    worker.model, worker.optimizer
                )
                result = original_step(*args, **kwargs)
                torch.cuda.synchronize()
                optimizer_probe["optimizer_masters_after"] = _snapshot_optimizer_masters(worker.model, worker.optimizer)
                return result

            worker.optimizer.step = step_with_evidence
            scheduler_steps_before = worker.scheduler.num_steps
            loss_fn = ClippedPGLossFn(loss_config)
            worker.begin_train_step(loss_fn=loss_fn, gbs=BATCH_SIZE, mbs=1)
            try:
                train_batch, _ = _prepare_batch_for_worker(
                    _build_batch(rows),
                    policy_config=policy_config,
                    shared_prefix_mode=shared_prefix_mode,
                    stage="train",
                )
                worker.train_microbatch(train_batch)
                metrics = worker.finish_train_step()
            except Exception:
                worker.abort_train_step()
                raise
            calls_after_train = _runtime_path_snapshot(runtime_counts)
            _require_runtime_path_counts(
                calls_after_train,
                shared_prefix_enabled=expected_shared_prefix_enabled,
                phase="train",
                previous=calls_after_pre_logprob,
            )
            if optimizer_probe["calls"] != 1:
                raise ParityError(f"expected one optimizer step, got {optimizer_probe['calls']}")
            if worker.scheduler.num_steps != scheduler_steps_before + BATCH_SIZE:
                raise ParityError(
                    "scheduler did not advance by the captured global batch size: "
                    f"before={scheduler_steps_before} after={worker.scheduler.num_steps}"
                )
            evidence["training"] = _metric_evidence(metrics)
            evidence["gradients"] = optimizer_probe["gradients"]
            evidence["optimizer_state_initialization"] = optimizer_probe["optimizer_state_initialization"]
            evidence["optimizer_masters_before"] = optimizer_probe["optimizer_masters_before"]
            evidence["optimizer_masters_after"] = optimizer_probe["optimizer_masters_after"]
            evidence["selected_logprobs_after"] = _selected_logprobs(
                worker,
                rows,
                policy_config=policy_config,
                shared_prefix_mode=shared_prefix_mode,
            )
            calls_after_post_logprob = _runtime_path_snapshot(runtime_counts)
            _require_runtime_path_counts(
                calls_after_post_logprob,
                shared_prefix_enabled=expected_shared_prefix_enabled,
                phase="post-logprob",
                previous=calls_after_train,
            )
            evidence["runtime_path_calls"].update(
                {
                    "after_train": calls_after_train,
                    "after_post_logprob": calls_after_post_logprob,
                }
            )

        _atomic_json(output_dir / f"{arguments.arm}.rank{rank}.json", evidence)
        torch.distributed.barrier()
        if rank == 0:
            print(
                "NEMORL_CAPTURED_CITATION_ARM_GREEN "
                f"arm={arguments.arm} batch_sha256={batch_summary['sha256']} "
                f"selected_tokens={batch_summary['selected_tokens']} config_sha256={config_sha256}",
                flush=True,
            )
    finally:
        if restore_runtime_counters is not None:
            restore_runtime_counters()
        if worker is not None:
            worker.shutdown()
        destroy_parallel_state()


def _run_batch_preflight(arguments: argparse.Namespace) -> None:
    """Validate OFF/ON DP=1 batch preparation without loading Nano weights."""
    repo_root = Path(arguments.repo_root)
    try:
        resolved_repo_root = repo_root.resolve(strict=True)
    except OSError as error:
        raise ParityError(f"cannot resolve NeMo repo root {repo_root}: {error}") from error
    if (
        not repo_root.is_absolute()
        or not repo_root.is_dir()
        or repo_root.is_symlink()
        or resolved_repo_root != repo_root
    ):
        raise ParityError(f"NeMo repo root must be an absolute canonical non-symlink directory: {repo_root}")
    rows, batch_summary = _read_captured_rows(
        Path(arguments.batch),
        expected_sha256=arguments.expected_batch_sha256,
    )
    summaries: dict[str, Any] = {}
    fingerprints: set[str] = set()
    for mode in ("observe", "train"):
        policy_config, _loss_config, fingerprint = _build_policy_and_loss_config(
            arguments.model_path,
            repo_root=repo_root,
            shared_prefix_mode=mode,
        )
        summaries[mode] = _preflight_batch_preparation(
            rows,
            policy_config=policy_config,
            shared_prefix_mode=mode,
        )
        fingerprints.add(fingerprint)
    if len(fingerprints) != 1:
        raise ParityError(f"OFF/ON policy fingerprints differ after masking arm mode: {sorted(fingerprints)}")
    dense_orders = {stage: summaries["observe"][stage]["source_order"] for stage in ("logprob", "train")}
    print(
        "CAPTURED_CITATION_BATCH_PREFLIGHT_GREEN "
        f"batch_sha256={batch_summary['sha256']} rows={batch_summary['rows']} "
        f"dense_source_orders={json.dumps(dense_orders, separators=(',', ':'))} "
        f"config_sha256={next(iter(fingerprints))}",
        flush=True,
    )


def _load_evidence(root: Path, arm: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rank in range(WORLD_SIZE):
        path = root / f"{arm}.rank{rank}.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
            raise ParityError(f"missing, symlinked, or writable arm evidence: {path}")
        with path.open(encoding="utf-8", errors="strict") as stream:
            value = json.load(stream, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            raise ParityError(f"invalid evidence schema in {path}")
        if value.get("arm") != arm or value.get("rank") != rank:
            raise ParityError(f"arm/rank identity mismatch in {path}")
        results.append(value)
    return results


def _vector_metrics(reference: Sequence[float], candidate: Sequence[float], *, label: str) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise ParityError(f"{label} vectors must be nonempty and equal length: {len(reference)}/{len(candidate)}")
    ref = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(ref).all() or not np.isfinite(actual).all():
        raise ParityError(f"{label} contains non-finite values")
    difference = cast("NDArray[np.float64]", np.subtract(actual, ref))
    reference_norm = float(np.linalg.norm(ref))
    actual_norm = float(np.linalg.norm(actual))
    difference_norm = float(np.linalg.norm(difference))
    relative_l2 = (
        difference_norm / reference_norm if reference_norm else (0.0 if difference_norm == 0 else sys.float_info.max)
    )
    denominator = reference_norm * actual_norm
    cosine = (
        float(np.dot(ref, actual) / denominator) if denominator else (1.0 if reference_norm == actual_norm else 0.0)
    )
    return {
        "count": len(reference),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def _logprob_metrics(reference: Any, candidate: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(reference, list) or not isinstance(candidate, list):
        raise ParityError(f"{label} must contain per-sequence lists")
    if len(reference) != BATCH_SIZE or len(candidate) != BATCH_SIZE:
        raise ParityError(f"{label} must contain exactly K=4 sequences: {len(reference)}/{len(candidate)}")
    reference_sequences: list[list[float]] = []
    candidate_sequences: list[list[float]] = []
    for index, (reference_row, candidate_row) in enumerate(zip(reference, candidate, strict=True)):
        reference_values = _number_list(reference_row, label=f"{label} reference row {index}")
        candidate_values = _number_list(candidate_row, label=f"{label} candidate row {index}")
        if not reference_values or len(reference_values) != len(candidate_values):
            raise ParityError(
                f"{label} row {index} must be nonempty and equal width: "
                f"{len(reference_values)}/{len(candidate_values)}"
            )
        reference_sequences.append(reference_values)
        candidate_sequences.append(candidate_values)

    flattened_reference = [value for row in reference_sequences for value in row]
    flattened_candidate = [value for row in candidate_sequences for value in row]
    result: dict[str, Any] = _vector_metrics(flattened_reference, flattened_candidate, label=label)
    absolute_delta = cast(
        "NDArray[np.float64]",
        np.abs(
            np.subtract(
                np.asarray(flattened_candidate, dtype=np.float64),
                np.asarray(flattened_reference, dtype=np.float64),
            )
        ),
    )
    exp_was_clipped = bool(np.any(absolute_delta > 700.0))
    exp_absolute_delta = np.exp(np.minimum(absolute_delta, 700.0))
    sequence_mean_exp_absolute_delta: list[float] = []
    offset = 0
    for reference_row in reference_sequences:
        width = len(reference_row)
        sequence_mean_exp_absolute_delta.append(float(np.mean(exp_absolute_delta[offset : offset + width])))
        offset += width
    result.update(
        {
            "mean_exp_abs_delta": float(np.mean(exp_absolute_delta)),
            "max_exp_abs_delta": float(np.max(exp_absolute_delta)),
            "exp_abs_delta_clipped_at_700": exp_was_clipped,
            "sequence_mean_exp_abs_delta": sequence_mean_exp_absolute_delta,
            "max_sequence_mean_exp_abs_delta": max(sequence_mean_exp_absolute_delta),
            "filter_limit": LOGPROB_FILTER_LIMIT,
            "strict_limit": LOGPROB_STRICT_LIMIT,
        }
    )
    return result


def _family_gradient_vectors(evidence: dict[str, Any], family: str) -> tuple[list[float], list[float]]:
    gradients = evidence.get("gradients")
    if not isinstance(gradients, dict):
        raise ParityError(f"{evidence['arm']} rank {evidence['rank']} lacks gradients")
    names = sorted(name for name, item in gradients.items() if item.get("family") == family)
    if not names:
        raise ParityError(f"{evidence['arm']} rank {evidence['rank']} lacks {family} parameters")
    norms: list[float] = []
    samples: list[float] = []
    for name in names:
        item = gradients[name]
        if not item.get("present"):
            raise ParityError(f"finalized gradient is absent for required {family} parameter {name}")
        norms.append(_finite_float(item.get("l2"), label=f"{family} {name} l2"))
        samples.extend(_number_list(item.get("samples"), label=f"{family} {name} samples"))
    if math.sqrt(sum(value * value for value in norms)) == 0.0:
        raise ParityError(f"{evidence['arm']} rank {evidence['rank']} has zero {family} gradient")
    return norms, samples


def _validated_optimizer_storage(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityError(f"{label} optimizer storage is not a dictionary")
    kind = value.get("kind")
    if kind == "bf16-primary-int16-remainder":
        item = _exact_dict(
            value,
            keys={
                "kind",
                "primary_dtype",
                "primary_sha256",
                "remainder_dtype",
                "remainder_sha256",
            },
            label=label,
        )
        if item["primary_dtype"] != "torch.bfloat16" or item["remainder_dtype"] != "torch.int16":
            raise ParityError(f"{label} has invalid BF16/INT16 optimizer storage dtypes")
        _validated_sha256(item["primary_sha256"], label=f"{label} primary_sha256")
        _validated_sha256(item["remainder_sha256"], label=f"{label} remainder_sha256")
        return item
    if kind == "native-fp32-primary":
        item = _exact_dict(
            value,
            keys={"kind", "primary_dtype", "primary_sha256"},
            label=label,
        )
        if item["primary_dtype"] != "torch.float32":
            raise ParityError(f"{label} native primary is not FP32")
        _validated_sha256(item["primary_sha256"], label=f"{label} primary_sha256")
        return item
    if kind == "full-master":
        item = _exact_dict(
            value,
            keys={"kind", "master_dtype", "master_sha256"},
            label=label,
        )
        _validated_sha256(item["master_sha256"], label=f"{label} master_sha256")
        return item
    raise ParityError(f"{label} has unsupported optimizer storage kind {kind!r}")


def _validated_optimizer_master_snapshot(
    evidence: dict[str, Any],
    key: Literal["optimizer_masters_before", "optimizer_masters_after"],
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{evidence['arm']} rank {evidence['rank']} {key}"
    root = _exact_dict(
        evidence.get(key),
        keys={
            "parameters",
            "families",
            "local_shard_model_parameter_count",
            "peer_owned_model_parameter_count",
        },
        label=label,
    )
    parameters = root["parameters"]
    families = root["families"]
    if not isinstance(parameters, dict) or not isinstance(families, dict):
        raise ParityError(f"{label} has invalid parameter or family dictionaries")
    local_count = _int(
        root["local_shard_model_parameter_count"],
        label=f"{label} local_shard_model_parameter_count",
    )
    peer_count = _int(
        root["peer_owned_model_parameter_count"],
        label=f"{label} peer_owned_model_parameter_count",
    )
    if local_count <= 0 or local_count != len(parameters) or peer_count < 0:
        raise ParityError(
            f"{label} has invalid local/peer coverage: "
            f"local={local_count} parameters={len(parameters)} peer={peer_count}"
        )
    return parameters, families


def _family_update_evidence(evidence: dict[str, Any], family: str) -> dict[str, Any]:
    before, before_families = _validated_optimizer_master_snapshot(
        evidence,
        "optimizer_masters_before",
    )
    after, after_families = _validated_optimizer_master_snapshot(
        evidence,
        "optimizer_masters_after",
    )
    if set(before) != set(after):
        raise ParityError(f"{evidence['arm']} rank {evidence['rank']} changed optimizer master identities")
    names = sorted(name for name, item in before.items() if item.get("family") == family)
    updates: list[float] = []
    changed_tensors = 0
    changed_storage_tensors = 0
    changed_primary_tensors = 0
    changed_remainder_tensors = 0
    for name in names:
        if before[name].get("dtype") != "torch.float32" or after[name].get("dtype") != "torch.float32":
            raise ParityError(f"optimizer master for {name} is not recorded as FP32")
        invariants = ("family", "source", "sharded", "numel")
        if any(before[name].get(key) != after[name].get(key) for key in invariants):
            raise ParityError(f"optimizer master identity changed for {name}")
        old_storage = _validated_optimizer_storage(
            before[name].get("storage"),
            label=f"{name} before storage",
        )
        new_storage = _validated_optimizer_storage(
            after[name].get("storage"),
            label=f"{name} after storage",
        )
        storage_invariants = {key: value for key, value in old_storage.items() if not key.endswith("_sha256")}
        new_storage_invariants = {key: value for key, value in new_storage.items() if not key.endswith("_sha256")}
        if storage_invariants != new_storage_invariants:
            raise ParityError(f"optimizer persisted storage identity changed for {name}")
        old = _number_list(before[name].get("samples"), label=f"{name} before samples")
        new = _number_list(after[name].get("samples"), label=f"{name} after samples")
        if len(old) != len(new):
            raise ParityError(f"sampled optimizer master width changed for {name}")
        updates.extend(new_value - old_value for old_value, new_value in zip(old, new, strict=True))
        numel = _int(before[name].get("numel"), label=f"{name} numel")
        updates.extend(
            (
                (
                    _finite_float(after[name].get("sum"), label=f"{name} after sum")
                    - _finite_float(before[name].get("sum"), label=f"{name} before sum")
                )
                / math.sqrt(numel),
                _finite_float(after[name].get("l2"), label=f"{name} after l2")
                - _finite_float(before[name].get("l2"), label=f"{name} before l2"),
            )
        )
        changed_tensors += before[name].get("sha256") != after[name].get("sha256")
        changed_storage_tensors += old_storage != new_storage
        changed_primary_tensors += old_storage.get("primary_sha256") != new_storage.get("primary_sha256")
        changed_remainder_tensors += old_storage.get("remainder_sha256") != new_storage.get("remainder_sha256")
    if not updates:
        raise ParityError(f"no FP32 optimizer master snapshot for family {family}")
    if family not in before_families or family not in after_families:
        raise ParityError(f"optimizer master family summary is missing {family}")
    before_sha256 = before_families[family].get("sha256")
    after_sha256 = after_families[family].get("sha256")
    before_storage_sha256 = before_families[family].get("storage_sha256")
    after_storage_sha256 = after_families[family].get("storage_sha256")
    _validated_sha256(before_sha256, label=f"{family} before logical master sha256")
    _validated_sha256(after_sha256, label=f"{family} after logical master sha256")
    _validated_sha256(before_storage_sha256, label=f"{family} before persisted storage sha256")
    _validated_sha256(after_storage_sha256, label=f"{family} after persisted storage sha256")
    return {
        "vector": updates,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
        "hash_changed": before_sha256 != after_sha256,
        "before_storage_sha256": before_storage_sha256,
        "after_storage_sha256": after_storage_sha256,
        "storage_hash_changed": before_storage_sha256 != after_storage_sha256,
        "changed_tensor_count": changed_tensors,
        "changed_storage_tensor_count": changed_storage_tensors,
        "changed_primary_tensor_count": changed_primary_tensors,
        "changed_remainder_tensor_count": changed_remainder_tensors,
        "tensor_count": len(names),
        "numel": sum(_int(before[name].get("numel"), label=f"{name} numel") for name in names),
    }


def _aggregate_family_optimizer_updates(
    off: Sequence[dict[str, Any]],
    on: Sequence[dict[str, Any]],
    family: str,
) -> dict[str, Any]:
    """Aggregate rank-local optimizer shards in deterministic rank/name order."""
    if len(off) != WORLD_SIZE or len(on) != WORLD_SIZE:
        raise ParityError(f"{family} optimizer aggregation requires {WORLD_SIZE} OFF/ON ranks")
    off_vector: list[float] = []
    on_vector: list[float] = []
    off_before_digest = hashlib.sha256()
    on_before_digest = hashlib.sha256()
    off_after_digest = hashlib.sha256()
    on_after_digest = hashlib.sha256()
    off_counts = {
        "changed_tensor_count": 0,
        "changed_storage_tensor_count": 0,
        "changed_primary_tensor_count": 0,
        "changed_remainder_tensor_count": 0,
        "tensor_count": 0,
        "numel": 0,
    }
    on_counts = {key: 0 for key in off_counts}
    rank_reports: list[dict[str, Any]] = []

    for rank, (dense, shared) in enumerate(zip(off, on, strict=True)):
        off_before, _off_before_families = _validated_optimizer_master_snapshot(
            dense,
            "optimizer_masters_before",
        )
        on_before, _on_before_families = _validated_optimizer_master_snapshot(
            shared,
            "optimizer_masters_before",
        )
        off_after, _off_after_families = _validated_optimizer_master_snapshot(
            dense,
            "optimizer_masters_after",
        )
        on_after, _on_after_families = _validated_optimizer_master_snapshot(
            shared,
            "optimizer_masters_after",
        )
        off_names = sorted(name for name, item in off_before.items() if item.get("family") == family)
        on_names = sorted(name for name, item in on_before.items() if item.get("family") == family)
        if off_names != on_names:
            raise ParityError(
                f"rank {rank} {family} OFF/ON local optimizer shard identities differ: "
                f"{off_names[:5]}/{on_names[:5]}"
            )
        if any(name not in off_after or name not in on_after for name in off_names):
            raise ParityError(f"rank {rank} {family} post-step optimizer snapshot lost a shard")

        for name in off_names:
            if off_before[name] != on_before[name]:
                raise ParityError(
                    f"rank {rank} {family} initial OFF/ON logical master or persisted storage " f"differs for {name}"
                )
            for digest, value in (
                (off_before_digest, off_before[name]),
                (on_before_digest, on_before[name]),
                (off_after_digest, off_after[name]),
                (on_after_digest, on_after[name]),
            ):
                digest.update(_canonical_json_bytes({"rank": rank, "name": name, "value": value}))

        if not off_names:
            rank_reports.append({"rank": rank, "local_tensor_count": 0})
            continue
        off_update = _family_update_evidence(dense, family)
        on_update = _family_update_evidence(shared, family)
        off_vector.extend(off_update["vector"])
        on_vector.extend(on_update["vector"])
        for key in off_counts:
            off_counts[key] += _int(off_update[key], label=f"rank {rank} OFF {family} {key}")
            on_counts[key] += _int(on_update[key], label=f"rank {rank} ON {family} {key}")
        rank_reports.append(
            {
                "rank": rank,
                "local_tensor_count": len(off_names),
                "off_changed_tensor_count": off_update["changed_tensor_count"],
                "on_changed_tensor_count": on_update["changed_tensor_count"],
            }
        )

    if not off_vector or not on_vector or off_counts["tensor_count"] <= 0:
        raise ParityError(f"no globally owned optimizer master shard for required family {family}")
    initial_off_sha256 = off_before_digest.hexdigest()
    initial_on_sha256 = on_before_digest.hexdigest()
    if initial_off_sha256 != initial_on_sha256:
        raise ParityError(f"{family} aggregate initial OFF/ON optimizer fingerprints differ")
    metrics = _vector_metrics(
        off_vector,
        on_vector,
        label=f"global {family} FP32 optimizer master updates",
    )
    return {
        "initial_off_sha256": initial_off_sha256,
        "initial_on_sha256": initial_on_sha256,
        "initial_exact_match": True,
        "off_after_sha256": off_after_digest.hexdigest(),
        "on_after_sha256": on_after_digest.hexdigest(),
        "off": off_counts,
        "on": on_counts,
        "update_projection": metrics,
        "ranks": rank_reports,
    }


def _optimizer_update_aggregation_self_test() -> dict[str, int]:
    """Pin global-family aggregation and exact OFF/ON initial-state parity."""

    def digest(character: str) -> str:
        return character * 64

    def parameter_item(*, family: str, after: bool) -> dict[str, Any]:
        return {
            "family": family,
            "source": "te-bf16-int16-remainder",
            "dtype": "torch.float32",
            "sharded": True,
            "numel": 1,
            "sha256": digest("2" if after else "1"),
            "sum": 1.25 if after else 1.0,
            "l2": 1.25 if after else 1.0,
            "samples": [1.25 if after else 1.0],
            "storage": {
                "kind": "bf16-primary-int16-remainder",
                "primary_dtype": "torch.bfloat16",
                "primary_sha256": digest("4" if after else "3"),
                "remainder_dtype": "torch.int16",
                "remainder_sha256": digest("6" if after else "5"),
            },
        }

    def evidence(*, arm: str, rank: int) -> dict[str, Any]:
        family = "attention" if rank == 0 else "other"
        name = f"chunk0.rank{rank}.{family}.weight"
        before_item = parameter_item(family=family, after=False)
        after_item = parameter_item(family=family, after=True)
        before_family = {
            "sha256": before_item["sha256"],
            "storage_sha256": digest("7"),
            "tensor_count": 1,
            "numel": 1,
            "missing_model_parameter_count": 0,
        }
        after_family = {
            "sha256": after_item["sha256"],
            "storage_sha256": digest("8"),
            "tensor_count": 1,
            "numel": 1,
            "missing_model_parameter_count": 0,
        }
        return {
            "arm": arm,
            "rank": rank,
            "optimizer_masters_before": {
                "parameters": {name: before_item},
                "families": {family: before_family},
                "local_shard_model_parameter_count": 1,
                "peer_owned_model_parameter_count": WORLD_SIZE - 1,
            },
            "optimizer_masters_after": {
                "parameters": {name: after_item},
                "families": {family: after_family},
                "local_shard_model_parameter_count": 1,
                "peer_owned_model_parameter_count": WORLD_SIZE - 1,
            },
        }

    off = [evidence(arm="off", rank=rank) for rank in range(WORLD_SIZE)]
    on = [evidence(arm="on", rank=rank) for rank in range(WORLD_SIZE)]
    result = _aggregate_family_optimizer_updates(off, on, "attention")
    if (
        result["off"]["tensor_count"] != 1
        or result["on"]["changed_remainder_tensor_count"] != 1
        or result["update_projection"]["relative_l2"] != 0.0
    ):
        raise ParityError("optimizer aggregation self-test did not preserve global updates")

    rejected = 0
    try:
        _aggregate_family_optimizer_updates(off, on, "mtp")
    except ParityError:
        rejected += 1
    else:
        raise ParityError("optimizer aggregation self-test accepted an absent global family")

    mismatched = copy.deepcopy(on)
    mismatched[0]["optimizer_masters_before"]["parameters"]["chunk0.rank0.attention.weight"]["samples"] = [0.5]
    try:
        _aggregate_family_optimizer_updates(off, mismatched, "attention")
    except ParityError:
        rejected += 1
    else:
        raise ParityError("optimizer aggregation self-test accepted mismatched initial masters")
    return {"green_cases": 1, "rejected_cases": rejected}


def _record_check(
    report: dict[str, Any],
    *,
    label: str,
    passed: bool,
    details: Any,
) -> None:
    report["checks"][label] = {"passed": bool(passed), "details": details}
    if not passed:
        report["failures"].append({"check": label, "details": details})


def _exact_dict(value: Any, *, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be an object")
    actual_keys = set(value)
    if actual_keys != keys:
        raise ParityError(
            f"{label} schema differs: missing={sorted(keys - actual_keys)} " f"extra={sorted(actual_keys - keys)}"
        )
    return value


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ParityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validated_batch_summary(value: Any, *, arm: str, rank: int) -> dict[str, Any]:
    label = f"{arm} rank {rank} batch"
    batch = _exact_dict(
        value,
        keys={
            "sha256",
            "rows",
            "width",
            "input_lengths",
            "prompt_length",
            "rewards",
            "selected_tokens",
        },
        label=label,
    )
    _validated_sha256(batch["sha256"], label=f"{label} sha256")
    if _int(batch["rows"], label=f"{label} rows") != BATCH_SIZE:
        raise ParityError(f"{label} must describe exactly K=4 rows")
    width = _int(batch["width"], label=f"{label} width")
    prompt_length = _int(batch["prompt_length"], label=f"{label} prompt_length")
    input_lengths = _int_list(batch["input_lengths"], label=f"{label} input_lengths")
    rewards = _number_list(batch["rewards"], label=f"{label} rewards")
    selected_tokens = _int(batch["selected_tokens"], label=f"{label} selected_tokens")
    if width <= 0 or prompt_length <= 0 or selected_tokens <= 0:
        raise ParityError(f"{label} has nonpositive dimensions or selected-token count")
    if len(input_lengths) != BATCH_SIZE or any(
        not prompt_length < input_length <= width for input_length in input_lengths
    ):
        raise ParityError(f"{label} has invalid K=4 input lengths")
    if len(rewards) != BATCH_SIZE or len(set(rewards)) < 2:
        raise ParityError(f"{label} rewards must contain four nondegenerate values")
    return batch


def _validated_topology_evidence(value: Any, *, arm: str, rank: int) -> dict[str, Any]:
    label = f"{arm} rank {rank} topology"
    expected = _topology()
    topology = _exact_dict(value, keys=set(expected), label=label)
    boolean_keys = ("sequence_parallel", "mtp_use_repeated_layer", "mtp_detach_heads")
    integer_keys = tuple(key for key in expected if key not in boolean_keys)
    for key in integer_keys:
        if _int(topology[key], label=f"{label} {key}") != expected[key]:
            raise ParityError(f"{label} {key} differs: {topology[key]!r} != {expected[key]!r}")
    for key in boolean_keys:
        if topology[key] is not expected[key]:
            raise ParityError(f"{label} {key} differs: {topology[key]!r} is not {expected[key]!r}")
    return topology


def _validated_batch_preparation_evidence(
    value: Any,
    *,
    arm: Literal["off", "on", "on-mamba-replay", "on-mamba-packed-replay"],
    rank: int,
    batch: dict[str, Any],
) -> dict[str, Any]:
    """Validate the serialized proof of the exact DP=1 preparation contract."""
    label = f"{arm} rank {rank} batch preparation"
    preparation = _exact_dict(value, keys={"logprob", "train"}, label=label)
    input_lengths = _int_list(batch["input_lengths"], label=f"{label} input_lengths")
    prompt_length = _int(batch["prompt_length"], label=f"{label} prompt_length")
    padding_multiple = TP_SIZE * CP_SIZE * 2
    padded_lengths = [_round_up(length, padding_multiple) for length in input_lengths]
    dense_padded_tokens = sum(padded_lengths)
    shared_physical_length = prompt_length + sum(padded_length - prompt_length for padded_length in padded_lengths)
    shared_row_order = sorted(
        range(BATCH_SIZE),
        key=lambda row: (
            -(padded_lengths[row] - prompt_length),
            -(input_lengths[row] - prompt_length),
            row,
        ),
    )
    shared = arm != "off"

    validated_stages: dict[str, Any] = {}
    for stage in ("logprob", "train"):
        stage_label = f"{label} {stage}"
        stage_value = _exact_dict(
            preparation[stage],
            keys={
                "capacity",
                "microbatches",
                "source_order",
                "plan",
                "worker_num_microbatches",
            },
            label=stage_label,
        )
        capacity = _int(stage_value["capacity"], label=f"{stage_label} capacity")
        worker_num_microbatches = _int(
            stage_value["worker_num_microbatches"],
            label=f"{stage_label} worker_num_microbatches",
        )
        if capacity != PACKING_TOKENS or worker_num_microbatches != 1:
            raise ParityError(f"{stage_label} must use capacity={PACKING_TOKENS} and one worker microbatch")

        if shared:
            if stage_value["microbatches"] is not None or stage_value["source_order"] is not None:
                raise ParityError(f"{stage_label} must bypass dense metadata and retain DP=1 source order")
            plan = _exact_dict(
                stage_value["plan"],
                keys={
                    "execution_units",
                    "row_indices",
                    "slot_ids",
                    "shared_layout",
                    "physical_length",
                    "capacity",
                },
                label=f"{stage_label} plan",
            )
            if (
                _int(plan["execution_units"], label=f"{stage_label} execution_units") != 1
                or _int_list(plan["row_indices"], label=f"{stage_label} row_indices") != shared_row_order
                or _int_list(plan["slot_ids"], label=f"{stage_label} slot_ids") != [0] * BATCH_SIZE
                or plan["shared_layout"] is not True
                or _int(plan["physical_length"], label=f"{stage_label} physical_length") != shared_physical_length
                or _int(plan["capacity"], label=f"{stage_label} plan capacity") != PACKING_TOKENS
                or shared_physical_length > PACKING_TOKENS
            ):
                raise ParityError(f"{stage_label} is not the exact non-fallback K=4 shared star")
        else:
            if _int(stage_value["microbatches"], label=f"{stage_label} microbatches") != 1:
                raise ParityError(f"{stage_label} must contain one dense packed microbatch")
            source_order = _int_list(stage_value["source_order"], label=f"{stage_label} source_order")
            if sorted(source_order) != list(range(BATCH_SIZE)):
                raise ParityError(f"{stage_label} source order is not a K=4 permutation")
            plan = _exact_dict(
                stage_value["plan"],
                keys={
                    "source_order",
                    "micro_batch_indices",
                    "micro_batch_lengths",
                    "padded_tokens",
                    "capacity",
                },
                label=f"{stage_label} plan",
            )
            if (
                _int_list(plan["source_order"], label=f"{stage_label} plan source_order") != source_order
                or plan["micro_batch_indices"] != [[[0, BATCH_SIZE]]]
                or plan["micro_batch_lengths"] != [[dense_padded_tokens]]
                or _int(plan["padded_tokens"], label=f"{stage_label} padded_tokens") != dense_padded_tokens
                or _int(plan["capacity"], label=f"{stage_label} plan capacity") != PACKING_TOKENS
                or dense_padded_tokens > PACKING_TOKENS
            ):
                raise ParityError(f"{stage_label} is not the exact one-bin dense K=4 plan")
        validated_stages[stage] = stage_value
    return validated_stages


def _validated_optimizer_state_initialization(
    value: Any,
    *,
    arm: str,
    rank: int,
) -> dict[str, Any]:
    """Validate proof that official lazy-state init was not an optimizer step."""
    label = f"{arm} rank {rank} optimizer state initialization"
    root = _exact_dict(
        value,
        keys={
            "method",
            "mcore_init_state_fn_used",
            "mcore_init_state_fn_incompatibility",
            "store_param_remainders",
            "initialized_parts",
            "initialized_parameters",
            "bf16_remainder_parameters",
            "parameter_coverage",
            "primary_before",
            "primary_after",
            "gradients_before",
            "gradients_after",
            "group_steps_before",
            "group_steps_after",
            "scheduler_steps_before",
            "scheduler_steps_after",
            "unchanged",
        },
        label=label,
    )
    if root["method"] != "transformer-engine-initialize_state":
        raise ParityError(f"{label} did not use pinned TE initialize_state")
    if (
        root["mcore_init_state_fn_used"] is not False
        or root["mcore_init_state_fn_incompatibility"] != "calls-te-initialize_state-without-required-remainder-flag"
    ):
        raise ParityError(f"{label} did not record the pinned MCore/TE initializer incompatibility")
    if root["store_param_remainders"] is not True or root["unchanged"] is not True:
        raise ParityError(f"{label} did not retain exact-recipe remainder storage or no-mutation proof")

    initialized_parts = _int(root["initialized_parts"], label=f"{label} initialized_parts")
    initialized_parameters = _int(
        root["initialized_parameters"],
        label=f"{label} initialized_parameters",
    )
    bf16_remainder_parameters = _int(
        root["bf16_remainder_parameters"],
        label=f"{label} bf16_remainder_parameters",
    )
    if (
        initialized_parts <= 0
        or initialized_parameters <= 0
        or not 0 < bf16_remainder_parameters <= initialized_parameters
    ):
        raise ParityError(f"{label} has invalid initialization coverage")

    parameter_coverage = _exact_dict(
        root["parameter_coverage"],
        keys={
            "method",
            "trainable_model_parameter_count",
            "full_owned_model_parameter_count",
            "local_shard_model_parameter_count",
            "peer_owned_model_parameter_count",
            "full_family_counts",
            "local_shard_family_counts",
            "parts",
        },
        label=f"{label} parameter coverage",
    )
    if parameter_coverage["method"] != "ddp-full-buffer-ownership-plus-local-shard-bijection":
        raise ParityError(f"{label} used an unexpected parameter-coverage audit")
    trainable_count = _int(
        parameter_coverage["trainable_model_parameter_count"],
        label=f"{label} trainable_model_parameter_count",
    )
    full_owned_count = _int(
        parameter_coverage["full_owned_model_parameter_count"],
        label=f"{label} full_owned_model_parameter_count",
    )
    local_shard_count = _int(
        parameter_coverage["local_shard_model_parameter_count"],
        label=f"{label} local_shard_model_parameter_count",
    )
    peer_owned_count = _int(
        parameter_coverage["peer_owned_model_parameter_count"],
        label=f"{label} peer_owned_model_parameter_count",
    )
    if (
        trainable_count <= 0
        or full_owned_count != trainable_count
        or not 0 < local_shard_count <= trainable_count
        or peer_owned_count != trainable_count - local_shard_count
        or initialized_parameters != local_shard_count
    ):
        raise ParityError(
            f"{label} parameter coverage counts disagree: trainable={trainable_count} "
            f"full={full_owned_count} local={local_shard_count} peer={peer_owned_count} "
            f"initialized={initialized_parameters}"
        )
    full_family_counts = parameter_coverage["full_family_counts"]
    local_family_counts = parameter_coverage["local_shard_family_counts"]
    if not isinstance(full_family_counts, dict) or not isinstance(local_family_counts, dict):
        raise ParityError(f"{label} parameter family coverage is not a dictionary")
    validated_full_family_counts = {
        family: _int(count, label=f"{label} full family {family}") for family, count in full_family_counts.items()
    }
    validated_local_family_counts = {
        family: _int(count, label=f"{label} local family {family}") for family, count in local_family_counts.items()
    }
    if (
        any(count <= 0 for count in validated_full_family_counts.values())
        or any(count <= 0 for count in validated_local_family_counts.values())
        or sum(validated_full_family_counts.values()) != trainable_count
        or sum(validated_local_family_counts.values()) != local_shard_count
        or any(validated_full_family_counts.get(family, 0) <= 0 for family in REQUIRED_GRADIENT_FAMILIES)
    ):
        raise ParityError(f"{label} parameter family coverage is incomplete or inconsistent")
    parts = parameter_coverage["parts"]
    if not isinstance(parts, list) or not parts:
        raise ParityError(f"{label} parameter coverage has no optimizer parts")
    validated_parts: list[dict[str, int]] = []
    for expected_part, raw_part in enumerate(parts):
        part = _exact_dict(
            raw_part,
            keys={
                "part",
                "full_model_parameter_count",
                "local_shard_model_parameter_count",
                "raw_optimizer_shard_count",
            },
            label=f"{label} parameter coverage part {expected_part}",
        )
        part_index = _int(part["part"], label=f"{label} part index")
        part_full = _int(part["full_model_parameter_count"], label=f"{label} part full")
        part_local = _int(part["local_shard_model_parameter_count"], label=f"{label} part local")
        part_raw = _int(part["raw_optimizer_shard_count"], label=f"{label} part raw")
        if part_index != expected_part or part_full < 0 or part_local < 0 or part_raw != part_local:
            raise ParityError(f"{label} optimizer part coverage is invalid")
        validated_parts.append({"part": part_index, "full": part_full, "local": part_local, "raw": part_raw})
    if (
        sum(part["full"] for part in validated_parts) != full_owned_count
        or sum(part["local"] for part in validated_parts) != local_shard_count
        or sum(part["raw"] > 0 for part in validated_parts) != initialized_parts
    ):
        raise ParityError(f"{label} optimizer part totals disagree with initialization")

    def validated_fingerprint(
        fingerprint: Any,
        *,
        fingerprint_label: str,
        includes_absent: bool,
    ) -> dict[str, Any]:
        keys = {"sha256", "tensor_count", "numel"}
        if includes_absent:
            keys.add("absent_count")
        item = _exact_dict(fingerprint, keys=keys, label=fingerprint_label)
        _validated_sha256(item["sha256"], label=f"{fingerprint_label} sha256")
        if _int(item["tensor_count"], label=f"{fingerprint_label} tensor_count") <= 0:
            raise ParityError(f"{fingerprint_label} tensor_count must be positive")
        if _int(item["numel"], label=f"{fingerprint_label} numel") <= 0:
            raise ParityError(f"{fingerprint_label} numel must be positive")
        if includes_absent and _int(item["absent_count"], label=f"{fingerprint_label} absent_count") < 0:
            raise ParityError(f"{fingerprint_label} absent_count must be nonnegative")
        return item

    primary_before = validated_fingerprint(
        root["primary_before"],
        fingerprint_label=f"{label} primary_before",
        includes_absent=False,
    )
    primary_after = validated_fingerprint(
        root["primary_after"],
        fingerprint_label=f"{label} primary_after",
        includes_absent=False,
    )
    gradients_before = validated_fingerprint(
        root["gradients_before"],
        fingerprint_label=f"{label} gradients_before",
        includes_absent=True,
    )
    gradients_after = validated_fingerprint(
        root["gradients_after"],
        fingerprint_label=f"{label} gradients_after",
        includes_absent=True,
    )
    if primary_before != primary_after or gradients_before != gradients_after:
        raise ParityError(f"{label} fingerprints changed")

    group_steps_before = root["group_steps_before"]
    group_steps_after = root["group_steps_after"]
    if not isinstance(group_steps_before, list) or not group_steps_before or group_steps_after != group_steps_before:
        raise ParityError(f"{label} optimizer group steps changed or are absent")
    for index, group_step in enumerate(group_steps_before):
        item = _exact_dict(
            group_step,
            keys={"part", "group", "present"},
            label=f"{label} group step {index}",
        )
        _int(item["part"], label=f"{label} group step {index} part")
        _int(item["group"], label=f"{label} group step {index} group")
        if item["present"] is not False:
            raise ParityError(f"{label} group step {index} existed before the first real step")

    scheduler_steps_before = _int(
        root["scheduler_steps_before"],
        label=f"{label} scheduler_steps_before",
    )
    scheduler_steps_after = _int(
        root["scheduler_steps_after"],
        label=f"{label} scheduler_steps_after",
    )
    if scheduler_steps_before != scheduler_steps_after:
        raise ParityError(f"{label} scheduler steps changed")
    return {
        "method": root["method"],
        "mcore_init_state_fn_used": False,
        "mcore_init_state_fn_incompatibility": root["mcore_init_state_fn_incompatibility"],
        "store_param_remainders": True,
        "initialized_parts": initialized_parts,
        "initialized_parameters": initialized_parameters,
        "bf16_remainder_parameters": bf16_remainder_parameters,
        "parameter_coverage": {
            "trainable": trainable_count,
            "full_owned": full_owned_count,
            "local_shards": local_shard_count,
            "peer_owned": peer_owned_count,
            "full_family_counts": validated_full_family_counts,
            "local_family_counts": validated_local_family_counts,
            "parts": validated_parts,
        },
        "primary_sha256": primary_before["sha256"],
        "gradient_sha256": gradients_before["sha256"],
        "group_count": len(group_steps_before),
        "scheduler_steps": scheduler_steps_before,
        "unchanged": True,
    }


def _validated_arm_contract(
    evidence: dict[str, Any],
    *,
    arm: Literal["off", "on", "on-mamba-replay"],
    rank: int,
) -> dict[str, Any]:
    common_keys = {
        "schema",
        "arm",
        "rank",
        "local_rank",
        "batch",
        "config_sha256",
        "shared_prefix_mode",
        "topology",
        "repo_root",
        "model_path",
        "seed",
        "batch_preparation",
        "worker_shared_prefix_training_enabled",
        "runtime_path_implementations",
        "runtime_path_calls",
        "optimizer_master_reconstruction_self_test",
        "selected_logprobs_before",
        "diagnostic_only",
    }
    training_keys = {
        "training",
        "gradients",
        "optimizer_state_initialization",
        "optimizer_masters_before",
        "optimizer_masters_after",
        "selected_logprobs_after",
    }
    expected_keys = common_keys if arm in DIAGNOSTIC_ARMS else common_keys | training_keys
    _exact_dict(evidence, keys=expected_keys, label=f"{arm} rank {rank} evidence")
    if evidence["schema"] != SCHEMA or evidence["arm"] != arm:
        raise ParityError(f"{arm} rank {rank} has invalid schema or arm identity")
    if (
        _int(evidence["rank"], label=f"{arm} rank") != rank
        or _int(evidence["local_rank"], label=f"{arm} local_rank") != rank
    ):
        raise ParityError(f"{arm} rank {rank} rank/local-rank identity differs")
    if evidence["diagnostic_only"] is not (arm in DIAGNOSTIC_ARMS):
        raise ParityError(f"{arm} rank {rank} diagnostic_only flag differs")
    expected_self_test = _master_remainder_reconstruction_self_test()
    if evidence["optimizer_master_reconstruction_self_test"] != expected_self_test:
        raise ParityError(f"{arm} rank {rank} has invalid FP32 master reconstruction self-test")
    expected_mode = "observe" if arm == "off" else "train"
    if evidence["shared_prefix_mode"] != expected_mode:
        raise ParityError(f"{arm} rank {rank} shared-prefix mode differs")
    _validated_topology_evidence(evidence["topology"], arm=arm, rank=rank)
    if _int(evidence["seed"], label=f"{arm} rank {rank} seed") != 42:
        raise ParityError(f"{arm} rank {rank} seed must be 42")
    _validated_sha256(evidence["config_sha256"], label=f"{arm} rank {rank} config_sha256")
    for path_key in ("repo_root", "model_path"):
        path_value = evidence[path_key]
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ParityError(f"{arm} rank {rank} {path_key} must be an absolute path")
    batch = _validated_batch_summary(evidence["batch"], arm=arm, rank=rank)
    preparation = _validated_batch_preparation_evidence(
        evidence["batch_preparation"],
        arm=arm,
        rank=rank,
        batch=batch,
    )
    result = {"batch": batch, "batch_preparation": preparation}
    if arm not in DIAGNOSTIC_ARMS:
        result["optimizer_state_initialization"] = _validated_optimizer_state_initialization(
            evidence["optimizer_state_initialization"],
            arm=arm,
            rank=rank,
        )
    return result


def _validated_runtime_path_evidence(
    evidence: dict[str, Any],
    *,
    arm: Literal["off", "on", "on-mamba-replay", "on-mamba-packed-replay"],
) -> dict[str, Any]:
    expected_enabled = arm != "off"
    if evidence.get("worker_shared_prefix_training_enabled") is not expected_enabled:
        raise ParityError(
            f"{arm} worker shared-prefix flag is " f"{evidence.get('worker_shared_prefix_training_enabled')!r}"
        )
    implementations = evidence.get("runtime_path_implementations")
    expected_implementations = {
        "hybrid_stack": "forward_hybrid_stack_shared_prefix",
        "mamba_cp": (
            "_forward_mamba_layer_shared_prefix_cp_replay"
            if arm == "on-mamba-replay"
            else (
                "_forward_mamba_layer_shared_prefix_cp_packed_fused_oracle"
                if arm == "on-mamba-packed-replay"
                else "_forward_mamba_layer_shared_prefix_cp"
            )
        ),
        "mamba_variant": (
            "mamba_decomposed_replay"
            if arm == "on-mamba-replay"
            else ("mamba_packed_fused" if arm == "on-mamba-packed-replay" else "mamba_state_fork")
        ),
        "attention_cp": "flash_composed_forest_attention_cp",
    }
    if implementations != expected_implementations:
        raise ParityError(
            f"{arm} runtime implementations differ: expected="
            f"{expected_implementations!r} actual={implementations!r}"
        )
    raw_phases = evidence.get("runtime_path_calls")
    expected_phases = (
        {"after_pre_logprob"} if arm in DIAGNOSTIC_ARMS else {"after_pre_logprob", "after_train", "after_post_logprob"}
    )
    if not isinstance(raw_phases, dict) or set(raw_phases) != expected_phases:
        raise ParityError(
            f"{arm} runtime counter phases differ: expected={sorted(expected_phases)} "
            f"actual={sorted(raw_phases) if isinstance(raw_phases, dict) else raw_phases!r}"
        )
    phases: dict[str, dict[str, int]] = {}
    for phase, raw_counts in raw_phases.items():
        if not isinstance(raw_counts, dict) or set(raw_counts) != set(RUNTIME_PATH_COUNTERS):
            raise ParityError(f"{arm} {phase} has invalid runtime counters")
        phases[phase] = {name: _int(raw_counts[name], label=f"{arm} {phase} {name}") for name in RUNTIME_PATH_COUNTERS}
    pre = phases["after_pre_logprob"]
    if arm == "off":
        if any(count for phase in phases.values() for count in phase.values()):
            raise ParityError(f"OFF called shared-prefix runtime paths: {phases!r}")
    else:
        if any(pre[name] <= 0 for name in RUNTIME_REQUIRED_COUNTERS):
            raise ParityError(f"{arm} did not execute every shared-prefix runtime path: {pre!r}")
        expected_variant = expected_implementations["mamba_variant"]
        if pre[expected_variant] != pre["mamba_cp"] or any(
            pre[name] != 0 for name in MAMBA_RUNTIME_VARIANTS if name != expected_variant
        ):
            raise ParityError(f"{arm} Mamba runtime attestation is invalid: {pre!r}")
    if arm == "on":
        train = phases["after_train"]
        post = phases["after_post_logprob"]
        active_counters = (*RUNTIME_REQUIRED_COUNTERS, expected_implementations["mamba_variant"])
        if any(not pre[name] < train[name] < post[name] for name in active_counters):
            raise ParityError(f"ON runtime paths did not increase in every phase: {phases!r}")
        if any(post[name] != 0 for name in MAMBA_RUNTIME_VARIANTS if name != expected_implementations["mamba_variant"]):
            raise ParityError(f"ON called more than one Mamba implementation: {phases!r}")
    return {
        "worker_shared_prefix_training_enabled": expected_enabled,
        "implementations": implementations,
        "calls": phases,
    }


def _compare(arguments: argparse.Namespace) -> None:
    root = Path(arguments.evidence_dir)
    if not root.is_absolute() or not root.is_dir():
        raise ParityError(f"evidence directory must be absolute: {root}")
    output = Path(arguments.output)
    if not output.is_absolute() or output.parent != root:
        raise ParityError("comparison output must be an absolute direct child of the evidence directory")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RED",
        "topology": _topology(),
        "thresholds": {
            "logprob_filter_max_sequence_mean_exp_abs_delta": LOGPROB_FILTER_LIMIT,
            "logprob_strict_max_sequence_mean_exp_abs_delta": LOGPROB_STRICT_LIMIT,
            "logprob_relative_l2": 0.03,
            "logprob_cosine": 0.995,
            "gradient_norm_relative_l2": 0.05,
            "gradient_norm_cosine": 0.995,
            "gradient_sample_relative_l2": 0.10,
            "gradient_sample_cosine": 0.99,
            "optimizer_update_relative_l2": 0.10,
            "optimizer_update_cosine": 0.99,
        },
        "checks": {},
        "failures": [],
        "ranks": {},
    }
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm in ("off", "on", "on-mamba-replay"):
        try:
            arms[arm] = _load_evidence(root, arm)
        except (ParityError, OSError, ValueError, TypeError, KeyError) as error:
            _record_check(
                report,
                label=f"load evidence {arm}",
                passed=False,
                details=str(error),
            )
    if len(arms) != 3:
        _atomic_json(output, report)
        raise ParityError(f"sealed RED comparison report at {output}")

    off = arms["off"]
    on = arms["on"]
    replay = arms["on-mamba-replay"]
    validated_preparations: dict[str, list[dict[str, Any] | None]] = {}
    for arm in ("off", "on", "on-mamba-replay"):
        validated_preparations[arm] = [cast(dict[str, Any] | None, None) for _ in range(WORLD_SIZE)]
    for rank in range(WORLD_SIZE):
        dense = off[rank]
        shared = on[rank]
        replay_shared = replay[rank]
        rank_report: dict[str, Any] = {"families": {}}
        report["ranks"][str(rank)] = rank_report
        contract_reports: dict[str, Any] = {}
        for arm_name, arm_evidence in (
            ("off", dense),
            ("on", shared),
            ("on-mamba-replay", replay_shared),
        ):
            try:
                contract = _validated_arm_contract(
                    arm_evidence,
                    arm=arm_name,
                    rank=rank,
                )
                contract_reports[arm_name] = contract
                validated_preparations[arm_name][rank] = contract["batch_preparation"]
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} evidence contract",
                    passed=True,
                    details=contract,
                )
            except (ParityError, KeyError, TypeError, ValueError) as error:
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} evidence contract",
                    passed=False,
                    details=str(error),
                )
        rank_report["contracts"] = contract_reports
        off_coverage = (
            contract_reports.get("off", {}).get("optimizer_state_initialization", {}).get("parameter_coverage")
        )
        on_coverage = contract_reports.get("on", {}).get("optimizer_state_initialization", {}).get("parameter_coverage")
        _record_check(
            report,
            label=f"rank {rank} OFF/ON optimizer ownership topology",
            passed=off_coverage is not None and off_coverage == on_coverage,
            details={"off": off_coverage, "on": on_coverage},
        )
        _record_check(
            report,
            label=f"rank {rank} ON/replay batch preparation",
            passed=(
                validated_preparations["on"][rank] is not None
                and validated_preparations["on"][rank] == validated_preparations["on-mamba-replay"][rank]
            ),
            details={
                "on": validated_preparations["on"][rank],
                "on_mamba_replay": validated_preparations["on-mamba-replay"][rank],
            },
        )
        runtime_reports: dict[str, Any] = {}
        for arm_name, arm_evidence in (
            ("off", dense),
            ("on", shared),
            ("on-mamba-replay", replay_shared),
        ):
            try:
                runtime_reports[arm_name] = _validated_runtime_path_evidence(
                    arm_evidence,
                    arm=arm_name,
                )
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} runtime path",
                    passed=True,
                    details=runtime_reports[arm_name],
                )
            except (ParityError, KeyError, TypeError, ValueError) as error:
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} runtime path",
                    passed=False,
                    details=str(error),
                )
        rank_report["runtime_paths"] = runtime_reports
        try:
            invariant_keys = (
                "batch",
                "config_sha256",
                "topology",
                "repo_root",
                "model_path",
                "seed",
            )
            for key in invariant_keys:
                invariant_passed = dense.get(key) == shared.get(key) == replay_shared.get(key)
                _record_check(
                    report,
                    label=f"rank {rank} invariant {key}",
                    passed=invariant_passed,
                    details={
                        "off": dense.get(key),
                        "on": shared.get(key),
                        "on_mamba_replay": replay_shared.get(key),
                    },
                )
            modes = {
                "off": dense.get("shared_prefix_mode"),
                "on": shared.get("shared_prefix_mode"),
                "on_mamba_replay": replay_shared.get("shared_prefix_mode"),
            }
            _record_check(
                report,
                label=f"rank {rank} arm modes",
                passed=modes == {"off": "observe", "on": "train", "on_mamba_replay": "train"},
                details=modes,
            )

            before = _logprob_metrics(
                dense["selected_logprobs_before"],
                shared["selected_logprobs_before"],
                label=f"rank {rank} pre-update selected logprobs",
            )
            after = _logprob_metrics(
                dense["selected_logprobs_after"],
                shared["selected_logprobs_after"],
                label=f"rank {rank} post-update selected logprobs",
            )
            replay_before = _logprob_metrics(
                dense["selected_logprobs_before"],
                replay_shared["selected_logprobs_before"],
                label=f"rank {rank} replay-prefix selected logprobs",
            )
            for phase, metrics in (("pre-update", before), ("post-update", after)):
                _record_check(
                    report,
                    label=f"rank {rank} {phase} logprob filter limit",
                    passed=metrics["max_sequence_mean_exp_abs_delta"] < LOGPROB_FILTER_LIMIT,
                    details=metrics,
                )
                _record_check(
                    report,
                    label=f"rank {rank} {phase} logprob strict limit",
                    passed=metrics["max_sequence_mean_exp_abs_delta"] < LOGPROB_STRICT_LIMIT,
                    details=metrics,
                )
                _record_check(
                    report,
                    label=f"rank {rank} {phase} logprob vector parity",
                    passed=metrics["relative_l2"] < 0.03 and metrics["cosine"] > 0.995,
                    details=metrics,
                )

            dense_training = dense["training"]
            shared_training = shared["training"]
            off_loss = _finite_float(dense_training["global_loss"], label="off global_loss")
            on_loss = _finite_float(shared_training["global_loss"], label="on global_loss")
            loss_difference = abs(off_loss - on_loss)
            loss_limit = 0.02 + 0.03 * abs(off_loss)
            loss_evidence = {
                "off": off_loss,
                "on": on_loss,
                "absolute_difference": loss_difference,
                "limit": loss_limit,
            }
            _record_check(
                report,
                label=f"rank {rank} RL loss",
                passed=loss_difference <= loss_limit,
                details=loss_evidence,
            )
            mtp_differences: dict[str, Any] = {}
            for key in EXPECTED_MTP_LOSS_KEYS:
                off_value = _finite_float(dense_training["mtp"][key], label=f"off {key}")
                on_value = _finite_float(shared_training["mtp"][key], label=f"on {key}")
                difference = abs(off_value - on_value)
                limit = 0.02 + 0.03 * abs(off_value)
                mtp_differences[key] = {
                    "off": off_value,
                    "on": on_value,
                    "absolute_difference": difference,
                    "limit": limit,
                }
                _record_check(
                    report,
                    label=f"rank {rank} {key}",
                    passed=difference <= limit,
                    details=mtp_differences[key],
                )

            rank_report.update(
                {
                    "selected_logprobs_before": before,
                    "selected_logprobs_after": after,
                    "selected_logprobs_mamba_replay_before": replay_before,
                    "rl_loss": loss_evidence,
                    "mtp_losses": mtp_differences,
                }
            )
        except (ParityError, KeyError, TypeError, ValueError, IndexError) as error:
            _record_check(
                report,
                label=f"rank {rank} scalar and logprob evidence",
                passed=False,
                details=str(error),
            )

        for family in REQUIRED_GRADIENT_FAMILIES:
            try:
                off_norms, off_samples = _family_gradient_vectors(dense, family)
                on_norms, on_samples = _family_gradient_vectors(shared, family)
                norm_metrics = _vector_metrics(
                    off_norms,
                    on_norms,
                    label=f"rank {rank} {family} gradient norms",
                )
                sample_metrics = _vector_metrics(
                    off_samples,
                    on_samples,
                    label=f"rank {rank} {family} sampled gradients",
                )
                _record_check(
                    report,
                    label=f"rank {rank} {family} gradient norms",
                    passed=norm_metrics["relative_l2"] < 0.05 and norm_metrics["cosine"] > 0.995,
                    details=norm_metrics,
                )
                off_sample_norm = float(np.linalg.norm(np.asarray(off_samples, dtype=np.float64)))
                on_sample_norm = float(np.linalg.norm(np.asarray(on_samples, dtype=np.float64)))
                both_samples_zero = off_sample_norm <= 1.0e-12 and on_sample_norm <= 1.0e-12
                _record_check(
                    report,
                    label=f"rank {rank} {family} gradient samples",
                    passed=(
                        both_samples_zero or (sample_metrics["relative_l2"] < 0.10 and sample_metrics["cosine"] > 0.99)
                    ),
                    details={
                        "off_sample_norm": off_sample_norm,
                        "on_sample_norm": on_sample_norm,
                        "both_samples_zero": both_samples_zero,
                        **sample_metrics,
                    },
                )

                rank_report["families"][family] = {
                    "gradient_norm_vector": norm_metrics,
                    "gradient_samples": sample_metrics,
                }
            except (ParityError, KeyError, TypeError, ValueError, IndexError) as error:
                _record_check(
                    report,
                    label=f"rank {rank} {family} family evidence",
                    passed=False,
                    details=str(error),
                )

    report["optimizer_families"] = {}
    for family in REQUIRED_GRADIENT_FAMILIES:
        try:
            aggregate = _aggregate_family_optimizer_updates(off, on, family)
            report["optimizer_families"][family] = aggregate
            _record_check(
                report,
                label=f"global {family} initial optimizer master/storage parity",
                passed=aggregate["initial_exact_match"],
                details={
                    "off_sha256": aggregate["initial_off_sha256"],
                    "on_sha256": aggregate["initial_on_sha256"],
                },
            )
            for arm_name in ("off", "on"):
                arm_update = aggregate[arm_name]
                _record_check(
                    report,
                    label=f"global {family} {arm_name.upper()} optimizer master changed",
                    passed=(
                        arm_update["changed_tensor_count"] > 0
                        and arm_update["changed_storage_tensor_count"] > 0
                        and arm_update["changed_remainder_tensor_count"] > 0
                    ),
                    details=arm_update,
                )
            update_metrics = aggregate["update_projection"]
            _record_check(
                report,
                label=f"global {family} optimizer update parity",
                passed=update_metrics["relative_l2"] < 0.10 and update_metrics["cosine"] > 0.99,
                details=update_metrics,
            )
        except (ParityError, KeyError, TypeError, ValueError, IndexError) as error:
            _record_check(
                report,
                label=f"global {family} optimizer family evidence",
                passed=False,
                details=str(error),
            )

    # Batch plans and selected logprobs are replicated over TP/CP. Every rank
    # must therefore report bit-identical evidence within an arm. This catches
    # artifacts accidentally assembled from different processes or checkpoints.
    for arm_name, arm_values in (("off", off), ("on", on), ("on-mamba-replay", replay)):
        leader_preparation = validated_preparations[arm_name][0]
        leader_runtime_path = {
            "worker_shared_prefix_training_enabled": arm_values[0].get("worker_shared_prefix_training_enabled"),
            "runtime_path_implementations": arm_values[0].get("runtime_path_implementations"),
            "runtime_path_calls": arm_values[0].get("runtime_path_calls"),
        }
        leader_before = arm_values[0].get("selected_logprobs_before")
        leader_after = arm_values[0].get("selected_logprobs_after")
        for rank in range(1, WORLD_SIZE):
            _record_check(
                report,
                label=f"{arm_name} batch preparation rank0/rank{rank}",
                passed=(
                    leader_preparation is not None and validated_preparations[arm_name][rank] == leader_preparation
                ),
                details={
                    "rank0": leader_preparation,
                    f"rank{rank}": validated_preparations[arm_name][rank],
                },
            )
            rank_runtime_path = {
                "worker_shared_prefix_training_enabled": arm_values[rank].get("worker_shared_prefix_training_enabled"),
                "runtime_path_implementations": arm_values[rank].get("runtime_path_implementations"),
                "runtime_path_calls": arm_values[rank].get("runtime_path_calls"),
            }
            _record_check(
                report,
                label=f"{arm_name} runtime path rank0/rank{rank}",
                passed=rank_runtime_path == leader_runtime_path,
                details={
                    "rank0": leader_runtime_path,
                    f"rank{rank}": rank_runtime_path,
                },
            )
            _record_check(
                report,
                label=f"{arm_name} pre-update logprobs rank0/rank{rank}",
                passed=arm_values[rank].get("selected_logprobs_before") == leader_before,
                details="selected logprobs must be bit-identical after PP broadcast",
            )
            if arm_name != "on-mamba-replay":
                _record_check(
                    report,
                    label=f"{arm_name} post-update logprobs rank0/rank{rank}",
                    passed=(
                        leader_after is not None and arm_values[rank].get("selected_logprobs_after") == leader_after
                    ),
                    details="post-update selected logprobs must be bit-identical after PP broadcast",
                )

    rank_zero = report["ranks"].get("0", {})
    state_fork = rank_zero.get("selected_logprobs_before")
    replay_prefix = rank_zero.get("selected_logprobs_mamba_replay_before")
    if isinstance(state_fork, dict) and isinstance(replay_prefix, dict):
        state_fork_strict = state_fork["max_sequence_mean_exp_abs_delta"] < LOGPROB_STRICT_LIMIT
        replay_prefix_strict = replay_prefix["max_sequence_mean_exp_abs_delta"] < LOGPROB_STRICT_LIMIT
        if not state_fork_strict and replay_prefix_strict:
            report["mamba_diagnostic"] = "state-fork-only-forward-drift"
        elif not state_fork_strict and not replay_prefix_strict:
            report["mamba_diagnostic"] = "shared-path-forward-drift-not-isolated-to-state-fork"
        elif state_fork_strict and not replay_prefix_strict:
            report["mamba_diagnostic"] = "replay-prefix-only-forward-drift"
        else:
            report["mamba_diagnostic"] = "both-shared-mamba-paths-within-strict-limit"

    report["status"] = "GREEN" if not report["failures"] else "RED"
    _atomic_json(output, report)
    if report["status"] != "GREEN":
        print(
            "NEMORL_CAPTURED_CITATION_PARITY_RED " f"failures={len(report['failures'])} report={output}",
            flush=True,
        )
        raise ParityError(f"sealed RED comparison report at {output}")

    print(
        "NEMORL_CAPTURED_CITATION_PARITY_GREEN "
        f"batch_sha256={off[0]['batch']['sha256']} "
        f"pre_rel_l2={rank_zero['selected_logprobs_before']['relative_l2']:.9g} "
        f"post_rel_l2={rank_zero['selected_logprobs_after']['relative_l2']:.9g} "
        "mamba_statefork_rel_l2="
        f"{rank_zero['selected_logprobs_before']['relative_l2']:.9g} "
        "mamba_replay_rel_l2="
        f"{rank_zero['selected_logprobs_mamba_replay_before']['relative_l2']:.9g} "
        f"report={output}",
        flush=True,
    )


def _compare_packed_mamba(arguments: argparse.Namespace) -> None:
    """Gate the dense-packed Mamba fallback against the authoritative OFF arm."""
    root = Path(arguments.evidence_dir)
    if not root.is_absolute() or not root.is_dir():
        raise ParityError(f"evidence directory must be absolute: {root}")
    output = Path(arguments.output)
    if not output.is_absolute() or output.parent != root:
        raise ParityError("comparison output must be an absolute direct child of the evidence directory")
    report: dict[str, Any] = {
        "schema": "nemorl-shared-prefix-packed-mamba-comparison-v1",
        "evidence_schema": SCHEMA,
        "status": "RED",
        "topology": _topology(),
        "thresholds": {
            "logprob_filter_max_sequence_mean_exp_abs_delta": LOGPROB_FILTER_LIMIT,
            "logprob_strict_max_sequence_mean_exp_abs_delta": LOGPROB_STRICT_LIMIT,
            "logprob_relative_l2": 0.03,
            "logprob_cosine": 0.995,
        },
        "checks": {},
        "failures": [],
        "ranks": {},
    }
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm in ("off", "on-mamba-packed-replay"):
        try:
            arms[arm] = _load_evidence(root, arm)
            _record_check(
                report,
                label=f"load evidence {arm}",
                passed=True,
                details={"ranks": WORLD_SIZE, "schema": SCHEMA},
            )
        except (ParityError, OSError, ValueError, TypeError, KeyError) as error:
            _record_check(
                report,
                label=f"load evidence {arm}",
                passed=False,
                details=str(error),
            )
    if len(arms) != 2:
        _atomic_json(output, report)
        raise ParityError(f"sealed RED packed-Mamba comparison report at {output}")

    off = arms["off"]
    packed = arms["on-mamba-packed-replay"]
    preparations: dict[str, list[dict[str, Any] | None]] = {
        arm: [cast(dict[str, Any] | None, None) for _ in range(WORLD_SIZE)] for arm in arms
    }
    for rank in range(WORLD_SIZE):
        dense = off[rank]
        fallback = packed[rank]
        rank_report: dict[str, Any] = {}
        report["ranks"][str(rank)] = rank_report
        contracts: dict[str, Any] = {}
        for arm_name, evidence in (("off", dense), ("on-mamba-packed-replay", fallback)):
            try:
                contract = _validated_arm_contract(
                    evidence,
                    arm=cast(Any, arm_name),
                    rank=rank,
                )
                contracts[arm_name] = contract
                preparations[arm_name][rank] = contract["batch_preparation"]
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} evidence contract",
                    passed=True,
                    details=contract,
                )
            except (ParityError, KeyError, TypeError, ValueError) as error:
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} evidence contract",
                    passed=False,
                    details=str(error),
                )
        rank_report["contracts"] = contracts

        runtime_paths: dict[str, Any] = {}
        for arm_name, evidence in (("off", dense), ("on-mamba-packed-replay", fallback)):
            try:
                runtime_paths[arm_name] = _validated_runtime_path_evidence(
                    evidence,
                    arm=cast(Any, arm_name),
                )
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} runtime path",
                    passed=True,
                    details=runtime_paths[arm_name],
                )
            except (ParityError, KeyError, TypeError, ValueError) as error:
                _record_check(
                    report,
                    label=f"rank {rank} {arm_name} runtime path",
                    passed=False,
                    details=str(error),
                )
        rank_report["runtime_paths"] = runtime_paths

        invariant_keys = ("batch", "config_sha256", "topology", "repo_root", "model_path", "seed")
        for key in invariant_keys:
            _record_check(
                report,
                label=f"rank {rank} invariant {key}",
                passed=dense.get(key) == fallback.get(key),
                details={"off": dense.get(key), "on_mamba_packed_replay": fallback.get(key)},
            )
        try:
            metrics = _logprob_metrics(
                dense.get("selected_logprobs_before"),
                fallback.get("selected_logprobs_before"),
                label=f"rank {rank} OFF/packed-Mamba pre-update logprobs",
            )
            rank_report["selected_logprobs_before"] = metrics
            _record_check(
                report,
                label=f"rank {rank} packed-Mamba pre-update logprob vector parity",
                passed=metrics["relative_l2"] < 0.03 and metrics["cosine"] > 0.995,
                details=metrics,
            )
            _record_check(
                report,
                label=f"rank {rank} packed-Mamba pre-update logprob strict limit",
                passed=metrics["max_sequence_mean_exp_abs_delta"] < LOGPROB_STRICT_LIMIT,
                details=metrics,
            )
            _record_check(
                report,
                label=f"rank {rank} packed-Mamba pre-update logprob filter limit",
                passed=metrics["max_sequence_mean_exp_abs_delta"] < LOGPROB_FILTER_LIMIT,
                details=metrics,
            )
        except (ParityError, KeyError, TypeError, ValueError, IndexError) as error:
            _record_check(
                report,
                label=f"rank {rank} packed-Mamba pre-update logprob evidence",
                passed=False,
                details=str(error),
            )

    for arm_name, arm_values in arms.items():
        leader_preparation = preparations[arm_name][0]
        leader_runtime = {
            "worker_shared_prefix_training_enabled": arm_values[0].get("worker_shared_prefix_training_enabled"),
            "runtime_path_implementations": arm_values[0].get("runtime_path_implementations"),
            "runtime_path_calls": arm_values[0].get("runtime_path_calls"),
        }
        leader_logprobs = arm_values[0].get("selected_logprobs_before")
        for rank in range(1, WORLD_SIZE):
            rank_runtime = {
                "worker_shared_prefix_training_enabled": arm_values[rank].get("worker_shared_prefix_training_enabled"),
                "runtime_path_implementations": arm_values[rank].get("runtime_path_implementations"),
                "runtime_path_calls": arm_values[rank].get("runtime_path_calls"),
            }
            _record_check(
                report,
                label=f"{arm_name} batch preparation rank0/rank{rank}",
                passed=leader_preparation is not None and preparations[arm_name][rank] == leader_preparation,
                details={"rank0": leader_preparation, f"rank{rank}": preparations[arm_name][rank]},
            )
            _record_check(
                report,
                label=f"{arm_name} runtime path rank0/rank{rank}",
                passed=rank_runtime == leader_runtime,
                details={"rank0": leader_runtime, f"rank{rank}": rank_runtime},
            )
            _record_check(
                report,
                label=f"{arm_name} pre-update logprobs rank0/rank{rank}",
                passed=arm_values[rank].get("selected_logprobs_before") == leader_logprobs,
                details="selected logprobs must be bit-identical after PP broadcast",
            )

    report["status"] = "GREEN" if not report["failures"] else "RED"
    _atomic_json(output, report)
    if report["status"] != "GREEN":
        print(
            "NEMORL_CAPTURED_CITATION_PACKED_MAMBA_PARITY_RED " f"failures={len(report['failures'])} report={output}",
            flush=True,
        )
        raise ParityError(f"sealed RED packed-Mamba comparison report at {output}")
    rank_zero = report["ranks"]["0"]["selected_logprobs_before"]
    print(
        "NEMORL_CAPTURED_CITATION_PACKED_MAMBA_PARITY_GREEN "
        f"batch_sha256={off[0]['batch']['sha256']} "
        f"pre_rel_l2={rank_zero['relative_l2']:.9g} "
        f"pre_cosine={rank_zero['cosine']:.9g} "
        f"max_sequence_mean_exp_abs_delta={rank_zero['max_sequence_mean_exp_abs_delta']:.9g} "
        f"report={output}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-arm")
    run.add_argument(
        "--arm",
        choices=("off", "on", "on-mamba-replay", "on-mamba-packed-replay"),
        required=True,
    )
    run.add_argument("--repo-root", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--batch", required=True)
    run.add_argument("--expected-batch-sha256", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=42)
    preflight = subparsers.add_parser("preflight-batch")
    preflight.add_argument("--repo-root", required=True)
    preflight.add_argument("--model-path", required=True)
    preflight.add_argument("--batch", required=True)
    preflight.add_argument("--expected-batch-sha256", required=True)
    subparsers.add_parser("self-test-master-remainder")
    subparsers.add_parser("self-test-optimizer-ownership")
    subparsers.add_parser("self-test-optimizer-aggregation")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--evidence-dir", required=True)
    compare.add_argument("--output", required=True)
    compare_packed = subparsers.add_parser("compare-packed-mamba")
    compare_packed.add_argument("--evidence-dir", required=True)
    compare_packed.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run-arm":
            _run_arm(arguments)
        elif arguments.command == "preflight-batch":
            _run_batch_preflight(arguments)
        elif arguments.command == "self-test-master-remainder":
            result = _master_remainder_reconstruction_self_test()
            print(
                "CAPTURED_CITATION_MASTER_REMAINDER_SELF_TEST_GREEN "
                f"cases={result['cases']} sha256={result['master_sha256']}",
                flush=True,
            )
        elif arguments.command == "self-test-optimizer-ownership":
            result = _optimizer_model_parameter_coverage_self_test()
            print(
                "CAPTURED_CITATION_OPTIMIZER_OWNERSHIP_SELF_TEST_GREEN "
                f"green={result['green_cases']} rejected={result['rejected_cases']} "
                f"peer_owned={result['result']['peer_owned_model_parameter_count']}",
                flush=True,
            )
        elif arguments.command == "self-test-optimizer-aggregation":
            result = _optimizer_update_aggregation_self_test()
            print(
                "CAPTURED_CITATION_OPTIMIZER_AGGREGATION_SELF_TEST_GREEN "
                f"green={result['green_cases']} rejected={result['rejected_cases']}",
                flush=True,
            )
        elif arguments.command == "compare-packed-mamba":
            _compare_packed_mamba(arguments)
        else:
            _compare(arguments)
    except ParityError as error:
        print(f"CAPTURED_CITATION_PARITY_ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
