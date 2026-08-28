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

"""Tensor materialization for backend-neutral shared-prefix plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from nemo_rl.data.packing.shared_prefix import (
    SharedPrefixFallback,
    SharedPrefixLayout,
    SharedPrefixRow,
    plan_shared_prefix_bins,
)


@dataclass(frozen=True, slots=True)
class SharedPrefixTensorIndices:
    """Device tensors for token gather and completion logprob fan-out/scatter."""

    token_gather_rows: torch.Tensor
    token_gather_columns: torch.Tensor
    completion_positions: torch.Tensor
    predecessor_positions: torch.Tensor
    completion_scatter_rows: torch.Tensor
    completion_scatter_columns: torch.Tensor


@dataclass(frozen=True, slots=True)
class SharedPrefixTensorBin:
    """One unpadded, prompt-once global star ready for a backend adapter.

    When materialized, ``attention_allow_mask`` is a dense boolean
    ``[tokens, tokens]`` reference mask where ``True`` means attention is
    allowed. It is intended as the exact global correctness oracle; production
    long-context paths leave it as ``None`` and lower ``layout`` to fused
    structured attention instead of constructing this quadratic mask.
    """

    layout: SharedPrefixLayout
    packed_input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_allow_mask: torch.Tensor | None
    indices: SharedPrefixTensorIndices


@dataclass(frozen=True, slots=True)
class SharedPrefixTensorPlan:
    """Tensorized shared bins plus the planner's untouched fallback records."""

    shared_bins: tuple[SharedPrefixTensorBin, ...]
    fallbacks: tuple[SharedPrefixFallback, ...]

    @property
    def fallback_row_indices(self) -> tuple[int, ...]:
        """Source rows that remain on the conventional packing path."""
        return tuple(fallback.row.row_index for fallback in self.fallbacks)


@dataclass(frozen=True, slots=True)
class SharedPrefixContextParallelShard:
    """One standard zigzag CP shard of a padded shared-prefix star.

    ``global_token_indices`` maps the local sequence dimension back to the
    padded global physical order.  Real tokens occupy
    ``[0, tensor_bin.layout.total_length)``; any trailing positions are causal
    padding and are not represented in the logical star layout.
    """

    packed_input_ids: torch.Tensor
    position_ids: torch.Tensor
    global_token_indices: torch.Tensor
    padded_total_length: int


def get_shared_prefix_context_parallel_indices(
    padded_total_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return standard two-chunk causal CP indices in rank-local order.

    Rank ``r`` owns global chunk ``r`` followed by chunk ``2*CP-r-1``.  This
    is the same zigzag sequence ownership used by NeMo-RL's conventional MCore
    path, expressed explicitly so the shared-prefix output fan-out can route
    only scalar log-probabilities instead of gathering vocabulary logits.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if padded_total_length < 1:
        raise ValueError(
            f"padded_total_length must be positive, got {padded_total_length}"
        )
    if cp_size == 1:
        return torch.arange(
            padded_total_length,
            dtype=torch.long,
            device=device,
        )
    alignment = 2 * cp_size
    if padded_total_length % alignment != 0:
        raise ValueError(
            "shared-prefix CP length must be divisible by 2 * cp_size: "
            f"{padded_total_length} % {alignment} != 0"
        )

    chunk_length = padded_total_length // alignment
    first_chunk = cp_rank
    second_chunk = alignment - cp_rank - 1
    first = torch.arange(
        first_chunk * chunk_length,
        (first_chunk + 1) * chunk_length,
        dtype=torch.long,
        device=device,
    )
    second = torch.arange(
        second_chunk * chunk_length,
        (second_chunk + 1) * chunk_length,
        dtype=torch.long,
        device=device,
    )
    return torch.cat((first, second), dim=0)


def shard_shared_prefix_tensor_bin_for_context_parallel(
    tensor_bin: SharedPrefixTensorBin,
    *,
    cp_rank: int,
    cp_size: int,
    padding_token_id: int = 0,
) -> SharedPrefixContextParallelShard:
    """Pad and shard a materialized star for MCore context parallelism.

    The logical layout remains unpadded.  Padding is appended after all real
    branches, so causality guarantees that it cannot influence a real token; the
    MCore forest adapter masks padded queries/keys and NeMo-RL never selects
    their logits for the loss.
    """
    total_length = tensor_bin.layout.total_length
    alignment = 1 if cp_size == 1 else 2 * cp_size
    padded_total_length = ((total_length + alignment - 1) // alignment) * alignment
    pad_length = padded_total_length - total_length

    packed_input_ids = tensor_bin.packed_input_ids
    position_ids = tensor_bin.position_ids
    if pad_length:
        packed_input_ids = torch.nn.functional.pad(
            packed_input_ids,
            (0, pad_length),
            value=padding_token_id,
        )
        position_ids = torch.nn.functional.pad(
            position_ids,
            (0, pad_length),
            value=0,
        )

    global_token_indices = get_shared_prefix_context_parallel_indices(
        padded_total_length,
        cp_rank=cp_rank,
        cp_size=cp_size,
        device=packed_input_ids.device,
    )
    return SharedPrefixContextParallelShard(
        packed_input_ids=packed_input_ids.index_select(0, global_token_indices),
        position_ids=position_ids.index_select(0, global_token_indices),
        global_token_indices=global_token_indices,
        padded_total_length=padded_total_length,
    )


def build_star_attention_allow_mask(
    layout: SharedPrefixLayout,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Materialize the exact dense causal allow-mask for one global star.

    Prompt queries attend causally within the prompt. A completion query attends
    every prompt token and causally within its own suffix, but never a sibling
    completion. Self-attention is included.

    Args:
        layout: Exact-prompt star layout to lower.
        device: Device for the returned boolean tensor. Defaults to CPU.

    Returns:
        Boolean ``[layout.total_length, layout.total_length]`` allow-mask.

    Raises:
        ValueError: If the layout's branch spans do not exactly cover its packed
            completion region.
    """
    if layout.prompt_length < 1 or len(layout.row_indices) < 2:
        raise ValueError("a shared-prefix star requires a prompt and at least two rows")
    if len(layout.branch_starts) != len(layout.completion_lengths):
        raise ValueError("branch_starts and completion_lengths must have equal length")

    mask_device = torch.device("cpu") if device is None else torch.device(device)
    segment_ids = torch.zeros(
        layout.total_length,
        dtype=torch.long,
        device=mask_device,
    )

    expected_start = layout.prompt_length
    for branch_id, (branch_start, completion_length) in enumerate(
        zip(layout.branch_starts, layout.completion_lengths, strict=True),
        start=1,
    ):
        if branch_start != expected_start or completion_length < 1:
            raise ValueError(
                "shared-prefix branches must be positive and contiguous; "
                f"expected start {expected_start}, got start {branch_start} "
                f"with length {completion_length}"
            )
        branch_end = branch_start + completion_length
        segment_ids[branch_start:branch_end] = branch_id
        expected_start = branch_end
    if expected_start != layout.total_length:
        raise ValueError(
            "shared-prefix branch spans do not cover total_length: "
            f"covered {expected_start}, total {layout.total_length}"
        )

    packed_positions = torch.arange(layout.total_length, device=mask_device)
    query_positions = packed_positions[:, None]
    key_positions = packed_positions[None, :]
    query_segments = segment_ids[:, None]
    key_segments = segment_ids[None, :]
    causal = key_positions <= query_positions
    key_is_prompt = key_segments == 0
    same_segment = query_segments == key_segments
    return causal & (key_is_prompt | same_segment)


def materialize_shared_prefix_layout(
    input_ids: torch.Tensor,
    *,
    input_lengths: torch.Tensor,
    layout: SharedPrefixLayout,
    materialize_attention_mask: bool = True,
) -> SharedPrefixTensorBin:
    """Gather one planned star from conventional padded ``input_ids`` rows.

    Args:
        input_ids: Integer token tensor with shape ``[batch, sequence]``.
        input_lengths: Unpadded source-row lengths with shape ``[batch]``.
        layout: Planner output whose row indices address ``input_ids``.
        materialize_attention_mask: Whether to build the dense global correctness
            oracle. Fused backends should disable it and lower ``layout``
            directly so long-context execution does not allocate ``O(T^2)``
            storage.

    Returns:
        Unpadded packed tokens, prefix-continued positions, exact global allow-mask,
        and tensorized gather/fan-out/scatter indices on ``input_ids.device``.

    Raises:
        ValueError: If the source tensor cannot satisfy the layout or its prompt
            tokens no longer match the planned exact prompt.
    """
    _validate_input_ids(input_ids)
    batch_size, sequence_width = input_ids.shape
    input_lengths_cpu = _validate_length_vector(
        input_lengths,
        name="input_lengths",
        batch_size=batch_size,
        sequence_width=sequence_width,
    )
    if not layout.token_gather_rows or not layout.token_gather_columns:
        raise ValueError("shared-prefix layout has no token gather indices")
    if (
        len(layout.token_gather_rows) != layout.total_length
        or len(layout.token_gather_columns) != layout.total_length
    ):
        raise ValueError("token gather indices must have total_length entries")
    if min(layout.token_gather_rows) < 0 or max(layout.token_gather_rows) >= batch_size:
        raise ValueError("shared-prefix layout references a row outside input_ids")
    if (
        min(layout.token_gather_columns) < 0
        or max(layout.token_gather_columns) >= sequence_width
    ):
        raise ValueError("shared-prefix layout references a token outside input_ids")

    expected_prompt = torch.tensor(
        layout.prompt_token_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    for row_index, completion_length in zip(
        layout.row_indices,
        layout.completion_lengths,
        strict=True,
    ):
        if row_index < 0 or row_index >= batch_size:
            raise ValueError(
                f"shared-prefix layout references row {row_index} outside input_ids"
            )
        required_length = layout.prompt_length + completion_length
        source_length = int(input_lengths_cpu[row_index].item())
        if required_length > source_length:
            raise ValueError(
                f"source row {row_index} has length {source_length}, but layout "
                f"requires {required_length} tokens"
            )
        source_prompt = input_ids[row_index, : layout.prompt_length]
        if not torch.equal(source_prompt, expected_prompt):
            raise ValueError(
                f"source prompt for row {row_index} differs from the planned exact prompt"
            )

    device = input_ids.device
    indices = SharedPrefixTensorIndices(
        token_gather_rows=_as_long_tensor(layout.token_gather_rows, device=device),
        token_gather_columns=_as_long_tensor(
            layout.token_gather_columns,
            device=device,
        ),
        completion_positions=_as_long_tensor(
            layout.completion_positions,
            device=device,
        ),
        predecessor_positions=_as_long_tensor(
            layout.predecessor_positions,
            device=device,
        ),
        completion_scatter_rows=_as_long_tensor(
            layout.completion_scatter_rows,
            device=device,
        ),
        completion_scatter_columns=_as_long_tensor(
            layout.completion_scatter_columns,
            device=device,
        ),
    )
    packed_input_ids = input_ids[
        indices.token_gather_rows,
        indices.token_gather_columns,
    ]
    return SharedPrefixTensorBin(
        layout=layout,
        packed_input_ids=packed_input_ids,
        position_ids=_as_long_tensor(layout.position_ids, device=device),
        attention_allow_mask=(
            build_star_attention_allow_mask(layout, device=device)
            if materialize_attention_mask
            else None
        ),
        indices=indices,
    )


def build_shared_prefix_tensor_plan(
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: Sequence[str | None],
    bin_capacity: int,
    max_completions_per_bin: int = 16,
    materialize_attention_mask: bool = True,
) -> SharedPrefixTensorPlan:
    """Plan and materialize exact-prompt stars from a conventional padded batch.

    Args:
        input_ids: Integer token IDs with shape ``[batch, sequence]``.
        input_lengths: Unpadded total length of each source row.
        prompt_lengths: Prompt length of each source row. Completion tokens are
            the contiguous range ``[prompt_length, input_length)``.
        group_ids: Opaque rollout-group ID per source row.
        bin_capacity: Maximum deduplicated token count per shared bin.
        max_completions_per_bin: Maximum branches per shared bin.
        materialize_attention_mask: Whether each tensor bin should include the
            dense global reference mask. Set to ``False`` for fused model adapters
            that consume the structured layout directly.

    Returns:
        Tensorized shared bins on ``input_ids.device`` and explicit fallback
        records for every row not selected for sharing.

    Raises:
        TypeError: If a group ID is neither ``str`` nor ``None``.
        ValueError: If tensor shapes, lengths, or planner limits are invalid.
    """
    lengths_cpu, prompt_lengths_cpu = _validate_batch_inputs(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
    )
    input_ids_cpu = input_ids.detach().cpu()
    rows: list[SharedPrefixRow] = []
    for row_index, group_id in enumerate(group_ids):
        if group_id is not None and not isinstance(group_id, str):
            raise TypeError(
                f"group_ids[{row_index}] must be str or None, got "
                f"{type(group_id).__name__}"
            )
        prompt_length = int(prompt_lengths_cpu[row_index].item())
        input_length = int(lengths_cpu[row_index].item())
        prompt_token_ids = tuple(
            int(token) for token in input_ids_cpu[row_index, :prompt_length].tolist()
        )
        rows.append(
            SharedPrefixRow(
                row_index=row_index,
                group_id=group_id,
                prompt_token_ids=prompt_token_ids,
                completion_length=input_length - prompt_length,
            )
        )

    plan = plan_shared_prefix_bins(
        rows,
        bin_capacity=bin_capacity,
        max_completions_per_bin=max_completions_per_bin,
    )
    return SharedPrefixTensorPlan(
        shared_bins=tuple(
            materialize_shared_prefix_layout(
                input_ids,
                input_lengths=input_lengths,
                layout=layout,
                materialize_attention_mask=materialize_attention_mask,
            )
            for layout in plan.shared_bins
        ),
        fallbacks=plan.fallbacks,
    )


def _validate_input_ids(input_ids: torch.Tensor) -> None:
    """Validate the source token tensor without changing its device or dtype."""
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if (
        input_ids.is_floating_point()
        or input_ids.is_complex()
        or input_ids.dtype == torch.bool
    ):
        raise ValueError(f"input_ids must have an integer dtype, got {input_ids.dtype}")


def _validate_batch_inputs(
    *,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: Sequence[str | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate conventional batch metadata and return CPU integer lengths."""
    _validate_input_ids(input_ids)
    batch_size, sequence_width = input_ids.shape
    if isinstance(group_ids, str):
        raise TypeError("group_ids must be a sequence of per-row IDs, not one string")
    if len(group_ids) != batch_size:
        raise ValueError(
            f"group_ids must have {batch_size} entries, got {len(group_ids)}"
        )

    lengths_cpu = _validate_length_vector(
        input_lengths,
        name="input_lengths",
        batch_size=batch_size,
        sequence_width=sequence_width,
    )
    prompt_lengths_cpu = _validate_length_vector(
        prompt_lengths,
        name="prompt_lengths",
        batch_size=batch_size,
        sequence_width=sequence_width,
    )
    if bool(torch.any(prompt_lengths_cpu > lengths_cpu).item()):
        raise ValueError(
            "prompt_lengths must satisfy 0 <= prompt_length <= input_length"
        )
    return lengths_cpu, prompt_lengths_cpu


def _validate_length_vector(
    lengths: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    sequence_width: int,
) -> torch.Tensor:
    """Validate and copy one source-length vector to CPU long."""
    if lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(
            f"{name} must have shape [{batch_size}], got {tuple(lengths.shape)}"
        )
    if (
        lengths.is_floating_point()
        or lengths.is_complex()
        or lengths.dtype == torch.bool
    ):
        raise ValueError(f"{name} must have an integer dtype, got {lengths.dtype}")
    lengths_cpu = lengths.detach().cpu().to(torch.long)
    if bool(torch.any(lengths_cpu < 0).item()) or bool(
        torch.any(lengths_cpu > sequence_width).item()
    ):
        raise ValueError(f"{name} must be within input_ids width [0, {sequence_width}]")
    return lengths_cpu


def _as_long_tensor(
    values: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Convert immutable planner indices to a device-local long tensor."""
    return torch.tensor(values, dtype=torch.long, device=device)
