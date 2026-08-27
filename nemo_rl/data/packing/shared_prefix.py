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

"""Backend-neutral planning primitives for exact shared-prefix GRPO groups.

This module only describes how rows can be packed. It does not build tensors,
attention masks, or backend-specific model inputs. Consequently, a caller can
use the same plan for Megatron, DTensor, or an observational dry run.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SharedPrefixRow:
    """One prompt-completion row considered for shared-prefix packing.

    Attributes:
        row_index: Stable index of the row in the caller's input batch.
        group_id: Opaque GRPO rollout-group identity. ``None`` means that the
            row cannot be safely matched with peers.
        prompt_token_ids: Exact prompt token sequence. Rows only share a prefix
            when both this sequence and ``group_id`` are equal.
        completion_length: Number of contiguous completion tokens immediately
            following the prompt in the source row.
    """

    row_index: int
    group_id: str | None
    prompt_token_ids: tuple[int, ...]
    completion_length: int

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative")
        if self.completion_length < 0:
            raise ValueError("completion_length must be nonnegative")

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens in the source row."""
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        """Unpadded token count of the ordinary prompt-completion row."""
        return self.prompt_length + self.completion_length


@dataclass(frozen=True, slots=True)
class SharedPrefixLayout:
    """CPU representation of one exact-prompt GRPO star.

    Parallel index tuples deliberately remain ordinary Python values. A model
    adapter can tensorize them on its own device without making the data-plane
    planner depend on PyTorch or a specific training backend.

    ``token_gather_rows`` and ``token_gather_columns`` map every packed token to
    its source-batch coordinate. ``completion_positions`` identify target tokens
    in the packed sequence, while ``predecessor_positions`` identify the logits
    that predict them. ``completion_scatter_*`` map those logprobs back to the
    caller's conventional ``[row, sequence_length - 1]`` next-token view.
    """

    group_id: str
    prompt_token_ids: tuple[int, ...]
    row_indices: tuple[int, ...]
    completion_lengths: tuple[int, ...]
    total_length: int
    branch_starts: tuple[int, ...]
    position_ids: tuple[int, ...]
    token_gather_rows: tuple[int, ...]
    token_gather_columns: tuple[int, ...]
    completion_positions: tuple[int, ...]
    predecessor_positions: tuple[int, ...]
    completion_scatter_rows: tuple[int, ...]
    completion_scatter_columns: tuple[int, ...]

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens stored once in this layout."""
        return len(self.prompt_token_ids)

    @property
    def baseline_length(self) -> int:
        """Token count if every completion duplicated the prompt."""
        return len(self.row_indices) * self.prompt_length + sum(self.completion_lengths)

    @property
    def tokens_saved(self) -> int:
        """Prompt tokens removed relative to ordinary duplicated rows."""
        return self.baseline_length - self.total_length


class SharedPrefixFallbackReason(enum.Enum):
    """Reason a row was not placed in a shared-prefix star."""

    MISSING_GROUP_ID = "missing_group_id"
    EMPTY_PROMPT = "empty_prompt"
    EMPTY_COMPLETION = "empty_completion"
    SEQUENCE_EXCEEDS_BIN = "sequence_exceeds_bin"
    NO_EXACT_PROMPT_PEER = "no_exact_prompt_peer"
    PROMPT_MISMATCH = "prompt_mismatch"
    NO_CAPACITY_COMPATIBLE_PEER = "no_capacity_compatible_peer"


@dataclass(frozen=True, slots=True)
class SharedPrefixFallback:
    """One row routed away from shared-prefix packing.

    ``fits_block_diagonal_bin`` distinguishes a normal mixed fallback from an
    oversized row that the existing packer must reject or handle separately.
    """

    row: SharedPrefixRow
    reason: SharedPrefixFallbackReason
    fits_block_diagonal_bin: bool


@dataclass(frozen=True, slots=True)
class SharedPrefixPlan:
    """Deterministic partition of input rows into shared stars and fallbacks."""

    shared_bins: tuple[SharedPrefixLayout, ...]
    fallbacks: tuple[SharedPrefixFallback, ...]

    @property
    def shared_row_indices(self) -> tuple[int, ...]:
        """Input row indices covered by shared-prefix bins."""
        return tuple(
            row_index for layout in self.shared_bins for row_index in layout.row_indices
        )

    @property
    def fallback_row_indices(self) -> tuple[int, ...]:
        """Input row indices routed to the fallback path."""
        return tuple(fallback.row.row_index for fallback in self.fallbacks)


def build_shared_prefix_layout(
    rows: Sequence[SharedPrefixRow],
) -> SharedPrefixLayout:
    """Build one exact-prompt GRPO star from two or more rows.

    The first row supplies the single stored copy of the prompt. Each completion
    supplies its own contiguous suffix. The first completion token of every
    branch is predicted by the final prompt position; later tokens are predicted
    by the preceding token in the same branch.

    Args:
        rows: Rows with the same non-null group identity and exact prompt tokens.

    Returns:
        A backend-neutral star layout whose indices reference the caller's input
        row indices.

    Raises:
        ValueError: If fewer than two rows are supplied or rows do not form a
            valid exact-prompt group.
    """
    if len(rows) < 2:
        raise ValueError("a shared-prefix layout requires at least two rows")

    first = rows[0]
    group_id = first.group_id
    if group_id is None:
        raise ValueError("shared-prefix rows must have a group_id")
    if first.prompt_length == 0:
        raise ValueError("a shared-prefix layout requires a non-empty prompt")

    seen_row_indices: set[int] = set()
    for row in rows:
        if row.row_index in seen_row_indices:
            raise ValueError(f"duplicate row_index {row.row_index}")
        seen_row_indices.add(row.row_index)
        if row.group_id != group_id:
            raise ValueError("all shared-prefix rows must have the same group_id")
        if row.prompt_token_ids != first.prompt_token_ids:
            raise ValueError("all shared-prefix rows must have identical prompt tokens")
        if row.completion_length == 0:
            raise ValueError("all shared-prefix rows must have a non-empty completion")

    prompt_length = first.prompt_length
    token_gather_rows = [first.row_index] * prompt_length
    token_gather_columns = list(range(prompt_length))
    position_ids = list(range(prompt_length))
    branch_starts: list[int] = []
    completion_positions: list[int] = []
    predecessor_positions: list[int] = []
    completion_scatter_rows: list[int] = []
    completion_scatter_columns: list[int] = []
    packed_offset = prompt_length

    for row in rows:
        branch_starts.append(packed_offset)
        token_gather_rows.extend([row.row_index] * row.completion_length)
        token_gather_columns.extend(
            range(prompt_length, prompt_length + row.completion_length)
        )
        position_ids.extend(range(prompt_length, prompt_length + row.completion_length))
        for completion_offset in range(row.completion_length):
            packed_position = packed_offset + completion_offset
            predecessor_position = (
                prompt_length - 1 if completion_offset == 0 else packed_position - 1
            )
            completion_positions.append(packed_position)
            predecessor_positions.append(predecessor_position)
            completion_scatter_rows.append(row.row_index)
            completion_scatter_columns.append(prompt_length + completion_offset - 1)
        packed_offset += row.completion_length

    return SharedPrefixLayout(
        group_id=group_id,
        prompt_token_ids=first.prompt_token_ids,
        row_indices=tuple(row.row_index for row in rows),
        completion_lengths=tuple(row.completion_length for row in rows),
        total_length=packed_offset,
        branch_starts=tuple(branch_starts),
        position_ids=tuple(position_ids),
        token_gather_rows=tuple(token_gather_rows),
        token_gather_columns=tuple(token_gather_columns),
        completion_positions=tuple(completion_positions),
        predecessor_positions=tuple(predecessor_positions),
        completion_scatter_rows=tuple(completion_scatter_rows),
        completion_scatter_columns=tuple(completion_scatter_columns),
    )


def plan_shared_prefix_bins(
    rows: Sequence[SharedPrefixRow],
    *,
    bin_capacity: int,
    max_completions_per_bin: int = 16,
) -> SharedPrefixPlan:
    """Partition exact-prompt GRPO rows into shared stars and mixed fallbacks.

    Planning is invariant to the input sequence order: rows are first ordered by
    ``row_index``, exact groups are visited by their lowest row index, and each
    group uses first-fit decreasing with ``row_index`` as the tie-breaker.

    A shared bin always has at least two completions and satisfies both the token
    capacity and completion-count limit. Rows that cannot share are retained as
    explicit fallbacks instead of being silently dropped.

    Args:
        rows: Candidate prompt-completion rows.
        bin_capacity: Maximum number of deduplicated tokens in one shared bin.
        max_completions_per_bin: Maximum branches behind one stored prompt.

    Returns:
        A complete, deterministic partition of the input row indices.

    Raises:
        ValueError: If planner limits are invalid or row indices are duplicated.
    """
    if bin_capacity < 1:
        raise ValueError("bin_capacity must be positive")
    if max_completions_per_bin < 2:
        raise ValueError("max_completions_per_bin must be at least 2")

    ordered_rows = sorted(rows, key=lambda row: row.row_index)
    row_indices = [row.row_index for row in ordered_rows]
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("row_index values must be unique")

    fallbacks: list[SharedPrefixFallback] = []
    eligible_rows: list[SharedPrefixRow] = []
    for row in ordered_rows:
        reason: SharedPrefixFallbackReason | None = None
        if row.group_id is None:
            reason = SharedPrefixFallbackReason.MISSING_GROUP_ID
        elif row.prompt_length == 0:
            reason = SharedPrefixFallbackReason.EMPTY_PROMPT
        elif row.completion_length == 0:
            reason = SharedPrefixFallbackReason.EMPTY_COMPLETION
        elif row.total_length > bin_capacity:
            reason = SharedPrefixFallbackReason.SEQUENCE_EXCEEDS_BIN

        if reason is None:
            eligible_rows.append(row)
        else:
            fallbacks.append(
                SharedPrefixFallback(
                    row=row,
                    reason=reason,
                    fits_block_diagonal_bin=row.total_length <= bin_capacity,
                )
            )

    exact_groups: dict[tuple[str, tuple[int, ...]], list[SharedPrefixRow]] = {}
    eligible_group_sizes: dict[str, int] = {}
    for row in eligible_rows:
        assert row.group_id is not None
        exact_groups.setdefault((row.group_id, row.prompt_token_ids), []).append(row)
        eligible_group_sizes[row.group_id] = (
            eligible_group_sizes.get(row.group_id, 0) + 1
        )

    shared_bins: list[SharedPrefixLayout] = []
    for (group_id, _prompt_token_ids), exact_rows in exact_groups.items():
        if len(exact_rows) < 2:
            reason = (
                SharedPrefixFallbackReason.PROMPT_MISMATCH
                if eligible_group_sizes[group_id] > 1
                else SharedPrefixFallbackReason.NO_EXACT_PROMPT_PEER
            )
            fallbacks.append(
                SharedPrefixFallback(
                    row=exact_rows[0],
                    reason=reason,
                    fits_block_diagonal_bin=True,
                )
            )
            continue

        prompt_length = exact_rows[0].prompt_length
        physical_bins: list[list[SharedPrefixRow]] = []
        bin_completion_tokens: list[int] = []
        rows_by_size = sorted(
            exact_rows,
            key=lambda row: (-row.completion_length, row.row_index),
        )
        for row in rows_by_size:
            destination: int | None = None
            for bin_index, bin_rows in enumerate(physical_bins):
                has_branch_slot = len(bin_rows) < max_completions_per_bin
                has_token_space = (
                    prompt_length
                    + bin_completion_tokens[bin_index]
                    + row.completion_length
                    <= bin_capacity
                )
                if has_branch_slot and has_token_space:
                    destination = bin_index
                    break
            if destination is None:
                physical_bins.append([row])
                bin_completion_tokens.append(row.completion_length)
            else:
                physical_bins[destination].append(row)
                bin_completion_tokens[destination] += row.completion_length

        for bin_rows in physical_bins:
            if len(bin_rows) >= 2:
                shared_bins.append(build_shared_prefix_layout(bin_rows))
            else:
                fallbacks.append(
                    SharedPrefixFallback(
                        row=bin_rows[0],
                        reason=SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
                        fits_block_diagonal_bin=True,
                    )
                )

    fallbacks.sort(key=lambda fallback: fallback.row.row_index)
    return SharedPrefixPlan(
        shared_bins=tuple(shared_bins),
        fallbacks=tuple(fallbacks),
    )
