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
"""Observe exact-prompt shared-prefix opportunity without changing execution."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SharedPrefixOpportunity:
    """Batch-level token accounting for exact-prompt star sharing."""

    total_sequences: int
    eligible_sequences: int
    complete_groups: int
    fallback_sequences: int
    total_tokens: int
    prompt_tokens: int
    valid_loss_tokens: float
    non_loss_suffix_tokens: float
    shareable_prompt_tokens: int
    ideal_shared_token_work: int
    loss_ratio_upper_bound_saved_tokens: float

    def as_metrics(self) -> dict[str, float | int]:
        """Return scalar metrics suitable for the standard NeMo-RL logger."""
        total_tokens = float(self.total_tokens)
        total_sequences = float(self.total_sequences)
        ideal_work = float(self.ideal_shared_token_work)

        return {
            "shared_prefix/total_sequences": self.total_sequences,
            "shared_prefix/eligible_sequences": self.eligible_sequences,
            "shared_prefix/total_tokens": self.total_tokens,
            "shared_prefix/prompt_tokens": self.prompt_tokens,
            "shared_prefix/valid_loss_tokens": self.valid_loss_tokens,
            "shared_prefix/non_loss_suffix_tokens": self.non_loss_suffix_tokens,
            "shared_prefix/valid_to_total_token_ratio": (
                self.valid_loss_tokens / total_tokens if total_tokens else 0.0
            ),
            "shared_prefix/prompt_token_fraction": (
                self.prompt_tokens / total_tokens if total_tokens else 0.0
            ),
            "shared_prefix/non_loss_suffix_token_fraction": (
                self.non_loss_suffix_tokens / total_tokens if total_tokens else 0.0
            ),
            "shared_prefix/exact_group_sequence_coverage": (
                self.eligible_sequences / total_sequences if total_sequences else 0.0
            ),
            "shared_prefix/complete_groups": self.complete_groups,
            "shared_prefix/fallback_sequences": self.fallback_sequences,
            "shared_prefix/shareable_prompt_tokens": self.shareable_prompt_tokens,
            "shared_prefix/ideal_shared_token_work": self.ideal_shared_token_work,
            "shared_prefix/ideal_token_reduction": (
                self.shareable_prompt_tokens / total_tokens if total_tokens else 0.0
            ),
            "shared_prefix/ideal_token_work_speedup": (
                total_tokens / ideal_work if ideal_work else 1.0
            ),
            # This is the optimistic value derivable from W&B's
            # valid/total ratio alone. It treats every non-loss token as a
            # complete-group prompt token, so masked suffixes and fragmented
            # groups make it an upper bound rather than an execution forecast.
            "shared_prefix/loss_ratio_upper_bound_token_reduction": (
                self.loss_ratio_upper_bound_saved_tokens / total_tokens
                if total_tokens
                else 0.0
            ),
        }


def combine_shared_prefix_opportunities(
    opportunities: Sequence[SharedPrefixOpportunity],
) -> SharedPrefixOpportunity:
    """Combine disjoint batch observations before deriving ratios or speedups."""
    if not opportunities:
        raise ValueError("at least one shared-prefix opportunity is required")
    return SharedPrefixOpportunity(
        total_sequences=sum(item.total_sequences for item in opportunities),
        eligible_sequences=sum(item.eligible_sequences for item in opportunities),
        complete_groups=sum(item.complete_groups for item in opportunities),
        fallback_sequences=sum(item.fallback_sequences for item in opportunities),
        total_tokens=sum(item.total_tokens for item in opportunities),
        prompt_tokens=sum(item.prompt_tokens for item in opportunities),
        valid_loss_tokens=sum(item.valid_loss_tokens for item in opportunities),
        non_loss_suffix_tokens=sum(
            item.non_loss_suffix_tokens for item in opportunities
        ),
        shareable_prompt_tokens=sum(
            item.shareable_prompt_tokens for item in opportunities
        ),
        ideal_shared_token_work=sum(
            item.ideal_shared_token_work for item in opportunities
        ),
        loss_ratio_upper_bound_saved_tokens=sum(
            item.loss_ratio_upper_bound_saved_tokens for item in opportunities
        ),
    )


def observe_shared_prefix_opportunity(
    *,
    group_ids: Sequence[str | None],
    prompt_token_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    expected_group_size: int,
) -> SharedPrefixOpportunity:
    """Measure exact-prompt sharing opportunity for one driver-side batch.

    Stable rollout-group identity is authoritative. A group is eligible only
    when it has exactly ``expected_group_size`` rows and every row has the same
    complete, unpadded prompt token sequence. Incomplete groups and rows with no
    identity are counted as fallback; distinct IDs are never merged merely
    because their prompts happen to match. The calculation is CPU-only and does
    not reorder or mutate the training batch.

    ``valid_loss_tokens`` intentionally matches Megatron's GRPO accounting:
    ``token_mask[:, 1:] * sample_mask``. This makes the derived upper-bound
    metric directly comparable with ``train/global_valid_toks /
    train/total_num_tokens``.
    """
    if expected_group_size < 2:
        raise ValueError(
            "expected_group_size must be at least 2 for shared-prefix training, "
            f"got {expected_group_size}"
        )
    if prompt_token_ids.ndim != 2:
        raise ValueError(
            "prompt_token_ids must have shape [batch, sequence], "
            f"got {tuple(prompt_token_ids.shape)}"
        )
    if token_mask.ndim != 2:
        raise ValueError(
            f"token_mask must have shape [batch, sequence], got {tuple(token_mask.shape)}"
        )

    prompt_token_ids = prompt_token_ids.detach().cpu()
    prompt_lengths = prompt_lengths.detach().cpu().reshape(-1).to(torch.int64)
    input_lengths = input_lengths.detach().cpu().reshape(-1).to(torch.int64)
    token_mask = token_mask.detach().cpu()
    sample_mask = sample_mask.detach().cpu().reshape(-1)

    batch_size = prompt_token_ids.shape[0]
    row_counts = {
        "group_ids": len(group_ids),
        "prompt_lengths": prompt_lengths.numel(),
        "input_lengths": input_lengths.numel(),
        "token_mask": token_mask.shape[0],
        "sample_mask": sample_mask.numel(),
    }
    mismatched = {name: size for name, size in row_counts.items() if size != batch_size}
    if mismatched:
        raise ValueError(
            f"shared-prefix metric inputs must have {batch_size} rows; got {mismatched}"
        )

    max_prompt_length = prompt_token_ids.shape[1]
    if torch.any(prompt_lengths < 0) or torch.any(prompt_lengths > max_prompt_length):
        raise ValueError(
            "prompt_lengths must be within the prompt_token_ids width "
            f"[0, {max_prompt_length}]"
        )
    if torch.any(input_lengths < prompt_lengths):
        raise ValueError("input_lengths cannot be shorter than prompt_lengths")

    grouped_prompts: dict[str, list[tuple[int, ...]]] = {}
    missing_group_rows = 0
    for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt = tuple(prompt_token_ids[row_index, :prompt_length].tolist())
        group_id = group_ids[row_index]
        if group_id is None:
            missing_group_rows += 1
            continue
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                "shared-prefix group IDs must be non-empty strings or None; "
                f"row {row_index} has {group_id!r}"
            )
        grouped_prompts.setdefault(group_id, []).append(prompt)

    complete_groups = 0
    eligible_sequences = 0
    shareable_prompt_tokens = 0
    fallback_sequences = missing_group_rows
    for group_id, prompts in grouped_prompts.items():
        unique_prompts = set(prompts)
        if len(unique_prompts) != 1:
            raise ValueError(
                f"shared-prefix group {group_id!r} maps to multiple exact prompts"
            )
        if len(prompts) > expected_group_size:
            raise ValueError(
                f"shared-prefix group {group_id!r} has {len(prompts)} rows, "
                f"exceeding expected_group_size={expected_group_size}"
            )
        if len(prompts) < expected_group_size:
            fallback_sequences += len(prompts)
            continue
        prompt = next(iter(unique_prompts))
        complete_groups += 1
        eligible_sequences += expected_group_size
        shareable_prompt_tokens += len(prompt) * (expected_group_size - 1)

    total_tokens = int(input_lengths.sum().item())
    prompt_tokens = int(prompt_lengths.sum().item())
    valid_loss_tokens = float(
        (token_mask[:, 1:] * sample_mask.unsqueeze(-1)).sum().item()
    )
    non_loss_suffix_tokens = float(total_tokens - prompt_tokens) - valid_loss_tokens
    if non_loss_suffix_tokens < -1e-5:
        raise ValueError(
            "valid loss-token accounting exceeds the non-prompt suffix: "
            f"total={total_tokens}, prompt={prompt_tokens}, valid={valid_loss_tokens}"
        )

    ideal_shared_token_work = total_tokens - shareable_prompt_tokens
    if ideal_shared_token_work < 0:
        raise ValueError(
            "shared-prefix token savings exceed total token work; prompt/input "
            "length metadata are inconsistent"
        )

    loss_ratio_upper_bound_saved_tokens = (
        (float(total_tokens) - valid_loss_tokens)
        * (expected_group_size - 1)
        / expected_group_size
    )
    return SharedPrefixOpportunity(
        total_sequences=batch_size,
        eligible_sequences=eligible_sequences,
        complete_groups=complete_groups,
        fallback_sequences=fallback_sequences,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        valid_loss_tokens=valid_loss_tokens,
        non_loss_suffix_tokens=max(non_loss_suffix_tokens, 0.0),
        shareable_prompt_tokens=shareable_prompt_tokens,
        ideal_shared_token_work=ideal_shared_token_work,
        loss_ratio_upper_bound_saved_tokens=loss_ratio_upper_bound_saved_tokens,
    )
