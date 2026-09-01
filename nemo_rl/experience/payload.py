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

"""Producer-side payload helpers for the async-RL TQ path."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import pack_jagged_fields
from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord


def record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Mapping[str, int],
    include_shared_prefix_metadata: bool = False,
) -> BatchedDataDict[Any]:
    """Convert one prompt group's record into a packed BatchedDataDict of N rows.

    Args:
        record: Rollout's PromptGroupRecord with N completions to flatten into rows.
        pad_value_dict: Field-name → pad value used by batched_message_log_to_flat_message.
        include_shared_prefix_metadata: Add the exact prompt length needed by
            observation and shared-prefix training. Group identity is added by
            :func:`pack_payload`, which owns the final sample-ID namespace.

    Returns:
        BatchedDataDict with input_ids, input_lengths, generation_logprobs, token_mask,
        sample_mask, prompt_ids_for_adv, total_reward, truncated,
        response_token_lengths, and optional routed_experts.
    """
    # Lazy imports: grpo and llm_message_utils transitively pull
    # experience.rollouts, so importing at module top risks a cycle.
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.experience.rollouts import backfill_missing_routed_experts

    completions = record.completions
    n = len(completions)
    assert n > 0, "PromptGroupRecord has no completions"

    message_logs = [c.message_log for c in completions]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    prompt_lengths = torch.full((n,), prompt_token_count, dtype=torch.long)

    # Must precede the prompt extraction: it reuses the same message dicts, so
    # backfilling here also covers the prompt flatten below. Doing it only inside
    # add_grpo_token_loss_masks_and_generation_logprobs would be too late.
    backfill_missing_routed_experts(message_logs)

    prompt_message_logs = extract_initial_prompt_messages(message_logs, prompt_lengths)
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,  # type: ignore
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    total_reward = torch.tensor(
        [float(c.reward) for c in completions], dtype=torch.float32
    )
    truncated = torch.tensor([c.truncated for c in completions], dtype=torch.bool)

    def _first_assistant_token_length(completion: Any) -> int:
        for message in completion.message_log:
            if message["role"] != "assistant":
                continue
            token_ids = message.get("token_ids")
            if token_ids is None:
                continue
            return (
                int(token_ids.shape[0])
                if isinstance(token_ids, torch.Tensor)
                else len(token_ids)
            )
        return 0

    response_token_lengths = torch.tensor(
        [_first_assistant_token_length(c) for c in completions],
        dtype=torch.long,
    )
    sample_mask = torch.ones(n, dtype=torch.float32)

    train_data: dict[str, Any] = {
        "input_ids": flat["token_ids"],
        "input_lengths": input_lengths,
        "generation_logprobs": flat["generation_logprobs"],
        "token_mask": flat["token_loss_mask"],
        "sample_mask": sample_mask,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        "total_reward": total_reward,
        "truncated": truncated,
        "response_token_lengths": response_token_lengths,
    }
    if ROUTED_EXPERTS_FIELD in flat:
        train_data[ROUTED_EXPERTS_FIELD] = flat[ROUTED_EXPERTS_FIELD]
    if include_shared_prefix_metadata:
        train_data[SHARED_PREFIX_PROMPT_LENGTHS] = prompt_lengths
    return BatchedDataDict[Any](train_data)


def pack_payload(
    train_batch: Mapping[str, Any],
    *,
    weight_version: int,
    group_id: str,
    include_shared_prefix_metadata: bool = False,
) -> tuple[list[str], TensorDict, list[dict[str, Any]]]:
    """Pack a producer batch into (sample_ids, fields, tags) for put_samples.

    Args:
        train_batch: Mapping with at least input_lengths plus the tensor/object fields to send.
        weight_version: Trainer weight version stamped on every row's tag.
        group_id: Per-group identifier used as the sample_id prefix; the caller owns uniqueness.
        include_shared_prefix_metadata: Store the same stable group identity as
            an object column for policy workers. Disabled mode omits only the
            shared-prefix metadata; rollout fields such as ``truncated`` and
            ``response_token_lengths`` remain part of the current payload schema.

    Returns:
        sample_ids of the form {group_id}_g{i}, a jagged-packed TensorDict, and per-row tags.
    """
    lengths = train_batch["input_lengths"]
    n = int(lengths.shape[0])
    tensor_fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in train_batch.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    if include_shared_prefix_metadata:
        if not isinstance(lengths, torch.Tensor) or lengths.shape != (n,):
            shape = (
                tuple(lengths.shape)
                if isinstance(lengths, (torch.Tensor, np.ndarray))
                else None
            )
            raise ValueError(
                "shared-prefix input lengths must be a dense tensor with "
                f"shape ({n},), got {type(lengths).__name__} with shape {shape}"
            )
        prompt_lengths = tensor_fields.get(SHARED_PREFIX_PROMPT_LENGTHS)
        if prompt_lengths is None:
            raise ValueError(
                "shared-prefix payload metadata requires prompt lengths from "
                "record_to_train_batch"
            )
        if not isinstance(prompt_lengths, torch.Tensor) or prompt_lengths.shape != (n,):
            shape = (
                tuple(prompt_lengths.shape)
                if isinstance(prompt_lengths, (torch.Tensor, np.ndarray))
                else None
            )
            raise ValueError(
                "shared-prefix prompt lengths must be a dense tensor with "
                f"shape ({n},), got {type(prompt_lengths).__name__} with shape {shape}"
            )
        if torch.any(prompt_lengths < 0) or torch.any(prompt_lengths > lengths):
            raise ValueError(
                "shared-prefix prompt lengths must be nonnegative and no larger "
                "than input_lengths"
            )
        if not group_id:
            raise ValueError("shared-prefix group_id must be non-empty")
        tensor_fields[SHARED_PREFIX_GROUP_ID] = np.asarray(
            [group_id] * n,
            dtype=object,
        )
    fields_td = pack_jagged_fields(
        tensor_fields, lengths=lengths, token_aligned_fields=TOKEN_ALIGNED_FIELDS
    )
    sample_ids = [f"{group_id}_g{i}" for i in range(n)]
    tags = [{"weight_version": weight_version} for _ in range(n)]
    return sample_ids, fields_td, tags
