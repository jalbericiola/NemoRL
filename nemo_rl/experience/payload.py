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

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from nemo_rl.data_plane.codec import pack_jagged_fields
from nemo_rl.data_plane.column_io import TOKEN_ALIGNED_FIELDS
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    INVALID_TOOL_CALL_TOKEN_MASK,
    MALFORMED_THINKING_MESSAGE_COUNT,
    MALFORMED_THINKING_TOKEN_MASK,
    PRE_PENALTY_REWARD,
    RAW_ENVIRONMENT_REWARD,
    RESPONSE_TOKEN_LENGTHS,
    ROLLOUT_TRUNCATED,
    PromptGroupRecord,
    completion_reward_boundaries,
)


def _strict_optional_bool(values: Mapping[str, Any], key: str, *, context: str) -> bool:
    """Read an optional boolean without truthiness coercion."""
    if key not in values:
        return False
    value = values[key]
    if type(value) is not bool:
        raise TypeError(f"{context}.{key} must be a bool, got {type(value).__name__}")
    return value


def _generated_assistant_layout(
    message_logs: list[list[dict[str, Any]]],
    *,
    prompt_token_count: int,
) -> tuple[list[list[bool]], torch.Tensor]:
    """Locate rollout-generated assistants using both marker and prompt boundary.

    Prompt history may itself contain assistant turns (and reconstructed payloads may
    carry zero-filled ``generation_logprobs`` for them).  A response is therefore a
    marked assistant turn whose first token is beyond the immutable prompt boundary.
    Reward shaping retains the first such response length, matching the native
    single-response convention while excluding assistant prompt history.
    """
    layouts: list[list[bool]] = []
    response_lengths: list[int] = []
    for completion_idx, message_log in enumerate(message_logs):
        token_offset = 0
        layout: list[bool] = []
        response_length = 0
        found_response = False
        for message_idx, message in enumerate(message_log):
            token_ids = message.get("token_ids")
            if not isinstance(token_ids, torch.Tensor):
                raise TypeError(
                    "payload conversion requires tensor token_ids at "
                    f"completion {completion_idx}, message {message_idx}"
                )
            message_length = int(token_ids.shape[0])
            if token_offset < prompt_token_count < token_offset + message_length:
                raise ValueError(
                    "prompt token boundary splits a message at "
                    f"completion {completion_idx}, message {message_idx}"
                )
            generated_assistant = (
                token_offset >= prompt_token_count
                and message.get("role") == "assistant"
                and "generation_logprobs" in message
            )
            layout.append(generated_assistant)
            if generated_assistant and not found_response:
                response_length = message_length
                found_response = True
            token_offset += message_length
        if token_offset < prompt_token_count:
            raise ValueError(
                "completion is shorter than its prompt token boundary: "
                f"{token_offset} < {prompt_token_count} at completion "
                f"{completion_idx}"
            )
        layouts.append(layout)
        response_lengths.append(response_length)
    return layouts, torch.tensor(response_lengths, dtype=torch.long)


def _message_level_advantage_fields(
    message_logs: list[list[dict[str, Any]]],
    generated_assistant_layout: list[list[bool]],
) -> dict[str, torch.Tensor]:
    """Build exact token masks and message counts for Gym advantage penalties."""
    invalid_rows: list[torch.Tensor] = []
    malformed_rows: list[torch.Tensor] = []
    assistant_counts: list[int] = []
    invalid_counts: list[int] = []
    malformed_counts: list[int] = []
    overlap_counts: list[int] = []

    for completion_idx, message_log in enumerate(message_logs):
        invalid_parts: list[torch.Tensor] = []
        malformed_parts: list[torch.Tensor] = []
        assistant_count = 0
        invalid_count = 0
        malformed_count = 0
        overlap_count = 0
        for message, generated_assistant in zip(
            message_log,
            generated_assistant_layout[completion_idx],
            strict=True,
        ):
            token_ids = message.get("token_ids")
            if not isinstance(token_ids, torch.Tensor):
                raise TypeError(
                    "message-level advantage masks require tensor token_ids"
                )
            invalid_detector = _strict_optional_bool(
                message,
                "is_invalid_tool_call",
                context="message",
            )
            malformed_detector = _strict_optional_bool(
                message,
                "has_malformed_thinking",
                context="message",
            )
            invalid = generated_assistant and invalid_detector
            malformed = generated_assistant and malformed_detector
            invalid_parts.append(torch.full_like(token_ids, invalid, dtype=torch.bool))
            malformed_parts.append(
                torch.full_like(token_ids, malformed, dtype=torch.bool)
            )
            assistant_count += int(generated_assistant)
            invalid_count += int(invalid)
            malformed_count += int(malformed)
            overlap_count += int(invalid and malformed)

        invalid_rows.append(torch.cat(invalid_parts))
        malformed_rows.append(torch.cat(malformed_parts))
        assistant_counts.append(assistant_count)
        invalid_counts.append(invalid_count)
        malformed_counts.append(malformed_count)
        overlap_counts.append(overlap_count)

    return {
        INVALID_TOOL_CALL_TOKEN_MASK: pad_sequence(
            invalid_rows, batch_first=True, padding_value=False
        ),
        MALFORMED_THINKING_TOKEN_MASK: pad_sequence(
            malformed_rows, batch_first=True, padding_value=False
        ),
        GENERATED_ASSISTANT_MESSAGE_COUNT: torch.tensor(
            assistant_counts, dtype=torch.long
        ),
        INVALID_TOOL_CALL_MESSAGE_COUNT: torch.tensor(invalid_counts, dtype=torch.long),
        MALFORMED_THINKING_MESSAGE_COUNT: torch.tensor(
            malformed_counts, dtype=torch.long
        ),
        INVALID_AND_MALFORMED_MESSAGE_COUNT: torch.tensor(
            overlap_counts, dtype=torch.long
        ),
    }


def _completion_is_loss_masked(completion: Any) -> bool:
    """Return the producer-resolved Gym loss gate without inspecting env data."""
    value = completion.env_masked
    if type(value) is not bool:
        raise TypeError(
            f"Completion.env_masked must be a bool, got {type(value).__name__}"
        )
    return value


def record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Mapping[str, int],
    include_shared_prefix_metadata: bool = False,
    include_reward_processing_metadata: bool = False,
    include_message_advantage_metadata: bool = False,
) -> BatchedDataDict[Any]:
    """Convert one prompt group's record into a packed BatchedDataDict of N rows.

    Args:
        record: Rollout's PromptGroupRecord with N completions to flatten into rows.
        pad_value_dict: Field-name → pad value used by batched_message_log_to_flat_message.
        include_shared_prefix_metadata: Add the exact prompt length needed by
            observation and shared-prefix training. Group identity is added by
            :func:`pack_payload`, which owns the final sample-ID namespace.
        include_reward_processing_metadata: Retain truncation and response-length
            scalars used by overlong filtering and reward shaping.
        include_message_advantage_metadata: Retain exact Gym detector token masks
            and message counts used by configured advantage penalties.

    Returns:
        BatchedDataDict with input_ids, input_lengths, generation_logprobs, token_mask,
        sample_mask, prompt_ids_for_adv, total_reward, and optional routed_experts.
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
    if "loss_multiplier" not in record.metadata:
        raise ValueError("PromptGroupRecord metadata requires loss_multiplier")
    loss_multiplier = float(record.metadata["loss_multiplier"])
    if not math.isfinite(loss_multiplier) or loss_multiplier < 0:
        raise ValueError(
            "PromptGroupRecord loss_multiplier must be finite and nonnegative, "
            f"got {loss_multiplier}"
        )

    # The flattening helper adds loss masks and zero-filled logprobs in place.  Work
    # on shallow message copies so converting the same immutable rollout record a
    # second time cannot reinterpret prompt-history assistants as generated turns.
    message_logs = [
        [dict(message) for message in completion.message_log]
        for completion in completions
    ]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    generated_assistant_layout, response_token_lengths = _generated_assistant_layout(
        message_logs,
        prompt_token_count=prompt_token_count,
    )

    # Reconstructed payloads can contain zero-filled generation_logprobs on prompt
    # turns.  Strip those local synthetic markers before the GRPO helper derives
    # token loss masks.
    for message_log, generated_layout in zip(
        message_logs, generated_assistant_layout, strict=True
    ):
        for message, generated_assistant in zip(
            message_log, generated_layout, strict=True
        ):
            if not generated_assistant:
                message.pop("generation_logprobs", None)

    advantage_penalty_fields = (
        _message_level_advantage_fields(message_logs, generated_assistant_layout)
        if include_message_advantage_metadata
        else {}
    )
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

    reward_boundaries = [completion_reward_boundaries(c) for c in completions]
    raw_environment_reward = torch.tensor(
        [values[0] for values in reward_boundaries], dtype=torch.float32
    )
    pre_penalty_reward = torch.tensor(
        [values[1] for values in reward_boundaries], dtype=torch.float32
    )
    total_reward = torch.tensor(
        [values[2] for values in reward_boundaries], dtype=torch.float32
    )
    sample_mask = torch.tensor(
        [
            0.0 if _completion_is_loss_masked(c) else loss_multiplier
            for c in completions
        ],
        dtype=torch.float32,
    )

    train_data: dict[str, Any] = {
        "input_ids": flat["token_ids"],
        "input_lengths": input_lengths,
        "generation_logprobs": flat["generation_logprobs"],
        "token_mask": flat["token_loss_mask"],
        "sample_mask": sample_mask,
        "prompt_ids_for_adv": prompt_flat["token_ids"],
        RAW_ENVIRONMENT_REWARD: raw_environment_reward,
        PRE_PENALTY_REWARD: pre_penalty_reward,
        "total_reward": total_reward,
        **advantage_penalty_fields,
    }
    if include_reward_processing_metadata:
        train_data[ROLLOUT_TRUNCATED] = torch.tensor(
            [bool(c.truncated) for c in completions], dtype=torch.bool
        )
        train_data[RESPONSE_TOKEN_LENGTHS] = response_token_lengths
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
            an object column for policy workers. Disabled mode keeps the legacy
            payload schema byte-for-byte unchanged.

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
