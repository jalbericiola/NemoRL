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

from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.data.interfaces import LLMMessageLogType, VLMMessageLogType

# Gym reserves this prefix for its own row bookkeeping (task/rollout/attempt
# ids, failure flags). The values are identifiers, not measurements, so they are
# excluded from the agent metrics aggregated out of env_extras.
NEMO_GYM_RESERVED_KEY_PREFIX = "_ng_"
NEMO_GYM_TASK_INDEX_KEY = "_ng_task_index"
NEMO_GYM_ROLLOUT_INDEX_KEY = "_ng_rollout_index"
NEXT_NEMO_GYM_TASK_INDEX_KEY = "next_ng_task_index"

# Slim DataPlane columns used to preserve message-level Gym detector semantics
# without shipping the full Python message log through TransferQueue.
INVALID_TOOL_CALL_TOKEN_MASK = "invalid_tool_call_token_mask"
MALFORMED_THINKING_TOKEN_MASK = "malformed_thinking_token_mask"
GENERATED_ASSISTANT_MESSAGE_COUNT = "generated_assistant_message_count"
INVALID_TOOL_CALL_MESSAGE_COUNT = "invalid_tool_call_message_count"
MALFORMED_THINKING_MESSAGE_COUNT = "malformed_thinking_message_count"
INVALID_AND_MALFORMED_MESSAGE_COUNT = "invalid_and_malformed_message_count"
ROLLOUT_TRUNCATED = "truncated"
RESPONSE_TOKEN_LENGTHS = "response_token_lengths"


@dataclass
class Completion:
    """A single generated completion for one prompt."""

    message_log: LLMMessageLogType | VLMMessageLogType
    env_extras: Optional[dict[str, Any]]
    truncated: bool
    reward: float
    # Resolved, provenance-preserving Gym loss gate. Native environments leave
    # this false; payload consumers must never infer it from arbitrary env_extras.
    env_masked: bool = False


@dataclass
class PromptGroupRecord:
    """All completions for a single prompt, with prompt-level metadata."""

    prompt_idx: int
    prompt: LLMMessageLogType | VLMMessageLogType
    extra_env_info: Optional[dict[str, Any]]
    metadata: dict[str, Any]
    completions: list["Completion"]
    rollout_metrics: dict[str, Any]
