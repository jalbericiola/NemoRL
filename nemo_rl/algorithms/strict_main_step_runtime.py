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

"""Runtime bridge from consumed SingleController tensors to strict evidence."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from nemo_rl.algorithms.single_controller_utils.utils import (
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.codec import unwrap_wire_stripped_payload
from nemo_rl.experience.interfaces import (
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    MALFORMED_THINKING_MESSAGE_COUNT,
    PRE_PENALTY_REWARD,
    RAW_ENVIRONMENT_REWARD,
)
from nemo_rl.utils.strict_captured_replay_evidence import (
    TRANSCRIPT_BUNDLE_SCHEMA,
    build_verifier_request_derivation,
    load_evidence_document,
    main_transcript_bundle_path,
    model_response_token_geometry,
    validate_ledger_transcript_join,
    validate_transcript_bundle,
    validate_transcript_model_transport_join,
)
from nemo_rl.utils.strict_main_step_ledger import (
    MAIN_STEP1_LEDGER_SCHEMA,
    MAIN_STEP1_REWARD_PENALTY_TAGS,
    MAIN_STEP1_TAG_FIXTURE_ROW_INDEX,
    MAIN_STEP1_TAG_GENERATION_SEED,
    MAIN_STEP1_TAG_ROLLOUT_INDEX,
    MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256,
    build_main_step1_ledger,
    main_step1_runtime_contract,
    publish_main_step1_ledger,
    strict_main_step1_enabled,
)
from nemo_rl.utils.strict_model_transport import (
    MODEL_TRANSPORT_DIRECTORY,
    build_model_transport_manifest,
    load_finalized_model_transport_capture,
    publish_model_transport_manifest,
)


_STRICT_REQUIRED_FIELDS = (
    "input_ids",
    "input_lengths",
    RAW_ENVIRONMENT_REWARD,
    PRE_PENALTY_REWARD,
    SHARED_PREFIX_GROUP_ID,
    SHARED_PREFIX_PROMPT_LENGTHS,
    GENERATED_ASSISTANT_MESSAGE_COUNT,
    INVALID_TOOL_CALL_MESSAGE_COUNT,
    MALFORMED_THINKING_MESSAGE_COUNT,
    INVALID_AND_MALFORMED_MESSAGE_COUNT,
)
_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
_TRANSCRIPT_BINDING_NAMES = (
    "pair_manifest_sha256",
    "submission_receipt_sha256",
    "job_id",
    "run_id",
    "fixture_sha256",
    "verifier_source_sha256",
    "config_sha256",
    "snapshot_manifest_sha256",
)


class _DisabledStrictMainStep1Recorder:
    """Explicit class-level default for unit fixtures that bypass actor init."""

    enabled = False

    @staticmethod
    def required_fields() -> list[str]:
        return []

    @staticmethod
    def capture_consumed_rows(**_: Any) -> None:
        return None

    @staticmethod
    def publish_after_successful_step(**_: Any) -> None:
        return None


DISABLED_STRICT_MAIN_STEP1_RECORDER = _DisabledStrictMainStep1Recorder()


class StrictMainStep1Recorder:
    """Capture the K=4 rows consumed by optimizer step one and publish once."""

    def __init__(
        self,
        *,
        master_config: Any,
        shared_prefix_mode: str,
        current_step: int,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.enabled = strict_main_step1_enabled(environ)
        self._row_inputs: list[dict[str, Any]] = []
        self._published = False
        self._contract: dict[str, Any] | None = None
        self._generation: dict[str, Any] | None = None
        self._transcript_bundle_ref: dict[str, str] | None = None
        self._transcript_bundle_document: dict[str, Any] | None = None
        self._model_transport_policy: dict[str, Any] | None = None
        self._model_transport_bundle_ref: dict[str, Any] | None = None
        self._model_transport_bundle_document: dict[str, Any] | None = None
        self._model_transport_capture_ref: dict[str, Any] | None = None
        self._model_path: str | None = None
        if not self.enabled:
            return

        contract = main_step1_runtime_contract(environ)
        if current_step != 0 or isinstance(current_step, bool):
            raise RuntimeError(
                "strict main-step evidence requires a fresh run at training step zero"
            )
        grpo = master_config.grpo
        if grpo.num_prompts_per_step != 1 or grpo.num_generations_per_prompt != 4:
            raise RuntimeError(
                "strict main-step evidence requires one prompt and K=4 generations"
            )
        if shared_prefix_mode != contract["mode"]:
            raise RuntimeError(
                "strict arm does not match policy.shared_prefix_training.mode: "
                f"arm={contract['arm']!r}, expected={contract['mode']!r}, "
                f"actual={shared_prefix_mode!r}"
            )
        generation_config = master_config.policy["generation"]
        if generation_config.get("nemo_gym_add_seed_per_rollout") is not True:
            raise RuntimeError(
                "strict main-step evidence requires deterministic per-rollout Gym seeds"
            )
        vllm_config = generation_config.get("vllm_cfg")
        if (
            not isinstance(vllm_config, Mapping)
            or vllm_config.get("strict_model_transport") != "capture"
        ):
            raise RuntimeError(
                "strict main-step evidence requires "
                "policy.generation.vllm_cfg.strict_model_transport='capture'"
            )
        generation = {
            "seed_base": generation_config["nemo_gym_per_rollout_seed_base"],
            "max_new_tokens": generation_config["max_new_tokens"],
            "temperature": generation_config["temperature"],
            "top_k": generation_config["top_k"],
            "top_p": generation_config["top_p"],
        }
        model_path = generation_config.get("model_name")
        if not isinstance(model_path, str) or not model_path:
            raise RuntimeError(
                "strict main-step evidence requires generation.model_name"
            )
        # The ledger builder performs the final exact scalar/key validation.
        self._contract = contract
        self._generation = generation
        self._model_path = model_path

    def required_fields(self) -> list[str]:
        """Return extra DataPlane fields needed only by strict evidence."""
        return list(_STRICT_REQUIRED_FIELDS) if self.enabled else []

    def capture_consumed_rows(
        self,
        *,
        step_index: int,
        meta: KVBatchMeta,
        data: TensorDict,
        prompt_ids: torch.Tensor,
        verifier_rewards: torch.Tensor,
        processed_rewards: torch.Tensor,
        token_loss_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        advantages: torch.Tensor,
        invalid_advantage_enabled: bool,
        malformed_advantage_enabled: bool,
    ) -> None:
        """Copy final, unpadded trainer inputs before their DP rows are cleared."""
        if not self.enabled:
            return
        if step_index != 0 or isinstance(step_index, bool):
            return
        if self._published:
            raise RuntimeError("strict main-step ledger was already published")
        if self._row_inputs:
            raise RuntimeError(
                "strict main-step evidence received more than one training chunk"
            )
        if len(meta.sample_ids) != 4 or meta.tags is None or len(meta.tags) != 4:
            raise RuntimeError(
                "strict main-step evidence requires exactly four tagged trainer rows"
            )

        input_ids = tensor_field(data, "input_ids")
        input_lengths = squeeze_trailing_unit_dim(tensor_field(data, "input_lengths"))
        prompt_lengths = squeeze_trailing_unit_dim(
            tensor_field(data, SHARED_PREFIX_PROMPT_LENGTHS)
        )
        raw_rewards = squeeze_trailing_unit_dim(
            tensor_field(data, RAW_ENVIRONMENT_REWARD)
        )
        pre_penalty_rewards = squeeze_trailing_unit_dim(
            tensor_field(data, PRE_PENALTY_REWARD)
        )
        group_ids = _string_object_field(data, SHARED_PREFIX_GROUP_ID)
        invalid_counts = _message_count_values(
            data, INVALID_TOOL_CALL_MESSAGE_COUNT, expected_rows=4
        )
        malformed_counts = _message_count_values(
            data, MALFORMED_THINKING_MESSAGE_COUNT, expected_rows=4
        )
        overlap_counts = _message_count_values(
            data, INVALID_AND_MALFORMED_MESSAGE_COUNT, expected_rows=4
        )
        # Fetching the denominator is intentional even though the row schema only
        # stores booleans: it rejects impossible detector counts before evidence.
        assistant_counts = _message_count_values(
            data, GENERATED_ASSISTANT_MESSAGE_COUNT, expected_rows=4
        )
        _require_batch_rows(
            4,
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_ids=prompt_ids,
            prompt_lengths=prompt_lengths,
            raw_rewards=raw_rewards,
            pre_penalty_rewards=pre_penalty_rewards,
            verifier_rewards=verifier_rewards,
            processed_rewards=processed_rewards,
            token_loss_mask=token_loss_mask,
            sample_mask=sample_mask,
            advantages=advantages,
        )
        if len(group_ids) != 4:
            raise RuntimeError("strict main-step group-id column must have four rows")

        rows: list[dict[str, Any]] = []
        transcript_digests: set[str] = set()
        for row_index in range(4):
            tag = meta.tags[row_index]
            if not isinstance(tag, Mapping):
                raise RuntimeError("strict row tags must be mappings")
            fixture_row_index = _exact_tag_int(tag, MAIN_STEP1_TAG_FIXTURE_ROW_INDEX)
            rollout_index = _exact_tag_int(tag, MAIN_STEP1_TAG_ROLLOUT_INDEX)
            generation_seed = _exact_tag_int(tag, MAIN_STEP1_TAG_GENERATION_SEED)
            transcript_digests.add(
                _exact_tag_sha256(tag, MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256)
            )
            if fixture_row_index != 0 or rollout_index != row_index:
                raise RuntimeError(
                    "strict step one must consume fixture row zero in rollout order 0..3"
                )
            group_id = group_ids[row_index]
            if meta.sample_ids[row_index] != f"{group_id}_g{rollout_index}":
                raise RuntimeError(
                    "strict main-step sample ID is not bound to its shared-prefix group"
                )

            input_length = _exact_positive_tensor_int(
                input_lengths[row_index], name="input_length"
            )
            prompt_length = _exact_positive_tensor_int(
                prompt_lengths[row_index], name="prompt_length"
            )
            if input_length > 131_072 or prompt_length > 131_072:
                raise RuntimeError(
                    "strict input and prompt lengths must be at most 131072"
                )
            if prompt_length >= input_length:
                raise RuntimeError(
                    "strict main-step completion must contain at least one token"
                )
            token_ids = _int_tensor_list(input_ids[row_index, :input_length])
            prompt_token_ids = _int_tensor_list(prompt_ids[row_index, :prompt_length])
            if token_ids[:prompt_length] != prompt_token_ids:
                raise RuntimeError(
                    "strict advantage prompt is not the exact trainer-input prefix"
                )
            completion_token_ids = token_ids[prompt_length:]
            row_token_mask = _float_tensor_list(
                token_loss_mask[row_index, :input_length]
            )
            row_advantages = _float_tensor_list(advantages[row_index, :input_length])
            if any(value != 0.0 for value in row_token_mask[:prompt_length]):
                raise RuntimeError("strict prompt prefix has nonzero token loss mask")
            valid_loss_tokens = sum(value == 1.0 for value in row_token_mask)

            raw_invalid = invalid_counts[row_index] > 0
            raw_malformed = malformed_counts[row_index] > 0
            overlap = overlap_counts[row_index]
            if (
                overlap > invalid_counts[row_index]
                or overlap > malformed_counts[row_index]
                or invalid_counts[row_index] > assistant_counts[row_index]
                or malformed_counts[row_index] > assistant_counts[row_index]
            ):
                raise RuntimeError(
                    "strict row has inconsistent message detector counts"
                )
            effective_malformed_count = (
                malformed_counts[row_index]
                - (overlap if invalid_advantage_enabled else 0)
                if malformed_advantage_enabled
                else 0
            )
            penalty_flags = {
                name: _exact_tag_bool(tag, tag_name)
                for name, tag_name in MAIN_STEP1_REWARD_PENALTY_TAGS.items()
            }
            penalty_flags.update(
                {
                    "invalid_tool_call": (
                        raw_invalid if invalid_advantage_enabled else False
                    ),
                    "malformed_thinking": effective_malformed_count > 0,
                    "raw_invalid_tool_call": raw_invalid,
                    "raw_malformed_thinking": raw_malformed,
                    "invalid_and_malformed": overlap > 0,
                }
            )
            rows.append(
                {
                    "sample_index": row_index,
                    "sample_id": meta.sample_ids[row_index],
                    "shared_prefix_group_id": group_id,
                    "fixture_row_index": fixture_row_index,
                    "rollout_index": rollout_index,
                    "generation_seed": generation_seed,
                    "token_ids": token_ids,
                    "input_length": input_length,
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": completion_token_ids,
                    "token_loss_mask": row_token_mask,
                    "raw_environment_reward": _tensor_float(raw_rewards[row_index]),
                    "pre_penalty_environment_reward": _tensor_float(
                        pre_penalty_rewards[row_index]
                    ),
                    "penalty_flags": penalty_flags,
                    "verifier_reward": _tensor_float(verifier_rewards[row_index]),
                    "processed_reward": _tensor_float(processed_rewards[row_index]),
                    "sample_mask": _tensor_float(sample_mask[row_index]),
                    "advantages": row_advantages,
                    "valid_loss_tokens": valid_loss_tokens,
                    "total_tokens": input_length,
                }
            )
        if len(transcript_digests) != 1:
            raise RuntimeError(
                "strict consumed K=4 rows do not bind one transcript bundle"
            )
        transcript_digest = transcript_digests.pop()
        bound_rows, transcript_ref = self._bind_transcript_rows(
            rows=rows, expected_sha256=transcript_digest
        )
        if self._contract is None or self._generation is None:
            raise AssertionError("active strict main-step recorder has no contract")
        self._row_inputs = bound_rows
        self._transcript_bundle_ref = transcript_ref

    def _bind_transcript_rows(
        self, *, rows: list[dict[str, Any]], expected_sha256: str
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Stable-load the selected raw transcript and join it to trainer rows."""
        if self._contract is None or self._generation is None:
            raise AssertionError("active strict main-step recorder has no contract")
        path = main_transcript_bundle_path(self._contract["results_dir"])
        bundle, actual_sha256 = load_evidence_document(
            path=path,
            expected_sha256=expected_sha256,
            trailing_lf=False,
        )
        validate_transcript_bundle(bundle)
        expected_bindings = {
            name: self._contract["bindings"][name] for name in _TRANSCRIPT_BINDING_NAMES
        }
        verifier_request_derivation = bundle["verifier_request_derivation"]
        derivation_runtime = verifier_request_derivation["runtime"]
        expected_root = {
            "pair_id": self._contract["pair_id"],
            "environment": self._contract["environment"],
            "arm": self._contract["arm"],
            "mode": self._contract["mode"],
            "attempt_id": None,
            "step": 1,
            "sample_count": 4,
            "generation": self._generation,
            "bindings": expected_bindings,
            "verifier_request_derivation": build_verifier_request_derivation(
                gym_gitlink_commit=self._contract["gym_gitlink_commit"],
                gym_tree=self._contract["gym_tree"],
                openai_version=derivation_runtime["openai_version"],
                pydantic_version=derivation_runtime["pydantic_version"],
            ),
        }
        for name, expected in expected_root.items():
            if bundle.get(name) != expected:
                raise RuntimeError(
                    f"strict transcript bundle {name!r} differs from runtime contract"
                )
        entries = bundle.get("entries")
        if not isinstance(entries, list) or len(entries) != 4:
            raise RuntimeError(
                "strict transcript bundle must contain exact K=4 entries"
            )
        if self._model_path is None:
            raise AssertionError("active strict main-step recorder has no model path")
        source_root = Path(__file__).resolve().parents[2]
        model_transport_policy, model_transport_bundle, capture_ref, bundle_ref = (
            load_finalized_model_transport_capture(
                results_dir=self._contract["results_dir"],
                expected_bundle_ref=bundle["model_transport_bundle"],
                expected_policy_sha256=self._contract["model_transport_policy_sha256"],
                source_root=source_root,
                model_path=self._model_path,
                expected_generation_requests=[
                    entry["generation_request"] for entry in entries
                ],
                expected_model_responses=[entry["model_response"] for entry in entries],
            )
        )
        validate_transcript_model_transport_join(
            transcript_bundle=bundle,
            model_transport_bundle=model_transport_bundle,
            model_transport_policy=model_transport_policy,
            model_path=self._model_path,
        )

        bound_rows: list[dict[str, Any]] = []
        for index, (row, entry) in enumerate(zip(rows, entries, strict=True)):
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"strict transcript entry {index} is not an object")
            if any(
                entry.get(name) != row[name]
                for name in (
                    "sample_index",
                    "fixture_row_index",
                    "rollout_index",
                    "generation_seed",
                )
            ):
                raise RuntimeError(
                    f"strict transcript entry {index} identity differs from trainer row"
                )
            transcript_reward = entry.get("raw_environment_reward")
            if (
                type(transcript_reward) is not float
                or not math.isfinite(transcript_reward)
                or _float32_round_trip(transcript_reward)
                != row["raw_environment_reward"]
            ):
                raise RuntimeError(
                    f"strict transcript entry {index} native reward does not "
                    "round to the consumed float32 trainer reward"
                )
            _validate_model_response_token_join(
                entry.get("model_response"), row=row, row_index=index
            )
            bound_row = dict(row)
            bound_row.update(
                {
                    "request_sha256": entry["generation_request_sha256"],
                    "response_sha256": entry["model_response_sha256"],
                    "agent_run_request_sha256": entry["agent_run_request_sha256"],
                    "derived_verifier_request_sha256": entry[
                        "derived_verifier_request_sha256"
                    ],
                    "verifier_response_sha256": entry["verifier_response_sha256"],
                }
            )
            bound_rows.append(bound_row)
        reference = {
            "path": str(path),
            "schema": TRANSCRIPT_BUNDLE_SCHEMA,
            "sha256": actual_sha256,
        }
        self._transcript_bundle_document = bundle
        self._model_transport_policy = model_transport_policy
        self._model_transport_bundle_ref = bundle_ref
        self._model_transport_bundle_document = model_transport_bundle
        self._model_transport_capture_ref = capture_ref
        return bound_rows, reference

    def publish_after_successful_step(
        self, *, step_index: int, update_successful: bool
    ) -> tuple[Path, str] | None:
        """Publish only after ``finish_train_step`` returned successfully."""
        if not self.enabled:
            return None
        if step_index != 0 or isinstance(step_index, bool):
            return None
        if type(update_successful) is not bool or not update_successful:
            raise RuntimeError(
                "strict main-step evidence requires a successful optimizer update"
            )
        if self._published:
            raise RuntimeError("strict main-step ledger was already published")
        if len(self._row_inputs) != 4:
            raise RuntimeError(
                "successful strict optimizer step has no exact four-row evidence"
            )
        if (
            self._contract is None
            or self._generation is None
            or self._transcript_bundle_ref is None
            or self._transcript_bundle_document is None
            or self._model_transport_policy is None
            or self._model_transport_bundle_ref is None
            or self._model_transport_bundle_document is None
            or self._model_transport_capture_ref is None
            or self._model_path is None
        ):
            raise AssertionError("active strict main-step recorder has no contract")
        ledger = build_main_step1_ledger(
            pair_id=self._contract["pair_id"],
            environment=self._contract["environment"],
            arm=self._contract["arm"],
            mode=self._contract["mode"],
            generation=self._generation,
            bindings=self._contract["bindings"],
            transcript_bundle=self._transcript_bundle_ref,
            row_inputs=self._row_inputs,
            update_successful=update_successful,
        )
        validate_ledger_transcript_join(
            ledger=ledger,
            transcript_bundle=self._transcript_bundle_document,
        )
        path, digest = publish_main_step1_ledger(
            results_dir=self._contract["results_dir"], document=ledger
        )
        ledger_ref = {
            "path": str(path),
            "schema": MAIN_STEP1_LEDGER_SCHEMA,
            "sha256": digest,
        }
        manifest = build_model_transport_manifest(
            pair_id=self._contract["pair_id"],
            environment=self._contract["environment"],
            arm=self._contract["arm"],
            pair_manifest_sha256=self._contract["bindings"]["pair_manifest_sha256"],
            authenticated_job_id=self._contract["bindings"]["job_id"],
            submission_receipt_sha256=self._contract["bindings"][
                "submission_receipt_sha256"
            ],
            capture_server=self._model_transport_bundle_document["capture_server"],
            main_transcript_bundle=self._transcript_bundle_ref,
            main_ledger=ledger_ref,
            transport_bundle=self._model_transport_bundle_ref,
            transport_capture=self._model_transport_capture_ref,
            model_transport_policy_sha256=self._contract[
                "model_transport_policy_sha256"
            ],
            entry_count=self._model_transport_bundle_document["entry_count"],
            ordered_entries_sha256=self._model_transport_bundle_document[
                "ordered_entries_sha256"
            ],
        )
        publish_model_transport_manifest(
            transport_directory=(
                Path(self._contract["results_dir"]) / MODEL_TRANSPORT_DIRECTORY
            ),
            manifest=manifest,
        )
        self._published = True
        return path, digest


def _require_batch_rows(expected_rows: int, **values: torch.Tensor) -> None:
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            raise RuntimeError(f"strict field {name!r} must be a batched tensor")
        if int(value.shape[0]) != expected_rows:
            raise RuntimeError(
                f"strict field {name!r} has {value.shape[0]} rows, "
                f"expected {expected_rows}"
            )


def _message_count_values(
    data: TensorDict, field_name: str, *, expected_rows: int
) -> list[int]:
    values = squeeze_trailing_unit_dim(tensor_field(data, field_name))
    if (
        values.dtype not in _INTEGER_DTYPES
        or values.ndim != 1
        or int(values.shape[0]) != expected_rows
    ):
        raise RuntimeError(f"strict message count {field_name!r} has wrong shape")
    result: list[int] = []
    for item in values:
        value = int(item.item())
        if value < 0 or float(item.item()) != value:
            raise RuntimeError(
                f"strict message count {field_name!r} must be nonnegative integers"
            )
        result.append(value)
    return result


def _string_object_field(data: TensorDict, field_name: str) -> list[str]:
    value: Any = None
    for key, item in data.items(include_nested=False):
        if str(key) == field_name:
            value = item
            break
    if isinstance(value, NonTensorStack):
        items = value.tolist()
    elif isinstance(value, NonTensorData):
        wrapped = value.data
        if isinstance(wrapped, np.ndarray) and wrapped.dtype == object:
            items = wrapped.tolist()
        elif isinstance(wrapped, (list, tuple)):
            items = list(wrapped)
        else:
            items = [wrapped]
    elif isinstance(value, np.ndarray) and value.dtype == object:
        items = value.tolist()
    else:
        raise RuntimeError(f"strict object field {field_name!r} is unavailable")
    result: list[str] = []
    for item in items:
        decoded = unwrap_wire_stripped_payload(item)
        if not isinstance(decoded, str) or not decoded:
            raise RuntimeError(f"strict object field {field_name!r} is malformed")
        result.append(decoded)
    return result


def _exact_tag_int(tag: Mapping[str, Any], name: str) -> int:
    value = tag.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"strict row tag {name!r} must be a nonnegative int")
    return value


def _exact_tag_bool(tag: Mapping[str, Any], name: str) -> bool:
    value = tag.get(name)
    if type(value) is not bool:
        raise RuntimeError(f"strict row tag {name!r} must be an exact bool")
    return value


def _exact_tag_sha256(tag: Mapping[str, Any], name: str) -> str:
    value = tag.get(name)
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise RuntimeError(f"strict row tag {name!r} must be nonzero lowercase SHA-256")
    return value


def _validate_model_response_token_join(
    value: Any, *, row: Mapping[str, Any], row_index: int
) -> None:
    """Bind the raw folded model token transport to consumed tensors."""
    if not isinstance(value, Mapping) or "reward" in value:
        raise RuntimeError(
            f"strict transcript entry {row_index} model response is malformed"
        )
    try:
        prompt, completion, all_tokens = model_response_token_geometry(
            value, name=f"transcript entry {row_index} model_response"
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"strict transcript entry {row_index} has invalid token transport"
        ) from error
    if (
        prompt != row["prompt_token_ids"]
        or completion != row["completion_token_ids"]
        or all_tokens != row["token_ids"]
    ):
        raise RuntimeError(
            f"strict transcript entry {row_index} tokens differ from trainer input"
        )


def _exact_positive_tensor_int(value: torch.Tensor, *, name: str) -> int:
    if value.dtype not in _INTEGER_DTYPES:
        raise RuntimeError(f"strict {name} must have an integer tensor dtype")
    scalar = value.item()
    result = int(scalar)
    if isinstance(scalar, bool) or result <= 0 or float(scalar) != result:
        raise RuntimeError(f"strict {name} must be an exact positive integer")
    return result


def _int_tensor_list(value: torch.Tensor) -> list[int]:
    if value.dtype not in _INTEGER_DTYPES or value.ndim != 1:
        raise RuntimeError("strict token row must be one-dimensional")
    result = [int(item) for item in value.detach().cpu().tolist()]
    if len(result) > 131_072:
        raise RuntimeError("strict token row exceeds maximum length 131072")
    if any(item < 0 or item > 2_147_483_647 for item in result):
        raise RuntimeError("strict token IDs must fit the nonnegative int32 range")
    return result


def _float_tensor_list(value: torch.Tensor) -> list[float]:
    if not torch.is_floating_point(value) or value.ndim != 1:
        raise RuntimeError("strict float row must be one-dimensional")
    return [
        _finite_positive_zero(float(item)) for item in value.detach().cpu().tolist()
    ]


def _tensor_float(value: torch.Tensor) -> float:
    if not torch.is_floating_point(value):
        raise RuntimeError("strict reward/mask values must have floating tensor dtype")
    return _finite_positive_zero(float(value.item()))


def _float32_round_trip(value: float) -> float:
    """Round one native Gym reward exactly as the training TensorDict does."""
    rounded = struct.unpack(">f", struct.pack(">f", value))[0]
    return 0.0 if rounded == 0.0 else rounded


def _finite_positive_zero(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError("strict evidence cannot encode a non-finite tensor value")
    return 0.0 if value == 0.0 else value
