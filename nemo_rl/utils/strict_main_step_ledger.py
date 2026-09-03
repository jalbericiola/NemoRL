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

"""Fail-closed evidence for the first strict single-environment train step.

The strict shared-prefix A/B campaign needs one arm-local artifact containing
the exact rows consumed by optimizer step one.  Aggregate W&B metrics cannot
prove that the OFF and ON jobs trained on the same prompt/seed schedule, and a
digest printed by the job is not evidence unless its preimage is preserved.

This module owns that preimage.  It deliberately has no NeMo, Ray, Torch, or
W&B dependency: the SingleController converts the consumed tensors to plain
Python values and calls :func:`build_main_step1_ledger`, then publishes the
returned document with :func:`publish_main_step1_ledger`.  Replay/calibration
evidence is a separate artifact and must never call this publisher.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import stat
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MAIN_STEP1_LEDGER_SCHEMA = "nemo-rl-strict-main-step1-ledger-v5"
MAIN_STEP1_TRANSCRIPT_BUNDLE_SCHEMA = "nemo-rl-strict-step1-transcript-bundle-v4"
MAIN_STEP1_HASH_DOMAIN = "sha256-domain-nul-canonical-ascii-json-no-lf-v1"
MAIN_STEP1_EVIDENCE_DIRECTORY = "strict_pair_step1_evidence"
MAIN_STEP1_LEDGER_FILENAME = "main-ledger.json"

STRICT_PAIR_LAUNCH_MODE_ENV = "STRICT_PAIR_LAUNCH_MODE"
STRICT_PAIR_ARM_ENV = "STRICT_PAIR_ARM"

MAIN_STEP1_TAG_FIXTURE_ROW_INDEX = "strict_step1_fixture_row_index"
MAIN_STEP1_TAG_ROLLOUT_INDEX = "strict_step1_rollout_index"
MAIN_STEP1_TAG_GENERATION_SEED = "strict_step1_generation_seed"
MAIN_STEP1_TAG_TRANSCRIPT_BUNDLE_SHA256 = "strict_step1_transcript_bundle_sha256"
MAIN_STEP1_REWARD_PENALTY_TAGS = {
    "reasoning_equal_to_final_answer": (
        "strict_step1_penalty_reasoning_equal_to_final_answer"
    ),
    "empty_final_answer": "strict_step1_penalty_empty_final_answer",
    "unwanted_token": "strict_step1_penalty_unwanted_token",
    "malformed_think_tag": "strict_step1_penalty_malformed_think_tag",
}

_MAIN_STEP1_RUNTIME_BINDING_ENV = {
    "pair_manifest_sha256": "EXPECTED_PAIR_MANIFEST_SHA256",
    "submission_receipt_sha256": ("EXPECTED_STRICT_PAIR_SUBMISSION_RECEIPT_SHA256"),
    "job_id": "STRICT_PAIR_BOUND_JOB_ID",
    "run_id": "WANDB_RUN_ID",
    "restart_count": "STRICT_PAIR_BOUND_RESTART_COUNT",
    "fixture_sha256": "EXPECTED_STRICT_PAIR_FIXTURE_SHA256",
    "verifier_source_sha256": "EXPECTED_STRICT_PAIR_GYM_VERIFIER_SOURCE_SHA256",
    "snapshot_manifest_sha256": ("EXPECTED_STRICT_PREBUILT_SNAPSHOT_MANIFEST_SHA256"),
    "config_sha256": "EXPECTED_STRICT_PAIR_CONFIG_SHA256",
    "pair_campaign_sha256": "EXPECTED_STRICT_PAIR_CAMPAIGN_SHA256",
    "pair_campaign_reward_and_advantage_sha256": (
        "EXPECTED_STRICT_PAIR_REWARD_AND_ADVANTAGE_SHA256"
    ),
}

MAIN_STEP1_COMPARED_FIELDS = [
    "sample_index",
    "fixture_row_index",
    "rollout_index",
    "prompt_sha256",
    "request_sha256",
    "agent_run_request_sha256",
    "generation_seed",
    "token_ids",
    "input_length",
    "prompt_token_ids",
    "completion_token_ids",
    "token_loss_mask",
    "raw_environment_reward",
    "pre_penalty_environment_reward",
    "penalty_flags",
    "verifier_reward",
    "processed_reward",
    "sample_mask",
    "advantages",
    "valid_loss_tokens",
    "total_tokens",
]

MAIN_STEP1_ROOT_KEYS = frozenset(
    {
        "schema",
        "hash_domain",
        "pair_id",
        "environment",
        "arm",
        "mode",
        "step",
        "update_successful",
        "sample_count",
        "compared_fields",
        "generation",
        "bindings",
        "transcript_bundle",
        "rows",
        "step_totals",
        "cohort_sha256",
        "outputs_sha256",
        "rewards_sha256",
        "ordered_rows_sha256",
    }
)

MAIN_STEP1_BINDING_KEYS = frozenset(
    {
        "pair_manifest_sha256",
        "submission_receipt_sha256",
        "job_id",
        "run_id",
        "restart_count",
        "fixture_sha256",
        "verifier_source_sha256",
        "snapshot_manifest_sha256",
        "config_sha256",
        "pair_campaign_sha256",
        "pair_campaign_reward_and_advantage_sha256",
    }
)

MAIN_STEP1_GENERATION_KEYS = frozenset(
    {"seed_base", "max_new_tokens", "temperature", "top_k", "top_p"}
)

MAIN_STEP1_TRANSCRIPT_BUNDLE_REF_KEYS = frozenset({"path", "schema", "sha256"})

MAIN_STEP1_PENALTY_FLAG_KEYS = frozenset(
    {
        "reasoning_equal_to_final_answer",
        "empty_final_answer",
        "unwanted_token",
        "malformed_think_tag",
        "invalid_tool_call",
        "malformed_thinking",
        "raw_invalid_tool_call",
        "raw_malformed_thinking",
        "invalid_and_malformed",
    }
)

MAIN_STEP1_ROW_INPUT_KEYS = frozenset(
    {
        "sample_index",
        "sample_id",
        "shared_prefix_group_id",
        "fixture_row_index",
        "rollout_index",
        "generation_seed",
        "request_sha256",
        "response_sha256",
        "agent_run_request_sha256",
        "derived_verifier_request_sha256",
        "verifier_response_sha256",
        "token_ids",
        "input_length",
        "prompt_token_ids",
        "completion_token_ids",
        "token_loss_mask",
        "raw_environment_reward",
        "pre_penalty_environment_reward",
        "penalty_flags",
        "verifier_reward",
        "processed_reward",
        "sample_mask",
        "advantages",
        "valid_loss_tokens",
        "total_tokens",
    }
)

MAIN_STEP1_ROW_KEYS = MAIN_STEP1_ROW_INPUT_KEYS | frozenset(
    {"prompt_sha256", "row_sha256"}
)

MAIN_STEP1_TOTAL_KEYS = frozenset(
    {
        "raw_environment_reward_sum",
        "pre_penalty_environment_reward_sum",
        "verifier_reward_sum",
        "processed_reward_sum",
        "sample_mask_sum",
        "global_valid_toks",
        "total_num_tokens",
    }
)

_STRICT_ENVIRONMENTS = frozenset({"reasoning_gym", "citation", "freeform"})
_ARM_TO_MODE = {"off": "observe", "on": "train"}
_SHA256_HEX_LENGTH = 64
_MAX_SIGNED_INT64 = (1 << 63) - 1
_MAX_TOKEN_ID = (1 << 31) - 1
_MAX_SEQUENCE_LENGTH = 131_072
_HASH_PREFIX = b"nemo-rl-strict-v2"
_SAFE_PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_GIT_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}\Z")
_UUID4_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_ADVANTAGE_ABS_TOLERANCE = 2e-6
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_LEDGER_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def strict_main_step1_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the authenticated strict-submit runtime must emit a ledger.

    Both values are required so an unrelated run with a stray arm-like variable
    cannot activate a fail-closed campaign contract.  A partially configured
    strict submit raises instead of silently running without evidence.
    """
    values = os.environ if environ is None else environ
    launch_mode = values.get(STRICT_PAIR_LAUNCH_MODE_ENV)
    arm = values.get(STRICT_PAIR_ARM_ENV)
    if launch_mode == "submit":
        if arm not in _ARM_TO_MODE:
            raise RuntimeError(
                "strict pair submit requires STRICT_PAIR_ARM='off' or 'on' "
                "before NeMo runtime initialization"
            )
        return True
    if arm in _ARM_TO_MODE:
        raise RuntimeError(
            "STRICT_PAIR_ARM is set outside STRICT_PAIR_LAUNCH_MODE=submit; "
            "refusing ambiguous main-step evidence mode"
        )
    return False


def main_step1_runtime_contract(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load the wrapper-authenticated identity used by the runtime publisher.

    The wrapper exports these values only after independently validating the
    immutable Pair and submission receipt.  Missing or non-canonical values are
    fatal in strict-submit mode; the runtime never falls back to Slurm/W&B
    ambient variables whose contents were not bound by that wrapper.
    """
    values = os.environ if environ is None else environ
    if not strict_main_step1_enabled(values):
        raise RuntimeError("main step-one runtime contract is not active")

    required_environment_names = {
        "pair_id": "PAIR_ID",
        "environment": "STRICT_PAIR_ENVIRONMENT",
        "arm": STRICT_PAIR_ARM_ENV,
        "results_dir": "RESULTS_DIR",
        "fixture_path": "TRAIN_PATH",
        "model_transport_policy_sha256": (
            "EXPECTED_STRICT_PAIR_MODEL_TRANSPORT_POLICY_SHA256"
        ),
        "gym_gitlink_commit": "EXPECTED_GYM_GITLINK_COMMIT",
        "gym_tree": "EXPECTED_GYM_TREE",
    }
    resolved: dict[str, str] = {}
    for field_name, environment_name in required_environment_names.items():
        raw_value = values.get(environment_name)
        if raw_value is None:
            raise RuntimeError(
                f"strict main-step runtime is missing {environment_name}"
            )
        resolved[field_name] = raw_value

    raw_bindings: dict[str, str] = {}
    for field_name, environment_name in _MAIN_STEP1_RUNTIME_BINDING_ENV.items():
        raw_value = values.get(environment_name)
        if raw_value is None:
            raise RuntimeError(
                f"strict main-step runtime is missing {environment_name}"
            )
        raw_bindings[field_name] = raw_value

    restart_count_text = raw_bindings["restart_count"]
    if (
        not restart_count_text.isdecimal()
        or str(int(restart_count_text)) != restart_count_text
    ):
        raise ValueError(
            "STRICT_PAIR_BOUND_RESTART_COUNT must be a canonical nonnegative "
            "decimal string"
        )
    bindings: dict[str, Any] = dict(raw_bindings)
    bindings["restart_count"] = int(restart_count_text)
    binding_document = _validate_bindings(bindings)

    pair_id = resolved["pair_id"]
    environment = resolved["environment"]
    arm = resolved["arm"]
    _require_pair_id(pair_id, name="PAIR_ID")
    if environment not in _STRICT_ENVIRONMENTS:
        raise ValueError(
            "STRICT_PAIR_ENVIRONMENT must be one of "
            f"{sorted(_STRICT_ENVIRONMENTS)}, got {environment!r}"
        )
    if arm not in _ARM_TO_MODE:
        raise ValueError("STRICT_PAIR_ARM must be 'off' or 'on'")
    results_dir = str(
        _validate_canonical_absolute_path(resolved["results_dir"], name="RESULTS_DIR")
    )
    fixture_path = str(
        _validate_canonical_absolute_path(resolved["fixture_path"], name="TRAIN_PATH")
    )
    model_transport_policy_sha256 = _require_sha256(
        resolved["model_transport_policy_sha256"],
        name="EXPECTED_STRICT_PAIR_MODEL_TRANSPORT_POLICY_SHA256",
    )
    gym_gitlink_commit = _require_git_object_id(
        resolved["gym_gitlink_commit"], name="EXPECTED_GYM_GITLINK_COMMIT"
    )
    gym_tree = _require_git_object_id(resolved["gym_tree"], name="EXPECTED_GYM_TREE")
    return {
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "mode": _ARM_TO_MODE[arm],
        "results_dir": results_dir,
        "fixture_path": fixture_path,
        "model_transport_policy_sha256": model_transport_policy_sha256,
        "gym_gitlink_commit": gym_gitlink_commit,
        "gym_tree": gym_tree,
        "bindings": binding_document,
    }


def canonical_ascii_json(document: Any) -> bytes:
    """Serialize strict JSON as sorted compact ASCII with no trailing newline."""
    _reject_negative_zero(document, path="$")
    try:
        payload = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(
            "strict main-step ledger is not canonical ASCII JSON"
        ) from error
    if payload.endswith(b"\n"):
        raise AssertionError("canonical main-step ledger unexpectedly ended in LF")
    return payload


def domain_sha256(label: str, value: Any) -> str:
    """Hash one canonical value in the frozen strict-v2 NUL-separated domain."""
    _require_ascii_string(label, name="hash label", max_length=128)
    if "\x00" in label:
        raise ValueError("hash label must not contain NUL")
    preimage = _HASH_PREFIX + b"\x00" + label.encode("ascii") + b"\x00"
    return hashlib.sha256(preimage + canonical_ascii_json(value)).hexdigest()


def derive_nemo_gym_request_seed(
    *, seed_base: int, fixture_row_index: int, rollout_index: int
) -> int:
    """Recompute the exact signed-int63 request seed used by NeMo-Gym."""
    for name, value in (
        ("seed_base", seed_base),
        ("fixture_row_index", fixture_row_index),
        ("rollout_index", rollout_index),
    ):
        _require_nonnegative_int(value, name=name)
    identity = canonical_ascii_json([seed_base, fixture_row_index, rollout_index])
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def build_main_step1_ledger(
    *,
    pair_id: str,
    environment: str,
    arm: str,
    mode: str,
    generation: Mapping[str, Any],
    bindings: Mapping[str, Any],
    transcript_bundle: Mapping[str, Any],
    row_inputs: Sequence[Mapping[str, Any]],
    update_successful: bool,
) -> dict[str, Any]:
    """Build and fully validate the one allowed main step-one ledger."""
    if type(update_successful) is not bool or not update_successful:
        raise ValueError("main step-one ledger requires exact update_successful=true")
    _require_pair_id(pair_id, name="pair_id")
    if environment not in _STRICT_ENVIRONMENTS:
        raise ValueError(
            f"environment must be one of {sorted(_STRICT_ENVIRONMENTS)}, got {environment!r}"
        )
    if arm not in _ARM_TO_MODE:
        raise ValueError("arm must be 'off' or 'on'")
    if mode != _ARM_TO_MODE[arm]:
        raise ValueError(f"arm={arm!r} requires mode={_ARM_TO_MODE[arm]!r}")

    generation_document = _validate_generation(generation)
    binding_document = _validate_bindings(bindings)
    transcript_bundle_document = _validate_transcript_bundle_ref(transcript_bundle)
    if not isinstance(row_inputs, Sequence) or isinstance(
        row_inputs, (str, bytes, bytearray)
    ):
        raise TypeError("row_inputs must be a sequence of exactly four mappings")
    if len(row_inputs) != 4:
        raise ValueError(
            f"main step-one ledger requires exactly four rows, got {len(row_inputs)}"
        )

    rows: list[dict[str, Any]] = []
    for sample_index, raw_row in enumerate(row_inputs):
        rows.append(
            _build_row(
                raw_row,
                sample_index=sample_index,
                generation=generation_document,
                environment=environment,
            )
        )
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("main step-one ledger sample_id values must be unique")
    fixture_indices = {row["fixture_row_index"] for row in rows}
    if len(fixture_indices) != 1:
        raise ValueError(
            "strict one-prompt step must use one fixture_row_index across all rows"
        )
    group_ids = {row["shared_prefix_group_id"] for row in rows}
    if len(group_ids) != 1:
        raise ValueError(
            "strict one-prompt step must use one shared_prefix_group_id across K=4"
        )
    first_prompt = rows[0]["prompt_token_ids"]
    if any(row["prompt_token_ids"] != first_prompt for row in rows[1:]):
        raise ValueError(
            "strict one-prompt step must use identical prompt_token_ids across K=4"
        )
    for row in rows:
        if row["sample_mask"] != 1.0:
            raise ValueError("strict accepted step-one rows require sample_mask=1.0")
        if any(row["penalty_flags"].values()):
            raise ValueError(
                "strict accepted step-one rows require all penalties false"
            )
        reward = row["raw_environment_reward"]
        if any(
            row[name] != reward
            for name in (
                "pre_penalty_environment_reward",
                "verifier_reward",
                "processed_reward",
            )
        ):
            raise ValueError(
                "strict accepted step-one rows require raw, pre-penalty, "
                "verifier, and processed rewards to match rowwise"
            )
    expected_advantages = _expected_float32_rloo_advantages(
        [row["processed_reward"] for row in rows]
    )
    for sample_index, (row, expected_advantage) in enumerate(
        zip(rows, expected_advantages, strict=True)
    ):
        for token_index, actual_advantage in enumerate(row["advantages"]):
            if abs(actual_advantage - expected_advantage) > _ADVANTAGE_ABS_TOLERANCE:
                raise ValueError(
                    "strict row advantages differ from frozen float32 K4 RLOO: "
                    f"row={sample_index}, token={token_index}, "
                    f"expected={expected_advantage}, actual={actual_advantage}"
                )

    cohort_projection = [
        {
            "sample_index": row["sample_index"],
            "fixture_row_index": row["fixture_row_index"],
            "rollout_index": row["rollout_index"],
            "prompt_sha256": row["prompt_sha256"],
            "request_sha256": row["request_sha256"],
            "generation_seed": row["generation_seed"],
            "prompt_token_ids": row["prompt_token_ids"],
        }
        for row in rows
    ]
    outputs_projection = [
        {
            **cohort,
            "response_sha256": row["response_sha256"],
            "agent_run_request_sha256": row["agent_run_request_sha256"],
            "derived_verifier_request_sha256": row["derived_verifier_request_sha256"],
            "verifier_response_sha256": row["verifier_response_sha256"],
            "token_ids": row["token_ids"],
            "input_length": row["input_length"],
            "completion_token_ids": row["completion_token_ids"],
            "token_loss_mask": row["token_loss_mask"],
            "valid_loss_tokens": row["valid_loss_tokens"],
            "total_tokens": row["total_tokens"],
        }
        for cohort, row in zip(cohort_projection, rows, strict=True)
    ]
    rewards_projection = [
        {
            "sample_index": row["sample_index"],
            "raw_environment_reward": row["raw_environment_reward"],
            "pre_penalty_environment_reward": row["pre_penalty_environment_reward"],
            "penalty_flags": row["penalty_flags"],
            "verifier_reward": row["verifier_reward"],
            "processed_reward": row["processed_reward"],
            "sample_mask": row["sample_mask"],
            "advantages": row["advantages"],
        }
        for row in rows
    ]
    step_totals = {
        "raw_environment_reward_sum": sum(
            row["raw_environment_reward"] for row in rows
        ),
        "pre_penalty_environment_reward_sum": sum(
            row["pre_penalty_environment_reward"] for row in rows
        ),
        "verifier_reward_sum": sum(row["verifier_reward"] for row in rows),
        "processed_reward_sum": sum(row["processed_reward"] for row in rows),
        "sample_mask_sum": sum(row["sample_mask"] == 1.0 for row in rows),
        "global_valid_toks": sum(row["valid_loss_tokens"] for row in rows),
        "total_num_tokens": sum(row["total_tokens"] for row in rows),
    }
    document = {
        "schema": MAIN_STEP1_LEDGER_SCHEMA,
        "hash_domain": MAIN_STEP1_HASH_DOMAIN,
        "pair_id": pair_id,
        "environment": environment,
        "arm": arm,
        "mode": mode,
        "step": 1,
        "update_successful": True,
        "sample_count": len(rows),
        "compared_fields": list(MAIN_STEP1_COMPARED_FIELDS),
        "generation": generation_document,
        "bindings": binding_document,
        "transcript_bundle": transcript_bundle_document,
        "rows": rows,
        "step_totals": step_totals,
        "cohort_sha256": domain_sha256("step1-cohort", cohort_projection),
        "outputs_sha256": domain_sha256("step1-outputs", outputs_projection),
        "rewards_sha256": domain_sha256("step1-rewards", rewards_projection),
        "ordered_rows_sha256": domain_sha256("step1-ordered-rows", rows),
    }
    if set(document) != MAIN_STEP1_ROOT_KEYS:
        raise AssertionError("internal main-step ledger root keyset drift")
    canonical_ascii_json(document)
    return document


def validate_main_step1_ledger(document: Mapping[str, Any]) -> None:
    """Rebuild a ledger and reject any non-derived or malformed field."""
    _require_exact_keys(document, MAIN_STEP1_ROOT_KEYS, name="ledger")
    if document["schema"] != MAIN_STEP1_LEDGER_SCHEMA:
        raise ValueError("main-step ledger schema mismatch")
    if document["hash_domain"] != MAIN_STEP1_HASH_DOMAIN:
        raise ValueError("main-step ledger hash_domain mismatch")
    if type(document["step"]) is not int or document["step"] != 1:
        raise ValueError("main-step ledger step must be integer 1")
    if (
        type(document["update_successful"]) is not bool
        or not document["update_successful"]
    ):
        raise ValueError("main-step ledger update_successful must be exact true")
    if type(document["sample_count"]) is not int or document["sample_count"] != 4:
        raise ValueError("main-step ledger sample_count must be integer 4")
    if document["compared_fields"] != MAIN_STEP1_COMPARED_FIELDS:
        raise ValueError("main-step ledger compared_fields mismatch")
    rows = document["rows"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("main-step ledger rows must be a four-element list")
    for index, row in enumerate(rows):
        _require_exact_keys(row, MAIN_STEP1_ROW_KEYS, name=f"ledger.rows[{index}]")
    rebuilt = build_main_step1_ledger(
        pair_id=document["pair_id"],
        environment=document["environment"],
        arm=document["arm"],
        mode=document["mode"],
        generation=document["generation"],
        bindings=document["bindings"],
        transcript_bundle=document["transcript_bundle"],
        row_inputs=[
            {key: row[key] for key in MAIN_STEP1_ROW_INPUT_KEYS} for row in rows
        ],
        update_successful=document["update_successful"],
    )
    if dict(document) != rebuilt:
        raise ValueError("main-step ledger contains a non-derived or changed value")
    canonical_ascii_json(document)


def main_step1_ledger_path(results_dir: str) -> Path:
    """Return the only accepted absolute ledger path below ``RESULTS_DIR``."""
    results = _validate_canonical_absolute_path(results_dir, name="RESULTS_DIR")
    if results == Path("/"):
        raise ValueError("RESULTS_DIR must not be root")
    return results / MAIN_STEP1_EVIDENCE_DIRECTORY / MAIN_STEP1_LEDGER_FILENAME


def publish_main_step1_ledger(
    *, results_dir: str, document: Mapping[str, Any]
) -> tuple[Path, str]:
    """Exclusively publish a mode-0400, single-link canonical ledger inode."""
    validate_main_step1_ledger(document)
    payload = canonical_ascii_json(document)
    digest = hashlib.sha256(payload).hexdigest()
    results_path = _validate_canonical_absolute_path(results_dir, name="RESULTS_DIR")
    ledger_path = main_step1_ledger_path(results_dir)

    results_fd = _open_absolute_directory_without_symlinks(results_path)
    evidence_fd: int | None = None
    candidate_fd: int | None = None
    candidate_created = False
    candidate_name = f".{MAIN_STEP1_LEDGER_FILENAME}.candidate"
    try:
        results_stat = os.fstat(results_fd)
        _require_owned_directory(results_stat, name="RESULTS_DIR")
        try:
            os.mkdir(MAIN_STEP1_EVIDENCE_DIRECTORY, 0o700, dir_fd=results_fd)
            os.fsync(results_fd)
        except FileExistsError:
            pass
        evidence_fd = os.open(
            MAIN_STEP1_EVIDENCE_DIRECTORY,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=results_fd,
        )
        _require_owned_directory(os.fstat(evidence_fd), name="step1 evidence directory")

        candidate_fd = os.open(
            candidate_name,
            _LEDGER_OPEN_FLAGS,
            stat.S_IRUSR,
            dir_fd=evidence_fd,
        )
        candidate_created = True
        _write_all(candidate_fd, payload)
        os.fchmod(candidate_fd, stat.S_IRUSR)
        os.fsync(candidate_fd)
        candidate_stat = os.fstat(candidate_fd)
        if (
            not stat.S_ISREG(candidate_stat.st_mode)
            or stat.S_IMODE(candidate_stat.st_mode) != 0o400
            or candidate_stat.st_uid != os.geteuid()
            or candidate_stat.st_nlink != 1
            or candidate_stat.st_size != len(payload)
        ):
            raise RuntimeError("main-step ledger candidate inode validation failed")
        os.close(candidate_fd)
        candidate_fd = None

        try:
            os.link(
                candidate_name,
                MAIN_STEP1_LEDGER_FILENAME,
                src_dir_fd=evidence_fd,
                dst_dir_fd=evidence_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"main step-one ledger already exists: {ledger_path}"
            ) from error
        os.unlink(candidate_name, dir_fd=evidence_fd)
        candidate_created = False
        os.fsync(evidence_fd)

        ledger_fd = os.open(
            MAIN_STEP1_LEDGER_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=evidence_fd,
        )
        try:
            ledger_stat = os.fstat(ledger_fd)
            actual = _read_all(ledger_fd)
        finally:
            os.close(ledger_fd)
        if (
            not stat.S_ISREG(ledger_stat.st_mode)
            or stat.S_IMODE(ledger_stat.st_mode) != 0o400
            or ledger_stat.st_uid != os.geteuid()
            or ledger_stat.st_nlink != 1
            or actual != payload
            or hashlib.sha256(actual).hexdigest() != digest
        ):
            raise RuntimeError(
                "published main step-one ledger failed exact verification"
            )
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)
        if candidate_created and evidence_fd is not None:
            try:
                os.unlink(candidate_name, dir_fd=evidence_fd)
            except FileNotFoundError:
                pass
        if evidence_fd is not None:
            os.close(evidence_fd)
        os.close(results_fd)
    return ledger_path, digest


def _build_row(
    raw_row: Mapping[str, Any],
    *,
    sample_index: int,
    generation: Mapping[str, Any],
    environment: str,
) -> dict[str, Any]:
    _require_exact_keys(raw_row, MAIN_STEP1_ROW_INPUT_KEYS, name=f"row[{sample_index}]")
    if (
        type(raw_row["sample_index"]) is not int
        or raw_row["sample_index"] != sample_index
    ):
        raise ValueError(f"row[{sample_index}].sample_index must equal its list index")
    sample_id = raw_row["sample_id"]
    _require_ascii_string(
        sample_id,
        name=f"row[{sample_index}].sample_id",
        max_length=256,
    )
    shared_prefix_group_id = raw_row["shared_prefix_group_id"]
    if (
        type(shared_prefix_group_id) is not str
        or _UUID4_RE.fullmatch(shared_prefix_group_id) is None
    ):
        raise ValueError(
            f"row[{sample_index}].shared_prefix_group_id must be a canonical UUID4"
        )
    expected_sample_id = f"{shared_prefix_group_id}_g{sample_index}"
    if sample_id != expected_sample_id:
        raise ValueError(
            f"row[{sample_index}].sample_id must equal {expected_sample_id!r}"
        )

    fixture_row_index = raw_row["fixture_row_index"]
    rollout_index = raw_row["rollout_index"]
    generation_seed = raw_row["generation_seed"]
    _require_nonnegative_int(fixture_row_index, name="fixture_row_index")
    if fixture_row_index != 0:
        raise ValueError(
            "strict no-shuffle step one must consume fixture_row_index exactly 0"
        )
    if type(rollout_index) is not int or rollout_index != sample_index:
        raise ValueError(f"row[{sample_index}].rollout_index must equal sample_index")
    expected_seed = derive_nemo_gym_request_seed(
        seed_base=generation["seed_base"],
        fixture_row_index=fixture_row_index,
        rollout_index=rollout_index,
    )
    if type(generation_seed) is not int or generation_seed != expected_seed:
        raise ValueError(
            f"row[{sample_index}].generation_seed differs from the exact request seed"
        )

    input_length = raw_row["input_length"]
    total_tokens = raw_row["total_tokens"]
    valid_loss_tokens = raw_row["valid_loss_tokens"]
    _require_positive_int(input_length, name="input_length")
    _require_positive_int(total_tokens, name="total_tokens")
    _require_positive_int(valid_loss_tokens, name="valid_loss_tokens")
    if any(
        value > _MAX_SEQUENCE_LENGTH
        for value in (input_length, total_tokens, valid_loss_tokens)
    ):
        raise ValueError(
            "input_length, total_tokens, and valid_loss_tokens must be <= 131072"
        )
    if total_tokens != input_length:
        raise ValueError("row total_tokens must equal input_length")

    token_ids = _validate_token_ids(
        raw_row["token_ids"], name="token_ids", max_length=input_length
    )
    if len(token_ids) != input_length:
        raise ValueError("row token_ids length must equal input_length")
    prompt_token_ids = _validate_token_ids(
        raw_row["prompt_token_ids"],
        name="prompt_token_ids",
        max_length=input_length,
    )
    completion_token_ids = _validate_token_ids(
        raw_row["completion_token_ids"],
        name="completion_token_ids",
        max_length=generation["max_new_tokens"],
    )
    if token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise ValueError("prompt_token_ids must be the exact prefix of token_ids")
    if token_ids[len(prompt_token_ids) :] != completion_token_ids:
        raise ValueError(
            "completion_token_ids must be the contiguous suffix after prompt_token_ids"
        )
    token_loss_mask = _validate_binary_float_list(
        raw_row["token_loss_mask"],
        name="token_loss_mask",
        expected_length=input_length,
    )
    advantages = _validate_finite_float_list(
        raw_row["advantages"],
        name="advantages",
        expected_length=input_length,
    )
    if any(
        mask_value != 0.0 for mask_value in token_loss_mask[: len(prompt_token_ids)]
    ):
        raise ValueError(
            "token_loss_mask must be zero throughout the immutable prompt prefix"
        )
    if valid_loss_tokens != token_loss_mask.count(1.0):
        raise ValueError(
            "valid_loss_tokens must equal the number of ones in token_loss_mask"
        )

    raw_reward = _require_exact_float32(
        raw_row["raw_environment_reward"], name="raw_environment_reward"
    )
    if environment == "reasoning_gym":
        if not 0.0 <= raw_reward <= 1.0:
            raise ValueError(
                "reasoning_gym raw_environment_reward must be an exact float in [0,1]"
            )
    elif raw_reward not in (0.0, 1.0):
        raise ValueError(
            "citation/freeform raw_environment_reward must be exact float 0.0 or 1.0"
        )
    pre_penalty_reward = _require_exact_float32(
        raw_row["pre_penalty_environment_reward"],
        name="pre_penalty_environment_reward",
    )
    verifier_reward = _require_exact_float32(
        raw_row["verifier_reward"], name="verifier_reward"
    )
    processed_reward = _require_exact_float32(
        raw_row["processed_reward"], name="processed_reward"
    )
    sample_mask = _require_exact_float(raw_row["sample_mask"], name="sample_mask")
    if sample_mask not in (0.0, 1.0):
        raise ValueError("sample_mask must be exact float 0.0 or 1.0")
    penalty_flags = raw_row["penalty_flags"]
    _require_exact_keys(
        penalty_flags,
        MAIN_STEP1_PENALTY_FLAG_KEYS,
        name="penalty_flags",
    )
    if any(type(value) is not bool for value in penalty_flags.values()):
        raise TypeError("every penalty_flags value must be an exact bool")

    prompt_sha256 = domain_sha256("step1-prompt", prompt_token_ids)
    request_sha256 = _require_sha256(raw_row["request_sha256"], name="request_sha256")
    response_sha256 = _require_sha256(
        raw_row["response_sha256"], name="response_sha256"
    )
    agent_run_request_sha256 = _require_sha256(
        raw_row["agent_run_request_sha256"], name="agent_run_request_sha256"
    )
    derived_verifier_request_sha256 = _require_sha256(
        raw_row["derived_verifier_request_sha256"],
        name="derived_verifier_request_sha256",
    )
    verifier_response_sha256 = _require_sha256(
        raw_row["verifier_response_sha256"], name="verifier_response_sha256"
    )
    row = {
        "sample_index": sample_index,
        "sample_id": sample_id,
        "shared_prefix_group_id": shared_prefix_group_id,
        "fixture_row_index": fixture_row_index,
        "rollout_index": rollout_index,
        "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "agent_run_request_sha256": agent_run_request_sha256,
        "derived_verifier_request_sha256": derived_verifier_request_sha256,
        "verifier_response_sha256": verifier_response_sha256,
        "generation_seed": generation_seed,
        "token_ids": token_ids,
        "input_length": input_length,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "token_loss_mask": token_loss_mask,
        "raw_environment_reward": raw_reward,
        "pre_penalty_environment_reward": pre_penalty_reward,
        "penalty_flags": dict(penalty_flags),
        "verifier_reward": verifier_reward,
        "processed_reward": processed_reward,
        "sample_mask": sample_mask,
        "advantages": advantages,
        "valid_loss_tokens": valid_loss_tokens,
        "total_tokens": total_tokens,
    }
    row["row_sha256"] = domain_sha256("step1-row", row)
    if set(row) != MAIN_STEP1_ROW_KEYS:
        raise AssertionError("internal main-step row keyset drift")
    return row


def _validate_generation(generation: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(generation, MAIN_STEP1_GENERATION_KEYS, name="generation")
    seed_base = generation["seed_base"]
    max_new_tokens = generation["max_new_tokens"]
    _require_nonnegative_int(seed_base, name="generation.seed_base")
    _require_positive_int(max_new_tokens, name="generation.max_new_tokens")
    temperature = _require_exact_float(
        generation["temperature"], name="generation.temperature"
    )
    top_p = _require_exact_float(generation["top_p"], name="generation.top_p")
    top_k = generation["top_k"]
    if top_k is not None:
        _require_nonnegative_int(top_k, name="generation.top_k")
    if temperature < 0.0:
        raise ValueError("generation.temperature must be nonnegative")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("generation.top_p must be in [0, 1]")
    document = {
        "seed_base": seed_base,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }
    if document != {
        "seed_base": 42,
        "max_new_tokens": 768,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
    }:
        raise ValueError("strict main-step generation policy differs from campaign")
    return document


def _validate_transcript_bundle_ref(value: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_keys(
        value,
        MAIN_STEP1_TRANSCRIPT_BUNDLE_REF_KEYS,
        name="transcript_bundle",
    )
    path = _validate_canonical_absolute_path(
        value["path"], name="transcript_bundle.path"
    )
    if (
        path.name != "transcript-bundle.json"
        or path.parent.name != MAIN_STEP1_EVIDENCE_DIRECTORY
    ):
        raise ValueError(
            "transcript_bundle.path must end in "
            "strict_pair_step1_evidence/transcript-bundle.json"
        )
    if value["schema"] != MAIN_STEP1_TRANSCRIPT_BUNDLE_SCHEMA:
        raise ValueError("transcript_bundle.schema mismatch")
    digest = _require_sha256(value["sha256"], name="transcript_bundle.sha256")
    return {"path": str(path), "schema": value["schema"], "sha256": digest}


def _validate_bindings(bindings: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(bindings, MAIN_STEP1_BINDING_KEYS, name="bindings")
    document = dict(bindings)
    for name in MAIN_STEP1_BINDING_KEYS - {"job_id", "restart_count"}:
        _require_sha256(document[name], name=f"bindings.{name}")
    job_id = document["job_id"]
    _require_ascii_string(job_id, name="bindings.job_id", max_length=32)
    if (
        not job_id.isdecimal()
        or str(int(job_id)) != job_id
        or not 0 < int(job_id) <= _MAX_SIGNED_INT64
    ):
        raise ValueError("bindings.job_id must be a canonical positive decimal string")
    restart_count = document["restart_count"]
    _require_nonnegative_int(restart_count, name="bindings.restart_count")
    if restart_count != 0:
        raise ValueError("strict main-step acceptance requires restart_count=0")
    return document


def _validate_token_ids(value: Any, *, name: str, max_length: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{name} must be a nonempty list")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length {max_length}")
    result: list[int] = []
    for index, token_id in enumerate(value):
        _require_nonnegative_int(token_id, name=f"{name}[{index}]")
        if token_id > _MAX_TOKEN_ID:
            raise ValueError(f"{name}[{index}] exceeds signed int32 token range")
        result.append(token_id)
    return result


def _validate_binary_float_list(
    value: Any, *, name: str, expected_length: int
) -> list[float]:
    values = _validate_finite_float_list(
        value, name=name, expected_length=expected_length
    )
    if any(item not in (0.0, 1.0) for item in values):
        raise ValueError(f"{name} must contain only exact float 0.0 or 1.0")
    return values


def _validate_finite_float_list(
    value: Any, *, name: str, expected_length: int
) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise TypeError(f"{name} must be a list of length {expected_length}")
    return [
        _require_exact_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _float32(value: float) -> float:
    """Round exactly once to the IEEE-754 binary32 value used by Torch."""
    rounded = struct.unpack(">f", struct.pack(">f", float(value)))[0]
    return 0.0 if rounded == 0.0 else rounded


def _expected_float32_rloo_advantages(rewards: Sequence[float]) -> list[float]:
    """Recompute the frozen K=4 leave-one-out GRPO scalar per response."""
    if len(rewards) != 4:
        raise ValueError("strict RLOO advantage closure requires exactly K=4 rewards")
    rounded_rewards = [_float32(reward) for reward in rewards]
    expected: list[float] = []
    for sample_index, reward in enumerate(rounded_rewards):
        peers = [
            peer
            for peer_index, peer in enumerate(rounded_rewards)
            if peer_index != sample_index
        ]
        peer_sum = _float32(_float32(peers[0] + peers[1]) + peers[2])
        mean = _float32(peer_sum / 3.0)
        squares = [_float32(peer * peer) for peer in peers]
        square_sum = _float32(_float32(squares[0] + squares[1]) + squares[2])
        square_mean = _float32(square_sum / 3.0)
        variance = _float32(_float32(square_mean - _float32(mean * mean)) * 1.5)
        std = _float32(math.sqrt(max(variance, 0.0)))
        delta = _float32(reward - mean)
        advantage = _float32(delta / _float32(std + 1e-6)) if std > 0.0 else delta
        expected.append(_float32(min(20.0, max(-20.0, advantage))))
    return expected


def _require_exact_keys(value: Any, expected: frozenset[str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keyset mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_pair_id(value: Any, *, name: str) -> str:
    if type(value) is not str or _SAFE_PAIR_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}")
    return value


def _require_ascii_string(value: Any, *, name: str, max_length: int) -> str:
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or len(value) > max_length
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError(
            f"{name} must be nonempty printable ASCII of at most {max_length} characters"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * _SHA256_HEX_LENGTH
    ):
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256 hex digest")
    return value


def _require_git_object_id(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or _GIT_OBJECT_ID_RE.fullmatch(value) is None
        or value == "0" * 40
    ):
        raise ValueError(f"{name} must be a nonzero lowercase 40-hex Git object ID")
    return value


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be an exact nonnegative int")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypeError(f"{name} must be an exact positive int")
    return value


def _require_exact_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be an exact finite float")
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise ValueError(f"{name} must not be negative zero")
    return value


def _require_exact_float32(value: Any, *, name: str) -> float:
    result = _require_exact_float(value, name=name)
    try:
        rounded = _float32(result)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{name} is outside the finite IEEE float32 range") from error
    if rounded != result:
        raise ValueError(f"{name} must be an exact IEEE float32 round-trip value")
    return result


def _reject_negative_zero(value: Any, *, path: str) -> None:
    if type(value) is float:
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(f"negative zero is forbidden at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"canonical JSON object key at {path} is not a string")
            _reject_negative_zero(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_negative_zero(item, path=f"{path}[{index}]")


def _validate_canonical_absolute_path(value: str, *, name: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty canonical absolute path")
    path = Path(value)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or posixpath.normpath(value) != value
        or not path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
    ):
        raise ValueError(f"{name} must be a canonical absolute path, got {value!r}")
    return path


def _open_absolute_directory_without_symlinks(path: Path) -> int:
    directory_fd = os.open("/", _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
    except OSError:
        os.close(directory_fd)
        raise
    return directory_fd


def _require_owned_directory(metadata: os.stat_result, *, name: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError(f"{name} must be an EUID-owned mode-0700 directory")


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("main-step ledger write made no progress")
        offset += written


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
