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

"""Closed result-inventory profiles for strict captured replay.

Profiles are selected only by an exact caller-supplied environment/profile pair.
They are immutable code data: result contents cannot add a profile, choose a
profile, or extend a profile's admitted filesystem inventory.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMAT_RESULT_DIRECTORIES = (".", "strict_gym_child_runtime")
FORMAT_RESULT_FILES = (
    "evidence-index.json",
    "model-transport-replay-consumption.json",
    "replay-ledger.json",
    "strict_gym_child_runtime/index.json",
    "strict_gym_child_runtime/format-verification-call-00000001.json",
    "strict_gym_child_runtime/format-verification-call-00000002.json",
    "strict_gym_child_runtime/format-verification-call-00000003.json",
    "strict_gym_child_runtime/format-verification-call-00000004.json",
    "strict_gym_child_runtime/format-verification-call-index.json",
    "strict_gym_child_runtime/format-verification-closed.json",
    "strict_gym_child_runtime/resource.json",
    "strict_gym_child_runtime/spec.json",
    "transcript-bundle.json",
)
FORMAT_RESULT_FILE_SCHEMAS = (
    "nemo-rl-strict-captured-replay-evidence-index-v4",
    "nemo-rl-strict-model-transport-replay-consumption-v3",
    "nemo-rl-strict-captured-replay-step1-ledger-v5",
    "nemo-rl-strict-gym-child-index-v1",
    "nemo-rl-strict-format-verification-call-v1",
    "nemo-rl-strict-format-verification-call-v1",
    "nemo-rl-strict-format-verification-call-v1",
    "nemo-rl-strict-format-verification-call-v1",
    "nemo-rl-strict-format-verification-call-index-v1",
    "nemo-rl-strict-format-verification-closed-v1",
    "nemo-rl-strict-gym-child-receipt-v1",
    "nemo-rl-strict-gym-child-spec-v1",
    "nemo-rl-strict-step1-transcript-bundle-v4",
)
FORMAT_RESULT_ANCHOR_PATHS = frozenset(
    {
        "evidence-index.json",
        "model-transport-replay-consumption.json",
        "replay-ledger.json",
        "strict_gym_child_runtime/format-verification-call-index.json",
        "transcript-bundle.json",
    }
)
FORMAT_SCORER_TERMINAL_INDEX_PATH = (
    "strict_gym_child_runtime/format-verification-call-index.json"
)

REASONING_RESULT_DIRECTORIES = (".", "strict_gym_child_runtime")
REASONING_RESULT_FILES = (
    "evidence-index.json",
    "model-transport-replay-consumption.json",
    "replay-ledger.json",
    "strict_gym_child_runtime/index.json",
    "strict_gym_child_runtime/reasoning-score-call-00000001.json",
    "strict_gym_child_runtime/reasoning-score-call-00000002.json",
    "strict_gym_child_runtime/reasoning-score-call-00000003.json",
    "strict_gym_child_runtime/reasoning-score-call-00000004.json",
    "strict_gym_child_runtime/reasoning-score-call-index.json",
    "strict_gym_child_runtime/reasoning-score-closed.json",
    "strict_gym_child_runtime/resource.json",
    "strict_gym_child_runtime/spec.json",
    "transcript-bundle.json",
)
REASONING_RESULT_FILE_SCHEMAS = (
    "nemo-rl-strict-captured-replay-evidence-index-v4",
    "nemo-rl-strict-model-transport-replay-consumption-v3",
    "nemo-rl-strict-captured-replay-step1-ledger-v5",
    "nemo-rl-strict-gym-child-index-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-v1",
    "nemo-rl-strict-reasoning-score-call-index-v1",
    "nemo-rl-strict-reasoning-score-closed-v1",
    "nemo-rl-strict-gym-child-receipt-v1",
    "nemo-rl-strict-gym-child-spec-v1",
    "nemo-rl-strict-step1-transcript-bundle-v4",
)
REASONING_RESULT_ANCHOR_PATHS = frozenset(
    {
        "evidence-index.json",
        "model-transport-replay-consumption.json",
        "replay-ledger.json",
        "strict_gym_child_runtime/reasoning-score-call-index.json",
        "transcript-bundle.json",
    }
)
REASONING_SCORER_TERMINAL_INDEX_PATH = (
    "strict_gym_child_runtime/reasoning-score-call-index.json"
)


@dataclass(frozen=True, slots=True)
class StrictCapturedReplayProfile:
    """One immutable, code-admitted terminal result profile."""

    environment: str
    profile_id: str
    verifier_type: str
    method: str
    resource_config_path_name: str
    disabled_config_path_name: str
    resource_app_path: str
    resource_app_sha256: str
    resource_config_path: str
    resource_config_sha256: str
    scorer_config_path_name: str
    scorer_config_path: str
    scorer_config_sha256: str
    requirements_path: str
    requirements_sha256: str
    fixture_path: str
    fixture_sha256: str
    fixture_rows: int
    call_schema: str
    closed_schema: str
    call_index_schema: str
    result_directories: tuple[str, ...]
    result_files: tuple[str, ...]
    result_file_schemas: tuple[str, ...]
    result_anchor_paths: frozenset[str]
    scorer_terminal_index_path: str


STRICT_CAPTURED_REPLAY_PROFILES = (
    StrictCapturedReplayProfile(
        environment="citation",
        profile_id="citation-string-match-v1",
        verifier_type="string_match",
        method="_verify_string_match",
        resource_config_path_name="citation_format",
        disabled_config_path_name="citation_format_simple_agent",
        resource_app_path="resources_servers/format_verification/app.py",
        resource_app_sha256=(
            "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
        ),
        resource_config_path=(
            "resources_servers/format_verification/configs/citation_format.yaml"
        ),
        resource_config_sha256=(
            "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
        ),
        scorer_config_path_name="citation_format",
        scorer_config_path=(
            "resources_servers/format_verification/configs/citation_format.yaml"
        ),
        scorer_config_sha256=(
            "da549a29c31219d8eeb14ea23f888c05479578c61619f684387efd97fadb0796"
        ),
        requirements_path="resources_servers/format_verification/requirements.txt",
        requirements_sha256=(
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        fixture_path="tests/unit/tools/data/citation_example.jsonl",
        fixture_sha256=(
            "d5b56a41c5e8a220d196c58727b87648d86384550f7a04b5a5d2f224e17213cc"
        ),
        fixture_rows=5,
        call_schema="nemo-rl-strict-format-verification-call-v1",
        closed_schema="nemo-rl-strict-format-verification-closed-v1",
        call_index_schema="nemo-rl-strict-format-verification-call-index-v1",
        result_directories=FORMAT_RESULT_DIRECTORIES,
        result_files=FORMAT_RESULT_FILES,
        result_file_schemas=FORMAT_RESULT_FILE_SCHEMAS,
        result_anchor_paths=FORMAT_RESULT_ANCHOR_PATHS,
        scorer_terminal_index_path=FORMAT_SCORER_TERMINAL_INDEX_PATH,
    ),
    StrictCapturedReplayProfile(
        environment="freeform",
        profile_id="freeform-regex-v1",
        verifier_type="regex",
        method="_verify_regex",
        resource_config_path_name="freeform_formatting",
        disabled_config_path_name="freeform_formatting_simple_agent",
        resource_app_path="resources_servers/format_verification/app.py",
        resource_app_sha256=(
            "6e0a4bd8eae96b073598f12b68805772644a0c60e5a5ef61f09ad12ea8761f36"
        ),
        resource_config_path=(
            "resources_servers/format_verification/configs/freeform_formatting.yaml"
        ),
        resource_config_sha256=(
            "92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa"
        ),
        scorer_config_path_name="freeform_formatting",
        scorer_config_path=(
            "resources_servers/format_verification/configs/freeform_formatting.yaml"
        ),
        scorer_config_sha256=(
            "92a38a70b922f9dcd837a7336c8ce5b13588cb3c1a85d05270486601d18ba6aa"
        ),
        requirements_path="resources_servers/format_verification/requirements.txt",
        requirements_sha256=(
            "18e0d5e99020599c4d033912b39d4569276b1b9278db73469ea9708742cfaa7d"
        ),
        fixture_path="tests/unit/tools/data/freeform_example.jsonl",
        fixture_sha256=(
            "8869b42f6a946833c1ca3a37316907fd3d621e460a3288ed309f1ca52ca67399"
        ),
        fixture_rows=5,
        call_schema="nemo-rl-strict-format-verification-call-v1",
        closed_schema="nemo-rl-strict-format-verification-closed-v1",
        call_index_schema="nemo-rl-strict-format-verification-call-index-v1",
        result_directories=FORMAT_RESULT_DIRECTORIES,
        result_files=FORMAT_RESULT_FILES,
        result_file_schemas=FORMAT_RESULT_FILE_SCHEMAS,
        result_anchor_paths=FORMAT_RESULT_ANCHOR_PATHS,
        scorer_terminal_index_path=FORMAT_SCORER_TERMINAL_INDEX_PATH,
    ),
    StrictCapturedReplayProfile(
        environment="reasoning_gym",
        profile_id="reasoning-gym-exact-match-v1",
        verifier_type="score_answer",
        method="KnightsKnavesDataset.score_answer",
        resource_config_path_name="reasoning_gym",
        disabled_config_path_name="reasoning_gym_simple_agent",
        resource_app_path="resources_servers/reasoning_gym/app.py",
        resource_app_sha256=(
            "3a35c5d27392dae05499ceefac04e9c32ad963b51a54d77bb470ee59b1fe3127"
        ),
        resource_config_path=(
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml"
        ),
        resource_config_sha256=(
            "bdbb459a4a920bc47cf84b1d7dc30aeaa9be35cf0dfac09c77879e45b62a52ab"
        ),
        # ``scorer_config_path_name`` is the resolved RunHelper target key,
        # not the selected YAML filename stem.  The resource-only YAML still
        # resolves the admitted scorer under ``reasoning_gym``.
        scorer_config_path_name="reasoning_gym",
        scorer_config_path=(
            "resources_servers/reasoning_gym/configs/resources_only.yaml"
        ),
        scorer_config_sha256=(
            "e11a3084f050e4c24101550f63efe71ac6c10f3bc125489ba7293cd81778de68"
        ),
        requirements_path="resources_servers/reasoning_gym/requirements.txt",
        requirements_sha256=(
            "b00b45db433d797d8a5c5c5602f24ab94d9d5620d83b4bef21fbee851287d411"
        ),
        fixture_path="tests/unit/tools/data/reasoning_gym_example.jsonl",
        fixture_sha256=(
            "da8ebd2b43d002ba9a6946fe458db7df8bf7e1b3068be3e2f9f014bfdd5229ce"
        ),
        fixture_rows=5,
        call_schema="nemo-rl-strict-reasoning-score-call-v1",
        closed_schema="nemo-rl-strict-reasoning-score-closed-v1",
        call_index_schema="nemo-rl-strict-reasoning-score-call-index-v1",
        result_directories=REASONING_RESULT_DIRECTORIES,
        result_files=REASONING_RESULT_FILES,
        result_file_schemas=REASONING_RESULT_FILE_SCHEMAS,
        result_anchor_paths=REASONING_RESULT_ANCHOR_PATHS,
        scorer_terminal_index_path=REASONING_SCORER_TERMINAL_INDEX_PATH,
    ),
)


def get_strict_captured_replay_profile(
    *,
    expected_environment: str,
    expected_profile_id: str,
) -> StrictCapturedReplayProfile:
    """Return the profile for one exact admitted pair; never infer either field."""
    if type(expected_environment) is not str or type(expected_profile_id) is not str:
        raise ValueError(
            "strict captured-replay environment/profile must be exact strings"
        )
    for profile in STRICT_CAPTURED_REPLAY_PROFILES:
        if (
            profile.environment == expected_environment
            and profile.profile_id == expected_profile_id
        ):
            return profile
    raise ValueError("unsupported strict captured-replay environment/profile pair")


__all__ = [
    "FORMAT_RESULT_ANCHOR_PATHS",
    "FORMAT_RESULT_DIRECTORIES",
    "FORMAT_RESULT_FILES",
    "FORMAT_RESULT_FILE_SCHEMAS",
    "FORMAT_SCORER_TERMINAL_INDEX_PATH",
    "REASONING_RESULT_ANCHOR_PATHS",
    "REASONING_RESULT_DIRECTORIES",
    "REASONING_RESULT_FILES",
    "REASONING_RESULT_FILE_SCHEMAS",
    "REASONING_SCORER_TERMINAL_INDEX_PATH",
    "STRICT_CAPTURED_REPLAY_PROFILES",
    "StrictCapturedReplayProfile",
    "get_strict_captured_replay_profile",
]
