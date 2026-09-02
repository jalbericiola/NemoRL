# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Tier-0 smoke tests — pre-commit gates.

Cheapest tier: catches drift in module paths, registry keys, and the
public ABC surface. Each test runs in milliseconds and never touches
real Ray / vLLM / TQ.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch


def test_sync_utils_module_imports() -> None:
    """Catches FQN drift after the algorithms.sync_utils consolidation."""
    from nemo_rl.experience.sync_rollout_actor import (
        SyncRolloutActor,
        kv_first_write,
    )

    # ``SyncRolloutActor`` is wrapped by ``@ray.remote`` into
    # ``ActorClass(SyncRolloutActor)`` — the wrapper has no
    # ``__name__`` attribute. Check via ``repr`` instead.
    assert "SyncRolloutActor" in repr(SyncRolloutActor)
    assert callable(kv_first_write)


def test_sync_rollout_actor_resolves_exact_training_sample_mask() -> None:
    from nemo_rl.experience.sync_rollout_actor import _resolve_training_sample_mask

    loss_multiplier = torch.tensor([1.0, 0.5, 0.0])
    resolved = _resolve_training_sample_mask(
        loss_multiplier,
        torch.tensor([False, True, True]),
    )

    assert resolved.tolist() == [1.0, 0.0, 0.0]
    assert loss_multiplier.tolist() == [1.0, 0.5, 0.0]
    assert resolved.data_ptr() != loss_multiplier.data_ptr()


@pytest.mark.parametrize(
    ("mask_sample", "error_type", "match"),
    [
        (torch.tensor([0, 1]), TypeError, "dtype bool"),
        (torch.tensor([[False, True]]), ValueError, "shape must match"),
    ],
)
def test_sync_rollout_actor_rejects_invalid_gym_mask(
    mask_sample: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    from nemo_rl.experience.sync_rollout_actor import _resolve_training_sample_mask

    with pytest.raises(error_type, match=match):
        _resolve_training_sample_mask(torch.ones(2), mask_sample)


@pytest.mark.parametrize(("gate_enabled", "expected_mask"), [(True, 0.0), (False, 0.5)])
def test_sync_rollout_actor_writes_one_resolved_mask_and_reward_schema(
    monkeypatch: pytest.MonkeyPatch,
    gate_enabled: bool,
    expected_mask: float,
) -> None:
    import nemo_rl.environments.nemo_gym as nemo_gym_mod
    import nemo_rl.experience.sync_rollout_actor as actor_mod
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    actor_cls = actor_mod.SyncRolloutActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor.policy_generation = None
    actor.tokenizer = SimpleNamespace(pad_token_id=0)
    actor.task_to_env = {}
    actor._dp_client = object()
    actor.master_config = SimpleNamespace(
        policy={
            "generation": {},
            "make_sequence_length_divisible_by": 1,
            "router_replay": {"enabled": False},
        },
        env={
            "should_mask_flagged_samples": gate_enabled,
            "nemo_gym": {},
        },
        grpo=SimpleNamespace(
            deduplicate_multimodal_data=True,
            debug_payload_metrics=False,
        ),
        logger={"wandb_enabled": False, "wandb": {}},
        reward_penalties=None,
    )

    raw_extras = {"instance_config": {"mask_sample": True}}
    captured: dict[str, object] = {}

    def fake_run_nemo_gym_rollout_sync(**kwargs):
        captured["gate"] = kwargs["mask_env_flagged_samples"]
        final_data = {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "prompt",
                        "token_ids": torch.tensor([1]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([2, 3]),
                        "generation_logprobs": torch.tensor([-0.1, -0.2]),
                    },
                ]
            ],
            "length": torch.tensor([1]),
            "loss_multiplier": torch.tensor([0.5]),
            "raw_environment_reward": torch.tensor([1.0]),
            "pre_penalty_reward": torch.tensor([0.8]),
            "total_reward": torch.tensor([0.0]),
            "truncated": torch.tensor([False]),
            "extra_env_info": [raw_extras],
        }
        if kwargs["mask_env_flagged_samples"]:
            final_data["mask_sample"] = torch.tensor([True])
        return SimpleNamespace(
            final_batch=BatchedDataDict(final_data),
            rollout_metrics={},
        )

    def fake_first_write(data, **kwargs):
        captured["bulk"] = data
        return SimpleNamespace(sample_ids=kwargs["sample_ids"])

    monkeypatch.setattr(nemo_gym_mod, "should_use_nemo_gym", lambda _cfg: True)
    monkeypatch.setattr(
        actor_mod,
        "run_nemo_gym_rollout_sync",
        fake_run_nemo_gym_rollout_sync,
    )
    monkeypatch.setattr(actor_mod, "get_nemo_gym_thinking_tags", lambda _env: [])
    monkeypatch.setattr(
        actor_mod,
        "get_shared_prefix_training_config",
        lambda _policy: SimpleNamespace(mode="off"),
    )
    monkeypatch.setattr(actor_mod, "trace_rollout_payload", lambda **_kwargs: None)
    monkeypatch.setattr(actor_mod, "kv_first_write", fake_first_write)

    _, driver_carry, _, _ = actor.rollout_to_tq(
        BatchedDataDict({"row": [0]}),
        partition_id="train",
    )

    bulk = captured["bulk"]
    assert captured["gate"] is gate_enabled
    assert bulk["sample_mask"].tolist() == [expected_mask]
    assert driver_carry["loss_multiplier"].tolist() == [expected_mask]
    assert bulk["raw_environment_reward"].tolist() == [1.0]
    assert bulk["pre_penalty_reward"].tolist() == pytest.approx([0.8])
    assert bulk["total_reward"].tolist() == [0.0]
    assert bulk["extra_env_info"].tolist() == [raw_extras]
    assert raw_extras["instance_config"]["mask_sample"] is True


def test_sync_rollout_actor_registered_under_vllm_tier() -> None:
    """Multinode runs depend on this — without it, tensordict missing on
    worker nodes (real bug seen in job 11614968)."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    fqn = "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    env = get_actor_python_env(fqn)
    # Same tier as vLLM workers / AsyncTrajectoryCollector / ReplayBuffer.
    # Allow either the resolved exec path or the SYSTEM-override sentinel.
    assert env in (PY_EXECUTABLES.VLLM, PY_EXECUTABLES.SYSTEM), (
        f"unexpected env tier for {fqn}: {env!r}"
    )


def test_kvbatchmeta_schema_unchanged() -> None:
    """Schema break check — KVBatchMeta is the cross-process boundary;
    adding/removing a field silently would break adapters that pickle it."""
    from nemo_rl.data_plane.interfaces import KVBatchMeta

    expected_fields = {
        "partition_id",
        "task_name",
        "sample_ids",
        "fields",
        "sequence_lengths",
        "extra_info",
        "tags",
    }
    actual_fields = {f.name for f in KVBatchMeta.__dataclass_fields__.values()}
    assert actual_fields == expected_fields, (
        f"KVBatchMeta schema drifted. expected={expected_fields}, "
        f"actual={actual_fields}"
    )


def test_dataplane_client_abc_surface() -> None:
    """Catches accidental ABC method removal / rename — e.g. dropping
    ``clear_samples`` would break step-end teardown silently."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    expected_methods = {
        # task-mediated
        "register_partition",
        "claim_meta",
        "get_data",
        "check_consumption_status",
        # direct-by-key
        "put_samples",
        "get_samples",
        "clear_samples",
        # lifecycle
        "close",
    }
    actual_methods = {
        name
        for name, member in inspect.getmembers(DataPlaneClient, callable)
        if not name.startswith("_") and getattr(member, "__isabstractmethod__", False)
    }
    assert expected_methods.issubset(actual_methods), (
        f"DataPlaneClient ABC missing methods: {expected_methods - actual_methods}"
    )


def test_async_and_sync_actors_share_env_tier() -> None:
    """Sync should mirror async's env tier — both drive vLLM and write
    tensordict to TQ, so they need the same VLLM venv."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    sync_env = get_actor_python_env(
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    )
    async_env = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
    )
    assert sync_env == async_env, (
        f"Sync vs async env tier drift: sync={sync_env!r}, async={async_env!r}"
    )


def test_sync_rollout_actor_prompt_extraction_and_masks_match_grpo() -> None:
    """TQ rollouts must mirror GRPO's length-based prompt extraction."""
    import torch

    from nemo_rl.experience.sync_rollout_actor import (
        _flatten_rollout_message_log_for_tq,
    )

    message_logs = [
        [
            {"role": "user", "content": "first", "token_ids": torch.tensor([1, 2])},
            {
                "role": "assistant",
                "content": "history",
                "token_ids": torch.tensor([3, 4]),
            },
            {"role": "user", "content": "next", "token_ids": torch.tensor([5])},
            {
                "role": "assistant",
                "content": "generated",
                "token_ids": torch.tensor([6, 7]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
            },
        ]
    ]

    flat, _input_lengths, prompt_flat = _flatten_rollout_message_log_for_tq(
        message_logs,
        torch.tensor([5]),
        pad_token_id=0,
        make_sequence_length_divisible_by=1,
    )

    assert torch.equal(prompt_flat["token_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(
        flat["token_loss_mask"],
        torch.tensor([[0, 0, 0, 0, 0, 1, 1]]),
    )
    assert torch.allclose(
        flat["generation_logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]]),
    )
