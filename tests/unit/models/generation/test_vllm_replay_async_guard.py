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

"""Fail-closed guards for direct async APIs during captured-cohort replay."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.vllm_replay_bundle import VllmReplayDirectGenerationError


class _RecordingWorkerGroup:
    """Minimal worker group proving whether the driver dispatched live work."""

    dp_size = 1

    def __init__(self) -> None:
        self.dispatched_methods: list[str] = []

    def get_dp_leader_worker_idx(self, dp_shard_idx: int) -> int:
        assert dp_shard_idx == 0
        return 0

    def run_single_worker_single_data(
        self,
        *,
        method_name: str,
        worker_idx: int,
        data: object,
        greedy: bool,
    ) -> AsyncGenerator[Awaitable[tuple[int, BatchedDataDict]], None]:
        del worker_idx, data, greedy
        self.dispatched_methods.append(method_name)
        return _one_live_result()

    def shutdown(self, *, cleanup_method: str) -> bool:
        assert cleanup_method == "shutdown"
        return True


async def _one_live_result() -> AsyncGenerator[Awaitable[tuple[int, BatchedDataDict]], None]:
    async def result_ref() -> tuple[int, BatchedDataDict]:
        return (0, BatchedDataDict({"texts": ["live"]}))

    yield result_ref()


def _config(*, replay_required: bool) -> dict[str, Any]:
    vllm_cfg = {"async_engine": True}
    if replay_required:
        vllm_cfg.update(
            {
                "expose_http_server": True,
                "replay_bundle_sha256": "0" * 64,
                "replay_required": True,
            }
        )
    return {"_debug_payload_metrics": False, "vllm_cfg": vllm_cfg}


def _data(method_name: str) -> BatchedDataDict:
    if method_name == "generate_text_async":
        return BatchedDataDict({"prompts": ["prompt"]})
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "input_lengths": torch.tensor([2], dtype=torch.long),
        }
    )


def _generation(*, replay_required: bool) -> VllmGeneration:
    generation = object.__new__(VllmGeneration)
    generation.cfg = _config(replay_required=replay_required)
    generation.worker_group = _RecordingWorkerGroup()
    generation.current_generate_dp_shard_idx = 0
    generation.fleet_monitor = None
    generation.fleet_selector = None
    generation.weight_synchronizer = None
    return generation


def _worker(*, replay_required: bool) -> VllmAsyncGenerationWorkerImpl:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = _config(replay_required=replay_required)
    worker._vllm_replay_bundle = object() if replay_required else None
    worker.llm = MagicMock()
    return worker


async def _collect(generator: AsyncIterable[object]) -> list[object]:
    return [item async for item in generator]


def _call_direct_async(
    target: VllmGeneration | VllmAsyncGenerationWorkerImpl,
    method_name: str,
    data: BatchedDataDict,
) -> AsyncIterable[object]:
    if method_name == "generate_async":
        return target.generate_async(data)
    if method_name == "generate_text_async":
        return target.generate_text_async(data)
    raise ValueError(f"unsupported direct async method {method_name!r}")


@pytest.mark.parametrize("method_name", ["generate_async", "generate_text_async"])
def test_required_replay_rejects_driver_direct_async_before_dispatch(
    method_name: str,
) -> None:
    generation = _generation(replay_required=True)

    with pytest.raises(
        VllmReplayDirectGenerationError,
        match=rf"{method_name}.*only be claimed through /v1/chat/completions",
    ):
        asyncio.run(_collect(_call_direct_async(generation, method_name, _data(method_name))))

    assert generation.worker_group.dispatched_methods == []


@pytest.mark.parametrize("method_name", ["generate_async", "generate_text_async"])
def test_required_replay_rejects_worker_direct_async_before_live_engine(
    method_name: str,
) -> None:
    worker = _worker(replay_required=True)

    with pytest.raises(
        VllmReplayDirectGenerationError,
        match=rf"{method_name}.*only be claimed through /v1/chat/completions",
    ):
        asyncio.run(_collect(_call_direct_async(worker, method_name, _data(method_name))))

    worker.llm.generate.assert_not_called()


@pytest.mark.parametrize("method_name", ["generate_async", "generate_text_async"])
def test_live_driver_direct_async_dispatch_is_unchanged(method_name: str) -> None:
    generation = _generation(replay_required=False)

    results = asyncio.run(_collect(_call_direct_async(generation, method_name, _data(method_name))))

    assert len(results) == 1
    assert generation.worker_group.dispatched_methods == [method_name]


@pytest.mark.parametrize("method_name", ["generate_async", "generate_text_async"])
def test_live_worker_empty_direct_async_is_unchanged(method_name: str) -> None:
    worker = _worker(replay_required=False)
    if method_name == "generate_text_async":
        data = BatchedDataDict({"prompts": []})
    else:
        data = BatchedDataDict(
            {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "input_lengths": torch.empty((0,), dtype=torch.long),
            }
        )

    assert asyncio.run(_collect(_call_direct_async(worker, method_name, data))) == []
    worker.llm.generate.assert_not_called()
