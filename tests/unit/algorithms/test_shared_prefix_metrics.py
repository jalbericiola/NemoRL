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

import pytest
import torch

from nemo_rl.algorithms.shared_prefix_metrics import (
    combine_shared_prefix_opportunities,
    observe_shared_prefix_opportunity,
)


def test_observe_shared_prefix_opportunity_separates_exact_and_ratio_estimates():
    opportunity = observe_shared_prefix_opportunity(
        group_ids=["group-a", "group-a", None],
        prompt_token_ids=torch.tensor(
            [
                [11, 12],
                [11, 12],
                [21, 0],
            ]
        ),
        prompt_lengths=torch.tensor([2, 2, 1]),
        input_lengths=torch.tensor([5, 4, 3]),
        token_mask=torch.tensor(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=torch.float32,
        ),
        sample_mask=torch.ones(3),
        expected_group_size=2,
    )

    assert opportunity.total_sequences == 3
    assert opportunity.eligible_sequences == 2
    assert opportunity.complete_groups == 1
    assert opportunity.fallback_sequences == 1
    assert opportunity.total_tokens == 12
    assert opportunity.prompt_tokens == 5
    assert opportunity.valid_loss_tokens == 4
    assert opportunity.non_loss_suffix_tokens == 3
    assert opportunity.shareable_prompt_tokens == 2
    assert opportunity.ideal_shared_token_work == 10

    metrics = opportunity.as_metrics()
    assert metrics["shared_prefix/valid_to_total_token_ratio"] == pytest.approx(1 / 3)
    assert metrics["shared_prefix/exact_group_sequence_coverage"] == pytest.approx(
        2 / 3
    )
    assert metrics["shared_prefix/ideal_token_reduction"] == pytest.approx(1 / 6)
    assert metrics[
        "shared_prefix/loss_ratio_upper_bound_token_reduction"
    ] == pytest.approx(1 / 3)


def test_observe_shared_prefix_opportunity_keeps_identical_prompt_ids_separate():
    opportunity = observe_shared_prefix_opportunity(
        group_ids=["group-a", "group-a", "group-b", "group-b"],
        prompt_token_ids=torch.tensor([[1, 2]] * 4),
        prompt_lengths=torch.tensor([2] * 4),
        input_lengths=torch.tensor([3] * 4),
        token_mask=torch.tensor([[0, 0, 1]] * 4),
        sample_mask=torch.ones(4),
        expected_group_size=2,
    )

    assert opportunity.complete_groups == 2
    assert opportunity.eligible_sequences == 4
    assert opportunity.fallback_sequences == 0
    assert opportunity.shareable_prompt_tokens == 4


def test_observe_shared_prefix_opportunity_matches_sample_mask_accounting():
    opportunity = observe_shared_prefix_opportunity(
        group_ids=["group-a", "group-a"],
        prompt_token_ids=torch.tensor([[1], [1]]),
        prompt_lengths=torch.tensor([1, 1]),
        input_lengths=torch.tensor([3, 3]),
        token_mask=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        sample_mask=torch.tensor([1.0, 0.0]),
        expected_group_size=2,
    )

    assert opportunity.valid_loss_tokens == 2
    assert opportunity.non_loss_suffix_tokens == 2


def test_observe_shared_prefix_opportunity_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="shorter than prompt_lengths"):
        observe_shared_prefix_opportunity(
            group_ids=["group-a", "group-a"],
            prompt_token_ids=torch.tensor([[1, 2], [1, 2]]),
            prompt_lengths=torch.tensor([2, 2]),
            input_lengths=torch.tensor([1, 3]),
            token_mask=torch.zeros((2, 3)),
            sample_mask=torch.ones(2),
            expected_group_size=2,
        )


def test_combine_shared_prefix_opportunities_derives_step_ratios_from_totals():
    first = observe_shared_prefix_opportunity(
        group_ids=["group-a", "group-a"],
        prompt_token_ids=torch.tensor([[1], [1]]),
        prompt_lengths=torch.tensor([1, 1]),
        input_lengths=torch.tensor([3, 3]),
        token_mask=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        sample_mask=torch.ones(2),
        expected_group_size=2,
    )
    second = observe_shared_prefix_opportunity(
        group_ids=["group-b", "group-b"],
        prompt_token_ids=torch.tensor([[2, 3], [2, 3]]),
        prompt_lengths=torch.tensor([2, 2]),
        input_lengths=torch.tensor([3, 3]),
        token_mask=torch.tensor([[0, 0, 1], [0, 0, 1]]),
        sample_mask=torch.ones(2),
        expected_group_size=2,
    )

    combined = combine_shared_prefix_opportunities([first, second])

    assert combined.total_sequences == 4
    assert combined.complete_groups == 2
    assert combined.total_tokens == 12
    assert combined.prompt_tokens == 6
    assert combined.valid_loss_tokens == 6
    assert combined.shareable_prompt_tokens == 3
    assert combined.as_metrics()[
        "shared_prefix/ideal_token_reduction"
    ] == pytest.approx(0.25)


def test_combine_shared_prefix_opportunities_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        combine_shared_prefix_opportunities([])


def test_observe_shared_prefix_opportunity_does_not_merge_partial_group_ids():
    opportunity = observe_shared_prefix_opportunity(
        group_ids=["group-a", "group-b"],
        prompt_token_ids=torch.tensor([[1, 2], [1, 2]]),
        prompt_lengths=torch.tensor([2, 2]),
        input_lengths=torch.tensor([3, 3]),
        token_mask=torch.tensor([[0, 0, 1], [0, 0, 1]]),
        sample_mask=torch.ones(2),
        expected_group_size=2,
    )

    assert opportunity.complete_groups == 0
    assert opportunity.eligible_sequences == 0
    assert opportunity.fallback_sequences == 2
    assert opportunity.shareable_prompt_tokens == 0


def test_observe_shared_prefix_opportunity_rejects_prompt_mismatch_within_id():
    with pytest.raises(ValueError, match="maps to multiple exact prompts"):
        observe_shared_prefix_opportunity(
            group_ids=["group-a", "group-a"],
            prompt_token_ids=torch.tensor([[1, 2], [1, 3]]),
            prompt_lengths=torch.tensor([2, 2]),
            input_lengths=torch.tensor([3, 3]),
            token_mask=torch.tensor([[0, 0, 1], [0, 0, 1]]),
            sample_mask=torch.ones(2),
            expected_group_size=2,
        )
