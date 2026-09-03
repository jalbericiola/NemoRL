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

"""CPU tests for the split-API presharded wrappers and TQPolicy fan-out.

These two layers sit between the SC driver and the backend state machine
and were previously exercised only by the GPU-gated parity test — the
latent bugs the PR #2683 review surfaced (futures consumed with the wrong
API, an unused per-microbatch return) lived exactly here. Pin the
contracts cheaply:
  - ``*_presharded`` wrappers: pass-through begin/finish/abort, the
    fetch → attach → backend chain in ``train_microbatch_presharded``
    (returning None), and the ``is_replica_leader`` tag on finish.
  - TQPolicy driver: single-data futures consumed via ``ray.get``,
    replica-twin dedup in ``finish_train_step`` aggregation, and
    ``train_microbatches_from_meta`` returning None.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS, ROUTED_EXPERTS_FIELD
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.models.policy import validate_shared_prefix_training_config
from nemo_rl.models.policy.tq_policy import TQPolicy, _aggregate_train_results


class _SplitStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(self, is_leader: bool = True):
        self.calls: list[tuple] = []
        self._leader = is_leader

    def _fetch(self, meta):
        self.calls.append(("fetch", meta))
        return {"data_from": meta}

    def _attach_or_repack_pack_metadata(self, data, meta):
        self.calls.append(("attach", meta))
        return data

    def _is_replica_leader(self) -> bool:
        return self._leader

    # backend split API
    def begin_train_step(self, loss_fn, gbs=None, mbs=None):
        self.calls.append(("begin", loss_fn, gbs, mbs))

    def train_microbatch(self, data):
        self.calls.append(("train_microbatch", data))

    def finish_train_step(self):
        self.calls.append(("finish",))
        return {
            "global_loss": 1.0,
            "grad_norm": 0.5,
            "all_mb_metrics": {"loss": [1.0]},
            "mtp_metrics": {"mtp_1_loss": 0.25, "grad_norm": 1.25},
        }

    def abort_train_step(self):
        self.calls.append(("abort",))


def _meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["s0", "s1"],
    )


def _mtp5_metrics() -> dict[str, float]:
    metrics = {
        metric_name: head_index / 10
        for head_index in range(1, 6)
        for metric_name in (
            f"mtp_{head_index}_loss",
            f"mtp_{head_index}_acceptance_rate",
        )
    }
    metrics["grad_norm"] = 1.25
    return metrics


def _train_result(
    *,
    mtp_metrics: dict[str, float] | None = None,
    draft_grad_norm: float | None = None,
    leader: bool = True,
    update_successful: bool = True,
) -> dict:
    result = {
        "global_loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {"loss": [0.1]},
        "is_replica_leader": leader,
        "update_successful": update_successful,
    }
    if mtp_metrics is not None:
        result["mtp_metrics"] = mtp_metrics
    if draft_grad_norm is not None:
        result["draft_grad_norm"] = draft_grad_norm
    return result


class TestPreshardedWrappers:
    def test_begin_forwards_args(self):
        w = _SplitStubWorker()
        loss_fn = object()
        w.begin_train_step_presharded(loss_fn=loss_fn, gbs=8, mbs=2)
        assert w.calls == [("begin", loss_fn, 8, 2)]

    def test_train_microbatch_fetches_attaches_then_dispatches(self):
        w = _SplitStubWorker()
        meta = _meta()
        out = w.train_microbatch_presharded(meta=meta)
        assert out is None  # metrics accumulate in the open-step state
        assert [c[0] for c in w.calls] == ["fetch", "attach", "train_microbatch"]
        assert w.calls[-1][1] == {"data_from": meta}

    def test_finish_tags_replica_leader(self):
        leader = _SplitStubWorker(is_leader=True)
        twin = _SplitStubWorker(is_leader=False)
        assert leader.finish_train_step_presharded()["is_replica_leader"] is True
        result = twin.finish_train_step_presharded()
        assert result["is_replica_leader"] is False
        # backend payload passes through untouched
        assert result["global_loss"] == 1.0
        assert result["mtp_metrics"] == {
            "mtp_1_loss": 0.25,
            "grad_norm": 1.25,
        }

    def test_abort_forwards(self):
        w = _SplitStubWorker()
        w.abort_train_step_presharded()
        assert w.calls == [("abort",)]


def _make_tq_policy(
    *,
    mtp_num_layers: int = 0,
    mtp_detach_heads: bool = False,
    clip_grad: float = 0.0,
    draft_enabled: bool = False,
) -> tuple[TQPolicy, MagicMock]:
    """Bare TQPolicy with the attributes the split fan-out touches."""
    p = object.__new__(TQPolicy)
    p.cfg = {
        "train_global_batch_size": 8,
        "train_micro_batch_size": 2,
        "megatron_cfg": {
            "mtp_num_layers": mtp_num_layers,
            "mtp_detach_heads": mtp_detach_heads,
            "optimizer": {"clip_grad": clip_grad},
        },
        "draft": {"enabled": draft_enabled},
    }
    p.shared_prefix_training_config = validate_shared_prefix_training_config(p.cfg)
    p._router_replay_enabled = False
    p.flops_tracker = None
    wg = MagicMock()
    wg.run_all_workers_single_data.return_value = ["f0", "f1"]
    p.worker_group = wg
    p.sharding_annotations = MagicMock()
    p.sharding_annotations.get_axis_size.return_value = 2
    return p, wg


class TestTQPolicySplitFanout:
    def test_begin_consumes_single_data_futures_with_ray_get(self):
        """run_all_workers_single_data returns plain ObjectRefs, not a
        MultiWorkerFuture — the fan-out must ray.get them (PR #2683
        review; first execution of this path raised AttributeError)."""
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.begin_train_step(loss_fn="LF")
        wg.run_all_workers_single_data.assert_called_once_with(
            "begin_train_step_presharded", loss_fn="LF", gbs=8, mbs=2
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])
        wg.get_all_worker_results.assert_not_called()

    def test_train_microbatches_from_meta_dispatches_and_returns_none(self):
        p, wg = _make_tq_policy()
        assert p.shared_prefix_training_config.mode == "disabled"
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            out = p.train_microbatches_from_meta(meta)
        assert out is None
        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == list(DP_TRAIN_FIELDS)
        assert ROUTED_EXPERTS_FIELD not in train_meta.fields
        assert (
            wg.run_all_workers_sharded_data.call_args.args[0]
            == "train_microbatch_presharded"
        )
        # sharded dispatch returns a MultiWorkerFuture → waited via
        # get_all_worker_results (unlike the single-data fan-outs)
        wg.get_all_worker_results.assert_called_once()

    def test_train_microbatches_requests_routed_experts_for_router_replay(self):
        p, _ = _make_tq_policy()
        p._router_replay_enabled = True
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            p.train_microbatches_from_meta(meta)

        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == [*DP_TRAIN_FIELDS, ROUTED_EXPERTS_FIELD]

    def test_finish_dedupes_replica_twins(self):
        """TP/CP twins return identical metric copies; aggregating without
        the is_replica_leader filter inflates every per-token metric."""

        def _result(leader: bool) -> dict:
            return {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [0.1]},
                "is_replica_leader": leader,
                "update_successful": True,
            }

        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            # 2 DP leaders + 2 TP twins
            mock_ray.get.return_value = [
                _result(True),
                _result(False),
                _result(True),
                _result(False),
            ]
            out = p.finish_train_step()
        assert out["all_mb_metrics"]["loss"] == [0.1, 0.1]  # twins dropped
        # _aggregate_train_results surfaces global_loss under "loss"
        assert out["loss"] == 1.0
        assert out["update_successful"] is True

    def test_finish_propagates_one_exact_copy_of_mtp_training_metrics(self):
        """MTP metrics are global replicas, not per-DP values to sum."""
        mtp_metrics = _mtp5_metrics()
        p, _ = _make_tq_policy(
            mtp_num_layers=5,
            mtp_detach_heads=True,
            clip_grad=1.0,
            draft_enabled=True,
        )
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = [
                _train_result(
                    mtp_metrics=mtp_metrics,
                    draft_grad_norm=2.25,
                    leader=True,
                ),
                _train_result(
                    mtp_metrics=dict(mtp_metrics),
                    draft_grad_norm=2.25,
                    leader=False,
                ),
                _train_result(
                    mtp_metrics=dict(reversed(list(mtp_metrics.items()))),
                    draft_grad_norm=2.25,
                    leader=True,
                ),
                _train_result(
                    mtp_metrics=dict(reversed(list(mtp_metrics.items()))),
                    draft_grad_norm=2.25,
                    leader=False,
                ),
            ]
            out = p.finish_train_step()

        assert out["mtp_metrics"] == dict(sorted(mtp_metrics.items()))
        assert out["mtp_metrics"]["grad_norm"] == pytest.approx(1.25)
        assert out["mtp_metrics"]["mtp_5_loss"] == pytest.approx(0.5)
        assert out["draft_grad_norm"] == pytest.approx(2.25)

    def test_finish_rejects_mtp_drift_on_nonleader_replica(self):
        mtp_metrics = _mtp5_metrics()
        drifted = dict(mtp_metrics)
        drifted["mtp_4_loss"] += 0.01
        p, _ = _make_tq_policy(
            mtp_num_layers=5,
            mtp_detach_heads=True,
            clip_grad=1.0,
        )
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="mtp_metrics/mtp_4_loss"),
        ):
            mock_ray.get.return_value = [
                _train_result(mtp_metrics=mtp_metrics, leader=True),
                _train_result(mtp_metrics=drifted, leader=False),
            ]
            p.finish_train_step()

    def test_finish_requires_configured_mtp_schema(self):
        p, _ = _make_tq_policy(mtp_num_layers=5)
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="required replicated mtp_metrics"),
        ):
            mock_ray.get.return_value = [_train_result(), _train_result()]
            p.finish_train_step()

    def test_finish_rejects_incomplete_configured_mtp_schema(self):
        incomplete = _mtp5_metrics()
        incomplete.pop("mtp_5_acceptance_rate")
        p, _ = _make_tq_policy(
            mtp_num_layers=5,
            mtp_detach_heads=True,
            clip_grad=1.0,
        )
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="schema does not match configuration"),
        ):
            mock_ray.get.return_value = [
                _train_result(mtp_metrics=incomplete),
                _train_result(mtp_metrics=incomplete),
            ]
            p.finish_train_step()

    def test_finish_rejects_mtp_telemetry_when_disabled(self):
        p, _ = _make_tq_policy()
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="feature is disabled"),
        ):
            mock_ray.get.return_value = [
                _train_result(mtp_metrics=_mtp5_metrics()),
                _train_result(mtp_metrics=_mtp5_metrics()),
            ]
            p.finish_train_step()

    def test_finish_requires_configured_draft_grad_norm(self):
        p, _ = _make_tq_policy(clip_grad=1.0, draft_enabled=True)
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="required replicated draft_grad_norm"),
        ):
            mock_ray.get.return_value = [_train_result(), _train_result()]
            p.finish_train_step()

    def test_finish_rejects_draft_drift_on_nonleader_replica(self):
        p, _ = _make_tq_policy(clip_grad=1.0, draft_enabled=True)
        with (
            patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray,
            pytest.raises(RuntimeError, match="draft_grad_norm"),
        ):
            mock_ray.get.return_value = [
                _train_result(draft_grad_norm=2.25, leader=True),
                _train_result(draft_grad_norm=9.0, leader=False),
            ]
            p.finish_train_step()

    def test_mtp_replica_aggregation_is_order_independent(self):
        mtp_metrics = _mtp5_metrics()
        forward = [
            _train_result(mtp_metrics=mtp_metrics),
            _train_result(mtp_metrics=dict(reversed(list(mtp_metrics.items())))),
        ]

        assert (
            _aggregate_train_results(forward)["mtp_metrics"]
            == (_aggregate_train_results(list(reversed(forward)))["mtp_metrics"])
        )

    def test_mtp_replica_aggregation_rejects_partial_telemetry(self):
        with pytest.raises(RuntimeError, match="missing from result"):
            _aggregate_train_results(
                [
                    _train_result(mtp_metrics=_mtp5_metrics()),
                    _train_result(),
                ]
            )

    def test_mtp_replica_aggregation_rejects_disagreement(self):
        drifted = _mtp5_metrics()
        drifted["mtp_3_loss"] += 0.01
        with pytest.raises(RuntimeError, match="mtp_metrics/mtp_3_loss"):
            _aggregate_train_results(
                [
                    _train_result(mtp_metrics=_mtp5_metrics()),
                    _train_result(mtp_metrics=drifted),
                ]
            )

    @pytest.mark.parametrize(
        "invalid",
        [
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
            torch.tensor([1.0, 2.0]),
        ],
    )
    def test_mtp_replica_aggregation_rejects_invalid_scalars(self, invalid):
        invalid_metrics = _mtp5_metrics()
        invalid_metrics["mtp_3_loss"] = invalid
        with pytest.raises((TypeError, ValueError), match="mtp_metrics/mtp_3_loss"):
            _aggregate_train_results(
                [
                    _train_result(mtp_metrics=invalid_metrics),
                    _train_result(mtp_metrics=invalid_metrics),
                ]
            )

    @pytest.mark.parametrize("invalid", [None, [1.0], "not-a-map"])
    def test_mtp_replica_aggregation_rejects_nonmapping(self, invalid):
        result = _train_result(mtp_metrics=_mtp5_metrics())
        result["mtp_metrics"] = invalid
        with pytest.raises(TypeError, match="must be a mapping"):
            _aggregate_train_results([result, result])

    def test_mtp_replica_aggregation_rejects_nonstring_keys(self):
        metrics = _mtp5_metrics()
        metrics[1] = 0.5
        with pytest.raises(TypeError, match="keys must be strings"):
            _aggregate_train_results(
                [
                    _train_result(mtp_metrics=metrics),
                    _train_result(mtp_metrics=metrics),
                ]
            )

    def test_draft_grad_norm_replica_aggregation_rejects_disagreement(self):
        with pytest.raises(RuntimeError, match="draft_grad_norm"):
            _aggregate_train_results(
                [
                    _train_result(draft_grad_norm=1.0),
                    _train_result(draft_grad_norm=2.0),
                ]
            )

    def test_draft_grad_norm_replica_aggregation_rejects_partial_telemetry(self):
        with pytest.raises(RuntimeError, match="missing from result"):
            _aggregate_train_results(
                [
                    _train_result(draft_grad_norm=1.0),
                    _train_result(),
                ]
            )

    @pytest.mark.parametrize(
        "invalid",
        [
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
            torch.tensor([1.0, 2.0]),
        ],
    )
    def test_draft_grad_norm_replica_aggregation_rejects_invalid_scalar(self, invalid):
        with pytest.raises((TypeError, ValueError), match="draft_grad_norm"):
            _aggregate_train_results(
                [
                    _train_result(draft_grad_norm=invalid),
                    _train_result(draft_grad_norm=invalid),
                ]
            )

    def test_abort_consumes_single_data_futures_with_ray_get(self):
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.abort_train_step()
        wg.run_all_workers_single_data.assert_called_once_with(
            "abort_train_step_presharded"
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])
