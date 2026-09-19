"""Tests for Y4 finite scaling diagnostics."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.yang_mills import (  # noqa: E402
    FiniteScalingPoint,
    FiniteScalingReport,
    run_finite_scaling_study,
)


def _small_scaling() -> FiniteScalingReport:
    return run_finite_scaling_study(
        n_values=(6, 8),
        topologies=("cycle", "complete"),
        seeds=(3,),
        target_u6_ratios=(0.75,),
        gauge_seed=5,
    )


class TestFiniteScalingStudy:
    """Y4 finite scaling diagnostic contracts."""

    def test_scaling_report_shape_and_scope(self):
        report = _small_scaling()
        assert isinstance(report, FiniteScalingReport)
        assert report.verdict == "FINITE_SCALING_EVIDENCE"
        assert len(report.points) == 4
        assert report.summary["n_points"] == 4
        assert report.summary["n_groups"] == 2
        assert report.summary["scope"] == ("finite_scaling_diagnostic_not_clay_proof")
        assert report.summary["all_self_adjoint"]
        assert report.summary["all_gauge_invariant"]

    def test_points_and_grouped_scaling_are_populated(self):
        report = _small_scaling()
        for point in report.points:
            assert isinstance(point, FiniteScalingPoint)
            assert point.gap > 0.0
            assert point.lambda1 >= point.lambda0
            assert point.observed_u6_ratio == pytest.approx(0.75, abs=1e-10)
            assert point.potential_magnitude_ratio == pytest.approx(1.5, abs=1e-10)
            assert point.potential_magnitude_ratio_to_u6_drift_scale == pytest.approx(
                0.75,
                abs=1e-10,
            )
            assert not point.potential_magnitude_within_warning
            assert not point.u6_drift_assessed
            assert point.metadata["finite_scope"] == "Y4_finite_scaling_only"
            assert not point.metadata["u6_drift_assessed"]
        for group in report.grouped_scaling.values():
            assert group["positive_at_all_sizes"]
            assert group["scope"] == "finite_group_scaling_not_continuum_limit"
            assert not group["u6_drift_assessed"]
            assert group["target_potential_magnitude_ratio"] == pytest.approx(1.5)
            assert len(group["n_values"]) == 2
            assert len(group["mean_gaps"]) == 2

    def test_scaling_study_is_reproducible(self):
        first = _small_scaling()
        second = _small_scaling()
        assert first.verdict == second.verdict
        assert first.summary["mean_gap"] == pytest.approx(
            second.summary["mean_gap"],
            abs=1e-14,
        )
        assert [point.gap for point in first.points] == pytest.approx(
            [point.gap for point in second.points],
            abs=1e-14,
        )
        assert set(first.grouped_scaling) == set(second.grouped_scaling)
        for key, first_group in first.grouped_scaling.items():
            second_group = second.grouped_scaling[key]
            assert first_group["n_values"] == second_group["n_values"]
            assert first_group["mean_gaps"] == pytest.approx(
                second_group["mean_gaps"],
                abs=1e-14,
            )
            assert first_group["loglog_slope"] == pytest.approx(
                second_group["loglog_slope"],
                abs=1e-14,
            )

    def test_zero_graph_pressure_can_report_gap_collapse(self):
        report = run_finite_scaling_study(
            n_values=(3,),
            topologies=("complete",),
            seeds=(1,),
            target_u6_ratios=(0.0,),
            gauge_seed=1,
            eigen_tolerance=10.0,
        )
        assert report.verdict == "GAP_COLLAPSE_OBSERVED"

    def test_invalid_scaling_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            run_finite_scaling_study(n_values=())
        with pytest.raises(ValueError):
            run_finite_scaling_study(n_values=(1,))
        with pytest.raises(ValueError):
            run_finite_scaling_study(target_u6_ratios=(-1.0,))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_values": (2.5,)},
            {"n_values": (True,)},
            {"seeds": (False,)},
            {"target_u6_ratios": (math.nan,)},
            {"target_u6_ratios": (math.inf,)},
            {"target_u6_ratios": (True,)},
            {"phase_spread": math.nan},
            {"tolerance": 0.0},
        ],
    )
    def test_scaling_rejects_silent_coercions_and_nonfinite_controls(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            run_finite_scaling_study(**kwargs)

    def test_grid_scaling_uses_realized_node_count(self):
        report = run_finite_scaling_study(
            n_values=(8, 15),
            topologies=("grid",),
            seeds=(2,),
            target_u6_ratios=(0.25,),
        )

        assert [point.n for point in report.points] == [8, 15]
        assert [point.actual_n_nodes for point in report.points] == [4, 9]
        assert [point.metadata["requested_n"] for point in report.points] == [8, 15]
        group = next(iter(report.grouped_scaling.values()))
        assert group["n_values"] == [4, 9]

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.run_finite_scaling_study)
