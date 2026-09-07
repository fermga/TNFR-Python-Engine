"""Tests for the legacy U6-named Y2 potential-magnitude sweep."""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.yang_mills import (  # noqa: E402
    U6ConfinementSweepPoint,
    U6ConfinementSweepReport,
    run_u6_confinement_sweep,
)


def _small_sweep() -> U6ConfinementSweepReport:
    return run_u6_confinement_sweep(
        n_values=(8,),
        topologies=("cycle", "complete"),
        seeds=(5,),
        target_u6_ratios=(0.25, 1.25),
        gauge_seed=7,
    )


class TestU6ConfinementSweep:
    """Y2 finite sweep contracts."""

    def test_sweep_report_shape_and_scope(self):
        report = _small_sweep()
        assert isinstance(report, U6ConfinementSweepReport)
        assert report.verdict == "EMPIRICAL_FINITE_GRAPH_ONLY"
        assert len(report.points) == 4
        assert report.summary["n_points"] == 4
        assert report.summary["scope"] == ("finite_graph_y2_empirical_not_clay_proof")
        assert report.summary["n_confined"] == 2
        assert report.summary["n_unconfined"] == 2
        assert report.summary["n_below_pi_magnitude_scale"] == 2
        assert report.summary["n_at_or_above_pi_magnitude_scale"] == 2
        assert not report.summary["u6_drift_assessed"]
        assert report.summary["u6_definition"] == (
            "mean_i |Phi_s_after(i)-Phi_s_before(i)| < threshold"
        )
        assert report.summary["curvature_is_numerical_residual"]
        assert report.summary["mean_pure_gauge_consistency_residual"] == (
            report.summary["mean_yang_mills_residual"]
        )
        assert report.summary["cycle_closure_residual_active_fraction"] == (
            report.summary["curvature_active_fraction"]
        )

    def test_points_track_u6_targets_and_gap_contracts(self):
        report = _small_sweep()
        for point in report.points:
            assert isinstance(point, U6ConfinementSweepPoint)
            assert point.observed_u6_ratio == pytest.approx(
                point.target_u6_ratio,
                abs=1e-10,
            )
            assert point.u6_confined is (point.observed_u6_ratio < 1.0)
            assert point.potential_magnitude_ratio == pytest.approx(
                2.0 * point.observed_u6_ratio
            )
            assert (
                point.potential_magnitude_ratio_to_u6_drift_scale
                == point.observed_u6_ratio
            )
            assert point.potential_magnitude_below_pi_scale is (
                point.potential_magnitude_ratio < 1.0
            )
            assert not point.u6_drift_assessed
            assert point.is_self_adjoint
            assert point.gauge_invariant
            assert point.gap >= 0.0
            assert point.lambda1 >= point.lambda0
            assert point.metadata["finite_scope"] == ("Y2_empirical_finite_graph_only")
            assert not point.metadata["u6_drift_assessed"]
            assert point.metadata["legacy_u6_fields_are_magnitude_proxies"]
            assert point.metadata["u6_aggregation"] == (
                "mean_absolute_nodewise_drift"
            )
            assert point.metadata["u6_definition"] == (
                "mean_i |Phi_s_after(i)-Phi_s_before(i)| < threshold"
            )
            assert point.metadata["connection_scope"] == (
                "derived_exact_vertex_phase_one_form"
            )
            assert point.metadata["curvature_is_numerical_residual"]
            assert point.mean_pure_gauge_consistency_residual == (
                point.mean_yang_mills_residual
            )
            assert point.max_pure_gauge_consistency_residual == (
                point.max_yang_mills_residual
            )
            assert point.cycle_closure_residual_active is point.curvature_active
            assert point.curvature_is_numerical_residual

    def test_sweep_is_reproducible(self):
        first = _small_sweep()
        second = _small_sweep()
        assert first.verdict == second.verdict
        assert first.summary["mean_gap"] == pytest.approx(
            second.summary["mean_gap"],
            abs=1e-14,
        )
        assert first.summary["u6_gap_correlation"] == pytest.approx(
            second.summary["u6_gap_correlation"],
            abs=1e-14,
        )
        assert first.summary["potential_magnitude_gap_correlation"] == pytest.approx(
            second.summary["potential_magnitude_gap_correlation"],
            abs=1e-14,
        )
        assert [point.gap for point in first.points] == pytest.approx(
            [point.gap for point in second.points],
            abs=1e-14,
        )

    def test_legacy_pi_over_two_group_differs_from_pi_over_four_warning(self):
        report = run_u6_confinement_sweep(
            n_values=(8,),
            topologies=("cycle",),
            seeds=(3,),
            target_u6_ratios=(0.75,),
        )
        point = report.points[0]

        assert point.u6_confined
        assert point.potential_magnitude_ratio == pytest.approx(1.5)
        assert not point.potential_magnitude_within_warning
        assert report.summary["n_confined"] == 1
        assert report.summary["n_outside_potential_magnitude_warning"] == 1

    def test_invalid_sweep_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(n_values=())
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(target_u6_ratios=(-0.1,))
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(n_values=(1,))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_values": (2.5,)},
            {"n_values": (True,)},
            {"seeds": (False,)},
            {"target_u6_ratios": (math.nan,)},
            {"target_u6_ratios": (math.inf,)},
            {"target_u6_ratios": (True,)},
            {"tolerance": 0.0},
        ],
    )
    def test_sweep_rejects_silent_coercions_and_nonfinite_controls(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            run_u6_confinement_sweep(**kwargs)

    def test_grid_point_records_realized_node_count(self):
        report = run_u6_confinement_sweep(
            n_values=(8,),
            topologies=("grid",),
            seeds=(1,),
            target_u6_ratios=(0.25,),
        )
        point = report.points[0]

        assert point.n == 8
        assert point.actual_n_nodes == 4
        assert point.metadata["requested_n"] == 8
        assert point.metadata["actual_n_nodes"] == 4

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.run_u6_confinement_sweep)
