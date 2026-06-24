"""Tests for TNFR–Yang–Mills Y2 U6 confinement sweeps."""

from __future__ import annotations

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

    def test_points_track_u6_targets_and_gap_contracts(self):
        report = _small_sweep()
        for point in report.points:
            assert isinstance(point, U6ConfinementSweepPoint)
            assert point.observed_u6_ratio == pytest.approx(
                point.target_u6_ratio,
                abs=1e-10,
            )
            assert point.u6_confined is (point.observed_u6_ratio < 1.0)
            assert point.is_self_adjoint
            assert point.gauge_invariant
            assert point.gap >= 0.0
            assert point.lambda1 >= point.lambda0
            assert point.metadata["finite_scope"] == ("Y2_empirical_finite_graph_only")

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
        assert [point.gap for point in first.points] == pytest.approx(
            [point.gap for point in second.points],
            abs=1e-14,
        )

    def test_invalid_sweep_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(n_values=())
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(target_u6_ratios=(-0.1,))
        with pytest.raises(ValueError):
            run_u6_confinement_sweep(n_values=(1,))

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.run_u6_confinement_sweep)
