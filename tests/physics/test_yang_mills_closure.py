"""Tests for Y5 closure / obstruction classification."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.yang_mills import (  # noqa: E402
    YangMillsClosureReport,
    audit_nonabelian_derivability,
    classify_yang_mills_closure,
    run_finite_scaling_study,
)


def _small_scaling_kwargs():
    return {
        "n_values": (6, 8),
        "topologies": ("cycle", "complete"),
        "seeds": (3,),
        "target_u6_ratios": (0.75,),
        "gauge_seed": 5,
    }


class TestYangMillsClosureClassification:
    """Y5 separates finite TNFR results from Clay-strength closure."""

    def test_current_programme_classifies_branch_b_obstruction(self):
        report = classify_yang_mills_closure(
            derivability_kwargs={"seed": 7},
            scaling_kwargs=_small_scaling_kwargs(),
        )
        assert isinstance(report, YangMillsClosureReport)
        assert report.finite_tnfr_branch == "A_FINITE_U1_DIAGNOSTIC_SURFACE"
        assert report.clay_strength_branch == (
            "B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION"
        )
        assert report.verdict == "BRANCH_B_OBSTRUCTION_CLASSIFIED"
        assert not report.clay_problem_resolved
        assert "derive_component_mixing_connection" in report.open_requirements
        assert "prove_continuum_thermodynamic_liminf_gap_bound" in (
            report.open_requirements
        )
        assert report.evidence["canonical_gauge_group"] == "U(1)"
        assert report.metadata["scope"] == (
            "Y5_closure_obstruction_not_clay_proof"
        )

    def test_existing_reports_can_be_reused(self):
        derivability = audit_nonabelian_derivability(seed=7)
        scaling = run_finite_scaling_study(**_small_scaling_kwargs())
        report = classify_yang_mills_closure(
            derivability_report=derivability,
            scaling_report=scaling,
        )
        assert report.evidence["derivability_verdict"] == derivability.verdict
        assert report.evidence["scaling_verdict"] == scaling.verdict
        assert report.evidence["finite_group_count"] == scaling.summary[
            "n_groups"
        ]

    def test_sampled_gap_collapse_is_recorded_but_y3_still_blocks_clay(self):
        collapse_scaling = run_finite_scaling_study(
            n_values=(3,),
            topologies=("complete",),
            seeds=(1,),
            target_u6_ratios=(0.0,),
            gauge_seed=1,
            eigen_tolerance=10.0,
        )
        report = classify_yang_mills_closure(
            scaling_report=collapse_scaling,
            derivability_kwargs={"seed": 1},
        )
        assert report.finite_tnfr_branch == "C_FINITE_SAMPLE_GAP_COLLAPSE"
        assert report.clay_strength_branch == (
            "B_REQUIRES_NEW_CANONICAL_NONABELIAN_DERIVATION"
        )
        assert report.verdict == "BRANCH_B_OBSTRUCTION_CLASSIFIED"
        assert "stabilize_positive_finite_gap_surface" in (
            report.open_requirements
        )

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.classify_yang_mills_closure)
