"""Tests for Y3 non-Abelian derivability audits."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.yang_mills import (  # noqa: E402
    NonAbelianCandidateAudit,
    NonAbelianDerivabilityReport,
    audit_nonabelian_derivability,
    build_structural_gauge_graph,
)


class TestNonAbelianDerivabilityAudit:
    """Y3 keeps non-Abelian promotion behind derivability evidence."""

    def test_default_audit_reports_open_derivability_gap(self):
        report = audit_nonabelian_derivability(seed=7)
        assert isinstance(report, NonAbelianDerivabilityReport)
        assert report.canonical_gauge_group == "U(1)"
        assert report.u1_baseline_confirmed
        assert not report.nonabelian_derived
        assert report.verdict == "OPEN_DERIVABILITY_GAP"
        assert report.summary["scope"] == (
            "Y3_derivability_audit_not_nonabelian_promotion"
        )
        assert {candidate.route for candidate in report.candidates} == {
            "u5_nested_epi_multiplet",
            "thol_remesh_internal_space",
            "cycle_basis_bundle",
        }

    def test_nested_epi_route_still_lacks_canonical_connection(self):
        graph = build_structural_gauge_graph(8, topology="cycle", seed=8)
        for node in graph.nodes():
            graph.nodes[node]["EPI"] = ["child_a", "child_b"]
        report = audit_nonabelian_derivability(
            graph,
            routes=("u5_nested_epi_multiplet",),
        )
        candidate = report.candidates[0]
        assert isinstance(candidate, NonAbelianCandidateAudit)
        assert candidate.has_multiplet
        assert candidate.nodal_derivable
        assert not candidate.has_canonical_connection
        assert not candidate.has_noncommuting_generators
        assert candidate.status == (
            "OPEN_MULTIPLET_WITHOUT_CANONICAL_CONNECTION"
        )
        assert report.verdict == "OPEN_DERIVABILITY_GAP"

    def test_cycle_bundle_route_rejects_external_basis_selection(self):
        graph = build_structural_gauge_graph(8, topology="complete", seed=9)
        report = audit_nonabelian_derivability(
            graph,
            routes=("cycle_basis_bundle",),
        )
        candidate = report.candidates[0]
        assert candidate.has_multiplet
        assert candidate.requires_external_labels
        assert not candidate.nodal_derivable
        assert candidate.status == "FAILED_BASIS_DEPENDENT_EXTERNAL_SELECTION"
        assert report.summary["cycle_rank"] > 1

    def test_unknown_route_rejected(self):
        with pytest.raises(ValueError):
            audit_nonabelian_derivability(routes=("su3_by_name",))

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.audit_nonabelian_derivability)
