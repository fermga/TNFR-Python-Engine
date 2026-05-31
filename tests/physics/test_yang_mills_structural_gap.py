"""Tests for TNFR–Yang–Mills Y1 finite structural gauge gap diagnostics.

These tests validate the TNFR-native finite-graph attack surface documented in
``theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md``.  The diagnostic is read-only with
respect to EPI and derives its operator from canonical gauge telemetry
(Ψ, A, F, Φ_s), not from an external quantum ontology.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.yang_mills import (  # noqa: E402
    StructuralGaugeGapOperator,
    StructuralGaugeGapResult,
    build_structural_gauge_gap_operator,
    build_structural_gauge_graph,
    compute_structural_gauge_gap,
)


class TestStructuralGaugeGraph:
    """Y1 graph construction creates TNFR-ready structural gauge graphs."""

    @pytest.mark.parametrize(
        "topology",
        ["cycle", "complete", "watts_strogatz", "grid"],
    )
    def test_graph_has_canonical_attributes(self, topology):
        graph = build_structural_gauge_graph(12, topology=topology, seed=7)
        assert graph.number_of_nodes() >= 4 if topology == "grid" else 12
        for node in graph.nodes():
            assert "phase" in graph.nodes[node]
            assert "frequency" in graph.nodes[node]
            assert "delta_nfr" in graph.nodes[node]
            assert "EPI" in graph.nodes[node]
            assert graph.nodes[node]["frequency"] > 0.0
        assert graph.graph["tnfr_program"] == "TNFR-Yang-Mills-Y1"
        assert graph.graph["seed"] == 7

    def test_invalid_topology_rejected(self):
        with pytest.raises(ValueError):
            build_structural_gauge_graph(8, topology="external_group_label")


class TestStructuralGaugeGapOperator:
    """Finite H_YM^TNFR operator contracts."""

    def test_operator_is_dataclass_and_hermitian(self):
        graph = build_structural_gauge_graph(10, topology="complete", seed=42)
        op = build_structural_gauge_gap_operator(graph)
        assert isinstance(op, StructuralGaugeGapOperator)
        assert op.matrix.shape == (
            graph.number_of_nodes(),
            graph.number_of_nodes(),
        )
        defect = np.max(np.abs(op.matrix - op.matrix.conjugate().T))
        assert defect < 1e-12

    def test_operator_metadata_records_tnfr_scope(self):
        graph = build_structural_gauge_graph(10, topology="cycle", seed=42)
        op = build_structural_gauge_gap_operator(graph)
        assert op.metadata["operator"] == "H_YM_TNFR = L_A + V_F + V_U6"
        assert (
            op.metadata["scope"]
            == "finite_graph_y1_diagnostic_not_clay_proof"
        )
        assert op.metadata["n_nodes"] == graph.number_of_nodes()
        assert op.metadata["yang_mills_action"] >= 0.0
        assert op.metadata["gauge_coupling_constant"] >= 0.0

    def test_negative_weights_rejected(self):
        graph = build_structural_gauge_graph(8, topology="cycle", seed=42)
        with pytest.raises(ValueError):
            build_structural_gauge_gap_operator(graph, curvature_weight=-1.0)


class TestStructuralGaugeGapDiagnostic:
    """Y1 diagnostic report contracts."""

    def test_gap_result_positive_and_self_adjoint(self):
        graph = build_structural_gauge_graph(12, topology="complete", seed=11)
        result = compute_structural_gauge_gap(graph, gauge_seed=99)
        assert isinstance(result, StructuralGaugeGapResult)
        assert result.is_self_adjoint
        assert result.self_adjoint_deviation < 1e-12
        assert result.lambda1 >= result.lambda0
        assert result.gap >= 0.0
        assert result.verdict in {
            "FINITE_POSITIVE_STRUCTURAL_GAP",
            "FINITE_GAP_NOT_RESOLVED",
        }

    def test_spectrum_is_gauge_invariant(self):
        graph = build_structural_gauge_graph(12, topology="cycle", seed=12)
        result = compute_structural_gauge_gap(graph, gauge_seed=123)
        assert result.gauge_invariant
        assert result.gauge_spectral_deviation < 1e-9
        assert np.allclose(
            result.eigenvalues,
            result.transformed_eigenvalues,
            atol=1e-9,
            rtol=1e-9,
        )

    def test_diagnostic_is_reproducible(self):
        g1 = build_structural_gauge_graph(
            14,
            topology="watts_strogatz",
            seed=44,
        )
        g2 = build_structural_gauge_graph(
            14,
            topology="watts_strogatz",
            seed=44,
        )
        r1 = compute_structural_gauge_gap(g1, gauge_seed=55)
        r2 = compute_structural_gauge_gap(g2, gauge_seed=55)
        assert r1.gap == pytest.approx(r2.gap, abs=1e-14)
        assert r1.gauge_spectral_deviation == pytest.approx(
            r2.gauge_spectral_deviation,
            abs=1e-14,
        )
        assert np.allclose(r1.eigenvalues, r2.eigenvalues, atol=1e-14)

    def test_diagnostic_does_not_mutate_epi(self):
        graph = build_structural_gauge_graph(10, topology="complete", seed=91)
        epi_before = {node: graph.nodes[node]["EPI"] for node in graph.nodes()}
        phase_before = {
            node: graph.nodes[node]["phase"] for node in graph.nodes()
        }
        _ = compute_structural_gauge_gap(graph, gauge_seed=91)
        epi_after = {node: graph.nodes[node]["EPI"] for node in graph.nodes()}
        phase_after = {
            node: graph.nodes[node]["phase"] for node in graph.nodes()
        }
        assert epi_after == epi_before
        assert phase_after == phase_before

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.compute_structural_gauge_gap)
        assert callable(yang_mills.build_structural_gauge_graph)
