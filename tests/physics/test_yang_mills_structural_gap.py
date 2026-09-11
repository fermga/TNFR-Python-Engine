"""Tests for TNFR–Yang–Mills Y1 finite structural gauge gap diagnostics.

These tests validate the finite-graph diagnostic surface documented in
``theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md``. The diagnostic is read-only with
respect to EPI and uses canonical fields (Ψ, Φ_s) plus the auxiliary pure-gauge
connection A=d(arg Ψ) and its numerical cycle-closure residual F.
"""

from __future__ import annotations

import math
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
from tnfr.operators import validate_affine_epi_graph_input  # noqa: E402
from tnfr.types import Glyph, scalarize_epi  # noqa: E402


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
            assert graph.nodes[node]["EPI"] == 0.0
            assert math.isfinite(scalarize_epi(graph.nodes[node]["EPI"]))
            validate_affine_epi_graph_input(graph, node, Glyph.AL)
        assert graph.graph["tnfr_program"] == "TNFR-Yang-Mills-Y1"
        assert graph.graph["seed"] == 7

    def test_scalar_epi_initialization_does_not_shift_seeded_structural_draws(self):
        graph = build_structural_gauge_graph(4, topology="cycle", seed=7)

        observed = [
            (
                graph.nodes[node]["phase"],
                graph.nodes[node]["frequency"],
                graph.nodes[node]["delta_nfr"],
            )
            for node in graph
        ]
        expected = [
            (-2.3158732757276175, 0.8205485521961549, -0.043966849601505306),
            (-2.375578027333452, 0.8988427563170096, -0.07915755126950805),
            (-2.3234718139862984, 0.837655543001637, -0.00513040754500467),
            (-2.3752914131426444, 0.42274048968061867, -0.039220865975340066),
        ]

        assert observed == pytest.approx(expected, abs=1e-15, rel=0.0)

    def test_invalid_topology_rejected(self):
        with pytest.raises(ValueError):
            build_structural_gauge_graph(8, topology="external_group_label")


class TestStructuralGaugeGapOperator:
    """Finite structural gauge diagnostic contracts."""

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
        assert op.metadata["operator"] == "H_structural = L_A + V_F + V_Phi"
        assert op.metadata["scope"] == "finite_graph_y1_diagnostic_not_clay_proof"
        assert op.metadata["n_nodes"] == graph.number_of_nodes()
        assert op.metadata["yang_mills_action"] >= 0.0
        assert op.metadata["gauge_coupling_constant"] >= 0.0
        assert not op.metadata["u6_drift_assessed"]
        assert op.metadata["u6_reference_required"]
        assert op.metadata["legacy_u6_fields_are_magnitude_proxies"]
        assert op.metadata["u6_aggregation"] == "mean_absolute_nodewise_drift"
        assert op.metadata["u6_comparison"] == "strict_less_than"
        assert op.metadata["u6_drift_threshold"] == pytest.approx(math.pi / 2.0)
        assert op.metadata["potential_magnitude_warning_threshold"] == pytest.approx(
            math.pi / 4.0
        )
        assert op.metadata["legacy_u6_normalization_scale"] == pytest.approx(
            math.pi / 2.0
        )
        assert op.potential_magnitude_penalty is op.confinement_potential
        assert op.metadata["grammar_rules_total"] == 1
        assert set(op.metadata["grammar_rules_unassessed"]) == {
            "U1",
            "U2",
            "U4",
            "U5",
            "U6",
        }

    def test_negative_weights_rejected(self):
        graph = build_structural_gauge_graph(8, topology="cycle", seed=42)
        with pytest.raises(ValueError):
            build_structural_gauge_gap_operator(graph, curvature_weight=-1.0)

    def test_canonical_connection_is_flat_by_construction(self):
        graph = build_structural_gauge_graph(10, topology="complete", seed=42)
        op = build_structural_gauge_gap_operator(graph)

        assert op.metadata["canonical_connection_flat_by_construction"]
        assert op.metadata["canonical_connection_is_pure_gauge"]
        assert op.metadata["curvature_is_numerical_residual"]
        assert op.metadata["curvature_potential_zero_within_tolerance"]
        assert op.metadata["max_abs_curvature_residual"] < 1e-12
        assert max(op.curvature_potential.values(), default=0.0) == 0.0

    def test_zero_potential_operator_is_unitarily_equivalent_to_graph_laplacian(self):
        import networkx as nx

        graph = build_structural_gauge_graph(9, topology="cycle", seed=19)
        for node in graph:
            graph.nodes[node]["delta_nfr"] = 0.0
        op = build_structural_gauge_gap_operator(graph)
        ordinary = nx.laplacian_matrix(
            graph,
            nodelist=list(op.node_order),
            weight="weight",
        ).toarray()

        assert np.allclose(
            np.linalg.eigvalsh(op.matrix),
            np.linalg.eigvalsh(ordinary),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_non_gauge_equivalent_supplied_connection_is_rejected(self):
        graph = build_structural_gauge_graph(3, topology="complete", seed=42)
        connection = {(0, 1): 0.0, (1, 2): 0.0, (2, 0): 0.5}

        with pytest.raises(ValueError, match="gauge-equivalent"):
            build_structural_gauge_gap_operator(graph, connection=connection)

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"n": 3.5}, ValueError),
            ({"n": True}, ValueError),
            ({"phase_spread": math.nan}, ValueError),
            ({"phase_spread": math.pi}, ValueError),
            ({"delta_nfr_scale": math.inf}, ValueError),
            ({"n": 2, "topology": "watts_strogatz"}, ValueError),
        ],
    )
    def test_graph_builder_rejects_nonfinite_or_coerced_inputs(self, kwargs, error):
        with pytest.raises(error):
            build_structural_gauge_graph(**kwargs)

    def test_directed_and_multigraph_inputs_are_rejected(self):
        import networkx as nx

        directed = nx.DiGraph([(0, 1), (1, 0)])
        multi = nx.MultiGraph([(0, 1)])
        for graph in (directed, multi):
            with pytest.raises(ValueError, match="undirected simple graph"):
                build_structural_gauge_gap_operator(graph)


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

    def test_signed_seeds_are_reproducible(self):
        g1 = build_structural_gauge_graph(8, topology="cycle", seed=-7)
        g2 = build_structural_gauge_graph(8, topology="cycle", seed=-7)
        r1 = compute_structural_gauge_gap(g1, gauge_seed=-11)
        r2 = compute_structural_gauge_gap(g2, gauge_seed=-11)

        assert np.allclose(r1.eigenvalues, r2.eigenvalues, atol=1e-14)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tolerance": 0.0},
            {"tolerance": math.nan},
            {"eigen_tolerance": -1.0},
            {"curvature_weight": math.inf},
            {"confinement_weight": math.nan},
        ],
    )
    def test_diagnostic_rejects_invalid_numeric_controls(self, kwargs):
        graph = build_structural_gauge_graph(8, topology="cycle", seed=2)
        with pytest.raises((TypeError, ValueError)):
            compute_structural_gauge_gap(graph, **kwargs)

    def test_diagnostic_does_not_mutate_epi(self):
        graph = build_structural_gauge_graph(10, topology="complete", seed=91)
        epi_before = {node: graph.nodes[node]["EPI"] for node in graph.nodes()}
        phase_before = {node: graph.nodes[node]["phase"] for node in graph.nodes()}
        _ = compute_structural_gauge_gap(graph, gauge_seed=91)
        epi_after = {node: graph.nodes[node]["EPI"] for node in graph.nodes()}
        phase_after = {node: graph.nodes[node]["phase"] for node in graph.nodes()}
        assert epi_after == epi_before
        assert phase_after == phase_before

    def test_diagnostic_spectrum_is_independent_of_epi_representation(self):
        graph = build_structural_gauge_graph(10, topology="complete", seed=91)
        scalar_result = compute_structural_gauge_gap(graph, gauge_seed=91)
        for index, node in enumerate(graph):
            graph.nodes[node]["EPI"] = f"legacy_diagnostic_label_{index}"

        labelled_result = compute_structural_gauge_gap(graph, gauge_seed=91)

        assert np.array_equal(scalar_result.eigenvalues, labelled_result.eigenvalues)
        assert scalar_result.gap == labelled_result.gap

    def test_imports_from_package_root(self):
        from tnfr import yang_mills

        assert callable(yang_mills.compute_structural_gauge_gap)
        assert callable(yang_mills.build_structural_gauge_graph)
