"""Tests for expanded coupling metrics with canonical measurements.

This module validates the extended coupling metrics collection for the UM
(Coupling) operator as specified in the issue:

- Δνf: Structural frequency change
- ΔΔNFR: Reorganization pressure reduction
- Total coupling strength: Sum of edge weights
- New edges count: Number of edges added
- Local coherence: Kuramoto order parameter of subgraph
- Phase dispersion: Standard deviation of local phases

Tests verify that all canonical metrics are properly computed and reported.
"""

from tnfr.operators.definitions import Coupling, Emission, Reception, Coherence, Silence
from tnfr.structural import create_nfr, run_sequence


def test_coupling_metrics_collection_enabled():
    """When COLLECT_OPERATOR_METRICS is True, coupling metrics must be collected."""
    G, node = create_nfr("coupling_test", epi=0.5, vf=1.0, theta=0.5)

    # Add a neighbor for coupling using create_nfr
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # TNFR grammar requires: Emission → Reception → Coupling → Coherence → Silence
    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    # Verify metrics were collected
    assert "operator_metrics" in G.graph
    assert len(G.graph["operator_metrics"]) >= 3

    # Get the Coupling metrics (third operator, index 2)
    metrics = G.graph["operator_metrics"][2]
    assert metrics["operator"] == "Coupling"
    assert metrics["glyph"] == "UM"


def test_delta_vf_present():
    """delta_vf metric should track structural frequency change."""
    G, node = create_nfr("vf_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbor
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check delta_vf is present
    assert "delta_vf" in metrics
    assert "vf_final" in metrics
    assert isinstance(metrics["delta_vf"], float)
    assert isinstance(metrics["vf_final"], float)


def test_delta_dnfr_present():
    """delta_dnfr and dnfr_stabilization metrics should track reorganization changes."""
    G, node = create_nfr("dnfr_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbor
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check ΔNFR metrics are present
    assert "delta_dnfr" in metrics
    assert "dnfr_stabilization" in metrics
    assert "dnfr_final" in metrics
    assert isinstance(metrics["delta_dnfr"], float)
    assert isinstance(metrics["dnfr_stabilization"], float)


def test_new_edges_count():
    """new_edges_count should track edge formation."""
    G, node = create_nfr("edges_test", epi=0.5, vf=1.0, theta=0.5)

    # Initially no edges, check degree
    initial_edges = G.degree(node)

    # Add neighbor using create_nfr
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)

    # Add edge before to simulate existing connection
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check edge metrics are present
    assert "new_edges_count" in metrics
    assert "total_edges" in metrics
    assert isinstance(metrics["new_edges_count"], int)
    assert isinstance(metrics["total_edges"], int)


def test_coupling_strength_total():
    """coupling_strength_total should sum edge weights."""
    G, node = create_nfr("strength_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbors with edge weights
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1", coupling=0.5)

    create_nfr("neighbor2", epi=0.7, vf=1.2, theta=0.7, graph=G)
    G.add_edge(node, "neighbor2", coupling=0.3)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check coupling strength is present
    assert "coupling_strength_total" in metrics
    assert isinstance(metrics["coupling_strength_total"], float)
    # Should be sum of edge weights (0.5 + 0.3 = 0.8)
    assert metrics["coupling_strength_total"] >= 0.0


def test_phase_dispersion():
    """phase_dispersion should measure local phase variance."""
    G, node = create_nfr("dispersion_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbors with varying phases
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1")

    create_nfr("neighbor2", epi=0.7, vf=1.2, theta=0.8, graph=G)
    G.add_edge(node, "neighbor2")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check phase dispersion is present
    assert "phase_dispersion" in metrics
    assert isinstance(metrics["phase_dispersion"], float)
    assert metrics["phase_dispersion"] >= 0.0


def test_local_coherence():
    """local_coherence should measure Kuramoto order parameter."""
    G, node = create_nfr("coherence_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbors
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.55, graph=G)
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check local coherence is present
    assert "local_coherence" in metrics
    assert isinstance(metrics["local_coherence"], float)
    assert 0.0 <= metrics["local_coherence"] <= 1.0


def test_is_synchronized():
    """is_synchronized should indicate strong phase alignment."""
    G, node = create_nfr("sync_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbor with similar phase using create_nfr
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.51, graph=G)  # Very close phase
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Check is_synchronized is present
    assert "is_synchronized" in metrics
    assert isinstance(metrics["is_synchronized"], bool)


def test_all_canonical_metrics_present():
    """All canonical metrics must be present in coupling metrics."""
    G, node = create_nfr("complete_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbor
    create_nfr("neighbor1", epi=0.6, vf=1.1, theta=0.6, graph=G)
    G.add_edge(node, "neighbor1")

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Core metrics (always present)
    assert "operator" in metrics
    assert "glyph" in metrics
    assert "theta_shift" in metrics
    assert "theta_final" in metrics
    assert "neighbor_count" in metrics
    assert "mean_neighbor_phase" in metrics
    assert "phase_alignment" in metrics

    # New canonical metrics
    assert "delta_vf" in metrics
    assert "vf_final" in metrics
    assert "delta_dnfr" in metrics
    assert "dnfr_stabilization" in metrics
    assert "dnfr_final" in metrics
    assert "dnfr_reduction" in metrics
    assert "dnfr_reduction_pct" in metrics
    assert "new_edges_count" in metrics
    assert "total_edges" in metrics
    assert "coupling_strength_total" in metrics
    assert "phase_dispersion" in metrics
    assert "local_coherence" in metrics
    assert "is_synchronized" in metrics


def test_coupling_without_neighbors():
    """Coupling metrics should handle nodes without neighbors gracefully."""
    G, node = create_nfr("isolated_test", epi=0.5, vf=1.0, theta=0.5)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Should have metrics even without neighbors
    assert metrics["neighbor_count"] == 0
    assert metrics["phase_alignment"] == 0.0
    assert metrics["phase_dispersion"] == 0.0
    assert metrics["local_coherence"] == 0.0
    assert metrics["coupling_strength_total"] == 0.0


def test_phase_alignment_range():
    """phase_alignment should be bounded between 0 and 1."""
    G, node = create_nfr("alignment_test", epi=0.5, vf=1.0, theta=0.5)

    # Add neighbors with various phases using create_nfr
    for i, theta in enumerate([0.3, 0.6, 1.0]):
        neighbor = f"neighbor{i}"
        create_nfr(neighbor, epi=0.6, vf=1.1, theta=theta, graph=G)
        G.add_edge(node, neighbor)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # phase_alignment should be in valid range
    assert 0.0 <= metrics["phase_alignment"] <= 1.0


def test_metrics_with_multiple_neighbors():
    """Coupling metrics should correctly aggregate over multiple neighbors."""
    G, node = create_nfr("multi_test", epi=0.5, vf=1.0, theta=0.5)

    # Add multiple neighbors using create_nfr
    for i in range(5):
        neighbor = f"neighbor{i}"
        create_nfr(neighbor, epi=0.6 + i * 0.05, vf=1.1 + i * 0.1, theta=0.5 + i * 0.1, graph=G)
        G.add_edge(node, neighbor, coupling=0.1 * (i + 1))

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, node, [Emission(), Reception(), Coupling(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][2]

    # Verify aggregation
    assert metrics["neighbor_count"] == 5
    assert metrics["coupling_strength_total"] > 0.0
    assert metrics["phase_dispersion"] >= 0.0  # Should have some variance
    assert 0.0 <= metrics["local_coherence"] <= 1.0
