"""Unit tests for bidirectional phase synchronization in UM operator.

This module tests the canonical TNFR requirement that coupling should implement
bidirectional phase synchronization: φᵢ(t) ≈ φⱼ(t) for coupled nodes.
"""

import math

import pytest

from tnfr.operators import apply_glyph
from tnfr.structural import create_nfr
from tnfr.utils import angle_diff


def test_um_bidirectional_basic_synchronization(graph_canon):
    """Test that bidirectional UM mutually adjusts node and neighbor phases."""
    G = graph_canon()

    # Create three nodes with different phases
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=math.pi / 2, EPI=1.0, Si=0.5)
    G.add_node(2, theta=math.pi, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)

    # Enable bidirectional mode (default, but explicit for clarity)
    G.graph["UM_BIDIRECTIONAL"] = True

    # Store initial phases
    theta_0_before = G.nodes[0]["theta"]
    theta_1_before = G.nodes[1]["theta"]
    theta_2_before = G.nodes[2]["theta"]

    # Apply UM operator to node 0
    apply_glyph(G, 0, "UM")

    # Get final phases
    theta_0_after = G.nodes[0]["theta"]
    theta_1_after = G.nodes[1]["theta"]
    theta_2_after = G.nodes[2]["theta"]

    # In bidirectional mode:
    # 1. Node 0 should move toward consensus
    # 2. Neighbors 1 and 2 should also move toward consensus
    # 3. All three should be closer to each other than before

    # Node 0 should increase (moving toward higher phases)
    assert theta_0_after > theta_0_before

    # Node 1 should stay approximately the same or change
    # (it's already close to the middle)

    # Node 2 should decrease (moving down from π)
    assert theta_2_after < theta_2_before

    # Verify that phases are now more synchronized
    initial_spread = max(theta_0_before, theta_1_before, theta_2_before) - min(
        theta_0_before, theta_1_before, theta_2_before
    )
    final_spread = max(theta_0_after, theta_1_after, theta_2_after) - min(
        theta_0_after, theta_1_after, theta_2_after
    )

    assert final_spread < initial_spread, "Phases should be more synchronized"


def test_um_bidirectional_vs_unidirectional(graph_canon):
    """Compare bidirectional and unidirectional UM behaviors."""
    G = graph_canon()

    # Create identical initial configurations
    # Use phases where none is exactly at the consensus
    initial_theta_node = 0.0
    initial_theta_neighbor1 = math.pi / 2
    initial_theta_neighbor2 = 2 * math.pi / 3

    # Test bidirectional mode
    G.add_node(0, theta=initial_theta_node, EPI=1.0, Si=0.5)
    G.add_node(1, theta=initial_theta_neighbor1, EPI=1.0, Si=0.5)
    G.add_node(2, theta=initial_theta_neighbor2, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_BIDIRECTIONAL"] = True

    apply_glyph(G, 0, "UM")

    theta_0_bidirectional = G.nodes[0]["theta"]
    theta_1_bidirectional = G.nodes[1]["theta"]
    theta_2_bidirectional = G.nodes[2]["theta"]

    # Reset graph for unidirectional test
    G.nodes[0]["theta"] = initial_theta_node
    G.nodes[1]["theta"] = initial_theta_neighbor1
    G.nodes[2]["theta"] = initial_theta_neighbor2
    G.graph["UM_BIDIRECTIONAL"] = False

    apply_glyph(G, 0, "UM")

    theta_0_unidirectional = G.nodes[0]["theta"]
    theta_1_unidirectional = G.nodes[1]["theta"]
    theta_2_unidirectional = G.nodes[2]["theta"]

    # Verify differences:
    # 1. Bidirectional: all three nodes should change
    assert theta_0_bidirectional != initial_theta_node
    assert theta_1_bidirectional != initial_theta_neighbor1
    assert theta_2_bidirectional != initial_theta_neighbor2

    # 2. Unidirectional: only node 0 should change
    assert theta_0_unidirectional != initial_theta_node
    assert theta_1_unidirectional == initial_theta_neighbor1
    assert theta_2_unidirectional == initial_theta_neighbor2

    # 3. Node 0 should move differently in the two modes
    assert theta_0_bidirectional != theta_0_unidirectional


def test_um_bidirectional_preserves_consensus_phase(graph_canon):
    """Test that bidirectional UM moves all nodes toward consensus phase."""
    G = graph_canon()

    # Create nodes with known phases
    theta_0 = 0.0
    theta_1 = math.pi / 4
    theta_2 = math.pi / 2

    G.add_node(0, theta=theta_0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=theta_1, EPI=1.0, Si=0.5)
    G.add_node(2, theta=theta_2, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_BIDIRECTIONAL"] = True

    # Compute expected consensus phase (circular mean)
    cos_sum = math.cos(theta_0) + math.cos(theta_1) + math.cos(theta_2)
    sin_sum = math.sin(theta_0) + math.sin(theta_1) + math.sin(theta_2)
    consensus_phase = math.atan2(sin_sum, cos_sum)

    # Apply UM
    apply_glyph(G, 0, "UM")

    # All nodes should move toward consensus
    k = 0.25  # Default UM_theta_push

    expected_theta_0 = theta_0 + k * angle_diff(consensus_phase, theta_0)
    expected_theta_1 = theta_1 + k * angle_diff(consensus_phase, theta_1)
    expected_theta_2 = theta_2 + k * angle_diff(consensus_phase, theta_2)

    assert G.nodes[0]["theta"] == pytest.approx(expected_theta_0, abs=1e-6)
    assert G.nodes[1]["theta"] == pytest.approx(expected_theta_1, abs=1e-6)
    assert G.nodes[2]["theta"] == pytest.approx(expected_theta_2, abs=1e-6)


def test_um_bidirectional_wraps_at_boundaries(graph_canon):
    """Test bidirectional UM correctly handles phase wrapping at ±π."""
    G = graph_canon()

    # Create nodes near the ±π boundary
    theta_0 = -math.pi + 0.1
    theta_1 = math.pi - 0.1
    theta_2 = 0.0

    G.add_node(0, theta=theta_0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=theta_1, EPI=1.0, Si=0.5)
    G.add_node(2, theta=theta_2, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_BIDIRECTIONAL"] = True

    apply_glyph(G, 0, "UM")

    # All phases should remain in [-π, π)
    assert -math.pi <= G.nodes[0]["theta"] < math.pi
    assert -math.pi <= G.nodes[1]["theta"] < math.pi
    assert -math.pi <= G.nodes[2]["theta"] < math.pi


def test_um_bidirectional_with_custom_push_factor(graph_canon):
    """Test bidirectional UM respects custom UM_theta_push factor."""
    G = graph_canon()

    theta_0 = 0.0
    theta_1 = math.pi / 2

    G.add_node(0, theta=theta_0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=theta_1, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.graph["UM_BIDIRECTIONAL"] = True

    # Test with stronger coupling (higher k)
    custom_k = 0.5
    G.graph["GLYPH_FACTORS"] = {"UM_theta_push": custom_k}

    # Compute consensus
    consensus = math.atan2(
        math.sin(theta_0) + math.sin(theta_1),
        math.cos(theta_0) + math.cos(theta_1),
    )

    apply_glyph(G, 0, "UM")

    expected_theta_0 = theta_0 + custom_k * angle_diff(consensus, theta_0)
    expected_theta_1 = theta_1 + custom_k * angle_diff(consensus, theta_1)

    assert G.nodes[0]["theta"] == pytest.approx(expected_theta_0, abs=1e-6)
    assert G.nodes[1]["theta"] == pytest.approx(expected_theta_1, abs=1e-6)


def test_um_bidirectional_single_neighbor(graph_canon):
    """Test bidirectional UM with a single neighbor."""
    G = graph_canon()

    theta_0 = 0.0
    theta_1 = math.pi / 2

    G.add_node(0, theta=theta_0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=theta_1, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.graph["UM_BIDIRECTIONAL"] = True

    apply_glyph(G, 0, "UM")

    # With two nodes, consensus is their circular mean
    consensus = math.atan2(
        math.sin(theta_0) + math.sin(theta_1),
        math.cos(theta_0) + math.cos(theta_1),
    )

    k = 0.25
    expected_theta_0 = theta_0 + k * angle_diff(consensus, theta_0)
    expected_theta_1 = theta_1 + k * angle_diff(consensus, theta_1)

    assert G.nodes[0]["theta"] == pytest.approx(expected_theta_0, abs=1e-6)
    assert G.nodes[1]["theta"] == pytest.approx(expected_theta_1, abs=1e-6)


def test_um_bidirectional_no_neighbors(graph_canon):
    """Test bidirectional UM with no neighbors (edge case)."""
    G = graph_canon()

    theta_0 = math.pi / 4

    G.add_node(0, theta=theta_0, EPI=1.0, Si=0.5)
    G.graph["UM_BIDIRECTIONAL"] = True

    # Apply UM to isolated node
    apply_glyph(G, 0, "UM")

    # Phase should remain unchanged
    assert G.nodes[0]["theta"] == pytest.approx(theta_0, abs=1e-9)


def test_um_bidirectional_improves_phase_coherence(graph_canon):
    """Test that bidirectional UM reduces phase dispersion (improves coherence)."""
    G = graph_canon()

    # Create a network with dispersed phases
    phases = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]
    for i, theta in enumerate(phases):
        G.add_node(i, theta=theta, EPI=1.0, Si=0.5)
        if i > 0:
            G.add_edge(0, i)

    G.graph["UM_BIDIRECTIONAL"] = True

    # Measure initial phase dispersion (standard deviation of angles)
    def phase_dispersion(graph, nodes):
        """Compute circular variance as a measure of phase dispersion."""
        cos_sum = sum(math.cos(graph.nodes[n]["theta"]) for n in nodes)
        sin_sum = sum(math.sin(graph.nodes[n]["theta"]) for n in nodes)
        r = math.sqrt(cos_sum**2 + sin_sum**2) / len(nodes)
        return 1 - r  # Circular variance (0 = perfect coherence, 1 = max dispersion)

    nodes = list(range(len(phases)))
    initial_dispersion = phase_dispersion(G, nodes)

    # Apply UM
    apply_glyph(G, 0, "UM")

    final_dispersion = phase_dispersion(G, nodes)

    # Phase dispersion should decrease (coherence improves)
    assert final_dispersion < initial_dispersion


def test_um_bidirectional_multiple_applications_converge(graph_canon):
    """Test that repeated bidirectional UM applications converge to consensus."""
    G = graph_canon()

    # Create nodes with dispersed phases
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=math.pi / 2, EPI=1.0, Si=0.5)
    G.add_node(2, theta=math.pi, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_BIDIRECTIONAL"] = True

    # Compute initial consensus
    initial_phases = [G.nodes[i]["theta"] for i in range(3)]
    initial_consensus = math.atan2(
        sum(math.sin(p) for p in initial_phases),
        sum(math.cos(p) for p in initial_phases),
    )

    # Apply UM multiple times
    for _ in range(10):
        apply_glyph(G, 0, "UM")

    # All phases should be very close to the initial consensus
    final_phases = [G.nodes[i]["theta"] for i in range(3)]

    for phase in final_phases:
        # Phases should converge toward consensus (within reasonable tolerance)
        # Note: They won't reach exactly the same value unless k=1.0
        diff = abs(angle_diff(phase, initial_consensus))
        # After 10 iterations with k=0.25, we expect significant convergence
        assert diff < 0.5  # Should be closer than initial dispersion


def test_um_bidirectional_default_behavior(graph_canon):
    """Test that bidirectional mode is enabled by default."""
    G = graph_canon()

    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5)
    G.add_node(1, theta=math.pi / 2, EPI=1.0, Si=0.5)
    G.add_edge(0, 1)

    # Don't set UM_BIDIRECTIONAL - should default to True
    theta_1_before = G.nodes[1]["theta"]

    apply_glyph(G, 0, "UM")

    theta_1_after = G.nodes[1]["theta"]

    # Neighbor should have moved (confirming bidirectional is default)
    assert theta_1_after != theta_1_before
