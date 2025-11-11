"""Unit tests for structural frequency (νf) synchronization in UM operator.

This module tests the canonical TNFR requirement that coupling synchronizes
not only phases but also structural frequencies (νf) between coupled nodes,
as required by the nodal equation: ∂EPI/∂t = νf · ΔNFR(t).
"""

import math
import pytest

from tnfr.operators import apply_glyph
from tnfr.alias import set_vf, get_attr
from tnfr.constants.aliases import ALIAS_VF


def get_vf(G, node_id):
    """Helper to get νf using canonical accessor."""
    return get_attr(G.nodes[node_id], ALIAS_VF, 0.0)


def init_node_with_vf(G, node_id, theta, vf_value):
    """Helper to add node and set νf canonically."""
    G.add_node(node_id, theta=theta, EPI=1.0, Si=0.5)
    set_vf(G, node_id, vf_value)


def test_um_vf_basic_synchronization(graph_canon):
    """Test that UM synchronizes νf between coupled nodes."""
    G = graph_canon()

    # Create nodes and initialize vf using canonical setter
    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 3.0)
    init_node_with_vf(G, 2, 1.0, 5.0)
    G.add_edge(0, 1)
    G.add_edge(0, 2)

    # Enable νf synchronization (default, but explicit for clarity)
    G.graph["UM_SYNC_VF"] = True

    # Store initial frequencies using canonical getter
    vf_0_before = get_vf(G, 0)
    vf_1_before = get_vf(G, 1)
    vf_2_before = get_vf(G, 2)

    # Apply UM operator to node 0
    apply_glyph(G, 0, "UM")

    # Get final frequencies using canonical getter
    vf_0_after = get_vf(G, 0)

    # Node 0 should move toward the mean of its neighbors
    # Mean of neighbors: (3.0 + 5.0) / 2 = 4.0
    expected_mean = (vf_1_before + vf_2_before) / 2

    # With k_vf = 0.10 (default), node 0 should move 10% toward the mean
    # vf_0_new = 1.0 + 0.10 * (4.0 - 1.0) = 1.3
    k_vf = 0.10
    expected_vf_0 = vf_0_before + k_vf * (expected_mean - vf_0_before)

    assert vf_0_after == pytest.approx(expected_vf_0, abs=1e-6)

    # Verify node moved toward the mean
    assert abs(vf_0_after - expected_mean) < abs(vf_0_before - expected_mean)


def test_um_vf_sync_disabled(graph_canon):
    """Test that νf synchronization can be disabled."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 5.0)
    G.add_edge(0, 1)

    # Disable νf synchronization
    G.graph["UM_SYNC_VF"] = False

    vf_0_before = get_vf(G, 0)

    apply_glyph(G, 0, "UM")

    vf_0_after = get_vf(G, 0)

    # νf should remain unchanged when disabled
    assert vf_0_after == pytest.approx(vf_0_before, abs=1e-9)


def test_um_vf_convergence_multiple_iterations(graph_canon):
    """Test that repeated UM applications converge νf values."""
    G = graph_canon()

    # Create nodes with dispersed frequencies
    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 5.0)
    init_node_with_vf(G, 2, 1.0, 9.0)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 2)

    G.graph["UM_SYNC_VF"] = True

    # Record initial frequencies
    initial_vfs = [get_vf(G, i) for i in range(3)]
    initial_mean = sum(initial_vfs) / len(initial_vfs)

    # Apply UM to all nodes multiple times
    for _ in range(20):
        for node_id in range(3):
            apply_glyph(G, node_id, "UM")

    # After many iterations, all frequencies should be close to the initial mean
    final_vfs = [get_vf(G, i) for i in range(3)]

    # All nodes should converge toward the initial mean
    for vf in final_vfs:
        # They should be much closer to the mean than initially
        assert abs(vf - initial_mean) < 2.0  # Tolerance for convergence


def test_um_vf_sync_with_custom_factor(graph_canon):
    """Test νf synchronization respects custom UM_vf_sync factor."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 5.0)
    G.add_edge(0, 1)

    G.graph["UM_SYNC_VF"] = True

    # Test with stronger coupling
    custom_k_vf = 0.5
    G.graph["GLYPH_FACTORS"] = {"UM_vf_sync": custom_k_vf}

    vf_0_before = get_vf(G, 0)
    vf_1_before = get_vf(G, 1)

    apply_glyph(G, 0, "UM")

    vf_0_after = get_vf(G, 0)

    # Compute expected change with custom factor
    vf_mean = vf_1_before  # Only one neighbor
    expected_vf_0 = vf_0_before + custom_k_vf * (vf_mean - vf_0_before)

    assert vf_0_after == pytest.approx(expected_vf_0, abs=1e-6)


def test_um_vf_sync_no_neighbors(graph_canon):
    """Test νf synchronization with no neighbors (edge case)."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 3.0)
    G.graph["UM_SYNC_VF"] = True

    vf_before = get_vf(G, 0)

    apply_glyph(G, 0, "UM")

    vf_after = get_vf(G, 0)

    # νf should remain unchanged when isolated
    assert vf_after == pytest.approx(vf_before, abs=1e-9)


def test_um_vf_sync_single_neighbor(graph_canon):
    """Test νf synchronization with a single neighbor."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 2.0)
    init_node_with_vf(G, 1, 0.5, 8.0)
    G.add_edge(0, 1)

    G.graph["UM_SYNC_VF"] = True

    vf_0_before = get_vf(G, 0)
    vf_1_before = get_vf(G, 1)

    apply_glyph(G, 0, "UM")

    vf_0_after = get_vf(G, 0)

    # With single neighbor, mean = neighbor's vf
    k_vf = 0.10
    expected_vf_0 = vf_0_before + k_vf * (vf_1_before - vf_0_before)

    assert vf_0_after == pytest.approx(expected_vf_0, abs=1e-6)


def test_um_vf_sync_preserves_positive_frequency(graph_canon):
    """Test that νf synchronization maintains positive frequencies."""
    G = graph_canon()

    # Create nodes with various positive frequencies
    init_node_with_vf(G, 0, 0.0, 0.1)
    init_node_with_vf(G, 1, 0.5, 0.5)
    G.add_edge(0, 1)

    G.graph["UM_SYNC_VF"] = True

    apply_glyph(G, 0, "UM")

    # All frequencies should remain positive
    assert get_vf(G, 0) > 0
    assert get_vf(G, 1) > 0


def test_um_vf_sync_reduces_frequency_dispersion(graph_canon):
    """Test that νf synchronization reduces frequency dispersion in network."""
    G = graph_canon()

    # Create a star network with dispersed frequencies
    frequencies = [1.0, 3.0, 5.0, 7.0, 9.0]
    for i, vf in enumerate(frequencies):
        init_node_with_vf(G, i, i * 0.5, vf)
        if i > 0:
            G.add_edge(0, i)

    G.graph["UM_SYNC_VF"] = True

    # Measure initial dispersion (standard deviation)
    def frequency_dispersion(graph, nodes):
        vfs = [get_vf(graph, n) for n in nodes]
        mean = sum(vfs) / len(vfs)
        variance = sum((vf - mean) ** 2 for vf in vfs) / len(vfs)
        return math.sqrt(variance)

    nodes = list(range(len(frequencies)))
    initial_dispersion = frequency_dispersion(G, nodes)

    # Apply UM to central node
    apply_glyph(G, 0, "UM")

    final_dispersion = frequency_dispersion(G, nodes)

    # Dispersion should decrease (frequencies become more similar)
    assert final_dispersion < initial_dispersion


def test_um_vf_sync_with_bidirectional_phase(graph_canon):
    """Test that νf sync works correctly with bidirectional phase sync."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 2.0)
    init_node_with_vf(G, 1, 1.0, 6.0)
    G.add_edge(0, 1)

    # Enable both bidirectional phase and νf sync
    G.graph["UM_BIDIRECTIONAL"] = True
    G.graph["UM_SYNC_VF"] = True

    theta_0_before = G.nodes[0]["theta"]
    theta_1_before = G.nodes[1]["theta"]
    vf_0_before = get_vf(G, 0)

    apply_glyph(G, 0, "UM")

    theta_0_after = G.nodes[0]["theta"]
    theta_1_after = G.nodes[1]["theta"]
    vf_0_after = get_vf(G, 0)

    # Both phase and frequency should change
    assert theta_0_after != theta_0_before
    assert theta_1_after != theta_1_before  # Bidirectional
    assert vf_0_after != vf_0_before


def test_um_vf_sync_default_enabled(graph_canon):
    """Test that νf synchronization is enabled by default."""
    G = graph_canon()

    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 5.0)
    G.add_edge(0, 1)

    # Don't set UM_SYNC_VF - should default to True
    vf_0_before = get_vf(G, 0)

    apply_glyph(G, 0, "UM")

    vf_0_after = get_vf(G, 0)

    # νf should have changed (confirming sync is default)
    assert vf_0_after != vf_0_before


def test_um_vf_sync_asymmetric_network(graph_canon):
    """Test νf synchronization in asymmetric network topology."""
    G = graph_canon()

    # Create a chain: 0 -- 1 -- 2
    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 5.0)
    init_node_with_vf(G, 2, 1.0, 9.0)
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    G.graph["UM_SYNC_VF"] = True

    # Apply UM to middle node
    vf_1_before = get_vf(G, 1)

    apply_glyph(G, 1, "UM")

    vf_1_after = get_vf(G, 1)

    # Middle node should move toward mean of its neighbors
    vf_mean = (get_vf(G, 0) + get_vf(G, 2)) / 2
    k_vf = 0.10
    expected_vf_1 = vf_1_before + k_vf * (vf_mean - vf_1_before)

    assert vf_1_after == pytest.approx(expected_vf_1, abs=1e-6)


def test_um_vf_sync_impact_on_nodal_equation(graph_canon):
    """Test that νf synchronization affects the nodal equation evolution.

    The nodal equation ∂EPI/∂t = νf · ΔNFR(t) shows that νf directly
    influences structural evolution. This test verifies that synchronized
    νf values lead to more coherent network evolution.
    """
    G = graph_canon()

    # Create two nodes with different frequencies
    init_node_with_vf(G, 0, 0.0, 1.0)
    init_node_with_vf(G, 1, 0.5, 9.0)
    G.add_edge(0, 1)

    G.graph["UM_SYNC_VF"] = True

    vf_0_initial = get_vf(G, 0)
    vf_1_initial = get_vf(G, 1)

    # Large initial disparity
    initial_vf_diff = abs(vf_1_initial - vf_0_initial)

    # Apply UM multiple times
    for _ in range(10):
        apply_glyph(G, 0, "UM")

    vf_0_final = get_vf(G, 0)
    vf_1_final = get_vf(G, 1)

    final_vf_diff = abs(vf_1_final - vf_0_final)

    # Frequency difference should decrease significantly
    assert final_vf_diff < initial_vf_diff

    # Both frequencies should be closer to their initial mean
    initial_mean = (vf_0_initial + vf_1_initial) / 2
    assert abs(vf_0_final - initial_mean) < abs(vf_0_initial - initial_mean)
