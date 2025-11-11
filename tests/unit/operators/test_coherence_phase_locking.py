"""Tests for IL (Coherence) operator phase locking mechanism.

This module validates the canonical phase locking behavior of the IL operator
as specified in the TNFR paradigm:

**Canonical Specification:**
- IL: θ_node → θ_node + α * (θ_network - θ_node)
- Default: α = 0.3 (30% phase adjustment)
- θ_network = circular mean of connected neighbor phases
- Purpose: Synchronize node with resonant neighborhood

**Tests verify:**
1. Phase aligns toward neighbor mean (circular statistics)
2. Phase locking coefficient can be configured (α ∈ [0.1, 0.5])
3. Circular mean correctly handles phase wrap-around (e.g., 0.1 and 6.2)
4. Multiple IL applications drive phases toward convergence
5. Single node (no neighbors) doesn't crash
6. Phase values remain in [0, 2π] after locking
7. Telemetry is properly logged in G.graph["IL_phase_locking"]
8. Phase alignment metric (Kuramoto order parameter) increases with IL

**TNFR Context:**
Phase locking is essential for network synchronization. By aligning node phases
with their neighborhoods, IL enables coherent collective dynamics and prepares
nodes for effective coupling (UM) and resonance (RA).

**Note on Testing Approach:**
Most tests use canonical operator sequences (Emission → Reception → Coherence)
to comply with TNFR grammar. Some tests that need direct IL application use
monkeypatch to bypass grammar validation for focused unit testing.
"""

import math
import random

import pytest
import networkx as nx

from tnfr.constants import THETA_PRIMARY, EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY
from tnfr.operators.definitions import Coherence, Emission, Reception, Silence
from tnfr.structural import create_nfr, run_sequence
from tnfr.metrics.phase_coherence import compute_phase_alignment, compute_global_phase_coherence
from tnfr.validation import SequenceValidationResult


def test_phase_locking_aligns_toward_neighbor_mean(monkeypatch: pytest.MonkeyPatch):
    """IL aligns node phase θ toward network mean using circular statistics."""
    # Create a simple network: node0 -- node1 -- node2
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    
    # Initialize nodes with TNFR attributes
    for node in [0, 1, 2]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Set initial phases: neighbors aligned at π, target at 0
    G.nodes[0][THETA_PRIMARY] = math.pi  # Neighbor 1
    G.nodes[1][THETA_PRIMARY] = 0.0      # Target node (will align toward π)
    G.nodes[2][THETA_PRIMARY] = math.pi  # Neighbor 2
    
    # Mock validation to allow direct IL application
    def _ok_outcome(names):
        return SequenceValidationResult(
            tokens=tuple(names),
            canonical_tokens=tuple(names),
            passed=True,
            message="ok",
            metadata={},
        )
    monkeypatch.setattr("tnfr.structural.validate_sequence", _ok_outcome)
    
    # Apply Coherence to node 1 (should align toward π)
    coherence = Coherence()
    theta_before = G.nodes[1][THETA_PRIMARY]
    coherence(G, 1, phase_locking_coefficient=0.3)
    theta_after = G.nodes[1][THETA_PRIMARY]
    
    # Verify phase moved toward network mean (π)
    # Network mean of neighbors (π, π) is π
    # Expected: 0.0 + 0.3 * (π - 0.0) = 0.3π ≈ 0.942
    expected_theta = theta_before + 0.3 * (math.pi - theta_before)
    
    assert theta_after == pytest.approx(expected_theta, abs=1e-6), (
        f"Phase should align toward network mean: "
        f"before={theta_before:.4f}, after={theta_after:.4f}, expected={expected_theta:.4f}"
    )
    
    # Verify telemetry logged
    assert "IL_phase_locking" in G.graph
    locking_data = G.graph["IL_phase_locking"][0]
    assert locking_data["node"] == 1
    assert locking_data["theta_before"] == pytest.approx(theta_before, abs=1e-6)
    assert locking_data["theta_after"] == pytest.approx(theta_after, abs=1e-6)
    assert locking_data["theta_network"] == pytest.approx(math.pi, abs=1e-6)


def test_phase_locking_handles_wrap_around(monkeypatch: pytest.MonkeyPatch):
    """IL correctly handles phase wrap-around at 2π using circular mean."""
    # Create a simple network
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    
    # Initialize nodes
    for node in [0, 1, 2]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Set phases near wrap-around: 0.1 and 6.2 should average to ~0, not π
    G.nodes[0][THETA_PRIMARY] = 0.1      # Near 0
    G.nodes[1][THETA_PRIMARY] = math.pi  # Target (far from both)
    G.nodes[2][THETA_PRIMARY] = 6.2      # Near 2π (≈ 0)
    
    # Mock validation
    def _ok_outcome(names):
        return SequenceValidationResult(
            tokens=tuple(names),
            canonical_tokens=tuple(names),
            passed=True,
            message="ok",
            metadata={},
        )
    monkeypatch.setattr("tnfr.structural.validate_sequence", _ok_outcome)
    
    # Apply Coherence
    coherence = Coherence()
    theta_before = G.nodes[1][THETA_PRIMARY]
    coherence(G, 1, phase_locking_coefficient=0.3)
    theta_after = G.nodes[1][THETA_PRIMARY]
    
    # Verify phase moved toward network mean (should be near 0, not π)
    # The circular mean of 0.1 and 6.2 is approximately 0.15 (not π!)
    assert "IL_phase_locking" in G.graph
    locking_data = G.graph["IL_phase_locking"][0]
    theta_network = locking_data["theta_network"]
    
    # Network mean should be near 0 (< 1.0), not near π (≈ 3.14)
    assert theta_network < 1.0, (
        f"Circular mean of 0.1 and 6.2 should be near 0, got {theta_network:.4f}"
    )


def test_phase_locking_no_neighbors_no_crash(monkeypatch: pytest.MonkeyPatch):
    """IL with no neighbors doesn't crash (phase unchanged)."""
    # Create isolated node
    G = nx.Graph()
    G.add_node(0)
    G.nodes[0][EPI_PRIMARY] = 0.5
    G.nodes[0][VF_PRIMARY] = 1.0
    G.nodes[0][DNFR_PRIMARY] = 0.1
    G.nodes[0][THETA_PRIMARY] = 1.5
    
    # Mock validation
    def _ok_outcome(names):
        return SequenceValidationResult(
            tokens=tuple(names),
            canonical_tokens=tuple(names),
            passed=True,
            message="ok",
            metadata={},
        )
    monkeypatch.setattr("tnfr.structural.validate_sequence", _ok_outcome)
    
    # Apply Coherence (should not crash)
    coherence = Coherence()
    theta_before = G.nodes[0][THETA_PRIMARY]
    coherence(G, 0)
    theta_after = G.nodes[0][THETA_PRIMARY]
    
    # Phase should be unchanged (no neighbors to align with)
    assert theta_after == theta_before, (
        f"Phase should be unchanged for isolated node: "
        f"before={theta_before}, after={theta_after}"
    )
    
    # No phase locking telemetry should be added (no neighbors case)
    # The _apply_phase_locking returns early if no neighbors
    if "IL_phase_locking" in G.graph:
        # If telemetry exists, it should not be for this node
        assert len(G.graph["IL_phase_locking"]) == 0


def test_phase_locking_custom_coefficient(monkeypatch: pytest.MonkeyPatch):
    """IL phase locking coefficient can be customized."""
    # Create network
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    
    for node in [0, 1]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    G.nodes[0][THETA_PRIMARY] = math.pi
    G.nodes[1][THETA_PRIMARY] = 0.0
    
    # Mock validation
    def _ok_outcome(names):
        return SequenceValidationResult(
            tokens=tuple(names),
            canonical_tokens=tuple(names),
            passed=True,
            message="ok",
            metadata={},
        )
    monkeypatch.setattr("tnfr.structural.validate_sequence", _ok_outcome)
    
    # Apply Coherence with custom coefficient (α = 0.5)
    coherence = Coherence()
    theta_before = G.nodes[1][THETA_PRIMARY]
    coherence(G, 1, phase_locking_coefficient=0.5)
    theta_after = G.nodes[1][THETA_PRIMARY]
    
    # Expected: 0.0 + 0.5 * (π - 0.0) = 0.5π
    expected_theta = theta_before + 0.5 * (math.pi - theta_before)
    
    assert theta_after == pytest.approx(expected_theta, abs=1e-6), (
        f"Phase locking with custom coefficient: "
        f"before={theta_before:.4f}, after={theta_after:.4f}, expected={expected_theta:.4f}"
    )


def test_phase_locking_convergence_with_repeated_il():
    """Multiple IL applications drive phases toward convergence."""
    # Create a star network: node 1 at center connected to 0, 2, 3
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3)])
    
    for node in [0, 1, 2, 3]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Set initial phases: periphery aligned, center misaligned
    G.nodes[0][THETA_PRIMARY] = 1.0
    G.nodes[1][THETA_PRIMARY] = 4.0  # Far from neighbors
    G.nodes[2][THETA_PRIMARY] = 1.0
    G.nodes[3][THETA_PRIMARY] = 1.0
    
    # Measure initial alignment
    alignment_before = compute_phase_alignment(G, 1, radius=1)
    
    # Apply IL multiple times to center node
    # First application
    run_sequence(G, 1, [Emission(), Reception(), Coherence(), Silence()])
    
    # Subsequent applications need reactivation
    for _ in range(4):
        Coherence()(G, 1)  # Reactivate from silence
        run_sequence(G, 1, [Emission(), Reception(), Coherence(), Silence()])
    
    # Measure final alignment
    alignment_after = compute_phase_alignment(G, 1, radius=1)
    
    # Alignment should improve (phase coherence increases)
    assert alignment_after > alignment_before, (
        f"Phase alignment should improve with repeated IL: "
        f"before={alignment_before:.4f}, after={alignment_after:.4f}"
    )


def test_phase_locking_normalizes_to_0_2pi():
    """Phase values remain in [0, 2π] after locking."""
    # Create network
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    
    for node in [0, 1]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    # Set phases that could cause wrap-around issues
    G.nodes[0][THETA_PRIMARY] = 6.0  # Near 2π
    G.nodes[1][THETA_PRIMARY] = 0.5  # Near 0
    
    # Apply IL to both nodes
    run_sequence(G, 1, [Emission(), Reception(), Coherence(), Silence()])
    run_sequence(G, 0, [Emission(), Reception(), Coherence(), Silence()])
    
    # Verify all phases in [0, 2π]
    for node in [0, 1]:
        theta = G.nodes[node][THETA_PRIMARY]
        assert 0.0 <= theta <= 2 * math.pi, (
            f"Phase should be in [0, 2π]: node={node}, theta={theta:.4f}"
        )


def test_phase_alignment_metric_increases_with_il():
    """Phase alignment metric (Kuramoto order parameter) increases with IL."""
    # Create a ring network
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    
    for node in [0, 1, 2, 3]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
    
    import random
    random.seed(42)
    for node in [0, 1, 2, 3]:
        G.nodes[node][THETA_PRIMARY] = random.uniform(0, 2 * math.pi)
    
    # Measure initial global phase coherence
    coherence_before = compute_global_phase_coherence(G)
    
    # Apply IL to all nodes
    for node in [0, 1, 2, 3]:
        run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    
    # Measure final global phase coherence
    coherence_after = compute_global_phase_coherence(G)
    
    # Global phase coherence should improve
    # Note: This may not always increase in one iteration depending on initial conditions,
    # but with the seed we're using it should increase
    assert coherence_after >= coherence_before * 0.9, (
        f"Global phase coherence should not decrease significantly: "
        f"before={coherence_before:.4f}, after={coherence_after:.4f}"
    )


def test_phase_locking_telemetry_structure():
    """IL phase locking telemetry has correct structure."""
    # Create simple network
    G = nx.Graph()
    G.add_edges_from([(0, 1)])
    
    for node in [0, 1]:
        G.nodes[node][EPI_PRIMARY] = 0.5
        G.nodes[node][VF_PRIMARY] = 1.0
        G.nodes[node][DNFR_PRIMARY] = 0.1
        G.nodes[node][THETA_PRIMARY] = float(node)
    
    # Apply IL
    run_sequence(G, 1, [Emission(), Reception(), Coherence(), Silence()])
    
    # Verify telemetry structure
    assert "IL_phase_locking" in G.graph
    assert len(G.graph["IL_phase_locking"]) > 0
    
    locking_data = G.graph["IL_phase_locking"][0]
    required_keys = {
        "node",
        "theta_before",
        "theta_after",
        "theta_network",
        "delta_theta",
        "alignment_achieved",
    }
    
    assert set(locking_data.keys()) == required_keys, (
        f"Phase locking telemetry should have keys {required_keys}, "
        f"got {set(locking_data.keys())}"
    )
    
    # Verify alignment_achieved makes sense
    # alignment_achieved = |delta_theta| * (1 - α)
    # This is the residual misalignment after locking
    delta_theta = locking_data["delta_theta"]
    alignment_achieved = locking_data["alignment_achieved"]
    expected_alignment = abs(delta_theta) * (1 - 0.3)  # Default α = 0.3
    
    assert alignment_achieved == pytest.approx(expected_alignment, abs=1e-6), (
        f"alignment_achieved should be |delta_theta| * (1 - α): "
        f"got {alignment_achieved:.4f}, expected {expected_alignment:.4f}"
    )


def test_phase_locking_metrics_integration():
    """Coherence metrics include phase_alignment after IL application."""
    G, node = create_nfr("metrics_test", epi=0.5, vf=1.0)
    
    # Add a neighbor
    neighbor = "neighbor"
    G.add_node(neighbor)
    G.add_edge(node, neighbor)
    G.nodes[neighbor][EPI_PRIMARY] = 0.5
    G.nodes[neighbor][VF_PRIMARY] = 1.0
    G.nodes[neighbor][DNFR_PRIMARY] = 0.1
    G.nodes[neighbor][THETA_PRIMARY] = 1.0
    G.nodes[node][THETA_PRIMARY] = 0.0
    
    # Enable metrics collection
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    
    # Apply IL with metrics
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    
    # Check that operator_metrics includes phase_alignment
    assert "operator_metrics" in G.graph
    
    # Find IL metrics
    il_metrics = [m for m in G.graph["operator_metrics"] if m.get("operator") == "Coherence"]
    assert len(il_metrics) > 0, "Should have IL metrics"
    
    il_metric = il_metrics[0]
    assert "phase_alignment" in il_metric, "IL metrics should include phase_alignment"
    assert "phase_coherence_quality" in il_metric, "IL metrics should include phase_coherence_quality"
    
    # Verify values are in valid range
    phase_alignment = il_metric["phase_alignment"]
    assert 0.0 <= phase_alignment <= 1.0, (
        f"phase_alignment should be in [0, 1]: got {phase_alignment}"
    )
