"""Tests for unified phase compatibility calculations.

Validates the canonical phase coupling strength formula and compatibility
checks used by UM, RA, and THOL operators.

Tests ensure compliance with:
- TNFR physics (wave interference, constructive/destructive)
- Invariant #5 (explicit phase verification required)
- Grammar rule U3 (RESONANT COUPLING)
"""

import math
import pytest
import networkx as nx

from tnfr.metrics.phase_compatibility import (
    compute_phase_coupling_strength,
    is_phase_compatible,
    compute_network_phase_alignment,
)
from tnfr.constants.aliases import ALIAS_THETA


class TestComputePhaseCouplingStrength:
    """Test canonical phase coupling strength formula."""

    def test_perfect_alignment_returns_one(self):
        """Perfect phase alignment (Δφ = 0) should return coupling = 1.0."""
        assert compute_phase_coupling_strength(0.0, 0.0) == 1.0
        assert compute_phase_coupling_strength(1.0, 1.0) == 1.0
        assert compute_phase_coupling_strength(math.pi, math.pi) == 1.0

    def test_antiphase_returns_zero(self):
        """Antiphase (Δφ = π) should return coupling = 0.0 (destructive interference)."""
        result = compute_phase_coupling_strength(0.0, math.pi)
        assert abs(result - 0.0) < 1e-10, f"Expected 0.0, got {result}"
        
        result = compute_phase_coupling_strength(math.pi/2, 3*math.pi/2)
        assert abs(result - 0.0) < 1e-10, f"Expected 0.0, got {result}"

    def test_orthogonal_phases_return_half(self):
        """Orthogonal phases (Δφ = π/2) should return coupling = 0.5."""
        result = compute_phase_coupling_strength(0.0, math.pi/2)
        assert abs(result - 0.5) < 1e-10, f"Expected 0.5, got {result}"
        
        result = compute_phase_coupling_strength(math.pi, 3*math.pi/2)
        assert abs(result - 0.5) < 1e-10, f"Expected 0.5, got {result}"

    def test_small_phase_difference(self):
        """Small phase difference should yield high coupling strength."""
        # Δφ = 0.1 rad (~5.7°)
        result = compute_phase_coupling_strength(0.0, 0.1)
        assert result > 0.95, f"Expected > 0.95, got {result}"
        assert result < 1.0, f"Expected < 1.0, got {result}"

    def test_symmetry(self):
        """Coupling strength should be symmetric: f(a,b) = f(b,a)."""
        theta_a, theta_b = 0.5, 1.5
        forward = compute_phase_coupling_strength(theta_a, theta_b)
        backward = compute_phase_coupling_strength(theta_b, theta_a)
        assert abs(forward - backward) < 1e-10, \
            f"Not symmetric: f({theta_a},{theta_b})={forward}, f({theta_b},{theta_a})={backward}"

    def test_wrapping_at_boundaries(self):
        """Phase wrapping at 2π should be handled correctly."""
        # Phases 0.1 and 2π - 0.1 are only 0.2 radians apart
        result = compute_phase_coupling_strength(0.1, 2*math.pi - 0.1)
        expected = 1.0 - (0.2 / math.pi)  # ~0.936
        assert abs(result - expected) < 1e-10, \
            f"Wrapping failed: expected {expected}, got {result}"

    def test_linear_interpolation(self):
        """Coupling should decrease linearly with phase difference."""
        # Test points along the linear relationship
        test_points = [
            (0.0, 1.0),           # Δφ = 0 → coupling = 1.0
            (math.pi/4, 0.75),    # Δφ = π/4 → coupling = 0.75
            (math.pi/2, 0.5),     # Δφ = π/2 → coupling = 0.5
            (3*math.pi/4, 0.25),  # Δφ = 3π/4 → coupling = 0.25
            (math.pi, 0.0),       # Δφ = π → coupling = 0.0
        ]
        
        for phase_diff, expected_coupling in test_points:
            result = compute_phase_coupling_strength(0.0, phase_diff)
            assert abs(result - expected_coupling) < 1e-10, \
                f"At Δφ={phase_diff:.3f}: expected {expected_coupling}, got {result}"

    def test_output_range(self):
        """Output should always be in [0, 1]."""
        import random
        random.seed(42)
        
        for _ in range(100):
            theta_a = random.uniform(0, 2*math.pi)
            theta_b = random.uniform(0, 2*math.pi)
            result = compute_phase_coupling_strength(theta_a, theta_b)
            assert 0.0 <= result <= 1.0, \
                f"Out of range [0,1]: {result} for θ_a={theta_a}, θ_b={theta_b}"


class TestIsPhaseCompatible:
    """Test boolean phase compatibility check."""

    def test_default_threshold_accepts_orthogonal(self):
        """Default threshold (0.5) should accept orthogonal phases."""
        assert is_phase_compatible(0.0, math.pi/2, threshold=0.5)
        assert is_phase_compatible(0.0, math.pi/2 - 0.01, threshold=0.5)

    def test_default_threshold_rejects_beyond_orthogonal(self):
        """Default threshold (0.5) should reject phases beyond orthogonal."""
        assert not is_phase_compatible(0.0, math.pi/2 + 0.1, threshold=0.5)
        assert not is_phase_compatible(0.0, math.pi, threshold=0.5)

    def test_perfect_alignment_always_compatible(self):
        """Perfect alignment should be compatible for any threshold."""
        for threshold in [0.1, 0.5, 0.9, 0.99]:
            assert is_phase_compatible(0.0, 0.0, threshold=threshold), \
                f"Perfect alignment failed at threshold={threshold}"

    def test_antiphase_always_incompatible(self):
        """Antiphase should be incompatible for any positive threshold."""
        for threshold in [0.01, 0.1, 0.5, 0.9]:
            assert not is_phase_compatible(0.0, math.pi, threshold=threshold), \
                f"Antiphase incorrectly compatible at threshold={threshold}"

    def test_threshold_boundary_exact(self):
        """At threshold boundary, should be compatible (>=, not >)."""
        # Orthogonal gives coupling = 0.5
        assert is_phase_compatible(0.0, math.pi/2, threshold=0.5)
        
        # Just above threshold: incompatible
        # Need coupling < 0.5, which means Δφ > π/2
        assert not is_phase_compatible(0.0, math.pi/2 + 0.01, threshold=0.5)

    def test_high_threshold_restrictive(self):
        """High threshold (0.9) should be very restrictive."""
        # Requires coupling >= 0.9, i.e., Δφ <= 0.1π (~18°)
        assert is_phase_compatible(0.0, 0.1, threshold=0.9)
        assert not is_phase_compatible(0.0, math.pi/4, threshold=0.9)

    def test_low_threshold_permissive(self):
        """Low threshold (0.1) should be permissive."""
        # Allows coupling >= 0.1, i.e., Δφ <= 0.9π (~162°)
        assert is_phase_compatible(0.0, 2.8, threshold=0.1)  # Δφ ~ 2.8 < 0.9π
        assert not is_phase_compatible(0.0, math.pi - 0.01, threshold=0.1)

    def test_wrapping_compatibility(self):
        """Phases near 0 and 2π should be compatible."""
        # 0.1 and 2π-0.1 are only 0.2 radians apart
        assert is_phase_compatible(0.1, 2*math.pi - 0.1, threshold=0.5)


class TestComputeNetworkPhaseAlignment:
    """Test network phase alignment calculation (wrapper function)."""

    def test_perfect_alignment_returns_high_value(self):
        """Perfectly aligned network should return r ≈ 1.0."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # All nodes have same phase - use 'theta' key directly
        for node in G.nodes():
            G.nodes[node]['theta'] = 0.5
        
        alignment = compute_network_phase_alignment(G, node=1, radius=2)
        assert alignment > 0.99, f"Expected near-perfect alignment, got {alignment}"

    def test_divergent_phases_handled(self):
        """Network phase alignment should return valid values for divergent phases."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        
        # Set phases using individual alias elements (theta is first in tuple)
        for i, node in enumerate(G.nodes()):
            # Use 'theta' which is the first element of ALIAS_THETA tuple
            G.nodes[node]['theta'] = i * 0.5
        
        alignment = compute_network_phase_alignment(G, node=1, radius=2)
        # Should return valid alignment value in [0, 1]
        assert 0.0 <= alignment <= 1.0, f"Alignment out of range: {alignment}"

    def test_radius_parameter(self):
        """Radius parameter should control neighborhood size."""
        G = nx.path_graph(5)  # Linear chain: 0-1-2-3-4
        
        # Node 2 neighbors: radius=1 → [1,2,3], radius=2 → [0,1,2,3,4]
        for node in G.nodes():
            # Use 'theta' key directly
            G.nodes[node]['theta'] = node * 0.1
        
        alignment_r1 = compute_network_phase_alignment(G, node=2, radius=1)
        alignment_r2 = compute_network_phase_alignment(G, node=2, radius=2)
        
        # Both should be valid alignment values
        assert 0.0 <= alignment_r1 <= 1.0
        assert 0.0 <= alignment_r2 <= 1.0

    def test_isolated_node(self):
        """Node with no neighbors should handle gracefully."""
        G = nx.Graph()
        G.add_node(0)
        G.nodes[0]['theta'] = 0.5
        
        # Should not crash, return value for single node
        alignment = compute_network_phase_alignment(G, node=0, radius=1)
        assert 0.0 <= alignment <= 1.0

    def test_consistency_with_phase_coherence_module(self):
        """Should return same value as underlying phase_coherence function."""
        from tnfr.metrics.phase_coherence import compute_phase_alignment
        
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        for i, theta in enumerate([0.0, 0.1, 0.2]):
            # Use 'theta' key directly
            G.nodes[i]['theta'] = theta
        
        # Both functions should return identical results
        wrapper_result = compute_network_phase_alignment(G, node=1, radius=1)
        direct_result = compute_phase_alignment(G, node=1, radius=1)
        
        assert abs(wrapper_result - direct_result) < 1e-10, \
            f"Wrapper inconsistent: {wrapper_result} vs {direct_result}"


class TestPhysicsCompliance:
    """Test compliance with TNFR physics principles."""

    def test_constructive_interference_physics(self):
        """In-phase nodes should have maximum coupling (constructive interference)."""
        # Physics: aligned waves amplify each other
        coupling = compute_phase_coupling_strength(0.0, 0.01)
        assert coupling > 0.99, "In-phase should have near-maximum coupling"

    def test_destructive_interference_physics(self):
        """Antiphase nodes should have zero coupling (destructive interference)."""
        # Physics: opposing waves cancel each other
        coupling = compute_phase_coupling_strength(0.0, math.pi)
        assert coupling < 0.01, "Antiphase should have near-zero coupling"

    def test_invariant_5_enforcement(self):
        """Functions should support explicit phase verification (Invariant #5)."""
        # Invariant #5: "No coupling without explicit phase verification"
        # These functions provide the explicit verification mechanism
        
        # Case 1: Compatible phases
        theta_a, theta_b = 0.0, 0.1
        assert is_phase_compatible(theta_a, theta_b, threshold=0.5), \
            "Compatible phases should pass verification"
        
        # Case 2: Incompatible phases
        theta_a, theta_b = 0.0, math.pi
        assert not is_phase_compatible(theta_a, theta_b, threshold=0.5), \
            "Incompatible phases should fail verification"

    def test_grammar_u3_compliance(self):
        """Should support Grammar Rule U3 (RESONANT COUPLING)."""
        # U3: Resonance requires |φᵢ - φⱼ| ≤ Δφ_max
        # This function computes the compatibility check
        
        # Define Δφ_max = π/2 (orthogonal threshold)
        delta_phi_max = math.pi / 2
        threshold = 1.0 - (delta_phi_max / math.pi)  # = 0.5
        
        # Within Δφ_max: should be compatible
        assert is_phase_compatible(0.0, math.pi/4, threshold=threshold)
        
        # Beyond Δφ_max: should be incompatible
        assert not is_phase_compatible(0.0, math.pi/2 + 0.1, threshold=threshold)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_phases(self):
        """Both phases zero should return perfect coupling."""
        assert compute_phase_coupling_strength(0.0, 0.0) == 1.0

    def test_negative_phases_handled(self):
        """Negative phases should be handled correctly by angle_diff."""
        # angle_diff normalizes to shortest arc
        coupling = compute_phase_coupling_strength(-0.1, 0.1)
        expected = 1.0 - (0.2 / math.pi)
        assert abs(coupling - expected) < 1e-10

    def test_phases_beyond_2pi(self):
        """Phases > 2π should be handled correctly."""
        # angle_diff should handle wrap-around
        coupling = compute_phase_coupling_strength(0.0, 2*math.pi + 0.1)
        expected = 1.0 - (0.1 / math.pi)
        assert abs(coupling - expected) < 1e-10

    def test_zero_threshold(self):
        """Threshold = 0 should accept all phase differences."""
        assert is_phase_compatible(0.0, math.pi, threshold=0.0)

    def test_threshold_one(self):
        """Threshold = 1.0 should only accept perfect alignment."""
        assert is_phase_compatible(0.0, 0.0, threshold=1.0)
        assert not is_phase_compatible(0.0, 0.001, threshold=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
