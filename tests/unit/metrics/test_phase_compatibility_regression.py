"""Regression tests for phase compatibility refactoring.

Verifies that the unified phase compatibility functions produce identical
results to the original inline implementations that were replaced in UM
and THOL operators.
"""

import math
import pytest

from tnfr.metrics.phase_compatibility import compute_phase_coupling_strength
from tnfr.utils.numeric import angle_diff


class TestPhaseCompatibilityRegression:
    """Test that refactored code produces identical results to original."""

    def test_thol_propagation_formula_equivalence(self):
        """Verify THOL propagation coupling formula remains identical."""
        # Original THOL implementation:
        # phase_diff = abs(angle_diff(neighbor_theta, parent_theta))
        # coupling_strength = 1.0 - (phase_diff / math.pi)
        
        test_cases = [
            (0.0, 0.0),           # Perfect alignment
            (0.0, math.pi/2),     # Orthogonal
            (0.0, math.pi),       # Antiphase
            (0.1, 0.15),          # Small difference
            (0.0, 2*math.pi - 0.1),  # Wrap-around
            (math.pi/4, 3*math.pi/4),  # 90 degree separation
        ]
        
        for parent_theta, neighbor_theta in test_cases:
            # Original formula
            phase_diff_original = abs(angle_diff(neighbor_theta, parent_theta))
            coupling_original = 1.0 - (phase_diff_original / math.pi)
            
            # New unified formula
            coupling_unified = compute_phase_coupling_strength(parent_theta, neighbor_theta)
            
            assert abs(coupling_original - coupling_unified) < 1e-10, \
                f"Mismatch for θ_parent={parent_theta}, θ_neighbor={neighbor_theta}: " \
                f"original={coupling_original}, unified={coupling_unified}"

    def test_thol_capture_signals_formula_equivalence(self):
        """Verify THOL capture_network_signals coupling formula remains identical."""
        # Original implementation in capture_network_signals:
        # phase_diff = abs(n_theta - node_theta)
        # if phase_diff > math.pi:
        #     phase_diff = 2 * math.pi - phase_diff
        # coupling_strength = 1.0 - (phase_diff / math.pi)
        
        test_cases = [
            (0.0, 0.0),
            (0.0, math.pi/2),
            (0.0, math.pi),
            (0.1, 0.15),
            (0.0, 1.9 * math.pi),  # Should wrap to small difference
            (math.pi/6, 5*math.pi/6),  # 120 degree separation
        ]
        
        for node_theta, n_theta in test_cases:
            # Original formula (manual normalization)
            phase_diff_original = abs(n_theta - node_theta)
            if phase_diff_original > math.pi:
                phase_diff_original = 2 * math.pi - phase_diff_original
            coupling_original = 1.0 - (phase_diff_original / math.pi)
            
            # New unified formula (uses angle_diff internally)
            coupling_unified = compute_phase_coupling_strength(node_theta, n_theta)
            
            assert abs(coupling_original - coupling_unified) < 1e-10, \
                f"Mismatch for θ_node={node_theta}, θ_n={n_theta}: " \
                f"original={coupling_original}, unified={coupling_unified}"

    def test_um_phase_alignment_formula_equivalence(self):
        """Verify UM operator phase alignment formula remains identical."""
        # Original UM implementation for ΔNFR reduction:
        # dphi = abs(angle_diff(neighbor.theta, node.theta))
        # alignment = 1.0 - dphi / math.pi
        
        test_cases = [
            (0.0, 0.0),
            (0.0, math.pi/2),
            (0.0, math.pi),
            (0.1, 0.2),
            (math.pi, math.pi + 0.1),
            (0.1, 2*math.pi - 0.1),
        ]
        
        for node_theta, neighbor_theta in test_cases:
            # Original formula
            dphi_original = abs(angle_diff(neighbor_theta, node_theta))
            alignment_original = 1.0 - dphi_original / math.pi
            
            # New unified formula
            alignment_unified = compute_phase_coupling_strength(node_theta, neighbor_theta)
            
            assert abs(alignment_original - alignment_unified) < 1e-10, \
                f"Mismatch for θ_node={node_theta}, θ_neighbor={neighbor_theta}: " \
                f"original={alignment_original}, unified={alignment_unified}"

    def test_um_functional_links_formula_equivalence(self):
        """Verify UM functional link formation phase coupling remains identical."""
        # Original UM implementation for link formation:
        # th_j = j.theta
        # dphi = abs(angle_diff(th_j, th_i)) / math.pi
        # phase_coupling = (1 - dphi)
        
        test_cases = [
            (0.0, 0.0),
            (0.0, math.pi/2),
            (0.0, math.pi),
            (math.pi/4, 3*math.pi/4),
            (0.05, 0.1),
            (5.0, 5.5),
        ]
        
        for th_i, th_j in test_cases:
            # Original formula
            dphi_original = abs(angle_diff(th_j, th_i)) / math.pi
            phase_coupling_original = 1 - dphi_original
            
            # New unified formula
            phase_coupling_unified = compute_phase_coupling_strength(th_i, th_j)
            
            assert abs(phase_coupling_original - phase_coupling_unified) < 1e-10, \
                f"Mismatch for θ_i={th_i}, θ_j={th_j}: " \
                f"original={phase_coupling_original}, unified={phase_coupling_unified}"

    def test_comprehensive_equivalence_scan(self):
        """Scan through many angle combinations to ensure complete equivalence."""
        import random
        random.seed(42)
        
        # Test 100 random angle pairs
        for _ in range(100):
            theta_a = random.uniform(0, 2*math.pi)
            theta_b = random.uniform(0, 2*math.pi)
            
            # Original formula (most general form)
            phase_diff = abs(angle_diff(theta_b, theta_a))
            coupling_original = 1.0 - (phase_diff / math.pi)
            
            # Unified formula
            coupling_unified = compute_phase_coupling_strength(theta_a, theta_b)
            
            assert abs(coupling_original - coupling_unified) < 1e-10, \
                f"Random test failed for θ_a={theta_a}, θ_b={theta_b}"


class TestBackwardCompatibility:
    """Test that operator behavior is preserved after refactoring."""

    def test_um_compatibility_calculation_unchanged(self):
        """UM compatibility formula should produce same results as before."""
        # Test the full compatibility calculation used in UM functional links
        # compat = (1 - dphi) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
        
        th_i, th_j = 0.0, math.pi/4
        epi_i, epi_j = 0.8, 0.7
        si_i, si_j = 0.9, 0.85
        
        # Original calculation
        dphi_original = abs(angle_diff(th_j, th_i)) / math.pi
        epi_sim = 1.0 - abs(epi_i - epi_j) / (abs(epi_i) + abs(epi_j) + 1e-9)
        si_sim = 1.0 - abs(si_i - si_j)
        compat_original = (1 - dphi_original) * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
        
        # New calculation
        phase_coupling_unified = compute_phase_coupling_strength(th_i, th_j)
        compat_unified = phase_coupling_unified * 0.5 + 0.25 * epi_sim + 0.25 * si_sim
        
        assert abs(compat_original - compat_unified) < 1e-10, \
            f"UM compatibility mismatch: original={compat_original}, unified={compat_unified}"

    def test_thol_propagation_threshold_unchanged(self):
        """THOL propagation thresholding should work identically."""
        # Test that the same neighbors would be selected for propagation
        
        parent_theta = 0.0
        neighbor_thetas = [0.1, math.pi/3, math.pi/2, 2.0, math.pi, 3.0]
        threshold = 0.5
        
        # Original logic
        selected_original = []
        for neighbor_theta in neighbor_thetas:
            phase_diff = abs(angle_diff(neighbor_theta, parent_theta))
            coupling = 1.0 - (phase_diff / math.pi)
            if coupling >= threshold:
                selected_original.append(neighbor_theta)
        
        # New logic
        selected_unified = []
        for neighbor_theta in neighbor_thetas:
            coupling = compute_phase_coupling_strength(parent_theta, neighbor_theta)
            if coupling >= threshold:
                selected_unified.append(neighbor_theta)
        
        assert selected_original == selected_unified, \
            f"THOL propagation selection mismatch: " \
            f"original={selected_original}, unified={selected_unified}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
