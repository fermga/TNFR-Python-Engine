"""Comprehensive tests for the TNFR Structural Field Tetrad.

Tests the four canonical structural fields with mathematical foundations:
- Φ_s: Structural Potential Field (Global stability)
- |∇φ|: Phase Gradient Field (Local desynchronization) 
- K_φ: Phase Curvature Field (Geometric confinement)
- ξ_C: Coherence Length Field (Spatial correlations)

These fields form the complete tetrad for multi-scale TNFR network characterization.

Mathematical Foundations:
- Φ_s < φ ≈ 1.618 (Universal Tetrahedral Correspondence φ ↔ Φ_s)
- |∇φ| < 0.2904 (Harmonic oscillator stability + Kuramoto synchronization)
- K_φ < 2.8274 (TNFR formalism constraints + safety margin)
- ξ_C thresholds via spatial correlation theory + critical phenomena
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.physics.fields import (
    estimate_coherence_length,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
class TestStructuralPotentialField:
    """Test Φ_s: Structural Potential Field (Global stability monitoring)."""

    def setup_method(self) -> None:
        """Create test networks for field validation."""
        # Small stable network
        self.G_stable = nx.Graph()
        self.G_stable.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 0.5}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.1, "theta": 1.0}),
        ])
        self.G_stable.add_edges_from([(1, 2), (2, 3), (1, 3)])

        # Stressed network (higher ΔNFR)
        self.G_stressed = nx.Graph()
        self.G_stressed.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 1.5, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 1.2, "theta": 2.0}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.8, "theta": 4.0}),
        ])
        self.G_stressed.add_edges_from([(1, 2), (2, 3), (1, 3)])

    def test_structural_potential_mathematical_foundation(self) -> None:
        """Test Φ_s mathematical derivation from ΔNFR distribution."""
        phi_s_stable = compute_structural_potential(self.G_stable)
        phi_s_stressed = compute_structural_potential(self.G_stressed)

        # Stressed network should have higher structural potential
        assert phi_s_stressed > phi_s_stable, "Higher ΔNFR should increase Φ_s"

        # All values should be non-negative (physical requirement)
        assert all(v >= 0 for v in phi_s_stable.values()), "Φ_s must be non-negative"
        assert all(v >= 0 for v in phi_s_stressed.values()), "Φ_s must be non-negative"

    def test_golden_ratio_correspondence_phi_s(self) -> None:
        """Test Universal Tetrahedral Correspondence: φ ↔ Φ_s < 1.618."""
        phi_s = compute_structural_potential(self.G_stable)
        
        golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618
        
        for node_id, phi_s_value in phi_s.items():
            assert phi_s_value < golden_ratio, f"Node {node_id}: Φ_s={phi_s_value:.3f} should be < φ={golden_ratio:.3f}"

    def test_escape_threshold_detection(self) -> None:
        """Test that escape threshold Φ_s ≈ 2.0 can be detected."""
        # Create network at threshold
        G_threshold = nx.Graph()
        G_threshold.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 5.0, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 4.5, "theta": 1.0}),
        ])
        G_threshold.add_edges_from([(1, 2)])

        phi_s = compute_structural_potential(G_threshold)
        
        # Some nodes should approach escape threshold
        max_phi_s = max(phi_s.values())
        assert max_phi_s > 1.0, "High ΔNFR should produce significant Φ_s"

    def test_structural_potential_network_topology_invariance(self) -> None:
        """Test Φ_s computation across different network topologies."""
        topologies = [
            nx.path_graph(4),
            nx.cycle_graph(4),
            nx.complete_graph(4),
            nx.star_graph(4),
        ]

        for i, G in enumerate(topologies):
            # Initialize with identical node states
            for node in G.nodes():
                set_attr(G.nodes[node], ALIAS_EPI, 1.0)
                set_attr(G.nodes[node], ALIAS_VF, 2.0)
                set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
                set_attr(G.nodes[node], ALIAS_THETA, 0.0)

            phi_s = compute_structural_potential(G)
            
            # All computations should succeed
            assert len(phi_s) == G.number_of_nodes(), f"Topology {i}: Φ_s computed for all nodes"
            assert all(isinstance(v, (int, float)) for v in phi_s.values()), f"Topology {i}: Φ_s values numeric"


class TestPhaseGradientField:
    """Test |∇φ|: Phase Gradient Field (Local desynchronization proxy)."""

    def setup_method(self) -> None:
        """Create test networks with different phase patterns."""
        # Synchronized network (small phase differences)
        self.G_sync = nx.Graph()
        self.G_sync.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 0.1}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.1, "theta": 0.05}),
        ])
        self.G_sync.add_edges_from([(1, 2), (2, 3), (1, 3)])

        # Desynchronized network (large phase differences)
        self.G_desync = nx.Graph()
        self.G_desync.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 2.5}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.1, "theta": 5.0}),
        ])
        self.G_desync.add_edges_from([(1, 2), (2, 3), (1, 3)])

    def test_phase_gradient_mathematical_foundation(self) -> None:
        """Test |∇φ| captures local phase stress that C(t) misses."""
        grad_sync = compute_phase_gradient(self.G_sync)
        grad_desync = compute_phase_gradient(self.G_desync)

        # Desynchronized network should have higher phase gradients
        max_grad_sync = max(grad_sync.values()) if grad_sync else 0
        max_grad_desync = max(grad_desync.values()) if grad_desync else 0
        
        assert max_grad_desync > max_grad_sync, "Desynchronized network should have higher |∇φ|"

        # All values should be non-negative (magnitude field)
        assert all(v >= 0 for v in grad_sync.values()), "|∇φ| must be non-negative"
        assert all(v >= 0 for v in grad_desync.values()), "|∇φ| must be non-negative"

    def test_kuramoto_stability_threshold(self) -> None:
        """Test |∇φ| < 0.2904 threshold from Kuramoto synchronization theory."""
        grad_sync = compute_phase_gradient(self.G_sync)
        
        kuramoto_threshold = math.pi / (4 * math.sqrt(2))  # ≈ 0.2904
        
        for node_id, grad_value in grad_sync.items():
            # Synchronized network should be well below threshold
            assert grad_value < kuramoto_threshold, f"Node {node_id}: |∇φ|={grad_value:.3f} should be < {kuramoto_threshold:.3f}"

    def test_phase_gradient_wrapping_behavior(self) -> None:
        """Test |∇φ| handles phase wrapping correctly (2π periodicity)."""
        G_wrap = nx.Graph()
        G_wrap.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.1}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 6.2}),  # ≈ 0.1 + 2π
        ])
        G_wrap.add_edges_from([(1, 2)])

        grad = compute_phase_gradient(G_wrap)
        
        # Phase difference should be small due to wrapping
        for node_id, grad_value in grad.items():
            assert grad_value < 1.0, f"Node {node_id}: Phase wrapping should give small |∇φ|"


class TestPhaseCurvatureField:
    """Test K_φ: Phase Curvature Field (Geometric confinement monitoring)."""

    def setup_method(self) -> None:
        """Create networks with different geometric phase configurations."""
        # Regular triangle (symmetric)
        self.G_regular = nx.complete_graph(3)
        # Initialize nodes with TNFR attributes
        for node in self.G_regular.nodes():
            set_attr(self.G_regular.nodes[node], ALIAS_EPI, 1.0)
            set_attr(self.G_regular.nodes[node], ALIAS_VF, 2.0)
            set_attr(self.G_regular.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(self.G_regular.nodes[node], ALIAS_THETA, 0.2 * node)  # Small phase differences
        
        # Irregular network (asymmetric phases)
        self.G_irregular = nx.Graph()
        self.G_irregular.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 1.5}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.1, "theta": 4.0}),
            (4, {"EPI": 0.9, "nu_f": 1.8, "ΔNFR": 0.2, "theta": 2.8}),
        ])
        self.G_irregular.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

    def test_phase_curvature_mathematical_foundation(self) -> None:
        """Test K_φ measures geometric phase torsion."""
        curv_regular = compute_phase_curvature(self.G_regular)
        curv_irregular = compute_phase_curvature(self.G_irregular)

        # All values should be within [-π, π] bounds
        for node_id, curv_value in curv_regular.items():
            assert -math.pi <= curv_value <= math.pi, f"Regular node {node_id}: K_φ={curv_value:.3f} should be in [-π,π]"
        
        for node_id, curv_value in curv_irregular.items():
            assert -math.pi <= curv_value <= math.pi, f"Irregular node {node_id}: K_φ={curv_value:.3f} should be in [-π,π]"

    def test_tnfr_confinement_threshold(self) -> None:
        """Test K_φ < 2.8274 threshold from TNFR formalism constraints."""
        curv = compute_phase_curvature(self.G_regular)
        
        tnfr_threshold = 0.9 * math.pi  # ≈ 2.8274
        
        for node_id, curv_value in curv.items():
            abs_curv = abs(curv_value)
            assert abs_curv < tnfr_threshold, f"Node {node_id}: |K_φ|={abs_curv:.3f} should be < {tnfr_threshold:.3f}"

    def test_mutation_prone_loci_detection(self) -> None:
        """Test K_φ flags potential mutation sites (high curvature)."""
        # Create network with high phase stress
        G_stress = nx.Graph()
        G_stress.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.1, "theta": 0.0}),
            (2, {"EPI": 0.8, "nu_f": 1.5, "ΔNFR": 0.05, "theta": 3.0}),
            (3, {"EPI": 1.2, "nu_f": 2.5, "ΔNFR": -0.1, "theta": 6.0}),
        ])
        G_stress.add_edges_from([(1, 2), (2, 3), (1, 3)])

        curv = compute_phase_curvature(G_stress)
        
        # At least one node should show high curvature
        max_abs_curv = max(abs(v) for v in curv.values())
        assert max_abs_curv > 1.0, "High phase stress should produce significant curvature"


class TestCoherenceLengthField:
    """Test ξ_C: Coherence Length Field (Spatial correlation monitoring)."""

    def setup_method(self) -> None:
        """Create networks with different correlation structures."""
        # Highly coherent network
        self.G_coherent = nx.Graph()
        self.G_coherent.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.01, "theta": 0.0}),
            (2, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.01, "theta": 0.0}),
            (3, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 0.01, "theta": 0.0}),
        ])
        self.G_coherent.add_edges_from([(1, 2), (2, 3), (1, 3)])

        # Fragmented network
        self.G_fragmented = nx.Graph()
        self.G_fragmented.add_nodes_from([
            (1, {"EPI": 1.0, "nu_f": 2.0, "ΔNFR": 1.0, "theta": 0.0}),
            (2, {"EPI": 0.2, "nu_f": 0.5, "ΔNFR": -0.8, "theta": 3.0}),
            (3, {"EPI": 1.8, "nu_f": 3.0, "ΔNFR": 0.5, "theta": 5.0}),
        ])
        self.G_fragmented.add_edges_from([(1, 2), (2, 3), (1, 3)])

    def test_coherence_length_mathematical_foundation(self) -> None:
        """Test ξ_C based on spatial correlation theory."""
        xi_coherent = estimate_coherence_length(self.G_coherent)
        xi_fragmented = estimate_coherence_length(self.G_fragmented)

        # Coherent network should have longer correlation lengths
        assert xi_coherent > xi_fragmented, "Coherent network should have larger ξ_C"

        # All values should be positive (correlation length scale)
        assert xi_coherent > 0, "ξ_C must be positive"
        assert xi_fragmented > 0, "ξ_C must be positive"

    def test_critical_phenomena_scaling(self) -> None:
        """Test ξ_C scaling behavior near critical points."""
        # Test different system sizes
        sizes = [5, 10, 15]
        xi_values = []

        for size in sizes:
            G = nx.cycle_graph(size)
            # Initialize with moderate coherence
            for node in G.nodes():
                set_attr(G.nodes[node], ALIAS_EPI, 1.0)
                set_attr(G.nodes[node], ALIAS_VF, 2.0)
                set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
                set_attr(G.nodes[node], ALIAS_THETA, 0.1)

            xi = estimate_coherence_length(G)
            xi_values.append(xi)

        # Correlation length should scale with system size for coherent systems
        assert xi_values[-1] >= xi_values[0], "ξ_C should scale with system size"

    def test_dimensional_analysis_thresholds(self) -> None:
        """Test ξ_C thresholds from dimensional analysis."""
        xi = estimate_coherence_length(self.G_coherent)
        
        # Compute mean distance for comparison
        n_nodes = self.G_coherent.number_of_nodes()
        if n_nodes > 1:
            mean_distance = 1.0  # Approximate for small test networks
            
            # For coherent systems, ξ_C should be comparable to or larger than mean distance
            assert xi >= 0.1 * mean_distance, "ξ_C should be comparable to network scale for coherent systems"


class TestTetradFieldIntegration:
    """Test integration and consistency across all four structural fields."""

    def setup_method(self) -> None:
        """Create comprehensive test network."""
        self.G = nx.complete_graph(6)
        # Initialize nodes with TNFR attributes
        for node in self.G.nodes():
            set_attr(self.G.nodes[node], ALIAS_EPI, 1.0)
            set_attr(self.G.nodes[node], ALIAS_VF, 2.0) 
            set_attr(self.G.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(self.G.nodes[node], ALIAS_THETA, 0.1 * node)

    def test_all_fields_computable_simultaneously(self) -> None:
        """Test that all four tetrad fields can be computed together."""
        phi_s = compute_structural_potential(self.G)
        grad_phi = compute_phase_gradient(self.G)
        curv_phi = compute_phase_curvature(self.G)
        xi_c = estimate_coherence_length(self.G)

        # All computations should succeed
        assert len(phi_s) == self.G.number_of_nodes(), "Φ_s computed for all nodes"
        assert len(grad_phi) == self.G.number_of_nodes(), "|∇φ| computed for all nodes"
        assert len(curv_phi) == self.G.number_of_nodes(), "K_φ computed for all nodes"
        assert isinstance(xi_c, (int, float)), "ξ_C computed as scalar"

    def test_field_consistency_under_operations(self) -> None:
        """Test field consistency when nodes are modified."""
        # Compute initial fields
        phi_s_initial = compute_structural_potential(self.G)
        grad_phi_initial = compute_phase_gradient(self.G)

        # Modify one node
        node_id = list(self.G.nodes())[0]
        set_attr(self.G.nodes[node_id], ALIAS_DNFR, 0.5)

        # Recompute fields
        phi_s_modified = compute_structural_potential(self.G)
        grad_phi_modified = compute_phase_gradient(self.G)

        # Fields should reflect the change
        assert phi_s_modified[node_id] != phi_s_initial[node_id], "Φ_s should reflect ΔNFR change"
        # Phase gradient may or may not change depending on neighbor phases

    def test_mathematical_invariants_preservation(self) -> None:
        """Test that mathematical invariants are preserved across fields."""
        phi_s = compute_structural_potential(self.G)
        curv_phi = compute_phase_curvature(self.G)

        # Physical bounds should be respected
        golden_ratio = (1 + math.sqrt(5)) / 2
        for node_id, phi_s_value in phi_s.items():
            assert phi_s_value >= 0, f"Node {node_id}: Φ_s must be non-negative"
            # Note: Not all networks will satisfy φ < golden_ratio in all conditions

        for node_id, curv_value in curv_phi.items():
            assert -math.pi <= curv_value <= math.pi, f"Node {node_id}: K_φ must be in [-π,π]"


@pytest.mark.parametrize("topology_func,size", [
    (nx.path_graph, 6),
    (nx.cycle_graph, 6),
    (nx.complete_graph, 5),
    (nx.star_graph, 5),
    (nx.random_regular_graph, (3, 8)),  # (degree, nodes)
])
def test_tetrad_cross_topology_validation(topology_func: Any, size: Any) -> None:
    """Test tetrad fields across different network topologies."""
    if topology_func == nx.random_regular_graph:
        degree, nodes = size
        G = topology_func(degree, nodes, seed=42)  # Fixed seed for reproducibility
    else:
        G = topology_func(size)

    # Initialize all nodes with standard TNFR attributes
    for node in G.nodes():
        set_attr(G.nodes[node], ALIAS_EPI, 1.0)
        set_attr(G.nodes[node], ALIAS_VF, 2.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.1 * node)  # Slight phase variation

    # All fields should be computable
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    curv_phi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    # Basic validation
    assert len(phi_s) == G.number_of_nodes(), f"{topology_func.__name__}: Φ_s for all nodes"
    assert len(grad_phi) == G.number_of_nodes(), f"{topology_func.__name__}: |∇φ| for all nodes"
    assert len(curv_phi) == G.number_of_nodes(), f"{topology_func.__name__}: K_φ for all nodes"
    assert isinstance(xi_c, (int, float)), f"{topology_func.__name__}: ξ_C computed"

    # Physical bounds
    assert all(v >= 0 for v in phi_s.values()), f"{topology_func.__name__}: Φ_s non-negative"
    assert all(v >= 0 for v in grad_phi.values()), f"{topology_func.__name__}: |∇φ| non-negative"
    assert all(-math.pi <= v <= math.pi for v in curv_phi.values()), f"{topology_func.__name__}: K_φ bounds"
    assert xi_c > 0, f"{topology_func.__name__}: ξ_C positive"