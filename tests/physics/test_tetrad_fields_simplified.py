"""Simplified tests for the TNFR Structural Field Tetrad.

Tests the four canonical structural fields:
- Φ_s: Structural Potential Field (Global stability)
- |∇φ|: Phase Gradient Field (Local desynchronization) 
- K_φ: Phase Curvature Field (Geometric confinement)
- ξ_C: Coherence Length Field (Spatial correlations)

Focus on essential functionality and mathematical foundations.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.physics.fields import (
    estimate_coherence_length,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)


class TestStructuralFieldTetradBasic:
    """Test basic functionality of all four tetrad fields."""

    def setup_method(self) -> None:
        """Create basic test network."""
        self.G = nx.complete_graph(4)
        for node in self.G.nodes():
            set_attr(self.G.nodes[node], ALIAS_EPI, 1.0)
            set_attr(self.G.nodes[node], ALIAS_VF, 2.0)
            set_attr(self.G.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(self.G.nodes[node], ALIAS_THETA, 0.1 * node)

    def test_structural_potential_computation(self) -> None:
        """Test Φ_s can be computed successfully."""
        phi_s = compute_structural_potential(self.G)
        
        assert isinstance(phi_s, dict), "Φ_s should return dict"
        assert len(phi_s) == self.G.number_of_nodes(), "Φ_s for all nodes"
        assert all(v >= 0 for v in phi_s.values()), "Φ_s must be non-negative"
        assert all(isinstance(v, (int, float)) for v in phi_s.values()), "Φ_s must be numeric"

    def test_phase_gradient_computation(self) -> None:
        """Test |∇φ| can be computed successfully."""
        grad_phi = compute_phase_gradient(self.G)
        
        assert isinstance(grad_phi, dict), "|∇φ| should return dict"
        assert len(grad_phi) == self.G.number_of_nodes(), "|∇φ| for all nodes"
        assert all(v >= 0 for v in grad_phi.values()), "|∇φ| must be non-negative"
        assert all(isinstance(v, (int, float)) for v in grad_phi.values()), "|∇φ| must be numeric"

    def test_phase_curvature_computation(self) -> None:
        """Test K_φ can be computed successfully."""
        curv_phi = compute_phase_curvature(self.G)
        
        assert isinstance(curv_phi, dict), "K_φ should return dict"
        assert len(curv_phi) == self.G.number_of_nodes(), "K_φ for all nodes"
        assert all(isinstance(v, (int, float)) for v in curv_phi.values()), "K_φ must be numeric"
        # K_φ should be bounded by [-π, π]
        for node_id, curv_value in curv_phi.items():
            assert -math.pi <= curv_value <= math.pi, f"K_φ[{node_id}] = {curv_value} should be in [-π,π]"

    def test_coherence_length_computation(self) -> None:
        """Test ξ_C can be computed successfully."""
        xi_c = estimate_coherence_length(self.G)
        
        assert isinstance(xi_c, (int, float)), "ξ_C should return scalar"
        assert xi_c > 0, "ξ_C must be positive"

    def test_all_fields_simultaneously(self) -> None:
        """Test all tetrad fields can be computed together."""
        phi_s = compute_structural_potential(self.G)
        grad_phi = compute_phase_gradient(self.G)
        curv_phi = compute_phase_curvature(self.G)
        xi_c = estimate_coherence_length(self.G)

        # All should succeed
        assert phi_s is not None, "Φ_s computed"
        assert grad_phi is not None, "|∇φ| computed"
        assert curv_phi is not None, "K_φ computed"
        assert xi_c is not None, "ξ_C computed"


class TestStructuralFieldMathematicalFoundations:
    """Test mathematical foundations and thresholds."""

    def test_structural_potential_golden_ratio_bound(self) -> None:
        """Test Φ_s respects golden ratio correspondence."""
        # Create network with low ΔNFR (should be well below threshold)
        G = nx.complete_graph(3)
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, 1.0)
            set_attr(G.nodes[node], ALIAS_VF, 2.0)
            set_attr(G.nodes[node], ALIAS_DNFR, 0.01)  # Very small
            set_attr(G.nodes[node], ALIAS_THETA, 0.0)

        phi_s = compute_structural_potential(G)
        golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618

        for node_id, phi_s_value in phi_s.items():
            # For low ΔNFR networks, should be well below golden ratio
            assert phi_s_value < golden_ratio, f"Low ΔNFR: Φ_s[{node_id}] = {phi_s_value} < φ = {golden_ratio}"

    def test_phase_gradient_synchronization_detection(self) -> None:
        """Test |∇φ| detects synchronization vs desynchronization."""
        # Synchronized network
        G_sync = nx.complete_graph(3)
        for node in G_sync.nodes():
            set_attr(G_sync.nodes[node], ALIAS_EPI, 1.0)
            set_attr(G_sync.nodes[node], ALIAS_VF, 2.0)
            set_attr(G_sync.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(G_sync.nodes[node], ALIAS_THETA, 0.0)  # All same phase

        # Desynchronized network
        G_desync = nx.complete_graph(3)
        phase_values = [0.0, 2.0, 4.0]  # Large differences
        for i, node in enumerate(G_desync.nodes()):
            set_attr(G_desync.nodes[node], ALIAS_EPI, 1.0)
            set_attr(G_desync.nodes[node], ALIAS_VF, 2.0)
            set_attr(G_desync.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(G_desync.nodes[node], ALIAS_THETA, phase_values[i])

        grad_sync = compute_phase_gradient(G_sync)
        grad_desync = compute_phase_gradient(G_desync)

        # Synchronized should have lower gradients
        max_grad_sync = max(grad_sync.values()) if grad_sync else 0
        max_grad_desync = max(grad_desync.values()) if grad_desync else 0
        
        assert max_grad_desync > max_grad_sync, "Desynchronized network should have higher |∇φ|"

    def test_phase_curvature_bounds(self) -> None:
        """Test K_φ respects mathematical bounds."""
        G = nx.cycle_graph(4)  # Simple ring topology
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, 1.0)
            set_attr(G.nodes[node], ALIAS_VF, 2.0)
            set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(G.nodes[node], ALIAS_THETA, 0.5 * node)  # Linear phase progression

        curv = compute_phase_curvature(G)

        for node_id, curv_value in curv.items():
            assert -math.pi <= curv_value <= math.pi, f"K_φ[{node_id}] = {curv_value} must be in [-π,π]"

    def test_coherence_length_positive(self) -> None:
        """Test ξ_C is always positive for valid networks."""
        topologies = [nx.path_graph(4), nx.cycle_graph(4), nx.complete_graph(4)]
        
        for G in topologies:
            for node in G.nodes():
                set_attr(G.nodes[node], ALIAS_EPI, 1.0)
                set_attr(G.nodes[node], ALIAS_VF, 2.0)
                set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
                set_attr(G.nodes[node], ALIAS_THETA, 0.1)

            xi_c = estimate_coherence_length(G)
            assert xi_c > 0, f"ξ_C = {xi_c} must be positive for topology {type(G)}"


class TestStructuralFieldStabilityUnderModifications:
    """Test field stability when network is modified."""

    def setup_method(self) -> None:
        """Create test network."""
        self.G = nx.complete_graph(3)
        for node in self.G.nodes():
            set_attr(self.G.nodes[node], ALIAS_EPI, 1.0)
            set_attr(self.G.nodes[node], ALIAS_VF, 2.0)
            set_attr(self.G.nodes[node], ALIAS_DNFR, 0.1)
            set_attr(self.G.nodes[node], ALIAS_THETA, 0.1 * node)

    def test_field_response_to_dnfr_changes(self) -> None:
        """Test fields respond appropriately to ΔNFR changes."""
        # Compute initial fields
        phi_s_initial = compute_structural_potential(self.G)
        
        # Modify one node's ΔNFR
        node_id = 0
        set_attr(self.G.nodes[node_id], ALIAS_DNFR, 0.5)  # Increase significantly
        
        # Recompute
        phi_s_modified = compute_structural_potential(self.G)
        
        # Structural potential should change for the modified node
        assert phi_s_modified[node_id] != phi_s_initial[node_id], "Φ_s should reflect ΔNFR change"

    def test_field_response_to_phase_changes(self) -> None:
        """Test phase-based fields respond to phase changes."""
        # Initial phase gradient
        grad_initial = compute_phase_gradient(self.G)
        
        # Change phases to create larger differences
        set_attr(self.G.nodes[0], ALIAS_THETA, 0.0)
        set_attr(self.G.nodes[1], ALIAS_THETA, 3.0)  # Large difference
        set_attr(self.G.nodes[2], ALIAS_THETA, 6.0)
        
        # Recompute
        grad_modified = compute_phase_gradient(self.G)
        
        # Some gradients should change (though specific changes depend on implementation)
        gradients_changed = any(
            abs(grad_modified[node] - grad_initial[node]) > 0.01
            for node in self.G.nodes()
        )
        assert gradients_changed, "Phase gradient should respond to phase changes"


@pytest.mark.parametrize("topology,size", [
    (nx.path_graph, 5),
    (nx.cycle_graph, 5),
    (nx.complete_graph, 4),
    (nx.star_graph, 4),
])
def test_tetrad_cross_topology_robustness(topology: type, size: int) -> None:
    """Test tetrad fields work across different topologies."""
    G = topology(size)
    
    # Initialize with standard TNFR attributes
    for node in G.nodes():
        set_attr(G.nodes[node], ALIAS_EPI, 1.0)
        set_attr(G.nodes[node], ALIAS_VF, 2.0)
        set_attr(G.nodes[node], ALIAS_DNFR, 0.1)
        set_attr(G.nodes[node], ALIAS_THETA, 0.1 * node)

    # All fields should be computable without errors
    phi_s = compute_structural_potential(G)
    grad_phi = compute_phase_gradient(G)
    curv_phi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    # Basic validation
    assert len(phi_s) == G.number_of_nodes(), f"{topology.__name__}: Φ_s computed"
    assert len(grad_phi) == G.number_of_nodes(), f"{topology.__name__}: |∇φ| computed"  
    assert len(curv_phi) == G.number_of_nodes(), f"{topology.__name__}: K_φ computed"
    assert isinstance(xi_c, (int, float)), f"{topology.__name__}: ξ_C computed"
    assert xi_c > 0, f"{topology.__name__}: ξ_C positive"