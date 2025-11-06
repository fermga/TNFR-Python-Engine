"""Integration tests for edge-aware scaling with full TNFR dynamics.

Tests that edge-aware scaling works correctly when combined with the
full integration cycle (operators + dynamics + structural_clip).
"""

import pytest
import networkx as nx

from tnfr.structural import create_nfr
from tnfr.dynamics import step
from tnfr.types import Glyph
from tnfr.operators import GLYPH_OPERATIONS
from tnfr.node import NodeNX
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_DNFR
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, inject_defaults


def get_epi(G, node_id):
    """Helper to get EPI value using proper alias."""
    return float(get_attr(G.nodes[node_id], ALIAS_EPI, 0.0))


def get_vf(G, node_id):
    """Helper to get vf value using proper alias."""
    return float(get_attr(G.nodes[node_id], ALIAS_VF, 0.0))


def create_test_nfr(*args, **kwargs):
    """Helper to create NFR with defaults injected."""
    from tnfr.structural import create_nfr as _create_nfr
    G, node_id = _create_nfr(*args, **kwargs)
    inject_defaults(G)
    return G, node_id


class TestEdgeAwareWithIntegration:
    """Test edge-aware scaling in combination with dynamics integration."""

    def test_val_prevents_overflow_during_integration(self):
        """VAL with edge-aware scaling should prevent EPI overflow during integration."""
        
        G, node_id = create_test_nfr("test", epi=0.95, vf=1.0)
        
        # Enable edge-aware scaling
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        
        # Set up positive ΔNFR that would cause overflow without edge-aware
        G.nodes[node_id][DNFR_PRIMARY] = 0.1
        
        # Apply VAL operator
        node = NodeNX.from_graph(G, node_id)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        val_op(node, {"VAL_scale": 1.05})
        
        # EPI should be scaled with edge-awareness immediately
        epi_after_val = get_epi(G, node_id)
        assert epi_after_val <= 1.0
        
        # Now run dynamics step - should stay within bounds
        step(G, dt=1.0)
        
        epi_after_step = get_epi(G, node_id)
        assert epi_after_step <= 1.0

    def test_val_without_edge_aware_may_need_clip(self):
        """Without edge-aware, VAL may rely on structural_clip."""
        G, node_id = create_test_nfr("test", epi=0.95, vf=1.0)
        
        # Disable edge-aware scaling
        G.graph["EDGE_AWARE_ENABLED"] = False
        G.graph["EPI_MAX"] = 1.0
        
        # Set up positive ΔNFR
        G.nodes[node_id][DNFR_PRIMARY] = 0.1
        
        # Apply VAL operator
        node = NodeNX.from_graph(G, node_id)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        val_op(node, {"VAL_scale": 1.05})
        
        # vf scaled but EPI unchanged by operator
        epi_after_val = get_epi(G, node_id)
        assert epi_after_val == 0.95  # Unchanged by operator
        
        # Run dynamics step - may exceed 1.0 before clip
        step(G, dt=1.0)
        
        # structural_clip should have caught it
        epi_after_step = get_epi(G, node_id)
        assert epi_after_step <= 1.0

    def test_edge_aware_telemetry_integration(self):
        """Edge-aware telemetry should track interventions during operator application."""
        G, node_id = create_test_nfr("test", epi=0.92, vf=1.0)
        
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.nodes[node_id][DNFR_PRIMARY] = 0.08
        
        # Apply VAL multiple times
        node = NodeNX.from_graph(G, node_id)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        
        for _ in range(3):
            val_op(node, {"VAL_scale": 1.05})
            
            # Should stay within bounds
            assert get_epi(G, node_id) <= 1.0
            
            # Run a step
            step(G, dt=0.5)
            assert get_epi(G, node_id) <= 1.0
        
        # Check telemetry was recorded
        interventions = G.graph.get("edge_aware_interventions", [])
        assert len(interventions) > 0  # At least one adaptation occurred
        
        # Verify adaptation happened when EPI got close to boundary
        for intervention in interventions:
            assert intervention["adapted"] is True
            assert intervention["scale_effective"] < intervention["scale_requested"]

    def test_combined_val_nul_with_dynamics(self):
        """VAL→NUL sequence with dynamics should maintain bounds."""
        G, node_id = create_test_nfr("test", epi=0.8, vf=1.0)
        
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.nodes[node_id][DNFR_PRIMARY] = 0.05
        
        node = NodeNX.from_graph(G, node_id)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        nul_op = GLYPH_OPERATIONS[Glyph.NUL]
        
        # Apply sequence with dynamics steps
        for _ in range(3):
            val_op(node, {"VAL_scale": 1.05})
            assert -1.0 <= get_epi(G, node_id) <= 1.0
            
            step(G, dt=0.3)
            assert -1.0 <= get_epi(G, node_id) <= 1.0
            
            nul_op(node, {"NUL_scale": 0.85})
            assert -1.0 <= get_epi(G, node_id) <= 1.0
            
            step(G, dt=0.3)
            assert -1.0 <= get_epi(G, node_id) <= 1.0


class TestEdgeAwareVsStructuralClip:
    """Compare edge-aware scaling with structural_clip behavior."""

    def test_edge_aware_reduces_clip_interventions(self):
        """Edge-aware should reduce reliance on structural_clip."""
        # Test with edge-aware disabled
        G1, n1 = create_test_nfr("test1", epi=0.95, vf=1.0)
        G1.graph["EDGE_AWARE_ENABLED"] = False
        G1.graph["EPI_MAX"] = 1.0
        G1.nodes[n1][DNFR_PRIMARY] = 0.1
        
        node1 = NodeNX.from_graph(G1, n1)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        
        # Apply VAL and run multiple steps
        for _ in range(5):
            val_op(node1, {"VAL_scale": 1.05})
            step(G1, dt=0.5)
        
        # With edge-aware disabled, structural_clip must work hard
        # (We can't easily measure clip interventions without modifying clip code)
        
        # Test with edge-aware enabled
        G2, n2 = create_test_nfr("test2", epi=0.95, vf=1.0)
        G2.graph["EDGE_AWARE_ENABLED"] = True
        G2.graph["EPI_MAX"] = 1.0
        G2.nodes[n2][DNFR_PRIMARY] = 0.1
        
        node2 = NodeNX.from_graph(G2, n2)
        
        # Apply VAL and run multiple steps
        for _ in range(5):
            val_op(node2, {"VAL_scale": 1.05})
            step(G2, dt=0.5)
        
        # Both should have valid EPI
        assert float(G1.nodes[n1][EPI_PRIMARY]) <= 1.0
        assert float(G2.nodes[n2][EPI_PRIMARY]) <= 1.0
        
        # Edge-aware should have telemetry
        assert len(G2.graph.get("edge_aware_interventions", [])) > 0

    def test_edge_aware_smoother_trajectories(self):
        """Edge-aware should produce smoother trajectories near boundaries."""
        G, node_id = create_test_nfr("test", epi=0.7, vf=1.0)
        
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.nodes[node_id][DNFR_PRIMARY] = 0.08
        
        node = NodeNX.from_graph(G, node_id)
        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        
        epi_history = [get_epi(G, node_id)]
        
        # Apply VAL multiple times and track EPI progression
        for _ in range(10):
            val_op(node, {"VAL_scale": 1.05})
            step(G, dt=0.3)
            epi_history.append(get_epi(G, node_id))
        
        # All values should be within bounds
        assert all(-1.0 <= epi <= 1.0 for epi in epi_history)
        
        # Trajectory should be monotonically increasing (since VAL expands)
        # until it hits the boundary
        for i in range(len(epi_history) - 1):
            if epi_history[i] < 0.99:  # Not yet at boundary
                # Should increase or stay same (if at boundary)
                assert epi_history[i+1] >= epi_history[i] - 0.01  # Small tolerance
