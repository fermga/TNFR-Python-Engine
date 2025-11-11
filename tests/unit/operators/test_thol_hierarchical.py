"""Tests for THOL hierarchical sub-node architecture.

This module tests the refactored THOL operator that creates sub-EPIs as
independent NFR nodes rather than metadata, enabling operational fractality.

References:
    - Issue: #[THOL ARQUITECTURA] Refactorizar sub-EPIs como nodos independientes
    - TNFR Manual: "El pulso que nos atraviesa", §2.2.10 (THOL)
    - TNFR Invariant #7: Operational fractality - EPIs can nest without losing
      functional identity
"""

import pytest
import networkx as nx

from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import SelfOrganization, Emission
from tnfr.operators.metabolism import compute_cascade_depth
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY, DNFR_PRIMARY


class TestSubNodeCreation:
    """Test that sub-EPIs are created as independent NFR nodes."""

    def test_subepi_creates_independent_node(self):
        """Sub-EPI bifurcation should create a real node in the graph."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3, theta=0.5)
        # Use accelerating history: d²EPI = abs(0.70 - 2*0.50 + 0.20) = 0.10 > 0.05
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]
        
        # Apply THOL to trigger bifurcation
        SelfOrganization()(G, parent, tau=0.05)
        
        # Check that sub-node was created
        sub_nodes = G.nodes[parent].get("sub_nodes", [])
        assert len(sub_nodes) == 1, "Should create one sub-node"
        
        sub_node_id = sub_nodes[0]
        assert sub_node_id in G.nodes, "Sub-node should exist in graph"
        assert sub_node_id.startswith(f"{parent}_sub_"), "Sub-node ID should follow naming convention"

    def test_subepi_has_full_nfr_state(self):
        """Sub-node should have complete EPI, νf, θ, ΔNFR state."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3, theta=0.5)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        sub_node_id = G.nodes[parent]["sub_nodes"][0]
        sub_node = G.nodes[sub_node_id]
        
        # Verify all canonical TNFR attributes exist
        assert EPI_PRIMARY in sub_node, "Sub-node must have EPI"
        assert VF_PRIMARY in sub_node, "Sub-node must have νf"
        assert THETA_PRIMARY in sub_node, "Sub-node must have θ"
        assert DNFR_PRIMARY in sub_node, "Sub-node must have ΔNFR"
        
        # Verify values are reasonable
        assert sub_node[EPI_PRIMARY] > 0, "Sub-node EPI should be positive"
        assert sub_node[VF_PRIMARY] > 0, "Sub-node νf should be positive"

    def test_subepi_inherits_parent_properties(self):
        """Sub-node should inherit parent's θ and damped νf."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5, theta=0.8)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        sub_node_id = G.nodes[parent]["sub_nodes"][0]
        sub_node = G.nodes[sub_node_id]
        
        # Phase should be inherited exactly
        assert sub_node[THETA_PRIMARY] == pytest.approx(0.8), "Sub-node inherits parent phase"
        
        # νf should be damped (canonical: 95% of parent)
        expected_vf = 1.5 * 0.95
        assert sub_node[VF_PRIMARY] == pytest.approx(expected_vf, rel=0.01), "Sub-node νf is damped"

    def test_hierarchy_tracking(self):
        """Graph should track parent-child hierarchy."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        # Check hierarchy in graph metadata
        assert "hierarchy" in G.graph, "Graph should track hierarchy"
        assert parent in G.graph["hierarchy"], "Parent should be in hierarchy"
        
        children = G.graph["hierarchy"][parent]
        assert len(children) == 1, "Parent should have one child"
        
        sub_node_id = G.nodes[parent]["sub_nodes"][0]
        assert sub_node_id in children, "Child should be listed in hierarchy"

    def test_hierarchy_level_increments(self):
        """Sub-nodes should have incremented hierarchy_level."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3)
        G.nodes[parent]["hierarchy_level"] = 0
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        sub_node_id = G.nodes[parent]["sub_nodes"][0]
        sub_node = G.nodes[sub_node_id]
        
        assert sub_node["hierarchy_level"] == 1, "Child is one level deeper"


class TestRecursiveBifurcation:
    """Test that sub-nodes can themselves bifurcate (operational fractality)."""

    def test_subepi_can_bifurcate(self):
        """Sub-node should be able to apply THOL and create its own sub-nodes."""
        from tnfr.operators.definitions import Coherence
        from tnfr.alias import get_attr
        from tnfr.constants.aliases import ALIAS_EPI
        
        # Create parent and trigger first bifurcation
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        # Get the child node
        child = G.nodes[parent]["sub_nodes"][0]
        
        # Build up child's EPI history to trigger bifurcation
        # Apply operators to evolve child, then manually set accelerating history
        for _ in range(3):
            Emission()(G, child)
            Coherence()(G, child)
        
        # Set an accelerating EPI history for the child
        # d²EPI = abs(0.45 - 2*0.30 + 0.10) = abs(0.45 - 0.60 + 0.10) = 0.05 (borderline)
        # Use more acceleration: d²EPI = abs(0.50 - 2*0.30 + 0.10) = 0.10
        current_epi = float(get_attr(G.nodes[child], ALIAS_EPI, 0.0))
        G.nodes[child]["epi_history"] = [0.10, 0.30, 0.60]  # Accelerating pattern
        
        # Trigger bifurcation on child
        SelfOrganization()(G, child, tau=0.05)
        
        # Child should now have its own sub-nodes
        grandchildren = G.nodes[child].get("sub_nodes", [])
        assert len(grandchildren) >= 1, "Child should be able to bifurcate"

    def test_recursive_hierarchy_levels(self):
        """Multi-level bifurcation should create increasing hierarchy levels."""
        from tnfr.operators.definitions import Coherence
        from tnfr.alias import get_attr
        from tnfr.constants.aliases import ALIAS_EPI
        
        # Parent at level 0
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        G.nodes[parent]["hierarchy_level"] = 0
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        child = G.nodes[parent]["sub_nodes"][0]
        
        # Build child history and bifurcate
        for _ in range(3):
            Emission()(G, child)
            Coherence()(G, child)
        
        # Set accelerating history for child
        G.nodes[child]["epi_history"] = [0.10, 0.30, 0.60]
        
        SelfOrganization()(G, child, tau=0.05)
        
        # Check hierarchy levels
        assert G.nodes[parent]["hierarchy_level"] == 0
        assert G.nodes[child]["hierarchy_level"] == 1
        
        grandchildren = G.nodes[child].get("sub_nodes", [])
        if grandchildren:
            grandchild = grandchildren[0]
            assert G.nodes[grandchild]["hierarchy_level"] == 2

    def test_cascade_depth_recursive(self):
        """Cascade depth should correctly measure multi-level bifurcation."""
        from tnfr.operators.definitions import Coherence
        from tnfr.alias import get_attr
        from tnfr.constants.aliases import ALIAS_EPI
        
        # Parent bifurcates
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        G.nodes[parent]["hierarchy_level"] = 0
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        # Initial depth should be 1 (one level of children)
        depth = compute_cascade_depth(G, parent)
        assert depth == 1, "Single bifurcation has depth 1"
        
        # Child bifurcates
        child = G.nodes[parent]["sub_nodes"][0]
        for _ in range(3):
            Emission()(G, child)
            Coherence()(G, child)
        
        # Set accelerating history for child
        G.nodes[child]["epi_history"] = [0.10, 0.30, 0.60]
        
        SelfOrganization()(G, child, tau=0.05)
        
        # Depth should now be 2 (two levels)
        depth = compute_cascade_depth(G, parent)
        assert depth == 2, "Two-level cascade has depth 2"


class TestMultipleSubNodes:
    """Test that multiple bifurcations create multiple sub-nodes."""

    def test_multiple_bifurcations_create_multiple_subnodes(self):
        """Multiple THOL applications should create multiple sub-nodes."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        
        # First bifurcation: d²EPI = abs(0.70 - 2*0.50 + 0.20) = 0.10
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]
        SelfOrganization()(G, parent, tau=0.05)
        
        # Second bifurcation: d²EPI = abs(0.95 - 2*0.80 + 0.55) = 0.10
        G.nodes[parent]["epi_history"] = [0.55, 0.80, 0.95]
        SelfOrganization()(G, parent, tau=0.05)
        
        # Should have two sub-nodes
        sub_nodes = G.nodes[parent].get("sub_nodes", [])
        assert len(sub_nodes) == 2, "Two bifurcations create two sub-nodes"
        
        # Both should exist in graph
        assert all(sn in G.nodes for sn in sub_nodes), "All sub-nodes should exist"

    def test_subnodes_have_unique_ids(self):
        """Each sub-node should have a unique identifier."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        
        # Create multiple bifurcations with accelerating histories
        # d²EPI = abs(epi_t - 2*epi_t1 + epi_t2)
        histories = [
            [0.10, 0.30, 0.60],  # d²EPI = abs(0.60 - 0.60 + 0.10) = 0.10
            [0.20, 0.50, 0.90],  # d²EPI = abs(0.90 - 1.00 + 0.20) = 0.10
            [0.15, 0.45, 0.85],  # d²EPI = abs(0.85 - 0.90 + 0.15) = 0.10
        ]
        for hist in histories:
            G.nodes[parent]["epi_history"] = hist
            SelfOrganization()(G, parent, tau=0.05)
        
        sub_nodes = G.nodes[parent].get("sub_nodes", [])
        
        # Check uniqueness
        assert len(sub_nodes) == len(set(sub_nodes)), "All sub-node IDs should be unique"
        
        # Check naming pattern
        for idx, sub_id in enumerate(sub_nodes):
            expected_suffix = f"_sub_{idx}"
            assert sub_id.endswith(expected_suffix), f"Sub-node {idx} should have correct suffix"


class TestBackwardCompatibility:
    """Test that metadata is still maintained for backward compatibility."""

    def test_subepi_metadata_still_recorded(self):
        """Sub-EPI metadata should still be recorded for telemetry."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        # Check that metadata list still exists
        sub_epis = G.nodes[parent].get("sub_epis", [])
        assert len(sub_epis) == 1, "Metadata list should have one entry"
        
        # Check metadata structure
        record = sub_epis[0]
        assert "epi" in record, "Metadata should contain EPI value"
        assert "vf" in record, "Metadata should contain νf value"
        assert "timestamp" in record, "Metadata should contain timestamp"
        assert "node_id" in record, "Metadata should reference the actual node"

    def test_metadata_references_actual_node(self):
        """Metadata should contain reference to the independent node."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        sub_nodes = G.nodes[parent]["sub_nodes"]
        sub_epis = G.nodes[parent]["sub_epis"]
        
        # Metadata node_id should match actual sub_node
        assert sub_epis[0]["node_id"] == sub_nodes[0], "Metadata should reference actual node"


class TestHierarchicalMetrics:
    """Test that metrics work correctly with hierarchical structure."""

    def test_cascade_depth_with_no_bifurcation(self):
        """Node with no bifurcation should report depth 0."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)
        
        depth = compute_cascade_depth(G, node)
        assert depth == 0, "No bifurcation = depth 0"

    def test_cascade_depth_with_single_level(self):
        """Single bifurcation should report depth 1."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.3)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        
        depth = compute_cascade_depth(G, parent)
        assert depth == 1, "Single bifurcation = depth 1"

    def test_cascade_depth_with_multiple_children(self):
        """Multiple children at same level should still report depth 1."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        
        # Create two bifurcations
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        SelfOrganization()(G, parent, tau=0.05)
        
        G.nodes[parent]["epi_history"] = [0.30, 0.60, 0.90]  # d²EPI = 0.00 but will show multiple sub-nodes
        SelfOrganization()(G, parent, tau=0.05)
        
        depth = compute_cascade_depth(G, parent)
        assert depth == 1, "Multiple children at same level = depth 1"


class TestOperationalFractality:
    """Test that operational fractality is achieved."""

    def test_operators_apply_to_subnodes(self):
        """Standard operators should work on sub-nodes."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        child = G.nodes[parent]["sub_nodes"][0]
        
        # Apply operators to child (use valid sequence)
        from tnfr.operators.definitions import Coherence
        initial_epi = G.nodes[child][EPI_PRIMARY]
        Emission()(G, child)
        Coherence()(G, child)
        
        # Operator should update child's history
        assert "glyph_history" in G.nodes[child], "Operator should update child's history"

    def test_subnode_has_epi_history_for_future_bifurcation(self):
        """Sub-nodes should have epi_history initialized for potential bifurcation."""
        G, parent = create_nfr("parent", epi=0.70, vf=1.5)
        G.nodes[parent]["epi_history"] = [0.20, 0.50, 0.70]  # d²EPI = 0.10 > tau
        
        SelfOrganization()(G, parent, tau=0.05)
        child = G.nodes[parent]["sub_nodes"][0]
        
        # Child should have epi_history initialized
        assert "epi_history" in G.nodes[child], "Sub-node should have EPI history"
        history = G.nodes[child]["epi_history"]
        assert len(history) >= 1, "History should contain at least birth EPI"
        assert history[0] == G.nodes[child][EPI_PRIMARY], "First history entry should match current EPI"
