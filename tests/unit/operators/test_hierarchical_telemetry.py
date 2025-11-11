"""Tests for hierarchical depth telemetry in nested THOL bifurcations.

This module tests the implementation of explicit bifurcation level tracking,
hierarchy path recording, and depth validation for nested THOL structures,
supporting operational fractality analysis and debugging.

References:
    - Issue: [THOL][Enhancement] Implementar telemetría de profundidad jerárquica
    - THOL_ENCAPSULATION_GUIDE.md: Section "Nested THOL"
    - AGENTS.md: Invariant #7 (Operational Fractality)
"""

import logging
import pytest
import networkx as nx

from tnfr.structural import create_nfr
from tnfr.operators.definitions import SelfOrganization, Emission, Coherence
from tnfr.operators.metabolism import compute_hierarchical_depth
from tnfr.visualization.hierarchy import print_bifurcation_hierarchy, get_hierarchy_info
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.alias import get_attr


class TestBifurcationLevelTracking:
    """Test bifurcation_level field in sub-EPI records."""

    def test_single_level_bifurcation_has_level_1(self):
        """First bifurcation creates sub-EPI with level 1."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Set up for bifurcation
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]

        # Apply THOL
        SelfOrganization()(G, node, tau=0.05)

        # Check sub-EPI has bifurcation_level
        sub_epis = G.nodes[node].get("sub_epis", [])
        assert len(sub_epis) == 1, "Should create one sub-EPI"

        sub_epi = sub_epis[0]
        assert "bifurcation_level" in sub_epi, "Should have bifurcation_level field"
        assert sub_epi["bifurcation_level"] == 1, "First level should be 1"

    def test_nested_bifurcation_increments_level(self):
        """Nested bifurcation creates sub-EPI with incremented level."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # First bifurcation
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Get the sub-node
        sub_epis = G.nodes[node].get("sub_epis", [])
        sub_node_id = sub_epis[0]["node_id"]

        # Prepare sub-node for bifurcation with proper acceleration
        # d²EPI/dt² = EPI_t - 2*EPI_{t-1} + EPI_{t-2}
        # = 0.20 - 2*0.15 + 0.05 = 0.20 - 0.30 + 0.05 = -0.05 (abs = 0.05)
        # We need abs(d2_epi) > tau, so let's make it 0.10
        # 0.25 - 2*0.15 + 0.05 = 0.25 - 0.30 + 0.05 = 0.0 (not enough)
        # Let's use: 0.30 - 2*0.15 + 0.05 = 0.30 - 0.30 + 0.05 = 0.05 (borderline)
        # Better: 0.35 - 2*0.15 + 0.05 = 0.35 - 0.30 + 0.05 = 0.10 (good!)
        assert sub_node_id in G.nodes, "Sub-node should exist"
        G.nodes[sub_node_id]["epi_history"] = [0.05, 0.15, 0.35]

        # Second bifurcation (nested)
        SelfOrganization()(G, sub_node_id, tau=0.05)

        # Check nested sub-EPI has level 2
        nested_sub_epis = G.nodes[sub_node_id].get("sub_epis", [])
        assert len(nested_sub_epis) == 1, "Sub-node should create sub-EPI"

        nested_sub_epi = nested_sub_epis[0]
        assert nested_sub_epi["bifurcation_level"] == 2, "Second level should be 2"

    def test_three_level_bifurcation_has_level_3(self):
        """Three-level nested bifurcation reaches level 3."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Level 1
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Level 2
        sub_node_1 = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node_1]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node_1, tau=0.05)

        # Level 3
        sub_node_2 = G.nodes[sub_node_1]["sub_epis"][0]["node_id"]
        G.nodes[sub_node_2]["epi_history"] = [0.02, 0.08, 0.20]
        SelfOrganization()(G, sub_node_2, tau=0.05)

        # Verify level 3
        sub_epis_3 = G.nodes[sub_node_2].get("sub_epis", [])
        assert len(sub_epis_3) == 1, "Level 3 should create sub-EPI"
        assert sub_epis_3[0]["bifurcation_level"] == 3, "Third level should be 3"


class TestHierarchyPathTracking:
    """Test hierarchy_path field recording parent chain."""

    def test_root_node_has_empty_path(self):
        """Root node (level 0) has empty hierarchy path."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        # Initial state - no bifurcation yet
        path = G.nodes[node].get("_hierarchy_path", [])
        assert path == [], "Root should have empty path"

    def test_first_bifurcation_records_parent_in_path(self):
        """First bifurcation records parent node in hierarchy_path."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Bifurcate
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Check path in sub-EPI metadata
        sub_epi = G.nodes[node]["sub_epis"][0]
        assert "hierarchy_path" in sub_epi, "Should have hierarchy_path"
        assert sub_epi["hierarchy_path"] == [node], "Should contain parent"

    def test_nested_bifurcation_builds_full_path(self):
        """Nested bifurcations build complete ancestry chain."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Level 1
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)
        sub_node_1 = G.nodes[node]["sub_epis"][0]["node_id"]

        # Level 2
        G.nodes[sub_node_1]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node_1, tau=0.05)
        sub_node_2 = G.nodes[sub_node_1]["sub_epis"][0]["node_id"]

        # Check level 2 path includes both ancestors
        sub_epi_2 = G.nodes[sub_node_1]["sub_epis"][0]
        expected_path = [node, sub_node_1]
        assert (
            sub_epi_2["hierarchy_path"] == expected_path
        ), f"Level 2 should have full path: {expected_path}"


class TestComputeHierarchicalDepth:
    """Test compute_hierarchical_depth() function."""

    def test_no_bifurcation_returns_zero(self):
        """Node with no sub-EPIs returns depth 0."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        depth = compute_hierarchical_depth(G, node)
        assert depth == 0, "No bifurcation should have depth 0"

    def test_single_bifurcation_returns_one(self):
        """Single-level bifurcation returns depth 1."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        depth = compute_hierarchical_depth(G, node)
        assert depth == 1, "Single bifurcation should have depth 1"

    def test_nested_bifurcation_returns_max_level(self):
        """Nested bifurcation returns maximum level."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Create 2-level structure
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node, tau=0.05)

        # Root should report depth 2 (maximum level in tree)
        depth = compute_hierarchical_depth(G, node)
        assert depth == 2, "Two-level bifurcation should have depth 2"

    def test_multiple_branches_returns_max_depth(self):
        """Multiple branches returns deepest level."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # First bifurcation - creates 1 sub-EPI
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Second bifurcation - creates another sub-EPI at level 1
        current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        G.nodes[node]["epi_history"].append(current_epi)
        G.nodes[node]["epi_history"].append(current_epi + 0.15)
        SelfOrganization()(G, node, tau=0.05)

        # Nested bifurcation in first sub-node (reaches level 2)
        sub_node_1 = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node_1]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node_1, tau=0.05)

        # Depth should be 2 (from deepest branch)
        depth = compute_hierarchical_depth(G, node)
        assert depth == 2, "Should return deepest branch depth"


class TestMaxDepthValidation:
    """Test THOL_MAX_BIFURCATION_DEPTH validation."""

    def test_no_warning_below_max_depth(self, caplog):
        """No warning when bifurcation level below maximum."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)
        G.graph["THOL_MAX_BIFURCATION_DEPTH"] = 3

        # First bifurcation (level 1)
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]

        with caplog.at_level(logging.WARNING):
            SelfOrganization()(G, node, tau=0.05)

        assert "maximum" not in caplog.text.lower(), "Should not warn when below max depth"

    def test_warning_at_max_depth(self, caplog):
        """Warning issued when bifurcating at max bifurcation depth."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)
        G.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1  # Max at level 1

        # Build to level 1
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Try to bifurcate at level 1 (which would create level 2, exceeding max)
        sub_node_1 = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node_1]["epi_history"] = [0.05, 0.15, 0.35]

        with caplog.at_level(logging.WARNING):
            SelfOrganization()(G, sub_node_1, tau=0.05)

        # Should warn about depth
        assert "depth" in caplog.text.lower(), "Should warn about depth"
        assert "maximum" in caplog.text.lower(), "Should mention maximum"

    def test_warning_recorded_in_node(self):
        """Depth warning recorded in node attributes."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)
        G.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1

        # Build to level 1 (at max)
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node, tau=0.05)

        # Check warning flag
        assert G.nodes[sub_node].get(
            "_thol_max_depth_warning", False
        ), "Warning should be recorded in node"

    def test_warning_recorded_in_graph_events(self):
        """Depth warnings recorded in graph metadata."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)
        G.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1

        # Build to max
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node, tau=0.05)

        # Check graph events
        events = G.graph.get("thol_depth_warnings", [])
        assert len(events) > 0, "Should record depth warning event"
        assert events[0]["node"] == sub_node, "Should record correct node"
        assert events[0]["depth"] == 1, "Should record depth"


class TestHierarchyVisualization:
    """Test print_bifurcation_hierarchy and get_hierarchy_info."""

    def test_print_single_level(self, capsys):
        """Print single-level bifurcation."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Print hierarchy
        print_bifurcation_hierarchy(G, node)

        captured = capsys.readouterr()
        assert "Node test" in captured.out, "Should print root node"
        assert "Sub-EPI 1" in captured.out, "Should print sub-EPI"
        assert "level=1" in captured.out, "Should show level"

    def test_print_nested_bifurcation(self, capsys):
        """Print nested bifurcation structure."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Level 1
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Level 2
        sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node, tau=0.05)

        # Print
        print_bifurcation_hierarchy(G, node)

        captured = capsys.readouterr()
        assert "level=0" in captured.out, "Should show root level"
        assert "level=1" in captured.out, "Should show first level"
        assert "level=2" in captured.out, "Should show second level"

    def test_get_hierarchy_info_single_level(self):
        """Get hierarchy info for single-level bifurcation."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        info = get_hierarchy_info(G, node)

        assert info["node"] == node, "Should identify node"
        assert info["bifurcation_level"] == 0, "Root should be level 0"
        assert info["sub_epi_count"] == 1, "Should count sub-EPIs"
        assert info["max_depth"] == 1, "Should report max depth"
        assert info["total_descendants"] == 1, "Should count descendants"

    def test_get_hierarchy_info_nested(self):
        """Get hierarchy info for nested bifurcation."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        # Level 1
        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        # Level 2
        sub_node = G.nodes[node]["sub_epis"][0]["node_id"]
        G.nodes[sub_node]["epi_history"] = [0.05, 0.15, 0.35]
        SelfOrganization()(G, sub_node, tau=0.05)

        info = get_hierarchy_info(G, node)

        assert info["max_depth"] == 2, "Should detect 2-level depth"
        assert info["total_descendants"] == 2, "Should count all descendants"


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_sub_epi_without_level_defaults_to_one(self):
        """Sub-EPIs without bifurcation_level default to 1."""
        G, node = create_nfr("test", epi=0.50, vf=1.0)

        # Manually create sub-EPI without bifurcation_level (legacy)
        G.nodes[node]["sub_epis"] = [{"epi": 0.15, "vf": 1.0}]

        # compute_hierarchical_depth should handle gracefully
        depth = compute_hierarchical_depth(G, node)
        assert depth == 1, "Should default to 1 for backward compatibility"

    def test_existing_telemetry_fields_preserved(self):
        """New fields don't break existing sub-EPI structure."""
        G, node = create_nfr("test", epi=0.50, vf=1.0, theta=0.1)

        G.nodes[node]["epi_history"] = [0.20, 0.38, 0.50]
        SelfOrganization()(G, node, tau=0.05)

        sub_epi = G.nodes[node]["sub_epis"][0]

        # All existing fields should be present
        assert "epi" in sub_epi, "Should have epi"
        assert "vf" in sub_epi, "Should have vf"
        assert "timestamp" in sub_epi, "Should have timestamp"
        assert "d2_epi" in sub_epi, "Should have d2_epi"
        assert "tau" in sub_epi, "Should have tau"
        assert "node_id" in sub_epi, "Should have node_id"
        assert "metabolized" in sub_epi, "Should have metabolized"

        # New fields added
        assert "bifurcation_level" in sub_epi, "Should have bifurcation_level"
        assert "hierarchy_path" in sub_epi, "Should have hierarchy_path"
