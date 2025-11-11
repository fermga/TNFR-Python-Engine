"""Tests for REMESH boundary preservation.

Verifies that REMESH operations respect EPI boundaries and use
unified structural_clip for consistency with other operators.
"""

import pytest
import networkx as nx
from collections import deque

from tnfr.operators.remesh import apply_network_remesh
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.constants import inject_defaults


class TestREMESHBoundaryPreservation:
    """Test that REMESH respects EPI boundaries."""

    def test_remesh_clips_epi_overflow(self):
        """REMESH should clip EPI when historical mixing exceeds bounds."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)

        # Set current EPI values near upper bound
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, 0.95)

        # Create history with values that would cause overflow
        # Historical values above current
        hist = deque()
        for _ in range(10):
            snapshot = {node: 0.98 for node in G.nodes()}
            hist.append(snapshot)

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.6  # High alpha to amplify historical values
        G.graph["EPI_MAX"] = 1.0

        # Apply REMESH
        apply_network_remesh(G)

        # Verify all EPI values are within bounds
        for node in G.nodes():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert epi <= 1.0, f"Node {node} EPI {epi} exceeds EPI_MAX"
            assert epi >= -1.0, f"Node {node} EPI {epi} below EPI_MIN"

    def test_remesh_clips_epi_underflow(self):
        """REMESH should clip EPI when historical mixing goes below bounds."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)

        # Set current EPI values near lower bound
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, -0.95)

        # Create history with values that would cause underflow
        hist = deque()
        for _ in range(10):
            snapshot = {node: -0.98 for node in G.nodes()}
            hist.append(snapshot)

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.6
        G.graph["EPI_MIN"] = -1.0

        # Apply REMESH
        apply_network_remesh(G)

        # Verify all EPI values are within bounds
        for node in G.nodes():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert epi >= -1.0, f"Node {node} EPI {epi} below EPI_MIN"
            assert epi <= 1.0, f"Node {node} EPI {epi} exceeds EPI_MAX"

    def test_remesh_preserves_in_bounds_values(self):
        """REMESH should not alter values already within bounds."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3])
        inject_defaults(G)

        # Set safe middle-range values
        initial_values = {1: 0.5, 2: 0.3, 3: -0.2}
        for node, epi in initial_values.items():
            set_attr(G.nodes[node], ALIAS_EPI, epi)

        # Create history with similar safe values
        hist = deque()
        for _ in range(10):
            snapshot = {node: epi + 0.05 for node, epi in initial_values.items()}
            hist.append(snapshot)

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.3

        # Apply REMESH
        apply_network_remesh(G)

        # Verify values stayed in safe range (no clipping needed)
        for node in G.nodes():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert -1.0 <= epi <= 1.0
            # Should be close to weighted average (not clipped at boundary)
            assert abs(epi) < 0.9  # Not pushed to extremes

    def test_remesh_uses_soft_clip_mode(self):
        """REMESH should respect soft clip mode when configured."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)

        # Set values near boundary
        for node in G.nodes():
            set_attr(G.nodes[node], ALIAS_EPI, 0.92)

        # History with higher values
        hist = deque()
        for _ in range(10):
            snapshot = {node: 0.96 for node in G.nodes()}
            hist.append(snapshot)

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.5
        G.graph["EPI_MAX"] = 1.0
        G.graph["CLIP_MODE"] = "soft"  # Use soft clipping

        # Apply REMESH
        apply_network_remesh(G)

        # Verify bounds respected (soft clip should still keep within bounds)
        for node in G.nodes():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert epi <= 1.0
            assert epi >= -1.0

    def test_remesh_with_mixed_signs(self):
        """REMESH should handle nodes with mixed positive/negative EPI."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2, 3, 4])
        inject_defaults(G)

        # Mix of positive and negative values
        initial_values = {1: 0.9, 2: -0.9, 3: 0.5, 4: -0.5}
        for node, epi in initial_values.items():
            set_attr(G.nodes[node], ALIAS_EPI, epi)

        # History with extreme values
        hist = deque()
        for _ in range(10):
            snapshot = {1: 0.95, 2: -0.95, 3: 0.7, 4: -0.7}
            hist.append(snapshot)

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.7  # High alpha

        # Apply REMESH
        apply_network_remesh(G)

        # All should be within bounds
        for node in G.nodes():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert -1.0 <= epi <= 1.0

    def test_remesh_insufficient_history_no_clip_needed(self):
        """REMESH with insufficient history should not modify EPI."""
        G = nx.DiGraph()
        G.add_nodes_from([1, 2])
        inject_defaults(G)

        initial_values = {1: 0.5, 2: 0.3}
        for node, epi in initial_values.items():
            set_attr(G.nodes[node], ALIAS_EPI, epi)

        # Insufficient history
        hist = deque()
        hist.append({1: 0.6, 2: 0.4})

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4

        # Apply REMESH (should return early)
        apply_network_remesh(G)

        # Values should be unchanged
        for node, expected_epi in initial_values.items():
            epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
            assert abs(epi - expected_epi) < 1e-9


class TestREMESHUnifiedFunctions:
    """Test that REMESH uses unified structural functions."""

    def test_remesh_uses_structural_clip(self):
        """REMESH should use the unified structural_clip function."""
        # This is verified by the implementation - structural_clip is imported
        # and used in apply_network_remesh
        from tnfr.operators.remesh import apply_network_remesh
        import inspect

        source = inspect.getsource(apply_network_remesh)
        assert "structural_clip" in source, "REMESH should use unified structural_clip"
        assert "from ..dynamics.structural_clip import structural_clip" in source

    def test_remesh_respects_global_clip_mode(self):
        """REMESH should respect the global CLIP_MODE setting."""
        G = nx.DiGraph()
        G.add_node(1)
        inject_defaults(G)

        set_attr(G.nodes[1], ALIAS_EPI, 0.95)

        hist = deque()
        for _ in range(10):
            hist.append({1: 0.98})

        G.graph["_epi_hist"] = hist
        G.graph["REMESH_TAU_GLOBAL"] = 8
        G.graph["REMESH_TAU_LOCAL"] = 4
        G.graph["REMESH_ALPHA"] = 0.6

        # Test both clip modes
        for clip_mode in ["hard", "soft"]:
            G.graph["CLIP_MODE"] = clip_mode
            set_attr(G.nodes[1], ALIAS_EPI, 0.95)  # Reset

            apply_network_remesh(G)

            epi = float(get_attr(G.nodes[1], ALIAS_EPI, 0.0))
            assert epi <= 1.0, f"Failed with clip_mode={clip_mode}"
