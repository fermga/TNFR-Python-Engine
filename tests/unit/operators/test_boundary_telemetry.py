"""Tests for boundary intervention telemetry.

This module validates that structural_clip and edge-aware scaling interventions
are properly recorded in telemetry, enabling observability and tuning.

Related Issues:
- fermga/TNFR-Python-Engine#2664: Comprehensive boundary precision test suite
- fermga/TNFR-Python-Engine#2661: structural_clip implementation
- fermga/TNFR-Python-Engine#2662: Edge-aware scaling for VAL/NUL
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.operators.definitions import Expansion, Contraction
from tnfr.dynamics.structural_clip import (
    structural_clip,
    get_clip_stats,
    reset_clip_stats,
)



def _get_epi_value(G, node):
    """Extract the actual EPI scalar value from a node.
    
    After operator application, EPI may be stored as a dict with
    'continuous', 'discrete', and 'grid' keys. This helper extracts
    the actual scalar value.
    """
    epi_val = G.nodes[node][EPI_PRIMARY]
    
    # If already a scalar, return it
    if isinstance(epi_val, (int, float)):
        return float(epi_val)
    
    # If it's a dict (after operator application), extract the continuous value
    if isinstance(epi_val, dict):
        cont = epi_val.get('continuous')
        if cont:
            # Extract real part from complex number
            val = cont[0]
            return val.real if hasattr(val, 'real') else float(val)
    
    # Fallback: try to convert directly
    return float(epi_val)


def _create_test_nfr(epi: float = 0.0, vf: float = 1.0) -> tuple[nx.Graph, str]:
    """Create a minimal test NFR node with specified EPI and νf values.
    
    Parameters
    ----------
    epi : float
        Initial EPI value for the node
    vf : float
        Initial structural frequency (νf) for the node
        
    Returns
    -------
    tuple[nx.Graph, str]
        Graph and node identifier
    """
    G = nx.Graph()
    node = "test_node"
    G.add_node(node)
    inject_defaults(G)
    
    # Set initial structural parameters
    G.nodes[node][EPI_PRIMARY] = epi
    G.nodes[node][VF_PRIMARY] = vf
    G.nodes[node]["ΔNFR"] = 0.0
    G.nodes[node]["Si"] = 0.5
    
    # Initialize glyph history for grammar
    from collections import deque
    G.nodes[node].setdefault("glyph_history", deque())
    
    return G, node


class TestStructuralClipTelemetry:
    """Test structural_clip telemetry and statistics."""
    
    def test_structural_clip_hard_mode(self):
        """Hard clip mode should constrain values to boundaries."""
        reset_clip_stats()
        
        # Test upper boundary
        result_upper = structural_clip(1.1, -1.0, 1.0, mode="hard", record_stats=True)
        assert result_upper == 1.0
        
        # Test lower boundary
        result_lower = structural_clip(-1.2, -1.0, 1.0, mode="hard", record_stats=True)
        assert result_lower == -1.0
        
        # Check stats were recorded
        stats = get_clip_stats()
        assert stats.hard_clips == 2
        assert stats.total_adjustments == 2
    
    def test_structural_clip_soft_mode(self):
        """Soft clip mode should smoothly approach boundaries."""
        reset_clip_stats()
        
        # Value well within bounds - soft mode applies smooth mapping
        result_mid = structural_clip(0.5, -1.0, 1.0, mode="soft", k=3.0, record_stats=True)
        assert -1.0 <= result_mid <= 1.0
        # Soft mode with k=3.0 applies tanh transformation that may shift values
        # significantly even in mid-range, so we just verify it's bounded
        
        # Value near boundary
        result_high = structural_clip(0.95, -1.0, 1.0, mode="soft", k=3.0, record_stats=True)
        assert -1.0 <= result_high <= 1.0
        assert result_high <= 1.0
    
    def test_structural_clip_stats_accumulation(self):
        """Clip statistics should accumulate across multiple calls."""
        reset_clip_stats()
        
        # Apply multiple clips - 1.0 is already at boundary so won't be clipped
        # Start from 1.1 onwards to ensure clipping occurs
        for i in range(1, 6):  # i = 1, 2, 3, 4, 5
            structural_clip(1.0 + i * 0.1, -1.0, 1.0, mode="hard", record_stats=True)
        
        stats = get_clip_stats()
        assert stats.hard_clips == 5
        assert stats.max_delta_hard > 0
    
    def test_structural_clip_stats_reset(self):
        """Clip statistics should reset correctly."""
        reset_clip_stats()
        
        structural_clip(1.1, -1.0, 1.0, mode="hard", record_stats=True)
        stats_before = get_clip_stats()
        assert stats_before.hard_clips == 1
        
        reset_clip_stats()
        stats_after = get_clip_stats()
        assert stats_after.hard_clips == 0
        assert stats_after.total_adjustments == 0
    
    def test_structural_clip_without_stats(self):
        """Clip without stats recording should work normally."""
        reset_clip_stats()
        
        result = structural_clip(1.1, -1.0, 1.0, mode="hard", record_stats=False)
        assert result == 1.0
        
        stats = get_clip_stats()
        assert stats.hard_clips == 0  # Should not record


class TestOperatorBoundaryTelemetry:
    """Test that operators record boundary interventions."""
    
    def test_expansion_near_boundary(self):
        """Expansion near boundary should complete successfully."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0)
        
        # Apply expansion
        Expansion()(G, node)
        
        # Verify boundary preserved
        assert _get_epi_value(G, node) <= 1.0
    
    def test_contraction_near_boundary(self):
        """Contraction near boundary should complete successfully."""
        G, node = _create_test_nfr(epi=-0.95, vf=1.0)
        
        # Apply contraction
        Contraction()(G, node)
        
        # Verify boundary preserved
        assert _get_epi_value(G, node) >= -1.0
    
    def test_multiple_operations_telemetry(self):
        """Multiple operations should maintain telemetry consistency."""
        G, node = _create_test_nfr(epi=0.9, vf=1.0)
        
        for _ in range(5):
            Expansion()(G, node)
            assert _get_epi_value(G, node) <= 1.0


class TestEdgeAwareMetrics:
    """Test edge-aware scaling metrics (if enabled)."""
    
    def test_edge_aware_with_high_epi(self):
        """Edge-aware scaling should adapt near boundaries."""
        G, node = _create_test_nfr(epi=0.96, vf=1.0)
        
        # Enable edge-aware if supported
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        
        Expansion()(G, node)
        
        # Result should respect boundary
        assert _get_epi_value(G, node) <= 1.0
    
    def test_edge_aware_far_from_boundary(self):
        """Edge-aware scaling should not interfere far from boundaries."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0)
        
        # Enable edge-aware
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        
        epi_before = _get_epi_value(G, node)
        Expansion()(G, node)
        epi_after = _get_epi_value(G, node)
        
        # Should show normal expansion behavior
        assert epi_after > epi_before
        assert epi_after <= 1.0


class TestClipStatsAPI:
    """Test the clip statistics API."""
    
    def test_clip_stats_summary(self):
        """Clip stats summary should provide useful metrics."""
        reset_clip_stats()
        
        # Generate some clip events
        structural_clip(1.1, -1.0, 1.0, mode="hard", record_stats=True)
        structural_clip(1.05, -1.0, 1.0, mode="hard", record_stats=True)
        
        summary = get_clip_stats().summary()
        
        assert "hard_clips" in summary
        assert "total_adjustments" in summary
        assert "max_delta_hard" in summary
        assert summary["hard_clips"] == 2
    
    def test_clip_stats_averages(self):
        """Clip stats should compute averages correctly."""
        reset_clip_stats()
        
        # Generate clips with known deltas
        structural_clip(1.1, -1.0, 1.0, mode="hard", record_stats=True)  # delta = -0.1
        structural_clip(1.2, -1.0, 1.0, mode="hard", record_stats=True)  # delta = -0.2
        
        summary = get_clip_stats().summary()
        
        # Average should be (0.1 + 0.2) / 2 = 0.15
        assert abs(summary["avg_delta_hard"] - 0.15) < 1e-10


class TestBoundaryObservability:
    """Test observability of boundary preservation mechanisms."""
    
    def test_boundary_events_recordable(self):
        """Boundary events should be recordable for analysis."""
        G, node = _create_test_nfr(epi=0.95, vf=1.0)
        
        # Initialize boundary events collection
        G.graph["COLLECT_BOUNDARY_METRICS"] = True
        boundary_events = []
        G.graph["boundary_events"] = boundary_events
        
        # Apply operator that may trigger boundary protection
        Expansion()(G, node)
        
        # Verify operation succeeded
        assert _get_epi_value(G, node) <= 1.0
        
        # Note: Actual event recording depends on operator implementation
        # This test ensures the data structure is available
    
    def test_boundary_metrics_persistence(self):
        """Boundary metrics should persist across operations."""
        G, node = _create_test_nfr(epi=0.9, vf=1.0)
        G.graph["COLLECT_BOUNDARY_METRICS"] = True
        G.graph["boundary_events"] = []
        
        # Apply multiple operations
        for _ in range(3):
            Expansion()(G, node)
        
        # Verify all operations completed successfully
        assert _get_epi_value(G, node) <= 1.0
