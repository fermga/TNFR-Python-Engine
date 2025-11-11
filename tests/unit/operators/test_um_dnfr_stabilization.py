"""Unit tests for ΔNFR reduction through mutual stabilization in UM operator.

This module tests the canonical TNFR requirement that coupling should reduce
reorganization pressure (ΔNFR) through mutual stabilization. The reduction is
proportional to phase alignment strength: well-coupled nodes experience stronger
ΔNFR reduction, promoting structural coherence.
"""

import math

import pytest

from tnfr.operators import apply_glyph
from tnfr.structural import create_nfr


def test_um_reduces_dnfr_basic(graph_canon):
    """Test that UM coupling reduces ΔNFR in aligned nodes."""
    G = graph_canon()
    
    # Create two nodes with aligned phases and positive ΔNFR
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=0.5, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=0.5, vf=1.0)
    G.add_edge(0, 1)
    
    # Enable ΔNFR stabilization (default)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    
    # Apply UM operator
    apply_glyph(G, 0, "UM")
    
    dnfr_after = G.nodes[0]["dnfr"]
    
    # ΔNFR should be reduced due to coupling
    assert dnfr_after < dnfr_before, "ΔNFR should decrease after coupling"
    
    # Verify the reduction is reasonable (not too extreme)
    reduction_pct = (dnfr_before - dnfr_after) / dnfr_before
    assert 0.0 < reduction_pct < 0.2, "ΔNFR reduction should be moderate"


def test_um_dnfr_reduction_proportional_to_alignment(graph_canon):
    """Test that ΔNFR reduction is stronger for better-aligned nodes."""
    G = graph_canon()
    
    # Test case 1: Well-aligned nodes (small phase difference)
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before_aligned = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after_aligned = G.nodes[0]["dnfr"]
    reduction_aligned = dnfr_before_aligned - dnfr_after_aligned
    
    # Reset graph for test case 2
    G.clear()
    
    # Test case 2: Poorly-aligned nodes (large phase difference)
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=math.pi * 0.8, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before_misaligned = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after_misaligned = G.nodes[0]["dnfr"]
    reduction_misaligned = dnfr_before_misaligned - dnfr_after_misaligned
    
    # Well-aligned nodes should have stronger ΔNFR reduction
    assert reduction_aligned > reduction_misaligned, \
        "Better-aligned nodes should have stronger ΔNFR reduction"


def test_um_dnfr_reduction_with_multiple_neighbors(graph_canon):
    """Test ΔNFR reduction with multiple neighbors."""
    G = graph_canon()
    
    # Create central node with multiple well-aligned neighbors
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(2, theta=0.15, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(3, theta=-0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # With multiple aligned neighbors, should have significant reduction
    assert dnfr_after < dnfr_before
    reduction_pct = (dnfr_before - dnfr_after) / dnfr_before
    assert reduction_pct > 0.05, "Multiple aligned neighbors should provide substantial reduction"


def test_um_dnfr_no_reduction_without_neighbors(graph_canon):
    """Test that isolated nodes don't experience ΔNFR reduction."""
    G = graph_canon()
    
    # Create isolated node
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # ΔNFR should remain unchanged for isolated nodes
    assert dnfr_after == pytest.approx(dnfr_before, abs=1e-9)


def test_um_dnfr_stabilization_can_be_disabled(graph_canon):
    """Test that ΔNFR stabilization can be disabled via configuration."""
    G = graph_canon()
    
    # Create coupled nodes
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    
    # Disable ΔNFR stabilization
    G.graph["UM_STABILIZE_DNFR"] = False
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # ΔNFR should remain unchanged when disabled
    assert dnfr_after == pytest.approx(dnfr_before, abs=1e-9)


def test_um_dnfr_custom_reduction_factor(graph_canon):
    """Test UM respects custom UM_dnfr_reduction factor."""
    G = graph_canon()
    
    # Create well-aligned nodes
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)  # Perfect alignment
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    # Test with default factor (0.15)
    dnfr_before_default = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after_default = G.nodes[0]["dnfr"]
    reduction_default = dnfr_before_default - dnfr_after_default
    
    # Reset
    G.nodes[0]["dnfr"] = 1.0
    
    # Test with custom higher factor (0.30)
    G.graph["GLYPH_FACTORS"] = {"UM_dnfr_reduction": 0.30}
    dnfr_before_custom = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after_custom = G.nodes[0]["dnfr"]
    reduction_custom = dnfr_before_custom - dnfr_after_custom
    
    # Higher factor should produce stronger reduction
    assert reduction_custom > reduction_default


def test_um_dnfr_reduction_with_negative_dnfr(graph_canon):
    """Test that ΔNFR reduction works correctly with negative ΔNFR."""
    G = graph_canon()
    
    # Create nodes with negative ΔNFR (contraction)
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=-0.5, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=-0.5, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # Negative ΔNFR should move toward zero (reduction in magnitude)
    assert abs(dnfr_after) < abs(dnfr_before)
    # Both should still be negative
    assert dnfr_after < 0


def test_um_dnfr_reduction_with_opposite_phases(graph_canon):
    """Test ΔNFR reduction with opposite phase nodes (worst alignment)."""
    G = graph_canon()
    
    # Create nodes with opposite phases
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=math.pi, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # With opposite phases (alignment ≈ 0), reduction should be minimal
    reduction = dnfr_before - dnfr_after
    assert reduction >= 0, "Should have zero or minimal reduction"
    assert reduction < 0.05, "Opposite phases should provide minimal stabilization"


def test_um_dnfr_reduction_preserves_sign(graph_canon):
    """Test that ΔNFR reduction doesn't change the sign of ΔNFR."""
    G = graph_canon()
    
    # Test positive ΔNFR
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=0.2, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=0.2, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    apply_glyph(G, 0, "UM")
    assert G.nodes[0]["dnfr"] >= 0, "Positive ΔNFR should remain positive"
    
    # Reset and test negative ΔNFR
    G.clear()
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=-0.2, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=-0.2, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    apply_glyph(G, 0, "UM")
    assert G.nodes[0]["dnfr"] <= 0, "Negative ΔNFR should remain negative"


def test_um_dnfr_stabilization_enabled_by_default(graph_canon):
    """Test that ΔNFR stabilization is enabled by default."""
    G = graph_canon()
    
    # Create coupled nodes without explicitly setting UM_STABILIZE_DNFR
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # ΔNFR should be reduced (stabilization enabled by default)
    assert dnfr_after < dnfr_before


def test_um_dnfr_reduction_with_mixed_alignments(graph_canon):
    """Test ΔNFR reduction when node has both aligned and misaligned neighbors."""
    G = graph_canon()
    
    # Central node with mixed neighbors
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)  # Well-aligned
    G.add_node(2, theta=math.pi * 0.9, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)  # Misaligned
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    apply_glyph(G, 0, "UM")
    dnfr_after = G.nodes[0]["dnfr"]
    
    # Should have moderate reduction (average of alignments)
    reduction_pct = (dnfr_before - dnfr_after) / dnfr_before
    assert 0.02 < reduction_pct < 0.15, "Mixed alignments should give moderate reduction"


def test_um_dnfr_reduction_integrates_with_phase_sync(graph_canon):
    """Test that ΔNFR reduction works alongside phase synchronization."""
    G = graph_canon()
    
    # Create nodes with phase difference
    theta_0_before = 0.0
    theta_1_before = math.pi / 4
    G.add_node(0, theta=theta_0_before, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_node(1, theta=theta_1_before, EPI=1.0, Si=0.5, dnfr=1.0, vf=1.0)
    G.add_edge(0, 1)
    G.graph["UM_STABILIZE_DNFR"] = True
    G.graph["UM_BIDIRECTIONAL"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    
    apply_glyph(G, 0, "UM")
    
    theta_0_after = G.nodes[0]["theta"]
    theta_1_after = G.nodes[1]["theta"]
    dnfr_after = G.nodes[0]["dnfr"]
    
    # Both phase synchronization and ΔNFR reduction should occur
    assert theta_0_after != theta_0_before, "Phase should change"
    assert theta_1_after != theta_1_before, "Neighbor phase should change (bidirectional)"
    assert dnfr_after < dnfr_before, "ΔNFR should be reduced"


def test_um_dnfr_reduction_affects_sense_index_trajectory(graph_canon):
    """Test that ΔNFR reduction contributes to improved stability (Si trajectory)."""
    G = graph_canon()
    
    # Create network with moderate ΔNFR
    G.add_node(0, theta=0.0, EPI=1.0, Si=0.4, dnfr=0.8, vf=1.0)
    G.add_node(1, theta=0.1, EPI=1.0, Si=0.5, dnfr=0.3, vf=1.0)
    G.add_node(2, theta=0.05, EPI=1.0, Si=0.5, dnfr=0.3, vf=1.0)
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.graph["UM_STABILIZE_DNFR"] = True
    
    dnfr_before = G.nodes[0]["dnfr"]
    si_before = G.nodes[0]["Si"]
    
    # Apply coupling
    apply_glyph(G, 0, "UM")
    
    dnfr_after = G.nodes[0]["dnfr"]
    
    # ΔNFR should be reduced
    assert dnfr_after < dnfr_before
    
    # Lower ΔNFR generally indicates more stability
    # (Si computation uses ΔNFR, so reducing ΔNFR can improve Si in subsequent steps)
    reduction_pct = (dnfr_before - dnfr_after) / dnfr_before
    assert reduction_pct > 0.05, "Should have meaningful ΔNFR reduction"
