"""Tests for edge-aware scaling in VAL and NUL operators.

This module tests the implementation of adaptive scaling that prevents
EPI overflow/underflow by adjusting scale factors near boundaries.

Issue: fermga/TNFR-Python-Engine#TBD
Depends on: fermga/TNFR-Python-Engine#2661 (structural_clip)
"""

import pytest
import networkx as nx

from tnfr.operators import (
    _compute_val_edge_aware_scale,
    _compute_nul_edge_aware_scale,
    GLYPH_OPERATIONS,
)
from tnfr.types import Glyph
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
from tnfr.node import NodeNX


class TestEdgeAwareScaleFunctions:
    """Test edge-aware scale computation functions."""

    def test_val_far_from_boundary(self):
        """VAL with EPI far from boundary should use full scale."""
        scale = _compute_val_edge_aware_scale(
            epi_current=0.5, scale=1.05, epi_max=1.0, epsilon=1e-12
        )
        assert scale == 1.05

    def test_val_near_upper_boundary(self):
        """VAL with EPI near upper boundary should adapt scale."""
        # EPI = 0.96, scale = 1.05 would give 1.008 > 1.0
        # Should adapt to roughly 1.0/0.96 ≈ 1.0417
        scale = _compute_val_edge_aware_scale(
            epi_current=0.96, scale=1.05, epi_max=1.0, epsilon=1e-12
        )
        assert scale < 1.05  # Adapted
        assert abs(scale - 1.0417) < 0.001
        # Verify result stays in bounds
        assert 0.96 * scale <= 1.0

    def test_val_at_boundary(self):
        """VAL with EPI exactly at boundary should use scale of 1.0."""
        scale = _compute_val_edge_aware_scale(
            epi_current=1.0, scale=1.05, epi_max=1.0, epsilon=1e-12
        )
        assert scale == 1.0  # Can't expand further
        assert 1.0 * scale <= 1.0

    def test_val_negative_epi_far_from_boundary(self):
        """VAL with negative EPI far from boundary should use full scale."""
        scale = _compute_val_edge_aware_scale(
            epi_current=-0.5, scale=1.05, epi_max=1.0, epsilon=1e-12
        )
        assert scale == 1.05

    def test_val_near_zero(self):
        """VAL with EPI near zero should use full scale safely."""
        scale = _compute_val_edge_aware_scale(
            epi_current=1e-13, scale=1.05, epi_max=1.0, epsilon=1e-12
        )
        assert scale == 1.05

    def test_nul_positive_epi(self):
        """NUL with positive EPI should use full scale (safe contraction)."""
        scale = _compute_nul_edge_aware_scale(
            epi_current=0.5, scale=0.85, epi_min=-1.0, epsilon=1e-12
        )
        assert scale == 0.85

    def test_nul_negative_epi_far_from_boundary(self):
        """NUL with negative EPI far from boundary should use full scale."""
        scale = _compute_nul_edge_aware_scale(
            epi_current=-0.5, scale=0.85, epi_min=-1.0, epsilon=1e-12
        )
        assert scale == 0.85

    def test_nul_negative_epi_near_lower_boundary(self):
        """NUL with EPI near lower boundary contracts safely toward center."""
        scale = _compute_nul_edge_aware_scale(
            epi_current=-0.95, scale=0.85, epi_min=-1.0, epsilon=1e-12
        )
        # No adaptation needed - contraction is safe
        assert scale == 0.85
        # Verify result stays within bounds and moves toward center
        result = -0.95 * scale
        assert result >= -1.0
        assert abs(result) < 0.95  # Moved toward zero

    def test_nul_at_lower_boundary(self):
        """NUL with EPI exactly at lower boundary contracts safely toward center."""
        scale = _compute_nul_edge_aware_scale(
            epi_current=-1.0, scale=0.85, epi_min=-1.0, epsilon=1e-12
        )
        # NUL with scale < 1.0 always contracts toward zero (safe)
        assert scale == 0.85
        # Verify result moves toward center, not boundary
        assert -1.0 * scale == -0.85  # Closer to zero

    def test_nul_near_zero_negative(self):
        """NUL with negative EPI near zero should use full scale safely."""
        scale = _compute_nul_edge_aware_scale(
            epi_current=-1e-13, scale=0.85, epi_min=-1.0, epsilon=1e-12
        )
        assert scale == 0.85


class TestVALOperatorEdgeAware:
    """Test VAL (Expansion) operator with edge-aware scaling."""

    def test_val_scales_vf_normally(self):
        """VAL should always scale νf regardless of edge-aware setting."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["VAL_scale"] = 1.05

        node = NodeNX.from_graph(G, "n1")
        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph.get("GLYPH_FACTORS", {}))

        # νf should be scaled
        assert abs(node.vf - 1.05) < 1e-9

    def test_val_scales_epi_with_edge_aware_enabled(self):
        """VAL should scale EPI when edge-aware is enabled."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")
        epi_before = node.EPI

        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph["GLYPH_FACTORS"])

        # EPI should be scaled
        expected_epi = epi_before * 1.05
        assert abs(node.EPI - expected_epi) < 1e-9

    def test_val_adapts_scale_near_boundary(self):
        """VAL should adapt scale when EPI is near upper boundary."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.96, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")

        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph["GLYPH_FACTORS"])

        # EPI should not exceed EPI_MAX
        assert float(node.EPI) <= 1.0
        # EPI should be scaled but not by full 1.05
        assert float(node.EPI) < 0.96 * 1.05

    def test_val_telemetry_records_adaptation(self):
        """VAL should record telemetry when scale is adapted."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.96, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")

        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph["GLYPH_FACTORS"])

        # Check telemetry was recorded
        assert "edge_aware_interventions" in G.graph
        interventions = G.graph["edge_aware_interventions"]
        assert len(interventions) > 0

        last_intervention = interventions[-1]
        assert last_intervention["adapted"] is True
        assert last_intervention["scale_requested"] == 1.05
        assert last_intervention["scale_effective"] < 1.05
        assert last_intervention["epi_before"] == 0.96

    def test_val_no_telemetry_when_not_adapted(self):
        """VAL should not record telemetry when scale is not adapted."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")

        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph["GLYPH_FACTORS"])

        # No adaptation needed, no telemetry
        interventions = G.graph.get("edge_aware_interventions", [])
        assert len(interventions) == 0

    def test_val_disabled_edge_aware(self):
        """VAL should not scale EPI when edge-aware is disabled."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.96, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = False
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")
        epi_before = node.EPI

        op = GLYPH_OPERATIONS[Glyph.VAL]
        op(node, G.graph["GLYPH_FACTORS"])

        # νf scaled
        assert abs(node.vf - 1.05) < 1e-9
        # EPI unchanged
        assert float(node.EPI) == epi_before


class TestNULOperatorEdgeAware:
    """Test NUL (Contraction) operator with edge-aware scaling."""

    def test_nul_scales_vf_normally(self):
        """NUL should always scale νf regardless of edge-aware setting."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")
        op = GLYPH_OPERATIONS[Glyph.NUL]
        op(node, G.graph["GLYPH_FACTORS"])

        # νf should be scaled
        assert abs(node.vf - 0.85) < 1e-9

    def test_nul_scales_epi_with_edge_aware_enabled(self):
        """NUL should scale EPI when edge-aware is enabled."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")
        epi_before = node.EPI

        op = GLYPH_OPERATIONS[Glyph.NUL]
        op(node, G.graph["GLYPH_FACTORS"])

        # EPI should be scaled
        expected_epi = epi_before * 0.85
        assert abs(node.EPI - expected_epi) < 1e-9

    def test_nul_adapts_scale_near_lower_boundary(self):
        """NUL should adapt scale when negative EPI is near lower boundary."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: -0.95, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")

        op = GLYPH_OPERATIONS[Glyph.NUL]
        op(node, G.graph["GLYPH_FACTORS"])

        # EPI should not go below EPI_MIN
        assert float(node.EPI) >= -1.0

    def test_nul_positive_epi_safe_contraction(self):
        """NUL with positive EPI should contract safely toward center."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.8, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")

        op = GLYPH_OPERATIONS[Glyph.NUL]
        op(node, G.graph["GLYPH_FACTORS"])

        # Positive EPI contracts toward zero (always safe)
        expected = 0.8 * 0.85
        assert abs(node.EPI - expected) < 1e-9

    def test_nul_disabled_edge_aware(self):
        """NUL should not scale EPI when edge-aware is disabled."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: -0.95, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = False
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")
        epi_before = node.EPI

        op = GLYPH_OPERATIONS[Glyph.NUL]
        op(node, G.graph["GLYPH_FACTORS"])

        # νf scaled
        assert abs(node.vf - 0.85) < 1e-9
        # EPI unchanged
        assert float(node.EPI) == epi_before


class TestCombinedSequences:
    """Test combined VAL/NUL sequences maintain EPI bounds."""

    def test_val_nul_val_maintains_bounds(self):
        """Sequence VAL→NUL→VAL should keep EPI in bounds."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.85, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05, "NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")

        val_op = GLYPH_OPERATIONS[Glyph.VAL]
        nul_op = GLYPH_OPERATIONS[Glyph.NUL]

        # Apply sequence
        val_op(node, G.graph["GLYPH_FACTORS"])
        assert -1.0 <= float(node.EPI) <= 1.0

        nul_op(node, G.graph["GLYPH_FACTORS"])
        assert -1.0 <= float(node.EPI) <= 1.0

        val_op(node, G.graph["GLYPH_FACTORS"])
        assert -1.0 <= float(node.EPI) <= 1.0

    def test_multiple_val_near_boundary(self):
        """Multiple VAL applications near boundary should stay in bounds."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.8, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.05}

        node = NodeNX.from_graph(G, "n1")
        val_op = GLYPH_OPERATIONS[Glyph.VAL]

        # Apply VAL multiple times
        for _ in range(5):
            val_op(node, G.graph["GLYPH_FACTORS"])
            assert float(node.EPI) <= 1.0

    def test_multiple_nul_near_boundary(self):
        """Multiple NUL applications near boundary should stay in bounds."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: -0.8, VF_PRIMARY: 1.0})
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MIN"] = -1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12
        G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}

        node = NodeNX.from_graph(G, "n1")
        nul_op = GLYPH_OPERATIONS[Glyph.NUL]

        # Apply NUL multiple times
        for _ in range(5):
            nul_op(node, G.graph["GLYPH_FACTORS"])
            assert float(node.EPI) >= -1.0
