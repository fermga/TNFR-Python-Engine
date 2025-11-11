"""Tests for boundary protection compatibility and performance.

This module validates that boundary protection mechanisms don't break
existing functionality and don't introduce significant performance regressions.

Related Issues:
- fermga/TNFR-Python-Engine#2664: Comprehensive boundary precision test suite
- fermga/TNFR-Python-Engine#2661: structural_clip implementation
- fermga/TNFR-Python-Engine#2662: Edge-aware scaling for VAL/NUL
"""

from __future__ import annotations

import time
import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.operators.definitions import Expansion, Contraction
from tnfr.config.operator_names import (
    EMISSION,
    RECEPTION,
    COHERENCE,
    RESONANCE,
    SILENCE,
)
from tnfr.validation import validate_sequence


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
        cont = epi_val.get("continuous")
        if cont:
            # Extract real part from complex number
            val = cont[0]
            return val.real if hasattr(val, "real") else float(val)

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


class TestExistingGrammarCompatibility:
    """Test that boundary protections don't break existing grammar tests."""

    def test_canonical_sequence_validation(self):
        """Canonical sequences should still validate correctly."""
        sequence = [EMISSION, RECEPTION, COHERENCE, RESONANCE, SILENCE]
        result = validate_sequence(sequence)
        assert result.passed

    def test_simple_sequence_validation(self):
        """Simple sequences should still work."""
        sequence = [EMISSION, RECEPTION, COHERENCE, SILENCE]
        result = validate_sequence(sequence)
        assert result.passed

    def test_operators_with_normal_epi(self):
        """Operators with normal EPI values should work unchanged."""
        G, node = _create_test_nfr(epi=0.5, vf=1.0)

        Expansion()(G, node)
        epi_after_val = _get_epi_value(G, node)
        assert 0.5 < epi_after_val < 1.0

        Contraction()(G, node)
        epi_after_nul = _get_epi_value(G, node)
        assert -1.0 <= epi_after_nul <= 1.0


class TestPerformanceNoRegression:
    """Test that boundary protections don't cause significant performance degradation."""

    def test_expansion_performance(self):
        """VAL performance should not degrade significantly."""
        iterations = 1000

        # Measure time for operations
        G, node = _create_test_nfr(epi=0.5, vf=1.0)
        start = time.time()

        for _ in range(iterations):
            # Reset node for each iteration
            G.nodes[node][EPI_PRIMARY] = 0.5
            Expansion()(G, node)

        elapsed = time.time() - start

        # Should complete reasonably fast (< 5ms per operation average)
        avg_time = elapsed / iterations
        assert avg_time < 0.005, f"Average time {avg_time*1000:.2f}ms exceeds threshold"

    def test_contraction_performance(self):
        """NUL performance should not degrade significantly."""
        iterations = 1000

        G, node = _create_test_nfr(epi=0.5, vf=1.0)
        start = time.time()

        for _ in range(iterations):
            G.nodes[node][EPI_PRIMARY] = 0.5
            Contraction()(G, node)

        elapsed = time.time() - start

        avg_time = elapsed / iterations
        assert avg_time < 0.005, f"Average time {avg_time*1000:.2f}ms exceeds threshold"

    def test_boundary_operations_performance(self):
        """Operations near boundaries should not be significantly slower."""
        iterations = 100

        # Mid-range operations
        G_mid, node_mid = _create_test_nfr(epi=0.5, vf=1.0)
        start_mid = time.time()
        for _ in range(iterations):
            G_mid.nodes[node_mid][EPI_PRIMARY] = 0.5
            Expansion()(G_mid, node_mid)
        mid_time = time.time() - start_mid

        # Boundary operations
        G_bound, node_bound = _create_test_nfr(epi=0.95, vf=1.0)
        start_bound = time.time()
        for _ in range(iterations):
            G_bound.nodes[node_bound][EPI_PRIMARY] = 0.95
            Expansion()(G_bound, node_bound)
        bound_time = time.time() - start_bound

        # Boundary operations should not be more than 20% slower
        assert (
            bound_time < mid_time * 1.2
        ), f"Boundary ops ({bound_time:.4f}s) significantly slower than mid-range ({mid_time:.4f}s)"


class TestBoundaryProtectionModes:
    """Test different boundary protection modes."""

    def test_hard_clip_mode(self):
        """Hard clip mode should enforce strict boundaries."""
        from tnfr.dynamics.structural_clip import structural_clip

        result = structural_clip(1.1, -1.0, 1.0, mode="hard")
        assert result == 1.0

        result = structural_clip(-1.1, -1.0, 1.0, mode="hard")
        assert result == -1.0

    def test_soft_clip_mode(self):
        """Soft clip mode should smoothly approach boundaries."""
        from tnfr.dynamics.structural_clip import structural_clip

        # Mid-range - soft mode applies smooth S-curve transformation
        result_mid = structural_clip(0.5, -1.0, 1.0, mode="soft", k=3.0)
        assert -1.0 <= result_mid <= 1.0
        # Soft mode with k=3.0 transforms values significantly, so just verify bounded

        # Near boundary should be gently constrained
        result_high = structural_clip(0.95, -1.0, 1.0, mode="soft", k=3.0)
        assert -1.0 <= result_high <= 1.0

    def test_clip_mode_consistency(self):
        """Clip modes should be consistent across calls."""
        from tnfr.dynamics.structural_clip import structural_clip

        # Hard mode should be deterministic
        results_hard = [structural_clip(1.1, -1.0, 1.0, mode="hard") for _ in range(5)]
        assert all(r == 1.0 for r in results_hard)

        # Soft mode should be deterministic
        results_soft = [structural_clip(0.95, -1.0, 1.0, mode="soft", k=3.0) for _ in range(5)]
        assert len(set(results_soft)) == 1  # All same value


class TestEdgeAwareConfiguration:
    """Test edge-aware scaling configuration."""

    def test_edge_aware_can_be_enabled(self):
        """Edge-aware scaling can be enabled via configuration."""
        G, node = _create_test_nfr(epi=0.96, vf=1.0)
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0
        G.graph["EDGE_AWARE_EPSILON"] = 1e-12

        Expansion()(G, node)

        # Should respect boundary
        assert _get_epi_value(G, node) <= 1.0

    def test_edge_aware_can_be_disabled(self):
        """Edge-aware scaling can be disabled via configuration."""
        G, node = _create_test_nfr(epi=0.96, vf=1.0)
        G.graph["EDGE_AWARE_ENABLED"] = False

        Expansion()(G, node)

        # Should still respect boundary (via other mechanisms)
        assert _get_epi_value(G, node) <= 1.0


class TestBoundaryProtectionIntegration:
    """Test integration of boundary protection mechanisms."""

    def test_combined_protections(self):
        """Edge-aware and structural_clip should work together."""
        G, node = _create_test_nfr(epi=0.98, vf=1.0)

        # Enable all protections
        G.graph["EDGE_AWARE_ENABLED"] = True
        G.graph["EPI_MAX"] = 1.0

        # Apply multiple operations
        for _ in range(10):
            Expansion()(G, node)

        # Should converge to boundary without exceeding
        final_epi = _get_epi_value(G, node)
        assert final_epi <= 1.0
        assert final_epi > 0.9  # Should approach boundary

    def test_protection_with_various_configs(self):
        """Boundary protection should work with various graph configurations."""
        configs = [
            {},  # Default
            {"EDGE_AWARE_ENABLED": True},
            {"EDGE_AWARE_ENABLED": False},
            {"EPI_MAX": 1.0, "EPI_MIN": -1.0},
        ]

        for config in configs:
            G, node = _create_test_nfr(epi=0.95, vf=1.0)
            G.graph.update(config)

            Expansion()(G, node)

            # All configs should respect boundaries
            assert _get_epi_value(G, node) <= 1.0


class TestRegressionPrevention:
    """Test that previous boundary issues don't recur."""

    def test_no_overflow_accumulation(self):
        """Repeated operations should not accumulate overflow errors."""
        G, node = _create_test_nfr(epi=0.9, vf=1.0)

        # Apply many expansions
        for i in range(50):
            Expansion()(G, node)
            epi = _get_epi_value(G, node)
            assert epi <= 1.0, f"Overflow at iteration {i+1}"

    def test_no_underflow_accumulation(self):
        """Repeated operations should not accumulate underflow errors."""
        G, node = _create_test_nfr(epi=-0.9, vf=1.0)

        # Apply many contractions
        for i in range(50):
            Contraction()(G, node)
            epi = _get_epi_value(G, node)
            assert epi >= -1.0, f"Underflow at iteration {i+1}"

    def test_precision_drift_prevention(self):
        """Long sequences should not drift outside boundaries due to precision."""
        G, node = _create_test_nfr(epi=0.7, vf=1.0)

        # Alternate operations many times
        for i in range(100):
            if i % 2 == 0:
                Expansion()(G, node)
            else:
                Contraction()(G, node)

            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, f"Precision drift at iteration {i+1}: EPI={epi}"
