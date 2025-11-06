"""Tests for boundary precision in structural operators.

This module validates that operators correctly handle EPI values near
structural boundaries (±1.0) without violating invariants due to
numerical precision issues.

Related Issues:
- fermga/TNFR-Python-Engine#2664: Comprehensive boundary precision test suite
- fermga/TNFR-Python-Engine#2661: structural_clip implementation
- fermga/TNFR-Python-Engine#2662: Edge-aware scaling for VAL/NUL
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.operators.definitions import Expansion, Contraction
from tnfr.config.operator_names import EXPANSION, CONTRACTION


def _get_epi_value(G: nx.Graph, node: str) -> float:
    """Extract the actual EPI scalar value from a node.
    
    After operator application, EPI may be stored as a dict with
    'continuous', 'discrete', and 'grid' keys. This helper extracts
    the actual scalar value.
    
    Parameters
    ----------
    G : nx.Graph
        Graph containing the node
    node : str
        Node identifier
        
    Returns
    -------
    float
        The scalar EPI value
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


class TestExpansionAtUpperBoundary:
    """Test VAL (Expansion) operator behavior near upper EPI boundary."""
    
    def test_expansion_at_upper_boundary_0_95(self):
        """VAL with EPI=0.95 should not exceed EPI_MAX."""
        G, node = _create_test_nfr(epi=0.95)
        Expansion()(G, node)
        assert _get_epi_value(G, node) <= 1.0
        
    def test_expansion_at_upper_boundary_0_99(self):
        """VAL with EPI=0.99 should not exceed EPI_MAX."""
        G, node = _create_test_nfr(epi=0.99)
        Expansion()(G, node)
        assert _get_epi_value(G, node) <= 1.0
        
    def test_expansion_at_upper_boundary_0_999(self):
        """VAL with EPI=0.999 should not exceed EPI_MAX."""
        G, node = _create_test_nfr(epi=0.999)
        Expansion()(G, node)
        assert _get_epi_value(G, node) <= 1.0
        
    def test_expansion_at_upper_boundary_near_one(self):
        """VAL with EPI very close to 1.0 should not exceed EPI_MAX."""
        G, node = _create_test_nfr(epi=1.0 - 1e-10)
        Expansion()(G, node)
        result = _get_epi_value(G, node)
        assert result <= 1.0
        # Allow for floating point tolerance
        assert result <= 1.0 or math.isclose(result, 1.0, rel_tol=1e-9, abs_tol=1e-12)
        
    def test_expansion_maintains_structural_coherence(self):
        """VAL near boundary should maintain all TNFR structural invariants."""
        test_cases = [0.95, 0.99, 0.999, 1.0 - 1e-10]
        for epi in test_cases:
            G, node = _create_test_nfr(epi=epi)
            Expansion()(G, node)
            result_epi = _get_epi_value(G, node)
            result_vf = G.nodes[node][VF_PRIMARY]
            
            # Verify EPI stays in bounds
            assert -1.0 <= result_epi <= 1.0, \
                f"EPI {result_epi} out of bounds for initial EPI={epi}"
            # Verify νf remains positive
            assert result_vf > 0, \
                f"νf {result_vf} should remain positive for EPI={epi}"


class TestContractionAtLowerBoundary:
    """Test NUL (Contraction) operator behavior near lower EPI boundary."""
    
    def test_contraction_at_lower_boundary_neg_0_95(self):
        """NUL with EPI=-0.95 should not fall below EPI_MIN."""
        G, node = _create_test_nfr(epi=-0.95)
        Contraction()(G, node)
        assert _get_epi_value(G, node) >= -1.0
        
    def test_contraction_at_lower_boundary_neg_0_99(self):
        """NUL with EPI=-0.99 should not fall below EPI_MIN."""
        G, node = _create_test_nfr(epi=-0.99)
        Contraction()(G, node)
        assert _get_epi_value(G, node) >= -1.0
        
    def test_contraction_at_lower_boundary_neg_0_999(self):
        """NUL with EPI=-0.999 should not fall below EPI_MIN."""
        G, node = _create_test_nfr(epi=-0.999)
        Contraction()(G, node)
        assert _get_epi_value(G, node) >= -1.0
        
    def test_contraction_at_lower_boundary_near_neg_one(self):
        """NUL with EPI very close to -1.0 should not fall below EPI_MIN."""
        G, node = _create_test_nfr(epi=-1.0 + 1e-10)
        Contraction()(G, node)
        result = _get_epi_value(G, node)
        assert result >= -1.0
        # Allow for floating point tolerance
        assert result >= -1.0 or math.isclose(result, -1.0, rel_tol=1e-9, abs_tol=1e-12)
        
    def test_contraction_maintains_structural_coherence(self):
        """NUL near boundary should maintain all TNFR structural invariants."""
        test_cases = [-0.95, -0.99, -0.999, -1.0 + 1e-10]
        for epi in test_cases:
            G, node = _create_test_nfr(epi=epi)
            Contraction()(G, node)
            result_epi = _get_epi_value(G, node)
            result_vf = G.nodes[node][VF_PRIMARY]
            
            # Verify EPI stays in bounds
            assert -1.0 <= result_epi <= 1.0, \
                f"EPI {result_epi} out of bounds for initial EPI={epi}"
            # Verify νf remains positive (NUL reduces but shouldn't zero it completely)
            assert result_vf >= 0, \
                f"νf {result_vf} should remain non-negative for EPI={epi}"


class TestBoundaryPrecisionWithIsClose:
    """Test that boundary comparisons use appropriate floating-point tolerances."""
    
    def test_expansion_boundary_with_tolerance(self):
        """Comparisons at boundary should use math.isclose for precision."""
        G, node = _create_test_nfr(epi=0.999)
        Expansion()(G, node)
        result = _get_epi_value(G, node)
        
        # Primary assertion: within bounds
        assert result <= 1.0, f"EPI {result} exceeds upper boundary"
        
        # Secondary: if slightly over due to precision, check with tolerance
        if result > 1.0:
            assert math.isclose(result, 1.0, rel_tol=1e-9, abs_tol=1e-12), \
                f"EPI {result} exceeds boundary beyond tolerance"
    
    def test_contraction_boundary_with_tolerance(self):
        """Comparisons at lower boundary should use math.isclose for precision."""
        G, node = _create_test_nfr(epi=-0.999)
        Contraction()(G, node)
        result = _get_epi_value(G, node)
        
        # Primary assertion: within bounds
        assert result >= -1.0, f"EPI {result} falls below lower boundary"
        
        # Secondary: if slightly under due to precision, check with tolerance
        if result < -1.0:
            assert math.isclose(result, -1.0, rel_tol=1e-9, abs_tol=1e-12), \
                f"EPI {result} falls below boundary beyond tolerance"
    
    def test_midrange_values_remain_stable(self):
        """Operators on mid-range EPI values should not cause precision issues."""
        test_cases = [0.0, 0.5, -0.5, 0.8, -0.8]
        for epi in test_cases:
            G, node = _create_test_nfr(epi=epi)
            Expansion()(G, node)
            result = _get_epi_value(G, node)
            assert -1.0 <= result <= 1.0, \
                f"Mid-range EPI {epi} led to out-of-bounds result {result}"


class TestBoundaryExtremes:
    """Test behavior at exact boundary values."""
    
    def test_expansion_at_exact_upper_boundary(self):
        """VAL at EPI=1.0 should handle the boundary gracefully."""
        G, node = _create_test_nfr(epi=1.0)
        Expansion()(G, node)
        result = _get_epi_value(G, node)
        assert result <= 1.0
        assert math.isclose(result, 1.0, rel_tol=1e-9, abs_tol=1e-12)
    
    def test_contraction_at_exact_lower_boundary(self):
        """NUL at EPI=-1.0 should handle the boundary gracefully."""
        G, node = _create_test_nfr(epi=-1.0)
        Contraction()(G, node)
        result = _get_epi_value(G, node)
        assert result >= -1.0
        # NUL contracts toward zero, so should move away from -1.0
        assert result > -1.0 or math.isclose(result, -1.0, rel_tol=1e-9)
    
    def test_zero_crossing_stable(self):
        """Operators near zero should maintain precision."""
        for epi in [-1e-10, 0.0, 1e-10]:
            G, node = _create_test_nfr(epi=epi)
            Expansion()(G, node)
            result = _get_epi_value(G, node)
            assert -1.0 <= result <= 1.0
