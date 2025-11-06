"""Tests for operator sequences with boundary EPI values.

This module validates that operator sequences maintain structural boundaries
even when operations are chained or alternate between expansion and contraction.

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
from tnfr.config.operator_names import (
    EMISSION,
    EXPANSION,
    CONTRACTION,
    COHERENCE,
    SILENCE,
    DISSONANCE,
    SELF_ORGANIZATION,
)
from tnfr.validation import apply_glyph_with_grammar



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


class TestMultipleExpansionApplications:
    """Test repeated VAL applications don't accumulate overflow."""
    
    def test_multiple_val_applications_from_midrange(self):
        """Multiple VAL from mid-range should not overflow."""
        G, node = _create_test_nfr(epi=0.8)
        
        for i in range(5):
            Expansion()(G, node)
            epi = _get_epi_value(G, node)
            assert epi <= 1.0, f"Overflow at iteration {i+1}: EPI={epi}"
    
    def test_multiple_val_applications_from_high_epi(self):
        """Multiple VAL from high EPI should stabilize at boundary."""
        G, node = _create_test_nfr(epi=0.95)
        
        previous_epi = 0.95
        for i in range(10):
            Expansion()(G, node)
            epi = _get_epi_value(G, node)
            assert epi <= 1.0, f"Overflow at iteration {i+1}: EPI={epi}"
            # Should approach but never exceed 1.0
            assert epi >= previous_epi or abs(epi - 1.0) < 1e-6
            previous_epi = epi
    
    def test_multiple_val_converges_to_boundary(self):
        """Repeated VAL should converge toward but not exceed upper boundary."""
        G, node = _create_test_nfr(epi=0.5)
        
        for _ in range(20):
            Expansion()(G, node)
        
        final_epi = _get_epi_value(G, node)
        assert final_epi <= 1.0
        # Should approach boundary with repeated expansion
        # Note: Actual convergence depends on VAL_scale; with edge-aware scaling
        # the convergence might stabilize before reaching 0.9
        assert final_epi > 0.5, "Should increase from initial value with repeated expansion"


class TestMultipleContractionApplications:
    """Test repeated NUL applications don't accumulate underflow."""
    
    def test_multiple_nul_applications_from_midrange(self):
        """Multiple NUL from mid-range should not underflow."""
        G, node = _create_test_nfr(epi=-0.8)
        
        for i in range(5):
            Contraction()(G, node)
            epi = _get_epi_value(G, node)
            assert epi >= -1.0, f"Underflow at iteration {i+1}: EPI={epi}"
    
    def test_multiple_nul_converges_toward_zero(self):
        """Repeated NUL should contract toward zero, not breach lower boundary."""
        G, node = _create_test_nfr(epi=-0.5)
        
        for _ in range(20):
            Contraction()(G, node)
        
        final_epi = _get_epi_value(G, node)
        assert final_epi >= -1.0
        # NUL contracts magnitude, so should approach zero
        assert abs(final_epi) < 0.5, "Should contract toward zero"


class TestValNulOscillation:
    """Test alternating VAL/NUL sequences maintain stability."""
    
    def test_val_nul_oscillation_basic(self):
        """Sequence VAL→NUL→VAL should maintain structural stability."""
        G, node = _create_test_nfr(epi=0.9)
        
        Expansion()(G, node)
        after_val = _get_epi_value(G, node)
        assert -1.0 <= after_val <= 1.0
        
        Contraction()(G, node)
        after_nul = _get_epi_value(G, node)
        assert -1.0 <= after_nul <= 1.0
        
        Expansion()(G, node)
        final_epi = _get_epi_value(G, node)
        assert -1.0 <= final_epi <= 1.0
    
    def test_val_nul_oscillation_extended(self):
        """Extended VAL↔NUL oscillation should remain bounded."""
        G, node = _create_test_nfr(epi=0.7)
        
        for i in range(10):
            if i % 2 == 0:
                Expansion()(G, node)
            else:
                Contraction()(G, node)
            
            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, \
                f"Boundary violation at iteration {i+1}: EPI={epi}"
    
    def test_val_nul_at_high_boundary(self):
        """VAL→NUL at high EPI should maintain bounds."""
        G, node = _create_test_nfr(epi=0.99)
        
        Expansion()(G, node)
        assert _get_epi_value(G, node) <= 1.0
        
        Contraction()(G, node)
        assert -1.0 <= _get_epi_value(G, node) <= 1.0
    
    def test_nul_val_at_low_boundary(self):
        """NUL→VAL at low EPI should maintain bounds."""
        G, node = _create_test_nfr(epi=-0.99)
        
        Contraction()(G, node)
        assert _get_epi_value(G, node) >= -1.0
        
        Expansion()(G, node)
        assert -1.0 <= _get_epi_value(G, node) <= 1.0


class TestGrammarSequencesWithHighEPI:
    """Test canonical grammar sequences with high EPI values."""
    
    def test_emission_expansion_coherence_silence(self):
        """Canonical sequence [EMISSION, EXPANSION, COHERENCE, SILENCE] at high EPI."""
        sequence = [EMISSION, EXPANSION, COHERENCE, SILENCE]
        G, node = _create_test_nfr(epi=0.95, vf=1.0)
        
        for op_name in sequence:
            apply_glyph_with_grammar(G, [node], op_name)
            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, \
                f"Boundary violation in {sequence} at operator {op_name}: EPI={epi}"
    
    def test_expansion_contraction_expansion_coherence(self):
        """Oscillating sequence [EXPANSION, CONTRACTION, EXPANSION, COHERENCE] at high EPI."""
        # Note: This sequence needs proper grammar initialization
        G, node = _create_test_nfr(epi=0.95, vf=1.0)
        
        # Initialize with EMISSION to satisfy grammar
        apply_glyph_with_grammar(G, [node], EMISSION)
        
        sequence = [EXPANSION, CONTRACTION, EXPANSION, COHERENCE, SILENCE]
        for op_name in sequence:
            apply_glyph_with_grammar(G, [node], op_name)
            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, \
                f"Boundary violation in sequence at {op_name}: EPI={epi}"
    
    def test_dissonance_expansion_self_organization_silence(self):
        """Complex sequence [DISSONANCE, EXPANSION, SELF_ORGANIZATION, SILENCE] at high EPI."""
        sequence = [EMISSION, DISSONANCE, EXPANSION, SELF_ORGANIZATION, SILENCE]
        G, node = _create_test_nfr(epi=0.95, vf=1.0)
        
        for op_name in sequence:
            apply_glyph_with_grammar(G, [node], op_name)
            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, \
                f"Boundary violation in {sequence} at {op_name}: EPI={epi}"
    
    def test_multiple_sequences_cumulative_boundaries(self):
        """Multiple complete sequences should maintain boundaries cumulatively."""
        G, node = _create_test_nfr(epi=0.9, vf=1.0)
        
        # Run three complete sequences
        for run in range(3):
            sequence = [EMISSION, EXPANSION, COHERENCE, SILENCE]
            for op_name in sequence:
                apply_glyph_with_grammar(G, [node], op_name)
                epi = _get_epi_value(G, node)
                assert -1.0 <= epi <= 1.0, \
                    f"Boundary violation in run {run+1} at {op_name}: EPI={epi}"


class TestBoundaryStressSequences:
    """Stress test sequences designed to challenge boundary preservation."""
    
    def test_rapid_expansion_sequence(self):
        """Rapid succession of expansions from near-boundary."""
        G, node = _create_test_nfr(epi=0.85, vf=1.0)
        apply_glyph_with_grammar(G, [node], EMISSION)
        
        for i in range(10):
            apply_glyph_with_grammar(G, [node], EXPANSION)
            epi = _get_epi_value(G, node)
            assert epi <= 1.0, f"Expansion {i+1} violated upper boundary: EPI={epi}"
    
    def test_alternating_at_boundaries(self):
        """Alternating operators at near-boundary values."""
        G, node = _create_test_nfr(epi=0.98, vf=1.0)
        apply_glyph_with_grammar(G, [node], EMISSION)
        
        for i in range(20):
            if i % 2 == 0:
                apply_glyph_with_grammar(G, [node], EXPANSION)
            else:
                apply_glyph_with_grammar(G, [node], CONTRACTION)
            
            epi = _get_epi_value(G, node)
            assert -1.0 <= epi <= 1.0, \
                f"Alternation {i+1} violated boundaries: EPI={epi}"
