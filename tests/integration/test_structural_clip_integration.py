"""Integration test for structural_clip with VAL and NUL operators.

This test verifies that the structural boundary preservation works correctly
when VAL (Expansion) and NUL (Contraction) operators are applied in sequences
that would otherwise cause EPI overflow.
"""

import pytest
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators import Expansion, Contraction
from tnfr.dynamics import update_epi_via_nodal_equation


def test_val_expansion_respects_epi_max():
    """Test that VAL operator with high EPI values respects EPI_MAX boundary.
    
    This is the critical case from the issue: EPI close to 1.0 with VAL_scale=1.15
    should not push EPI above EPI_MAX=1.0 after integration.
    """
    # Create a node with EPI close to the critical threshold
    G, node = create_nfr("test_val", epi=0.95, vf=1.0)
    
    # Set initial dnfr to enable evolution
    G.nodes[node]["DNFR"] = 0.1
    
    # Set VAL_scale to the default 1.15
    G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.15}
    
    # Apply VAL (Expansion) operator
    val_op = Expansion()
    val_op(G, node)
    
    # Run integration step
    update_epi_via_nodal_equation(G, dt=1.0)
    
    # EPI should be constrained to EPI_MAX
    epi_after = G.nodes[node]["EPI"]
    assert epi_after <= 1.0, f"EPI {epi_after} exceeds EPI_MAX=1.0"
    assert epi_after >= -1.0, f"EPI {epi_after} below EPI_MIN=-1.0"


def test_nul_contraction_respects_epi_min():
    """Test that NUL operator with low EPI values respects EPI_MIN boundary."""
    # Create a node with EPI at negative edge
    G, node = create_nfr("test_nul", epi=-0.95, vf=1.0)
    
    # Set initial dnfr to enable evolution
    G.nodes[node]["DNFR"] = -0.1
    
    # Set NUL_scale to the default 0.85
    G.graph["GLYPH_FACTORS"] = {"NUL_scale": 0.85}
    
    # Apply NUL (Contraction) operator
    nul_op = Contraction()
    nul_op(G, node)
    
    # Run integration step
    update_epi_via_nodal_equation(G, dt=1.0)
    
    # EPI should be constrained to EPI_MIN
    epi_after = G.nodes[node]["EPI"]
    assert epi_after <= 1.0, f"EPI {epi_after} exceeds EPI_MAX=1.0"
    assert epi_after >= -1.0, f"EPI {epi_after} below EPI_MIN=-1.0"


def test_repeated_val_applications_stay_bounded():
    """Test that repeated VAL applications don't accumulate to overflow."""
    G, node = create_nfr("test_repeated_val", epi=0.8, vf=1.0)
    G.nodes[node]["DNFR"] = 0.05
    G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.15}
    
    # Apply VAL multiple times
    val_op = Expansion()
    for _ in range(5):
        val_op(G, node)
        update_epi_via_nodal_equation(G, dt=1.0)
        
        epi = G.nodes[node]["EPI"]
        assert -1.0 <= epi <= 1.0, f"EPI {epi} out of bounds after iteration"


def test_val_nul_sequence_maintains_bounds():
    """Test that alternating VAL and NUL operators maintain boundaries."""
    G, node = create_nfr("test_val_nul_seq", epi=0.9, vf=1.0)
    G.nodes[node]["DNFR"] = 0.08
    G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.15, "NUL_scale": 0.85}
    
    val_op = Expansion()
    nul_op = Contraction()
    
    # Apply sequence: VAL, NUL, VAL, NUL
    for op in [val_op, nul_op, val_op, nul_op]:
        op(G, node)
        update_epi_via_nodal_equation(G, dt=1.0)
        
        epi = G.nodes[node]["EPI"]
        assert -1.0 <= epi <= 1.0, f"EPI {epi} out of bounds in sequence"


def test_soft_mode_clip_configuration():
    """Test that soft mode clipping can be configured."""
    G, node = create_nfr("test_soft", epi=0.95, vf=1.0)
    G.nodes[node]["DNFR"] = 0.1
    
    # Enable soft clipping mode
    G.graph["CLIP_MODE"] = "soft"
    G.graph["CLIP_SOFT_K"] = 5.0
    G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.15}
    
    val_op = Expansion()
    val_op(G, node)
    update_epi_via_nodal_equation(G, dt=1.0)
    
    epi_after = G.nodes[node]["EPI"]
    # Soft mode should still constrain to bounds
    assert -1.0 <= epi_after <= 1.0
    # And should be close to the boundary with smooth transition
    assert epi_after > 0.8


def test_custom_epi_bounds_respected():
    """Test that custom EPI_MIN/EPI_MAX values are respected."""
    G, node = create_nfr("test_custom_bounds", epi=0.4, vf=1.0)
    G.nodes[node]["DNFR"] = 0.1
    
    # Set custom bounds (narrower range)
    G.graph["EPI_MIN"] = -0.5
    G.graph["EPI_MAX"] = 0.5
    G.graph["GLYPH_FACTORS"] = {"VAL_scale": 1.15}
    
    val_op = Expansion()
    val_op(G, node)
    update_epi_via_nodal_equation(G, dt=1.0)
    
    epi_after = G.nodes[node]["EPI"]
    # Should respect custom bounds
    assert -0.5 <= epi_after <= 0.5, f"EPI {epi_after} exceeds custom bounds"
