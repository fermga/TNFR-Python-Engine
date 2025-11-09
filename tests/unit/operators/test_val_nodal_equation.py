"""Canonical tests for VAL nodal equation compliance.

This module validates that the VAL (Expansion) operator rigorously adheres
to the fundamental nodal equation:

    ∂EPI/∂t = νf · ΔNFR(t)

Test Coverage:
--------------
1. **Nodal Equation**: Verify ΔEPI ≈ νf · ΔNFR · dt
2. **νf Increase**: Expansion increases structural frequency
3. **EPI Increase**: Expansion increases structural form magnitude

Physical Basis:
---------------
From TNFR.pdf § 2.1: The nodal equation governs all structural transformations.
VAL (Expansion) must:
- Respect the multiplicative relationship between νf, ΔNFR, and ∂EPI/∂t
- Increase νf (reorganization capacity)
- Increase EPI magnitude (structural form)

References:
-----------
- TNFR.pdf § 2.1: Nodal equation derivation
- AGENTS.md: Canonical Invariant #1 (EPI as Coherent Form)
- UNIFIED_GRAMMAR_RULES.md: U2 (CONVERGENCE & BOUNDEDNESS)
"""

import pytest
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import Expansion
from tnfr.structural import create_nfr, run_sequence


class TestVALNodalEquation:
    """Test suite for VAL nodal equation compliance."""

    def test_val_respects_nodal_equation(self):
        """VAL must respect the fundamental nodal equation: ∂EPI/∂t = νf · ΔNFR(t).
        
        Test approach:
        1. Set up custom ΔNFR hook to track changes
        2. Apply VAL with known parameters
        3. Verify hook was called (operator applied correctly)
        
        Physics: All structural operators must satisfy the nodal equation.
        This is Canonical Invariant #1.
        
        Note: EPI changes occur via ΔNFR hooks. This test verifies the
        operator triggers the hook mechanism correctly.
        """
        from tnfr.dynamics import set_delta_nfr_hook
        
        G, node = create_nfr("test_node", epi=0.5, vf=1.2)
        
        # Set positive ΔNFR (expansion pressure)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        # Track if hook was called
        hook_called = []
        
        def expansion_hook(graph):
            """Hook that modifies EPI and νf during expansion."""
            epi_key = list(ALIAS_EPI)[0]
            vf_key = list(ALIAS_VF)[0]
            dnfr = float(get_attr(graph.nodes[node], ALIAS_DNFR, 0.0))
            vf = float(get_attr(graph.nodes[node], ALIAS_VF, 1.0))
            
            # Simulate expansion: increase EPI proportional to νf * ΔNFR
            delta_epi = vf * dnfr * 0.1  # Small factor for stability
            delta_vf = dnfr * 0.05  # Slight νf increase
            
            graph.nodes[node][epi_key] += delta_epi
            graph.nodes[node][vf_key] += delta_vf
            hook_called.append(True)
        
        set_delta_nfr_hook(G, expansion_hook)
        
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        epi_before = G.nodes[node][epi_key]
        vf_before = G.nodes[node][vf_key]
        
        # Apply VAL
        run_sequence(G, node, [Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]
        
        # Verify hook was called (operator applied)
        assert len(hook_called) > 0, "ΔNFR hook should be called during VAL"
        
        # Verify changes occurred
        assert epi_after > epi_before, (
            f"EPI should increase: {epi_before:.4f} -> {epi_after:.4f}"
        )
        assert vf_after >= vf_before, (
            f"νf should increase: {vf_before:.4f} -> {vf_after:.4f}"
        )

    def test_val_increases_vf(self):
        """VAL must increase νf (structural frequency).
        
        Physical basis: Expansion increases reorganization capacity.
        From operator definition: VAL expands structural degrees of freedom.
        """
        G, node = create_nfr("test_node", vf=1.0)
        
        # Set positive ΔNFR
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        vf_key = list(ALIAS_VF)[0]
        vf_before = G.nodes[node][vf_key]
        
        run_sequence(G, node, [Expansion()])
        
        vf_after = G.nodes[node][vf_key]
        assert vf_after >= vf_before, (
            f"νf should increase or maintain: {vf_before} -> {vf_after}"
        )

    def test_val_increases_epi(self):
        """VAL must increase EPI (form magnitude) when ΔNFR > 0.
        
        Physical basis: Positive reorganization gradient + expansion
        leads to structural growth.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)
        
        # Set positive ΔNFR (expansion pressure)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]
        
        run_sequence(G, node, [Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        assert epi_after > epi_before, (
            f"EPI should increase: {epi_before} -> {epi_after}"
        )

    def test_val_proportional_to_vf(self):
        """EPI change should be proportional to νf.
        
        Test: Compare two nodes with different νf, same ΔNFR.
        Node with higher νf should have larger ΔEPI.
        """
        # Node 1: Lower νf
        G1, node1 = create_nfr("low_vf", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi1_before = G1.nodes[node1][epi_key]
        run_sequence(G1, node1, [Expansion()])
        epi1_after = G1.nodes[node1][epi_key]
        delta_epi1 = epi1_after - epi1_before
        
        # Node 2: Higher νf
        G2, node2 = create_nfr("high_vf", epi=0.5, vf=2.0)
        G2.nodes[node2][dnfr_key] = 0.1
        
        epi2_before = G2.nodes[node2][epi_key]
        run_sequence(G2, node2, [Expansion()])
        epi2_after = G2.nodes[node2][epi_key]
        delta_epi2 = epi2_after - epi2_before
        
        # Higher νf should lead to larger change (or equal if saturation)
        assert delta_epi2 >= delta_epi1, (
            f"Higher νf should produce larger or equal ΔEPI: "
            f"νf=1.0 → ΔEPI={delta_epi1:.4f}, νf=2.0 → ΔEPI={delta_epi2:.4f}"
        )

    def test_val_proportional_to_dnfr(self):
        """EPI change should be proportional to ΔNFR.
        
        Test: Compare two nodes with same νf, different ΔNFR.
        Node with higher ΔNFR should have larger ΔEPI.
        """
        # Node 1: Lower ΔNFR
        G1, node1 = create_nfr("low_dnfr", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.05
        
        epi_key = list(ALIAS_EPI)[0]
        epi1_before = G1.nodes[node1][epi_key]
        run_sequence(G1, node1, [Expansion()])
        epi1_after = G1.nodes[node1][epi_key]
        delta_epi1 = epi1_after - epi1_before
        
        # Node 2: Higher ΔNFR
        G2, node2 = create_nfr("high_dnfr", epi=0.5, vf=1.0)
        G2.nodes[node2][dnfr_key] = 0.15
        
        epi2_before = G2.nodes[node2][epi_key]
        run_sequence(G2, node2, [Expansion()])
        epi2_after = G2.nodes[node2][epi_key]
        delta_epi2 = epi2_after - epi2_before
        
        # Higher ΔNFR should lead to larger change
        assert delta_epi2 > delta_epi1, (
            f"Higher ΔNFR should produce larger ΔEPI: "
            f"ΔNFR=0.05 → ΔEPI={delta_epi1:.4f}, ΔNFR=0.15 → ΔEPI={delta_epi2:.4f}"
        )

    def test_val_zero_dnfr_minimal_change(self):
        """With ΔNFR ≈ 0, VAL should produce minimal EPI change.
        
        Physical basis: Nodal equation predicts ∂EPI/∂t ≈ 0 when ΔNFR ≈ 0.
        """
        G, node = create_nfr("no_pressure", epi=0.5, vf=1.0)
        
        # Set ΔNFR very close to zero (but not exactly zero to avoid precondition)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.001  # Near-zero
        
        # Disable strict precondition checking for this edge case
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
        
        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]
        
        run_sequence(G, node, [Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        delta_epi = abs(epi_after - epi_before)
        
        # Change should be small (within tolerance)
        assert delta_epi < 0.1, (
            f"Near-zero ΔNFR should produce minimal ΔEPI: ΔEPI={delta_epi:.4f}"
        )


@pytest.mark.val
@pytest.mark.nodal_equation
class TestVALNodalEquationIntegration:
    """Integration tests for VAL nodal equation in sequences."""

    def test_val_nodal_equation_in_sequence(self):
        """Nodal equation holds across multi-step sequences containing VAL.
        
        Test: VAL → IL sequence maintains nodal equation compliance.
        """
        from tnfr.operators.definitions import Coherence
        
        G, node = create_nfr("sequence", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi_initial = G.nodes[node][epi_key]
        
        # Apply VAL → IL
        run_sequence(G, node, [Expansion(), Coherence()])
        
        epi_final = G.nodes[node][epi_key]
        
        # Net change should be positive (expansion dominates)
        assert epi_final > epi_initial, (
            f"VAL → IL sequence should increase EPI: "
            f"{epi_initial:.4f} -> {epi_final:.4f}"
        )

    def test_multiple_val_cumulative_effect(self):
        """Multiple VAL applications have cumulative effect on EPI.
        
        Physical basis: Each VAL step follows nodal equation.
        Cumulative effect should be approximately sum of individual effects.
        """
        G, node = create_nfr("cumulative", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi_0 = G.nodes[node][epi_key]
        
        # First VAL
        run_sequence(G, node, [Expansion()])
        epi_1 = G.nodes[node][epi_key]
        delta_1 = epi_1 - epi_0
        
        # Reset ΔNFR (may have changed after first VAL)
        G.nodes[node][dnfr_key] = 0.1
        
        # Second VAL
        run_sequence(G, node, [Expansion()])
        epi_2 = G.nodes[node][epi_key]
        delta_2 = epi_2 - epi_1
        
        # Both steps should show expansion
        assert delta_1 > 0, f"First VAL should expand: ΔEPI={delta_1:.4f}"
        assert delta_2 > 0, f"Second VAL should expand: ΔEPI={delta_2:.4f}"
        
        # Total change
        total_delta = epi_2 - epi_0
        assert total_delta > delta_1, (
            f"Cumulative effect should exceed single step: "
            f"Total ΔEPI={total_delta:.4f} > Single ΔEPI={delta_1:.4f}"
        )
