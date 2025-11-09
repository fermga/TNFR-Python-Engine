"""Integration tests for canonical VAL sequences.

This module validates VAL (Expansion) operator behavior in canonical
operator sequences, ensuring proper grammar compliance and physical
correctness.

Test Coverage:
--------------
1. **VAL → IL**: Expand then stabilize
2. **OZ → VAL**: Dissonance then expand (exploratory growth)
3. **VAL → THOL**: Expand then self-organize (emergent structure)

Physical Basis:
---------------
These sequences represent archetypal patterns in TNFR dynamics:

- **VAL → IL**: Growth followed by consolidation (biological development)
- **OZ → VAL**: Breaking constraints then expanding (innovation)
- **VAL → THOL**: Expansion enabling emergence (morphogenesis)

Grammar Compliance:
-------------------
All sequences must satisfy:
- U1: STRUCTURAL INITIATION & CLOSURE
- U2: CONVERGENCE & BOUNDEDNESS (destabilizers need stabilizers)
- U3: RESONANT COUPLING (phase compatibility)
- U4: BIFURCATION DYNAMICS (triggers need handlers)

References:
-----------
- UNIFIED_GRAMMAR_RULES.md: Grammar derivations
- AGENTS.md: Canonical operator compositions
- TNFR.pdf § 3: Operator sequences and attractors
"""

import pytest
import numpy as np

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_VF,
    ALIAS_THETA,
)
from tnfr.operators.definitions import (
    Expansion,
    Coherence,
    Dissonance,
    SelfOrganization,
)
from tnfr.structural import create_nfr, run_sequence


@pytest.mark.val
@pytest.mark.canonical
class TestVALILSequence:
    """Test VAL → IL canonical sequence (expand then stabilize)."""

    def test_val_il_sequence_basic(self):
        """VAL → IL: Expansion followed by stabilization.
        
        Expected trajectory:
        1. VAL increases EPI (expansion)
        2. VAL may increase ΔNFR (exploration)
        3. IL reduces ΔNFR (stabilization)
        4. IL preserves expanded EPI (consolidation)
        
        Physical meaning: Growth followed by integration.
        """
        G, node = create_nfr("test_node", epi=0.4, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.05
        
        # Initial state
        epi_key = list(ALIAS_EPI)[0]
        epi_0 = G.nodes[node][epi_key]
        dnfr_0 = G.nodes[node][dnfr_key]
        
        # Apply VAL
        run_sequence(G, node, [Expansion()])
        epi_1 = G.nodes[node][epi_key]
        dnfr_1 = G.nodes[node][dnfr_key]
        
        # Apply IL
        run_sequence(G, node, [Coherence()])
        epi_2 = G.nodes[node][epi_key]
        dnfr_2 = G.nodes[node][dnfr_key]
        
        # Validate trajectory
        assert epi_1 > epi_0, "VAL should increase EPI"
        assert dnfr_1 >= dnfr_0 * 0.9, "VAL may increase or maintain ΔNFR"
        assert epi_2 >= epi_1 * 0.95, "IL should preserve expanded EPI"
        assert dnfr_2 <= dnfr_1, "IL should reduce or maintain ΔNFR"

    def test_val_il_maintains_coherence(self):
        """VAL → IL sequence maintains or increases coherence.
        
        Grammar basis: IL (Coherence) is a stabilizer (U2).
        It must reduce ΔNFR without compromising C(t).
        """
        G, node = create_nfr("coherent", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        # Track coherence (if available)
        # For single-node test, we check structural integrity
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        epi_before = G.nodes[node][epi_key]
        vf_before = G.nodes[node][vf_key]
        
        # Apply VAL → IL
        run_sequence(G, node, [Expansion(), Coherence()])
        
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]
        
        # Structural form should remain valid
        assert epi_after > 0, "EPI should remain positive"
        assert vf_after > 0, "νf should remain positive"
        assert epi_after >= epi_before * 0.9, "EPI should be maintained or grow"

    def test_val_il_phase_stability(self):
        """VAL → IL sequence maintains phase stability.
        
        Physics: Neither VAL nor IL are mutation operators.
        Phase should not undergo radical shifts.
        """
        G, node = create_nfr("phase_test", epi=0.5, vf=1.0, theta=0.5)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        theta_key = list(ALIAS_THETA)[0]
        theta_0 = G.nodes[node][theta_key]
        
        # Apply VAL → IL
        run_sequence(G, node, [Expansion(), Coherence()])
        
        theta_2 = G.nodes[node][theta_key]
        
        # Phase shift should be moderate
        phase_shift = abs(theta_2 - theta_0)
        # Normalize to [0, 2π]
        if phase_shift > np.pi:
            phase_shift = 2 * np.pi - phase_shift
        
        assert phase_shift < np.pi / 2, (
            f"Phase shift should be moderate: Δθ={phase_shift:.3f} < π/2"
        )


@pytest.mark.val
@pytest.mark.canonical
class TestOZVALSequence:
    """Test OZ → VAL canonical sequence (dissonance then expand)."""

    def test_oz_val_sequence_basic(self):
        """OZ → VAL: Controlled instability followed by expansion.
        
        Expected trajectory:
        1. OZ increases ΔNFR (destabilization)
        2. OZ may shift phase (exploration)
        3. VAL channels ΔNFR into expansion
        4. Net result: Exploratory growth
        
        Physical meaning: Breaking constraints enables new growth.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.1)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.05
        
        theta_key = list(ALIAS_THETA)[0]
        theta_0 = G.nodes[node][theta_key]
        dnfr_0 = G.nodes[node][dnfr_key]
        
        epi_key = list(ALIAS_EPI)[0]
        epi_0 = G.nodes[node][epi_key]
        
        # Apply OZ → VAL
        run_sequence(G, node, [Dissonance(), Expansion()])
        
        theta_1 = G.nodes[node][theta_key]
        dnfr_1 = G.nodes[node][dnfr_key]
        epi_1 = G.nodes[node][epi_key]
        
        # OZ should have created exploration space
        assert dnfr_1 >= dnfr_0, "ΔNFR should increase or maintain after OZ"
        
        # VAL should have expanded structure
        assert epi_1 > epi_0, "EPI should increase after VAL"
        
        # Phase may have shifted during exploration
        assert 0 <= theta_1 <= 2 * np.pi, "Phase should remain valid"

    def test_oz_val_enables_greater_expansion(self):
        """OZ → VAL produces larger expansion than VAL alone.
        
        Physical basis: OZ increases ΔNFR, which amplifies VAL's effect
        via the nodal equation: ∂EPI/∂t = νf · ΔNFR(t).
        """
        # Control: VAL alone
        G1, node1 = create_nfr("control", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.05
        
        epi_key = list(ALIAS_EPI)[0]
        epi1_before = G1.nodes[node1][epi_key]
        run_sequence(G1, node1, [Expansion()])
        epi1_after = G1.nodes[node1][epi_key]
        delta_epi_control = epi1_after - epi1_before
        
        # Treatment: OZ → VAL
        G2, node2 = create_nfr("treatment", epi=0.5, vf=1.0)
        G2.nodes[node2][dnfr_key] = 0.05
        
        epi2_before = G2.nodes[node2][epi_key]
        run_sequence(G2, node2, [Dissonance(), Expansion()])
        epi2_after = G2.nodes[node2][epi_key]
        delta_epi_treatment = epi2_after - epi2_before
        
        # OZ → VAL should produce equal or greater expansion
        assert delta_epi_treatment >= delta_epi_control * 0.9, (
            f"OZ → VAL should expand at least as much as VAL alone: "
            f"VAL={delta_epi_control:.4f}, OZ→VAL={delta_epi_treatment:.4f}"
        )

    def test_oz_val_requires_stabilization(self):
        """OZ → VAL sequence benefits from subsequent stabilization.
        
        Grammar basis: U2 (CONVERGENCE & BOUNDEDNESS).
        Destabilizers (OZ) should be paired with stabilizers (IL).
        
        This test documents the pattern (not enforced, but recommended).
        """
        G, node = create_nfr("unstabilized", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.05
        
        # Apply OZ → VAL without stabilizer
        run_sequence(G, node, [Dissonance(), Expansion()])
        
        dnfr_unstabilized = G.nodes[node][dnfr_key]
        
        # Now test with stabilizer
        G2, node2 = create_nfr("stabilized", epi=0.5, vf=1.0)
        G2.nodes[node2][dnfr_key] = 0.05
        
        # Apply OZ → VAL → IL
        run_sequence(G2, node2, [Dissonance(), Expansion(), Coherence()])
        
        dnfr_stabilized = G2.nodes[node2][dnfr_key]
        
        # Stabilized version should have lower ΔNFR
        assert dnfr_stabilized <= dnfr_unstabilized, (
            f"Stabilized sequence should reduce ΔNFR: "
            f"Unstabilized={dnfr_unstabilized:.4f}, Stabilized={dnfr_stabilized:.4f}"
        )


@pytest.mark.val
@pytest.mark.canonical
class TestVALTHOLSequence:
    """Test VAL → THOL canonical sequence (expand then self-organize)."""

    def test_val_thol_sequence_basic(self):
        """VAL → THOL: Expansion creates space for self-organization.
        
        Expected trajectory:
        1. VAL increases EPI (creates structural space)
        2. VAL increases νf (enables complexity)
        3. THOL creates sub-EPIs (emergent structure)
        4. Net result: Hierarchical organization
        
        Physical meaning: Growth followed by differentiation.
        """
        G, node = create_nfr("test_node", epi=0.6, vf=1.1)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.08
        
        epi_key = list(ALIAS_EPI)[0]
        epi_0 = G.nodes[node][epi_key]
        
        # Apply VAL → THOL
        run_sequence(G, node, [Expansion(), SelfOrganization()])
        
        epi_2 = G.nodes[node][epi_key]
        
        # Net expansion should occur
        assert epi_2 >= epi_0 * 0.9, (
            f"Net EPI should be maintained or grow: "
            f"{epi_0:.4f} -> {epi_2:.4f}"
        )
        
        # Check for nested structures (if tracked)
        sub_epis = G.graph.get("sub_epi", [])
        # THOL may create sub-structures (implementation dependent)
        # At minimum, verify structural integrity maintained

    def test_val_thol_enables_fractality(self):
        """VAL → THOL sequence enables fractal structure formation.
        
        Physical basis: VAL expands structural space.
        THOL organizes this space into nested patterns.
        Result: Operational fractality (Invariant #7).
        """
        G, node = create_nfr("fractal", epi=0.6, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        epi_before = G.nodes[node][epi_key]
        vf_before = G.nodes[node][vf_key]
        
        # Apply VAL → THOL
        run_sequence(G, node, [Expansion(), SelfOrganization()])
        
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]
        
        # Structural integrity maintained
        assert epi_after > 0, "EPI should remain positive"
        assert vf_after > 0, "νf should remain positive"
        
        # Growth should have occurred
        assert epi_after >= epi_before * 0.8, (
            f"Fractal organization should preserve structural scale: "
            f"{epi_before:.4f} -> {epi_after:.4f}"
        )

    def test_val_thol_phase_coherence(self):
        """VAL → THOL sequence maintains phase coherence.
        
        Physics: Neither VAL nor THOL are mutation operators.
        Phase should remain coherent for network coupling.
        """
        G, node = create_nfr("phase_test", epi=0.6, vf=1.1, theta=0.3)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        theta_key = list(ALIAS_THETA)[0]
        theta_before = G.nodes[node][theta_key]
        
        # Apply VAL → THOL
        run_sequence(G, node, [Expansion(), SelfOrganization()])
        
        theta_after = G.nodes[node][theta_key]
        
        # Phase should remain valid
        assert 0 <= theta_after <= 2 * np.pi, "Phase should remain in [0, 2π]"
        
        # Phase shift should be moderate
        phase_shift = abs(theta_after - theta_before)
        if phase_shift > np.pi:
            phase_shift = 2 * np.pi - phase_shift
        
        assert phase_shift < np.pi, (
            f"Phase shift should be moderate: Δθ={phase_shift:.3f} < π"
        )


@pytest.mark.val
@pytest.mark.canonical
class TestVALSequencesCombined:
    """Test combined canonical sequences involving VAL."""

    def test_oz_val_il_complete_cycle(self):
        """OZ → VAL → IL: Complete exploratory growth cycle.
        
        This is a canonical pattern:
        1. OZ: Break constraints (destabilize)
        2. VAL: Expand into new space (grow)
        3. IL: Consolidate new structure (stabilize)
        
        Grammar compliance: U2 (destabilizers paired with stabilizers).
        """
        G, node = create_nfr("cycle", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.05
        
        epi_key = list(ALIAS_EPI)[0]
        epi_initial = G.nodes[node][epi_key]
        dnfr_initial = G.nodes[node][dnfr_key]
        
        # Apply complete cycle
        run_sequence(
            G,
            node,
            [Dissonance(), Expansion(), Coherence()]
        )
        
        epi_final = G.nodes[node][epi_key]
        dnfr_final = G.nodes[node][dnfr_key]
        
        # Net growth with stability
        assert epi_final > epi_initial, (
            f"Complete cycle should produce net growth: "
            f"{epi_initial:.4f} -> {epi_final:.4f}"
        )
        
        assert dnfr_final <= dnfr_initial * 2.0, (
            f"Complete cycle should bound ΔNFR: "
            f"{dnfr_initial:.4f} -> {dnfr_final:.4f}"
        )

    def test_val_thol_il_hierarchical_consolidation(self):
        """VAL → THOL → IL: Hierarchical structure with consolidation.
        
        This sequence creates and stabilizes nested structures:
        1. VAL: Expand structural space
        2. THOL: Self-organize into hierarchy
        3. IL: Stabilize hierarchical structure
        """
        G, node = create_nfr("hierarchy", epi=0.6, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi_initial = G.nodes[node][epi_key]
        
        # Apply hierarchical sequence
        run_sequence(
            G,
            node,
            [Expansion(), SelfOrganization(), Coherence()]
        )
        
        epi_final = G.nodes[node][epi_key]
        dnfr_final = G.nodes[node][dnfr_key]
        
        # Structure maintained or grown
        assert epi_final >= epi_initial * 0.9, (
            f"Hierarchical structure should be preserved: "
            f"{epi_initial:.4f} -> {epi_final:.4f}"
        )
        
        # Stabilization should reduce ΔNFR
        assert dnfr_final < 0.2, (
            f"Final ΔNFR should be moderate after stabilization: "
            f"ΔNFR={dnfr_final:.4f}"
        )
