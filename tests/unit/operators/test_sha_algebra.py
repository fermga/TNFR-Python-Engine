"""Test suite for SHA algebraic properties validation.

This module validates the algebraic properties of SHA (Silence) operator
according to TNFR.pdf Section 3.2.4 "Notación funcional de operadores glíficos".

Theoretical Foundation
----------------------
From TNFR.pdf §3.2.4 (p. 227-230) and nodal equation ∂EPI/∂t = νf · ΔNFR(t):

1. **SHA as Structural Identity**:
   SHA(g(ω)) ≈ g(ω) for structure (EPI)
   SHA preserves structural results, acting as identity for EPI component.

2. **Idempotence**:
   SHA^n = SHA for all n ≥ 1
   Multiple applications have same effect as single application.

3. **Commutativity with NUL**:
   SHA ∘ NUL = NUL ∘ SHA
   Silence and contraction can be applied in either order.

Physical Emergence
------------------
These properties emerge from TNFR physics, not arbitrary design:

- **Identity**: SHA reduces νf → 0, freezing ∂EPI/∂t but preserving EPI
- **Idempotence**: νf minimum is saturated; more SHA cannot reduce further
- **Commutativity**: SHA and NUL reduce orthogonal dimensions (νf vs complexity)

Test Coverage
-------------
- Identity property with Emission and complex sequences
- Idempotence through consistent SHA behavior across contexts
- Commutativity with Contraction operator
- All tests respect TNFR grammar constraints (C1, C2, C3)
"""

from __future__ import annotations

import pytest

from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
from tnfr.structural import create_nfr, run_sequence
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import (
    Silence,
    Emission,
    Reception,
    Coherence,
    Resonance,
    Dissonance,
    Contraction,
    Transition,
)
from tnfr.operators.algebra import (
    validate_identity_property,
    validate_idempotence,
    validate_commutativity_nul,
)


class TestSHAIdentityProperty:
    """Test SHA as structural identity element in operator composition."""
    
    def test_sha_identity_with_emission(self):
        """Validate SHA acts as identity after Emission: SHA(AL(ω)) ≈ AL(ω).
        
        Tests that SHA preserves structural state (EPI) created by Emission.
        
        Grammar-valid sequences:
        - Path 1: AL → IL → OZ (without SHA)
        - Path 2: AL → IL → SHA (with SHA as terminator)
        
        Both should have equivalent EPI values, demonstrating SHA's identity property.
        """
        G, node = create_nfr("emit_test", epi=0.50, vf=1.00)
        
        # Validate identity property
        assert validate_identity_property(G, node, Emission()), (
            "SHA failed identity property with Emission operator"
        )
    
    def test_sha_identity_preserves_epi_not_vf(self):
        """SHA as identity applies to EPI (structure), not νf (frequency).
        
        Clarifies that SHA preserves structural form but modifies reorganization
        capacity. This is the correct interpretation of SHA as "identity" in
        structural algebra.
        
        SHA is identity for the structural component (EPI), not the dynamic
        component (νf). This distinction emerges from the nodal equation.
        """
        G, node = create_nfr("test_epi", epi=0.60, vf=1.20)
        
        # Sequence ending with SHA
        G_sha = G.copy()
        run_sequence(G_sha, node, [Emission(), Coherence(), Silence()])
        epi_sha = float(get_attr(G_sha.nodes[node], ALIAS_EPI, 0.0))
        vf_sha = float(get_attr(G_sha.nodes[node], ALIAS_VF, 0.0))
        
        # Sequence ending with Dissonance (different terminator)
        G_oz = G.copy()
        run_sequence(G_oz, node, [Emission(), Coherence(), Dissonance()])
        epi_oz = float(get_attr(G_oz.nodes[node], ALIAS_EPI, 0.0))
        vf_oz = float(get_attr(G_oz.nodes[node], ALIAS_VF, 0.0))
        
        # EPI should be similar (both preserve structure after coherence)
        assert abs(epi_sha - epi_oz) < 0.05, (
            f"EPI diverged: SHA={epi_sha}, OZ={epi_oz}"
        )
        
        # νf should be minimal with SHA (characteristic effect)
        assert vf_sha < 0.15, (
            f"SHA did not reduce νf sufficiently: {vf_sha}"
        )
        
        # νf with Dissonance should be higher (maintains activation)
        assert vf_oz > vf_sha, (
            f"Dissonance should maintain higher νf than Silence: {vf_oz} vs {vf_sha}"
        )
    
    def test_sha_identity_in_complex_sequence(self):
        """Validate SHA identity in complex operator sequences.
        
        Tests: AL → EN → IL → RA → SHA vs AL → EN → IL → RA → OZ
        
        SHA should preserve the complex structural state similarly to other
        terminators. This validates identity property in realistic scenarios.
        """
        G, node = create_nfr("complex", epi=0.55, vf=1.05)
        
        # Path with SHA terminator
        G_sha = G.copy()
        run_sequence(G_sha, node, [
            Emission(),
            Reception(),
            Coherence(),
            Resonance(),
            Silence()
        ])
        epi_sha = float(get_attr(G_sha.nodes[node], ALIAS_EPI, 0.0))
        
        # Path with Dissonance terminator  
        G_oz = G.copy()
        run_sequence(G_oz, node, [
            Emission(),
            Reception(),
            Coherence(),
            Resonance(),
            Dissonance()
        ])
        epi_oz = float(get_attr(G_oz.nodes[node], ALIAS_EPI, 0.0))
        
        # Both terminators should preserve structural result from the sequence
        tolerance = 0.05  # Relaxed for complex sequence
        assert abs(epi_sha - epi_oz) < tolerance, (
            f"SHA identity failed in complex sequence: {epi_sha} vs {epi_oz}"
        )


class TestSHAIdempotence:
    """Test SHA idempotence property: SHA^n = SHA."""
    
    def test_sha_idempotence_basic(self):
        """Validate SHA behavior is consistent across different contexts.
        
        SHA should always reduce νf to near-zero and preserve EPI, regardless
        of where it appears in a sequence. This is the essence of idempotence:
        the effect is saturated and consistent.
        
        Physical basis: Once νf ≈ 0, further reduction is impossible.
        """
        G, node = create_nfr("idempotent_basic", epi=0.65, vf=1.30)
        
        assert validate_idempotence(G, node), (
            "SHA failed idempotence: inconsistent behavior across contexts"
        )
    
    def test_sha_characteristic_effect_on_vf(self):
        """Validate SHA's characteristic effect: always reduces νf to near-zero.
        
        This is the operational definition of idempotence for SHA:
        its effect on νf is always the same (reduction to minimum).
        
        Tests SHA in multiple contexts; all should show minimal νf.
        """
        G, node = create_nfr("vf_test", epi=0.58, vf=1.15)
        
        # Test 1: SHA after Emission + Coherence
        G1 = G.copy()
        run_sequence(G1, node, [Emission(), Coherence(), Silence()])
        vf_1 = float(get_attr(G1.nodes[node], ALIAS_VF, 0.0))
        
        # Test 2: SHA after Emission + Coherence + Resonance
        G2 = G.copy()
        run_sequence(G2, node, [Emission(), Coherence(), Resonance(), Silence()])
        vf_2 = float(get_attr(G2.nodes[node], ALIAS_VF, 0.0))
        
        # Test 3: SHA after complex sequence
        G3 = G.copy()
        run_sequence(G3, node, [
            Emission(),
            Reception(),
            Coherence(),
            Silence()
        ])
        vf_3 = float(get_attr(G3.nodes[node], ALIAS_VF, 0.0))
        
        # All should have minimal νf (SHA's characteristic effect)
        threshold = 0.15
        assert vf_1 < threshold, f"SHA failed to reduce νf in context 1: {vf_1}"
        assert vf_2 < threshold, f"SHA failed to reduce νf in context 2: {vf_2}"
        assert vf_3 < threshold, f"SHA failed to reduce νf in context 3: {vf_3}"
        
        # All should be similar (idempotent behavior)
        max_diff = max(abs(vf_1 - vf_2), abs(vf_2 - vf_3), abs(vf_1 - vf_3))
        assert max_diff < 0.05, (
            f"SHA behavior inconsistent: νf values {vf_1}, {vf_2}, {vf_3}"
        )
    
    def test_sha_preserves_epi_consistently(self):
        """Validate SHA preserves EPI in all contexts (identity aspect).
        
        SHA should not alter the structural form, only the reorganization rate.
        This test validates EPI preservation is consistent across contexts.
        """
        G, node = create_nfr("epi_preserve", epi=0.70, vf=1.25)
        
        # Measure EPI at different points in sequence ending with Dissonance
        G_before = G.copy()
        run_sequence(G_before, node, [Emission(), Coherence(), Dissonance()])
        epi_before = float(get_attr(G_before.nodes[node], ALIAS_EPI, 0.0))
        
        # Measure EPI after SHA (terminator)
        G_after = G.copy()
        run_sequence(G_after, node, [Emission(), Coherence(), Silence()])
        epi_after = float(get_attr(G_after.nodes[node], ALIAS_EPI, 0.0))
        
        # SHA should preserve EPI similarly to other terminators (within tolerance)
        assert abs(epi_before - epi_after) < 0.05, (
            f"SHA altered EPI vs Dissonance: {epi_before} vs {epi_after}"
        )


class TestSHANULCommutativity:
    """Test commutativity property: SHA ∘ NUL = NUL ∘ SHA."""
    
    def test_sha_nul_commutativity_basic(self):
        """Validate SHA and NUL commute: SHA(NUL(ω)) ≈ NUL(SHA(ω)).
        
        Physical basis: SHA and NUL reduce orthogonal dimensions:
        - SHA reduces νf (reorganization capacity)
        - NUL reduces EPI complexity (structural dimensionality)
        
        Order of reduction doesn't affect final state in equilibrium.
        """
        G, node = create_nfr("commute_basic", epi=0.55, vf=1.10)
        
        assert validate_commutativity_nul(G, node), (
            "SHA and NUL failed commutativity test"
        )
    
    def test_sha_nul_commutativity_after_activation(self):
        """Validate SHA-NUL commutativity after node activation.
        
        After AL → IL, applying SHA→NUL or NUL→SHA should yield equivalent states.
        Tests commutativity in realistic scenario with prior structure.
        """
        G, node = create_nfr("commute_activated", epi=0.48, vf=1.05)
        
        # Establish initial activated state
        run_sequence(G, node, [Emission(), Coherence(), Silence()])
        
        # Test commutativity from this state
        # Note: Starting from silenced state, test with new sequences
        G_test = G.copy()
        
        assert validate_commutativity_nul(G_test, node), (
            "SHA-NUL commutativity failed after activation sequence"
        )
    
    def test_sha_nul_order_independence(self):
        """Explicitly validate both orderings produce same EPI and νf.
        
        Direct comparison of NAV→SHA→NUL vs NAV→NUL→SHA paths.
        Validates that order truly doesn't matter for final state.
        """
        G, node = create_nfr("order_test", epi=0.62, vf=1.18)
        
        # Path 1: NAV → SHA → NUL
        G1 = G.copy()
        run_sequence(G1, node, [Transition(), Silence(), Contraction()])
        epi_sha_nul = float(get_attr(G1.nodes[node], ALIAS_EPI, 0.0))
        vf_sha_nul = float(get_attr(G1.nodes[node], ALIAS_VF, 0.0))
        
        # Path 2: NAV → NUL → SHA  
        G2 = G.copy()
        run_sequence(G2, node, [Transition(), Contraction(), Silence()])
        epi_nul_sha = float(get_attr(G2.nodes[node], ALIAS_EPI, 0.0))
        vf_nul_sha = float(get_attr(G2.nodes[node], ALIAS_VF, 0.0))
        
        # Validate order independence
        tolerance = 0.02
        assert abs(epi_sha_nul - epi_nul_sha) < tolerance, (
            f"EPI differs by order: SHA→NUL={epi_sha_nul}, NUL→SHA={epi_nul_sha}"
        )
        assert abs(vf_sha_nul - vf_nul_sha) < tolerance, (
            f"νf differs by order: SHA→NUL={vf_sha_nul}, NUL→SHA={vf_nul_sha}"
        )


class TestAlgebraicPropertiesIntegration:
    """Integration tests combining multiple algebraic properties."""
    
    def test_combined_properties_realistic_scenario(self):
        """Validate multiple algebraic properties in realistic sequence.
        
        Tests identity, idempotence, and structural preservation together
        in a scenario mimicking real usage (learning → consolidation → storage).
        """
        G, node = create_nfr("realistic", epi=0.52, vf=1.08)
        
        # Learning phase: AL → EN → IL → OZ (acquire, stabilize, close with dissonance)
        G_learned = G.copy()
        run_sequence(G_learned, node, [Emission(), Reception(), Coherence(), Dissonance()])
        epi_learned = float(get_attr(G_learned.nodes[node], ALIAS_EPI, 0.0))
        
        # Storage phase: AL → EN → IL → SHA (same but close with silence for storage)
        G_stored = G.copy()
        run_sequence(G_stored, node, [
            Emission(),
            Reception(),
            Coherence(),
            Silence()
        ])
        epi_stored = float(get_attr(G_stored.nodes[node], ALIAS_EPI, 0.0))
        vf_stored = float(get_attr(G_stored.nodes[node], ALIAS_VF, 0.0))
        
        # Identity: SHA should preserve learned structure (similarly to OZ)
        assert abs(epi_learned - epi_stored) < 0.05, (
            f"SHA failed to preserve learned structure: {epi_learned} vs {epi_stored}"
        )
        
        # Idempotence: νf should be minimal (characteristic SHA effect)
        assert vf_stored < 0.15, (
            f"SHA failed to achieve characteristic low νf: {vf_stored}"
        )
    
    def test_algebraic_closure(self):
        """Validate algebraic closure: complex compositions remain valid.
        
        Tests: (SHA ∘ IL) ∘ (RA ∘ AL) produces valid structural state.
        All algebraic operations should yield physically coherent results.
        """
        G, node = create_nfr("closure_test", epi=0.47, vf=0.98)
        
        # Complex composition
        sequence = [
            Emission(),     # AL
            Resonance(),    # RA  
            Coherence(),    # IL
            Silence(),      # SHA
        ]
        
        # Should not raise and should produce valid state
        run_sequence(G, node, sequence)
        
        # Validate resulting state is coherent
        final_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        final_vf = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
        
        assert final_epi >= 0.0, f"EPI became negative (invalid): {final_epi}"
        assert final_vf >= 0.0, f"νf became negative (invalid): {final_vf}"
        assert final_vf < 0.15, f"SHA did not reduce νf sufficiently: {final_vf}"
        
        # Structure should be well-formed (positive EPI)
        assert final_epi > 0.3, (
            f"Structure degraded to near-zero: {final_epi}"
        )
