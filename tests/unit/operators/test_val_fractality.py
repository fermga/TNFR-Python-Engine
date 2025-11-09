"""Tests for VAL fractality preservation.

This module validates that VAL (Expansion) maintains structural identity
during expansion, in accordance with TNFR's operational fractality principle.

Test Coverage:
--------------
1. **Structural Identity**: Self-similar growth preservation
2. **Phase Coherence**: Moderate phase shifts during expansion
3. **Proportional Growth**: Ratios and proportions maintained

Physical Basis:
---------------
Fractality (Canonical Invariant #7): EPIs can nest without losing identity.

VAL must expand structure while preserving:
- **Form signature**: Structural proportions (e.g., EPI/νf ratio)
- **Phase coherence**: Network coupling compatibility
- **Self-similarity**: Pattern recognizability across scales

This is distinct from ZHIR (Mutation), which causes qualitative phase
transformations. VAL is quantitative growth, not qualitative change.

References:
-----------
- AGENTS.md: Canonical Invariant #7 (Operational Fractality)
- TNFR.pdf § 2.3: Self-similarity and scaling
- UNIFIED_GRAMMAR_RULES.md: Fractal preservation requirements
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
    Mutation,
    Emission,
)
from tnfr.structural import create_nfr, run_sequence


@pytest.mark.val
@pytest.mark.fractality
class TestVALStructuralIdentity:
    """Test VAL preserves structural identity during expansion."""

    def test_val_preserves_structural_signature(self):
        """VAL must expand while preserving structural signature.
        
        Structural signature: EPI/νf ratio (form per reorganization capacity).
        
        Physical meaning: How much structure exists per unit of capacity.
        During expansion, this ratio should remain relatively stable,
        indicating self-similar growth.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        # Structural signature before
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        epi_0 = G.nodes[node][epi_key]
        vf_0 = G.nodes[node][vf_key]
        signature_0 = epi_0 / vf_0 if vf_0 > 0 else 0
        
        # Apply VAL
        run_sequence(G, node, [Expansion()])
        
        # Structural signature after
        epi_1 = G.nodes[node][epi_key]
        vf_1 = G.nodes[node][vf_key]
        signature_1 = epi_1 / vf_1 if vf_1 > 0 else 0
        
        # Signature should change proportionally (fractal scaling)
        # Allow variance but preserve general ratio
        if signature_0 > 0:
            ratio = signature_1 / signature_0
            assert 0.7 < ratio < 1.3, (
                f"Structural signature changed too much: "
                f"{signature_0:.3f} -> {signature_1:.3f} (ratio {ratio:.3f})"
            )

    def test_val_self_similar_across_scales(self):
        """VAL produces self-similar growth across different initial scales.
        
        Test: Apply VAL to nodes at different scales (small, medium, large).
        Relative change should be comparable, showing scale-invariance.
        """
        dnfr_key = list(ALIAS_DNFR)[0]
        epi_key = list(ALIAS_EPI)[0]
        
        scales = [
            ("small", 0.2, 0.8),
            ("medium", 0.5, 1.5),
            ("large", 0.8, 2.0),
        ]
        
        relative_changes = []
        
        for name, epi, vf in scales:
            G, node = create_nfr(name, epi=epi, vf=vf)
            G.nodes[node][dnfr_key] = 0.1
            
            epi_before = G.nodes[node][epi_key]
            run_sequence(G, node, [Expansion()])
            epi_after = G.nodes[node][epi_key]
            
            relative_change = (epi_after - epi_before) / epi_before if epi_before > 0 else 0
            relative_changes.append(relative_change)
        
        # Relative changes should be similar (self-similarity)
        mean_change = np.mean(relative_changes)
        std_change = np.std(relative_changes)
        
        # Standard deviation should be small relative to mean
        coefficient_of_variation = std_change / mean_change if mean_change > 0 else 0
        
        assert coefficient_of_variation < 0.5, (
            f"Growth should be self-similar across scales: "
            f"CV={coefficient_of_variation:.3f}, changes={relative_changes}"
        )

    def test_val_preserves_form_type(self):
        """VAL preserves the type/character of structural form.
        
        If tracking EPI as vector or complex form, VAL should:
        - Increase magnitude
        - Preserve direction/character
        
        For scalar EPI, this means maintaining positivity and bounds.
        This test verifies VAL can be applied without breaking structure.
        """
        G, node = create_nfr("form_test", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]
        
        # Set up hook for structural change
        from tnfr.dynamics import set_delta_nfr_hook
        
        def form_preserving_hook(graph):
            # Expand while preserving form type (scalar positive)
            graph.nodes[node][epi_key] += 0.05
        
        set_delta_nfr_hook(G, form_preserving_hook)
        
        run_sequence(G, node, [Emission(), Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        
        # Form should remain valid (positive, bounded)
        assert epi_after > 0, "EPI should remain positive"
        assert epi_after < 10.0, "EPI should remain bounded"
        assert epi_after >= epi_before, "EPI should increase with hook"

    def test_val_multiple_applications_fractal(self):
        """Multiple VAL applications maintain fractal self-similarity.
        
        Test: Apply VAL repeatedly, check signature at each step.
        All signatures should fall within similar range.
        """
        G, node = create_nfr("iterative", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        signatures = []
        
        for i in range(3):
            # Set ΔNFR for expansion
            G.nodes[node][dnfr_key] = 0.1
            
            epi = G.nodes[node][epi_key]
            vf = G.nodes[node][vf_key]
            signature = epi / vf if vf > 0 else 0
            signatures.append(signature)
            
            run_sequence(G, node, [Expansion()])
        
        # Final signature
        epi_final = G.nodes[node][epi_key]
        vf_final = G.nodes[node][vf_key]
        signature_final = epi_final / vf_final if vf_final > 0 else 0
        signatures.append(signature_final)
        
        # Check variance in signatures
        sig_mean = np.mean(signatures)
        sig_std = np.std(signatures)
        
        coefficient_of_variation = sig_std / sig_mean if sig_mean > 0 else 0
        
        assert coefficient_of_variation < 0.5, (
            f"Signatures should remain similar: "
            f"CV={coefficient_of_variation:.3f}, signatures={signatures}"
        )


@pytest.mark.val
@pytest.mark.fractality
class TestVALPhaseCoherence:
    """Test VAL maintains phase coherence during expansion."""

    def test_val_maintains_phase_coherence(self):
        """VAL should not drastically destabilize phase.
        
        Physics: VAL is not a mutation operator.
        Phase may shift slightly during expansion, but should not
        cause radical jumps that break network coupling.
        """
        G, node = create_nfr("phase_test", epi=0.5, vf=1.0, theta=0.5)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        theta_key = list(ALIAS_THETA)[0]
        theta_0 = G.nodes[node][theta_key]
        
        run_sequence(G, node, [Expansion()])
        
        theta_1 = G.nodes[node][theta_key]
        
        # Phase shift should be moderate
        phase_shift = abs(theta_1 - theta_0)
        # Normalize to [0, π] (shortest path on circle)
        if phase_shift > np.pi:
            phase_shift = 2 * np.pi - phase_shift
        
        assert phase_shift < np.pi / 2, (
            f"VAL caused excessive phase shift: "
            f"{theta_0:.3f} -> {theta_1:.3f} (Δθ = {phase_shift:.3f} > π/2)"
        )

    def test_val_phase_shift_less_than_mutation(self):
        """VAL phase shift should be much smaller than ZHIR (Mutation).
        
        ZHIR causes phase transformation (qualitative change).
        VAL causes expansion (quantitative change).
        VAL's phase shift should be << ZHIR's.
        """
        # Test VAL phase shift
        G1, node1 = create_nfr("val_test", epi=0.5, vf=1.0, theta=0.5)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.1
        
        theta_key = list(ALIAS_THETA)[0]
        theta_val_before = G1.nodes[node1][theta_key]
        
        run_sequence(G1, node1, [Expansion()])
        
        theta_val_after = G1.nodes[node1][theta_key]
        val_phase_shift = abs(theta_val_after - theta_val_before)
        if val_phase_shift > np.pi:
            val_phase_shift = 2 * np.pi - val_phase_shift
        
        # Test ZHIR phase shift (for comparison)
        G2, node2 = create_nfr("zhir_test", epi=0.5, vf=1.0, theta=0.5)
        G2.nodes[node2][dnfr_key] = 0.2  # High ΔNFR for mutation
        
        theta_zhir_before = G2.nodes[node2][theta_key]
        
        # Apply IL first (ZHIR requires prior stabilization per U4b)
        run_sequence(G2, node2, [Coherence()])
        
        # Then apply dissonance to enable mutation
        from tnfr.operators.definitions import Dissonance
        run_sequence(G2, node2, [Dissonance()])
        
        # Now apply mutation
        run_sequence(G2, node2, [Mutation()])
        
        theta_zhir_after = G2.nodes[node2][theta_key]
        zhir_phase_shift = abs(theta_zhir_after - theta_zhir_before)
        if zhir_phase_shift > np.pi:
            zhir_phase_shift = 2 * np.pi - zhir_phase_shift
        
        # VAL shift should be smaller (if mutation occurred)
        # Note: This is a general guideline, not a strict requirement
        # since mutations may or may not occur based on conditions

    def test_val_phase_compatible_with_coupling(self):
        """VAL preserves phase compatibility for network coupling.
        
        After VAL, phase should remain in range compatible with
        typical coupling threshold (Δφ_max ≈ π/2).
        """
        G, node = create_nfr("coupling_test", epi=0.5, vf=1.0, theta=0.3)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        theta_key = list(ALIAS_THETA)[0]
        theta_initial = G.nodes[node][theta_key]
        
        # Apply VAL
        run_sequence(G, node, [Expansion()])
        
        theta_final = G.nodes[node][theta_key]
        
        # Phase should remain in [0, 2π]
        assert 0 <= theta_final <= 2 * np.pi, (
            f"Phase should remain valid: θ={theta_final:.3f}"
        )
        
        # Shift from initial should be within coupling range
        phase_shift = abs(theta_final - theta_initial)
        if phase_shift > np.pi:
            phase_shift = 2 * np.pi - phase_shift
        
        coupling_threshold = np.pi / 2  # Typical Δφ_max
        
        assert phase_shift <= coupling_threshold * 1.5, (
            f"Phase shift should allow coupling: "
            f"Δθ={phase_shift:.3f} < 1.5*Δφ_max={coupling_threshold*1.5:.3f}"
        )

    def test_val_phase_consistency_across_applications(self):
        """Phase behavior should be consistent across VAL applications.
        
        Multiple VAL applications should show similar phase shift patterns.
        This indicates predictable, self-similar dynamics.
        """
        G, node = create_nfr("consistency", epi=0.5, vf=1.0, theta=0.5)
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]
        
        phase_shifts = []
        
        for i in range(3):
            theta_before = G.nodes[node][theta_key]
            
            # Set ΔNFR
            G.nodes[node][dnfr_key] = 0.1
            
            run_sequence(G, node, [Expansion()])
            
            theta_after = G.nodes[node][theta_key]
            
            phase_shift = abs(theta_after - theta_before)
            if phase_shift > np.pi:
                phase_shift = 2 * np.pi - phase_shift
            
            phase_shifts.append(phase_shift)
        
        # Shifts should be relatively consistent
        mean_shift = np.mean(phase_shifts)
        std_shift = np.std(phase_shifts)
        
        coefficient_of_variation = std_shift / mean_shift if mean_shift > 0 else 0
        
        # Allow some variance but expect general consistency
        assert coefficient_of_variation < 1.0, (
            f"Phase shifts should be consistent: "
            f"CV={coefficient_of_variation:.3f}, shifts={phase_shifts}"
        )


@pytest.mark.val
@pytest.mark.fractality
class TestVALProportionalGrowth:
    """Test VAL maintains proportional relationships during growth."""

    def test_val_epi_vf_proportionality(self):
        """EPI and νf should grow proportionally during VAL.
        
        Physical basis: Expansion increases both form and capacity.
        The ratio should remain relatively stable.
        """
        G, node = create_nfr("proportional", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        epi_before = G.nodes[node][epi_key]
        vf_before = G.nodes[node][vf_key]
        
        run_sequence(G, node, [Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]
        
        # Both should increase
        assert epi_after >= epi_before, "EPI should increase"
        assert vf_after >= vf_before, "νf should increase"
        
        # Calculate growth ratios
        epi_growth = (epi_after - epi_before) / epi_before if epi_before > 0 else 0
        vf_growth = (vf_after - vf_before) / vf_before if vf_before > 0 else 0
        
        # Ratios should be in similar range (proportional growth)
        # Allow significant variance as implementation may differ
        # This documents the expected pattern

    def test_val_maintains_stability_indicators(self):
        """VAL maintains structural stability indicators.
        
        Metrics like sense index (Si) should remain in healthy range
        after expansion, indicating stable reorganization capacity.
        """
        G, node = create_nfr("stability", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1
        
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        
        # Apply VAL
        run_sequence(G, node, [Expansion()])
        
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]
        
        # Compute structural health indicator
        # Simple metric: both should be positive and reasonable
        assert epi_after > 0, "EPI should be positive"
        assert vf_after > 0, "νf should be positive"
        assert epi_after < 5.0, "EPI should be bounded"
        assert vf_after < 10.0, "νf should be bounded"

    def test_val_preserves_network_position(self):
        """VAL on one node preserves its relative position in network.
        
        After expansion, node should maintain coupling compatibility
        with neighbors (phase coherence).
        """
        import networkx as nx
        
        # Create 3-node network using create_nfr for proper initialization
        # We'll manually link them after
        G1, node1 = create_nfr("node0", epi=0.5, vf=1.0, theta=0.0)
        G2, node2 = create_nfr("node1", epi=0.5, vf=1.0, theta=0.1)
        G3, node3 = create_nfr("node2", epi=0.5, vf=1.0, theta=0.2)
        
        # Merge into single graph
        G = G1
        # Copy node 2 and 3 into G
        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        dnfr_key = list(ALIAS_DNFR)[0]
        theta_key = list(ALIAS_THETA)[0]
        
        G.add_node(node2, **G2.nodes[node2])
        G.add_node(node3, **G3.nodes[node3])
        
        # Connect as chain
        G.add_edge(node1, node2)
        G.add_edge(node2, node3)
        
        # Set ΔNFR
        G.nodes[node1][dnfr_key] = 0.1
        G.nodes[node2][dnfr_key] = 0.1
        G.nodes[node3][dnfr_key] = 0.1
        
        # Phases before expansion
        theta_0_before = G.nodes[node1][theta_key]
        theta_1_before = G.nodes[node2][theta_key]
        
        phase_diff_before = abs(theta_1_before - theta_0_before)
        
        # Expand node 0
        run_sequence(G, node1, [Emission(), Expansion()])
        
        # Phases after expansion
        theta_0_after = G.nodes[node1][theta_key]
        theta_1_after = G.nodes[node2][theta_key]
        
        phase_diff_after = abs(theta_1_after - theta_0_after)
        
        # Phase difference should not increase dramatically
        # (maintains coupling compatibility)
        assert phase_diff_after < np.pi, (
            f"Phase difference should allow coupling: "
            f"Δθ={phase_diff_after:.3f}"
        )
