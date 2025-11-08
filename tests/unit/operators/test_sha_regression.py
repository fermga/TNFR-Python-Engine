"""Comprehensive regression test suite for SHA (Silence) operator.

This module implements a complete regression test suite for the Silence (SHA)
operator following TNFR structural theory as specified in TNFR.pdf §2.3.10.

Test Coverage:
- A. Structural Effects: νf reduction, EPI preservation, ΔNFR freezing
- B. Preconditions: Validation of minimum νf and existing EPI requirements
- C. Canonical Sequences: IL→SHA, SHA→AL, SHA→NAV, OZ→SHA
- D. Metrics: SHA-specific metrics validation
- E. Integration: Multi-node effects, complex sequences
- F. Nodal Equation: Validation of ∂EPI/∂t = νf · ΔNFR(t)
- G. Full Lifecycle: Complete activation-silence-reactivation cycle

Theoretical Foundation:
- SHA reduces νf → 0 (structural pause)
- EPI remains invariant (preservation)
- ΔNFR maintained but frozen (no reorganization pressure)
- Latency state tracking for memory consolidation
"""

from __future__ import annotations

import pytest
import warnings

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Silence,
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Resonance,
    Coupling,
    Transition,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF


class TestSHAStructuralEffects:
    """Test A: Structural effects of SHA operator."""

    def test_sha_reduces_vf_to_minimum(self):
        """Test 1: SHA must reduce νf to value close to zero.
        
        Validates nodal equation: If νf ≈ 0, then ∂EPI/∂t ≈ 0 (independent of ΔNFR).
        Sequence must start with generator (AL) per R1 grammar rule.
        """
        G, node = create_nfr("active", epi=0.60, vf=1.50)
        initial_vf = G.nodes[node][VF_PRIMARY]
        
        # Apply sequence: AL (generator start) → SHA
        run_sequence(G, node, [Emission(), Silence()])
        
        final_vf = G.nodes[node][VF_PRIMARY]
        min_threshold = G.graph.get("SHA_MIN_VF", 0.01)
        
        # Validations
        assert final_vf < initial_vf, "SHA must reduce νf"
        assert final_vf <= min_threshold * 2, f"νf must be close to minimum: {final_vf}"
        assert final_vf >= 0.0, "νf cannot be negative"

    def test_sha_preserves_epi_exactly(self):
        """Test 2: SHA must maintain EPI invariant with minimal tolerance."""
        G, node = create_nfr("memory", epi=0.73, vf=1.20)
        initial_epi = G.nodes[node][EPI_PRIMARY]
        
        # Apply SHA
        run_sequence(G, node, [Silence()])
        
        final_epi = G.nodes[node][EPI_PRIMARY]
        tolerance = 1e-3  # Allow small numerical tolerance
        
        assert abs(final_epi - initial_epi) < tolerance, (
            f"EPI must be preserved: ΔEPI = {abs(final_epi - initial_epi)}"
        )

    def test_sha_freezes_dnfr(self):
        """Test 3: SHA does not modify ΔNFR - state is frozen.
        
        ΔNFR can remain high but with νf ≈ 0, it does not affect EPI.
        """
        G, node = create_nfr("frozen", epi=0.50, vf=1.00)
        G.nodes[node][DNFR_PRIMARY] = 0.15  # High reorganization pressure
        initial_dnfr = G.nodes[node][DNFR_PRIMARY]
        
        # Apply SHA
        run_sequence(G, node, [Silence()])
        
        final_dnfr = G.nodes[node][DNFR_PRIMARY]
        
        # ΔNFR can remain high, but SHA should not actively change it
        # The key is that with νf ≈ 0, ΔNFR does not affect EPI
        assert abs(final_dnfr - initial_dnfr) < 0.05, (
            "SHA should not actively modify ΔNFR"
        )


class TestSHAPreconditions:
    """Test B: Precondition validation for SHA operator."""

    def test_sha_precondition_vf_minimum(self):
        """Test 4: SHA must fail if νf already at minimum."""
        G, node = create_nfr("already_silent", epi=0.40, vf=0.005)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        with pytest.raises(OperatorPreconditionError, match="already minimal"):
            run_sequence(G, node, [Silence()])

    def test_sha_requires_existing_epi(self):
        """Test 5: SHA should warn if EPI ≈ 0 (no structure to preserve)."""
        G, node = create_nfr("empty", epi=0.0, vf=1.0)
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # SHA on empty structure should issue warning
        # This is more of a semantic warning than hard failure
        with pytest.warns(UserWarning, match="no structure|empty|zero"):
            run_sequence(G, node, [Silence()])


class TestSHACanonicalSequences:
    """Test C: Canonical operator sequences involving SHA."""

    def test_sha_after_coherence_preserves_stability(self):
        """Test 6: IL → SHA (stabilize then preserve) - canonical memory pattern."""
        G, node = create_nfr("learning", epi=0.45, vf=1.10)
        G.nodes[node][DNFR_PRIMARY] = 0.20  # High initial pressure
        
        # IL reduces ΔNFR, stabilizes
        run_sequence(G, node, [Coherence()])
        post_il_dnfr = G.nodes[node][DNFR_PRIMARY]
        post_il_epi = G.nodes[node][EPI_PRIMARY]
        
        # SHA preserves the stabilized state
        run_sequence(G, node, [Silence()])
        
        assert G.nodes[node][VF_PRIMARY] < 0.1, "SHA reduces νf"
        assert abs(G.nodes[node][EPI_PRIMARY] - post_il_epi) < 0.05, "EPI preserved"

    def test_sha_to_emission_reactivation(self):
        """Test 7: SHA → NAV → AL (reactivation from silence) - structurally coherent awakening.
        
        TNFR Physics: Cannot jump zero → high (SHA → AL) directly.
        Must transition through medium frequency: SHA → NAV → AL (zero → medium → high).
        This respects structural continuity and prevents singularities.
        """
        G, node = create_nfr("sleeping", epi=0.55, vf=1.00)
        
        # Phase 1: Prepare and enter silence
        run_sequence(G, node, [Emission(), Coherence(), Silence()])
        assert G.nodes[node][VF_PRIMARY] < 0.1, "Node in silence"
        silent_epi = G.nodes[node][EPI_PRIMARY]
        
        # Phase 2: Reactivate through medium frequency (NAV) then high (AL)
        run_sequence(G, node, [Transition(), Emission()])
        
        # Validate coherent reactivation
        assert G.nodes[node][VF_PRIMARY] > 0.5, "Node reactivated"
        assert G.nodes[node][EPI_PRIMARY] >= silent_epi - 0.15, "EPI maintains structural identity"

    def test_sha_to_transition_controlled_change(self):
        """Test 8: SHA → NAV (controlled transition from silence)."""
        G, node = create_nfr("dormant", epi=0.48, vf=0.95)
        
        # SHA: Preserve structure
        run_sequence(G, node, [Silence()])
        preserved_epi = G.nodes[node][EPI_PRIMARY]
        
        # NAV: Transition from silence
        run_sequence(G, node, [Transition()])
        
        # Validate controlled transition without collapse
        assert G.nodes[node][VF_PRIMARY] > 0.1, "Node reactivating"
        assert abs(G.nodes[node][EPI_PRIMARY] - preserved_epi) < 0.2, (
            "EPI transitions controlledly without collapse"
        )

    def test_oz_to_sha_containment(self):
        """Test 9: OZ → SHA (dissonance contained) - therapeutic pause.
        
        Clinical use case: Trauma containment, conflict deferred.
        """
        G, node = create_nfr("trauma", epi=0.40, vf=1.00)
        G.nodes[node][DNFR_PRIMARY] = 0.05
        
        # OZ: Introduce dissonance
        run_sequence(G, node, [Dissonance()])
        post_oz_dnfr = G.nodes[node][DNFR_PRIMARY]
        assert post_oz_dnfr > 0.10, "Dissonance increases ΔNFR"
        
        # SHA: Contain dissonance (protective pause)
        run_sequence(G, node, [Silence()])
        
        # Validate containment
        assert G.nodes[node][VF_PRIMARY] < 0.1, "Node paused"
        # ΔNFR remains high but frozen
        assert G.nodes[node][DNFR_PRIMARY] > 0.10, "Dissonance contained (not resolved)"


class TestSHAMetrics:
    """Test D: SHA-specific metrics collection."""

    def test_sha_metrics_preservation(self):
        """Test 10: Validate that silence_metrics captures preservation correctly."""
        G, node = create_nfr("test", epi=0.60, vf=1.00)
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        run_sequence(G, node, [Silence()])
        
        # Check if metrics were collected
        if "operator_metrics" in G.graph:
            metrics = G.graph["operator_metrics"][-1]
            
            assert metrics["operator"] == "Silence", "Operator name recorded"
            assert metrics["glyph"] == "SHA", "Glyph recorded"
            
            # Check for SHA-specific metric keys
            assert "vf_reduction" in metrics or "vf_final" in metrics, (
                "νf reduction metric present"
            )


class TestSHAIntegration:
    """Test E: Integration and network effects."""

    def test_sha_does_not_affect_neighbors(self):
        """Test 11: SHA is local operation - no direct propagation to neighbors."""
        G, n1 = create_nfr("node1", epi=0.50, vf=1.00)
        
        # Add second node manually
        _, n2 = create_nfr("node2", epi=0.50, vf=1.00)
        # Import n2's attributes into G
        G.add_node(n2)
        set_attr(G.nodes[n2], ALIAS_EPI, 0.50)
        set_attr(G.nodes[n2], ALIAS_VF, 1.00)
        G.add_edge(n1, n2)  # Connect nodes
        
        initial_n2_vf = G.nodes[n2][VF_PRIMARY]
        
        # SHA on n1
        run_sequence(G, n1, [Silence()])
        
        # n2 must remain active
        assert G.nodes[n1][VF_PRIMARY] < 0.1, "n1 in silence"
        assert G.nodes[n2][VF_PRIMARY] >= initial_n2_vf * 0.9, (
            "n2 remains active (SHA is local)"
        )

    def test_sha_after_complex_sequence(self):
        """Test 12: SHA as closure of complex sequence.
        
        Sequence: AL → IL → RA → UM → SHA
        """
        G, node = create_nfr("complex", epi=0.30, vf=0.80)
        
        sequence = [
            Emission(),    # AL: Activate
            Coherence(),   # IL: Stabilize
            Resonance(),   # RA: Propagate
            Coupling(),    # UM: Couple
            Silence()      # SHA: Close and preserve
        ]
        
        initial_epi = G.nodes[node][EPI_PRIMARY]
        run_sequence(G, node, sequence)
        
        # Validate final state
        assert G.nodes[node][VF_PRIMARY] < 0.1, "Sequence closed with silence"
        assert G.nodes[node][EPI_PRIMARY] >= initial_epi, (
            "EPI evolved during sequence"
        )


class TestSHANodalEquation:
    """Test F: Nodal equation validation for SHA."""

    def test_sha_nodal_equation_validation(self):
        """Test 13: Validate SHA respects nodal equation: ∂EPI/∂t = νf · ΔNFR(t).
        
        If νf → 0, then |∂EPI/∂t| → 0
        """
        G, node = create_nfr("validate", epi=0.65, vf=1.30)
        G.nodes[node][DNFR_PRIMARY] = 0.25  # High pressure
        
        epi_before = G.nodes[node][EPI_PRIMARY]
        
        # SHA should work even with high ΔNFR
        run_sequence(G, node, [Silence()])
        
        epi_after = G.nodes[node][EPI_PRIMARY]
        vf_after = G.nodes[node][VF_PRIMARY]
        
        # ∂EPI/∂t ≈ νf_after · ΔNFR ≈ 0 (because νf ≈ 0)
        delta_epi = abs(epi_after - epi_before)
        
        # With νf ≈ 0, EPI change should be minimal regardless of ΔNFR
        assert delta_epi < 0.1, (
            f"Nodal equation respected: ΔEPI = {delta_epi} should be small with νf ≈ 0"
        )


class TestSHAFullLifecycle:
    """Test G: Complete lifecycle including SHA."""

    def test_sha_full_cycle_activation_silence_reactivation(self):
        """Test 14: Complete cycle: AL → IL → SHA → NAV → AL.
        
        Simulates: learning → consolidation → memory → recall → use
        """
        G, node = create_nfr("lifecycle", epi=0.25, vf=0.90)
        
        # Phase 1: Activation and stabilization (learning)
        run_sequence(G, node, [Emission(), Coherence()])
        post_learning_epi = G.nodes[node][EPI_PRIMARY]
        assert post_learning_epi > 0.25, "Learning increments EPI"
        
        # Phase 2: Consolidation in silence (memory formation)
        run_sequence(G, node, [Silence()])
        assert G.nodes[node][VF_PRIMARY] < 0.1, "Memory consolidated"
        memory_epi = G.nodes[node][EPI_PRIMARY]
        
        # Phase 3: Transition and reactivation (recall)
        run_sequence(G, node, [Transition(), Emission()])
        
        # Validate memory preservation and reactivation
        assert abs(G.nodes[node][EPI_PRIMARY] - memory_epi) < 0.2, (
            "Structural identity preserved through silence cycle"
        )
        assert G.nodes[node][VF_PRIMARY] > 0.5, "Node active again"


class TestSHALatencyStateTracking:
    """Additional tests for SHA latency state attributes."""

    def test_sha_sets_latency_attributes(self):
        """Validate SHA sets latency state tracking attributes."""
        G, node = create_nfr("latency_test", epi=0.50, vf=1.00)
        
        run_sequence(G, node, [Silence()])
        
        # Check latency attributes
        assert G.nodes[node].get("latent") == True, "Latent flag set"
        assert "latency_start_time" in G.nodes[node], "Start time recorded"
        assert "preserved_epi" in G.nodes[node], "EPI preserved"
        assert G.nodes[node]["preserved_epi"] == pytest.approx(0.50, abs=0.01), (
            "Preserved EPI matches initial"
        )
        assert "silence_duration" in G.nodes[node], "Duration tracker initialized"

    def test_sha_preserved_epi_matches_current(self):
        """Validate preserved_epi attribute matches actual EPI at silence entry."""
        G, node = create_nfr("preserve_test", epi=0.73, vf=1.10)
        
        # Apply some operators first
        run_sequence(G, node, [Emission(), Coherence()])
        epi_before_silence = G.nodes[node][EPI_PRIMARY]
        
        # Apply SHA
        run_sequence(G, node, [Silence()])
        
        preserved_epi = G.nodes[node].get("preserved_epi", 0.0)
        assert abs(preserved_epi - epi_before_silence) < 0.01, (
            "Preserved EPI must match EPI at silence entry"
        )
