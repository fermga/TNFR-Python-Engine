"""Tests for ZHIR (Mutation) canonical contracts.

This module tests the canonical contracts that ZHIR MUST satisfy,
as specified in AGENTS.md and TNFR theory:

1. Preserves EPI sign (structural identity at sign level)
2. Does not collapse νf (maintains reorganization capacity)
3. Satisfies monotonicity requirements
4. Maintains structural bounds

Test Coverage:
- EPI sign preservation
- νf preservation (no collapse to zero)
- Structural bounds maintenance
- Contract validation in various scenarios

References:
- AGENTS.md §11 (Mutation contracts)
- TNFR.pdf §2.2.11 (ZHIR physics)
- Canonical Invariants (AGENTS.md §Invariants)
"""

import pytest
import math
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Mutation,
    Coherence,
    Dissonance,
    Emission,
    Silence,
    Transition,
)


class TestZHIREPISignPreservation:
    """Test ZHIR preserves EPI sign (identity at sign level)."""

    def test_zhir_preserves_positive_epi_sign(self):
        """ZHIR MUST NOT change positive EPI to negative."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        epi_before = G.nodes[node]["EPI"]
        assert epi_before > 0, "Test setup: EPI should be positive"

        # Apply mutation
        Mutation()(G, node)

        epi_after = G.nodes[node]["EPI"]

        # CRITICAL CONTRACT: Positive EPI must remain positive
        assert (
            epi_after > 0
        ), f"ZHIR violated sign preservation: positive EPI {epi_before} became {epi_after}"

    def test_zhir_preserves_negative_epi_sign(self):
        """ZHIR MUST NOT change negative EPI to positive."""
        G, node = create_nfr("test", epi=-0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.7, -0.6, -0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        epi_before = G.nodes[node]["EPI"]
        assert epi_before < 0, "Test setup: EPI should be negative"

        # Apply mutation
        Mutation()(G, node)

        epi_after = G.nodes[node]["EPI"]

        # CRITICAL CONTRACT: Negative EPI must remain negative
        assert (
            epi_after < 0
        ), f"ZHIR violated sign preservation: negative EPI {epi_before} became {epi_after}"

    def test_zhir_preserves_sign_in_canonical_sequence(self):
        """EPI sign preserved through IL → OZ → ZHIR sequence."""
        G, node = create_nfr("test", epi=0.6, vf=1.0)
        G.nodes[node]["epi_history"] = [0.4, 0.5, 0.6]

        epi_initial = G.nodes[node]["EPI"]
        sign_initial = 1 if epi_initial > 0 else -1

        # Apply canonical sequence
        run_sequence(G, node, [Transition(), Coherence(), Dissonance(), Mutation(), Silence()])

        epi_final = G.nodes[node]["EPI"]
        sign_final = 1 if epi_final > 0 else -1

        # Sign must be preserved
        assert sign_initial == sign_final, f"Sign changed from {sign_initial} to {sign_final}"

    def test_zhir_handles_zero_epi(self):
        """ZHIR with EPI=0 is edge case (no sign to preserve)."""
        G, node = create_nfr("test", epi=0.0, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.05, 0.0, 0.0]
        G.nodes[node]["delta_nfr"] = 0.1

        # Should not raise error
        Mutation()(G, node)

        epi_after = G.nodes[node]["EPI"]
        # Result can be positive, negative, or zero (no sign to preserve at 0)
        # Contract is satisfied as long as it doesn't crash

    def test_zhir_preserves_sign_with_high_transformation(self):
        """Sign preserved even with strong transformation (high ΔNFR)."""
        G, node = create_nfr("test", epi=0.7, vf=1.5)
        G.nodes[node]["epi_history"] = [0.4, 0.55, 0.7]
        G.nodes[node]["delta_nfr"] = 0.8  # High transformation pressure

        epi_before = G.nodes[node]["EPI"]
        assert epi_before > 0

        # Apply with strong destabilizer first
        run_sequence(G, node, [Transition(), Dissonance(), Mutation(), Coherence(), Silence()])

        epi_after = G.nodes[node]["EPI"]

        # Even with strong transformation, sign must be preserved
        assert epi_after > 0, "Strong transformation violated sign preservation"


class TestZHIRVfPreservation:
    """Test ZHIR does not collapse structural frequency (νf)."""

    def test_zhir_does_not_collapse_vf(self):
        """ZHIR MUST NOT reduce νf to zero (would kill the node)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        vf_before = G.nodes[node]["νf"]
        assert vf_before > 0, "Test setup: νf should be positive"

        # Apply mutation
        Mutation()(G, node)

        vf_after = G.nodes[node]["νf"]

        # CRITICAL CONTRACT: νf must remain positive
        assert vf_after > 0, f"ZHIR collapsed νf: {vf_before} → {vf_after} (node death)"

    def test_zhir_does_not_drastically_reduce_vf(self):
        """ZHIR should not drastically reduce νf (>50% reduction suspicious)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        vf_before = G.nodes[node]["νf"]

        # Apply mutation
        Mutation()(G, node)

        vf_after = G.nodes[node]["νf"]

        # νf should not drop by more than 50% in single mutation
        # (This is a soft contract - strong drops indicate potential issue)
        assert (
            vf_after > 0.5 * vf_before
        ), f"ZHIR drastically reduced νf: {vf_before} → {vf_after} (>50% drop)"

    def test_zhir_preserves_vf_in_multiple_applications(self):
        """Multiple ZHIR applications should not progressively collapse νf."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        vf_initial = G.nodes[node]["νf"]

        # Apply 5 mutation cycles
        for i in range(5):
            run_sequence(G, node, [Transition(), Coherence(), Dissonance(), Mutation(), Silence()])

        vf_final = G.nodes[node]["νf"]

        # νf should not have collapsed to near-zero
        assert vf_final > 0.1 * vf_initial, f"Multiple ZHIR collapsed νf: {vf_initial} → {vf_final}"

    def test_zhir_with_low_initial_vf(self):
        """ZHIR with already low νf should not collapse it further."""
        G, node = create_nfr("test", epi=0.5, vf=0.2)  # Already low
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        vf_before = 0.2

        # Apply mutation
        Mutation()(G, node)

        vf_after = G.nodes[node]["νf"]

        # Should remain positive
        assert vf_after > 0, "ZHIR collapsed already-low νf"

        # Should not drop drastically
        assert vf_after > 0.5 * vf_before, f"ZHIR made low νf worse: {vf_before} → {vf_after}"

    def test_zhir_vf_metrics_tracked(self):
        """νf changes should be tracked in metrics."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.graph["COLLECT_OPERATOR_METRICS"] = True

        vf_before = G.nodes[node]["νf"]

        Mutation()(G, node)

        metrics = G.graph["operator_metrics"][-1]

        # Metrics should track νf
        assert "vf_final" in metrics
        assert "delta_vf" in metrics

        # delta_vf should reflect actual change
        vf_after = G.nodes[node]["νf"]
        expected_delta = vf_after - vf_before
        assert abs(metrics["delta_vf"] - expected_delta) < 0.01


class TestZHIRStructuralBounds:
    """Test ZHIR maintains structural bounds (EPI ∈ [-1, 1])."""

    def test_zhir_respects_epi_upper_bound(self):
        """ZHIR should not push EPI > 1.0."""
        G, node = create_nfr("test", epi=0.95, vf=1.0)
        G.nodes[node]["epi_history"] = [0.85, 0.90, 0.95]
        G.nodes[node]["delta_nfr"] = 0.5  # Strong expansion pressure

        # Apply with destabilizer
        run_sequence(G, node, [Transition(), Dissonance(), Mutation(), Coherence(), Silence()])

        epi_after = G.nodes[node]["EPI"]

        # Should respect upper bound
        assert epi_after <= 1.0, f"ZHIR violated upper bound: EPI = {epi_after} > 1.0"

    def test_zhir_respects_epi_lower_bound(self):
        """ZHIR should not push EPI < -1.0."""
        G, node = create_nfr("test", epi=-0.95, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.85, -0.90, -0.95]
        G.nodes[node]["delta_nfr"] = -0.5  # Strong contraction pressure

        # Apply with destabilizer
        run_sequence(G, node, [Transition(), Dissonance(), Mutation(), Coherence(), Silence()])

        epi_after = G.nodes[node]["EPI"]

        # Should respect lower bound
        assert epi_after >= -1.0, f"ZHIR violated lower bound: EPI = {epi_after} < -1.0"

    def test_zhir_phase_wraps_correctly(self):
        """Phase (θ) should wrap in [0, 2π)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["theta"] = 1.9 * math.pi  # Near 2π
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.5  # Will push beyond 2π
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.5}

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase must be in valid range
        assert (
            0 <= theta_after < 2 * math.pi
        ), f"ZHIR failed to wrap phase: θ = {theta_after} not in [0, 2π)"


class TestZHIRContractIntegration:
    """Integration tests for contract satisfaction."""

    def test_all_contracts_satisfied_in_typical_use(self):
        """Typical ZHIR usage should satisfy all contracts."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=1.0)
        G.nodes[node]["structural_type"] = "test_pattern"
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        epi_before = G.nodes[node]["EPI"]
        vf_before = G.nodes[node]["νf"]
        sign_before = 1 if epi_before > 0 else -1
        identity_before = G.nodes[node]["structural_type"]

        # Apply canonical sequence
        run_sequence(
            G,
            node,
            [
                Transition(),
                Coherence(),
                Dissonance(),
                Mutation(),
                Coherence(),
                Silence(),
            ],
        )

        epi_after = G.nodes[node]["EPI"]
        vf_after = G.nodes[node]["νf"]
        sign_after = 1 if epi_after > 0 else -1
        identity_after = G.nodes[node]["structural_type"]
        theta_after = G.nodes[node]["theta"]

        # Check all contracts
        assert sign_after == sign_before, "Sign preservation violated"
        assert vf_after > 0, "νf collapse violated"
        assert identity_after == identity_before, "Identity preservation violated"
        assert -1.0 <= epi_after <= 1.0, "EPI bounds violated"
        assert 0 <= theta_after < 2 * math.pi, "Phase bounds violated"

    def test_contracts_under_extreme_conditions(self):
        """Contracts should hold even under extreme transformations."""
        G, node = create_nfr("test", epi=0.8, vf=2.0, theta=0.1)
        G.nodes[node]["EPI_kind"] = "extreme_pattern"
        G.nodes[node]["epi_history"] = [0.5, 0.65, 0.8]
        G.nodes[node]["delta_nfr"] = 0.9  # Very high pressure

        epi_before = G.nodes[node]["EPI"]
        vf_before = G.nodes[node]["νf"]
        sign_before = 1 if epi_before > 0 else -1

        # Apply with extreme conditions
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.9}
        run_sequence(G, node, [Transition(), Dissonance(), Mutation(), Coherence(), Silence()])

        epi_after = G.nodes[node]["EPI"]
        vf_after = G.nodes[node]["νf"]
        sign_after = 1 if epi_after > 0 else -1

        # Contracts must still hold
        assert sign_after == sign_before, "Extreme: Sign violated"
        assert vf_after > 0, "Extreme: νf collapsed"
        assert -1.0 <= epi_after <= 1.0, "Extreme: EPI bounds violated"

    def test_contracts_with_negative_dnfr(self):
        """Contracts should hold with negative ΔNFR (contraction)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.7, 0.6, 0.5]  # Decreasing
        G.nodes[node]["delta_nfr"] = -0.4  # Negative pressure

        epi_before = G.nodes[node]["EPI"]
        sign_before = 1

        Mutation()(G, node)

        epi_after = G.nodes[node]["EPI"]
        sign_after = 1 if epi_after > 0 else -1

        # Sign must be preserved even with negative ΔNFR
        assert sign_after == sign_before, "Negative ΔNFR violated sign"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
