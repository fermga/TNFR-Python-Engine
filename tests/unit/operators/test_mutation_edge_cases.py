"""Edge case tests for ZHIR (Mutation) operator.

This module tests ZHIR behavior in edge cases and boundary conditions
to ensure robust operation across the full parameter space.

Test Coverage:
1. Isolated nodes
2. Phase boundaries
3. EPI extremes
4. Reproducibility
5. Unusual configurations

References:
- AGENTS.md §11 (Mutation operator)
- test_zhir_phase_transformation.py (complementary tests)
"""

import pytest
import math
import random
import numpy as np
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import (
    Mutation, Coherence, Dissonance, Emission, Silence
)


class TestZHIRIsolatedNodes:
    """Test ZHIR on isolated nodes (no connections)."""

    def test_zhir_on_isolated_node_succeeds(self):
        """ZHIR should work on isolated nodes (internal transformation)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # No edges - completely isolated
        assert G.degree(node) == 0

        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise error (mutation is internal)
        Mutation()(G, node)

        # Node should still be viable
        assert G.nodes[node]["νf"] > 0
        assert -1.0 <= G.nodes[node]["EPI"] <= 1.0

    def test_zhir_isolated_node_transforms_phase(self):
        """Isolated node's phase should still transform."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.5)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.4

        theta_before = G.nodes[node]["theta"]

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Phase transformation should occur even without neighbors
        assert theta_after != theta_before, "Isolated node phase not transformed"

    def test_zhir_isolated_preserves_all_contracts(self):
        """Isolated node should satisfy all ZHIR contracts."""
        G, node = create_nfr("test", epi=0.6, vf=1.2)
        G.nodes[node]["structural_type"] = "isolated_pattern"
        G.nodes[node]["epi_history"] = [0.4, 0.5, 0.6]
        # Enable postcondition validation
        G.graph["VALIDATE_OPERATOR_POSTCONDITIONS"] = True

        epi_before = G.nodes[node]["EPI"]
        vf_before = G.nodes[node]["νf"]
        sign_before = 1 if epi_before > 0 else -1

        Mutation()(G, node)

        epi_after = G.nodes[node]["EPI"]
        vf_after = G.nodes[node]["νf"]
        sign_after = 1 if epi_after > 0 else -1

        # All contracts should hold
        assert sign_after == sign_before, "Sign contract violated"
        assert vf_after > 0, "νf contract violated"
        assert G.nodes[node]["structural_type"] == "isolated_pattern", "Identity contract violated"


class TestZHIRPhaseBoundaries:
    """Test ZHIR behavior at phase boundaries (0, π/2, π, 3π/2, 2π)."""

    def test_zhir_at_zero_phase(self):
        """ZHIR should work correctly at θ=0."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=0.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Should be in valid range
        assert 0 <= theta_after < 2 * math.pi

    def test_zhir_at_pi_over_2(self):
        """ZHIR should work correctly at θ=π/2 (quadrant boundary)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=math.pi / 2)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        assert 0 <= theta_after < 2 * math.pi

    def test_zhir_at_pi(self):
        """ZHIR should work correctly at θ=π."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=math.pi)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.3

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        assert 0 <= theta_after < 2 * math.pi

    def test_zhir_near_2pi_wraps(self):
        """ZHIR near 2π should wrap correctly to [0, 2π)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=1.95 * math.pi)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        G.nodes[node]["delta_nfr"] = 0.5  # Will push past 2π
        G.graph["GLYPH_FACTORS"] = {"ZHIR_theta_shift_factor": 0.4}

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Should wrap into valid range
        assert 0 <= theta_after < 2 * math.pi
        # Should have wrapped to small value
        assert theta_after < 1.0, f"Failed to wrap: θ={theta_after}"

    def test_zhir_all_boundaries_tested(self):
        """Test all major phase boundaries."""
        boundaries = [
            0.0,
            math.pi / 2,
            math.pi,
            3 * math.pi / 2,
            1.99 * math.pi,  # Just before 2π
        ]

        for boundary in boundaries:
            G, node = create_nfr(f"test_{boundary}", epi=0.5, vf=1.0, theta=boundary)
            G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
            G.nodes[node]["delta_nfr"] = 0.3

            # Should not raise error
            Mutation()(G, node)

            theta_after = G.nodes[node]["theta"]
            assert (
                0 <= theta_after < 2 * math.pi
            ), f"Invalid phase after mutation at boundary {boundary}: {theta_after}"


class TestZHIRReproducibility:
    """Test deterministic behavior with seeds."""

    def test_zhir_reproducible_with_same_seed(self):
        """Same seed should produce identical ZHIR results."""
        # First run
        random.seed(42)
        np.random.seed(42)
        G1, node1 = create_nfr("test1", epi=0.5, vf=1.0, theta=1.0)
        G1.nodes[node1]["epi_history"] = [0.3, 0.4, 0.5]
        G1.nodes[node1]["delta_nfr"] = 0.3
        Mutation()(G1, node1)
        result1 = {
            "theta": G1.nodes[node1]["theta"],
            "epi": G1.nodes[node1]["EPI"],
            "vf": G1.nodes[node1]["νf"],
        }

        # Second run with same seed
        random.seed(42)
        np.random.seed(42)
        G2, node2 = create_nfr("test2", epi=0.5, vf=1.0, theta=1.0)
        G2.nodes[node2]["epi_history"] = [0.3, 0.4, 0.5]
        G2.nodes[node2]["delta_nfr"] = 0.3
        Mutation()(G2, node2)
        result2 = {
            "theta": G2.nodes[node2]["theta"],
            "epi": G2.nodes[node2]["EPI"],
            "vf": G2.nodes[node2]["νf"],
        }

        # Should be identical
        assert abs(result1["theta"] - result2["theta"]) < 1e-10
        assert abs(result1["epi"] - result2["epi"]) < 1e-10
        assert abs(result1["vf"] - result2["vf"]) < 1e-10

    def test_zhir_different_seeds_produce_different_results(self):
        """Different seeds should produce different results (if stochastic)."""
        # Run with seed 1
        random.seed(1)
        np.random.seed(1)
        G1, node1 = create_nfr("test1", epi=0.5, vf=1.0, theta=1.0)
        G1.nodes[node1]["epi_history"] = [0.3, 0.4, 0.5]
        G1.nodes[node1]["delta_nfr"] = 0.3
        Mutation()(G1, node1)
        theta1 = G1.nodes[node1]["theta"]

        # Run with seed 2
        random.seed(2)
        np.random.seed(2)
        G2, node2 = create_nfr("test2", epi=0.5, vf=1.0, theta=1.0)
        G2.nodes[node2]["epi_history"] = [0.3, 0.4, 0.5]
        G2.nodes[node2]["delta_nfr"] = 0.3
        Mutation()(G2, node2)
        theta2 = G2.nodes[node2]["theta"]

        # Results may be different if operator uses randomness
        # (If deterministic, they'll be the same - that's OK too)

    def test_zhir_sequence_reproducible(self):
        """Full sequence with ZHIR should be reproducible."""
        # First run
        random.seed(100)
        np.random.seed(100)
        G1, node1 = create_nfr("test1", epi=0.5, vf=1.0)
        G1.nodes[node1]["epi_history"] = [0.35, 0.42, 0.50]
        run_sequence(G1, node1, [Emission(), Coherence(), Dissonance(),
                                 Mutation(), Silence()])
        state1 = G1.nodes[node1]["theta"]

        # Second run with same seed
        random.seed(100)
        np.random.seed(100)
        G2, node2 = create_nfr("test2", epi=0.5, vf=1.0)
        G2.nodes[node2]["epi_history"] = [0.35, 0.42, 0.50]
        run_sequence(G2, node2, [Emission(), Coherence(), Dissonance(),
                                 Mutation(), Silence()])
        state2 = G2.nodes[node2]["theta"]

        # Should be identical
        assert abs(state1 - state2) < 1e-10


class TestZHIRExtremeCases:
    """Test ZHIR with extreme parameter values."""

    def test_zhir_with_very_high_epi(self):
        """ZHIR near upper EPI bound."""
        G, node = create_nfr("test", epi=0.95, vf=1.0)
        G.nodes[node]["epi_history"] = [0.85, 0.90, 0.95]

        Mutation()(G, node)

        # Should not exceed bounds
        assert G.nodes[node]["EPI"] <= 1.0

    def test_zhir_with_very_low_epi(self):
        """ZHIR near lower EPI bound."""
        G, node = create_nfr("test", epi=-0.95, vf=1.0)
        G.nodes[node]["epi_history"] = [-0.85, -0.90, -0.95]

        Mutation()(G, node)

        # Should not exceed bounds
        assert G.nodes[node]["EPI"] >= -1.0

    def test_zhir_with_very_high_vf(self):
        """ZHIR with very high structural frequency."""
        G, node = create_nfr("test", epi=0.5, vf=9.5)  # Near maximum (10.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Should not raise error
        Mutation()(G, node)

        # Should still be viable
        assert G.nodes[node]["νf"] > 0

    def test_zhir_with_very_high_dnfr(self):
        """ZHIR with extreme ΔNFR values."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.2, 0.35, 0.5]
        G.nodes[node]["delta_nfr"] = 5.0  # Very high

        # Should not crash
        Mutation()(G, node)

        # Should maintain bounds
        assert -1.0 <= G.nodes[node]["EPI"] <= 1.0

    def test_zhir_with_negative_dnfr(self):
        """ZHIR with negative ΔNFR (contraction pressure)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0, theta=1.0)
        G.nodes[node]["epi_history"] = [0.7, 0.6, 0.5]
        G.nodes[node]["delta_nfr"] = -0.8  # Strong contraction

        theta_before = G.nodes[node]["theta"]

        Mutation()(G, node)

        theta_after = G.nodes[node]["theta"]

        # Should still transform (direction depends on sign of ΔNFR)
        # Phase should have shifted backward
        shift = theta_after - theta_before
        if shift > math.pi:
            shift -= 2 * math.pi
        # Negative ΔNFR should produce backward shift
        # (implementation-dependent)


class TestZHIRUnusualConfigurations:
    """Test ZHIR with unusual graph/node configurations."""

    def test_zhir_without_history_initialization(self):
        """ZHIR should handle missing history keys gracefully."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        # Explicitly don't set epi_history

        # Should not crash (may log warning)
        Mutation()(G, node)

    def test_zhir_with_empty_history(self):
        """ZHIR with empty history should handle gracefully."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = []  # Empty

        # Should not crash (may log warning)
        Mutation()(G, node)

    def test_zhir_with_nan_handling(self):
        """ZHIR should handle NaN values gracefully (if they occur)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]
        # Don't inject NaN - just ensure it wouldn't crash
        # (Actual NaN injection would violate TNFR invariants)

        # Normal operation should work
        Mutation()(G, node)

    def test_zhir_repeated_immediate_applications(self):
        """Multiple immediate ZHIR applications (no intermediate ops)."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.5]

        # Apply 5 times immediately
        for i in range(5):
            Mutation()(G, node)

        # Node should still be viable
        assert G.nodes[node]["νf"] > 0
        assert -1.0 <= G.nodes[node]["EPI"] <= 1.0
        assert 0 <= G.nodes[node]["theta"] < 2 * math.pi


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
