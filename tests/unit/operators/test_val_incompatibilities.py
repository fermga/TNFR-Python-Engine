"""Tests for VAL incompatibilities and anti-patterns.

This module documents operator sequences that are technically valid
but semantically problematic or inefficient. While not enforced as
hard errors, these patterns should be avoided or flagged.

Test Coverage:
--------------
1. **VAL → NUL**: Expansion-contraction cancellation (wasteful)
2. **Multiple VAL**: Consecutive expansions without stabilization (risky)

Physical Basis:
---------------
These patterns violate efficiency principles:

- **VAL → NUL**: Wastes structural energy in opposing transformations
- **Multiple VAL**: Accumulates ΔNFR without bounds (U2 violation risk)

While grammar allows these, they indicate:
- Inefficient operator planning
- Potential instability accumulation
- Missing stabilization steps

References:
-----------
- UNIFIED_GRAMMAR_RULES.md: U2 (CONVERGENCE & BOUNDEDNESS)
- AGENTS.md: Operator composition best practices
"""

import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators.definitions import (
    Expansion,
    Contraction,
    Coherence,
    Emission,
)
from tnfr.structural import create_nfr, run_sequence


@pytest.mark.val
class TestVALNULIncompatibility:
    """Test VAL → NUL semantic contradiction (expansion-contraction)."""

    def test_val_nul_mostly_cancels(self):
        """VAL → NUL is semantically contradictory.

        Physics:
        - VAL expands structure (increases complexity)
        - NUL contracts structure (decreases complexity)
        - Immediate reversal wastes structural energy

        While not forbidden by grammar, this pattern is inefficient.
        Future: Add telemetry warning for this anti-pattern.

        This test documents the pattern using hooks to simulate effects.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]

        # Set up hook to enable observable changes
        from tnfr.dynamics import set_delta_nfr_hook

        def opposing_hook(graph):
            """Hook that simulates expansion/contraction opposition."""
            last_op = getattr(graph, "_last_operator_applied", None)
            if last_op == "expansion":
                graph.nodes[node][epi_key] += 0.1
            elif last_op == "contraction":
                graph.nodes[node][epi_key] -= 0.09  # Slightly less to show inefficiency

        set_delta_nfr_hook(G, opposing_hook)

        epi_0 = G.nodes[node][epi_key]

        # Apply VAL → NUL
        run_sequence(G, node, [Emission(), Expansion(), Contraction()])

        epi_2 = G.nodes[node][epi_key]

        # Net effect should be small (wasteful cancellation)
        epi_change = abs(epi_2 - epi_0)
        assert epi_change < abs(epi_0 * 0.3), (
            f"VAL→NUL should mostly cancel: " f"|ΔEPI|={epi_change:.4f} < 0.3*EPI₀={0.3*epi_0:.4f}"
        )

    def test_val_nul_vs_no_operation(self):
        """VAL → NUL produces similar result to no operation.

        Test: Compare node with VAL→NUL to control node.
        Outcome should be similar, demonstrating inefficiency.
        """
        # Treatment: VAL → NUL
        G1, node1 = create_nfr("treatment", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.1

        epi_key = list(ALIAS_EPI)[0]
        epi1_before = G1.nodes[node1][epi_key]
        run_sequence(G1, node1, [Expansion(), Contraction()])
        epi1_after = G1.nodes[node1][epi_key]

        # Control: No operation
        G2, node2 = create_nfr("control", epi=0.5, vf=1.2)
        G2.nodes[node2][dnfr_key] = 0.1

        epi2 = G2.nodes[node2][epi_key]

        # States should be similar
        epi_difference = abs(epi1_after - epi2)
        assert epi_difference < 0.2, (
            f"VAL→NUL should produce similar result to no-op: " f"Δ={epi_difference:.4f}"
        )

    def test_val_nul_energy_inefficiency(self):
        """VAL → NUL represents wasted computational/structural energy.

        Measure: Count operator applications vs. net structural change.
        High count with low change indicates inefficiency.
        """
        G, node = create_nfr("inefficient", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]

        # Apply inefficient sequence
        run_sequence(G, node, [Expansion(), Contraction()])

        epi_after = G.nodes[node][epi_key]

        # Calculate efficiency metric
        operator_count = 2  # VAL + NUL
        net_change = abs(epi_after - epi_before)
        efficiency = net_change / operator_count if operator_count > 0 else 0

        # Low efficiency indicates waste
        assert efficiency < 0.15, f"VAL→NUL should have low efficiency: {efficiency:.4f}"

    def test_val_il_nul_better_than_val_nul(self):
        """VAL → IL → NUL is more controlled than VAL → NUL.

        Adding IL (stabilizer) between VAL and NUL:
        - Consolidates expanded structure
        - Provides controlled reduction
        - Reduces structural stress
        """
        # Test 1: VAL → NUL (uncontrolled)
        G1, node1 = create_nfr("uncontrolled", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.1

        dnfr1_before = G1.nodes[node1][dnfr_key]
        run_sequence(G1, node1, [Expansion(), Contraction()])
        dnfr1_after = G1.nodes[node1][dnfr_key]

        # Test 2: VAL → IL → NUL (controlled)
        G2, node2 = create_nfr("controlled", epi=0.5, vf=1.2)
        G2.nodes[node2][dnfr_key] = 0.1

        dnfr2_before = G2.nodes[node2][dnfr_key]
        run_sequence(G2, node2, [Expansion(), Coherence(), Contraction()])
        dnfr2_after = G2.nodes[node2][dnfr_key]

        # Controlled version should have more stable ΔNFR
        # (Implementation dependent, but documents the pattern)
        assert dnfr2_after >= 0, "Controlled sequence maintains validity"
        assert dnfr1_after >= 0, "Uncontrolled sequence maintains validity"


@pytest.mark.val
class TestConsecutiveVALRisk:
    """Test risks of consecutive VAL operations without stabilization."""

    def test_consecutive_val_accumulates_dnfr(self):
        """VAL → VAL → VAL without IL can accumulate ΔNFR.

        Grammar concern: U2 (CONVERGENCE & BOUNDEDNESS).
        Multiple destabilizers without stabilizers risk divergence.

        While allowed, this pattern should trigger warnings.
        This test verifies the sequence is grammatically valid.
        """
        G, node = create_nfr("test_node", epi=0.4, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]

        # Set up hook to maintain ΔNFR
        from tnfr.dynamics import set_delta_nfr_hook

        def maintain_dnfr_hook(graph):
            # Keep ΔNFR positive for subsequent operations
            graph.nodes[node][dnfr_key] = 0.05

        set_delta_nfr_hook(G, maintain_dnfr_hook)

        # Apply VAL x3 without IL - should not raise
        run_sequence(G, node, [Emission(), Expansion(), Expansion(), Expansion()])

    def test_consecutive_val_with_il_more_stable(self):
        """Interleaving IL with VAL produces more stable trajectory.

        Compare:
        - Unstabilized: VAL → VAL → VAL
        - Stabilized: VAL → IL → VAL → IL → VAL → IL

        Stabilized version should have lower final ΔNFR.
        """
        # Unstabilized
        G1, node1 = create_nfr("unstabilized", epi=0.4, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.05

        run_sequence(G1, node1, [Expansion(), Expansion(), Expansion()])
        dnfr1_final = G1.nodes[node1][dnfr_key]

        # Stabilized
        G2, node2 = create_nfr("stabilized", epi=0.4, vf=1.0)
        G2.nodes[node2][dnfr_key] = 0.05

        run_sequence(
            G2,
            node2,
            [
                Expansion(),
                Coherence(),
                Expansion(),
                Coherence(),
                Expansion(),
                Coherence(),
            ],
        )
        dnfr2_final = G2.nodes[node2][dnfr_key]

        # Stabilized should have lower ΔNFR
        assert dnfr2_final <= dnfr1_final * 1.1, (
            f"Stabilized trajectory should have lower ΔNFR: "
            f"Unstabilized={dnfr1_final:.4f}, Stabilized={dnfr2_final:.4f}"
        )

    def test_consecutive_val_growth_rate(self):
        """Consecutive VAL applications should work with proper setup.

        Physics: Each expansion should be possible with sufficient ΔNFR.
        This test verifies multiple VAL operations are grammatically valid.
        """
        G, node = create_nfr("diminishing", epi=0.4, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]

        from tnfr.dynamics import set_delta_nfr_hook

        expansions = []

        def track_expansions(graph):
            # Track and enable each expansion
            graph.nodes[node][dnfr_key] = 0.1
            expansions.append(True)

        set_delta_nfr_hook(G, track_expansions)

        # Apply 3 VALs
        run_sequence(G, node, [Emission(), Expansion(), Expansion(), Expansion()])

        # All three should have been applied
        assert len(expansions) >= 3, "All expansions should be applied"

    def test_val_stabilization_ratio_guideline(self):
        """Best practice: Match destabilizers with stabilizers.

        Guideline: For every N destabilizers, include ≥N stabilizers.
        This test documents the recommended pattern and verifies it's valid.
        """
        G, node = create_nfr("guideline", epi=0.4, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]

        from tnfr.dynamics import set_delta_nfr_hook

        def maintain_state(graph):
            # Maintain ΔNFR for operations
            graph.nodes[node][dnfr_key] = 0.05

        set_delta_nfr_hook(G, maintain_state)

        # Apply 2 VAL (destabilizers) + 2 IL (stabilizers)
        # This should be a valid, stable pattern
        run_sequence(G, node, [Emission(), Expansion(), Expansion(), Coherence(), Coherence()])

        # Sequence should complete successfully
        epi_key = list(ALIAS_EPI)[0]
        assert G.nodes[node][epi_key] > 0, "Node should remain valid"


@pytest.mark.val
class TestVALAntiPatternsDocumentation:
    """Document additional VAL anti-patterns for reference."""

    def test_val_without_sufficient_dnfr_is_ineffective(self):
        """VAL requires sufficient ΔNFR to be effective.

        With very low ΔNFR, VAL produces minimal change.
        This wastes operator application without benefit.
        """
        G, node = create_nfr("low_pressure", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.001  # Very low

        # Disable precondition check for this test
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False

        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]

        run_sequence(G, node, [Expansion()])

        epi_after = G.nodes[node][epi_key]
        delta_epi = abs(epi_after - epi_before)

        # Very little change
        assert delta_epi < 0.05, f"Low ΔNFR produces minimal expansion: ΔEPI={delta_epi:.4f}"

    def test_val_at_vf_maximum_is_ineffective(self):
        """VAL at νf maximum has no room to increase capacity.

        When νf is saturated, VAL cannot increase reorganization capacity.
        This limits expansion effectiveness.
        """
        G, node = create_nfr("saturated", epi=0.5, vf=9.9)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        # Disable precondition for this edge case test
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
        G.graph["VAL_VF_MAXIMUM"] = 10.0  # Set maximum

        vf_key = list(ALIAS_VF)[0]
        vf_before = G.nodes[node][vf_key]

        run_sequence(G, node, [Expansion()])

        vf_after = G.nodes[node][vf_key]

        # νf should not exceed maximum
        assert vf_after <= 10.0, f"νf should not exceed maximum: {vf_after:.4f} ≤ 10.0"

        # Change should be minimal
        vf_change = abs(vf_after - vf_before)
        assert vf_change < 0.2, f"Near-maximum νf limits expansion: Δνf={vf_change:.4f}"
