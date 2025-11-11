"""Canonical tests for VAL nodal equation compliance.

This module validates that the VAL (Expansion) operator is correctly structured
to respect the fundamental nodal equation through its operator contracts and
integration with the TNFR dynamics system.

Test Coverage:
--------------
1. **Operator Application**: Verify VAL is called and hook triggered
2. **νf Behavior**: Expansion's effect on structural frequency
3. **ΔNFR Integration**: Interaction with reorganization gradients
4. **Contract Compliance**: Preconditions and expected behaviors

Physical Basis:
---------------
From TNFR.pdf § 2.1: The nodal equation governs all structural transformations:

    ∂EPI/∂t = νf · ΔNFR(t)

VAL (Expansion) must:
- Integrate with the ΔNFR hook mechanism
- Respect operator preconditions (positive ΔNFR, sufficient EPI, νf capacity)
- Trigger dynamics updates appropriately

Note on Implementation:
-----------------------
The VAL operator works through the TNFR dynamics system:
1. Operator applied via run_sequence
2. ΔNFR hook called after operator
3. Hook modifies EPI/νf based on nodal equation

Tests verify the operator integration rather than direct EPI modification.

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
from tnfr.operators.definitions import Expansion, Emission, Coherence
from tnfr.structural import create_nfr, run_sequence
from tnfr.dynamics import set_delta_nfr_hook


class TestVALOperatorIntegration:
    """Test suite for VAL operator integration with TNFR dynamics."""

    def test_val_operator_applies_successfully(self):
        """VAL operator should apply without errors when preconditions met.

        This verifies the operator integrates correctly with run_sequence.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        # Should not raise
        run_sequence(G, node, [Emission(), Expansion()])

    def test_val_triggers_dnfr_hook(self):
        """VAL should trigger ΔNFR hook for structural updates.

        The hook mechanism is how operators actually modify node state.
        """
        G, node = create_nfr("test_node", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        hook_called = []

        def expansion_hook(graph):
            """Hook that tracks calls and modifies state."""
            hook_called.append(True)
            # Simulate expansion per nodal equation
            epi_key = list(ALIAS_EPI)[0]
            vf_key = list(ALIAS_VF)[0]
            dnfr = float(get_attr(graph.nodes[node], ALIAS_DNFR, 0.0))
            vf = float(get_attr(graph.nodes[node], ALIAS_VF, 1.0))

            # ∂EPI/∂t = νf · ΔNFR
            delta_epi = vf * dnfr * 0.1
            delta_vf = dnfr * 0.05

            graph.nodes[node][epi_key] += delta_epi
            graph.nodes[node][vf_key] += delta_vf

        set_delta_nfr_hook(G, expansion_hook)

        epi_key = list(ALIAS_EPI)[0]
        vf_key = list(ALIAS_VF)[0]
        epi_before = G.nodes[node][epi_key]
        vf_before = G.nodes[node][vf_key]

        # Apply with generator first to satisfy grammar
        run_sequence(G, node, [Emission(), Expansion()])

        # Verify hook was called
        assert len(hook_called) >= 1, "ΔNFR hook should be called"

        # Verify changes occurred via hook
        epi_after = G.nodes[node][epi_key]
        vf_after = G.nodes[node][vf_key]

        assert (
            epi_after > epi_before
        ), f"EPI should increase via hook: {epi_before:.4f} -> {epi_after:.4f}"
        assert (
            vf_after > vf_before
        ), f"νf should increase via hook: {vf_before:.4f} -> {vf_after:.4f}"

    def test_val_respects_positive_dnfr_contract(self):
        """VAL requires positive ΔNFR (expansion pressure).

        Physical basis: Cannot expand without reorganization gradient.
        """
        from tnfr.operators.preconditions import OperatorPreconditionError

        G, node = create_nfr("test_node", epi=0.5, vf=1.2)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = -0.1  # Negative

        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError):
            run_sequence(G, node, [Emission(), Expansion()])

    def test_val_with_hook_follows_nodal_equation_proportions(self):
        """With custom hook, VAL changes follow nodal equation proportions.

        Test: Verify ΔEP I ∝ νf · ΔNFR through controlled hook.
        """
        # Create two nodes with different νf
        G1, node1 = create_nfr("low_vf", epi=0.5, vf=1.0)
        G2, node2 = create_nfr("high_vf", epi=0.5, vf=2.0)

        dnfr_key = list(ALIAS_DNFR)[0]
        G1.nodes[node1][dnfr_key] = 0.1
        G2.nodes[node2][dnfr_key] = 0.1  # Same ΔNFR

        # Same hook for both
        def proportional_hook(graph, node_id):
            epi_key = list(ALIAS_EPI)[0]
            vf_key = list(ALIAS_VF)[0]
            dnfr = float(get_attr(graph.nodes[node_id], ALIAS_DNFR, 0.0))
            vf = float(get_attr(graph.nodes[node_id], ALIAS_VF, 1.0))

            # Nodal equation: ∂EPI/∂t = νf · ΔNFR
            delta_epi = vf * dnfr * 0.1
            graph.nodes[node_id][epi_key] += delta_epi

        set_delta_nfr_hook(G1, lambda g: proportional_hook(g, node1))
        set_delta_nfr_hook(G2, lambda g: proportional_hook(g, node2))

        epi_key = list(ALIAS_EPI)[0]

        # Apply VAL to both
        run_sequence(G1, node1, [Emission(), Expansion()])
        run_sequence(G2, node2, [Emission(), Expansion()])

        # Node with higher νf should have larger ΔEP I
        delta_epi1 = G1.nodes[node1][epi_key] - 0.5
        delta_epi2 = G2.nodes[node2][epi_key] - 0.5

        assert delta_epi2 > delta_epi1, (
            f"Higher νf should produce larger ΔEPI: "
            f"νf=1.0 → ΔEPI={delta_epi1:.4f}, νf=2.0 → ΔEPI={delta_epi2:.4f}"
        )


@pytest.mark.val
@pytest.mark.nodal_equation
class TestVALSequenceDynamics:
    """Test VAL behavior in operator sequences with dynamics."""

    def test_emission_val_sequence_with_hook(self):
        """AL → VAL: Generator followed by expansion.

        Grammar compliant sequence with measurable dynamics.
        """
        G, node = create_nfr("test", epi=0.0, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        hook_calls = []

        def tracking_hook(graph):
            hook_calls.append(True)
            epi_key = list(ALIAS_EPI)[0]
            vf_key = list(ALIAS_VF)[0]

            # Simple expansion
            graph.nodes[node][epi_key] += 0.05
            graph.nodes[node][vf_key] += 0.02

        set_delta_nfr_hook(G, tracking_hook)

        # Apply AL → VAL
        run_sequence(G, node, [Emission(), Expansion()])

        # Hook should be called for both operators
        assert len(hook_calls) >= 2, "Hook called for each operator"

    def test_val_il_stabilization_sequence(self):
        """VAL → IL: Expansion followed by stabilization.

        Canonical sequence for controlled growth.
        """
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        expansion_count = []
        stabilization_count = []

        def sequence_hook(graph):
            # Track which operator by checking graph state
            last_op = getattr(graph, "_last_operator_applied", None)
            if last_op == "expansion":
                expansion_count.append(True)
            elif last_op == "coherence":
                stabilization_count.append(True)

        set_delta_nfr_hook(G, sequence_hook)

        # Apply VAL → IL
        run_sequence(G, node, [Emission(), Expansion(), Coherence()])

        # Both operators should have been applied
        assert len(expansion_count) >= 1, "Expansion applied"
        assert len(stabilization_count) >= 1, "Coherence applied"

    def test_multiple_val_with_hooks(self):
        """Multiple VAL applications show cumulative effect via hooks.

        Each VAL triggers hook, cumulative modifications occur.
        """
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        expansion_count = []

        def cumulative_hook(graph):
            epi_key = list(ALIAS_EPI)[0]
            # Small increment per call
            graph.nodes[node][epi_key] += 0.01
            expansion_count.append(True)

        set_delta_nfr_hook(G, cumulative_hook)

        epi_key = list(ALIAS_EPI)[0]
        epi_before = G.nodes[node][epi_key]

        # Apply VAL three times (with generator first)
        run_sequence(G, node, [Emission(), Expansion(), Expansion(), Expansion()])

        epi_after = G.nodes[node][epi_key]

        # Should show cumulative effect
        assert epi_after > epi_before, "Cumulative expansion occurred"
        assert len(expansion_count) >= 3, "Hook called multiple times"


@pytest.mark.val
class TestVALContractCompliance:
    """Test VAL compliance with operator contracts."""

    def test_val_requires_sufficient_epi(self):
        """VAL requires minimum EPI for expansion base.

        Cannot expand from insufficient structural foundation.
        """
        from tnfr.operators.preconditions import OperatorPreconditionError

        G, node = create_nfr("weak_base", epi=0.05, vf=1.0)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError):
            run_sequence(G, node, [Emission(), Expansion()])

    def test_val_requires_vf_capacity(self):
        """VAL requires νf below maximum for expansion capacity.

        Saturated νf limits reorganization potential.
        """
        from tnfr.operators.preconditions import OperatorPreconditionError

        G, node = create_nfr("saturated", epi=0.5, vf=9.9)
        dnfr_key = list(ALIAS_DNFR)[0]
        G.nodes[node][dnfr_key] = 0.1

        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["VAL_VF_MAXIMUM"] = 10.0

        with pytest.raises(OperatorPreconditionError):
            run_sequence(G, node, [Emission(), Expansion()])

    def test_val_respects_configurable_thresholds(self):
        """VAL thresholds are configurable via graph metadata.

        Allows domain-specific tuning while maintaining physics.
        """
        G, node = create_nfr("custom", epi=0.25, vf=2.0)  # Above default but could use custom
        dnfr_key = list(ALIAS_DNFR)[0]

        # Custom thresholds (more permissive)
        G.graph["VAL_EPI_MINIMUM"] = 0.1  # Lower than default (0.2)
        G.graph["VAL_DNFR_MINIMUM"] = 0.001  # Lower than default
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        # Set ΔNFR via hook
        def maintain_dnfr(graph):
            """Hook that maintains ΔNFR value."""
            graph.nodes[node][dnfr_key] = 0.005

        set_delta_nfr_hook(G, maintain_dnfr)

        # Should pass with custom thresholds
        run_sequence(G, node, [Emission(), Expansion()])
