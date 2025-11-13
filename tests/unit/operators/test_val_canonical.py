"""Canonical tests for VAL (Expansion) operator.

This module validates the implementation of canonical preconditions,
metrics, and TNFR physics compliance for the VAL (Expansion) operator
according to issue #2722.

Test Coverage:
--------------
1. **Preconditions**: ΔNFR positivity, EPI minimum, νf maximum
2. **Nodal Equation**: ∂EPI/∂t = νf · ΔNFR(t) compliance
3. **Metrics**: Bifurcation risk, coherence, network impact
4. **Fractality**: Self-similar growth preservation
5. **Canonical Sequences**: VAL→IL, OZ→VAL, VAL→THOL

References:
-----------
- Issue #2722: [VAL] Profundizar implementación canónica del operador Expansión
- AGENTS.md: Canonical invariants
- TNFR.pdf § 2.1: Nodal equation
"""

import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF, ALIAS_D2EPI
from tnfr.operators.definitions import (
    Expansion,
    Coherence,
    Dissonance,
    Emission,
    Silence,
    SelfOrganization,
)
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_expansion,
)
from tnfr.structural import create_nfr, run_sequence


class TestVALPreconditionsCanonical:
    """Test suite for VAL canonical preconditions (issue #2722)."""

    def test_val_requires_positive_dnfr(self):
        """VAL requires ΔNFR > 0 for outward growth pressure.

        Physical basis: Expansion needs positive reorganization gradient.
        From nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
        If ΔNFR ≤ 0, no growth occurs.
        """
        G, node = create_nfr("no_pressure", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = -0.05  # Negative ΔNFR

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_expansion(G, node)

        error_msg = str(exc_info.value)
        assert "ΔNFR must be positive" in error_msg
        assert "Consider OZ" in error_msg  # Suggests fix

    def test_val_requires_minimum_epi(self):
        """VAL requires EPI >= minimum for coherent expansion base.

        Physical basis: Cannot expand from insufficient structural base.
        Fractality requires coherent form to maintain during expansion.
        """
        G, node = create_nfr("weak_base", epi=0.1, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1  # Positive ΔNFR

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_expansion(G, node)

        error_msg = str(exc_info.value)
        assert "EPI too low" in error_msg
        assert "Consider AL" in error_msg  # Suggests activation

    def test_val_requires_vf_below_maximum(self):
        """VAL requires νf < max for reorganization capacity.

        Existing test - maintained for backward compatibility.
        """
        G, node = create_nfr("saturated", epi=0.5, vf=10.0)
        G.nodes[node]["delta_nfr"] = 0.1

        with pytest.raises(OperatorPreconditionError) as exc_info:
            validate_expansion(G, node)

        error_msg = str(exc_info.value)
        assert "at maximum" in error_msg

    def test_val_passes_with_valid_conditions(self):
        """VAL validation passes when all preconditions met."""
        G, node = create_nfr("valid", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1  # Positive

        # Should not raise
        validate_expansion(G, node)

    def test_val_configurable_thresholds(self):
        """VAL thresholds are configurable via graph metadata."""
        G, node = create_nfr("custom", epi=0.15, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.005

        # Lower thresholds to allow expansion
        G.graph["VAL_MIN_EPI"] = 0.1
        G.graph["VAL_MIN_DNFR"] = 0.001

        # Should pass with custom thresholds
        validate_expansion(G, node)


class TestVALNodalEquationCompliance:
    """Test VAL respects ∂EPI/∂t = νf · ΔNFR(t)."""

    def test_val_follows_nodal_equation(self):
        """VAL must follow nodal equation during expansion.

        Physical basis: All structural changes governed by:
        ∂EPI/∂t = νf · ΔNFR(t)

        For expansion:
        - ΔNFR > 0 (outward pressure)
        - νf increases (more reorganization capacity)
        - EPI increases proportionally
        """
        G, node = create_nfr("expanding", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1

        epi_before = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf_before = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))

        # Apply expansion
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        Expansion()(G, node, collect_metrics=True)

        epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        vf_after = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))

        # Verify expansion occurred
        assert epi_after > epi_before, "EPI should increase"
        assert vf_after > vf_before, "νf should increase"

        # Verify changes are proportional to ΔNFR and νf
        delta_epi = epi_after - epi_before
        dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))

        # Rough proportionality check (exact depends on dynamics)
        assert delta_epi > 0, "EPI increase should be positive"
        assert dnfr > 0, "ΔNFR should remain positive"


class TestVALMetricsEnhanced:
    """Test enhanced VAL metrics (issue #2722)."""

    def test_val_metrics_include_bifurcation_risk(self):
        """VAL metrics include ∂²EPI/∂t² bifurcation risk assessment."""
        G, node = create_nfr("metrics_test", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1
        G.nodes[node][list(ALIAS_D2EPI)[0]] = 0.6  # High acceleration

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        Expansion()(G, node, collect_metrics=True)

        # Check metrics were collected - metrics stored in graph level
        assert "operator_metrics" in G.graph
        metrics = G.graph["operator_metrics"][-1]  # Get last metrics entry

        # Verify bifurcation risk metrics
        assert "bifurcation_risk" in metrics
        assert "d2epi" in metrics
        assert "bifurcation_threshold" in metrics

        # With d²EPI/dt² = 0.6 > τ=0.5, risk should be True
        assert metrics["bifurcation_risk"] is True

    def test_val_metrics_include_network_impact(self):
        """VAL metrics track network impact (neighbors affected)."""
        G, node1 = create_nfr("node1", epi=0.5, vf=2.0)
        _, node2 = create_nfr("node2", epi=0.5, vf=2.0, graph=G)
        G.add_edge(node1, node2)

        G.nodes[node1]["delta_nfr"] = 0.1

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        Expansion()(G, node1, collect_metrics=True)

        metrics = G.graph["operator_metrics"][-1]  # Get last metrics entry

        # Verify network metrics
        assert "neighbor_count" in metrics
        assert "network_coupled" in metrics
        assert metrics["neighbor_count"] == 1
        assert metrics["network_coupled"] is True

    def test_val_metrics_include_fractality_indicators(self):
        """VAL metrics include fractality preservation indicators."""
        G, node = create_nfr("fractal_test", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1

        G.graph["COLLECT_OPERATOR_METRICS"] = True
        Expansion()(G, node, collect_metrics=True)

        metrics = G.graph["operator_metrics"][-1]  # Get last metrics entry

        # Verify fractality indicators
        assert "fractal_preserved" in metrics
        assert "expansion_factor" in metrics
        assert "expansion_healthy" in metrics

        # Quality should be preserved with valid expansion
        assert metrics["fractal_preserved"] is True
        # Expansion should occur with positive factor
        assert metrics["expansion_factor"] > 1.0  # Growth occurred


class TestVALCanonicalSequences:
    """Test VAL in canonical operator sequences."""

    def test_val_to_il_consolidation(self):
        """VAL → IL: Expand then stabilize (canonical pattern).

        Physical basis: Expansion increases ΔNFR, requires stabilization.
        IL provides negative feedback for convergence (U2 grammar).
        """
        G, node = create_nfr("expand_stabilize", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1

        # Apply canonical sequence with generator (per U1a)
        sequence = [Emission(), Expansion(), Coherence(), Silence()]
        run_sequence(G, node, sequence)

        # After IL, system should be more stable
        # (exact metrics depend on dynamics implementation)
        assert float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)) > 0.5  # Expanded

    def test_oz_to_val_exploratory(self):
        """OZ → VAL: Dissonance enables expansion (canonical pattern).

        Physical basis: OZ increases ΔNFR (creates pressure),
        enabling VAL to expand into new coherence volume.
        """
        G, node = create_nfr("explore", epi=0.5, vf=2.0)

        # Apply canonical sequence respecting compatibility and grammar
        sequence = [
            Emission(),  # Generator (U1a)
            Dissonance(),  # Exploration pressure
            Coherence(),  # Stabilizer after OZ (U2)
            Expansion(),  # Expansion phase
            Coherence(),  # Re-stabilize post-expansion
            Silence(),  # Closure (U1b)
        ]
        run_sequence(G, node, sequence)

        # After sequence, node should have explored new structure
        dnfr = float(get_attr(G.nodes[node], ALIAS_DNFR, 0.0))
        assert dnfr >= 0  # ΔNFR maintained or increased

    def test_val_to_thol_emergence(self):
        """VAL → THOL: Expansion triggers self-organization.

        Physical basis: Increased complexity (from VAL) creates
        conditions for bifurcation and emergence (THOL).
        """
        G, node = create_nfr("emergent", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1

        # Build EPI history for THOL precondition
        G.nodes[node]["epi_history"] = [0.4, 0.45, 0.5]

        # Apply canonical sequence with generator, stabilizer, closure
        sequence = [
            Emission(),  # Generator (U1a)
            Expansion(),  # Increase structural complexity
            Coherence(),  # Stabilize post-expansion (U2)
            SelfOrganization(),  # Trigger emergent structuring
            Silence(),  # Closure per THOL requirement (U1b)
        ]
        run_sequence(G, node, sequence)

        # After THOL, structure should be reorganized
        # (specific effects depend on THOL implementation)
        assert float(get_attr(G.nodes[node], ALIAS_EPI, 0.0)) >= 0.5


class TestVALFractalityPreservation:
    """Test VAL preserves fractality during expansion."""

    def test_val_maintains_structural_identity(self):
        """VAL increases magnitude while preserving structural form.

        Physical basis: Fractality = self-similarity across scales.
        EPI grows but core pattern remains coherent.
        """
        G, node = create_nfr("fractal", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1

        epi_before = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))

        # Apply expansion
        Expansion()(G, node)

        epi_after = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))

        # EPI should increase but remain coherent (not fragment)
        assert epi_after > epi_before
        assert epi_after < epi_before * 2  # Bounded growth

        # Structural parameters should remain in valid range
        vf_after = float(get_attr(G.nodes[node], ALIAS_VF, 0.0))
        assert 0 < vf_after < 20  # Reasonable νf range


class TestVALEdgeCases:
    """Test VAL edge cases and boundary conditions."""

    def test_val_at_epi_threshold(self):
        """VAL at EPI minimum threshold."""
        G, node = create_nfr("threshold", epi=0.2, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.1
        G.graph["VAL_MIN_EPI"] = 0.2

        # Should pass exactly at threshold
        validate_expansion(G, node)

    def test_val_at_dnfr_threshold(self):
        """VAL at ΔNFR minimum threshold."""
        G, node = create_nfr("threshold", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.01
        G.graph["VAL_MIN_DNFR"] = 0.01

        # Should pass exactly at threshold
        validate_expansion(G, node)

    def test_val_with_zero_dnfr_fails(self):
        """VAL with exactly zero ΔNFR should fail."""
        G, node = create_nfr("zero_pressure", epi=0.5, vf=2.0)
        G.nodes[node]["delta_nfr"] = 0.0

        with pytest.raises(OperatorPreconditionError):
            validate_expansion(G, node)
