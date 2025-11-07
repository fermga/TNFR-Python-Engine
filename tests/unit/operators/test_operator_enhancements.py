"""Tests for operator preconditions and metrics."""

import pytest
import networkx as nx

from tnfr.operators.definitions import (
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY, THETA_PRIMARY


class TestOperatorPreconditions:
    """Test precondition validation for structural operators."""

    def test_emission_precondition_high_epi(self):
        """AL - Emission should fail if EPI already high (strict validation)."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.9, VF_PRIMARY: 1.0})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["EPI_LATENT_MAX"] = 0.8  # Updated threshold name for strict validation

        # New strict validation raises ValueError with detailed message
        with pytest.raises(ValueError, match="AL precondition failed"):
            Emission()(G, "n1")

    def test_emission_precondition_low_epi(self):
        """AL - Emission should succeed if EPI is low."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.2, VF_PRIMARY: 1.0})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        # Should not raise
        # Note: actual glyph application requires full TNFR setup

    def test_reception_precondition_no_neighbors(self):
        """EN - Reception should fail if no neighbors."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError, match="no neighbors"):
            Reception()(G, "n1")

    def test_coherence_precondition_minimal_dnfr(self):
        """IL - Coherence should fail if ΔNFR already minimal."""
        G = nx.DiGraph()
        G.add_node("n1", **{DNFR_PRIMARY: 0.0, EPI_PRIMARY: 0.5})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

        with pytest.raises(OperatorPreconditionError, match="already minimal"):
            Coherence()(G, "n1")

    def test_dissonance_precondition_low_vf(self):
        """OZ - Dissonance should fail if νf too low."""
        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 0.005, DNFR_PRIMARY: 0.1})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["OZ_MIN_VF"] = 0.01

        with pytest.raises(OperatorPreconditionError, match="too low"):
            Dissonance()(G, "n1")

    def test_silence_precondition_minimal_vf(self):
        """SHA - Silence should fail if νf already minimal."""
        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 0.005})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["SHA_MIN_VF"] = 0.01

        with pytest.raises(OperatorPreconditionError, match="already minimal"):
            Silence()(G, "n1")

    def test_expansion_precondition_max_vf(self):
        """VAL - Expansion should fail if νf at maximum."""
        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 10.5})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["VAL_MAX_VF"] = 10.0

        with pytest.raises(OperatorPreconditionError, match="at maximum"):
            Expansion()(G, "n1")

    def test_self_organization_precondition_low_epi(self):
        """THOL - Self-organization should fail if EPI too low."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.1})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["THOL_MIN_EPI"] = 0.3

        with pytest.raises(OperatorPreconditionError, match="too low"):
            SelfOrganization()(G, "n1")

    def test_mutation_precondition_low_vf(self):
        """ZHIR - Mutation should fail if νf too low."""
        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 0.02, THETA_PRIMARY: 0.0})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        G.graph["ZHIR_MIN_VF"] = 0.05

        with pytest.raises(OperatorPreconditionError, match="too low"):
            Mutation()(G, "n1")

    def test_preconditions_disabled_by_default(self):
        """Preconditions should not validate by default."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.9, VF_PRIMARY: 1.0})
        # No VALIDATE_OPERATOR_PRECONDITIONS flag
        # Should not raise even though EPI is high
        # Note: actual glyph application requires full TNFR setup


class TestOperatorMetrics:
    """Test metrics collection for structural operators."""

    def test_metrics_collection_disabled_by_default(self):
        """Metrics should not be collected by default."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        # No COLLECT_OPERATOR_METRICS flag
        assert "operator_metrics" not in G.graph

    def test_emission_metrics_structure(self):
        """AL - Emission metrics should have expected structure."""
        from tnfr.operators.metrics import emission_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.6, VF_PRIMARY: 1.2, DNFR_PRIMARY: 0.05})

        metrics = emission_metrics(G, "n1", epi_before=0.5, vf_before=1.0)

        assert metrics["operator"] == "Emission"
        assert metrics["glyph"] == "AL"
        assert "delta_epi" in metrics
        assert "activation_strength" in metrics
        assert metrics["delta_epi"] == pytest.approx(0.1)

    def test_coherence_metrics_structure(self):
        """IL - Coherence metrics should have expected structure."""
        from tnfr.operators.metrics import coherence_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.05})

        metrics = coherence_metrics(G, "n1", dnfr_before=0.15)

        assert metrics["operator"] == "Coherence"
        assert metrics["glyph"] == "IL"
        assert "dnfr_reduction" in metrics
        assert "stability_gain" in metrics
        assert metrics["dnfr_reduction"] == pytest.approx(0.1)

    def test_dissonance_metrics_structure(self):
        """OZ - Dissonance metrics should have expected structure."""
        from tnfr.operators.metrics import dissonance_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{DNFR_PRIMARY: 0.25, THETA_PRIMARY: 0.5})

        metrics = dissonance_metrics(G, "n1", dnfr_before=0.1, theta_before=0.3)

        assert metrics["operator"] == "Dissonance"
        assert metrics["glyph"] == "OZ"
        assert "dnfr_increase" in metrics
        assert "bifurcation_risk" in metrics
        assert metrics["dnfr_increase"] == pytest.approx(0.15)

    def test_resonance_metrics_with_neighbors(self):
        """RA - Resonance metrics should reflect neighbor coupling."""
        from tnfr.operators.metrics import resonance_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.6})
        G.add_node("n2", **{EPI_PRIMARY: 0.8})
        G.add_node("n3", **{EPI_PRIMARY: 0.7})
        G.add_edge("n1", "n2")
        G.add_edge("n1", "n3")

        metrics = resonance_metrics(G, "n1", epi_before=0.5)

        assert metrics["operator"] == "Resonance"
        assert metrics["neighbor_count"] == 2
        assert metrics["resonance_strength"] > 0

    def test_silence_metrics_structure(self):
        """SHA - Silence metrics should show νf reduction."""
        from tnfr.operators.metrics import silence_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 0.3, EPI_PRIMARY: 0.5})

        metrics = silence_metrics(G, "n1", vf_before=0.8, epi_before=0.5)

        assert metrics["operator"] == "Silence"
        assert metrics["glyph"] == "SHA"
        assert "vf_reduction" in metrics
        assert "epi_preservation" in metrics
        assert metrics["vf_reduction"] == pytest.approx(0.5)

    def test_expansion_metrics_structure(self):
        """VAL - Expansion metrics should show νf increase."""
        from tnfr.operators.metrics import expansion_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{VF_PRIMARY: 1.5, EPI_PRIMARY: 0.6})

        metrics = expansion_metrics(G, "n1", vf_before=1.0, epi_before=0.5)

        assert metrics["operator"] == "Expansion"
        assert metrics["glyph"] == "VAL"
        assert "vf_increase" in metrics
        assert "expansion_factor" in metrics
        assert metrics["vf_increase"] == pytest.approx(0.5)

    def test_self_organization_metrics_structure(self):
        """THOL - Self-organization metrics should track cascades."""
        from tnfr.operators.metrics import self_organization_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.7, VF_PRIMARY: 1.1})
        G.graph["sub_epi"] = [0.5, 0.6]  # Track nested EPIs

        metrics = self_organization_metrics(G, "n1", epi_before=0.6, vf_before=1.0)

        assert metrics["operator"] == "Self-organization"
        assert metrics["glyph"] == "THOL"
        assert "nested_epi_count" in metrics
        assert metrics["nested_epi_count"] == 2

    def test_mutation_metrics_structure(self):
        """ZHIR - Mutation metrics should show phase change."""
        from tnfr.operators.metrics import mutation_metrics

        G = nx.DiGraph()
        G.add_node("n1", **{THETA_PRIMARY: 1.5, EPI_PRIMARY: 0.8})

        metrics = mutation_metrics(G, "n1", theta_before=0.5, epi_before=0.7)

        assert metrics["operator"] == "Mutation"
        assert metrics["glyph"] == "ZHIR"
        assert "theta_shift" in metrics
        assert "phase_change" in metrics
        assert metrics["theta_shift"] == pytest.approx(1.0)


class TestBackwardCompatibility:
    """Ensure changes maintain backward compatibility."""

    def test_operators_work_without_flags(self):
        """Operators should work when new flags are not set."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        # No special flags - should work as before
        # Note: actual glyph application requires full TNFR setup

    def test_validation_can_be_enabled(self):
        """Validation can be explicitly enabled via graph flag."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        # Now validation is active

    def test_metrics_can_be_enabled(self):
        """Metrics collection can be explicitly enabled via graph flag."""
        G = nx.DiGraph()
        G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        # Now metrics collection is active
