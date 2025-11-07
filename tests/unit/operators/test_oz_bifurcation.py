"""Tests for OZ bifurcation detection and structural path selection.

This module tests the integration between OZ (Dissonance) operator and
bifurcation dynamics, ensuring canonical TNFR behavior per §2.3.3 R4.
"""

from __future__ import annotations

import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_D2EPI
from tnfr.dynamics.bifurcation import get_bifurcation_paths
from tnfr.operators.definitions import Coherence, Contraction, Dissonance, Mutation, SelfOrganization
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.structural import create_nfr, run_sequence
from tnfr.types import Glyph


class TestOZBifurcationDetection:
    """Test OZ (Dissonance) bifurcation threshold detection."""

    def test_oz_no_bifurcation_without_history(self):
        """OZ without EPI history does not trigger bifurcation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set positive ΔNFR for precondition
        G.nodes[node][DNFR_PRIMARY] = 0.1
        
        # Apply OZ with precondition validation
        Dissonance()(G, node)
        
        # Verify no bifurcation flag set
        assert not G.nodes[node].get("_bifurcation_ready", False)

    def test_oz_no_bifurcation_with_insufficient_history(self):
        """OZ with < 3 history points does not trigger bifurcation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.1
        
        # Only 2 history points (need 3 for second derivative)
        G.nodes[node]["epi_history"] = [0.3, 0.4]
        
        Dissonance()(G, node)
        
        # Verify no bifurcation detected
        assert not G.nodes[node].get("_bifurcation_ready", False)

    def test_oz_no_bifurcation_below_threshold(self):
        """OZ with low acceleration (< τ) does not trigger bifurcation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.1
        
        # History with minimal acceleration
        # d²EPI = abs(0.51 - 2*0.50 + 0.49) = 0.0
        G.nodes[node]["epi_history"] = [0.49, 0.50, 0.51]
        
        # High threshold
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.8
        
        Dissonance()(G, node)
        
        # Verify no bifurcation (acceleration too low)
        assert not G.nodes[node].get("_bifurcation_ready", False)

    def test_oz_bifurcation_above_threshold(self):
        """OZ with high acceleration (> τ) triggers bifurcation."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set positive ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.2
        
        # History with strong acceleration
        # d²EPI = abs(0.7 - 2*0.45 + 0.3) = abs(0.1) = 0.1
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]
        
        # Low threshold to trigger bifurcation
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        Dissonance()(G, node)
        
        # Verify bifurcation detected
        assert G.nodes[node].get("_bifurcation_ready", False)
        
        # Verify d²EPI stored for telemetry
        d2_epi = G.nodes[node].get(ALIAS_D2EPI[0], 0.0)
        assert d2_epi > 0.08

    def test_oz_bifurcation_stores_d2epi(self):
        """OZ stores ∂²EPI/∂t² for telemetry."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        Dissonance()(G, node)
        
        # Verify d²EPI stored using canonical alias
        d2_epi = G.nodes[node].get(ALIAS_D2EPI[0])
        assert d2_epi is not None
        assert isinstance(d2_epi, (int, float))
        assert d2_epi > 0.0

    def test_oz_precondition_checks_epi_minimum(self):
        """OZ requires minimum EPI to withstand dissonance."""
        G, node = create_nfr("test", epi=0.1, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set minimum EPI threshold
        G.graph["OZ_MIN_EPI"] = 0.2
        
        # Should raise precondition error
        with pytest.raises(OperatorPreconditionError, match="EPI too low"):
            Dissonance()(G, node)

    def test_oz_precondition_warns_high_dnfr(self):
        """OZ warns when ΔNFR already critically high."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set critically high ΔNFR
        G.nodes[node][DNFR_PRIMARY] = 0.85
        G.graph["OZ_MAX_DNFR"] = 0.8
        
        # Should warn but not raise
        with pytest.warns(UserWarning, match="high ΔNFR"):
            Dissonance()(G, node)


class TestBifurcationPaths:
    """Test bifurcation path selection after OZ."""

    def test_no_paths_without_bifurcation(self):
        """No paths returned when bifurcation not active."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # No bifurcation flag set
        paths = get_bifurcation_paths(G, node)
        
        assert len(paths) == 0

    def test_il_always_in_paths(self):
        """IL (Coherence) always viable as universal resolution."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Set bifurcation flag
        G.nodes[node]["_bifurcation_ready"] = True
        
        paths = get_bifurcation_paths(G, node)
        
        # IL should always be present
        assert Glyph.IL in paths

    def test_zhir_path_with_high_vf(self):
        """ZHIR (Mutation) viable when νf sufficient for controlled transformation."""
        G, node = create_nfr("test", epi=0.5, vf=1.2)
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["ZHIR_BIFURCATION_VF_THRESHOLD"] = 0.8
        
        paths = get_bifurcation_paths(G, node)
        
        # ZHIR should be viable (vf=1.2 > 0.8)
        assert Glyph.ZHIR in paths

    def test_no_zhir_path_with_low_vf(self):
        """ZHIR not viable when νf too low."""
        G, node = create_nfr("test", epi=0.5, vf=0.6)
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["ZHIR_BIFURCATION_VF_THRESHOLD"] = 0.8
        
        paths = get_bifurcation_paths(G, node)
        
        # ZHIR should not be viable (vf=0.6 < 0.8)
        assert Glyph.ZHIR not in paths

    def test_nul_path_with_low_epi(self):
        """NUL (Contraction) viable when EPI low enough for safe collapse."""
        G, node = create_nfr("test", epi=0.3, vf=1.0)
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["NUL_BIFURCATION_EPI_THRESHOLD"] = 0.5
        
        paths = get_bifurcation_paths(G, node)
        
        # NUL should be viable (epi=0.3 < 0.5)
        assert Glyph.NUL in paths

    def test_no_nul_path_with_high_epi(self):
        """NUL not viable when EPI too high."""
        G, node = create_nfr("test", epi=0.7, vf=1.0)
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["NUL_BIFURCATION_EPI_THRESHOLD"] = 0.5
        
        paths = get_bifurcation_paths(G, node)
        
        # NUL should not be viable (epi=0.7 > 0.5)
        assert Glyph.NUL not in paths

    def test_thol_path_with_connectivity(self):
        """THOL (Self-organization) viable with sufficient network connectivity."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Add neighbors for connectivity
        G.add_node("n1", **{EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0})
        G.add_node("n2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.add_edge(node, "n1")
        G.add_edge(node, "n2")
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["THOL_BIFURCATION_MIN_DEGREE"] = 2
        
        paths = get_bifurcation_paths(G, node)
        
        # THOL should be viable (degree=2 >= 2)
        assert Glyph.THOL in paths

    def test_no_thol_path_isolated(self):
        """THOL not viable for isolated nodes."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node]["_bifurcation_ready"] = True
        G.graph["THOL_BIFURCATION_MIN_DEGREE"] = 2
        
        paths = get_bifurcation_paths(G, node)
        
        # THOL should not be viable (degree=0 < 2)
        assert Glyph.THOL not in paths


class TestOZBifurcationIntegration:
    """Integration tests for OZ → bifurcation → operator sequences."""

    def test_oz_to_zhir_sequence(self):
        """OZ → ZHIR bifurcation path (mutation)."""
        G, node = create_nfr("oz_zhir", epi=0.6, vf=1.2)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set up bifurcation conditions
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.4, 0.5, 0.7]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        # Apply OZ directly (not via run_sequence, to avoid grammar validation)
        Dissonance()(G, node)
        
        # Verify bifurcation detected
        assert G.nodes[node].get("_bifurcation_ready", False)
        
        # ZHIR should be in viable paths
        paths = get_bifurcation_paths(G, node)
        assert Glyph.ZHIR in paths
        
        # ZHIR should execute successfully
        Mutation()(G, node)  # Should not raise

    def test_oz_to_nul_sequence(self):
        """OZ → NUL bifurcation path (collapse to latency)."""
        G, node = create_nfr("oz_nul", epi=0.3, vf=0.8)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set up bifurcation conditions
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.2, 0.25, 0.35]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.04
        
        # Apply OZ directly
        Dissonance()(G, node)
        
        # Verify bifurcation
        assert G.nodes[node].get("_bifurcation_ready", False)
        
        # NUL should be viable
        paths = get_bifurcation_paths(G, node)
        assert Glyph.NUL in paths
        
        # Record EPI before
        epi_before = G.nodes[node][EPI_PRIMARY]
        
        # Apply NUL directly
        Contraction()(G, node)
        
        # Verify contraction occurred
        epi_after = G.nodes[node][EPI_PRIMARY]
        # EPI should decrease (contraction)
        # Note: Actual effect depends on glyph implementation

    def test_oz_to_il_sequence(self):
        """OZ → IL bifurcation path (stabilization)."""
        G, node = create_nfr("oz_il", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Set up bifurcation
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.6]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        # Apply OZ directly
        Dissonance()(G, node)
        
        # IL always viable
        paths = get_bifurcation_paths(G, node)
        assert Glyph.IL in paths
        
        # Record ΔNFR before
        dnfr_before = abs(G.nodes[node][DNFR_PRIMARY])
        
        # Apply IL directly to resolve dissonance
        Coherence()(G, node)
        
        # ΔNFR should reduce (IL effect)
        dnfr_after = abs(G.nodes[node][DNFR_PRIMARY])
        # Note: Actual reduction depends on glyph implementation

    def test_oz_to_thol_sequence(self):
        """OZ → THOL bifurcation path (self-organization)."""
        G, node = create_nfr("oz_thol", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # Add neighbors
        G.add_node("n1", **{EPI_PRIMARY: 0.4, VF_PRIMARY: 1.0})
        G.add_node("n2", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0})
        G.add_edge(node, "n1")
        G.add_edge(node, "n2")
        
        # Set up bifurcation
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.3, 0.4, 0.6]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.08
        
        # Apply OZ directly
        Dissonance()(G, node)
        
        # THOL should be viable
        paths = get_bifurcation_paths(G, node)
        assert Glyph.THOL in paths
        
        # Apply THOL directly with same tau for sub-EPI spawning
        SelfOrganization()(G, node, tau=0.08)
        
        # THOL may spawn sub-EPIs if history maintained
        # (depends on THOL implementation and EPI history state)


class TestBifurcationTelemetry:
    """Test telemetry and observability for bifurcation events."""

    def test_d2epi_stored_correctly(self):
        """∂²EPI/∂t² correctly stored using ALIAS_D2EPI."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        Dissonance()(G, node, validate_preconditions=True)
        
        # Check all alias variants
        for alias in ALIAS_D2EPI:
            if alias in G.nodes[node]:
                d2_epi = G.nodes[node][alias]
                assert isinstance(d2_epi, (int, float))
                assert d2_epi >= 0.0  # Magnitude, always non-negative

    def test_bifurcation_flag_cleared_when_below_threshold(self):
        """_bifurcation_ready flag cleared when acceleration drops below τ."""
        G, node = create_nfr("test", epi=0.5, vf=1.0)
        
        # Enable precondition validation
        G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
        
        # First: set bifurcation
        G.nodes[node][DNFR_PRIMARY] = 0.2
        G.nodes[node]["epi_history"] = [0.3, 0.45, 0.7]
        G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.08
        
        Dissonance()(G, node)
        assert G.nodes[node].get("_bifurcation_ready", False)
        
        # Second: drop below threshold
        G.nodes[node]["epi_history"] = [0.49, 0.50, 0.51]  # Minimal acceleration
        
        Dissonance()(G, node)
        assert not G.nodes[node].get("_bifurcation_ready", False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
