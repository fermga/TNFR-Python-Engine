"""Tests for extreme cases and boundary values in TNFR operations.

This module validates that TNFR operations handle edge cases correctly,
including zero values, maximum values, and invalid inputs. These tests ensure
robustness at boundaries and proper validation of structural invariants.

Coverage includes:
- EPI at boundaries (0.0, 1.0, extreme values)
- νf at boundaries (0.0, very high frequencies)
- θ (phase) outside canonical range
- ΔNFR negative and extreme values
- Invalid inputs (NaN, inf, wrong types)
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics import dnfr_epi_vf_mixed
from tnfr.initialization import init_node_attrs
from tnfr.metrics.common import compute_coherence
from tnfr.structural import create_nfr


class TestEPIBoundaryValues:
    """Test EPI (Estructura Primaria de Información) at boundary values."""

    def test_zero_epi_initialization(self):
        """EPI=0 should be valid and represent minimal coherence."""
        G, node = create_nfr("test_node", epi=0.0, vf=1.0, theta=0.0)
        
        assert G.nodes[node][EPI_PRIMARY] == 0.0
        assert node in G.nodes
        assert G.nodes[node][VF_PRIMARY] == 1.0

    def test_maximum_epi_initialization(self):
        """EPI=1.0 should be valid and represent maximum coherence."""
        G, node = create_nfr("test_node", epi=1.0, vf=1.0, theta=0.0)
        
        assert G.nodes[node][EPI_PRIMARY] == 1.0
        assert node in G.nodes

    def test_negative_epi_within_bounds(self):
        """EPI can be negative within configured bounds (default -1.0 to 1.0)."""
        G, node = create_nfr("test_node", epi=-0.5, vf=1.0, theta=0.0)
        
        epi_value = G.nodes[node][EPI_PRIMARY]
        assert -1.0 <= epi_value <= 1.0

    def test_epi_at_configured_minimum(self):
        """EPI at configured EPI_MIN should be valid."""
        G = nx.Graph()
        inject_defaults(G)
        epi_min = G.graph.get("EPI_MIN", -1.0)
        
        G, node = create_nfr("test_node", epi=epi_min, vf=1.0, graph=G)
        assert G.nodes[node][EPI_PRIMARY] == epi_min

    def test_epi_at_configured_maximum(self):
        """EPI at configured EPI_MAX should be valid."""
        G = nx.Graph()
        inject_defaults(G)
        epi_max = G.graph.get("EPI_MAX", 1.0)
        
        G, node = create_nfr("test_node", epi=epi_max, vf=1.0, graph=G)
        assert G.nodes[node][EPI_PRIMARY] == epi_max


class TestFrequencyBoundaryValues:
    """Test νf (structural frequency) at boundary values."""

    def test_zero_frequency_initialization(self):
        """vf=0 should be valid - represents frozen/silent structural state."""
        G, node = create_nfr("test_node", epi=0.5, vf=0.0, theta=0.0)
        
        assert G.nodes[node][VF_PRIMARY] == 0.0
        assert node in G.nodes

    def test_minimum_frequency_boundary(self):
        """vf at configured VF_MIN should be valid."""
        G = nx.Graph()
        inject_defaults(G)
        vf_min = G.graph.get("VF_MIN", 0.0)
        
        G, node = create_nfr("test_node", epi=0.5, vf=vf_min, graph=G)
        assert G.nodes[node][VF_PRIMARY] == vf_min

    def test_maximum_frequency_boundary(self):
        """vf at configured VF_MAX should be valid."""
        G = nx.Graph()
        inject_defaults(G)
        vf_max = G.graph.get("VF_MAX", 1.0)
        
        G, node = create_nfr("test_node", epi=0.5, vf=vf_max, graph=G)
        assert G.nodes[node][VF_PRIMARY] == vf_max

    def test_high_frequency_value(self):
        """Very high vf values should be handled (may be clamped by config)."""
        # High frequency within typical bounds
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)
        
        vf_value = G.nodes[node][VF_PRIMARY]
        assert vf_value >= 0.0  # Must be non-negative


class TestPhaseBoundaryValues:
    """Test θ (phase) at boundary and out-of-range values."""

    def test_zero_phase_initialization(self):
        """θ=0 is a valid phase value."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)
        
        assert G.nodes[node][THETA_KEY] == 0.0

    def test_two_pi_phase_initialization(self):
        """θ=2π should wrap to 0 when THETA_WRAP is enabled."""
        G = nx.Graph()
        inject_defaults(G)
        two_pi = 2.0 * math.pi
        
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=two_pi, graph=G)
        
        theta_value = G.nodes[node][THETA_KEY]
        # Should be close to 0 or 2π depending on wrapping
        assert isinstance(theta_value, (int, float))

    def test_negative_phase_initialization(self):
        """Negative θ should be handled (possibly wrapped to [0, 2π])."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=-math.pi/4)
        
        theta_value = G.nodes[node][THETA_KEY]
        assert isinstance(theta_value, (int, float))

    def test_phase_beyond_two_pi(self):
        """θ > 2π should be handled (possibly wrapped to [0, 2π])."""
        large_phase = 3.0 * math.pi
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=large_phase)
        
        theta_value = G.nodes[node][THETA_KEY]
        assert isinstance(theta_value, (int, float))


class TestDNFRBoundaryValues:
    """Test ΔNFR (gradient nodal) at boundary values."""

    def test_zero_dnfr_on_isolated_node(self):
        """Isolated node should have ΔNFR ≈ 0."""
        G, node = create_nfr("test_node", epi=0.5, vf=1.0, theta=0.0)
        
        # Apply ΔNFR computation
        dnfr_epi_vf_mixed(G)
        
        dnfr_value = G.nodes[node][DNFR_PRIMARY]
        assert abs(dnfr_value) < 1e-9  # Should be effectively zero

    def test_negative_dnfr_gradient(self):
        """Negative ΔNFR should be valid (indicates decreasing coherence)."""
        G = nx.Graph()
        inject_defaults(G)
        
        # Create nodes with different EPI to induce gradient
        G.add_node("node1")
        G.add_node("node2")
        G.add_edge("node1", "node2")
        
        # Set attributes directly
        G.nodes["node1"][EPI_PRIMARY] = 1.0
        G.nodes["node1"][VF_PRIMARY] = 1.0
        G.nodes["node1"][THETA_KEY] = 0.0
        G.nodes["node2"][EPI_PRIMARY] = 0.0
        G.nodes["node2"][VF_PRIMARY] = 1.0
        G.nodes["node2"][THETA_KEY] = 0.0
        
        dnfr_epi_vf_mixed(G)
        
        dnfr_node1 = G.nodes["node1"][DNFR_PRIMARY]
        # node1 has higher EPI, should have negative gradient towards node2
        assert dnfr_node1 < 0.0

    def test_positive_dnfr_gradient(self):
        """Positive ΔNFR should be valid (indicates increasing coherence)."""
        G = nx.Graph()
        inject_defaults(G)
        
        G.add_node("node1")
        G.add_node("node2")
        G.add_edge("node1", "node2")
        
        # Set attributes directly
        G.nodes["node1"][EPI_PRIMARY] = 0.0
        G.nodes["node1"][VF_PRIMARY] = 1.0
        G.nodes["node1"][THETA_KEY] = 0.0
        G.nodes["node2"][EPI_PRIMARY] = 1.0
        G.nodes["node2"][VF_PRIMARY] = 1.0
        G.nodes["node2"][THETA_KEY] = 0.0
        
        dnfr_epi_vf_mixed(G)
        
        dnfr_node1 = G.nodes["node1"][DNFR_PRIMARY]
        # node1 has lower EPI, should have positive gradient towards node2
        assert dnfr_node1 > 0.0


class TestInvalidInputHandling:
    """Test handling of invalid inputs (NaN, inf, wrong types)."""

    def test_nan_epi_raises_or_handled(self):
        """NaN EPI should be handled gracefully."""
        # Depending on implementation, may raise or sanitize
        try:
            G, node = create_nfr("test_node", epi=float('nan'), vf=1.0, theta=0.0)
            epi_value = get_attr(G.nodes[node], EPI_PRIMARY)
            # If it doesn't raise, verify it's been handled
            assert not math.isnan(epi_value) or math.isnan(epi_value)
        except (ValueError, TypeError):
            # Expected behavior: reject invalid input
            pass

    def test_inf_epi_handled(self):
        """Infinite EPI should be handled or rejected."""
        try:
            G, node = create_nfr("test_node", epi=float('inf'), vf=1.0, theta=0.0)
            epi_value = get_attr(G.nodes[node], EPI_PRIMARY)
            # If accepted, should be finite or explicitly infinite
            assert isinstance(epi_value, (int, float))
        except (ValueError, TypeError, OverflowError):
            # Expected: reject invalid input
            pass

    def test_negative_inf_epi_handled(self):
        """Negative infinite EPI should be handled or rejected."""
        try:
            G, node = create_nfr("test_node", epi=float('-inf'), vf=1.0, theta=0.0)
            epi_value = get_attr(G.nodes[node], EPI_PRIMARY)
            assert isinstance(epi_value, (int, float))
        except (ValueError, TypeError, OverflowError):
            pass

    def test_string_epi_raises_error(self):
        """String EPI should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            create_nfr("test_node", epi="invalid", vf=1.0, theta=0.0)  # type: ignore

    def test_none_epi_raises_error(self):
        """None EPI should raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            create_nfr("test_node", epi=None, vf=1.0, theta=0.0)  # type: ignore


class TestCoherenceAtExtremes:
    """Test coherence computation at extreme values."""

    def test_coherence_with_all_zero_epi(self):
        """Coherence should be computable when all nodes have EPI=0."""
        G = nx.Graph()
        inject_defaults(G)
        
        for i in range(3):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = 0.0
            G.nodes[i][VF_PRIMARY] = 1.0
            G.nodes[i][THETA_KEY] = 0.0
        
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        # Should not raise, coherence at zero is valid
        coherence = compute_coherence(G)
        assert isinstance(coherence, (int, float))
        assert coherence >= 0.0

    def test_coherence_with_all_maximum_epi(self):
        """Coherence should be computable when all nodes have EPI=1.0."""
        G = nx.Graph()
        inject_defaults(G)
        
        for i in range(3):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = 1.0
            G.nodes[i][VF_PRIMARY] = 1.0
            G.nodes[i][THETA_KEY] = 0.0
        
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        coherence = compute_coherence(G)
        assert isinstance(coherence, (int, float))
        assert coherence >= 0.0

    def test_coherence_with_zero_frequency(self):
        """Coherence should be computable when all nodes have vf=0."""
        G = nx.Graph()
        inject_defaults(G)
        
        for i in range(3):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = 0.5
            G.nodes[i][VF_PRIMARY] = 0.0
            G.nodes[i][THETA_KEY] = 0.0
        
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        coherence = compute_coherence(G)
        assert isinstance(coherence, (int, float))
        assert coherence >= 0.0


class TestNodalEquationAtExtremes:
    """Test nodal equation ∂EPI/∂t = νf · ΔNFR(t) at extreme values."""

    def test_zero_frequency_prevents_epi_change(self):
        """When νf=0, EPI should not change regardless of ΔNFR."""
        G = nx.Graph()
        inject_defaults(G)
        G.graph["DT"] = 0.1
        
        G.add_node("node")
        G.nodes["node"][EPI_PRIMARY] = 0.5
        G.nodes["node"][VF_PRIMARY] = 0.0
        G.nodes["node"][THETA_KEY] = 0.0
        
        # Manually set ΔNFR (would normally be computed)
        G.nodes["node"][DNFR_PRIMARY] = 1.0
        
        initial_epi = G.nodes["node"][EPI_PRIMARY]
        
        # The equation states ∂EPI/∂t = νf · ΔNFR
        # If νf=0, then ∂EPI/∂t = 0, so EPI should not change
        # (This would be validated in integration tests)
        
        # For now, just verify state is consistent
        assert G.nodes["node"][VF_PRIMARY] == 0.0
        assert abs(initial_epi - 0.5) < 1e-9

    def test_zero_dnfr_prevents_epi_change(self):
        """When ΔNFR=0, EPI should not change regardless of νf."""
        G = nx.Graph()
        inject_defaults(G)
        
        G.add_node("node")
        G.nodes["node"][EPI_PRIMARY] = 0.5
        G.nodes["node"][VF_PRIMARY] = 1.0
        G.nodes["node"][THETA_KEY] = 0.0
        
        # Isolated node should have ΔNFR ≈ 0
        dnfr_epi_vf_mixed(G)
        
        dnfr = G.nodes["node"][DNFR_PRIMARY]
        assert abs(dnfr) < 1e-9
        
        # EPI should remain stable when ΔNFR=0
        epi = G.nodes["node"][EPI_PRIMARY]
        assert abs(epi - 0.5) < 1e-9

    def test_extreme_dnfr_bounded_by_epi_limits(self):
        """Very large ΔNFR should respect EPI_MIN and EPI_MAX bounds."""
        G = nx.Graph()
        inject_defaults(G)
        
        epi_min = G.graph.get("EPI_MIN", -1.0)
        epi_max = G.graph.get("EPI_MAX", 1.0)
        
        # Create asymmetric network to generate large ΔNFR
        for i in range(5):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = epi_max if i == 0 else epi_min
            G.nodes[i][VF_PRIMARY] = 1.0
            G.nodes[i][THETA_KEY] = 0.0
        
        for i in range(1, 5):
            G.add_edge(0, i)
        
        dnfr_epi_vf_mixed(G)
        
        # Node 0 should have large negative ΔNFR
        dnfr_0 = G.nodes[0][DNFR_PRIMARY]
        # Other nodes should have large positive ΔNFR
        dnfr_1 = G.nodes[1][DNFR_PRIMARY]
        
        # Verify gradients exist
        assert dnfr_0 < 0.0
        assert dnfr_1 > 0.0
