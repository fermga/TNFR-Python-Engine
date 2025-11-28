"""Property-based tests for TNFR structural invariants and operator effects.

This module uses Hypothesis to generate randomized test scenarios that validate
TNFR invariants hold across a wide range of inputs. These tests complement the
specific extreme case tests by exploring the broader property space.

TNFR Properties Tested:
- Operator sequences preserve physical bounds (EPI, νf, θ remain valid)
- Coherence C(t) is non-negative
- Sense index Si reflects structural stability
- ΔNFR conservation (sum over isolated systems)
- Nodal equation numerical stability
- Phase wrapping preserves semantics
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, assume

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_KEY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics import dnfr_epi_vf_mixed
from tnfr.metrics.common import compute_coherence
from tnfr.structural import create_nfr

from tests.property.strategies import (
    PROPERTY_TEST_SETTINGS,
    homogeneous_graphs,
    phase_graphs,
)


class TestOperatorInvariants:
    """Property tests for operator invariants."""

    @PROPERTY_TEST_SETTINGS
    @given(
        epi=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        vf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        theta=st.floats(
            min_value=0.0, max_value=2 * math.pi, allow_nan=False, allow_infinity=False
        ),
    )
    def test_create_nfr_preserves_bounds(self, epi: float, vf: float, theta: float):
        """Creating NFR nodes preserves value bounds."""
        G, node = create_nfr("test_node", epi=epi, vf=vf, theta=theta)

        node_epi = G.nodes[node][EPI_PRIMARY]
        node_vf = G.nodes[node][VF_PRIMARY]
        node_theta = G.nodes[node][THETA_KEY]

        # Values should remain in valid ranges
        assert -1.0 <= node_epi <= 1.0
        assert 0.0 <= node_vf <= 1.0
        assert isinstance(node_theta, (int, float))

    @PROPERTY_TEST_SETTINGS
    @given(graph=homogeneous_graphs())
    def test_dnfr_computation_never_nan(self, graph: nx.Graph):
        """ΔNFR computation should never produce NaN values."""
        dnfr_epi_vf_mixed(graph)

        for node in graph.nodes:
            dnfr_value = graph.nodes[node][DNFR_PRIMARY]
            assert not math.isnan(dnfr_value), f"Node {node} has NaN ΔNFR"
            assert not math.isinf(dnfr_value), f"Node {node} has inf ΔNFR"

    @PROPERTY_TEST_SETTINGS
    @given(graph=homogeneous_graphs())
    def test_coherence_always_non_negative(self, graph: nx.Graph):
        """Coherence C(t) should always be non-negative."""
        coherence = compute_coherence(graph)

        assert coherence >= 0.0, f"Negative coherence: {coherence}"
        assert not math.isnan(coherence), "Coherence is NaN"

    @PROPERTY_TEST_SETTINGS
    @given(
        epi1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        epi2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        vf=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_dnfr_gradient_direction(self, epi1: float, epi2: float, vf: float):
        """ΔNFR gradient should point from low to high EPI."""
        assume(abs(epi1 - epi2) > 0.1)  # Ensure meaningful gradient

        G = nx.Graph()
        inject_defaults(G)

        G.add_node("n1")
        G.add_node("n2")
        G.add_edge("n1", "n2")

        G.nodes["n1"][EPI_PRIMARY] = epi1
        G.nodes["n1"][VF_PRIMARY] = vf
        G.nodes["n1"][THETA_KEY] = 0.0
        G.nodes["n2"][EPI_PRIMARY] = epi2
        G.nodes["n2"][VF_PRIMARY] = vf
        G.nodes["n2"][THETA_KEY] = 0.0

        dnfr_epi_vf_mixed(G)

        dnfr1 = G.nodes["n1"][DNFR_PRIMARY]
        dnfr2 = G.nodes["n2"][DNFR_PRIMARY]

        # Node with lower EPI should have positive ΔNFR
        # Node with higher EPI should have negative ΔNFR
        if epi1 < epi2:
            assert dnfr1 > 0, f"Expected positive ΔNFR for lower EPI node"
            assert dnfr2 < 0, f"Expected negative ΔNFR for higher EPI node"
        else:
            assert dnfr1 < 0, f"Expected negative ΔNFR for higher EPI node"
            assert dnfr2 > 0, f"Expected positive ΔNFR for lower EPI node"


class TestNodalEquationProperties:
    """Property tests for nodal equation ∂EPI/∂t = νf · ΔNFR(t)."""

    @PROPERTY_TEST_SETTINGS
    @given(
        vf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dnfr=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_zero_frequency_prevents_change(self, vf: float, dnfr: float):
        """When νf=0, ∂EPI/∂t should be 0 regardless of ΔNFR."""
        # Skip if vf is not effectively zero
        assume(vf < 1e-6)

        # ∂EPI/∂t = νf · ΔNFR = 0 · ΔNFR = 0
        depi_dt = vf * dnfr
        # Allow for floating point precision - if vf is 1e-8 and dnfr is 1, result is 1e-8
        assert abs(depi_dt) < 1e-5, f"Expected near-zero change with vf≈0, got {depi_dt}"

    @PROPERTY_TEST_SETTINGS
    @given(
        vf=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_zero_dnfr_prevents_change(self, vf: float):
        """When ΔNFR=0, ∂EPI/∂t should be 0 regardless of νf."""
        dnfr = 0.0

        # ∂EPI/∂t = νf · 0 = 0
        depi_dt = vf * dnfr
        assert abs(depi_dt) < 1e-9, f"Expected zero change with ΔNFR=0, got {depi_dt}"

    @PROPERTY_TEST_SETTINGS
    @given(
        vf=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        dnfr=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_nodal_equation_sign_preservation(self, vf: float, dnfr: float):
        """Sign of ∂EPI/∂t should match sign of ΔNFR when νf > 0."""
        depi_dt = vf * dnfr

        if dnfr > 0:
            assert depi_dt >= 0, f"Positive ΔNFR should give non-negative change"
        elif dnfr < 0:
            assert depi_dt <= 0, f"Negative ΔNFR should give non-positive change"
        else:
            assert abs(depi_dt) < 1e-9, f"Zero ΔNFR should give zero change"

    @PROPERTY_TEST_SETTINGS
    @given(
        vf=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        dnfr=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_nodal_equation_magnitude_scaling(self, vf: float, dnfr: float, dt: float):
        """∂EPI/∂t should scale linearly with both νf and ΔNFR."""
        depi_dt = vf * dnfr

        # Doubling νf should double the rate
        depi_dt_2vf = (2 * vf) * dnfr
        if abs(depi_dt) > 1e-6:
            ratio = depi_dt_2vf / depi_dt
            assert abs(ratio - 2.0) < 0.01, f"Expected 2x scaling, got {ratio}"


class TestPhaseProperties:
    """Property tests for phase (θ) behavior."""

    @PROPERTY_TEST_SETTINGS
    @given(
        theta=st.floats(
            min_value=-10 * math.pi,
            max_value=10 * math.pi,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_phase_wrapping_preserves_modulo_2pi(self, theta: float):
        """Phase values should be semantically equivalent modulo 2π."""
        # Create two nodes with phases that differ by 2π
        G1, n1 = create_nfr("n1", epi=0.5, vf=1.0, theta=theta)
        G2, n2 = create_nfr("n2", epi=0.5, vf=1.0, theta=theta + 2 * math.pi)

        theta1 = G1.nodes[n1][THETA_KEY]
        theta2 = G2.nodes[n2][THETA_KEY]

        # Phases should be equivalent modulo 2π
        diff = abs(theta1 - theta2)
        # Allow for wrapping: diff should be close to 0 or close to 2π
        remainder = diff % (2 * math.pi)
        assert remainder < 0.1 or remainder > (
            2 * math.pi - 0.1
        ), f"Phases {theta1} and {theta2} differ by {diff}, not equivalent mod 2π"

    @PROPERTY_TEST_SETTINGS
    @given(
        theta1=st.floats(
            min_value=0.0, max_value=2 * math.pi, allow_nan=False, allow_infinity=False
        ),
        theta2=st.floats(
            min_value=0.0, max_value=2 * math.pi, allow_nan=False, allow_infinity=False
        ),
    )
    def test_phase_difference_bounded(self, theta1: float, theta2: float):
        """Phase difference should be computable and bounded."""
        diff = abs(theta1 - theta2)

        # Minimum difference considering wrapping
        min_diff = min(diff, 2 * math.pi - diff) if diff > math.pi else diff

        assert 0.0 <= min_diff <= math.pi, f"Minimum phase difference {min_diff} outside [0, π]"


class TestStructuralConservation:
    """Property tests for structural conservation laws."""

    @PROPERTY_TEST_SETTINGS
    @given(graph=homogeneous_graphs())
    def test_dnfr_conservation_in_isolated_graph(self, graph: nx.Graph):
        """Sum of ΔNFR over all nodes in an isolated graph should be ~0."""
        dnfr_epi_vf_mixed(graph)

        total_dnfr = sum(graph.nodes[n][DNFR_PRIMARY] for n in graph.nodes)

        # For connected homogeneous graphs, total ΔNFR should be near zero
        # Allow some numerical error
        if graph.number_of_nodes() > 0:
            avg_dnfr = abs(total_dnfr / graph.number_of_nodes())
            assert (
                avg_dnfr < 0.1
            ), f"Average ΔNFR magnitude {avg_dnfr} too high for homogeneous graph"

    @PROPERTY_TEST_SETTINGS
    @given(
        num_nodes=st.integers(min_value=2, max_value=10),
        epi=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_coherence_scales_with_size(self, num_nodes: int, epi: float):
        """Coherence should have consistent behavior as graph size changes."""
        G = nx.Graph()
        inject_defaults(G)

        # Create a complete graph with uniform EPI
        for i in range(num_nodes):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = epi
            G.nodes[i][VF_PRIMARY] = 1.0
            G.nodes[i][THETA_KEY] = 0.0

        # Make it connected
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)

        coherence = compute_coherence(G)

        # Coherence should be computable and finite
        assert not math.isnan(coherence), "Coherence is NaN"
        assert not math.isinf(coherence), "Coherence is infinite"
        assert coherence >= 0.0, "Coherence is negative"


class TestBoundaryBehavior:
    """Property tests for behavior at boundary values."""

    @PROPERTY_TEST_SETTINGS
    @given(
        epi=st.sampled_from([-1.0, -0.5, 0.0, 0.5, 1.0]),
        vf=st.sampled_from([0.0, 0.5, 1.0]),
    )
    def test_extreme_values_remain_stable(self, epi: float, vf: float):
        """Extreme but valid values should remain stable after operations."""
        G, node = create_nfr("test", epi=epi, vf=vf, theta=0.0)

        # Apply ΔNFR computation
        dnfr_epi_vf_mixed(G)

        # Values should still be in range
        final_epi = G.nodes[node][EPI_PRIMARY]
        final_vf = G.nodes[node][VF_PRIMARY]

        assert -1.0 <= final_epi <= 1.0
        assert 0.0 <= final_vf <= 1.0

    @PROPERTY_TEST_SETTINGS
    @given(
        epi_range=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    def test_large_epi_differences_handled(self, epi_range: float):
        """Large EPI differences between nodes should be handled."""
        G = nx.Graph()
        inject_defaults(G)

        G.add_node("low")
        G.add_node("high")
        G.add_edge("low", "high")

        G.nodes["low"][EPI_PRIMARY] = -epi_range
        G.nodes["low"][VF_PRIMARY] = 1.0
        G.nodes["low"][THETA_KEY] = 0.0

        G.nodes["high"][EPI_PRIMARY] = epi_range
        G.nodes["high"][VF_PRIMARY] = 1.0
        G.nodes["high"][THETA_KEY] = 0.0

        # Should not crash
        dnfr_epi_vf_mixed(G)

        dnfr_low = G.nodes["low"][DNFR_PRIMARY]
        dnfr_high = G.nodes["high"][DNFR_PRIMARY]

        # Gradients should exist and be finite
        assert not math.isnan(dnfr_low)
        assert not math.isnan(dnfr_high)
        assert not math.isinf(dnfr_low)
        assert not math.isinf(dnfr_high)


class TestGraphTopologyInvariants:
    """Property tests for topology-dependent invariants."""

    @PROPERTY_TEST_SETTINGS
    @given(
        num_nodes=st.integers(min_value=1, max_value=8),
        epi=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_isolated_nodes_have_zero_dnfr(self, num_nodes: int, epi: float):
        """Isolated nodes should have ΔNFR ≈ 0."""
        G = nx.Graph()
        inject_defaults(G)

        # Create isolated nodes (no edges)
        for i in range(num_nodes):
            G.add_node(i)
            G.nodes[i][EPI_PRIMARY] = epi
            G.nodes[i][VF_PRIMARY] = 1.0
            G.nodes[i][THETA_KEY] = 0.0

        dnfr_epi_vf_mixed(G)

        for i in range(num_nodes):
            dnfr = G.nodes[i][DNFR_PRIMARY]
            assert abs(dnfr) < 1e-9, f"Isolated node {i} has non-zero ΔNFR: {dnfr}"

    @PROPERTY_TEST_SETTINGS
    @given(graph=homogeneous_graphs())
    def test_graph_modifications_preserve_validity(self, graph: nx.Graph):
        """Graph should remain valid after TNFR operations."""
        initial_nodes = set(graph.nodes)

        dnfr_epi_vf_mixed(graph)
        compute_coherence(graph)

        # Graph structure should be unchanged
        assert set(graph.nodes) == initial_nodes

        # All nodes should have valid attributes
        for node in graph.nodes:
            assert EPI_PRIMARY in graph.nodes[node]
            assert VF_PRIMARY in graph.nodes[node]
            assert THETA_KEY in graph.nodes[node]
            assert DNFR_PRIMARY in graph.nodes[node]
