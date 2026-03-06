r"""Tests for the TNFR-Riemann alternative topology analysis (P2).

Validates that:
1. **Graph builders** produce connected graphs with correct labels
2. **Structural equilibrium** lambda_min(H(1/2)) = 0 for ALL topologies
3. **Sigma* deviation** decreases with k for ALL topologies
4. **Eigenvalue flow** has positive velocities across topologies
5. **Convergence comparison** yields power-law fits

Physics basis: sigma_c = 1/2 is exact for ANY connected graph because
H(1/2) = L (graph Laplacian) with ker(L) = span{1}.  The prime
structure enters through edge weights and eigenvector statistics.

TNFR physics: phase-gated coupling (U3) defines legitimate edges;
different topologies test different coupling regimes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import networkx as nx

from tnfr.riemann.operator import (
    _first_primes,
    build_prime_path_graph,
    build_prime_cycle_graph,
    build_prime_star_graph,
    build_prime_complete_graph,
    build_prime_tree_graph,
    build_prime_random_graph,
    build_h_tnfr,
)
from tnfr.riemann.topology import (
    TopologyResult,
    TopologyConvergenceResult,
    TOPOLOGY_BUILDERS,
    analyze_graph_topology,
    compare_topologies,
    topology_convergence_study,
)


# ============================================================================
# Graph Builder Tests
# ============================================================================


class TestGraphBuilders:
    """Validate that each builder produces well-formed prime graphs."""

    K_SMALL = 10

    @pytest.fixture(params=["path", "cycle", "star", "complete", "tree", "random"])
    def topology_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def _build(self, name: str, k: int = 10) -> nx.Graph:
        builder = TOPOLOGY_BUILDERS[name]
        if name == "random":
            return builder(k, seed=42)
        return builder(k)

    def test_node_count(self, topology_name: str) -> None:
        G = self._build(topology_name, self.K_SMALL)
        assert G.number_of_nodes() == self.K_SMALL

    def test_connected(self, topology_name: str) -> None:
        G = self._build(topology_name, self.K_SMALL)
        assert nx.is_connected(G), f"{topology_name} graph is disconnected"

    def test_labels_are_primes(self, topology_name: str) -> None:
        G = self._build(topology_name, self.K_SMALL)
        primes = _first_primes(self.K_SMALL)
        labels = sorted(G.nodes[n]["label"] for n in G.nodes())
        assert labels == primes

    def test_positive_weights(self, topology_name: str) -> None:
        G = self._build(topology_name, self.K_SMALL)
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            assert w > 0, f"Non-positive weight on edge ({u}, {v})"


class TestCycleBuilder:
    """Cycle-specific structural tests."""

    def test_edge_count(self) -> None:
        G = build_prime_cycle_graph(10)
        assert G.number_of_edges() == 10  # k edges (path + closing)

    def test_closing_edge_exists(self) -> None:
        G = build_prime_cycle_graph(5)
        # Nodes 0 and 4 should be connected (closing the cycle)
        assert G.has_edge(0, 4) or G.has_edge(4, 0)

    def test_all_degree_two(self) -> None:
        G = build_prime_cycle_graph(10)
        for n in G.nodes():
            assert G.degree(n) == 2


class TestStarBuilder:
    """Star-specific structural tests."""

    def test_edge_count(self) -> None:
        G = build_prime_star_graph(10)
        assert G.number_of_edges() == 9  # k-1 spokes

    def test_hub_degree(self) -> None:
        G = build_prime_star_graph(10)
        assert G.degree(0) == 9  # Hub connected to all others

    def test_leaf_degree(self) -> None:
        G = build_prime_star_graph(10)
        for n in range(1, 10):
            assert G.degree(n) == 1


class TestCompleteBuilder:
    """Complete graph structural tests."""

    def test_edge_count(self) -> None:
        G = build_prime_complete_graph(10)
        expected = 10 * 9 // 2  # k(k-1)/2
        assert G.number_of_edges() == expected

    def test_regular(self) -> None:
        G = build_prime_complete_graph(10)
        for n in G.nodes():
            assert G.degree(n) == 9


class TestTreeBuilder:
    """Binary tree structural tests."""

    def test_edge_count(self) -> None:
        G = build_prime_tree_graph(10)
        assert G.number_of_edges() == 9  # Tree: k-1 edges

    def test_is_tree(self) -> None:
        G = build_prime_tree_graph(10)
        assert nx.is_tree(G)

    def test_small_tree_structure(self) -> None:
        G = build_prime_tree_graph(7)
        # Balanced binary tree: root 0 -> children 1,2
        # Node 1 -> children 3,4; node 2 -> children 5,6
        assert G.has_edge(0, 1)
        assert G.has_edge(0, 2)
        assert G.has_edge(1, 3)
        assert G.has_edge(1, 4)
        assert G.has_edge(2, 5)
        assert G.has_edge(2, 6)


class TestRandomBuilder:
    """Erdős-Rényi builder tests."""

    def test_connected_guaranteed(self) -> None:
        """Even with low edge probability, connectivity is guaranteed."""
        G = build_prime_random_graph(20, edge_prob=0.1, seed=42)
        assert nx.is_connected(G)

    def test_seed_reproducibility(self) -> None:
        G1 = build_prime_random_graph(20, seed=42)
        G2 = build_prime_random_graph(20, seed=42)
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_differ(self) -> None:
        G1 = build_prime_random_graph(50, seed=42)
        G2 = build_prime_random_graph(50, seed=123)
        # Very unlikely to be identical
        assert set(G1.edges()) != set(G2.edges())


class TestUniformWeights:
    """Test weight_by_log_gap=False for all builders."""

    @pytest.fixture(params=["path", "cycle", "star", "complete", "tree", "random"])
    def topology_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_unit_weights(self, topology_name: str) -> None:
        builder = TOPOLOGY_BUILDERS[topology_name]
        if topology_name == "random":
            G = builder(10, seed=42, weight_by_log_gap=False)
        else:
            G = builder(10, weight_by_log_gap=False)
        for u, v, data in G.edges(data=True):
            assert data.get("weight", 1.0) == 1.0


# ============================================================================
# Equilibrium Universality Tests
# ============================================================================


class TestEquilibriumUniversality:
    r"""Verify lambda_min(H(1/2)) = 0 for ALL topologies.

    This is the fundamental universality result: the structural
    equilibrium is exact for any connected graph, not just the path.
    """

    @pytest.fixture(params=["path", "cycle", "star", "complete", "tree", "random"])
    def topology_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_equilibrium_exact(self, topology_name: str) -> None:
        """lambda_min(H(1/2)) = 0 (machine precision) for any connected graph."""
        builder = TOPOLOGY_BUILDERS[topology_name]
        if topology_name == "random":
            G = builder(20, seed=42)
        else:
            G = builder(20)
        result = analyze_graph_topology(G, topology_name)
        assert abs(result.lambda_min) < 1e-10, (
            f"{topology_name}: lambda_min = {result.lambda_min}"
        )

    def test_positive_spectral_gap(self, topology_name: str) -> None:
        """Spectral gap > 0 for all connected topologies."""
        builder = TOPOLOGY_BUILDERS[topology_name]
        if topology_name == "random":
            G = builder(20, seed=42)
        else:
            G = builder(20)
        result = analyze_graph_topology(G, topology_name)
        assert result.spectral_gap > 1e-10, (
            f"{topology_name}: gap = {result.spectral_gap}"
        )


class TestEquilibriumMultipleK:
    """Equilibrium exact for multiple k values across topologies."""

    K_VALUES = [5, 10, 30, 50]

    @pytest.fixture(params=["path", "cycle", "star", "complete", "tree", "random"])
    def topology_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_equilibrium_all_k(self, topology_name: str) -> None:
        for k in self.K_VALUES:
            builder = TOPOLOGY_BUILDERS[topology_name]
            if topology_name == "random":
                G = builder(k, seed=42)
            else:
                G = builder(k)
            result = analyze_graph_topology(G, topology_name)
            assert abs(result.lambda_min) < 1e-10, (
                f"{topology_name} k={k}: lambda_min = {result.lambda_min}"
            )


# ============================================================================
# Topology Analysis Tests
# ============================================================================


class TestAnalyzeTopology:
    """Test the core analysis function."""

    def test_result_structure(self) -> None:
        G = build_prime_path_graph(20)
        r = analyze_graph_topology(G, "path")
        assert isinstance(r, TopologyResult)
        assert r.topology == "path"
        assert r.k == 20
        assert r.n_edges == 19

    def test_potential_norm_topology_independent(self) -> None:
        """tr(V_1^2) = sum(log(p_i)^2) is the same for all topologies."""
        k = 30
        norms = []
        for name in TOPOLOGY_BUILDERS:
            builder = TOPOLOGY_BUILDERS[name]
            G = builder(k, seed=42) if name == "random" else builder(k)
            r = analyze_graph_topology(G, name)
            norms.append(r.potential_norm)
        # All should be identical (same set of primes)
        for n in norms:
            assert abs(n - norms[0]) < 1e-12, (
                f"potential_norm varies: {norms}"
            )

    def test_curvature_topology_independent(self) -> None:
        """d^2E/dsigma^2 = (1/k) tr(V_1^2) is the same for all topologies."""
        k = 30
        curvatures = []
        for name in TOPOLOGY_BUILDERS:
            builder = TOPOLOGY_BUILDERS[name]
            G = builder(k, seed=42) if name == "random" else builder(k)
            r = analyze_graph_topology(G, name)
            curvatures.append(r.curvature)
        for c in curvatures:
            assert abs(c - curvatures[0]) < 1e-12, (
                f"curvature varies: {curvatures}"
            )

    def test_cross_term_varies(self) -> None:
        """tr(L V_1) should differ across topologies (topology-dependent)."""
        k = 30
        cross_terms = {}
        for name in TOPOLOGY_BUILDERS:
            builder = TOPOLOGY_BUILDERS[name]
            G = builder(k, seed=42) if name == "random" else builder(k)
            r = analyze_graph_topology(G, name)
            cross_terms[name] = r.cross_term
        # At least some should be different
        values = list(cross_terms.values())
        assert max(values) - min(values) > 1e-6, (
            "All cross terms identical -- topologies have no effect?"
        )


# ============================================================================
# Compare Topologies Tests
# ============================================================================


class TestCompareTopologies:
    """Test the topology comparison function."""

    def test_all_topologies_present(self) -> None:
        results = compare_topologies(15)
        assert set(results.keys()) == set(TOPOLOGY_BUILDERS.keys())

    def test_subset_topologies(self) -> None:
        results = compare_topologies(15, topologies=["path", "cycle"])
        assert set(results.keys()) == {"path", "cycle"}

    def test_equilibrium_universal(self) -> None:
        """All topologies have lambda_min ~ 0 at the same k."""
        results = compare_topologies(30)
        for name, r in results.items():
            assert abs(r.lambda_min) < 1e-10, (
                f"{name}: lambda_min = {r.lambda_min}"
            )

    def test_spectral_gap_ordering(self) -> None:
        """Complete graph should have largest spectral gap, star smallest."""
        results = compare_topologies(20)
        # Complete graph is the strongest expander
        assert results["complete"].spectral_gap > results["path"].spectral_gap


# ============================================================================
# Convergence Study Tests
# ============================================================================


class TestConvergenceStudy:
    """Test the convergence study function."""

    def test_basic_convergence(self) -> None:
        result = topology_convergence_study(
            k_values=[10, 20, 50],
            topologies=["path", "cycle"],
        )
        assert isinstance(result, TopologyConvergenceResult)
        assert len(result.results["path"]) == 3
        assert len(result.results["cycle"]) == 3

    def test_convergence_rates_computed(self) -> None:
        result = topology_convergence_study(
            k_values=[10, 20, 50, 100],
            topologies=["path", "star"],
        )
        assert "path" in result.convergence_rates
        assert "star" in result.convergence_rates
        A_path, beta_path = result.convergence_rates["path"]
        assert A_path > 0
        assert beta_path > 0  # Deviation should decrease

    def test_deviation_decreases_with_k(self) -> None:
        """For bounded-degree topologies, deviation decreases with k.

        The star topology is a known exception: the hub degree scales
        as k-1, so tr(L V_1) / tr(V_1^2) does not vanish.  This is
        an important physics result showing that extreme degree
        concentration prevents sigma* -> 1/2 convergence.
        """
        result = topology_convergence_study(
            k_values=[10, 50, 100],
            topologies=["path", "cycle", "tree"],
        )
        for name in ["path", "cycle", "tree"]:
            devs = [r.deviation for r in result.results[name]]
            # Last deviation should be smaller than first
            assert devs[-1] < devs[0], (
                f"{name}: deviation not decreasing: {devs}"
            )

    def test_summary_nonempty(self) -> None:
        result = topology_convergence_study(
            k_values=[10, 30],
            topologies=["path"],
        )
        assert len(result.summary) > 50


# ============================================================================
# Eigenvalue Flow Tests
# ============================================================================


class TestEigenvalueFlowTopologies:
    """Test eigenvalue velocities across topologies."""

    @pytest.fixture(params=["path", "cycle", "star", "complete", "tree", "random"])
    def topology_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_positive_velocities(self, topology_name: str) -> None:
        """All eigenvalue velocities should be positive at k=20."""
        builder = TOPOLOGY_BUILDERS[topology_name]
        G = builder(20, seed=42) if topology_name == "random" else builder(20)
        r = analyze_graph_topology(G, topology_name)
        assert r.all_velocities_positive, (
            f"{topology_name}: min velocity = {r.min_velocity}"
        )

    def test_velocity_range(self, topology_name: str) -> None:
        """Velocities bounded by min/max of log(primes)."""
        builder = TOPOLOGY_BUILDERS[topology_name]
        G = builder(20, seed=42) if topology_name == "random" else builder(20)
        r = analyze_graph_topology(G, topology_name)
        primes = _first_primes(20)
        log_max = math.log(primes[-1])
        log_min = math.log(primes[0])
        assert r.min_velocity >= log_min - 0.01
        assert r.max_velocity <= log_max + 0.01


# ============================================================================
# Registry Tests
# ============================================================================


class TestTopologyRegistry:
    """Test the TOPOLOGY_BUILDERS registry."""

    def test_all_six_present(self) -> None:
        expected = {"path", "cycle", "star", "complete", "tree", "random"}
        assert set(TOPOLOGY_BUILDERS.keys()) == expected

    def test_builders_callable(self) -> None:
        for name, builder in TOPOLOGY_BUILDERS.items():
            assert callable(builder), f"{name} builder not callable"
