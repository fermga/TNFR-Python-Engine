"""Performance tests for vectorized ΔNFR computation."""

import pytest
import networkx as nx
from time import perf_counter

from tnfr.backends import get_backend


class TestVectorizedPerformance:
    """Performance validation for vectorized backend."""

    def test_vectorized_faster_on_large_graphs(self):
        """Vectorized backend should not be significantly slower on large graphs.

        Note: Without Numba JIT, the two-pass algorithm may be slower than
        the standard backend. This test ensures it's not catastrophically slow.
        With Numba JIT compilation, expect 2-3x speedup.
        """
        # Create a moderately large graph
        G = nx.erdos_renyi_graph(300, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5

        # Set weights to enable gradients
        G.graph["DNFR_WEIGHTS"] = {
            "phase": 0.3,
            "epi": 0.3,
            "vf": 0.2,
            "topo": 0.2,
        }

        # Time standard backend
        backend_std = get_backend("numpy")
        t0 = perf_counter()
        backend_std.compute_delta_nfr(G)
        time_std = perf_counter() - t0

        # Reset
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.0

        # Time optimized backend
        backend_opt = get_backend("optimized_numpy")
        t0 = perf_counter()
        backend_opt.compute_delta_nfr(G)
        time_opt = perf_counter() - t0

        # The optimized backend should not be more than 5x slower
        # Without Numba: expected ~2x slower (two-pass overhead)
        # With Numba: expected 2-3x faster
        speedup = time_std / time_opt if time_opt > 0 else 1.0

        assert speedup >= 0.2, (
            f"Optimized backend catastrophically slow: {speedup:.2f}x "
            f"(std={time_std:.4f}s, opt={time_opt:.4f}s). "
            f"Without Numba, expect ~0.5x. With Numba, expect 2-3x."
        )

    def test_vectorized_profiling_markers(self):
        """Profiling should indicate vectorized path used."""
        G = nx.erdos_renyi_graph(200, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5

        backend = get_backend("optimized_numpy")
        profile = {}
        backend.compute_delta_nfr(G, profile=profile)

        # Should use vectorized fused path for large graphs
        assert profile.get("dnfr_optimization") == "vectorized_fused"

        # Should have timing metrics from vectorized implementation
        assert "dnfr_workspace_alloc" in profile
        assert "dnfr_fused_compute" in profile

    def test_small_graphs_use_standard_path(self):
        """Small graphs (<100 nodes) should use standard implementation."""
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5

        backend = get_backend("optimized_numpy")
        profile = {}
        backend.compute_delta_nfr(G, profile=profile)

        # Small graphs should use standard path
        assert profile.get("dnfr_optimization") == "standard_small_graph"

    def test_vectorized_handles_empty_graph(self):
        """Vectorized implementation handles empty graphs correctly."""
        G = nx.erdos_renyi_graph(200, 0.0, seed=42)  # No edges
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5

        backend = get_backend("optimized_numpy")
        backend.compute_delta_nfr(G)

        # All ΔNFR should be 0 (no edges means no gradients)
        for node in G.nodes():
            assert G.nodes[node]["ΔNFR"] == 0.0

    def test_vectorized_preserves_semantics_with_varying_weights(self):
        """Vectorized implementation respects different weight configurations."""
        G = nx.erdos_renyi_graph(150, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = float(node) * 0.01
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5 + float(node) * 0.001

        # Test with phase-only weights
        G.graph["DNFR_WEIGHTS"] = {"phase": 1.0}

        backend_std = get_backend("numpy")
        backend_std.compute_delta_nfr(G)
        dnfr_std = {n: G.nodes[n]["ΔNFR"] for n in G.nodes()}

        # Reset and compute with optimized
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.0

        backend_opt = get_backend("optimized_numpy")
        backend_opt.compute_delta_nfr(G)
        dnfr_opt = {n: G.nodes[n]["ΔNFR"] for n in G.nodes()}

        # Results should match (currently delegates to standard for fidelity)
        for node in G.nodes():
            assert abs(dnfr_std[node] - dnfr_opt[node]) < 1e-10
