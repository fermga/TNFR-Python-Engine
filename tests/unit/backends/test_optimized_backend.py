"""Tests for optimized NumPy backend."""

import pytest
import networkx as nx

from tnfr.backends import get_backend, available_backends


class TestOptimizedBackendAvailability:
    """Test optimized backend registration."""
    
    def test_optimized_backend_registered(self):
        """Optimized NumPy backend should be registered."""
        backends = available_backends()
        assert "optimized_numpy" in backends or "optimized" in backends
    
    def test_can_get_optimized_backend(self):
        """Should be able to get optimized backend."""
        backend = get_backend("optimized_numpy")
        assert backend.name == "optimized_numpy"
    
    def test_optimized_alias_works(self):
        """'optimized' alias should work."""
        backend = get_backend("optimized")
        assert backend.name == "optimized_numpy"


class TestOptimizedBackendInterface:
    """Test optimized backend implements required interface."""
    
    def test_has_name_property(self):
        """Backend has name property."""
        backend = get_backend("optimized_numpy")
        assert backend.name == "optimized_numpy"
    
    def test_has_supports_gpu(self):
        """Backend has supports_gpu property."""
        backend = get_backend("optimized_numpy")
        assert isinstance(backend.supports_gpu, bool)
    
    def test_has_supports_jit(self):
        """Backend has supports_jit property."""
        backend = get_backend("optimized_numpy")
        assert isinstance(backend.supports_jit, bool)
    
    def test_has_compute_delta_nfr(self):
        """Backend implements compute_delta_nfr."""
        backend = get_backend("optimized_numpy")
        assert hasattr(backend, "compute_delta_nfr")
        assert callable(backend.compute_delta_nfr)
    
    def test_has_compute_si(self):
        """Backend implements compute_si."""
        backend = get_backend("optimized_numpy")
        assert hasattr(backend, "compute_si")
        assert callable(backend.compute_si)


class TestOptimizedBackendComputation:
    """Test optimized backend produces correct results."""
    
    def test_computes_delta_nfr_small_graph(self):
        """Optimized backend computes ΔNFR for small graphs."""
        G = nx.erdos_renyi_graph(30, 0.3, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5
        
        backend = get_backend("optimized_numpy")
        backend.compute_delta_nfr(G)
        
        assert "ΔNFR" in G.nodes[0]
    
    def test_computes_delta_nfr_large_graph(self):
        """Optimized backend computes ΔNFR for large graphs."""
        G = nx.erdos_renyi_graph(200, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5
        
        backend = get_backend("optimized_numpy")
        backend.compute_delta_nfr(G)
        
        assert "ΔNFR" in G.nodes[0]
    
    def test_computes_si(self):
        """Optimized backend computes Si."""
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 0.8
            G.nodes[node]["delta_nfr"] = 0.1
        
        backend = get_backend("optimized_numpy")
        si_values = backend.compute_si(G, inplace=False)
        
        assert len(si_values) == G.number_of_nodes()
        assert all(0.0 <= v <= 1.0 for v in si_values.values())
    
    def test_delta_nfr_matches_standard_backend(self):
        """Optimized backend produces same results as standard."""
        G = nx.erdos_renyi_graph(100, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5
        
        # Compute with standard backend
        backend_std = get_backend("numpy")
        backend_std.compute_delta_nfr(G)
        dnfr_std = {n: G.nodes[n]["ΔNFR"] for n in G.nodes()}
        
        # Reset and compute with optimized backend
        for node in G.nodes():
            G.nodes[node]["ΔNFR"] = 0.0
        
        backend_opt = get_backend("optimized_numpy")
        backend_opt.compute_delta_nfr(G)
        dnfr_opt = {n: G.nodes[n]["ΔNFR"] for n in G.nodes()}
        
        # Results should match within numerical tolerance
        for node in G.nodes():
            assert abs(dnfr_std[node] - dnfr_opt[node]) < 1e-10
    
    def test_si_matches_standard_backend(self):
        """Optimized Si matches standard backend."""
        G = nx.erdos_renyi_graph(100, 0.2, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 0.8
            G.nodes[node]["delta_nfr"] = 0.2
        
        backend_std = get_backend("numpy")
        si_std = backend_std.compute_si(G, inplace=False)
        
        backend_opt = get_backend("optimized_numpy")
        si_opt = backend_opt.compute_si(G, inplace=False)
        
        for node in G.nodes():
            assert abs(si_std[node] - si_opt[node]) < 1e-10


class TestOptimizedBackendProfiling:
    """Test profiling markers in optimized backend."""
    
    def test_dnfr_profile_has_optimization_marker(self):
        """Profile dict contains optimization marker."""
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["epi"] = 0.5
        
        backend = get_backend("optimized_numpy")
        profile = {}
        backend.compute_delta_nfr(G, profile=profile)
        
        assert "dnfr_optimization" in profile
        assert profile["dnfr_optimization"] in [
            "standard_small_graph",
            "enhanced_vectorization",
        ]
    
    def test_si_profile_has_optimization_marker(self):
        """Profile dict contains Si optimization marker."""
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        for node in G.nodes():
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["nu_f"] = 0.8
            G.nodes[node]["delta_nfr"] = 0.1
        
        backend = get_backend("optimized_numpy")
        profile = {}
        backend.compute_si(G, inplace=True, profile=profile)
        
        assert "si_optimization" in profile


class TestOptimizedBackendCaching:
    """Test workspace caching in optimized backend."""
    
    def test_has_clear_cache_method(self):
        """Backend has clear_cache method."""
        backend = get_backend("optimized_numpy")
        assert hasattr(backend, "clear_cache")
        assert callable(backend.clear_cache)
    
    def test_clear_cache_runs(self):
        """clear_cache can be called without error."""
        backend = get_backend("optimized_numpy")
        backend.clear_cache()  # Should not raise
