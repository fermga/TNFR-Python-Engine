"""Tests for optimization features."""

import pytest
import networkx as nx


class TestOptimizations:
    """Test suite for advanced optimizations."""

    def test_adaptive_partitioning(self):
        """Test adaptive partition sizing based on network characteristics."""
        from tnfr.parallel import FractalPartitioner

        # Create networks with different characteristics
        # Dense network
        G_dense = nx.erdos_renyi_graph(100, 0.5)
        for node in G_dense.nodes():
            G_dense.nodes[node]["vf"] = 1.0
            G_dense.nodes[node]["phase"] = 0.0

        partitioner_adaptive = FractalPartitioner(adaptive=True, max_partition_size=None)
        partitions_dense = partitioner_adaptive.partition_network(G_dense)

        # Adaptive partitioning should create partitions
        assert len(partitions_dense) >= 1

        # Sparse network
        G_sparse = nx.erdos_renyi_graph(100, 0.05)
        for node in G_sparse.nodes():
            G_sparse.nodes[node]["vf"] = 1.0
            G_sparse.nodes[node]["phase"] = 0.0

        partitions_sparse = partitioner_adaptive.partition_network(G_sparse)

        # Should handle sparse networks too
        assert len(partitions_sparse) >= 1

    def test_spatial_indexing(self):
        """Test spatial indexing for O(n log n) neighbor finding."""
        from tnfr.parallel import FractalPartitioner

        G = nx.erdos_renyi_graph(200, 0.1)
        for node in G.nodes():
            G.nodes[node]["vf"] = float(node) / 10.0  # Vary frequencies
            G.nodes[node]["phase"] = float(node) / 50.0  # Vary phases

        # With spatial indexing
        partitioner_spatial = FractalPartitioner(use_spatial_index=True, max_partition_size=50)
        partitions_spatial = partitioner_spatial.partition_network(G)

        # Should create partitions
        assert len(partitions_spatial) > 0

        # Without spatial indexing
        partitioner_no_spatial = FractalPartitioner(use_spatial_index=False, max_partition_size=50)
        partitions_no_spatial = partitioner_no_spatial.partition_network(G)

        # Should also work without spatial indexing
        assert len(partitions_no_spatial) > 0

    def test_cache_aware_distribution(self):
        """Test cache-aware work distribution."""
        from tnfr.parallel import TNFRParallelEngine

        # Create engine with cache-aware distribution
        engine = TNFRParallelEngine(max_workers=4, cache_aware=True)

        # Create mock partitions
        G = nx.erdos_renyi_graph(100, 0.1)
        for node in G.nodes():
            G.nodes[node]["vf"] = 1.0
            G.nodes[node]["phase"] = 0.0

        from tnfr.parallel.partitioner import FractalPartitioner

        partitioner = FractalPartitioner(max_partition_size=20)
        partitions = partitioner.partition_network(G)

        # Test work distribution
        if len(partitions) > 0:
            work_chunks = engine._distribute_work_cache_aware(partitions, num_workers=4)

            # Should distribute work across workers
            assert len(work_chunks) == 4

            # All partitions should be assigned
            total_assigned = sum(len(chunk) for chunk in work_chunks)
            assert total_assigned == len(partitions)

    def test_gpu_engine_numpy_backend(self):
        """Test GPU engine with NumPy fallback."""
        from tnfr.parallel import TNFRGPUEngine

        engine = TNFRGPUEngine(backend="numpy")

        # Create small test graph
        G = nx.Graph([(0, 1), (1, 2), (2, 3)])
        for node in G.nodes():
            G.nodes[node]["epi"] = 0.5 + node * 0.1
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["phase"] = 0.0

        # Compute ΔNFR
        result = engine.compute_delta_nfr_from_graph(G)

        # Should return results for all nodes
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result.values())

    def test_gpu_engine_jax_backend(self):
        """Test GPU engine with JAX backend (if available)."""
        try:
            import jax

            HAS_JAX = True
        except ImportError:
            HAS_JAX = False

        if not HAS_JAX:
            pytest.skip("JAX not available")

        from tnfr.parallel import TNFRGPUEngine

        engine = TNFRGPUEngine(backend="jax")

        # Create test graph
        G = nx.Graph([(0, 1), (1, 2)])
        for node in G.nodes():
            G.nodes[node]["epi"] = 0.5
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["phase"] = 0.0

        # Should be able to compute
        result = engine.compute_delta_nfr_from_graph(G)
        assert len(result) == 3

    def test_distributed_engine_si_computation(self):
        """Test distributed Si computation (fallback to multiprocessing)."""
        from tnfr.parallel import TNFRDistributedEngine

        # Create distributed engine (will fallback without Ray/Dask)
        engine = TNFRDistributedEngine(backend="auto")

        # Create small test network
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        for node in G.nodes():
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["delta_nfr"] = 0.0

        # Compute Si using distributed engine
        result = engine.compute_si_distributed(G, chunk_size=2)

        # Should return results
        assert "si_values" in result
        assert "backend" in result
        assert len(result["si_values"]) == 4

    def test_distributed_simulate_network(self):
        """Test distributed network simulation."""
        from tnfr.parallel import TNFRDistributedEngine

        engine = TNFRDistributedEngine(backend="auto")

        # Simulate small network
        result = engine.simulate_large_network(
            node_count=50, edge_probability=0.1, operator_sequences=[], chunk_size=10
        )

        # Should return results with statistics
        assert "si_values" in result
        assert "network_stats" in result
        assert result["network_stats"]["nodes"] == 50

    def test_optimization_integration(self):
        """Test that all optimizations work together."""
        from tnfr.parallel import (
            FractalPartitioner,
            TNFRParallelEngine,
            TNFRGPUEngine,
            TNFRDistributedEngine,
            TNFRAutoScaler,
        )

        # Create test network
        G = nx.erdos_renyi_graph(100, 0.1)
        for node in G.nodes():
            G.nodes[node]["vf"] = 1.0
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["epi"] = 0.5
            G.nodes[node]["delta_nfr"] = 0.0
            G.nodes[node]["nu_f"] = 1.0

        # Test adaptive partitioner with spatial indexing
        partitioner = FractalPartitioner(adaptive=True, use_spatial_index=True)
        partitions = partitioner.partition_network(G)
        assert len(partitions) > 0

        # Test cache-aware parallel engine
        engine = TNFRParallelEngine(max_workers=2, cache_aware=True)
        si_result = engine.compute_si_parallel(G)
        assert len(si_result) == 100

        # Test GPU engine (numpy backend)
        gpu_engine = TNFRGPUEngine(backend="numpy")
        dnfr_result = gpu_engine.compute_delta_nfr_from_graph(G)
        assert len(dnfr_result) == 100

        # Test auto-scaler recommendation
        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(graph_size=100)
        assert "backend" in strategy

        print("\n✓ All optimizations integrated successfully")
