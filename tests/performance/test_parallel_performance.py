"""Performance tests for parallel computation."""

import pytest
import networkx as nx
import time


@pytest.mark.slow
class TestParallelPerformance:
    """Performance tests for parallel TNFR computation."""

    def _create_test_network(self, size: int):
        """Create a test network with TNFR attributes."""
        G = nx.erdos_renyi_graph(size, 0.1, seed=42)

        # Add TNFR attributes
        for node in G.nodes():
            G.nodes[node]["nu_f"] = 1.0
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["epi"] = 0.5
            G.nodes[node]["delta_nfr"] = 0.0

        return G

    def test_partitioner_performance(self):
        """Test partitioner scales reasonably with network size."""
        from tnfr.parallel import FractalPartitioner

        sizes = [50, 100, 200]
        times = []

        for size in sizes:
            G = self._create_test_network(size)
            partitioner = FractalPartitioner(max_partition_size=50)

            start = time.time()
            partitions = partitioner.partition_network(G)
            elapsed = time.time() - start
            times.append(elapsed)

            print(f"\nSize {size}: {elapsed:.3f}s, {len(partitions)} partitions")

            # Should complete in reasonable time
            # Note: Current implementation is O(n²) for dense graphs
            # Future optimization could use spatial indexing for large networks
            assert elapsed < 10.0, f"Partitioning {size} nodes took {elapsed:.1f}s"

    def test_parallel_engine_recommendation(self):
        """Test that engine makes sensible recommendations."""
        from tnfr.parallel import TNFRParallelEngine

        engine = TNFRParallelEngine(max_workers=4)

        # Small network should recommend serial
        assert engine.recommend_workers(20) == 1

        # Medium network should use some parallelism
        workers_medium = engine.recommend_workers(200)
        assert 1 < workers_medium <= 4

        # Large network should use full parallelism
        workers_large = engine.recommend_workers(1000)
        assert workers_large == 4

    def test_auto_scaler_recommendations(self):
        """Test auto-scaler provides reasonable strategies."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()

        # Small network
        strategy = scaler.recommend_execution_strategy(
            graph_size=50, available_memory_gb=8.0
        )
        assert strategy["backend"] == "sequential"
        assert strategy["estimated_time_minutes"] < 1.0

        # Medium network
        strategy = scaler.recommend_execution_strategy(
            graph_size=500, available_memory_gb=8.0
        )
        assert strategy["backend"] == "multiprocessing"
        assert strategy["workers"] > 1

        print(f"\nMedium network strategy: {strategy['explanation']}")
        print(f"Estimated time: {strategy['estimated_time_minutes']:.2f} minutes")
        print(f"Estimated memory: {strategy['estimated_memory_gb']:.2f} GB")

    def test_monitor_overhead(self):
        """Test that monitoring has minimal overhead."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()

        # Measure overhead of monitoring
        start = time.time()
        monitor.start_monitoring(expected_nodes=1000, workers=4)
        time.sleep(0.01)  # Simulate minimal work
        metrics = monitor.stop_monitoring(final_coherence=0.9, initial_coherence=0.8)
        elapsed = time.time() - start

        # Monitoring overhead should be negligible
        assert elapsed < 0.1  # Should complete in < 100ms
        assert metrics.duration_seconds < 0.1

        print(
            f"\nMonitoring overhead: {(elapsed - metrics.duration_seconds) * 1000:.1f}ms"
        )

    @pytest.mark.parametrize("size", [100, 200])
    def test_compute_si_scaling(self, size):
        """Test that Si computation scales with network size."""
        from tnfr.metrics.sense_index import compute_Si

        G = self._create_test_network(size)

        # Measure serial execution
        start = time.time()
        result_serial = compute_Si(G, inplace=False, n_jobs=1)
        time_serial = time.time() - start

        # Measure parallel execution
        start = time.time()
        result_parallel = compute_Si(G, inplace=False, n_jobs=2)
        time_parallel = time.time() - start

        # Verify results are consistent
        assert len(result_serial) == size
        assert len(result_parallel) == size

        # For small networks, parallel may not be faster due to overhead
        # But it should at least complete successfully
        print(
            f"\nSize {size}: Serial={time_serial:.3f}s, Parallel={time_parallel:.3f}s"
        )

    def test_integration_example(self):
        """Integration test showing typical usage pattern."""
        from tnfr.parallel import (
            TNFRParallelEngine,
            TNFRAutoScaler,
            ParallelExecutionMonitor,
        )

        # Create test network
        G = self._create_test_network(150)

        # Get recommendation
        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=len(G), available_memory_gb=8.0
        )

        # Create engine with recommended workers
        engine = TNFRParallelEngine(max_workers=strategy.get("workers", 1))

        # Monitor execution
        monitor = ParallelExecutionMonitor()
        monitor.start_monitoring(expected_nodes=len(G), workers=engine.max_workers)

        # Execute parallel computation
        si_results = engine.compute_si_parallel(G)

        # Stop monitoring
        initial_coherence = 0.7
        final_coherence = sum(si_results.values()) / len(si_results)
        metrics = monitor.stop_monitoring(
            final_coherence=final_coherence, initial_coherence=initial_coherence
        )

        # Verify results
        assert len(si_results) == len(G)
        assert metrics.nodes_processed == len(G)
        assert metrics.duration_seconds > 0

        print(f"\n{'='*60}")
        print(f"Integration Test Results:")
        print(f"  Network size: {len(G)} nodes")
        print(f"  Strategy: {strategy['backend']}")
        print(f"  Workers: {engine.max_workers}")
        print(f"  Duration: {metrics.duration_seconds:.3f}s")
        print(f"  Throughput: {metrics.operations_per_second:.0f} ops/s")
        print(f"  Coherence improvement: {metrics.coherence_improvement:.3f}")
        print(f"{'='*60}")
