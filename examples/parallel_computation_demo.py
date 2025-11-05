#!/usr/bin/env python3
"""Demonstration of parallel computation features in TNFR.

This example shows how to use the parallel computation infrastructure to
accelerate TNFR operations on medium to large networks while preserving all
structural invariants.
"""

import networkx as nx
from tnfr.parallel import (
    FractalPartitioner,
    TNFRParallelEngine,
    TNFRAutoScaler,
    ParallelExecutionMonitor,
)


def create_sample_network(size: int = 100) -> nx.Graph:
    """Create a sample TNFR network for demonstration.
    
    Parameters
    ----------
    size : int
        Number of nodes in the network
        
    Returns
    -------
    nx.Graph
        Network with TNFR attributes
    """
    print(f"Creating sample network with {size} nodes...")
    
    # Create random network
    G = nx.erdos_renyi_graph(size, 0.1, seed=42)
    
    # Add TNFR attributes efficiently using batch operations
    nx.set_node_attributes(G, 1.0, "nu_f")  # Structural frequency
    nx.set_node_attributes(G, 0.0, "phase")  # Phase
    nx.set_node_attributes(G, 0.5, "epi")  # EPI
    nx.set_node_attributes(G, 0.0, "delta_nfr")  # ΔNFR
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.3f}")
    
    return G


def demo_fractal_partitioner(G: nx.Graph) -> None:
    """Demonstrate TNFR-aware network partitioning."""
    print("\n" + "="*60)
    print("1. Fractal Partitioner Demo")
    print("="*60)
    
    partitioner = FractalPartitioner(
        max_partition_size=50,
        coherence_threshold=0.3
    )
    
    print("\nPartitioning network based on structural coherence...")
    partitions = partitioner.partition_network(G)
    
    print(f"\nPartitioning complete:")
    print(f"  Total partitions: {len(partitions)}")
    
    for i, (node_set, subgraph) in enumerate(partitions, 1):
        print(f"  Partition {i}: {len(node_set)} nodes, "
              f"{subgraph.number_of_edges()} edges")


def demo_auto_scaler(graph_size: int) -> dict:
    """Demonstrate execution strategy recommendation."""
    print("\n" + "="*60)
    print("2. Auto-Scaler Demo")
    print("="*60)
    
    scaler = TNFRAutoScaler()
    
    print(f"\nGetting execution strategy for {graph_size}-node network...")
    strategy = scaler.recommend_execution_strategy(
        graph_size=graph_size,
        available_memory_gb=8.0,
        has_gpu=False
    )
    
    print("\nRecommended strategy:")
    print(f"  Backend: {strategy['backend']}")
    print(f"  Workers: {strategy.get('workers', 'N/A')}")
    print(f"  Explanation: {strategy['explanation']}")
    print(f"  Estimated time: {strategy['estimated_time_minutes']:.2f} minutes")
    print(f"  Estimated memory: {strategy['estimated_memory_gb']:.3f} GB")
    
    return strategy


def demo_parallel_engine(G: nx.Graph, workers: int = 2) -> None:
    """Demonstrate parallel computation."""
    print("\n" + "="*60)
    print("3. Parallel Engine Demo")
    print("="*60)
    
    engine = TNFRParallelEngine(
        max_workers=workers,
        execution_mode="threads"
    )
    
    print(f"\nComputing sense index with {engine.max_workers} workers...")
    
    # Use monitoring to track performance
    monitor = ParallelExecutionMonitor()
    monitor.start_monitoring(
        expected_nodes=len(G),
        workers=engine.max_workers
    )
    
    # Compute Si in parallel
    si_results = engine.compute_si_parallel(G)
    
    # Calculate coherence from Si values
    final_coherence = sum(si_results.values()) / len(si_results)
    metrics = monitor.stop_monitoring(
        final_coherence=final_coherence,
        initial_coherence=0.7
    )
    
    print(f"\nComputation complete:")
    print(f"  Nodes processed: {metrics.nodes_processed}")
    print(f"  Duration: {metrics.duration_seconds:.3f}s")
    print(f"  Throughput: {metrics.operations_per_second:.0f} ops/s")
    print(f"  Average Si: {final_coherence:.3f}")
    print(f"  Coherence improvement: {metrics.coherence_improvement:.3f}")


def demo_optimization_suggestions(monitor: ParallelExecutionMonitor) -> None:
    """Demonstrate optimization suggestions."""
    print("\n" + "="*60)
    print("4. Optimization Suggestions")
    print("="*60)
    
    suggestions = monitor.get_optimization_suggestions()
    
    print("\nPerformance suggestions:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")


def main():
    """Run all demonstrations."""
    print("="*60)
    print("TNFR Parallel Computation Demonstration")
    print("="*60)
    
    # Create sample network
    network_size = 100
    G = create_sample_network(network_size)
    
    # Demo 1: Fractal partitioning
    demo_fractal_partitioner(G)
    
    # Demo 2: Auto-scaling
    strategy = demo_auto_scaler(network_size)
    
    # Demo 3: Parallel execution
    workers = strategy.get("workers", 2)
    demo_parallel_engine(G, workers=min(workers, 4))
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60)
    
    print("\nKey takeaways:")
    print("  • FractalPartitioner respects TNFR coherence for partitioning")
    print("  • AutoScaler recommends optimal strategies based on network size")
    print("  • ParallelEngine integrates seamlessly with existing TNFR code")
    print("  • All TNFR structural invariants are preserved")
    print("\nFor massive networks (>10,000 nodes), consider:")
    print("  • Distributed backends (Ray/Dask) - requires optional deps")
    print("  • GPU acceleration (JAX/CuPy) - requires optional deps")


if __name__ == "__main__":
    main()
