"""Tetrad Scaling Benchmark - Phase 4 Demonstration

Tests how canonical tetrad fields (Φ_s, |∇φ|, K_φ, ξ_C) scale across:
- Multiple network sizes (20, 50, 100, 200, 500 nodes)
- Different topologies (ring, ws, scale_free, grid)
- Various precision modes (standard, high, research)
- Different telemetry densities (low, medium, high)

This demonstrates Phase 4 capabilities:
- CLI-driven parameter scaling
- Precision/telemetry integration (Phases 1-3)
- Multi-topology universality tests
- Performance benchmarking at scale

Physics Invariance:
- All measurements are read-only (U1-U6 preserved)
- Precision modes affect only numerical accuracy
- Telemetry density affects only snapshot detail
"""

import json
import time
import numpy as np

from tnfr.physics.canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.metrics.tetrad import collect_tetrad_snapshot
from tnfr.config import (
    get_precision_mode,
    get_telemetry_density,
)
from cli_utils import (
    create_benchmark_parser,
    resolve_seeds,
    resolve_node_sizes,
    setup_output_dir,
    apply_precision_config,
)
from benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes


def benchmark_tetrad_computation(
    G, topology_type, n_nodes, seed, precision_mode, telemetry_density
):
    """Benchmark tetrad field computation on a single network."""
    results = {
        "topology": topology_type,
        "n_nodes": n_nodes,
        "seed": seed,
        "precision_mode": precision_mode,
        "telemetry_density": telemetry_density,
        "timings": {},
        "tetrad_values": {},
    }
    
    # Benchmark Φ_s
    start = time.time()
    phi_s = compute_structural_potential(G)
    results["timings"]["phi_s"] = time.time() - start
    results["tetrad_values"]["phi_s_mean"] = np.mean(list(phi_s.values()))
    results["tetrad_values"]["phi_s_std"] = np.std(list(phi_s.values()))
    
    # Benchmark |∇φ|
    start = time.time()
    grad = compute_phase_gradient(G)
    results["timings"]["phase_grad"] = time.time() - start
    results["tetrad_values"]["phase_grad_mean"] = np.mean(list(grad.values()))
    results["tetrad_values"]["phase_grad_std"] = np.std(list(grad.values()))
    
    # Benchmark K_φ
    start = time.time()
    curv = compute_phase_curvature(G)
    results["timings"]["phase_curv"] = time.time() - start
    results["tetrad_values"]["phase_curv_mean"] = np.mean(list(curv.values()))
    results["tetrad_values"]["phase_curv_std"] = np.std(list(curv.values()))
    
    # Benchmark ξ_C
    start = time.time()
    xi_c = estimate_coherence_length(G)
    results["timings"]["xi_c"] = time.time() - start
    xi_c_val = float(xi_c) if np.isfinite(xi_c) else None
    results["tetrad_values"]["xi_c"] = xi_c_val
    
    # Benchmark tetrad snapshot
    start = time.time()
    snapshot = collect_tetrad_snapshot(G)
    results["timings"]["tetrad_snapshot"] = time.time() - start
    results["snapshot_size"] = len(json.dumps(snapshot))
    
    # Total time
    results["total_time"] = sum(results["timings"].values())
    
    return results


def run_scaling_experiment(
    topologies, node_sizes, seeds, output_dir, quiet=False
):
    """Run tetrad scaling experiments across configurations."""
    all_results = []
    
    precision_mode = get_precision_mode()
    telemetry_density = get_telemetry_density()
    
    for topology in topologies:
        for n_nodes in node_sizes:
            if not quiet:
                print(f"\n{'='*60}")
                print(f"Topology: {topology}, Nodes: {n_nodes}")
                print(f"{'='*60}")
            
            topology_times = []
            
            for seed in seeds:
                if not quiet:
                    print(f"  Seed {seed}...", end=" ", flush=True)
                
                # Create topology
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, nu_f=1.0, seed=seed)
                
                # Add some ΔNFR variation
                np.random.seed(seed)
                for node in G.nodes():
                    G.nodes[node]["delta_nfr"] = np.random.uniform(-1, 1)
                
                # Run benchmark
                result = benchmark_tetrad_computation(
                    G, topology, n_nodes, seed,
                    precision_mode, telemetry_density
                )
                
                all_results.append(result)
                topology_times.append(result["total_time"])
                
                if not quiet:
                    print(f"{result['total_time']:.3f}s")
            
            # Summary statistics
            if not quiet:
                mean_time = np.mean(topology_times)
                std_time = np.std(topology_times)
                print(f"  Average: {mean_time:.3f}s ± {std_time:.3f}s")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"tetrad_scaling_{timestamp}.jsonl"
    output_file = output_dir / filename
    
    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    if not quiet:
        print(f"\nResults saved to: {output_file}")
    
    return all_results, output_file


def print_summary(results, quiet=False):
    """Print summary statistics from results."""
    if quiet:
        return
    
    print(f"\n{'='*60}")
    print("Scaling Summary")
    print(f"{'='*60}")
    
    # Group by topology and size
    by_topo_size = {}
    for r in results:
        key = (r["topology"], r["n_nodes"])
        if key not in by_topo_size:
            by_topo_size[key] = []
        by_topo_size[key].append(r["total_time"])
    
    print("\nTiming Results (seconds):")
    print(f"{'Topology':<15} {'Nodes':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 60)
    
    for (topo, n_nodes), times in sorted(by_topo_size.items()):
        mean_t = np.mean(times)
        std_t = np.std(times)
        print(f"{topo:<15} {n_nodes:<10} {mean_t:<10.4f} {std_t:<10.4f}")
    
    # Performance scaling
    print("\nPerformance Insights:")
    
    # Find linear scaling coefficient
    all_nodes = [r["n_nodes"] for r in results]
    all_times = [r["total_time"] for r in results]
    
    if len(set(all_nodes)) > 1:
        # Simple linear fit
        coeffs = np.polyfit(all_nodes, all_times, 1)
        print(f"  Linear fit: T ≈ {coeffs[0]:.6f} * N + {coeffs[1]:.6f}")
        print(f"  Time per node: ~{coeffs[0]*1000:.3f} ms")


if __name__ == "__main__":
    parser = create_benchmark_parser(
        description="Tetrad Scaling Benchmark (Phase 4 Demo)",
        default_nodes=50,
        default_seeds=5,
        default_topologies=["ws", "scale_free"],
        add_precision_flags=True,
        add_param_grid=False,
    )
    
    args = parser.parse_args()
    
    # Apply configuration
    apply_precision_config(args)
    
    # Resolve parameters
    seeds = resolve_seeds(args)
    node_sizes = resolve_node_sizes(args)
    topologies = args.topologies
    
    # Setup output
    output_dir = setup_output_dir(args)
    
    # Print configuration
    if not args.quiet:
        print("="*60)
        print("Tetrad Scaling Benchmark - Phase 4")
        print("="*60)
        print("Configuration:")
        print(f"  Topologies: {topologies}")
        print(f"  Node sizes: {node_sizes}")
        print(f"  Seeds: {len(seeds)} runs")
        precision = get_precision_mode()
        telemetry = get_telemetry_density()
        print(f"  Precision: {precision}")
        print(f"  Telemetry: {telemetry}")
        print(f"  Output: {output_dir}")
    
    # Run experiments
    results, output_file = run_scaling_experiment(
        topologies, node_sizes, seeds, output_dir, args.quiet
    )
    
    # Print summary
    print_summary(results, args.quiet)
    
    if not args.quiet:
        print(f"\nBenchmark complete!")
        print(f"Total configurations: {len(results)}")
        print(f"Results: {output_file}")
