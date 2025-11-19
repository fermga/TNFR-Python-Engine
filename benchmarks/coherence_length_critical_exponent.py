#!/usr/bin/env python3
"""
Coherence Length Critical Exponent Extraction Experiments

This script runs fine-grained intensity sweeps to measure coherence length
xi_C near the critical transition I_c = 2.015, then fits power-law divergence
to extract critical exponent nu.

Tasks:
1. Fine-grained intensity sweep I ∈ [1.8, 2.2]
2. Extract critical exponent nu from xi_C ~ |I - I_c|^(-nu)
3. Classify universality class (mean-field, Ising, etc.)
4. Test reproducibility across topologies

Expected: nu ~ 0.5 (mean-field class, consistent with beta = 0.556)
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from tnfr.physics.fields import (
        estimate_coherence_length,
        fit_correlation_length_exponent,
        measure_phase_symmetry
    )
    from benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes

    from tnfr.operators.definitions import Dissonance, Coherence
    from tnfr.dynamics.dnfr import default_compute_delta_nfr
    from tnfr.metrics.common import compute_coherence
    from tnfr.config import DNFR_PRIMARY
    import networkx as nx
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root")
    sys.exit(1)

# Critical point constant for phase transitions
I_C = 2.015


def create_test_topology(
    topology_type: str, n_nodes: int = 50, seed: int = 42
) -> nx.Graph:
    """Create test topology for coherence length experiments."""
    # Create topology using benchmark utils
    G = create_tnfr_topology(topology_type, n_nodes, seed)
    
    # Initialize nodes with TNFR attributes
    initialize_tnfr_nodes(G, nu_f=1.0, seed=seed)
    
    return G


def run_intensity_sweep(
    topology: str,
    intensities: List[float],
    n_runs: int = 50,
    n_nodes: int = 50,
    seed_base: int = 42
) -> Dict[str, Any]:
    """Run coherence length measurements across intensity range."""
    
    results = {
        "topology": topology,
        "n_nodes": n_nodes,
        "n_runs": n_runs,
        "intensities": intensities,
        "xi_c_data": [],
        "symmetry_data": [],
        "coherence_data": []
    }
    
    print(f"Testing {topology} topology with {n_runs} runs per intensity...")
    
    for i, intensity in enumerate(intensities):
        print(f"  Intensity {intensity:.3f} ({i+1}/{len(intensities)})")
        
        xi_c_runs = []
        symmetry_runs = []
        coherence_runs = []
        
        for run in range(n_runs):
            seed = seed_base + i * 1000 + run
            np.random.seed(seed)
            
            try:
                # Create fresh topology
                G = create_test_topology(topology, n_nodes)
                
                # Create network-wide perturbation proportional to intensity
                # This simulates a system under increasing structural stress
                for node in G.nodes():
                    # Random perturbation scaled by intensity around critical point
                    delta_I = intensity - I_C  # Distance from critical point
                    base_dnfr = delta_I * np.random.normal(0, 0.5)
                    G.nodes[node][DNFR_PRIMARY] = base_dnfr
                
                # Apply multiple dissonance operators for network evolution
                # Select random nodes for perturbation based on intensity
                perturb_factor = 0.2 * n_nodes * abs(intensity - I_C)
                n_perturbed = max(1, int(perturb_factor))
                node_list = list(G.nodes())
                perturbed_nodes = np.random.choice(
                    node_list,
                    size=min(n_perturbed, len(node_list)),
                    replace=False
                )
                
                dissonance = Dissonance()
                for node in perturbed_nodes:
                    dissonance(G, node)
                
                # Let system evolve for several steps to develop correlations
                for step in range(10):  # Multiple evolution steps
                    default_compute_delta_nfr(G)
                    # Apply coherence to stabilize some nodes
                    coherence_op = Coherence()
                    node_list = list(G.nodes())
                    stabilized_nodes = np.random.choice(
                        node_list, size=max(1, n_nodes//4), replace=False
                    )
                    for node in stabilized_nodes:
                        coherence_op(G, node)
                
                # Final ΔNFR calculation after evolution
                default_compute_delta_nfr(G)
                
                # Measure coherence length (uses delta_nfr attribute)
                xi_c = estimate_coherence_length(G)
                xi_c_runs.append(xi_c)
                
                # Measure phase symmetry
                symmetry = measure_phase_symmetry(G)
                symmetry_runs.append(symmetry)
                
                # Measure global coherence
                coherence = compute_coherence(G)
                coherence_runs.append(coherence)
                
            except Exception as e:
                print(f"    Run {run} failed: {e}")
                # Use fallback values
                xi_c_runs.append(0.0)
                symmetry_runs.append({
                    "circular_variance": 0.5,
                    "n_clusters": 1,
                    "cluster_separation": 0.0,
                    "largest_cluster_fraction": 1.0
                })
                coherence_runs.append(0.5)
        
        # Store aggregated data
        results["xi_c_data"].append({
            "intensity": intensity,
            "values": xi_c_runs,
            "mean": float(np.mean(xi_c_runs)),
            "std": float(np.std(xi_c_runs)),
            "median": float(np.median(xi_c_runs))
        })
        
        results["symmetry_data"].append({
            "intensity": intensity,
            "circular_variance": [
                s["circular_variance"] for s in symmetry_runs
            ],
            "n_clusters": [s["n_clusters"] for s in symmetry_runs],
            "cluster_separation": [
                s["cluster_separation"] for s in symmetry_runs
            ],
            "largest_cluster_fraction": [
                s["largest_cluster_fraction"] for s in symmetry_runs
            ]
        })
        
        results["coherence_data"].append({
            "intensity": intensity,
            "values": coherence_runs,
            "mean": float(np.mean(coherence_runs)),
            "std": float(np.std(coherence_runs))
        })
    
    return results


def analyze_critical_exponent(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract critical exponent from coherence length data."""
    
    # Extract mean xi_C values
    intensities = np.array([d["intensity"] for d in results["xi_c_data"]])
    xi_c_means = np.array([d["mean"] for d in results["xi_c_data"]])
    
    # Filter out zero/invalid measurements
    valid_mask = (xi_c_means > 1e-6) & np.isfinite(xi_c_means)
    intensities_clean = intensities[valid_mask]
    xi_c_clean = xi_c_means[valid_mask]
    
    if len(intensities_clean) < 6:
        return {
            "success": False,
            "error": "Insufficient valid data points",
            "n_valid": len(intensities_clean)
        }
    
    # Fit critical exponent
    exponent_fit = fit_correlation_length_exponent(
        intensities_clean, xi_c_clean, I_c=2.015, min_distance=0.01
    )
    
    return {
        "success": True,
        "topology": results["topology"],
        "n_valid": len(intensities_clean),
        "exponent_fit": exponent_fit,
        "intensity_range": [
            float(np.min(intensities_clean)),
            float(np.max(intensities_clean))
        ],
        "xi_c_range": [
            float(np.min(xi_c_clean)),
            float(np.max(xi_c_clean))
        ]
    }


def main():
    """Run coherence length critical exponent experiments."""
    
    # Fine-grained intensity sweep around I_c = 2.015
    intensities = [
        1.8, 1.9, 1.95, 2.0, 2.01, 2.015, 2.02, 2.03, 2.04,
        2.05, 2.07, 2.1, 2.2
    ]
    
    topologies = ["ring", "ws", "scale_free", "tree", "grid"]
    
    # Results storage
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = (
        results_dir / f"coherence_length_critical_exponent_{timestamp}.jsonl"
    )
    
    print("Coherence Length Critical Exponent Extraction")
    print("=" * 50)
    print(f"Intensities: {intensities}")
    print(f"Topologies: {topologies}")
    print(f"Results: {results_file}")
    print()
    
    all_results = []
    
    try:
        for topology in topologies:
            print(f"Running {topology} experiments...")
            start_time = time.time()
            
            # Run intensity sweep
            topo_results = run_intensity_sweep(
                topology=topology,
                intensities=intensities,
                n_runs=30,  # Reduced for speed, increase to 50+ for final
                n_nodes=50,
                seed_base=12345
            )
            
            # Analyze critical exponent
            analysis = analyze_critical_exponent(topo_results)
            
            # Combine results
            combined_result = {
                "timestamp": timestamp,
                "experiment": "coherence_length_critical_exponent",
                "topology": topology,
                "raw_data": topo_results,
                "analysis": analysis,
                "duration_seconds": time.time() - start_time
            }
            
            all_results.append(combined_result)
            
            # Save incremental results
            with open(results_file, "a") as f:
                f.write(json.dumps(combined_result) + "\n")
            
            # Print summary
            if analysis["success"]:
                fit = analysis["exponent_fit"]
                nu_avg = (fit['nu_below'] + fit['nu_above']) / 2
                r2_avg = (fit['r_squared_below'] + fit['r_squared_above']) / 2
                print(f"  {topology:12} | nu_avg = {nu_avg:.3f} | "
                      f"R²_avg = {r2_avg:.3f} | "
                      f"class = {fit['universality_class']}")
            else:
                error_msg = analysis.get('error', 'unknown')
                print(f"  {topology:12} | FAILED: {error_msg}")
            
            print(f"  Duration: {time.time() - start_time:.1f}s")
            print()
    
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Partial results saved.")
    
    # Generate summary
    print("Critical Exponent Summary:")
    print("=" * 40)
    
    successful_analyses = [
        r["analysis"] for r in all_results if r["analysis"]["success"]
    ]
    
    if successful_analyses:
        # Collect all nu values
        all_nu_below = [
            a["exponent_fit"]["nu_below"] for a in successful_analyses
            if a["exponent_fit"]["nu_below"] > 0
        ]
        all_nu_above = [
            a["exponent_fit"]["nu_above"] for a in successful_analyses
            if a["exponent_fit"]["nu_above"] > 0
        ]
        
        if all_nu_below:
            nu_below_mean = np.mean(all_nu_below)
            nu_below_std = np.std(all_nu_below)
            print(f"nu (below I_c): {nu_below_mean:.3f} ± {nu_below_std:.3f}")
        if all_nu_above:
            nu_above_mean = np.mean(all_nu_above)
            nu_above_std = np.std(all_nu_above)
            print(f"nu (above I_c): {nu_above_mean:.3f} ± {nu_above_std:.3f}")
        
        # Universality classification
        classes = [
            a["exponent_fit"]["universality_class"]
            for a in successful_analyses
        ]
        from collections import Counter
        class_counts = Counter(classes)
        print(f"Universality classes: {dict(class_counts)}")
        
        if "mean-field" in class_counts:
            print("✓ Mean-field behavior detected (nu ~ 0.5)")
        
    print(f"\nComplete results saved to: {results_file}")


if __name__ == "__main__":
    from cli_utils import (
        create_benchmark_parser,
        resolve_seeds,
        resolve_node_sizes,
        setup_output_dir,
        apply_precision_config,
        get_param_grid_points,
    )
    
    # Create CLI parser
    parser = create_benchmark_parser(
        description="Coherence Length Critical Exponent Extraction",
        default_nodes=50,
        default_seeds=30,
        default_topologies=["ws", "scale_free"],
        add_precision_flags=True,
        add_param_grid=True,
    )
    # Lightweight Phase 4 test harness skip
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse arguments and configuration, then exit without "
            "executing sweeps"
        ),
    )
    
    args = parser.parse_args()
    
    # Apply precision/telemetry configuration
    apply_precision_config(args)
    
    # Resolve parameters
    seeds = resolve_seeds(args)
    node_sizes = resolve_node_sizes(args)
    topologies = args.topologies
    
    # Generate intensity grid
    if hasattr(args, "param_grid_resolution"):
        param_range = (
            args.param_range if hasattr(args, "param_range")
            else None
        )
        intensities = get_param_grid_points(
            resolution=args.param_grid_resolution,
            critical_point=I_C,
            param_range=param_range,
        )
    else:
        # Default intensity range
        intensities = list(np.linspace(1.8, 2.2, 25))
    
    # Setup output
    output_dir = setup_output_dir(args)
    
    # Store precision mode for use in results
    precision = args.precision if hasattr(args, "precision") else "standard"
    
    # Run main experiment with CLI parameters
    if not args.quiet:
        print("Configuration:")
        print(f"  Topologies: {topologies}")
        print(f"  Node sizes: {node_sizes}")
        print(f"  Seeds: {len(seeds)} runs")
        print(f"  Intensities: {len(intensities)} points")
        precision = (
            args.precision if hasattr(args, "precision")
            else "standard"
        )
        print(f"  Precision: {precision}")
        print(f"  Output: {output_dir}")
        print()

    if getattr(args, "dry_run", False):
        # Early exit for CLI parameter validation tests
        if not args.quiet:
            print(
                "[DRY-RUN] Skipping intensity sweeps; CLI parameters "
                "validated."
            )
        sys.exit(0)
    
    # Run experiments for each configuration
    for n_nodes in node_sizes:
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"Network size: {n_nodes} nodes")
            print(f"{'='*60}")
        
        # Call original main() logic with CLI params
        # (Would need to refactor main() to accept parameters)
        # For now, run inline
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"xi_c_critical_{n_nodes}nodes_{timestamp}.jsonl"
        results_file = output_dir / filename
        
        all_results = []
        
        for topology in topologies:
            if not args.quiet:
                print(f"\nRunning {topology} experiments...")
            start_time = time.time()
            
            # Run intensity sweep with CLI parameters
            topo_results = run_intensity_sweep(
                topology=topology,
                intensities=intensities,
                n_runs=len(seeds),
                n_nodes=n_nodes,
                seed_base=args.seed
            )
            
            # Analyze critical exponent
            analysis = analyze_critical_exponent(topo_results)
            
            # Combine results
            combined_result = {
                "timestamp": timestamp,
                "experiment": "coherence_length_critical_exponent_cli",
                "topology": topology,
                "n_nodes": n_nodes,
                "n_seeds": len(seeds),
                "precision_mode": precision,
                "raw_data": topo_results,
                "analysis": analysis,
                "duration_seconds": time.time() - start_time
            }
            
            all_results.append(combined_result)
            
            # Save incremental results
            with open(results_file, "a") as f:
                f.write(json.dumps(combined_result) + "\n")
            
            if not args.quiet:
                # Print summary
                if analysis["exponent_fit"]["success"]:
                    nu = analysis["exponent_fit"]["nu"]
                    nu_err = analysis["exponent_fit"]["nu_error"]
                    univ_class = analysis["exponent_fit"]["universality_class"]
                    print(f"  Critical exponent: nu = {nu:.3f} ± {nu_err:.3f}")
                    print(f"  Universality class: {univ_class}")
                else:
                    print(f"  Fit failed: {analysis['exponent_fit']['error']}")
                
                print(f"  Duration: {time.time() - start_time:.1f}s")
        
        if not args.quiet:
            print(f"\nResults saved to: {results_file}")
