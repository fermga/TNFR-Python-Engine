#!/usr/bin/env python3
"""
K_φ Asymptotic Freedom Investigation

Task 3: Test if |K_φ| variance decreases at larger scales following
var(K_φ) ~ 1/r^α pattern (asymptotic freedom analogy).
"""

import sys
import json
import random
from pathlib import Path
import argparse

import numpy as np
import networkx as nx
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import compute_phase_curvature
from benchmarks.benchmark_utils import (create_tnfr_topology,
                                        initialize_tnfr_nodes)
from src.tnfr.operators.definitions import Dissonance, Coherence


def asymptotic_freedom_investigation(topologies=None, n_nodes=50, n_tests=8, high_resolution=False, seed=42, quiet=False):
    """Test scale-dependent K_φ variance for asymptotic freedom.

    Parameters
    ----------
    topologies : list[str] | None
        Topologies to test. Defaults to ['ring','scale_free','tree','ws'] if None.
    n_nodes : int
        Number of nodes per topology.
    n_tests : int
        Number of seed experiments per topology.
    high_resolution : bool
        If True, extend hop distances for finer scaling analysis.
    seed : int
        Base RNG seed controlling per-test seeds.
    quiet : bool
        Suppress verbose progress & summary output (used in CI tests).
    """
    if topologies is None:
        topologies = ['ring', 'scale_free', 'tree', 'ws']

    if not quiet:
        print("🌌 K_φ Asymptotic Freedom Investigation")
        print("=" * 45)
        print(f"Topologies: {topologies}")
        print(f"Nodes per topology: {n_nodes}")
        print(f"Tests per topology: {n_tests}")
        print(f"High-resolution: {high_resolution}")

    # Base hop distances
    scales = [1, 2, 3, 4, 5, 7, 10]
    if high_resolution:
        scales = sorted(set(scales + [6, 8, 9, 12, 15]))

    rng = random.Random(seed)
    results = []

    for topology in topologies:
        if not quiet:
            print(f"\n🔭 {topology.upper()} - Multi-Scale K_φ Analysis:")
        for test_id in range(n_tests):
            local_seed = rng.randint(1000, 9999)
            try:
                G = create_tnfr_topology(topology, n_nodes, local_seed)
                initialize_tnfr_nodes(G, seed=local_seed)
                if not quiet:
                    print(f"  Test {test_id}: ", end="")
                dynamic_state = apply_dynamics_sequence(G)
                k_phi = compute_phase_curvature(G)
                scale_analysis = analyze_multiscale_variance(G, k_phi, scales)
                power_law_fit = fit_asymptotic_freedom_law(scale_analysis)
                result = {
                    'topology': topology,
                    'test_id': test_id,
                    'n_nodes': len(G.nodes()),
                    'n_edges': len(G.edges()),
                    'seed': local_seed,
                    'dynamic_state': dynamic_state,
                    'scale_analysis': scale_analysis,
                    'power_law_fit': power_law_fit,
                    'scales': scales,
                    'high_resolution': high_resolution,
                }
                results.append(result)
                alpha = power_law_fit['alpha']
                r_squared = power_law_fit['r_squared']
                evidence = "✅" if (alpha > 0 and r_squared > 0.5) else "❌"
                if not quiet:
                    print(f"α={alpha:.2f} R²={r_squared:.3f} {evidence}")
            except Exception as e:
                if not quiet:
                    print(f"ERROR: {e}")
                continue

    if results and not quiet:
        print("\n📊 ASYMPTOTIC FREEDOM ANALYSIS SUMMARY:")
        print(f"Total experiments: {len(results)}")
        
        # Power law statistics
        alphas = [r['power_law_fit']['alpha'] for r in results]
        r_squareds = [r['power_law_fit']['r_squared'] for r in results]
        
        positive_alphas = [a for a in alphas if a > 0]
        good_fits = [r for r in r_squareds if r > 0.5]
        
        print("\n🔬 Power Law Fit Statistics:")
        print(
            "   Alpha (α) mean: "
            f"{np.mean(alphas):.3f} ± {np.std(alphas):.3f}"
        )
        print(f"   Positive α rate: {len(positive_alphas)}/{len(alphas)} "
              f"({100*len(positive_alphas)/len(alphas):.1f}%)")
        print(f"   Good fits (R²>0.5): {len(good_fits)}/{len(r_squareds)} "
              f"({100*len(good_fits)/len(r_squareds):.1f}%)")
        print(f"   Mean R²: {np.mean(r_squareds):.3f}")
        
        # Asymptotic freedom evidence classification
        strong_evidence = sum(
            1 for r in results
            if (
                r['power_law_fit']['alpha'] > 0 and
                r['power_law_fit']['r_squared'] > 0.7
            )
        )

        moderate_evidence = sum(
            1 for r in results
            if (
                r['power_law_fit']['alpha'] > 0 and
                r['power_law_fit']['r_squared'] > 0.5
            )
        )
        
        print("\n🎯 Asymptotic Freedom Evidence:")
        print(
            "   Strong evidence (α>0, R²>0.7): "
            f"{strong_evidence}/{len(results)} "
            f"({100*strong_evidence/len(results):.1f}%)"
        )
        print(
            "   Moderate evidence (α>0, R²>0.5): "
            f"{moderate_evidence}/{len(results)} "
            f"({100*moderate_evidence/len(results):.1f}%)"
        )
        
        # Topology-specific analysis
        print("\n🗺️ Topology-Specific Asymptotic Behavior:")
        print(
            f"{'Topology':<12} {'Mean α':<8} "
            f"{'Mean R²':<8} {'Evidence':<12}"
        )
        print("-" * 50)
        
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            
            if topo_results:
                topo_alphas = [
                    r['power_law_fit']['alpha'] for r in topo_results
                ]
                topo_r2s = [
                    r['power_law_fit']['r_squared'] for r in topo_results
                ]
                
                mean_alpha = np.mean(topo_alphas)
                mean_r2 = np.mean(topo_r2s)
                
                # Evidence classification for topology
                topo_evidence = sum(
                    1 for r in topo_results
                    if (
                        r['power_law_fit']['alpha'] > 0 and
                        r['power_law_fit']['r_squared'] > 0.5
                    )
                )
                
                evidence_rate = topo_evidence / len(topo_results)
                if evidence_rate > 0.6:
                    evidence_label = "Strong"
                elif evidence_rate > 0.3:
                    evidence_label = "Weak"
                else:
                    evidence_label = "None"
                
                print(
                    f"{topology:<12} {mean_alpha:<8.3f} "
                    f"{mean_r2:<8.3f} {evidence_label:<12}"
                )
        
        # Scale-dependent patterns
        print("\n📏 Scale-Dependent K_φ Variance Patterns:")
        
        # Aggregate variance by scale across all experiments
        all_scales = scales
        scale_variances = {scale: [] for scale in all_scales}
        
        for result in results:
            for scale_data in result['scale_analysis']:
                scale = scale_data['scale']
                if scale in scale_variances:
                    scale_variances[scale].append(scale_data['variance'])
        
        print(
            f"{'Scale (hops)':<12} {'Mean Var':<10} "
            f"{'Std Var':<10} {'N_samples':<10}"
        )
        print("-" * 50)
        
        for scale in all_scales:
            if scale_variances[scale]:
                mean_var = np.mean(scale_variances[scale])
                std_var = np.std(scale_variances[scale])
                n_samples = len(scale_variances[scale])
                
                print(
                    f"{scale:<12} {mean_var:<10.4f} "
                    f"{std_var:<10.4f} {n_samples:<10}"
                )
        
        # Save results
        output_file = (
            PROJECT_ROOT / "benchmarks" / "results" /
            "asymptotic_freedom_analysis.jsonl"
        )
        with open(output_file, 'w') as f:
            for result in results:
                # Convert numpy types to native Python for JSON serialization
                json_safe_result = convert_numpy_types(result)
                f.write(json.dumps(json_safe_result) + '\n')
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # === CONCLUSIONS ===
        print("\n🔍 ASYMPTOTIC FREEDOM CONCLUSIONS:")
        
        overall_evidence_rate = moderate_evidence / len(results)
        
        if overall_evidence_rate > 0.6:
            print("   ✅ STRONG ASYMPTOTIC FREEDOM EVIDENCE")
            print(
                "      - " f"{overall_evidence_rate:.1%} of experiments show "
                "var(K_φ) ~ 1/r^α"
            )
            print(
                "      - Mean α = "
                f"{np.mean([a for a in alphas if a > 0]):.3f}"
            )
            print("      - Supports strong-like interaction analogy")
            print("      - Contributes to K_φ canonical promotion")
        elif overall_evidence_rate > 0.3:
            print("   ⚠️ MODERATE ASYMPTOTIC FREEDOM EVIDENCE")
            print(
                "      - " f"{overall_evidence_rate:.1%} of experiments show "
                "power law behavior"
            )
            print("      - Evidence varies by topology")
            print("      - May support qualified canonical status")
        else:
            print("   ❌ WEAK ASYMPTOTIC FREEDOM EVIDENCE")
            print(
                "      - Only "
                f"{overall_evidence_rate:.1%} show clear power law"
            )
            print("      - K_φ may not exhibit strong-like scale dependence")
            print("      - Challenges canonical promotion pathway")
        
        # Recommendations
        if strong_evidence > len(results) * 0.4:
            print("\n🚀 RECOMMENDATIONS:")
            print("   1. Proceed with K_φ canonical promotion")
            print("   2. Document asymptotic freedom as key evidence")
            print("   3. Include scale-dependent analysis in safety criteria")
        else:
            print("\n🔄 RECOMMENDATIONS:")
            print("   1. Investigate topology-specific scaling laws")
            print("   2. Test alternative scale-dependent metrics")
            print("   3. Consider ensemble averaging for cleaner signals")


def apply_dynamics_sequence(G):
    """Apply a sequence of operators to create realistic K_φ distribution."""
    # Random selection of nodes for dynamics
    dynamics_nodes = random.sample(list(G.nodes()), len(G.nodes())//3)
    
    operations_applied = []
    
    for node in dynamics_nodes:
        if random.random() < 0.6:  # 60% get dissonance
            dissonance = Dissonance()
            dissonance(G, node)
            operations_applied.append(('dissonance', node))
        
        if random.random() < 0.3:  # 30% get coherence
            coherence = Coherence()
            coherence(G, node)
            operations_applied.append(('coherence', node))
    
    return {
        'n_operations': len(operations_applied),
        'dynamics_nodes': len(dynamics_nodes),
        'operation_types': [op[0] for op in operations_applied]
    }


def analyze_multiscale_variance(G, k_phi, scales):
    """Compute K_φ variance at different neighborhood scales."""
    scale_analysis = []
    
    for r in scales:
        scale_k_phi = {}
        
        for node in G.nodes():
            try:
                # Get r-hop ego network
                ego_graph = nx.ego_graph(G, node, radius=r)
                ego_nodes = list(ego_graph.nodes())
                
                # Compute coarse-grained K_φ (average over r-hop neighborhood)
                if ego_nodes:
                    ego_k_phi_values = [
                        k_phi[n] for n in ego_nodes if n in k_phi
                    ]
                    if ego_k_phi_values:
                        scale_k_phi[node] = np.mean(ego_k_phi_values)
                    else:
                        scale_k_phi[node] = 0.0
                else:
                    scale_k_phi[node] = 0.0
                    
            except Exception:
                scale_k_phi[node] = 0.0
        
        # Compute variance at this scale
        k_phi_values = list(scale_k_phi.values())
        variance = np.var(k_phi_values) if k_phi_values else 0.0
        
        scale_analysis.append({
            'scale': r,
            'variance': variance,
            'n_nodes_analyzed': len(k_phi_values),
            'mean_k_phi': (
                np.mean([abs(k) for k in k_phi_values])
                if k_phi_values else 0.0
            )
        })
    
    return scale_analysis


def fit_asymptotic_freedom_law(scale_analysis):
    """Fit power law: var(K_φ) ~ 1/r^α to scale analysis data."""
    # Extract scales and variances
    scales = [s['scale'] for s in scale_analysis if s['variance'] > 1e-10]
    variances = [
        s['variance'] for s in scale_analysis if s['variance'] > 1e-10
    ]
    
    if len(scales) < 3:  # Need at least 3 points for meaningful fit
        return {
            'alpha': 0.0,
            'r_squared': 0.0,
            'fit_quality': 'insufficient_data',
            'n_points': len(scales)
        }
    
    try:
        # Transform to log-log space: log(var) = log(A) - α * log(r)
        log_scales = np.log(scales)
        log_variances = np.log(variances)
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales, log_variances
        )
        
        # α is the negative of slope (var ~ 1/r^α means log(var) ~ -α*log(r))
        alpha = -slope
        r_squared = r_value**2
        
        if r_squared > 0.8:
            fit_quality = 'excellent'
        elif r_squared > 0.6:
            fit_quality = 'good'
        else:
            fit_quality = 'poor'
        
        return {
            'alpha': alpha,
            'r_squared': r_squared,
            'fit_quality': fit_quality,
            'n_points': len(scales),
            'p_value': p_value,
            'std_error': std_err
        }
        
    except Exception as e:
        return {
            'alpha': 0.0,
            'r_squared': 0.0,
            'fit_quality': f'fit_failed_{str(e)[:20]}',
            'n_points': len(scales)
        }


def convert_numpy_types(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K_phi asymptotic freedom benchmark (variance scaling)"
    )
    parser.add_argument(
        "--topologies", nargs="+", default=["ring", "scale_free"],
        help="Topologies to test (default: ring scale_free)"
    )
    parser.add_argument(
        "--nodes", type=int, default=50,
        help="Nodes per topology (default: 50)"
    )
    parser.add_argument(
        "--seeds", type=int, default=8,
        help="Number of seed runs per topology (default: 8)"
    )
    parser.add_argument(
        "--high-resolution", action="store_true",
        help="Enable extended hop distances"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base RNG seed"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    cli_args = parser.parse_args()
    asymptotic_freedom_investigation(
        topologies=cli_args.topologies,
        n_nodes=cli_args.nodes,
        n_tests=cli_args.seeds,
        high_resolution=cli_args.high_resolution,
        seed=cli_args.seed,
        quiet=cli_args.quiet,
    )
