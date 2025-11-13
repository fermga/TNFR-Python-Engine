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


def asymptotic_freedom_investigation():
    """Test scale-dependent K_φ variance for asymptotic freedom."""
    print("🌌 K_φ Asymptotic Freedom Investigation")
    print("=" * 45)
    
    results = []
    topologies = ['ring', 'scale_free', 'tree', 'ws']
    n_nodes = 50  # Larger networks for multi-scale analysis
    scales = [1, 2, 3, 4, 5, 7, 10]  # Hop distances
    n_tests = 8
    
    for topology in topologies:
        print(f"\n🔭 {topology.upper()} - Multi-Scale K_φ Analysis:")
        
        for test_id in range(n_tests):
            seed = random.randint(1000, 9999)
            
            try:
                # Create larger network
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, seed=seed)
                
                print(f"  Test {test_id}: ", end="")
                
                # Apply some dynamics for realistic K_φ distribution
                dynamic_state = apply_dynamics_sequence(G)
                
                # Compute K_φ field
                k_phi = compute_phase_curvature(G)
                
                # Multi-scale variance analysis
                scale_analysis = analyze_multiscale_variance(G, k_phi, scales)
                
                # Fit power law: var(K_φ) ~ 1/r^α
                power_law_fit = fit_asymptotic_freedom_law(scale_analysis)
                
                result = {
                    'topology': topology,
                    'test_id': test_id,
                    'n_nodes': len(G.nodes()),
                    'n_edges': len(G.edges()),
                    'seed': seed,
                    'dynamic_state': dynamic_state,
                    'scale_analysis': scale_analysis,
                    'power_law_fit': power_law_fit
                }
                
                results.append(result)
                
                # Real-time feedback
                alpha = power_law_fit['alpha']
                r_squared = power_law_fit['r_squared']
                evidence = "✅" if (alpha > 0 and r_squared > 0.5) else "❌"
                
                print(f"α={alpha:.2f} R²={r_squared:.3f} {evidence}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # === COMPREHENSIVE ANALYSIS ===
    if results:
        print(f"\n📊 ASYMPTOTIC FREEDOM ANALYSIS SUMMARY:")
        print(f"Total experiments: {len(results)}")
        
        # Power law statistics
        alphas = [r['power_law_fit']['alpha'] for r in results]
        r_squareds = [r['power_law_fit']['r_squared'] for r in results]
        
        positive_alphas = [a for a in alphas if a > 0]
        good_fits = [r for r in r_squareds if r > 0.5]
        
        print(f"\n🔬 Power Law Fit Statistics:")
        print(f"   Alpha (α) mean: {np.mean(alphas):.3f} ± {np.std(alphas):.3f}")
        print(f"   Positive α rate: {len(positive_alphas)}/{len(alphas)} "
              f"({100*len(positive_alphas)/len(alphas):.1f}%)")
        print(f"   Good fits (R²>0.5): {len(good_fits)}/{len(r_squareds)} "
              f"({100*len(good_fits)/len(r_squareds):.1f}%)")
        print(f"   Mean R²: {np.mean(r_squareds):.3f}")
        
        # Asymptotic freedom evidence classification
        strong_evidence = sum(1 for r in results 
                            if r['power_law_fit']['alpha'] > 0 
                            and r['power_law_fit']['r_squared'] > 0.7)
        
        moderate_evidence = sum(1 for r in results
                              if r['power_law_fit']['alpha'] > 0
                              and r['power_law_fit']['r_squared'] > 0.5)
        
        print(f"\n🎯 Asymptotic Freedom Evidence:")
        print(f"   Strong evidence (α>0, R²>0.7): {strong_evidence}/{len(results)} "
              f"({100*strong_evidence/len(results):.1f}%)")
        print(f"   Moderate evidence (α>0, R²>0.5): {moderate_evidence}/{len(results)} "
              f"({100*moderate_evidence/len(results):.1f}%)")
        
        # Topology-specific analysis
        print(f"\n🗺️ Topology-Specific Asymptotic Behavior:")
        print(f"{'Topology':<12} {'Mean α':<8} {'Mean R²':<8} {'Evidence':<12}")
        print("-" * 50)
        
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            
            if topo_results:
                topo_alphas = [r['power_law_fit']['alpha'] for r in topo_results]
                topo_r2s = [r['power_law_fit']['r_squared'] for r in topo_results]
                
                mean_alpha = np.mean(topo_alphas)
                mean_r2 = np.mean(topo_r2s)
                
                # Evidence classification for topology
                topo_evidence = sum(1 for r in topo_results 
                                  if r['power_law_fit']['alpha'] > 0 
                                  and r['power_law_fit']['r_squared'] > 0.5)
                
                evidence_rate = topo_evidence / len(topo_results)
                evidence_label = "Strong" if evidence_rate > 0.6 else "Weak" if evidence_rate > 0.3 else "None"
                
                print(f"{topology:<12} {mean_alpha:<8.3f} {mean_r2:<8.3f} {evidence_label:<12}")
        
        # Scale-dependent patterns
        print(f"\n📏 Scale-Dependent K_φ Variance Patterns:")
        
        # Aggregate variance by scale across all experiments
        all_scales = scales
        scale_variances = {scale: [] for scale in all_scales}
        
        for result in results:
            for scale_data in result['scale_analysis']:
                scale = scale_data['scale']
                if scale in scale_variances:
                    scale_variances[scale].append(scale_data['variance'])
        
        print(f"{'Scale (hops)':<12} {'Mean Var':<10} {'Std Var':<10} {'N_samples':<10}")
        print("-" * 50)
        
        for scale in all_scales:
            if scale_variances[scale]:
                mean_var = np.mean(scale_variances[scale])
                std_var = np.std(scale_variances[scale])
                n_samples = len(scale_variances[scale])
                
                print(f"{scale:<12} {mean_var:<10.4f} {std_var:<10.4f} {n_samples:<10}")
        
        # Save results
        output_file = PROJECT_ROOT / "benchmarks" / "results" / "asymptotic_freedom_analysis.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                # Convert numpy types to native Python for JSON serialization
                json_safe_result = convert_numpy_types(result)
                f.write(json.dumps(json_safe_result) + '\n')
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # === CONCLUSIONS ===
        print(f"\n🔍 ASYMPTOTIC FREEDOM CONCLUSIONS:")
        
        overall_evidence_rate = moderate_evidence / len(results)
        
        if overall_evidence_rate > 0.6:
            print(f"   ✅ STRONG ASYMPTOTIC FREEDOM EVIDENCE")
            print(f"      - {overall_evidence_rate:.1%} of experiments show var(K_φ) ~ 1/r^α")
            print(f"      - Mean α = {np.mean([a for a in alphas if a > 0]):.3f}")
            print(f"      - Supports strong-like interaction analogy")
            print(f"      - Contributes to K_φ canonical promotion")
        elif overall_evidence_rate > 0.3:
            print(f"   ⚠️ MODERATE ASYMPTOTIC FREEDOM EVIDENCE")  
            print(f"      - {overall_evidence_rate:.1%} of experiments show power law behavior")
            print(f"      - Evidence varies by topology")
            print(f"      - May support qualified canonical status")
        else:
            print(f"   ❌ WEAK ASYMPTOTIC FREEDOM EVIDENCE")
            print(f"      - Only {overall_evidence_rate:.1%} show clear power law")
            print(f"      - K_φ may not exhibit strong-like scale dependence") 
            print(f"      - Challenges canonical promotion pathway")
        
        # Recommendations
        if strong_evidence > len(results) * 0.4:
            print(f"\n🚀 RECOMMENDATIONS:")
            print(f"   1. Proceed with K_φ canonical promotion")
            print(f"   2. Document asymptotic freedom as key evidence")
            print(f"   3. Include scale-dependent analysis in safety criteria")
        else:
            print(f"\n🔄 RECOMMENDATIONS:")
            print(f"   1. Investigate topology-specific scaling laws")
            print(f"   2. Test alternative scale-dependent metrics")
            print(f"   3. Consider ensemble averaging for cleaner signals")


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
                    ego_k_phi_values = [k_phi[n] for n in ego_nodes if n in k_phi]
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
            'mean_k_phi': np.mean([abs(k) for k in k_phi_values]) if k_phi_values else 0.0
        })
    
    return scale_analysis


def fit_asymptotic_freedom_law(scale_analysis):
    """Fit power law: var(K_φ) ~ 1/r^α to scale analysis data."""
    # Extract scales and variances
    scales = [s['scale'] for s in scale_analysis if s['variance'] > 1e-10]
    variances = [s['variance'] for s in scale_analysis if s['variance'] > 1e-10]
    
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
        
        fit_quality = 'excellent' if r_squared > 0.8 else 'good' if r_squared > 0.6 else 'poor'
        
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
    """Recursively convert numpy types to native Python for JSON serialization."""
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
    asymptotic_freedom_investigation()