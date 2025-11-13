#!/usr/bin/env python3
"""
Aggressive K_φ Fragmentation Test

Force fragmentation using high-intensity sequences to properly test
|K_φ| > 4.88 threshold hypothesis.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import compute_phase_curvature
from benchmarks.benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes
from src.tnfr.operators.definitions import Dissonance, Emission, Mutation


def force_fragmentation_test():
    """Test K_φ threshold with aggressive fragmentation sequences."""
    print("⚡ Aggressive K_φ Fragmentation Test")
    print("=" * 40)
    
    results = []
    topologies = ['ring', 'scale_free', 'tree', 'ws']
    n_nodes = 25
    n_tests = 15
    
    for topology in topologies:
        print(f"\n💥 {topology.upper()} - Aggressive Fragmentation:")
        
        for test_id in range(n_tests):
            seed = random.randint(1000, 9999)
            
            try:
                # Create graph
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, seed=seed)
                
                # Pre-fragmentation K_φ
                k_phi_pre = compute_phase_curvature(G)
                k_phi_pre_max = max(abs(k) for k in k_phi_pre.values())
                
                # AGGRESSIVE FRAGMENTATION SEQUENCE
                intensity_factor = random.uniform(2.0, 4.0)  # High intensity
                
                # Apply multiple dissonance bursts
                for burst in range(3):
                    # Random nodes for disruption
                    target_nodes = random.sample(list(G.nodes()), 
                                                min(5, len(G.nodes())))
                    
                    for node in target_nodes:
                        try:
                            # High-intensity dissonance
                            dissonance = Dissonance()
                            dissonance(G, node)
                            
                            # Random mutation for some nodes
                            if random.random() < 0.3:  # 30% chance
                                mutation = Mutation()
                                mutation(G, node)
                        
                        except Exception:
                            pass  # Skip failed operations
                
                # Post-fragmentation metrics
                k_phi_post = compute_phase_curvature(G)
                k_phi_post_max = max(abs(k) for k in k_phi_post.values())
                
                # Multiple fragmentation criteria
                components = list(nx.connected_components(G))
                largest_comp = max(len(c) for c in components)
                
                # Fragmentation indicators
                coherence_ratio = largest_comp / len(G.nodes())
                n_components = len(components)
                isolation_ratio = (n_components - 1) / len(G.nodes())
                
                # Multiple fragmentation thresholds
                frag_loose = coherence_ratio < 0.8    # Loose criterion
                frag_medium = coherence_ratio < 0.6   # Medium criterion  
                frag_strict = coherence_ratio < 0.4   # Strict criterion
                frag_severe = n_components > 3        # Component count
                
                result = {
                    'topology': topology,
                    'test_id': test_id,
                    'k_phi_pre_max': k_phi_pre_max,
                    'k_phi_post_max': k_phi_post_max,
                    'k_phi_delta': k_phi_post_max - k_phi_pre_max,
                    'coherence_ratio': coherence_ratio,
                    'n_components': n_components,
                    'isolation_ratio': isolation_ratio,
                    'frag_loose': frag_loose,
                    'frag_medium': frag_medium, 
                    'frag_strict': frag_strict,
                    'frag_severe': frag_severe,
                    'intensity_factor': intensity_factor,
                    'seed': seed
                }
                
                results.append(result)
                
                # Status indicators
                frag_indicators = []
                if frag_loose: frag_indicators.append("L")
                if frag_medium: frag_indicators.append("M")
                if frag_strict: frag_indicators.append("S")
                if frag_severe: frag_indicators.append("C")
                
                frag_status = "".join(frag_indicators) if frag_indicators else "OK"
                
                print(f"  T{test_id:2d}: K_φ {k_phi_pre_max:.2f}→{k_phi_post_max:.2f} "
                      f"coherence={coherence_ratio:.2f} comp={n_components} [{frag_status}]")
                      
            except Exception as e:
                print(f"  T{test_id:2d}: ERROR - {e}")
    
    # Comprehensive analysis
    if results:
        print(f"\n📊 COMPREHENSIVE ANALYSIS:")
        print(f"Total experiments: {len(results)}")
        
        # Test thresholds against all fragmentation criteria
        thresholds_to_test = [4.0, 4.5, 4.88, 5.0, 5.5]
        criteria = ['frag_loose', 'frag_medium', 'frag_strict', 'frag_severe']
        
        print(f"\n🎯 Threshold Performance Matrix:")
        print(f"{'Threshold':<10} {'Criterion':<12} {'Accuracy':<8} {'TP':<3} {'FP':<3} {'TN':<3} {'FN':<3}")
        print("-" * 55)
        
        for threshold in thresholds_to_test:
            for criterion in criteria:
                k_phi_values = [r['k_phi_post_max'] for r in results]
                actual_frag = [r[criterion] for r in results]
                
                predictions = [k > threshold for k in k_phi_values]
                
                tp = sum(p and a for p, a in zip(predictions, actual_frag))
                fp = sum(p and not a for p, a in zip(predictions, actual_frag))
                tn = sum(not p and not a for p, a in zip(predictions, actual_frag))
                fn = sum(not p and a for p, a in zip(predictions, actual_frag))
                
                accuracy = (tp + tn) / len(results) if len(results) > 0 else 0
                
                print(f"{threshold:<10.1f} {criterion:<12s} {accuracy:<8.3f} "
                      f"{tp:<3d} {fp:<3d} {tn:<3d} {fn:<3d}")
        
        # Best threshold per criterion
        print(f"\n🏆 Best Thresholds:")
        for criterion in criteria:
            best_acc = 0
            best_threshold = None
            
            for threshold in thresholds_to_test:
                k_phi_values = [r['k_phi_post_max'] for r in results]
                actual_frag = [r[criterion] for r in results]
                predictions = [k > threshold for k in k_phi_values]
                
                if len(set(actual_frag)) > 1:  # Need variation
                    tp = sum(p and a for p, a in zip(predictions, actual_frag))
                    tn = sum(not p and not a for p, a in zip(predictions, actual_frag))
                    acc = (tp + tn) / len(results)
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_threshold = threshold
            
            frag_rate = np.mean([r[criterion] for r in results])
            print(f"   {criterion:<12s}: threshold={best_threshold} "
                  f"acc={best_acc:.3f} frag_rate={frag_rate:.3f}")
        
        # K_φ delta analysis
        k_phi_deltas = [r['k_phi_delta'] for r in results]
        print(f"\n📈 K_φ Evolution:")
        print(f"   Δ|K_φ| mean: {np.mean(k_phi_deltas):.3f} ± {np.std(k_phi_deltas):.3f}")
        print(f"   Δ|K_φ| range: [{np.min(k_phi_deltas):.2f}, {np.max(k_phi_deltas):.2f}]")
        
        # Topology comparison
        print(f"\n🗺️ Topology Fragmentation Rates:")
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            if topo_results:
                rates = {}
                for criterion in criteria:
                    rates[criterion] = np.mean([r[criterion] for r in topo_results])
                
                avg_k_phi = np.mean([r['k_phi_post_max'] for r in topo_results])
                print(f"   {topology:<12s}: K_φ={avg_k_phi:.2f} " +
                      " ".join([f"{c[5:]}={rates[c]:.2f}" for c in criteria]))
        
        # Save results
        output_file = PROJECT_ROOT / "benchmarks" / "results" / "aggressive_k_phi_test.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Conclusions
        print(f"\n🔍 CONCLUSIONS:")
        
        # Check if any fragmentation occurred
        any_fragmentation = any([
            any(r[criterion] for r in results)
            for criterion in criteria
        ])
        
        if not any_fragmentation:
            print("   ❌ NO FRAGMENTATION ACHIEVED")
            print("      - Sequences not aggressive enough")
            print("      - Need stronger disruption methods")
            print("      - Consider direct EPI manipulation for testing")
        else:
            # Find best performing threshold
            best_overall = 0
            best_threshold_overall = None
            
            for threshold in thresholds_to_test:
                total_accuracy = 0
                valid_criteria = 0
                
                for criterion in criteria:
                    actual_frag = [r[criterion] for r in results]
                    if len(set(actual_frag)) > 1:  # Has variation
                        k_phi_values = [r['k_phi_post_max'] for r in results]
                        predictions = [k > threshold for k in k_phi_values]
                        
                        tp = sum(p and a for p, a in zip(predictions, actual_frag))
                        tn = sum(not p and not a for p, a in zip(predictions, actual_frag))
                        acc = (tp + tn) / len(results)
                        
                        total_accuracy += acc
                        valid_criteria += 1
                
                if valid_criteria > 0:
                    avg_acc = total_accuracy / valid_criteria
                    if avg_acc > best_overall:
                        best_overall = avg_acc
                        best_threshold_overall = threshold
            
            if best_threshold_overall:
                print(f"   ✅ OPTIMAL THRESHOLD: {best_threshold_overall}")
                print(f"      Average accuracy: {best_overall:.3f}")
                
                if best_threshold_overall == 4.88:
                    print("      ⭐ MATCHES LITERATURE VALUE 4.88!")
                elif abs(best_threshold_overall - 4.88) < 0.5:
                    print(f"      ⚠️ Close to literature (Δ={abs(best_threshold_overall-4.88):.2f})")
                else:
                    print(f"      ❓ Differs from literature (Δ={abs(best_threshold_overall-4.88):.2f})")


if __name__ == "__main__":
    force_fragmentation_test()