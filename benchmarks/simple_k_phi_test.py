#!/usr/bin/env python3
"""
Simplified Phase Curvature K_φ Investigation - Critical Threshold Validation

Quick proof-of-concept to test if |K_φ| ≈ 4.88 acts as fragmentation threshold.
"""

import sys
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import compute_phase_curvature
from benchmarks.benchmark_utils import (
    create_tnfr_topology,
    initialize_tnfr_nodes,
    generate_grammar_valid_sequence
)


def run_simple_k_phi_experiment():
    """Simple K_φ threshold test."""
    print("🔬 Simple Phase Curvature K_φ Threshold Investigation")
    print("=" * 55)
    
    # Test parameters
    topologies = ['ring', 'scale_free', 'tree', 'ws']
    n_nodes = 30
    n_tests = 20
    results = []
    
    for topology in topologies:
        print(f"\n📊 Testing {topology} topology:")
        
        for test_id in range(n_tests):
            seed = random.randint(1000, 9999)
            
            try:
                # Create and initialize graph
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, seed=seed)
                
                # Generate and apply sequence
                sequence_type = random.choice(['OZ_heavy', 'balanced'])
                intensity = 1.5 + np.random.uniform(-0.5, 0.5)  # Around critical
                
                operators = generate_grammar_valid_sequence(sequence_type, intensity)
                
                # Apply operators (simplified)
                for op in operators:
                    try:
                        op.apply(G)
                    except:
                        pass  # Skip failed ops
                
                # Compute K_φ metrics
                k_phi = compute_phase_curvature(G)
                k_phi_values = list(k_phi.values())
                
                if k_phi_values:
                    k_phi_max = max(abs(k) for k in k_phi_values)
                    k_phi_mean = np.mean([abs(k) for k in k_phi_values])
                    
                    # Simple fragmentation test (largest component ratio)
                    components = list(nx.connected_components(G))
                    largest_comp_size = max(len(c) for c in components)
                    coherence_ratio = largest_comp_size / len(G.nodes())
                    fragmented = coherence_ratio < 0.7  # Simple threshold
                    
                    result = {
                        'topology': topology,
                        'test_id': test_id,
                        'k_phi_max': k_phi_max,
                        'k_phi_mean': k_phi_mean,
                        'coherence_ratio': coherence_ratio,
                        'fragmented': fragmented,
                        'sequence_type': sequence_type,
                        'intensity': intensity,
                        'seed': seed
                    }
                    
                    results.append(result)
                    
                    # Real-time feedback
                    status = "FRAG" if fragmented else "OK"
                    print(f"  Test {test_id:2d}: |K_φ|_max={k_phi_max:.2f} "
                          f"coherence={coherence_ratio:.2f} [{status}]")
                
            except Exception as e:
                print(f"  Test {test_id:2d}: ERROR - {e}")
    
    # Analysis
    print(f"\n📈 Analysis Results:")
    print(f"Total experiments: {len(results)}")
    
    if results:
        # Test candidate threshold
        candidate_threshold = 4.88
        k_phi_maxes = [r['k_phi_max'] for r in results]
        fragmentation_flags = [r['fragmented'] for r in results]
        
        # Predictions using candidate threshold
        threshold_predictions = [k > candidate_threshold for k in k_phi_maxes]
        
        # Accuracy metrics
        correct_predictions = sum(pred == actual for pred, actual in 
                                zip(threshold_predictions, fragmentation_flags))
        accuracy = correct_predictions / len(results)
        
        # Breakdown by prediction type
        true_positives = sum(pred and actual for pred, actual in 
                           zip(threshold_predictions, fragmentation_flags))
        false_positives = sum(pred and not actual for pred, actual in 
                            zip(threshold_predictions, fragmentation_flags))
        false_negatives = sum(not pred and actual for pred, actual in 
                            zip(threshold_predictions, fragmentation_flags))
        true_negatives = sum(not pred and not actual for pred, actual in 
                           zip(threshold_predictions, fragmentation_flags))
        
        print(f"\n🎯 Threshold Analysis (|K_φ| > {candidate_threshold}):")
        print(f"   Overall Accuracy: {accuracy:.3f}")
        print(f"   True Positives:   {true_positives}")
        print(f"   False Positives:  {false_positives}")
        print(f"   True Negatives:   {true_negatives}")
        print(f"   False Negatives:  {false_negatives}")
        
        if true_positives + false_negatives > 0:
            sensitivity = true_positives / (true_positives + false_negatives)
            print(f"   Sensitivity:      {sensitivity:.3f}")
        
        if true_negatives + false_positives > 0:
            specificity = true_negatives / (true_negatives + false_positives)
            print(f"   Specificity:      {specificity:.3f}")
        
        # Statistics
        k_phi_max_mean = np.mean(k_phi_maxes)
        k_phi_max_std = np.std(k_phi_maxes)
        fragmentation_rate = np.mean(fragmentation_flags)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   |K_φ|_max mean:    {k_phi_max_mean:.3f} ± {k_phi_max_std:.3f}")
        print(f"   Fragmentation rate: {fragmentation_rate:.3f}")
        
        # Topology breakdown
        print(f"\n🗺️ Topology Breakdown:")
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            if topo_results:
                topo_k_phi = [r['k_phi_max'] for r in topo_results]
                topo_frag_rate = np.mean([r['fragmented'] for r in topo_results])
                topo_k_phi_mean = np.mean(topo_k_phi)
                
                # Topology-specific accuracy
                topo_predictions = [k > candidate_threshold for k in topo_k_phi]
                topo_actual = [r['fragmented'] for r in topo_results]
                topo_accuracy = np.mean([p == a for p, a in zip(topo_predictions, topo_actual)])
                
                print(f"   {topology:12s}: |K_φ|={topo_k_phi_mean:.2f} "
                      f"frag_rate={topo_frag_rate:.2f} acc={topo_accuracy:.2f}")
        
        # Save results
        output_file = PROJECT_ROOT / "benchmarks" / "results" / "simple_k_phi_test.jsonl"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Interpretation
        print(f"\n🔍 Preliminary Interpretation:")
        if accuracy >= 0.8:
            print(f"   ✅ Strong threshold evidence (acc={accuracy:.3f} ≥ 0.80)")
            print(f"      |K_φ| > {candidate_threshold} shows promising fragmentation prediction")
        elif accuracy >= 0.6:
            print(f"   ⚠️ Moderate threshold evidence (acc={accuracy:.3f})")
            print(f"      May need threshold optimization or topology-specific tuning")
        else:
            print(f"   ❌ Weak threshold evidence (acc={accuracy:.3f} < 0.60)")
            print(f"      Current threshold may not be universal predictor")
            
        # Next steps recommendation
        print(f"\n🚀 Recommended Next Steps:")
        if accuracy >= 0.7:
            print("   1. Scale up experiments (200+ per topology)")
            print("   2. Fine-tune threshold via ROC analysis")
            print("   3. Test confinement zone mapping")
        else:
            print("   1. Investigate alternative K_φ metrics (variance, peaks)")
            print("   2. Check topology-specific thresholds")
            print("   3. Validate fragmentation criteria")


if __name__ == "__main__":
    run_simple_k_phi_experiment()