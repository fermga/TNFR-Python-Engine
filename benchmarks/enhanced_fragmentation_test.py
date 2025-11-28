#!/usr/bin/env python3
"""
Enhanced Fragmentation Testing for K_φ Critical Threshold Validation

Task 1: Use aggressive multi-burst operator sequences to achieve strong
fragmentation and validate |K_φ| > 4.88 as universal threshold with ≥90%
accuracy.

Strategy: Apply cascading dissonance/expansion/mutation bursts to drive |K_φ|
beyond literature threshold and measure true/false positive rates.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tnfr.physics.fields import compute_phase_curvature  # noqa: E402
from benchmarks.benchmark_utils import (  # noqa: E402
    create_tnfr_topology,
    initialize_tnfr_nodes
)
from src.tnfr.operators.definitions import (  # noqa: E402
    Dissonance,
    Mutation,
    Expansion
)
from src.tnfr.config import DNFR_PRIMARY  # noqa: E402


def enhanced_fragmentation_investigation():
    """Aggressive fragmentation test with multi-burst sequences."""
    print("💥 Enhanced K_φ Fragmentation Investigation")
    print("=" * 45)
    
    results = []
    topologies = ['ring', 'scale_free', 'ws', 'tree']
    n_nodes = 25  # Smaller for concentrated effects
    n_tests = 15   # More tests for statistical power
    
    # Test multiple thresholds including literature value
    thresholds = [3.0, 4.0, 4.88, 5.5, 6.0]
    
    # Fragmentation criteria
    fragmentation_criteria = {
        'coherence_drop': 0.3,      # 30% coherence loss
        'dnfr_spike': 0.1,          # 10% ΔNFR increase
        'network_disconnect': 0.1,   # 10% edge loss
        'phase_chaos': 2.0          # Phase variance > 2.0
    }
    
    for topology in topologies:
        print(f"\n🌪️ {topology.upper()} - Enhanced Fragmentation:")
        
        for test_id in range(n_tests):
            seed = random.randint(1000, 9999)
            
            try:
                # Create and initialize graph
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, seed=seed)
                
                print(f"  Test {test_id:2d}: ", end="")
                
                # === BASELINE MEASUREMENT ===
                baseline_metrics = capture_fragmentation_metrics(G)
                k_phi_baseline = compute_phase_curvature(G)
                baseline_max_k_phi = max(
                    abs(k) for k in k_phi_baseline.values()
                )
                
                # === AGGRESSIVE FRAGMENTATION SEQUENCE ===
                fragmentation_applied = apply_aggressive_fragmentation(
                    G, intensity='extreme', target_k_phi=6.0
                )
                
                # === POST-FRAGMENTATION MEASUREMENT ===
                post_metrics = capture_fragmentation_metrics(G)
                k_phi_post = compute_phase_curvature(G)
                post_max_k_phi = max(abs(k) for k in k_phi_post.values())
                
                # === FRAGMENTATION DETECTION ===
                fragmentation_detected = detect_fragmentation(
                    baseline_metrics, post_metrics, fragmentation_criteria
                )
                
                # === THRESHOLD ANALYSIS ===
                threshold_analysis = {}
                for threshold in thresholds:
                    # True positive: fragmentation + |K_φ| > threshold
                    # False positive: no fragmentation + |K_φ| > threshold
                    # True negative: no fragmentation + |K_φ| <= threshold
                    # False negative: fragmentation + |K_φ| <= threshold
                    
                    k_phi_exceeds = post_max_k_phi > threshold
                    
                    if fragmentation_detected and k_phi_exceeds:
                        classification = 'true_positive'
                    elif fragmentation_detected and not k_phi_exceeds:
                        classification = 'false_negative'
                    elif not fragmentation_detected and k_phi_exceeds:
                        classification = 'false_positive'
                    else:
                        classification = 'true_negative'
                    
                    threshold_analysis[threshold] = {
                        'classification': classification,
                        'k_phi_exceeds': k_phi_exceeds,
                        'fragmentation_detected': fragmentation_detected
                    }
                
                # === RESULT COMPILATION ===
                result = {
                    'topology': topology,
                    'test_id': test_id,
                    'n_nodes': len(G.nodes()),
                    'n_edges': len(G.edges()),
                    'seed': seed,
                    'baseline_metrics': baseline_metrics,
                    'post_metrics': post_metrics,
                    'k_phi_baseline_max': baseline_max_k_phi,
                    'k_phi_post_max': post_max_k_phi,
                    'k_phi_increase': post_max_k_phi - baseline_max_k_phi,
                    'fragmentation_detected': fragmentation_detected,
                    'fragmentation_applied': fragmentation_applied,
                    'threshold_analysis': threshold_analysis,
                    'fragmentation_criteria': fragmentation_criteria
                }
                
                results.append(result)
                
                # Real-time feedback
                frag_symbol = "💥" if fragmentation_detected else "✅"
                critical_threshold = 4.88
                critical_analysis = threshold_analysis[critical_threshold]
                accuracy_symbol = (
                    "✅" if critical_analysis['classification']
                    in ['true_positive', 'true_negative']
                    else "❌"
                )
                
                print(f"K_φ: {baseline_max_k_phi:.1f}→{post_max_k_phi:.1f} | "
                      f"Frag: {frag_symbol} | 4.88: {accuracy_symbol}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # === COMPREHENSIVE THRESHOLD ANALYSIS ===
    if results:
        print("\n📊 ENHANCED FRAGMENTATION ANALYSIS SUMMARY:")
        print(f"Total experiments: {len(results)}")
        
        # Overall fragmentation success
        total_fragmentation = sum(
            1 for r in results if r['fragmentation_detected']
        )
        fragmentation_rate = total_fragmentation / len(results)
        
        print("\n💥 Fragmentation Achievement:")
        print(
            f"   Successful fragmentations: {total_fragmentation}/"
            f"{len(results)} ({fragmentation_rate:.1%})"
        )
        
        # K_φ elevation analysis
        k_phi_increases = [r['k_phi_increase'] for r in results]
        mean_increase = np.mean(k_phi_increases)
        max_increase = np.max(k_phi_increases)
        exceeded_488 = sum(1 for r in results if r['k_phi_post_max'] > 4.88)
        
        print("\n📈 K_φ Elevation:")
        print(f"   Mean increase: {mean_increase:.2f}")
        print(f"   Max increase: {max_increase:.2f}")
        print(f"   Exceeded 4.88: {exceeded_488}")
        
        # Threshold accuracy analysis
        print("\n🎯 Threshold Accuracy Analysis:")
        print(
            f"{'Threshold':<10} {'Accuracy':<8} {'Precision':<10} "
            f"{'Recall':<8} {'F1':<8}"
        )
        print("-" * 55)
        
        for threshold in thresholds:
            # Calculate confusion matrix
            tp = sum(
                1 for r in results
                if r['threshold_analysis'][threshold]['classification']
                == 'true_positive'
            )
            fp = sum(
                1 for r in results
                if r['threshold_analysis'][threshold]['classification']
                == 'false_positive'
            )
            tn = sum(
                1 for r in results
                if r['threshold_analysis'][threshold]['classification']
                == 'true_negative'
            )
            fn = sum(
                1 for r in results
                if r['threshold_analysis'][threshold]['classification']
                == 'false_negative'
            )
            
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            
            print(
                f"{threshold:<10.1f} {accuracy:<8.3f} {precision:<10.3f} "
                f"{recall:<8.3f} {f1:<8.3f}"
            )
        
        # Topology-specific performance
        print(f"\n🗺️ Topology-Specific Performance:")
        print(f"{'Topology':<12} {'Frag Rate':<10} {'K_φ Max':<8} {'4.88 Acc':<8}")
        print("-" * 45)
        
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            
            if topo_results:
                topo_frag_rate = sum(1 for r in topo_results if r['fragmentation_detected']) / len(topo_results)
                topo_max_k_phi = max(r['k_phi_post_max'] for r in topo_results)
                
                # Accuracy for critical threshold 4.88
                critical_threshold = 4.88
                topo_tp = sum(1 for r in topo_results if r['threshold_analysis'][critical_threshold]['classification'] == 'true_positive')
                topo_tn = sum(1 for r in topo_results if r['threshold_analysis'][critical_threshold]['classification'] == 'true_negative')
                topo_accuracy = (topo_tp + topo_tn) / len(topo_results)
                
                print(f"{topology:<12} {topo_frag_rate:<10.3f} {topo_max_k_phi:<8.2f} {topo_accuracy:<8.3f}")
        
        # Save results with numpy conversion
        output_file = PROJECT_ROOT / "benchmarks" / "results" / "enhanced_fragmentation_analysis.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(_convert_numpy_types(result)) + '\n')
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # === CRITICAL THRESHOLD CONCLUSIONS ===
        critical_threshold = 4.88
        critical_results = [r['threshold_analysis'][critical_threshold] for r in results]
        critical_tp = sum(1 for r in critical_results if r['classification'] == 'true_positive')
        critical_tn = sum(1 for r in critical_results if r['classification'] == 'true_negative')
        critical_accuracy = (critical_tp + critical_tn) / len(results)
        
        print(f"\n🔍 CRITICAL THRESHOLD (4.88) CONCLUSIONS:")
        print(f"   Accuracy: {critical_accuracy:.1%} (target: ≥90%)")
        
        if critical_accuracy >= 0.9:
            print("   ✅ THRESHOLD VALIDATION ACHIEVED")
            print("      - |K_φ| > 4.88 successfully predicts fragmentation")
            print("      - Literature threshold confirmed in TNFR implementation")
            print("      - Ready for canonical promotion")
        elif critical_accuracy >= 0.7:
            print("   ⚠️ MODERATE THRESHOLD VALIDATION")
            print(f"      - {critical_accuracy:.1%} accuracy achieved")
            print("      - May require topology-specific calibration")
            print("      - Consider alternative thresholds")
        else:
            print("   ❌ THRESHOLD VALIDATION FAILED")
            print(f"      - Only {critical_accuracy:.1%} accuracy achieved")
            print("      - Literature threshold may not apply to TNFR")
            print("      - Investigate alternative fragmentation indicators")
        
        # Alternative threshold recommendations
        best_threshold = None
        best_accuracy = 0.0
        
        for threshold in thresholds:
            threshold_results = [r['threshold_analysis'][threshold] for r in results]
            tp = sum(1 for r in threshold_results if r['classification'] == 'true_positive')
            tn = sum(1 for r in threshold_results if r['classification'] == 'true_negative')
            accuracy = (tp + tn) / len(results)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        if best_threshold != critical_threshold:
            print(f"\n🎯 ALTERNATIVE THRESHOLD RECOMMENDATION:")
            print(f"   Best performing threshold: {best_threshold:.1f}")
            print(f"   Accuracy: {best_accuracy:.1%}")
            print(f"   Consider using {best_threshold:.1f} instead of literature 4.88")


def apply_aggressive_fragmentation(G, intensity='extreme', target_k_phi=6.0):
    """Apply cascading fragmentation sequences to drive high |K_φ|."""
    operators_applied = []
    
    # Phase 1: Widespread dissonance burst
    dissonance_targets = random.sample(list(G.nodes()), min(8, len(G.nodes())))
    for node in dissonance_targets:
        for _ in range(3):  # Triple dissonance burst
            Dissonance()(G, node)
            operators_applied.append(('dissonance', node))
    
    # Phase 2: Strategic expansion (increase complexity)
    expansion_targets = random.sample(list(G.nodes()), min(5, len(G.nodes())))
    for node in expansion_targets:
        for _ in range(2):  # Double expansion
            Expansion()(G, node)
            operators_applied.append(('expansion', node))
    
    # Phase 3: Mutation cascade (phase disruption)
    mutation_targets = random.sample(list(G.nodes()), min(6, len(G.nodes())))
    for node in mutation_targets:
        Mutation()(G, node)
        operators_applied.append(('mutation', node))
    
    # Phase 4: Check and amplify if needed
    k_phi = compute_phase_curvature(G)
    max_k_phi = max(abs(k) for k in k_phi.values())
    
    if max_k_phi < target_k_phi and intensity == 'extreme':
        # Additional amplification
        high_k_phi_nodes = [node for node, k in k_phi.items() if abs(k) > np.mean(list(k_phi.values()))]
        for node in high_k_phi_nodes[:3]:
            for _ in range(2):
                Dissonance()(G, node)
                operators_applied.append(('amplification_dissonance', node))
    
    return {
        'total_operators': len(operators_applied),
        'operator_breakdown': {
            'dissonance': len([op for op in operators_applied if op[0] in ['dissonance', 'amplification_dissonance']]),
            'expansion': len([op for op in operators_applied if op[0] == 'expansion']),
            'mutation': len([op for op in operators_applied if op[0] == 'mutation'])
        },
        'intensity': intensity,
        'target_k_phi': target_k_phi
    }


def capture_fragmentation_metrics(G):
    """Capture comprehensive fragmentation indicators."""
    from src.tnfr.metrics.common import compute_coherence
    from src.tnfr.physics.fields import compute_phase_gradient
    
    # Coherence
    coherence = compute_coherence(G)
    
    # ΔNFR statistics
    dnfr_values = [G.nodes[node][DNFR_PRIMARY] for node in G.nodes()]
    dnfr_mean = np.mean(dnfr_values)
    dnfr_std = np.std(dnfr_values)
    
    # Phase statistics
    phases = [G.nodes[node].get('theta', 0.0) for node in G.nodes()]
    phase_variance = np.var(phases)
    
    # Network connectivity
    n_edges = len(G.edges())
    n_nodes = len(G.nodes())
    connectivity = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0.0
    
    # Phase gradient
    grad_field = compute_phase_gradient(G)
    grad_mean = np.mean(list(grad_field.values()))
    
    return {
        'coherence': float(coherence),
        'dnfr_mean': float(dnfr_mean),
        'dnfr_std': float(dnfr_std),
        'phase_variance': float(phase_variance),
        'connectivity': float(connectivity),
        'phase_grad_mean': float(grad_mean),
        'n_nodes': n_nodes,
        'n_edges': n_edges
    }


def detect_fragmentation(baseline_metrics, post_metrics, criteria):
    """Detect fragmentation based on multiple criteria."""
    # Coherence drop
    coherence_drop = baseline_metrics['coherence'] - post_metrics['coherence']
    coherence_fragmentation = coherence_drop > criteria['coherence_drop']
    
    # ΔNFR spike
    dnfr_increase = post_metrics['dnfr_std'] - baseline_metrics['dnfr_std']
    dnfr_fragmentation = dnfr_increase > criteria['dnfr_spike']
    
    # Network disconnection
    connectivity_drop = baseline_metrics['connectivity'] - post_metrics['connectivity']
    network_fragmentation = connectivity_drop > criteria['network_disconnect']
    
    # Phase chaos
    phase_chaos = post_metrics['phase_variance'] > criteria['phase_chaos']
    
    # Fragmentation if ANY criterion met (inclusive OR)
    fragmentation_detected = (
        coherence_fragmentation or 
        dnfr_fragmentation or 
        network_fragmentation or 
        phase_chaos
    )
    
    return fragmentation_detected


def _convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_numpy_types(v) for v in obj)
    
    # NumPy scalar and array handling
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    enhanced_fragmentation_investigation()