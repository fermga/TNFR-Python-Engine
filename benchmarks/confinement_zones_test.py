#!/usr/bin/env python3
"""
K_φ Confinement Zone Mapping Investigation

Test Task 2: Identify high |K_φ| zones as confinement regions and measure
ΔNFR localization dynamics during operator sequences.
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import networkx as nx


def _convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python for JSON serialization.

    Ensures json.dumps does not raise TypeError for np.integer, np.floating,
    or ndarray objects. Leaves other types unchanged.
    """
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


def confinement_zone_investigation():
    """Investigate K_φ confinement zones and ΔNFR localization."""
    # Lazy project imports to satisfy linter and ensure sys.path is set
    PROJECT_ROOT = Path(__file__).parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.tnfr.physics.fields import compute_phase_curvature  # noqa: E402
    from benchmarks.benchmark_utils import (  # noqa: E402
        create_tnfr_topology,
        initialize_tnfr_nodes,
    )
    from src.tnfr.operators.definitions import (  # noqa: E402
        Dissonance,
        Coherence,
        Mutation,
    )
    from src.tnfr.config import DNFR_PRIMARY  # noqa: E402
    print("🔒 K_φ Confinement Zone Mapping Investigation")
    print("=" * 50)
    
    results = []
    topologies = ['ring', 'scale_free', 'ws']
    n_nodes = 30
    n_tests = 10
    
    # Different K_φ thresholds to test
    k_phi_thresholds = [3.0, 4.0, 4.88, 5.5, 6.0]
    
    for topology in topologies:
        print(f"\n🌊 {topology.upper()} - Confinement Zone Analysis:")
        
        for test_id in range(n_tests):
            seed = random.randint(1000, 9999)
            
            try:
                # Create and initialize graph
                G = create_tnfr_topology(topology, n_nodes, seed)
                initialize_tnfr_nodes(G, seed=seed)
                
                print(f"  Test {test_id:2d}: ", end="")
                
                # === PHASE 1: Pre-disruption baseline ===
                k_phi_baseline = compute_phase_curvature(G)
                dnfr_baseline = {
                    node: G.nodes[node][DNFR_PRIMARY]
                    for node in G.nodes()
                }
                
                baseline_stats = analyze_field_distribution(
                    k_phi_baseline, dnfr_baseline, "Baseline"
                )
                
                # === PHASE 2: Apply disruption sequence ===
                # Target nodes with various K_φ levels
                disruption_targets = select_disruption_targets(
                    G, k_phi_baseline
                )
                
                for target_node in disruption_targets:
                    # Apply dissonance burst
                    dissonance = Dissonance()
                    dissonance(G, target_node)
                    
                    # Random mutation
                    if random.random() < 0.4:
                        mutation = Mutation()
                        mutation(G, target_node)
                
                # === PHASE 3: Post-disruption analysis ===
                k_phi_disrupted = compute_phase_curvature(G)
                dnfr_disrupted = {
                    node: G.nodes[node][DNFR_PRIMARY]
                    for node in G.nodes()
                }
                
                disrupted_stats = analyze_field_distribution(
                    k_phi_disrupted, dnfr_disrupted, "Disrupted"
                )
                
                # === PHASE 4: Confinement zone analysis ===
                confinement_analysis = {}
                
                for threshold in k_phi_thresholds:
                    zones = identify_confinement_zones(
                        G, k_phi_disrupted, threshold
                    )
                    
                    dnfr_capture = measure_dnfr_localization(
                        G, dnfr_disrupted, zones
                    )
                    
                    confinement_analysis[threshold] = {
                        'n_zones': len(zones),
                        'zone_sizes': [len(zone) for zone in zones],
                        'total_confined_nodes': sum(
                            len(zone) for zone in zones
                        ),
                        'dnfr_capture_rate': dnfr_capture,
                        'zone_connectivity': analyze_zone_connectivity(
                            G, zones
                        )
                    }
                
                # === PHASE 5: Apply stabilization ===
                coherence = Coherence()
                for node in G.nodes():
                    if random.random() < 0.3:  # Stabilize 30% of nodes
                        coherence(G, node)
                
                k_phi_stabilized = compute_phase_curvature(G)
                dnfr_stabilized = {
                    node: G.nodes[node][DNFR_PRIMARY]
                    for node in G.nodes()
                }
                
                stabilized_stats = analyze_field_distribution(
                    k_phi_stabilized, dnfr_stabilized, "Stabilized"
                )
                
                # === RESULTS COMPILATION ===
                result = {
                    'topology': topology,
                    'test_id': test_id,
                    'n_nodes': len(G.nodes()),
                    'n_edges': len(G.edges()),
                    'seed': seed,
                    'baseline_stats': baseline_stats,
                    'disrupted_stats': disrupted_stats,
                    'stabilized_stats': stabilized_stats,
                    'confinement_analysis': confinement_analysis,
                    'disruption_targets': disruption_targets
                }
                
                results.append(result)
                
                # Real-time summary
                best_threshold = max(
                    confinement_analysis.keys(),
                    key=lambda t: confinement_analysis[t]['dnfr_capture_rate']
                )
                best_capture = confinement_analysis[best_threshold][
                    'dnfr_capture_rate'
                ]
                best_zones = confinement_analysis[best_threshold]['n_zones']
                
                k_phi_evolution = {
                    'baseline_max': baseline_stats['k_phi_max'],
                    'disrupted_max': disrupted_stats['k_phi_max'],
                    'stabilized_max': stabilized_stats['k_phi_max']
                }
                
                print(f"K_φ: {k_phi_evolution['baseline_max']:.1f}→"
                      f"{k_phi_evolution['disrupted_max']:.1f}→"
                      f"{k_phi_evolution['stabilized_max']:.1f} | "
                      f"Zones: {best_zones} | Capture: {best_capture:.2f}")
                      
            except Exception as e:
                print(f"ERROR: {e}")
                continue
    
    # === COMPREHENSIVE ANALYSIS ===
    if results:
        print(f"\n📊 CONFINEMENT ZONE ANALYSIS SUMMARY:")
        print(f"Total experiments: {len(results)}")
        
        # Threshold performance analysis
        print("\n🎯 Threshold Performance (ΔNFR Capture Rates):")
        print(
            f"{'Threshold':<10} {'Mean':<8} {'Std':<8} {'Max':<8} {'>75%':<6}"
        )
        print("-" * 50)
        
        for threshold in k_phi_thresholds:
            capture_rates = []
            for result in results:
                if threshold in result['confinement_analysis']:
                    capture_rates.append(
                        result['confinement_analysis'][threshold][
                            'dnfr_capture_rate'
                        ]
                    )
            
            if capture_rates:
                mean_capture = np.mean(capture_rates)
                std_capture = np.std(capture_rates)
                max_capture = np.max(capture_rates)
                high_capture_count = sum(1 for r in capture_rates if r > 0.75)
                
                print(
                    f"{threshold:<10.1f} {mean_capture:<8.3f} "
                    f"{std_capture:<8.3f} {max_capture:<8.3f} {high_capture_count:<6d}"
                )
        
        # Zone dynamics analysis
        print("\n🌊 Zone Dynamics Across Phases:")
        phases = ['baseline', 'disrupted', 'stabilized']

        print(
            f"{'Phase':<12} {'K_φ Mean':<10} {'K_φ Max':<10} {'ΔNFR Mean':<12}"
        )
        print("-" * 50)
        
        for phase in phases:
            k_phi_means = []
            k_phi_maxs = []
            dnfr_means = []
            
            for result in results:
                stats = result[f'{phase}_stats']
                k_phi_means.append(stats['k_phi_mean'])
                k_phi_maxs.append(stats['k_phi_max'])
                dnfr_means.append(stats['dnfr_mean'])
            
            print(
                f"{phase.title():<12} {np.mean(k_phi_means):<10.3f} "
                f"{np.mean(k_phi_maxs):<10.3f} {np.mean(dnfr_means):<12.6f}"
            )
        
        # Topology comparison
        print("\n🗺️ Topology-Specific Confinement Patterns:")
        for topology in topologies:
            topo_results = [r for r in results if r['topology'] == topology]
            
            if topo_results:
                # Best threshold for this topology
                best_capture_by_threshold = {}
                for threshold in k_phi_thresholds:
                    captures = []
                    for result in topo_results:
                        if threshold in result['confinement_analysis']:
                            captures.append(
                                result['confinement_analysis'][threshold][
                                    'dnfr_capture_rate'
                                ]
                            )
                    if captures:
                        best_capture_by_threshold[threshold] = np.mean(captures)
                
                if best_capture_by_threshold:
                    best_threshold = max(
                        best_capture_by_threshold.keys(),
                        key=lambda t: best_capture_by_threshold[t]
                    )
                    best_performance = best_capture_by_threshold[best_threshold]
                    
                    print(
                        f"   {topology:<12s}: Best threshold = {best_threshold:.1f} "
                        f"(capture = {best_performance:.3f})"
                    )
        
        # Save detailed results
        output_file = (
            PROJECT_ROOT
            / "benchmarks"
            / "results"
            / "confinement_zones_analysis.jsonl"
        )
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(_convert_numpy_types(result)) + '\n')
        
        print(f"\n💾 Detailed results saved to: {output_file}")

        # === CONCLUSIONS ===
        print("\n🔍 CONFINEMENT MECHANISM CONCLUSIONS:")
        
        # Check for strong confinement evidence
        strong_evidence_threshold = 0.75  # 75% ΔNFR capture
        strong_evidence_cases = 0
        
        for result in results:
            for threshold, analysis in result['confinement_analysis'].items():
                if analysis['dnfr_capture_rate'] > strong_evidence_threshold:
                    strong_evidence_cases += 1
        
        total_threshold_tests = len(results) * len(k_phi_thresholds)
        strong_evidence_rate = strong_evidence_cases / total_threshold_tests
        
        print(
            f"\n   Strong Confinement Evidence: {strong_evidence_cases}/"
            f"{total_threshold_tests} ({strong_evidence_rate:.1%})"
        )
        
        # Moderate evidence band (partial confinement): capture > 0.20
        moderate_evidence_cases = 0
        for result in results:
            for threshold, analysis in result['confinement_analysis'].items():
                if analysis['dnfr_capture_rate'] > 0.20:
                    moderate_evidence_cases += 1
        moderate_rate = (
            moderate_evidence_cases / total_threshold_tests
            if total_threshold_tests > 0
            else 0.0
        )

        print(
            f"   Moderate Confinement Evidence (>20% capture): "
            f"{moderate_evidence_cases}/{total_threshold_tests} "
            f"({moderate_rate:.1%})"
        )

        if strong_evidence_rate > 0.2:  # 20% of cases show strong confinement
            print("   ✅ CONFINEMENT MECHANISM DETECTED")
            print("      - High |K_φ| zones successfully localize ΔNFR")
            print("      - Strong-like interaction regime validated")
            print("      - Supports canonical promotion pathway")
        elif strong_evidence_rate > 0.05:
            print("   ⚠️ WEAK CONFINEMENT EVIDENCE")
            print("      - Some localization observed but inconsistent")
            print("      - May need topology-specific thresholds")
            print("      - Requires further investigation")
        else:
            print("   ❌ NO CLEAR CONFINEMENT MECHANISM")
            print("      - ΔNFR remains distributed despite high |K_φ|")
            print("      - May not function as strong-like interaction")
            print("      - Consider alternative interpretations")


def analyze_field_distribution(k_phi_field, dnfr_field, phase_name):
    """Analyze statistical properties of K_φ and ΔNFR fields."""
    k_phi_values = list(k_phi_field.values())
    dnfr_values = list(dnfr_field.values())
    
    k_phi_abs = [abs(k) for k in k_phi_values]
    
    return {
        'phase': phase_name,
        'k_phi_mean': np.mean(k_phi_abs),
        'k_phi_max': np.max(k_phi_abs),
        'k_phi_std': np.std(k_phi_abs),
        'dnfr_mean': np.mean(dnfr_values),
        'dnfr_max': np.max(dnfr_values),
        'dnfr_std': np.std(dnfr_values),
        'n_nodes': len(k_phi_values)
    }


def select_disruption_targets(G, k_phi_field, n_targets=5):
    """Select nodes for disruption based on K_φ distribution."""
    # Target mix of high K_φ and random nodes
    k_phi_abs = {node: abs(k_phi_field[node]) for node in G.nodes()}
    
    # Top K_φ nodes
    sorted_by_k_phi = sorted(
        k_phi_abs.items(), key=lambda x: x[1], reverse=True
    )
    top_k_phi_nodes = [node for node, _ in sorted_by_k_phi[:n_targets//2]]
    
    # Random nodes
    random_nodes = random.sample(list(G.nodes()), n_targets//2)
    
    return list(set(top_k_phi_nodes + random_nodes))[:n_targets]


def identify_confinement_zones(G, k_phi_field, threshold):
    """Identify connected components of nodes with |K_φ| > threshold."""
    high_k_phi_nodes = [
        node for node, k_phi in k_phi_field.items()
        if abs(k_phi) > threshold
    ]
    
    if not high_k_phi_nodes:
        return []
    
    # Create subgraph of high K_φ nodes
    subgraph = G.subgraph(high_k_phi_nodes)
    
    # Find connected components (confinement zones)
    zones = [
        list(component) for component in nx.connected_components(subgraph)
    ]
    
    return zones


def measure_dnfr_localization(G, dnfr_field, zones):
    """Measure what fraction of total ΔNFR is captured within zones."""
    if not zones:
        return 0.0
    
    confined_nodes = set()
    for zone in zones:
        confined_nodes.update(zone)
    
    # Total ΔNFR in system
    total_dnfr = sum(abs(dnfr_field[node]) for node in G.nodes())
    
    # ΔNFR within confinement zones
    confined_dnfr = sum(
        abs(dnfr_field[node]) for node in confined_nodes if node in dnfr_field
    )
    
    if total_dnfr == 0:
        return 0.0
    
    return confined_dnfr / total_dnfr


def analyze_zone_connectivity(G, zones):
    """Analyze connectivity properties of confinement zones."""
    if not zones:
        return {'avg_zone_size': 0, 'max_zone_size': 0, 'total_zones': 0}
    
    zone_sizes = [len(zone) for zone in zones]
    
    return {
        'avg_zone_size': np.mean(zone_sizes),
        'max_zone_size': np.max(zone_sizes),
        'total_zones': len(zones),
        'size_distribution': zone_sizes
    }


if __name__ == "__main__":
    confinement_zone_investigation()
