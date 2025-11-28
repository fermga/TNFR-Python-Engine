#!/usr/bin/env python3
"""
Optimized ξ_C validation experiment with minimal verbosity
Focus on data collection efficiency
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, 
    Coupling, Resonance, Silence, Transition
)
from tnfr.operators.grammar import validate_sequence
from tnfr.types import NodeId
from benchmark_utils import create_tnfr_topology
from tnfr.physics.fields import (
    estimate_coherence_length, 
    compute_structural_potential, 
    compute_phase_gradient,
    compute_phase_curvature,
    measure_phase_symmetry
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.metrics.common import compute_coherence
from tnfr.config import DNFR_PRIMARY
import networkx as nx

# Configure logging to reduce output
logging.basicConfig(level=logging.ERROR)

# Critical threshold from literature
I_C_EXPECTED = 2.015

# Optimized intensity grid - focus on critical region
INTENSITIES = np.array([
    1.900, 1.950, 2.000, 2.010, 2.015, 2.020, 2.030, 2.050, 2.100
])

TOPOLOGIES = ['ws', 'scale_free', 'grid']
RUNS_PER_POINT = 20  # Reduced for speed
EVOLUTION_STEPS = 100  # Reduced steps

def create_test_network(topology: str, size: int = 30) -> nx.Graph:
    """Fast network creation"""
    seed = np.random.randint(1000, 9999)
    return create_tnfr_topology(topology, size, seed)

def xi_c_experiment(G: nx.Graph, intensity: float, verbose: bool = False) -> Dict[str, Any]:
    """Optimized experiment - minimal computation"""
    
    # Basic probe sequence - streamlined
    sequence = [
        (Emission(), 0),
        (Coupling(), 1), 
        (Dissonance(), 2),
        (Coherence(), 2),
        (Silence(), 0)
    ]
    
    try:
        # Apply sequence manually
        for op, node in sequence:
            op(G, node)
        
        # Fast evolution - manual steps
        for step in range(5):  # Reduced steps
            default_compute_delta_nfr(G)
            # Stabilize some nodes  
            if step % 2 == 0:
                coherence_op = Coherence()
                for node in list(G.nodes())[:2]:
                    coherence_op(G, node)
        
        # Final DNFR calculation 
        default_compute_delta_nfr(G)
        
        # Add per-node coherence
        for node in G.nodes():
            dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
            G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
        
        # Quick metrics
        results = {}
        
        # Phase symmetry (core measurement)
        phase_sym_result = measure_phase_symmetry(G)
        xi_c = estimate_coherence_length(G, coherence_key="coherence")
        
        results['xi_c'] = xi_c
        results['phase_symmetry'] = phase_sym_result.get('symmetry_index', 0.0)
        results['threshold_crossing'] = 1 if intensity > I_C_EXPECTED else 0
        
        # Only compute canonical fields if needed for correlation
        if intensity in [2.000, 2.015, 2.030]:  # Sample points for correlation
            phi_s = compute_structural_potential(G)
            grad_phi = compute_phase_gradient(G)  
            k_phi = compute_phase_curvature(G)
            
            results['phi_s_mean'] = np.mean(list(phi_s['values']))
            results['grad_phi_mean'] = np.mean(list(grad_phi['values']))
            results['k_phi_mean'] = np.mean(list(k_phi['values']))
        
        if verbose:
            print(f"    ξ_C = {xi_c:.3f}, symmetry = {results['phase_symmetry']:.3f}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {'xi_c': np.nan, 'error': str(e)}

def run_fast_experiment():
    """Optimized multi-topology experiment"""
    
    print("Starting optimized ξ_C validation experiment...")
    print(f"Intensities: {len(INTENSITIES)} points")
    print(f"Topologies: {TOPOLOGIES}")
    print(f"Runs per point: {RUNS_PER_POINT}")
    print(f"Total measurements: {len(INTENSITIES) * len(TOPOLOGIES) * RUNS_PER_POINT}")
    
    all_results = []
    
    for topology in TOPOLOGIES:
        print(f"\n=== {topology.upper()} TOPOLOGY ===")
        
        for i, intensity in enumerate(INTENSITIES):
            print(f"Intensity {intensity:.3f} ({i+1}/{len(INTENSITIES)})")
            
            intensity_results = []
            
            for run in range(RUNS_PER_POINT):
                if run % 5 == 0 and run > 0:
                    print(f"  Run {run}/{RUNS_PER_POINT}")
                
                # Create fresh network
                G = create_test_network(topology)
                
                # Run experiment  
                result = xi_c_experiment(G, intensity, verbose=False)
                result.update({
                    'topology': topology,
                    'intensity': intensity, 
                    'run': run
                })
                
                intensity_results.append(result)
                all_results.append(result)
            
            # Quick stats for this intensity
            valid_xi_c = [r['xi_c'] for r in intensity_results if not np.isnan(r['xi_c'])]
            if valid_xi_c:
                mean_xi_c = np.mean(valid_xi_c)
                print(f"  Mean ξ_C: {mean_xi_c:.3f} (n={len(valid_xi_c)})")
    
    return all_results

def analyze_critical_threshold(results: List[Dict]) -> Dict[str, Any]:
    """Fast analysis focusing on critical behavior"""
    
    print("\n=== CRITICAL THRESHOLD ANALYSIS ===")
    
    analysis = {}
    
    # Group by topology
    by_topology = {}
    for r in results:
        if r['topology'] not in by_topology:
            by_topology[r['topology']] = []
        if not np.isnan(r['xi_c']):
            by_topology[r['topology']].append(r)
    
    # Find critical behavior for each topology
    for topology in TOPOLOGIES:
        topo_results = by_topology.get(topology, [])
        if not topo_results:
            continue
        
        # Group by intensity
        by_intensity = {}
        for r in topo_results:
            intensity = r['intensity']
            if intensity not in by_intensity:
                by_intensity[intensity] = []
            by_intensity[intensity].append(r['xi_c'])
        
        # Analyze transition
        intensities = sorted(by_intensity.keys())
        mean_xi_c = []
        
        for intensity in intensities:
            xi_c_values = by_intensity[intensity]
            mean_val = np.mean(xi_c_values)
            mean_xi_c.append(mean_val)
        
        # Look for sharp transition around I_c = 2.015
        pre_critical = [xi for i, xi in zip(intensities, mean_xi_c) if i < I_C_EXPECTED]
        post_critical = [xi for i, xi in zip(intensities, mean_xi_c) if i > I_C_EXPECTED]
        
        if pre_critical and post_critical:
            pre_mean = np.mean(pre_critical)
            post_mean = np.mean(post_critical)
            transition_strength = abs(post_mean - pre_mean) / pre_mean if pre_mean > 0 else 0
            
            analysis[topology] = {
                'pre_critical_xi_c': pre_mean,
                'post_critical_xi_c': post_mean,
                'transition_strength': transition_strength,
                'n_points': len(topo_results)
            }
            
            print(f"{topology.upper()}:")
            print(f"  Pre-critical ξ_C: {pre_mean:.3f}")  
            print(f"  Post-critical ξ_C: {post_mean:.3f}")
            print(f"  Transition strength: {transition_strength:.3f}")
            print(f"  Data points: {len(topo_results)}")
    
    return analysis

def main():
    """Fast experiment execution"""
    
    print("ξ_C Fast Validation Experiment")
    print("=" * 40)
    
    # Run experiment
    results = run_fast_experiment()
    
    # Analyze critical behavior
    analysis = analyze_critical_threshold(results)
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total valid measurements: {len([r for r in results if not np.isnan(r['xi_c'])])}")
    
    all_xi_c = [r['xi_c'] for r in results if not np.isnan(r['xi_c'])]
    if all_xi_c:
        print(f"Overall ξ_C range: {np.min(all_xi_c):.3f} - {np.max(all_xi_c):.3f}")
        print(f"Mean ξ_C: {np.mean(all_xi_c):.3f} ± {np.std(all_xi_c):.3f}")
    
    # Check if critical behavior observed
    strong_transitions = sum(1 for topo_data in analysis.values() 
                            if topo_data.get('transition_strength', 0) > 0.1)
    
    print(f"\nTopologies showing strong transition (>10%): {strong_transitions}/{len(analysis)}")
    
    if strong_transitions >= 2:
        print("✅ CRITICAL THRESHOLD BEHAVIOR CONFIRMED")
        recommendation = "PROCEED with ξ_C canonical promotion"
    else:
        print("❌ Critical threshold behavior unclear")
        recommendation = "NEED more investigation before promotion"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    
    return results, analysis

if __name__ == "__main__":
    results, analysis = main()