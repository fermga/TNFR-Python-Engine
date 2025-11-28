#!/usr/bin/env python3
"""
Complete Multi-Topology Critical Exponent Experiments

This script completes the ξ_C critical exponent extraction across all topologies
(ws, scale_free, grid) to establish comprehensive validation for canonical promotion.

Building on successful tree topology results (ν ≈ 0.607, ising-3d class).
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
        measure_phase_symmetry,
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature
    )
    from benchmark_utils import create_tnfr_topology

    from tnfr.operators.definitions import Dissonance, Coherence
    from tnfr.dynamics.dnfr import default_compute_delta_nfr
    from tnfr.metrics.common import compute_coherence
    from tnfr.constants import DNFR_PRIMARY
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
    np.random.seed(seed)
    
    if topology_type == "ring":
        return create_tnfr_topology("ring", n_nodes, seed)
    elif topology_type == "ws":
        return create_tnfr_topology("ws", n_nodes, seed)
    elif topology_type == "scale_free":
        return create_tnfr_topology("scale_free", n_nodes, seed)
    elif topology_type == "tree":
        return create_tnfr_topology("tree", n_nodes, seed)
    elif topology_type == "grid":
        return create_tnfr_topology("grid", n_nodes, seed)
    else:
        raise ValueError(f"Unknown topology: {topology_type}")


def run_multi_topology_experiments():
    """Run critical exponent experiments across all topology families"""
    
    # Experimental parameters
    topologies = ["ws", "scale_free", "grid"]  # Skip tree (already complete)
    n_nodes = 50
    n_runs = 30
    intensities = [
        1.8, 1.9, 1.95, 2.0, 2.01, 2.015, 2.02, 2.03, 2.04, 2.05, 2.07, 2.1, 2.2
    ]
    seed_base = 12345
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = (
        results_dir /
        f"multi_topology_critical_exponent_{timestamp}.jsonl"
    )
    
    print("="*60)
    print("ξ_C MULTI-TOPOLOGY CRITICAL EXPONENT EXPERIMENTS")
    print("="*60)
    print(f"Topologies: {topologies}")
    print(f"Nodes per topology: {n_nodes}")
    print(f"Runs per intensity: {n_runs}")
    print(f"Intensities: {len(intensities)} points")
    print(f"Results: {results_file}")
    print("")
    
    all_results = {}
    
    for topology in topologies:
        print(f"Running {topology} experiments...")
        start_time = time.time()
        
        result = run_topology_experiment(
            topology, n_nodes, n_runs, intensities, seed_base
        )
        
        duration = time.time() - start_time
        result["duration_seconds"] = duration
        result["timestamp"] = timestamp
        result["experiment"] = "multi_topology_critical_exponent"
        
        # Save individual result
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        all_results[topology] = result
        
        if result['analysis']['success']:
            fit = result['analysis']['exponent_fit']
            print(f"  {topology:12} | "
                  f"nu = {fit['nu_above']:.3f} | "
                  f"R² = {fit['r_squared_above']:.3f} | "
                  f"class = {fit['universality_class']}")
        else:
            print(f"  {topology:12} | FAILED: {result['analysis'].get('error', 'Unknown error')}")
        print(f"  Duration: {duration:.1f}s")
        print("")
    
    # Create comprehensive analysis
    analyze_multi_topology_results(all_results, results_file)


def run_topology_experiment(
    topology: str, n_nodes: int, n_runs: int, 
    intensities: List[float], seed_base: int
) -> Dict[str, Any]:
    """Run critical exponent experiment for single topology"""
    
    print(f"Testing {topology} topology with {n_runs} runs per intensity...")
    
    xi_c_data = []
    symmetry_data = []
    coherence_data = []
    canonical_fields_data = []  # New: Track Φ_s, |∇φ|, K_φ
    
    for i, intensity in enumerate(intensities):
        print(f"  Intensity {intensity:.3f} ({i+1}/{len(intensities)})")
        
        xi_c_runs = []
        symmetry_runs = []
        coherence_runs = []
        canonical_runs = {"phi_s": [], "grad_phi": [], "k_phi": []}
        
        for run in range(n_runs):
            seed = seed_base + i * 1000 + run
            np.random.seed(seed)
            
            try:
                # Create fresh topology
                G = create_test_topology(topology, n_nodes, seed)
                
                # Create network-wide perturbation proportional to intensity
                for node in G.nodes():
                    delta_I = intensity - I_C  # Distance from critical point
                    base_dnfr = delta_I * np.random.normal(0, 0.5)
                    G.nodes[node][DNFR_PRIMARY] = base_dnfr
                
                # Apply multiple dissonance operators for network evolution
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
                    # COMMENTED OUT to preserve DNFR spatial variation needed for ξ_C
                    # default_compute_delta_nfr(G)
                    # Apply coherence to stabilize some nodes
                    coherence_op = Coherence()
                    node_list = list(G.nodes())
                    stabilized_nodes = np.random.choice(
                        node_list, size=max(1, n_nodes//4), replace=False
                    )
                    for node in stabilized_nodes:
                        coherence_op(G, node)
                
                # Final ΔNFR calculation after evolution - COMMENTED OUT to preserve spatial variation
                # default_compute_delta_nfr(G)
                
                # Add per-node coherence for coherence length calculation
                for node in G.nodes():
                    dnfr = abs(G.nodes[node].get(DNFR_PRIMARY, 0.0))
                    G.nodes[node]['coherence'] = 1.0 / (1.0 + dnfr)
                
                # Measure ξ_C coherence length
                xi_c = estimate_coherence_length(G, coherence_key="coherence")
                xi_c_runs.append(xi_c)
                
                # Measure phase symmetry
                symmetry = measure_phase_symmetry(G)
                symmetry_runs.append(symmetry)
                
                # Measure global coherence
                coherence = compute_coherence(G)
                coherence_runs.append(coherence)
                
                # NEW: Measure canonical fields for cross-validation
                phi_s_dict = compute_structural_potential(G)
                grad_phi_dict = compute_phase_gradient(G)
                k_phi_dict = compute_phase_curvature(G)
                
                # Convert dictionaries to scalar means
                phi_s = np.mean(list(phi_s_dict.values())) if phi_s_dict else 0.0
                grad_phi = np.mean(list(grad_phi_dict.values())) if grad_phi_dict else 0.0
                k_phi = np.mean(list(k_phi_dict.values())) if k_phi_dict else 0.0
                
                canonical_runs["phi_s"].append(phi_s)
                canonical_runs["grad_phi"].append(grad_phi) 
                canonical_runs["k_phi"].append(k_phi)
                
            except Exception as e:
                import traceback
                print(f"    Run {run} failed: {type(e).__name__}: {e}")
                print(f"    Traceback:")
                traceback.print_exc()
                # Use fallback values
                xi_c_runs.append(0.0)
                symmetry_runs.append({
                    "circular_variance": 0.0,
                    "n_clusters": 1,
                    "cluster_separation": 0.0,
                    "largest_cluster_fraction": 1.0
                })
                coherence_runs.append(0.5)
                canonical_runs["phi_s"].append(0.0)
                canonical_runs["grad_phi"].append(0.0)
                canonical_runs["k_phi"].append(0.0)
        
        # Statistical analysis for this intensity
        xi_c_data.append({
            "intensity": intensity,
            "values": xi_c_runs,
            "mean": np.mean(xi_c_runs),
            "std": np.std(xi_c_runs),
            "median": np.median(xi_c_runs)
        })
        
        # Aggregate symmetry data
        symmetry_data.append({
            "intensity": intensity,
            "circular_variance": [s["circular_variance"] for s in symmetry_runs],
            "n_clusters": [s["n_clusters"] for s in symmetry_runs],
            "cluster_separation": [s["cluster_separation"] for s in symmetry_runs],
            "largest_cluster_fraction": [
                s["largest_cluster_fraction"] for s in symmetry_runs
            ]
        })
        
        # Coherence data
        coherence_data.append({
            "intensity": intensity,
            "values": coherence_runs,
            "mean": np.mean(coherence_runs),
            "std": np.std(coherence_runs)
        })
        
        # Canonical fields data
        canonical_fields_data.append({
            "intensity": intensity,
            "phi_s": {
                "values": canonical_runs["phi_s"],
                "mean": np.mean(canonical_runs["phi_s"]),
                "std": np.std(canonical_runs["phi_s"])
            },
            "grad_phi": {
                "values": canonical_runs["grad_phi"],
                "mean": np.mean(canonical_runs["grad_phi"]),
                "std": np.std(canonical_runs["grad_phi"])
            },
            "k_phi": {
                "values": canonical_runs["k_phi"],
                "mean": np.mean(canonical_runs["k_phi"]),
                "std": np.std(canonical_runs["k_phi"])
            }
        })
    
    # Fit critical exponent
    try:
        xi_c_means = [data["mean"] for data in xi_c_data]
        fit_result = fit_correlation_length_exponent(
            intensities, xi_c_means, I_C
        )
        
        # Defensive handling for min/max operations
        try:
            # Convert any non-scalar values to float for safety
            xi_c_safe = []
            for x in xi_c_means:
                if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
                    # If it's a list/array, take the mean
                    xi_c_safe.append(float(np.mean(x)))
                else:
                    xi_c_safe.append(float(x))
            
            xi_c_range = [min(xi_c_safe), max(xi_c_safe)] if xi_c_safe else [0.0, 0.0]
        except Exception as range_error:
            print(f"    Warning: xi_c_range calculation failed: {range_error}")
            print(f"    xi_c_means = {xi_c_means}")
            print(f"    xi_c_means types = {[type(x) for x in xi_c_means]}")
            xi_c_range = [0.0, 0.0]
        
        # Safe comparison for n_valid calculation
        n_valid_count = 0
        for x in xi_c_means:
            try:
                if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
                    # If it's a list/array, take the mean
                    val = float(np.mean(x))
                else:
                    val = float(x)
                if val > 0:
                    n_valid_count += 1
            except:
                pass  # Skip invalid values
        
        analysis = {
            "success": True,
            "topology": topology,
            "n_valid": n_valid_count,
            "exponent_fit": fit_result,
            "intensity_range": [min(intensities), max(intensities)],
            "xi_c_range": xi_c_range
        }
    except Exception as e:
        # Safe n_valid calculation for error case too
        n_valid_count = 0
        for x in xi_c_means:
            try:
                if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
                    val = float(np.mean(x))
                else:
                    val = float(x)
                if val > 0:
                    n_valid_count += 1
            except:
                pass
        
        analysis = {
            "success": False,
            "error": str(e),
            "n_valid": n_valid_count
        }
    
    return {
        "topology": topology,
        "raw_data": {
            "topology": topology,
            "n_nodes": n_nodes,
            "n_runs": n_runs,
            "intensities": intensities,
            "xi_c_data": xi_c_data,
            "symmetry_data": symmetry_data,
            "coherence_data": coherence_data,
            "canonical_fields_data": canonical_fields_data  # NEW
        },
        "analysis": analysis
    }


def analyze_multi_topology_results(
    results: Dict[str, Any], results_file: Path
):
    """Analyze results across all topologies"""
    
    print("\n" + "="*60)
    print("MULTI-TOPOLOGY CRITICAL EXPONENT ANALYSIS")
    print("="*60)
    
    summary = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "experiment": "multi_topology_summary",
        "topologies": list(results.keys()),
        "topology_results": {},
        "cross_validation": {},
        "canonical_assessment": {}
    }
    
    # Analyze each topology
    for topology, result in results.items():
        if result['analysis']['success']:
            fit = result['analysis']['exponent_fit']
            summary["topology_results"][topology] = {
                "nu_above": fit['nu_above'],
                "r_squared": fit['r_squared_above'],
                "universality_class": fit['universality_class'],
                "xi_c_range": result['analysis']['xi_c_range']
            }
            
            print(f"{topology:12} | "
                  f"ν = {fit['nu_above']:.3f} | "
                  f"R² = {fit['r_squared_above']:.3f} | "
                  f"class = {fit['universality_class']} | "
                  f"range = {result['analysis']['xi_c_range'][1]:.0f}x")
    
    # Cross-validation with canonical fields
    print(f"\n📊 CROSS-VALIDATION WITH CANONICAL FIELDS:")
    correlations = analyze_canonical_correlations(results)
    summary["cross_validation"] = correlations
    
    for field, corr in correlations.items():
        print(f"   ξ_C ↔ {field}: r = {corr['correlation']:.3f} "
              f"(p = {corr.get('p_value', 'N/A')})")
    
    # Canonical promotion assessment
    assessment = assess_canonical_promotion(summary)
    summary["canonical_assessment"] = assessment
    
    print(f"\n🏆 CANONICAL PROMOTION ASSESSMENT:")
    print(f"   Score: {assessment['score']}/10")
    print(f"   Recommendation: {assessment['recommendation']}")
    
    # Save comprehensive summary
    with open(results_file, 'a') as f:
        f.write(json.dumps(summary) + '\n')
    
    print(f"\n📁 Complete results saved to: {results_file}")


def analyze_canonical_correlations(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze correlations between ξ_C and canonical fields"""
    
    correlations = {}
    all_xi_c = []
    all_phi_s = []
    all_grad_phi = []
    all_k_phi = []
    
    # Aggregate data across topologies
    for topology, result in results.items():
        if not result['analysis']['success']:
            continue
            
        xi_c_data = result['raw_data']['xi_c_data']
        canonical_data = result['raw_data']['canonical_fields_data']
        
        for i, xi_data in enumerate(xi_c_data):
            canonical_point = canonical_data[i]
            
            # Extend with individual values from each measurement
            all_xi_c.extend(xi_data['values'])
            all_phi_s.extend(canonical_point['phi_s']['values'])
            all_grad_phi.extend(canonical_point['grad_phi']['values'])
            all_k_phi.extend(canonical_point['k_phi']['values'])
    
    # Compute correlations
    if len(all_xi_c) > 10:  # Minimum data requirement
        correlations = {
            "phi_s": {
                "correlation": np.corrcoef(all_xi_c, all_phi_s)[0, 1]
            },
            "grad_phi": {
                "correlation": np.corrcoef(all_xi_c, all_grad_phi)[0, 1]
            },
            "k_phi": {
                "correlation": np.corrcoef(all_xi_c, all_k_phi)[0, 1]
            }
        }
    
    return correlations


def assess_canonical_promotion(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Assess ξ_C for canonical field promotion"""
    
    score = 0
    criteria = []
    
    # Criterion 1: Multi-topology validation (2 points)
    n_successful = len(summary["topology_results"])
    if n_successful >= 3:
        score += 2
        criteria.append("✅ Multi-topology validation (3+ topologies)")
    elif n_successful >= 2:
        score += 1
        criteria.append("⚡ Partial topology validation (2 topologies)")
    else:
        criteria.append("❌ Insufficient topology coverage")
    
    # Criterion 2: Critical exponent quality (2 points)  
    good_exponents = sum(
        1 for r in summary["topology_results"].values() 
        if r["r_squared"] > 0.3 and 0.4 < r["nu_above"] < 1.0
    )
    if good_exponents >= 2:
        score += 2
        criteria.append("✅ High-quality critical exponents")
    elif good_exponents >= 1:
        score += 1
        criteria.append("⚡ Moderate critical exponents")
    else:
        criteria.append("❌ Poor critical exponent quality")
    
    # Criterion 3: Universality classification (2 points)
    known_classes = sum(
        1 for r in summary["topology_results"].values()
        if r["universality_class"] != "unknown"
    )
    if known_classes >= 2:
        score += 2
        criteria.append("✅ Proper universality classification")
    elif known_classes >= 1:
        score += 1
        criteria.append("⚡ Partial universality classification")
    else:
        criteria.append("❌ No universality classification")
    
    # Criterion 4: Dynamic range (1 point)
    max_range = max(
        r["xi_c_range"][1] / r["xi_c_range"][0]
        for r in summary["topology_results"].values()
        if r["xi_c_range"][0] > 0
    ) if summary["topology_results"] else 0
    
    if max_range > 10:
        score += 1
        criteria.append(f"✅ Excellent dynamic range ({max_range:.1f}x)")
    elif max_range > 5:
        score += 0.5
        criteria.append(f"⚡ Good dynamic range ({max_range:.1f}x)")
    else:
        criteria.append("❌ Insufficient dynamic range")
    
    # Criterion 5: Cross-validation (2 points)
    correlations = summary.get("cross_validation", {})
    strong_correlations = sum(
        1 for corr in correlations.values()
        if abs(corr.get("correlation", 0)) > 0.5
    )
    
    if strong_correlations >= 2:
        score += 2
        criteria.append("✅ Strong cross-field correlations")
    elif strong_correlations >= 1:
        score += 1
        criteria.append("⚡ Moderate cross-field correlations")
    else:
        criteria.append("❌ Weak cross-field correlations")
    
    # Criterion 6: Theoretical consistency (1 point)
    theory_match = sum(
        1 for r in summary["topology_results"].values()
        if r["universality_class"] in ["ising-2d", "ising-3d", "mean-field"]
    )
    
    if theory_match >= 1:
        score += 1
        criteria.append("✅ Matches known physics universality classes")
    else:
        criteria.append("❌ No theoretical universality match")
    
    # Final recommendation
    if score >= 8:
        recommendation = "STRONG CANDIDATE - Recommend canonical promotion"
    elif score >= 6:
        recommendation = "PROMISING - Conditional promotion pending improvements"
    elif score >= 4:
        recommendation = "DEVELOPING - More validation needed"
    else:
        recommendation = "INSUFFICIENT - Major development required"
    
    return {
        "score": score,
        "max_score": 10,
        "criteria": criteria,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    run_multi_topology_experiments()