#!/usr/bin/env python3
"""U6 Temporal Ordering: Experimental Validation Suite.

This script implements the experimental protocols described in
docs/grammar/U6_TEMPORAL_ORDERING.md to measure and validate
temporal ordering requirements.

Experiments:
- A: τ_relax measurement across νf and topologies
- B: Nonlinear accumulation α(Δt) curves
- C: Bifurcation index B vs coherence fragmentation

Usage:
    python scripts/experiment_u6.py --experiment A --output results/u6_tau_relax.json
    python scripts/experiment_u6.py --experiment B --vf 1.0 --topology ring
    python scripts/experiment_u6.py --experiment C --all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Silence,
)
from tnfr.operators.grammar import GrammarValidator
from tnfr.operators.metrics import (
    compute_bifurcation_index,
    measure_nonlinear_accumulation,
    measure_tau_relax_observed,
)
from tnfr.structural import create_nfr
from tnfr.utils.topology import compute_k_top_spectral


def create_test_topologies() -> dict[str, Any]:
    """Create test networks with different topologies."""
    import networkx as nx
    
    topologies = {}
    
    # Star/radial (k_top ≈ 1.0)
    G_star = nx.star_graph(5)
    topologies["star"] = G_star
    
    # Ring/cyclic (k_top ≈ 0.16)
    G_ring = nx.cycle_graph(10)
    topologies["ring"] = G_ring
    
    # Grid (intermediate k_top)
    G_grid = nx.grid_2d_graph(4, 4)
    topologies["grid"] = G_grid
    
    # Random (variable k_top)
    G_random = nx.erdos_renyi_graph(10, 0.3, seed=42)
    topologies["random"] = G_random
    
    return topologies


def experiment_a_tau_relax(
    vf_range: list[float],
    topologies: dict[str, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Experiment A: Measure τ_relax across νf and topologies.

    Protocol:
    1. For each topology and νf value:
    2. Create network, apply Emission + Dissonance
    3. Monitor until |ΔNFR| < 0.05·|ΔNFR_0| and C(t) > 0.95·C_0
    4. Record τ_relax_observed
    5. Compare with estimated τ_relax = (k_top/νf)·ln(20)

    Returns
    -------
    dict
        Results including observed vs estimated τ_relax for each configuration
    """
    results = {
        "experiment": "A_tau_relax",
        "configurations": [],
    }

    for topo_name, G_topo in topologies.items():
        # Compute k_top for this topology
        k_top = compute_k_top_spectral(G_topo)
        
        print(f"\n=== Topology: {topo_name} (k_top={k_top:.3f}) ===")
        
        for vf in vf_range:
            print(f"  νf = {vf:.2f} Hz_str...")
            
            # Create TNFR network from topology
            G, node_id = create_nfr("test", epi=0.1, vf=vf)
            
            # Apply structure from topology (simplified: use topology as template)
            # In full implementation, would build actual TNFR network
            
            # Apply destabilizer
            Emission()(G, node_id)
            Dissonance()(G, node_id)
            
            # Measure τ_relax (snapshot, full monitoring not implemented)
            telemetry = measure_tau_relax_observed(G, node_id)
            
            # Estimate τ_relax
            tau_est = (k_top / vf) * 3.0  # k_op=1.0, ln(20)≈3.0
            
            config_result = {
                "topology": topo_name,
                "k_top": k_top,
                "vf": vf,
                "tau_relax_estimated": tau_est,
                "tau_relax_observed": telemetry["tau_relax_observed"],
                "dnfr_initial": telemetry["dnfr_initial"],
                "coherence_initial": telemetry["coherence_initial"],
            }
            
            results["configurations"].append(config_result)
            print(f"    τ_relax_estimated: {tau_est:.2f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def experiment_b_nonlinear_accumulation(
    dt_range: list[float],
    vf: float = 1.0,
    topology: str = "ring",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Experiment B: Measure α(Δt) nonlinear accumulation curves.

    Protocol:
    1. Apply first Dissonance, measure ΔNFR_0
    2. Wait Δt (varying from 0 to 5·τ_relax)
    3. Measure ΔNFR_before_second
    4. Apply second Dissonance
    5. Measure ΔNFR_actual
    6. Compute α(Δt)

    Expected: α(Δt) > 1 for Δt < τ_relax, α(Δt) → 1 for Δt ≥ τ_relax

    Returns
    -------
    dict
        Results including α(Δt) curves
    """
    topologies = create_test_topologies()
    G_topo = topologies.get(topology, topologies["ring"])
    k_top = compute_k_top_spectral(G_topo)
    
    tau_relax_est = (k_top / vf) * 3.0
    
    print(f"\n=== Experiment B: α(Δt) for {topology} topology ===")
    print(f"νf = {vf:.2f}, k_top = {k_top:.3f}, τ_relax ≈ {tau_relax_est:.2f}")
    
    results = {
        "experiment": "B_nonlinear_accumulation",
        "topology": topology,
        "vf": vf,
        "k_top": k_top,
        "tau_relax_estimated": tau_relax_est,
        "measurements": [],
    }
    
    for dt in dt_range:
        G, node_id = create_nfr("test", epi=0.1, vf=vf)
        
        # First destabilizer
        Emission()(G, node_id)
        from ..alias import get_attr
        from ..constants.aliases import ALIAS_DNFR
        dnfr_0 = abs(float(get_attr(G.nodes[node_id], ALIAS_DNFR, 0.0)))
        
        Dissonance()(G, node_id)
        
        # Simulate waiting (in full implementation, integrate dynamics)
        # For now, just track time separation
        
        # Before second destabilizer
        dnfr_before_second = abs(float(get_attr(G.nodes[node_id], ALIAS_DNFR, 0.0)))
        
        # Second destabilizer
        Dissonance()(G, node_id)
        
        # Measure α
        alpha_metrics = measure_nonlinear_accumulation(
            G, node_id, dnfr_0, dnfr_before_second, dt
        )
        
        results["measurements"].append({
            "dt": dt,
            "dt_normalized": dt / tau_relax_est,
            "alpha": alpha_metrics["alpha"],
            "severity": alpha_metrics["amplification_severity"],
            "nonlinear_regime": alpha_metrics["nonlinear_regime"],
        })
        
        print(f"  Δt={dt:.2f} (Δt/τ={dt/tau_relax_est:.2f}): α={alpha_metrics['alpha']:.2f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def experiment_c_bifurcation_index(
    test_sequences: list[tuple[str, list]],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Experiment C: B index vs coherence fragmentation.

    Protocol:
    1. For each test sequence:
    2. Run sequence, track B after each operator
    3. Measure C(t) trajectory
    4. Identify B_crit threshold where C(t) fragments

    Returns
    -------
    dict
        Results including B trajectories and C(t) correlation
    """
    print("\n=== Experiment C: B vs Coherence Fragmentation ===")
    
    results = {
        "experiment": "C_bifurcation_index",
        "sequences": [],
    }
    
    for seq_name, operators in test_sequences:
        print(f"\n  Sequence: {seq_name}")
        
        G, node_id = create_nfr("test", epi=0.0, vf=1.0)
        
        B_trajectory = []
        C_trajectory = []
        
        for i, op in enumerate(operators):
            op(G, node_id)
            
            # Compute B
            B_metrics = compute_bifurcation_index(G, node_id)
            B_trajectory.append(B_metrics["B"])
            
            # Estimate C(t) (simplified)
            from ..alias import get_attr
            from ..constants.aliases import ALIAS_DNFR
            dnfr = abs(float(get_attr(G.nodes[node_id], ALIAS_DNFR, 0.0)))
            C_est = 1.0 / (1.0 + dnfr)
            C_trajectory.append(C_est)
            
            print(f"    Step {i}: B={B_metrics['B']:.2f}, C≈{C_est:.2f}, risk={B_metrics['risk_level']}")
        
        results["sequences"].append({
            "name": seq_name,
            "B_trajectory": B_trajectory,
            "C_trajectory": C_trajectory,
            "max_B": max(B_trajectory),
            "min_C": min(C_trajectory),
        })

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="U6 Temporal Ordering Experimental Validation"
    )
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--vf", type=float, default=1.0, help="Structural frequency (Hz_str)"
    )
    parser.add_argument(
        "--topology",
        choices=["star", "ring", "grid", "random"],
        default="ring",
        help="Network topology",
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON path for results"
    )
    
    args = parser.parse_args()
    
    topologies = create_test_topologies()
    
    if args.experiment in ["A", "all"]:
        vf_range = [0.5, 1.0, 2.0, 5.0]
        experiment_a_tau_relax(vf_range, topologies, args.output)
    
    if args.experiment in ["B", "all"]:
        dt_range = np.linspace(0.1, 5.0, 10)
        experiment_b_nonlinear_accumulation(dt_range, args.vf, args.topology, args.output)
    
    if args.experiment in ["C", "all"]:
        test_sequences = [
            ("valid_spaced", [Emission(), Dissonance(), Coherence(), Dissonance(), Silence()]),
            ("violation_consecutive", [Emission(), Dissonance(), Dissonance(), Coherence(), Silence()]),
            ("triple_destabilizer", [Emission(), Dissonance(), Expansion(), Mutation(), Coherence(), Silence()]),
        ]
        experiment_c_bifurcation_index(test_sequences, args.output)
    
    print("\n=== Experiments Complete ===")


if __name__ == "__main__":
    main()
