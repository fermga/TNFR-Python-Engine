"""Phase Gradient Correlation Study - Alternative Metrics
==========================================================

Tests |∇φ| correlation with metrics that capture TNFR dynamics missed by C(t):
1. Mean ΔNFR: System-wide reorganization pressure
2. Max ΔNFR: Peak node stress (fragmentation indicator)
3. Sense Index (Si): Stable reorganization capacity

Background:
-----------
C(t) = 1 - (σ_ΔNFR / ΔNFR_max) is invariant to proportional scaling.
When all nodes change uniformly, C(t) remains constant even though
dynamics are occurring. Alternative metrics capture these changes.

Usage:
    python benchmarks/phase_gradient_alternative_metrics.py \
        --runs 10 --output alternative_metrics_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from tnfr.structural import run_sequence
from tnfr.metrics.coherence import compute_global_coherence
from tnfr.physics.fields import (
    compute_structural_potential,
    path_integrated_gradient,
)
from tnfr.constants import DNFR_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_THETA, ALIAS_DNFR
from benchmark_utils import (
    create_tnfr_topology,
    initialize_tnfr_nodes,
    generate_grammar_valid_sequence,
    validate_sequence_grammar,
)


@dataclass
class AlternativeMetricsResult:
    """Result from one experiment with alternative metrics."""
    run_id: int
    topology: str
    n_nodes: int
    intensity: float
    sequence_type: str
    
    # Traditional C(t)
    C_initial: float
    C_final: float
    delta_C: float
    
    # Alternative metrics
    mean_dnfr_initial: float
    mean_dnfr_final: float
    delta_mean_dnfr: float
    
    max_dnfr_initial: float
    max_dnfr_final: float
    delta_max_dnfr: float
    
    si_initial: float
    si_final: float
    delta_si: float
    
    # Phase gradient
    grad_phi_initial: float
    grad_phi_final: float
    delta_grad_phi: float
    
    # Structural potential
    phi_s_initial: float
    phi_s_final: float
    delta_phi_s: float
    
    # Metadata
    seed: int
    num_operators: int
    sequence_valid: bool
    error_message: Optional[str] = None


def compute_mean_dnfr(G: nx.Graph) -> float:
    """Compute mean ΔNFR across all nodes."""
    dnfr_values = []
    for node in G.nodes():
        node_data = G.nodes[node]
        for alias in ALIAS_DNFR:
            if alias in node_data:
                dnfr_values.append(abs(float(node_data[alias])))
                break
    
    if not dnfr_values:
        return 0.0
    return sum(dnfr_values) / len(dnfr_values)


def compute_max_dnfr(G: nx.Graph) -> float:
    """Compute maximum ΔNFR across all nodes."""
    dnfr_values = []
    for node in G.nodes():
        node_data = G.nodes[node]
        for alias in ALIAS_DNFR:
            if alias in node_data:
                dnfr_values.append(abs(float(node_data[alias])))
                break
    
    if not dnfr_values:
        return 0.0
    return max(dnfr_values)


def compute_sense_index(G: nx.Graph) -> float:
    """Compute Sense Index (Si) - capacity for stable reorganization.
    
    Si = νf / (1 + |ΔNFR|)
    
    Higher Si = better capacity to reorganize without fragmentation.
    """
    si_values = []
    for node in G.nodes():
        node_data = G.nodes[node]
        
        # Get νf
        nu_f = float(node_data.get(VF_PRIMARY, 1.0))
        
        # Get ΔNFR
        dnfr = 0.0
        for alias in ALIAS_DNFR:
            if alias in node_data:
                dnfr = abs(float(node_data[alias]))
                break
        
        si = nu_f / (1.0 + dnfr)
        si_values.append(si)
    
    if not si_values:
        return 1.0
    return sum(si_values) / len(si_values)


def compute_mean_phase_gradient(G: nx.Graph) -> float:
    """Compute mean |∇φ| across all nodes."""
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0
    
    gradients = []
    for node in nodes:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        
        try:
            # Get phase using alias system
            node_phase = 0.0
            node_data = G.nodes[node]
            for alias in ALIAS_THETA:
                if alias in node_data:
                    node_phase = float(node_data[alias])
                    break
            
            # Get neighbor phases
            neighbor_phases = []
            for nb in neighbors:
                nb_data = G.nodes[nb]
                for alias in ALIAS_THETA:
                    if alias in nb_data:
                        neighbor_phases.append(float(nb_data[alias]))
                        break
            
            if not neighbor_phases:
                continue
            
            # Mean phase difference
            phase_diffs = [abs(node_phase - nb_phase) 
                          for nb_phase in neighbor_phases]
            mean_diff = sum(phase_diffs) / len(phase_diffs)
            gradients.append(mean_diff)
        except Exception:
            continue
    
    return sum(gradients) / len(gradients) if gradients else 0.0


def compute_mean_phi_s(G: nx.Graph) -> float:
    """Compute mean Φ_s across all nodes."""
    phi_s_dict = compute_structural_potential(G)
    if not phi_s_dict:
        return 0.0
    return sum(phi_s_dict.values()) / len(phi_s_dict)


def run_experiment(
    run_id: int,
    topology: str,
    n_nodes: int,
    intensity: float,
    sequence_type: str,
    seed: int,
    nu_f: float = 1.0,
) -> AlternativeMetricsResult:
    """Run one experiment measuring alternative metrics."""
    try:
        # Create graph
        G = create_tnfr_topology(topology, n_nodes, seed)
        initialize_tnfr_nodes(G, nu_f=nu_f, seed=seed + 1)
        
        # Generate sequence
        sequence = generate_grammar_valid_sequence(
            sequence_type, intensity
        )
        
        # Validate grammar
        is_valid, error_msg = validate_sequence_grammar(sequence)
        if not is_valid:
            raise ValueError(f"Invalid sequence: {error_msg}")
        
        # Initial measurements
        C_init = compute_global_coherence(G)
        mean_dnfr_init = compute_mean_dnfr(G)
        max_dnfr_init = compute_max_dnfr(G)
        si_init = compute_sense_index(G)
        grad_phi_init = compute_mean_phase_gradient(G)
        phi_s_init = compute_mean_phi_s(G)
        
        # Apply sequence to ALL nodes
        for target_node in G.nodes():
            run_sequence(G, target_node, sequence)
        
        # Final measurements
        C_final = compute_global_coherence(G)
        mean_dnfr_final = compute_mean_dnfr(G)
        max_dnfr_final = compute_max_dnfr(G)
        si_final = compute_sense_index(G)
        grad_phi_final = compute_mean_phase_gradient(G)
        phi_s_final = compute_mean_phi_s(G)
        
        return AlternativeMetricsResult(
            run_id=run_id,
            topology=topology,
            n_nodes=n_nodes,
            intensity=intensity,
            sequence_type=sequence_type,
            C_initial=C_init,
            C_final=C_final,
            delta_C=C_final - C_init,
            mean_dnfr_initial=mean_dnfr_init,
            mean_dnfr_final=mean_dnfr_final,
            delta_mean_dnfr=mean_dnfr_final - mean_dnfr_init,
            max_dnfr_initial=max_dnfr_init,
            max_dnfr_final=max_dnfr_final,
            delta_max_dnfr=max_dnfr_final - max_dnfr_init,
            si_initial=si_init,
            si_final=si_final,
            delta_si=si_final - si_init,
            grad_phi_initial=grad_phi_init,
            grad_phi_final=grad_phi_final,
            delta_grad_phi=grad_phi_final - grad_phi_init,
            phi_s_initial=phi_s_init,
            phi_s_final=phi_s_final,
            delta_phi_s=phi_s_final - phi_s_init,
            seed=seed,
            num_operators=len(sequence),
            sequence_valid=True,
            error_message=None,
        )
    
    except Exception as e:
        return AlternativeMetricsResult(
            run_id=run_id,
            topology=topology,
            n_nodes=n_nodes,
            intensity=intensity,
            sequence_type=sequence_type,
            C_initial=0.0,
            C_final=0.0,
            delta_C=0.0,
            mean_dnfr_initial=0.0,
            mean_dnfr_final=0.0,
            delta_mean_dnfr=0.0,
            max_dnfr_initial=0.0,
            max_dnfr_final=0.0,
            delta_max_dnfr=0.0,
            si_initial=0.0,
            si_final=0.0,
            delta_si=0.0,
            grad_phi_initial=0.0,
            grad_phi_final=0.0,
            delta_grad_phi=0.0,
            phi_s_initial=0.0,
            phi_s_final=0.0,
            delta_phi_s=0.0,
            seed=seed,
            num_operators=0,
            sequence_valid=False,
            error_message=str(e),
        )


def main(argv: List[str]) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase gradient correlation with alternative metrics"
    )
    parser.add_argument(
        '--runs', type=int, default=10,
        help='Number of runs per condition'
    )
    parser.add_argument(
        '--output', type=str, default='alternative_metrics_results.jsonl',
        help='Output JSONL file'
    )
    parser.add_argument(
        '--n-nodes', type=int, default=6,
        help='Number of nodes in networks'
    )
    
    args = parser.parse_args(argv)
    
    # Experimental conditions
    topologies = ['ring', 'scale_free', 'ws', 'tree', 'grid']
    intensities = [1.5, 2.0, 2.5]
    sequence_types = ['RA_dominated', 'OZ_heavy', 'balanced']
    
    total_conditions = (len(topologies) * len(intensities) * 
                       len(sequence_types))
    total_experiments = total_conditions * args.runs
    
    print(f"Running {total_experiments} experiments...")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output
    
    print(f"Output: {output_path}")
    
    run_id = 0
    completed = 0
    
    with output_path.open('w', encoding='utf-8') as f:
        for topology in topologies:
            for intensity in intensities:
                for sequence_type in sequence_types:
                    for run_idx in range(args.runs):
                        seed = run_id + 42000
                        
                        result = run_experiment(
                            run_id=run_id,
                            topology=topology,
                            n_nodes=args.n_nodes,
                            intensity=intensity,
                            sequence_type=sequence_type,
                            seed=seed,
                        )
                        
                        f.write(json.dumps(asdict(result)) + '\n')
                        f.flush()
                        
                        run_id += 1
                        completed += 1
                        
                        if completed % 50 == 0:
                            print(f"Completed {completed}/{total_experiments}"
                                  f" runs...")
    
    print(f"\n✅ Completed {completed} experiments")
    print(f"Results saved to: {output_path}")
    print("\nNext: Analyze with correlation analysis script")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
