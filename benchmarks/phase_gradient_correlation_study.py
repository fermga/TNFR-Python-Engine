"""Phase Gradient Correlation Study
===================================

Extended correlation analysis to investigate if |∇φ| predictive power improves
under specific conditions:
- Different intensity ranges (I ∈ [1.5, 2.5])
- Different topologies (ring, scale_free, ws, tree, grid)
- Different operator sequence types (RA-dominated vs OZ-heavy)

Goal: Determine if |∇φ| correlation with coherence can reach |corr| > 0.5
(comparable to Φ_s baseline of -0.822) under specific conditions.

Output: JSONL results in benchmarks/results/phase_gradient_correlation_*.jsonl

Usage (PowerShell):
    python benchmarks/phase_gradient_correlation_study.py --runs 100

Status: RESEARCH - Part of |∇φ| validation for canonical promotion
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import networkx as nx

# Ensure TNFR modules are importable
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Import benchmark utilities
sys.path.insert(0, str(_ROOT / "benchmarks"))
from benchmark_utils import (  # noqa: E402
    create_tnfr_topology,
    initialize_tnfr_nodes,
    generate_grammar_valid_sequence,
    validate_sequence_grammar,
)

from tnfr.metrics.coherence import compute_global_coherence  # noqa: E402
from tnfr.physics.fields import (  # noqa: E402
    compute_phase_gradient,
    compute_structural_potential,
    path_integrated_gradient,
)
from tnfr.structural import run_sequence  # noqa: E402


@dataclass
class ExperimentResult:
    """Results from one experimental run."""
    run_id: int
    topology: str
    n_nodes: int
    intensity: float
    sequence_type: str
    
    # Initial state
    C_initial: float
    grad_phi_initial: float
    phi_s_initial: float
    
    # Final state
    C_final: float
    grad_phi_final: float
    phi_s_final: float
    
    # Changes
    delta_C: float
    delta_grad_phi: float
    delta_phi_s: float
    
    # Path-integrated metrics (for RA sequences)
    mean_path_gradient: float
    
    # Metadata
    seed: int
    num_operators: int
    sequence_valid: bool
    error_message: str


def compute_mean_phase_gradient(G: nx.Graph) -> float:
    """Compute mean |∇φ| across all nodes."""
    grad_dict = compute_phase_gradient(G)
    if not grad_dict:
        return 0.0
    return sum(grad_dict.values()) / len(grad_dict)


def compute_mean_phi_s(G: nx.Graph) -> float:
    """Compute mean Φ_s across all nodes."""
    phi_s_dict = compute_structural_potential(G, alpha=2.0)
    if not phi_s_dict:
        return 0.0
    return sum(phi_s_dict.values()) / len(phi_s_dict)


def compute_mean_path_gradient(
    G: nx.Graph,
    sample_size: int = 10
) -> float:
    """Sample random node pairs and compute mean path-integrated gradient."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0
    
    gradients = []
    for _ in range(min(sample_size, len(nodes) * (len(nodes) - 1) // 2)):
        i, j = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, i, j)
            pig = path_integrated_gradient(G, path)
            gradients.append(pig)
        except nx.NetworkXNoPath:
            continue
    
    if not gradients:
        return 0.0
    return sum(gradients) / len(gradients)


def run_experiment(
    run_id: int,
    topology: str,
    n_nodes: int,
    intensity: float,
    sequence_type: str,
    nu_f: float,
    seed: int,
) -> ExperimentResult:
    """Run single experiment and collect metrics."""
    # Create and initialize graph using utilities
    G = create_tnfr_topology(topology, n_nodes, seed)
    initialize_tnfr_nodes(G, nu_f=nu_f, seed=seed + 1)
    
    # Measure initial state
    C_init = compute_global_coherence(G)
    grad_phi_init = compute_mean_phase_gradient(G)
    phi_s_init = compute_mean_phi_s(G)
    
    # Generate grammar-valid sequence
    sequence = generate_grammar_valid_sequence(sequence_type, intensity)
    
    # Validate sequence
    is_valid, error_msg = validate_sequence_grammar(sequence)
    
    if not is_valid:
        # Return initial state if sequence invalid
        return ExperimentResult(
            run_id=run_id,
            topology=topology,
            n_nodes=n_nodes,
            intensity=intensity,
            sequence_type=sequence_type,
            C_initial=C_init,
            grad_phi_initial=grad_phi_init,
            phi_s_initial=phi_s_init,
            C_final=C_init,
            grad_phi_final=grad_phi_init,
            phi_s_final=phi_s_init,
            delta_C=0.0,
            delta_grad_phi=0.0,
            delta_phi_s=0.0,
            mean_path_gradient=0.0,
            seed=seed,
            num_operators=len(sequence),
            sequence_valid=False,
            error_message=error_msg,
        )
    
    # Apply complete sequence to ALL nodes (not just one)
    # This ensures fields actually change across the network
    try:
        for target_node in G.nodes():
            run_sequence(G, target_node, sequence)
        
        # Measure final state after sequence applied to all nodes
        C_final = compute_global_coherence(G)
        grad_phi_final = compute_mean_phase_gradient(G)
        phi_s_final = compute_mean_phi_s(G)
        
        execution_error = ""
        
    except Exception as e:
        # Sequence failed during execution
        C_final = C_init
        grad_phi_final = grad_phi_init
        phi_s_final = phi_s_init
        execution_error = str(e)[:200]  # Truncate long errors
    
    # Compute path gradients (for RA sequences)
    mean_pig = 0.0
    if sequence_type == 'RA_dominated':
        try:
            mean_pig = compute_mean_path_gradient(G, sample_size=10)
        except Exception:
            mean_pig = 0.0
    
    return ExperimentResult(
        run_id=run_id,
        topology=topology,
        n_nodes=n_nodes,
        intensity=intensity,
        sequence_type=sequence_type,
        C_initial=C_init,
        grad_phi_initial=grad_phi_init,
        phi_s_initial=phi_s_init,
        C_final=C_final,
        grad_phi_final=grad_phi_final,
        phi_s_final=phi_s_final,
        delta_C=C_final - C_init,
        delta_grad_phi=grad_phi_final - grad_phi_init,
        delta_phi_s=phi_s_final - phi_s_init,
        mean_path_gradient=mean_pig,
        seed=seed,
        num_operators=len(sequence),
        sequence_valid=is_valid,
        error_message=execution_error,
    )


def main(argv: List[str]) -> int:
    """Run phase gradient correlation study."""
    parser = argparse.ArgumentParser(description="Phase gradient correlation study")
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per condition')
    parser.add_argument('--output', type=str, default='phase_gradient_correlation.jsonl',
                        help='Output JSONL file (in benchmarks/results/)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--n-nodes', type=int, default=10, help='Number of nodes')
    parser.add_argument('--nu-f', type=float, default=1.0, help='Structural frequency')
    
    args = parser.parse_args(argv)
    
    # Output path
    results_dir = _ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / args.output
    
    # Experimental conditions
    topologies = ['ring', 'scale_free', 'ws', 'tree', 'grid']
    intensities = [1.5, 2.0, 2.5]  # Different stress levels
    sequence_types = ['RA_dominated', 'OZ_heavy', 'balanced']
    
    total_runs = len(topologies) * len(intensities) * len(sequence_types) * args.runs
    print(f"Running {total_runs} experiments...")
    print(f"Output: {output_path}")
    
    run_id = 0
    with output_path.open('w', encoding='utf-8') as f:
        for topology in topologies:
            for intensity in intensities:
                for seq_type in sequence_types:
                    for rep in range(args.runs):
                        run_id += 1
                        seed = args.seed + run_id
                        
                        try:
                            result = run_experiment(
                                run_id=run_id,
                                topology=topology,
                                n_nodes=args.n_nodes,
                                intensity=intensity,
                                sequence_type=seq_type,
                                nu_f=args.nu_f,
                                seed=seed,
                            )
                            
                            # Write result as JSONL
                            f.write(json.dumps(asdict(result)) + '\n')
                            f.flush()
                            
                            if run_id % 50 == 0:
                                print(f"Completed {run_id}/{total_runs} runs...")
                                
                        except Exception as e:
                            print(f"Error in run {run_id}: {e}")
                            continue
    
    print(f"\n✅ Completed {run_id} experiments")
    print(f"Results saved to: {output_path}")
    print("\nNext: Analyze with custom correlation analysis script")
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
