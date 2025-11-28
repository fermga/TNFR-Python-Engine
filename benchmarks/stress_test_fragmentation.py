"""Stress Test: High-Intensity Sequences to Force Fragmentation
================================================================

Creates deliberately unstable sequences to observe fragmentation dynamics
and test if |∇φ| provides early warning independent of Φ_s.

Strategy:
- Very high intensity (I = 3.0, 4.0, 5.0)
- Long sequences (15-20 operators)
- Multiple consecutive destabilizers
- Minimal stabilization

Goal: Force some runs into fragmentation to calibrate safety thresholds.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from tnfr.structural import run_sequence
from tnfr.operators.definitions import (
    Emission, Reception, Coherence, Dissonance, Resonance, Silence,
    Expansion, SelfOrganization, Coupling, Mutation
)
from tnfr.metrics.coherence import compute_global_coherence
from tnfr.physics.fields import compute_structural_potential, path_integrated_gradient
from benchmark_utils import create_tnfr_topology, initialize_tnfr_nodes


@dataclass
class StressTestResult:
    """Result from one stress test run."""
    run_id: int
    topology: str
    n_nodes: int
    intensity: float
    sequence_type: str
    C_initial: float
    C_final: float
    grad_phi_initial: float
    grad_phi_final: float
    phi_s_initial: float
    phi_s_final: float
    delta_C: float
    delta_grad_phi: float
    delta_phi_s: float
    seed: int
    num_operators: int
    fragmented: bool
    error_message: Optional[str] = None


def generate_stress_sequence(sequence_type: str, intensity: float) -> List:
    """Generate deliberately unstable sequences."""
    I = int(intensity)
    
    if sequence_type == 'destabilizer_heavy':
        # Multiple destabilizers with minimal stabilization
        return (
            [Emission()] +
            [Dissonance()] * I +
            [Coherence()] +
            [Expansion()] * (I - 1) +
            [SelfOrganization()] +
            [Dissonance()] * I +
            [Silence()]
        )
    
    elif sequence_type == 'mutation_cascade':
        # Cascade of mutations (phase transformations)
        return (
            [Emission()] +
            [Coherence()] +
            [Dissonance()] * 2 +
            [Mutation()] * I +
            [Coherence()] +
            [Silence()]
        )
    
    elif sequence_type == 'expansion_explosion':
        # Rapid dimensionality increase
        return (
            [Emission()] +
            [Coherence()] +
            [Expansion()] * I +
            [Dissonance()] * I +
            [SelfOrganization()] +
            [Silence()]
        )
    
    elif sequence_type == 'minimal_stabilization':
        # Barely compliant with U2
        return (
            [Emission()] +
            [Dissonance()] * (I * 2) +
            [Coherence()] +  # Just one stabilizer
            [Dissonance()] * I +
            [Silence()]
        )
    
    else:
        raise ValueError(f"Unknown sequence_type: {sequence_type}")


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
            node_phase = float(G.nodes[node].get('theta', 0.0))
            neighbor_phases = [float(G.nodes[nb].get('theta', 0.0)) for nb in neighbors]
            
            phase_diffs = [abs(node_phase - nb_phase) for nb_phase in neighbor_phases]
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


def run_stress_test(
    run_id: int,
    topology: str,
    n_nodes: int,
    intensity: float,
    sequence_type: str,
    seed: int
) -> StressTestResult:
    """Run one stress test."""
    try:
        # Create graph
        G = create_tnfr_topology(topology, n_nodes, seed)
        initialize_tnfr_nodes(G, nu_f=1.0, epi_range=(0.2, 0.8), seed=seed)
        
        # Initial measurements
        C_initial = compute_global_coherence(G)
        grad_phi_initial = compute_mean_phase_gradient(G)
        phi_s_initial = compute_mean_phi_s(G)
        
        # Generate and apply sequence
        sequence = generate_stress_sequence(sequence_type, intensity)
        
        # Apply to ALL nodes
        for target_node in G.nodes():
            run_sequence(G, target_node, sequence)
        
        # Final measurements
        C_final = compute_global_coherence(G)
        grad_phi_final = compute_mean_phase_gradient(G)
        phi_s_final = compute_mean_phi_s(G)
        
        # Compute deltas
        delta_C = C_final - C_initial
        delta_grad_phi = grad_phi_final - grad_phi_initial
        delta_phi_s = phi_s_final - phi_s_initial
        
        # Check fragmentation (C dropped significantly)
        fragmented = delta_C < -0.05
        
        return StressTestResult(
            run_id=run_id,
            topology=topology,
            n_nodes=n_nodes,
            intensity=intensity,
            sequence_type=sequence_type,
            C_initial=C_initial,
            C_final=C_final,
            grad_phi_initial=grad_phi_initial,
            grad_phi_final=grad_phi_final,
            phi_s_initial=phi_s_initial,
            phi_s_final=phi_s_final,
            delta_C=delta_C,
            delta_grad_phi=delta_grad_phi,
            delta_phi_s=delta_phi_s,
            seed=seed,
            num_operators=len(sequence),
            fragmented=fragmented,
            error_message=None
        )
    
    except Exception as e:
        return StressTestResult(
            run_id=run_id,
            topology=topology,
            n_nodes=n_nodes,
            intensity=intensity,
            sequence_type=sequence_type,
            C_initial=0.0,
            C_final=0.0,
            grad_phi_initial=0.0,
            grad_phi_final=0.0,
            phi_s_initial=0.0,
            phi_s_final=0.0,
            delta_C=0.0,
            delta_grad_phi=0.0,
            delta_phi_s=0.0,
            seed=seed,
            num_operators=0,
            fragmented=False,
            error_message=str(e)
        )


def main(argv: List[str]) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stress test")
    parser.add_argument('--runs', type=int, default=10, help='Runs per condition')
    parser.add_argument('--output', type=str, default='stress_test_results.jsonl')
    parser.add_argument('--n-nodes', type=int, default=6, help='Network size')
    
    args = parser.parse_args(argv)
    
    # Conditions
    topologies = ['ring', 'scale_free', 'ws']
    intensities = [3.0, 4.0, 5.0]  # HIGH intensity
    sequence_types = [
        'destabilizer_heavy',
        'mutation_cascade',
        'expansion_explosion',
        'minimal_stabilization'
    ]
    
    total = len(topologies) * len(intensities) * len(sequence_types) * args.runs
    print(f"Running {total} stress tests...")
    print(f"Conditions: {len(topologies)} topologies × {len(intensities)} intensities × {len(sequence_types)} sequences × {args.runs} runs")
    
    # Output
    output_path = Path(__file__).parent / args.output
    
    run_id = 0
    completed = 0
    fragmented_count = 0
    
    with output_path.open('w', encoding='utf-8') as f:
        for topology in topologies:
            for intensity in intensities:
                for sequence_type in sequence_types:
                    for run_idx in range(args.runs):
                        seed = run_id + 12345
                        
                        result = run_stress_test(
                            run_id=run_id,
                            topology=topology,
                            n_nodes=args.n_nodes,
                            intensity=intensity,
                            sequence_type=sequence_type,
                            seed=seed
                        )
                        
                        # Write result
                        f.write(json.dumps(asdict(result)) + '\n')
                        f.flush()
                        
                        run_id += 1
                        completed += 1
                        
                        if result.fragmented:
                            fragmented_count += 1
                        
                        # Progress
                        if completed % 10 == 0:
                            print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) | Fragmented: {fragmented_count}")
    
    print(f"\n✅ Completed {completed} stress tests")
    print(f"📊 Fragmented runs: {fragmented_count}/{completed} ({fragmented_count/completed*100:.1f}%)")
    print(f"📁 Results: {output_path}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]))
