"""U6 Sequence Simulator
=======================

Experimental benchmark tool to generate and compare TNFR operator sequences
that (a) comply with U6 (post-destabilizer spacing) vs (b) violate U6
(consecutive OZ/ZHIR/VAL without recovery), measuring effects on:

- C(t) trajectory (global coherence)
- ΔNFR spikes (max |ΔNFR| after destabilizer)
- Recovery time (steps until C(t) >= 0.9 * C_base post-destabilization)
- Fragmentation flag (C(t) < 0.3 sustained ≥ frag_window steps)
- τ_relax estimated (Liouvillian if available, spectral fallback)
- α_empirical ≈ τ_relax * 2π * νf (inverse of proposed formula)

Output: JSONL per line (one experiment) + summary in stdout.

Status: RESEARCH - Not integrated in canonical pipelines yet.

Usage (PowerShell):
    python benchmarks/u6_sequence_simulator.py --runs 20 --topologies star,ring,ws --seed 42 \
        --nu-freqs 0.5,1.0,2.0 --frag-window 5 --export u6_results.jsonl

Notes:
- Maintains reproducibility using fixed seeds.
- Does not modify the global persistent graph; each experiment creates its instance.
- Does not declare U6 as canonical: serves to collect evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import networkx as nx

# Import TNFR operators & metrics lazily (avoid heavy import costs al inicio)
# Ensure local 'src' is importable when running from repo root
from pathlib import Path as _Path  # local alias to avoid shadowing
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

from tnfr.metrics.common import compute_coherence  # type: ignore
from tnfr.operators.definitions import Emission, Coherence, Dissonance, Silence, Mutation, Expansion
from tnfr.operators.metrics_u6 import measure_tau_relax_observed
from tnfr.physics.fields import (  # type: ignore
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SequenceResult:
    topology: str
    n: int
    nu_f: float
    sequence_type: str  # 'valid_u6' | 'violate_u6'
    ops: List[str]
    coherence_initial: float
    coherence_final: float
    coherence_min: float
    fragmentation: bool
    recovery_steps: int
    max_dnfr_spike: float
    tau_relax: float | None
    tau_relax_spectral: float | None
    tau_relax_liouv: float | None
    slow_mode_real: float | None
    alpha_empirical: float | None
    steps: int
    min_spacing_steps: int
    destabilizer_count: int
    # Structural fields (research-phase telemetry)
    phi_s_mean_initial: float | None
    phi_s_mean_final: float | None
    grad_phi_mean_initial: float | None
    grad_phi_mean_final: float | None
    curv_phi_max_initial: float | None
    curv_phi_max_final: float | None
    xi_c_initial: float | None
    xi_c_final: float | None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_graph(topology: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    if topology == 'star':
        return nx.star_graph(n - 1)
    if topology == 'ring':
        return nx.cycle_graph(n)
    if topology == 'ws':  # small-world Watts-Strogatz
        return nx.watts_strogatz_graph(n, k=min(4, n-1), p=0.3, seed=seed)
    if topology == 'scale_free':
        return nx.scale_free_graph(n, seed=seed).to_undirected()
    if topology == 'tree':
        # Balanced binary tree (approximation for n nodes)
        # Use k-ary tree with branching factor 2
        import math
        height = max(1, int(math.log2(n)))
        G = nx.balanced_tree(r=2, h=height)
        # Trim to exactly n nodes if needed
        if G.number_of_nodes() > n:
            nodes_to_remove = list(G.nodes)[n:]
            G.remove_nodes_from(nodes_to_remove)
        return G
    if topology == 'grid':
        # 2D grid lattice (approximately sqrt(n) x sqrt(n))
        import math
        side = max(2, int(math.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        # Convert node labels to integers for consistency
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        # Trim to exactly n nodes if needed
        if G.number_of_nodes() > n:
            nodes_to_remove = list(G.nodes)[n:]
            G.remove_nodes_from(nodes_to_remove)
        return G
    raise ValueError(f"Unknown topology: {topology}")

def set_initial_state(G: nx.Graph, nu_f: float, seed: int) -> None:
    random.seed(seed)
    # Minimal attribute initialization for metrics
    for node in G.nodes:
        G.nodes[node]['vf'] = nu_f  # structural frequency
        # small random ΔNFR baseline
        G.nodes[node]['dnfr'] = random.uniform(0.01, 0.05)
        G.nodes[node]['delta_nfr'] = random.uniform(0.01, 0.05)  # alias for fields.py
        G.nodes[node]['depi'] = random.uniform(0.01, 0.05)
        G.nodes[node]['epi'] = random.uniform(0.4, 0.7)
        # Phase for phase gradient/curvature
        G.nodes[node]['phase'] = random.uniform(0.0, 6.283)
        # Coherence for coherence length
        G.nodes[node]['coherence'] = random.uniform(0.5, 0.9)

# Rough ΔNFR perturbation when applying destabilizers
def apply_operator(G: nx.Graph, op_name: str, seed: int, intensity: float = 1.0) -> None:
    """Apply operator with adjustable intensity multiplier."""
    random.seed(seed)
    if op_name == 'emission':
        for n in G.nodes:  # mimic activation
            G.nodes[n]['epi'] += random.uniform(0.05, 0.12) * intensity
            G.nodes[n]['vf'] = max(G.nodes[n]['vf'], random.uniform(0.8, 1.2) * intensity)
    elif op_name == 'coherence':
        for n in G.nodes:
            # Reduce dnfr and depi (stabilization)
            G.nodes[n]['dnfr'] *= (0.7 / intensity)
            G.nodes[n]['delta_nfr'] *= (0.7 / intensity)
            G.nodes[n]['depi'] *= (0.7 / intensity)
    elif op_name == 'dissonance':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.2, 0.4) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.2, 0.4) * intensity
            G.nodes[n]['depi'] += random.uniform(0.15, 0.3) * intensity
            # Perturb phase to create gradient
            G.nodes[n]['phase'] = (G.nodes[n]['phase'] + random.uniform(-0.5, 0.5) * intensity) % (2 * math.pi)
    elif op_name == 'mutation':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.25, 0.5) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.25, 0.5) * intensity
            G.nodes[n]['depi'] += random.uniform(0.2, 0.35) * intensity
            # phase-like surrogate with large jump
            G.nodes[n]['phase'] = (G.nodes[n]['phase'] + random.uniform(-math.pi/2, math.pi/2) * intensity) % (2 * math.pi)
            G.nodes[n]['theta'] = G.nodes[n]['phase']
    elif op_name == 'expansion':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.15, 0.3) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.15, 0.3) * intensity
            G.nodes[n]['depi'] += random.uniform(0.1, 0.25) * intensity
            G.nodes[n]['epi'] += random.uniform(0.02, 0.05) * intensity
    elif op_name == 'silence':
        for n in G.nodes:
            # Freeze evolution (vf lowered slightly)
            G.nodes[n]['vf'] *= (0.95 / intensity)
            G.nodes[n]['dnfr'] *= (0.85 / intensity)
            G.nodes[n]['delta_nfr'] *= (0.85 / intensity)
            G.nodes[n]['depi'] *= (0.85 / intensity)
    else:
        raise ValueError(f"Unknown operator name: {op_name}")

# Generate pair of sequences: valid vs violation
# Valid ensures spacing (coherence/self organization) between destabilizers
# Violation stacks destabilizers consecutively

def generate_sequences() -> Tuple[List[str], List[str]]:
    valid = [
        'emission', 'dissonance', 'coherence', 'expansion', 'coherence', 'dissonance', 'coherence', 'silence'
    ]
    violate = [
        'emission', 'dissonance', 'dissonance', 'mutation', 'expansion', 'coherence', 'silence'
    ]
    return valid, violate

def generate_aggressive_sequences() -> Tuple[List[str], List[str]]:
    """Generate more aggressive sequences to trigger bifurcations and fragmentation."""
    valid_aggressive = [
        'emission', 'dissonance', 'coherence', 'dissonance', 'coherence', 
        'dissonance', 'coherence', 'expansion', 'coherence', 'silence'
    ]
    violate_aggressive = [
        'emission', 'dissonance', 'dissonance', 'dissonance', 'expansion', 
        'mutation', 'mutation', 'coherence', 'silence'
    ]
    return valid_aggressive, violate_aggressive

# Compute minimum spacing in steps between destabilizers in a sequence
def min_destabilizer_spacing(ops: List[str]) -> Tuple[int, int]:
    dest = {'dissonance', 'mutation', 'expansion'}
    idxs = [i for i, op in enumerate(ops) if op in dest]
    if len(idxs) <= 1:
        return (len(ops), len(idxs))
    mins = min(idxs[i+1] - idxs[i] for i in range(len(idxs)-1))
    return (mins, len(idxs))

# Measure recovery: steps until coherence back near baseline proportion

def measure_recovery(coherence_series: List[float], base: float, threshold_ratio: float = 0.9) -> int:
    target = base * threshold_ratio
    for i, c in enumerate(coherence_series):
        if c >= target:
            return i
    return -1  # not recovered

# Fragmentation: coherence < frag_level for >= window consecutive steps

def detect_fragmentation(series: List[float], frag_level: float = 0.3, window: int = 5) -> bool:
    run = 0
    for c in series:
        if c < frag_level:
            run += 1
            if run >= window:
                return True
        else:
            run = 0
    return False

# Extract max dnfr spike (approx) from node attributes

def max_dnfr(G: nx.Graph) -> float:
    return max(abs(G.nodes[n].get('dnfr', 0.0)) for n in G.nodes) if G.number_of_nodes() else 0.0

# Estimate τ_relax using experimental U6 metric on a representative node

def estimate_tau_relax(G: nx.Graph) -> Tuple[float|None, float|None, float|None, float|None]:
    # Pick central node (max degree) as representative
    if G.number_of_nodes() == 0:
        return None, None, None, None
    # NetworkX DegreeView is iterable; cast to list for type checkers
    # Work around static type complaints: explicitly list degrees
    # Build degree list explicitly to avoid typing issues with DegreeView
    nodes = list(G.nodes)
    degs: List[Tuple[object, int]] = [(n, int(len(list(G.neighbors(n))))) for n in nodes]
    degs.sort(key=lambda x: x[1], reverse=True)
    node = degs[0][0]
    try:
        result = measure_tau_relax_observed(G, node)
        return (
            result.get('estimated_tau_relax'),
            result.get('estimated_tau_relax_spectral'),
            result.get('estimated_tau_relax_liouvillian'),
            result.get('liouvillian_slow_mode_real'),
        )
    except Exception:
        return None, None, None, None

# Compute structural fields (research-phase)
def compute_fields(G: nx.Graph) -> Tuple[float|None, float|None, float|None, float|None]:
    try:
        phi_s = compute_structural_potential(G)  # type: ignore[arg-type]
        grad = compute_phase_gradient(G)  # type: ignore[arg-type]
        curv = compute_phase_curvature(G)  # type: ignore[arg-type]
        xi_c = estimate_coherence_length(G)  # type: ignore[arg-type]
        
        phi_s_mean = sum(phi_s.values()) / max(len(phi_s), 1) if phi_s else None
        grad_mean = sum(grad.values()) / max(len(grad), 1) if grad else None
        curv_max = max(abs(v) for v in curv.values()) if curv else None
        
        return phi_s_mean, grad_mean, curv_max, xi_c
    except Exception:
        return None, None, None, None

# Main execution of a sequence over graph state

def run_sequence(G: nx.Graph, ops: List[str], base_seed: int, intensity: float = 1.0) -> Tuple[List[float], int]:
    coherence_series: List[float] = []
    for step, op in enumerate(ops):
        apply_operator(G, op, seed=base_seed + step, intensity=intensity)
        # compute_coherence expects GraphLike protocol; NetworkX Graph provides required attributes
        c = float(compute_coherence(G))  # type: ignore[arg-type]
        coherence_series.append(c)
    return coherence_series, len(ops)

# ---------------------------------------------------------------------------
# Core Simulation Logic
# ---------------------------------------------------------------------------

def simulate(topology: str, n: int, nu_f: float, seed: int, frag_window: int, aggressive: bool = False, intensity: float = 1.0) -> Tuple[SequenceResult, SequenceResult]:
    if aggressive:
        valid_ops, violate_ops = generate_aggressive_sequences()
    else:
        valid_ops, violate_ops = generate_sequences()

    # Base graph for both sequences (clone via copy)
    G_base = make_graph(topology, n, seed)
    set_initial_state(G_base, nu_f, seed)
    coherence_initial = float(compute_coherence(G_base))  # type: ignore[arg-type]
    
    # Compute initial structural fields
    phi_s_init, grad_phi_init, curv_phi_init, xi_c_init = compute_fields(G_base)

    # VALID sequence
    G_valid = G_base.copy()
    series_valid, steps_valid = run_sequence(G_valid, valid_ops, seed * 100, intensity=intensity)
    coherence_final_valid = series_valid[-1]
    coherence_min_valid = min(series_valid)
    recovery_valid = measure_recovery(series_valid, coherence_initial)
    frag_valid = detect_fragmentation(series_valid, window=frag_window)
    max_dnfr_valid = max_dnfr(G_valid)
    tau_relax_valid, tau_spectral_valid, tau_liouv_valid, slow_mode_real_valid = estimate_tau_relax(G_valid)
    alpha_emp_valid = None
    if tau_relax_valid is not None:
        # α ≈ τ_relax * 2π * νf (heurística inversa de τ = α/(2π νf))
        alpha_emp_valid = tau_relax_valid * 2 * math.pi * nu_f
    
    # Compute final structural fields for valid sequence
    phi_s_final_valid, grad_phi_final_valid, curv_phi_final_valid, xi_c_final_valid = compute_fields(G_valid)

    min_space_valid, dest_count_valid = min_destabilizer_spacing(valid_ops)
    res_valid = SequenceResult(
        topology=topology,
        n=n,
        nu_f=nu_f,
        sequence_type='valid_u6',
        ops=valid_ops,
        coherence_initial=coherence_initial,
        coherence_final=coherence_final_valid,
        coherence_min=coherence_min_valid,
        fragmentation=frag_valid,
        recovery_steps=recovery_valid,
        max_dnfr_spike=max_dnfr_valid,
        tau_relax=tau_relax_valid,
        tau_relax_spectral=tau_spectral_valid,
        tau_relax_liouv=tau_liouv_valid,
        slow_mode_real=slow_mode_real_valid,
        alpha_empirical=alpha_emp_valid,
        steps=steps_valid,
        min_spacing_steps=min_space_valid,
        destabilizer_count=dest_count_valid,
        phi_s_mean_initial=phi_s_init,
        phi_s_mean_final=phi_s_final_valid,
        grad_phi_mean_initial=grad_phi_init,
        grad_phi_mean_final=grad_phi_final_valid,
        curv_phi_max_initial=curv_phi_init,
        curv_phi_max_final=curv_phi_final_valid,
        xi_c_initial=xi_c_init,
        xi_c_final=xi_c_final_valid,
    )

    # VIOLATION sequence
    G_violate = G_base.copy()
    series_violate, steps_violate = run_sequence(G_violate, violate_ops, seed * 200, intensity=intensity)
    coherence_final_violate = series_violate[-1]
    coherence_min_violate = min(series_violate)
    recovery_violate = measure_recovery(series_violate, coherence_initial)
    frag_violate = detect_fragmentation(series_violate, window=frag_window)
    max_dnfr_violate = max_dnfr(G_violate)
    tau_relax_violate, tau_spectral_violate, tau_liouv_violate, slow_mode_real_violate = estimate_tau_relax(G_violate)
    alpha_emp_violate = None
    if tau_relax_violate is not None:
        alpha_emp_violate = tau_relax_violate * 2 * math.pi * nu_f
    
    # Compute final structural fields for violation sequence
    phi_s_final_violate, grad_phi_final_violate, curv_phi_final_violate, xi_c_final_violate = compute_fields(G_violate)

    min_space_violate, dest_count_violate = min_destabilizer_spacing(violate_ops)
    res_violate = SequenceResult(
        topology=topology,
        n=n,
        nu_f=nu_f,
        sequence_type='violate_u6',
        ops=violate_ops,
        coherence_initial=coherence_initial,
        coherence_final=coherence_final_violate,
        coherence_min=coherence_min_violate,
        fragmentation=frag_violate,
        recovery_steps=recovery_violate,
        max_dnfr_spike=max_dnfr_violate,
        tau_relax=tau_relax_violate,
        tau_relax_spectral=tau_spectral_violate,
        tau_relax_liouv=tau_liouv_violate,
        slow_mode_real=slow_mode_real_violate,
        alpha_empirical=alpha_emp_violate,
        steps=steps_violate,
        min_spacing_steps=min_space_violate,
        destabilizer_count=dest_count_violate,
        phi_s_mean_initial=phi_s_init,
        phi_s_mean_final=phi_s_final_violate,
        grad_phi_mean_initial=grad_phi_init,
        grad_phi_mean_final=grad_phi_final_violate,
        curv_phi_max_initial=curv_phi_init,
        curv_phi_max_final=curv_phi_final_violate,
        xi_c_initial=xi_c_init,
        xi_c_final=xi_c_final_violate,
    )

    return res_valid, res_violate

# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate TNFR sequences for U6 research.")
    p.add_argument('--runs', type=int, default=10, help='Number of runs per topology')
    p.add_argument('--topologies', type=str, default='star,ring,ws,scale_free', help='Comma-separated list')
    p.add_argument('--sizes', type=str, default='20', help='Graph sizes (comma-separated)')
    p.add_argument('--nu-freqs', type=str, default='1.0', help='νf values (comma-separated)')
    p.add_argument('--seed', type=int, default=42, help='Global base seed')
    p.add_argument('--frag-window', type=int, default=5, help='Consecutive window for fragmentation')
    p.add_argument('--export', type=str, default='u6_results.jsonl', help='Output JSONL path')
    p.add_argument('--aggressive', action='store_true', help='Use aggressive sequences (triple destabilizers)')
    p.add_argument('--intensity', type=float, default=1.0, help='Operator intensity multiplier (>1.0 for extreme stress)')
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    topologies = [t.strip() for t in args.topologies.split(',') if t.strip()]
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
    # Parse νf values
    if isinstance(args.nu_freqs, str):
        nu_freqs = [float(f) for f in args.nu_freqs.split(',') if f.strip()]
    else:
        nu_freqs = [float(args.nu_freqs)]

    out_path = Path(args.export)
    f_out = out_path.open('w', encoding='utf-8')

    summary_counts = {
        'total': 0,
        'fragmentations_valid': 0,
        'fragmentations_violate': 0,
        'recovery_fail_valid': 0,
        'recovery_fail_violate': 0,
    }

    for topology in topologies:
        for n in sizes:
            for nu_f in nu_freqs:
                for run in range(args.runs):
                    seed = args.seed + run + n
                    try:
                        res_valid, res_violate = simulate(topology, n, nu_f, seed, args.frag_window, args.aggressive, args.intensity)
                    except Exception as e:
                        print(f"[WARN] Simulation failed topo={topology} n={n} nu_f={nu_f}: {e}")
                        continue
                    for res in (res_valid, res_violate):
                        f_out.write(json.dumps(asdict(res), ensure_ascii=False) + '\n')
                        summary_counts['total'] += 1
                    if res_valid.fragmentation:
                        summary_counts['fragmentations_valid'] += 1
                    if res_violate.fragmentation:
                        summary_counts['fragmentations_violate'] += 1
                    if res_valid.recovery_steps == -1:
                        summary_counts['recovery_fail_valid'] += 1
                    if res_violate.recovery_steps == -1:
                        summary_counts['recovery_fail_violate'] += 1

    f_out.close()

    print("\n=== U6 Sequence Simulation Summary ===")
    print(f"Output file: {out_path}")
    print(f"Total experiments: {summary_counts['total']}")
    print(f"Fragmentations (valid): {summary_counts['fragmentations_valid']}")
    print(f"Fragmentations (violate): {summary_counts['fragmentations_violate']}")
    print(f"Recovery failures (valid): {summary_counts['recovery_fail_valid']}")
    print(f"Recovery failures (violate): {summary_counts['recovery_fail_violate']}")

    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
