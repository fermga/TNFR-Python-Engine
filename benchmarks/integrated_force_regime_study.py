"""Integrated Force Regime Orchestration Study
================================================

This benchmark orchestrates a unified, multi-task study of the Structural
Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) across topologies and operator regimes.

Tasks implemented (exported as JSONL records):
- Task 1: Field Interaction Matrix (per-snapshot correlations)
- Task 2: Composite Field Metrics (stress indices)
- Task 3: Force Regime Phase Diagram (intensity sweep and regime labels)
- Task 4: Temporal Orchestration Analysis (lead/lag among fields)
- Task 5: Operator-Field Coupling Analysis (per-operator deltas)
- Task 6: Cross-Domain Unification Test (consistency across topologies)

Status: Research harness for unified analysis; leverages CANONICAL fields.

Usage (PowerShell):
    python benchmarks/integrated_force_regime_study.py \
        --topologies ring,ws,scale_free,grid --sizes 30 \
        --runs 5 --seed 42 --export results/integrated_force_study.jsonl

The script writes JSONL lines, one per measurement/event, with a `task`
field indicating the task type. It avoids side effects on the codebase.
"""
# flake8: noqa
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import networkx as nx

# Ensure local src is importable when running from repo root
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))

# Import canonical field computations
from tnfr.physics.fields import (  # type: ignore  # noqa: E402
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

# ---------------------------------------------------------------------------
# Minimal graph helpers (self-contained; aligned with u6_sequence_simulator)
# ---------------------------------------------------------------------------


def make_graph(topology: str, n: int, seed: int) -> nx.Graph:
    random.seed(seed)
    if topology == 'ring':
        return nx.cycle_graph(n)
    if topology == 'ws':
        return nx.watts_strogatz_graph(n, k=min(4, n-1), p=0.3, seed=seed)
    if topology == 'scale_free':
        return nx.scale_free_graph(n, seed=seed).to_undirected()
    if topology == 'grid':
        import math as _m
        side = max(2, int(_m.sqrt(n)))
        G = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping)
    if topology == 'tree':
        G = nx.balanced_tree(r=2, h=max(1, int(math.log2(n))))
        if G.number_of_nodes() > n:
            nodes_to_remove = list(G.nodes())[n:]
            G.remove_nodes_from(nodes_to_remove)
        return G
    if topology == 'star':
        return nx.star_graph(n - 1)
    raise ValueError(f"Unknown topology: {topology}")


def set_initial_state(G: nx.Graph, nu_f: float, seed: int) -> None:
    random.seed(seed)
    for node in G.nodes:
        G.nodes[node]['vf'] = float(nu_f)
        G.nodes[node]['dnfr'] = random.uniform(0.01, 0.05)
        G.nodes[node]['delta_nfr'] = random.uniform(0.01, 0.05)
        G.nodes[node]['epi'] = random.uniform(0.4, 0.7)
        G.nodes[node]['phase'] = random.uniform(0.0, 2 * math.pi)
        G.nodes[node]['theta'] = G.nodes[node]['phase']
        # Coherence surrogate for ξ_C estimation
        G.nodes[node]['coherence'] = random.uniform(0.5, 0.9)


def apply_operator_like(
    G: nx.Graph,
    op: str,
    seed: int,
    intensity: float = 1.0,
) -> None:
    """Lightweight attribute perturbations mimicking operator effects.
    Read-only fields consumption happens outside; this only sets attributes
    consistent with telemetry expectations.
    """
    random.seed(seed)
    if op == 'emission':
        for n in G.nodes:
            G.nodes[n]['epi'] += random.uniform(0.05, 0.12) * intensity
            G.nodes[n]['vf'] = max(
                G.nodes[n]['vf'],
                random.uniform(0.8, 1.2) * intensity,
            )
    elif op == 'coherence':
        for n in G.nodes:
            G.nodes[n]['dnfr'] *= (0.7 / max(intensity, 1e-6))
            G.nodes[n]['delta_nfr'] *= (0.7 / max(intensity, 1e-6))
    elif op == 'dissonance':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.2, 0.4) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.2, 0.4) * intensity
            G.nodes[n]['phase'] = (
                G.nodes[n]['phase'] + random.uniform(-0.5, 0.5) * intensity
            ) % (2 * math.pi)
            G.nodes[n]['theta'] = G.nodes[n]['phase']
    elif op == 'mutation':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.25, 0.5) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.25, 0.5) * intensity
            G.nodes[n]['phase'] = (
                G.nodes[n]['phase']
                + random.uniform(-math.pi / 2, math.pi / 2) * intensity
            ) % (2 * math.pi)
            G.nodes[n]['theta'] = G.nodes[n]['phase']
    elif op == 'expansion':
        for n in G.nodes:
            G.nodes[n]['dnfr'] += random.uniform(0.15, 0.3) * intensity
            G.nodes[n]['delta_nfr'] += random.uniform(0.15, 0.3) * intensity
    elif op == 'silence':
        for n in G.nodes:
            G.nodes[n]['vf'] *= (0.95 / max(intensity, 1e-6))
            G.nodes[n]['dnfr'] *= (0.85 / max(intensity, 1e-6))
            G.nodes[n]['delta_nfr'] *= (0.85 / max(intensity, 1e-6))
    else:
        raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# Task 1: Field Interaction Matrix
# ---------------------------------------------------------------------------

def field_interaction_matrix(G: nx.Graph) -> Dict[str, Any]:
    phi_s = compute_structural_potential(G)  # Dict[node, float]
    grad = compute_phase_gradient(G)
    curv = compute_phase_curvature(G)
    dnfr = {n: float(G.nodes[n].get('delta_nfr', 0.0)) for n in G.nodes()}
    # local coherence proxy used by ξ_C
    coh_local = {n: 1.0 / (1.0 + abs(dnfr[n])) for n in G.nodes()}

    # Align vectors
    nodes = list(G.nodes())
    X = np.vstack([
        np.array([phi_s[n] for n in nodes], dtype=float),
        np.array([grad[n] for n in nodes], dtype=float),
        np.array([abs(curv[n]) for n in nodes], dtype=float),
        np.array([coh_local[n] for n in nodes], dtype=float),
        np.array([dnfr[n] for n in nodes], dtype=float),
    ])  # shape (5, N)

    labels = ["phi_s", "grad_phi", "abs_k_phi", "coh_local", "dnfr"]

    # Correlation matrix (Pearson)
    if X.shape[1] >= 3:
        C = np.corrcoef(X)
        corr = {
            f"{labels[i]}__{labels[j]}": float(C[i, j])
            for i in range(len(labels))
            for j in range(len(labels))
        }
    else:
        corr = {f"{a}__{b}": 0.0 for a in labels for b in labels}

    xi_c = float(estimate_coherence_length(G))

    return {
        "task": "field_interaction_matrix",
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "corr": corr,
        "xi_c": xi_c,
        "means": {
            "phi_s": float(np.mean(X[0])) if X.shape[1] else 0.0,
            "grad_phi": float(np.mean(X[1])) if X.shape[1] else 0.0,
            "abs_k_phi": float(np.mean(X[2])) if X.shape[1] else 0.0,
            "coh_local": float(np.mean(X[3])) if X.shape[1] else 0.0,
            "dnfr": float(np.mean(X[4])) if X.shape[1] else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Task 2: Composite Field Metrics
# ---------------------------------------------------------------------------

def composite_field_metrics(G: nx.Graph) -> Dict[str, Any]:
    phi_s = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    kphi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    nodes = list(G.nodes())
    if not nodes:
        return {
            "task": "composite_field_metrics",
            "S_local": 0.0,
            "S_global": 0.0,
        }

    v_grad = np.array([grad[n] for n in nodes], dtype=float)
    v_k = np.array([abs(kphi[n]) for n in nodes], dtype=float)
    v_phi = np.array([phi_s[n] for n in nodes], dtype=float)

    # Z-score helper with guard
    def z(x: np.ndarray) -> np.ndarray:
        mu, sd = float(np.mean(x)), float(np.std(x))
        return (x - mu) / (sd + 1e-9)

    # Local stress: gradient + curvature (node-wise), summarized by mean
    s_local_vec = z(v_grad) + z(v_k)
    S_local = float(np.mean(s_local_vec))

    # Global stress: global potential (mean) + normalized xi_c by diameter
    # Normalize xi_c to [0, ..] by graph diameter (topological)
    try:
        diam = (
            nx.diameter(G)
            if nx.is_connected(G)
            else 1
            + max(
                (
                    nx.diameter(CC)
                    for CC in (
                        G.subgraph(c).copy() for c in nx.connected_components(G)
                    )
                ),
                default=1,
            )
        )
    except Exception:
        diam = max(1, int(np.ceil(np.log2(max(1, G.number_of_nodes())))))

    xi_norm = float(xi_c) / max(1e-9, float(diam))
    # Higher xi_norm suggests critical/global correlation; add with mean phi_s
    S_global = float(np.mean(z(v_phi))) + xi_norm

    return {
        "task": "composite_field_metrics",
        "S_local": S_local,
        "S_global": S_global,
        "xi_c": float(xi_c),
        "xi_norm": float(xi_norm),
        "diameter": float(diam),
    }


# ---------------------------------------------------------------------------
# Task 3: Force Regime Phase Diagram (Intensity sweep)
# ---------------------------------------------------------------------------

def classify_regime(G: nx.Graph) -> str:
    # Thresholds per canonical docs
    grad = compute_phase_gradient(G)
    kphi = compute_phase_curvature(G)
    xi_c = estimate_coherence_length(G)

    grad_mean = float(np.mean(list(grad.values()))) if grad else 0.0
    kphi_max = float(np.max(np.abs(list(kphi.values())))) if kphi else 0.0

    # Diameter for xi_C normalization
    try:
        diam = (
            nx.diameter(G)
            if nx.is_connected(G)
            else 1
            + max(
                (
                    nx.diameter(CC)
                    for CC in (
                        G.subgraph(c).copy() for c in nx.connected_components(G)
                    )
                ),
                default=1,
            )
        )
    except Exception:
        diam = max(1, int(np.ceil(np.log2(max(1, G.number_of_nodes())))))

    xi_norm = float(xi_c) / max(1e-9, float(diam))

    # Simple regime labeling
    if (grad_mean < 0.38) and (kphi_max < 3.0) and (xi_norm < 1.0):
        return "stable_localized"
    if xi_norm >= 1.0:
        return "critical_global"
    if (grad_mean >= 0.38) or (kphi_max >= 3.0):
        return "high_stress"
    return "unknown"


def force_regime_phase_diagram(
    topology: str,
    n: int,
    seed: int,
    intensities: List[float],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for intensity_val in intensities:
        G = make_graph(topology, n, seed + int(intensity_val * 1000))
        set_initial_state(G, nu_f=1.0, seed=seed + 7)
        # Apply a small sequence influenced by intensity
        ops = ['emission', 'dissonance', 'coherence', 'expansion', 'silence']
        for step, op in enumerate(ops):
            apply_operator_like(
                G,
                op,
                seed=seed + 13 + step,
                intensity=float(intensity_val),
            )
        regime = classify_regime(G)
        comp = composite_field_metrics(G)
        out = {
            "task": "force_regime_phase_diagram",
            "topology": topology,
            "n": n,
            "intensity": float(intensity_val),
            "regime": regime,
            **{k: v for k, v in comp.items() if k != 'task'},
        }
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Task 4: Temporal Orchestration Analysis (lead/lag)
# ---------------------------------------------------------------------------

def temporal_orchestration_series(
    G: nx.Graph, seed: int, intensity: float = 1.0
) -> Dict[str, Any]:
    ops = [
        'emission',
        'dissonance',
        'coherence',
        'expansion',
        'coherence',
        'silence',
    ]
    series = {
        "phi_s_mean": [],
        "grad_phi_mean": [],
        "k_phi_max": [],
        "xi_c": [],
    }
    for step, op in enumerate(ops):
        apply_operator_like(G, op, seed=seed + step, intensity=intensity)
        phi_s = compute_structural_potential(G)
        grad = compute_phase_gradient(G)
        kphi = compute_phase_curvature(G)
        xi_c = estimate_coherence_length(G)
        series["phi_s_mean"].append(
            float(np.mean(list(phi_s.values()))) if phi_s else 0.0
        )
        series["grad_phi_mean"].append(
            float(np.mean(list(grad.values()))) if grad else 0.0
        )
        series["k_phi_max"].append(
            float(np.max(np.abs(list(kphi.values())))) if kphi else 0.0
        )
        series["xi_c"].append(float(xi_c))

    # Compute pairwise lead/lag via argmax cross-correlation
    def lead_lag(a: List[float], b: List[float]) -> int:
        x = np.array(a, dtype=float)
        y = np.array(b, dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-9)
        y = (y - y.mean()) / (y.std() + 1e-9)
        # Use explicit dot with rolled vector to avoid NumPy scalar-cast deprecation
        # Scaling is consistent across lags; argmax unaffected by constant factors
        L = len(x)
        lags = range(-L + 1, L)
        corrs = [float(np.dot(x, np.roll(y, lag))) for lag in lags]
        best = int(np.argmax(corrs)) - (L - 1)
        return best

    keys = list(series.keys())
    lead_lag_matrix: Dict[str, Dict[str, int]] = {k: {} for k in keys}
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                lead_lag_matrix[ki][kj] = 0
            else:
                lead_lag_matrix[ki][kj] = lead_lag(series[ki], series[kj])

    return {
        "task": "temporal_orchestration",
        "ops": ops,
        "series": series,
        "lead_lag": lead_lag_matrix,
    }


# ---------------------------------------------------------------------------
# Task 5: Operator-Field Coupling Analysis
# ---------------------------------------------------------------------------

def operator_field_coupling(topology: str, n: int, seed: int, repeats: int = 10, intensity: float = 1.0) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    op_names = ['emission', 'coherence', 'dissonance', 'mutation', 'expansion', 'silence']
    for op in op_names:
        deltas: List[Dict[str, float]] = []
        for r in range(repeats):
            G = make_graph(topology, n, seed + r)
            set_initial_state(G, nu_f=1.0, seed=seed + 3)
            # Before
            phi_b = compute_structural_potential(G)
            g_b = compute_phase_gradient(G)
            k_b = compute_phase_curvature(G)
            xi_b = float(estimate_coherence_length(G))
            # Apply operator-like perturbation
            apply_operator_like(G, op, seed=seed + 11 + r, intensity=intensity)
            # After
            phi_a = compute_structural_potential(G)
            g_a = compute_phase_gradient(G)
            k_a = compute_phase_curvature(G)
            xi_a = float(estimate_coherence_length(G))

            d_phi = float(np.mean(list(phi_a.values()))) - float(np.mean(list(phi_b.values())))
            d_g = float(np.mean(list(g_a.values()))) - float(np.mean(list(g_b.values())))
            d_k = float(np.max(np.abs(list(k_a.values())))) - float(np.max(np.abs(list(k_b.values()))))
            d_xi = float(xi_a - xi_b)
            deltas.append({"d_phi_s": d_phi, "d_grad_phi": d_g, "d_abs_k_phi_max": d_k, "d_xi_c": d_xi})

        # Aggregate
        agg = {
            "task": "operator_field_coupling",
            "topology": topology,
            "n": n,
            "operator": op,
            "intensity": float(intensity),
            "mean_deltas": {k: float(np.mean([d[k] for d in deltas])) for k in deltas[0].keys()} if deltas else {},
            "std_deltas": {k: float(np.std([d[k] for d in deltas])) for k in deltas[0].keys()} if deltas else {},
        }
        results.append(agg)
    return results


# ---------------------------------------------------------------------------
# Task 6: Cross-Domain Unification Test
# ---------------------------------------------------------------------------

def cross_domain_unification(topologies: List[str], n: int, seed: int) -> Dict[str, Any]:
    signatures: Dict[str, Dict[str, float]] = {}
    for topo in topologies:
        G = make_graph(topo, n, seed + hash(topo) % 1000)
        set_initial_state(G, nu_f=1.0, seed=seed + 5)
        # Interaction matrix as signature
        fim = field_interaction_matrix(G)
        # Select a stable subset of correlations
        corr = fim["corr"]
        sig = {
            "phi_s__coh_local": float(corr.get("phi_s__coh_local", 0.0)),
            "grad_phi__abs_k_phi": float(corr.get("grad_phi__abs_k_phi", 0.0)),
            "grad_phi__dnfr": float(corr.get("grad_phi__dnfr", 0.0)),
            "abs_k_phi__dnfr": float(corr.get("abs_k_phi__dnfr", 0.0)),
        }
        signatures[topo] = sig
    # Measure spread (max deviation across topologies) to assess unification
    keys = list(next(iter(signatures.values())).keys()) if signatures else []
    spread = {
        k: float(max(signatures[t][k] for t in topologies) - min(signatures[t][k] for t in topologies))
        for k in keys
    } if signatures else {}

    return {
        "task": "cross_domain_unification",
        "n": n,
        "topologies": topologies,
        "signatures": signatures,
        "spread": spread,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated Force Regime Orchestration Study")
    p.add_argument('--topologies', type=str, default='ring,ws,scale_free,grid')
    p.add_argument('--sizes', type=str, default='30')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--export', type=str, default='integrated_force_study.jsonl')
    p.add_argument('--intensity-sweep', type=str, default='0.5,0.8,1.0,1.2,1.5,2.0')
    p.add_argument('--op-intensity', type=float, default=1.0)
    p.add_argument('--coupling-repeats', type=int, default=10)
    return p.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    topologies = [t.strip() for t in args.topologies.split(',') if t.strip()]
    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
    intensities = [float(x) for x in args.intensity_sweep.split(',') if x.strip()]

    out_path = Path(args.export)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f_out:
        for topo in topologies:
            for n in sizes:
                for run in range(args.runs):
                    seed = args.seed + run + n
                    # Fresh graph for snapshot tasks
                    G = make_graph(topo, n, seed)
                    set_initial_state(G, nu_f=1.0, seed=seed)

                    # Task 1
                    rec1 = field_interaction_matrix(G)
                    rec1.update({"topology": topo, "task": "field_interaction_matrix", "n": n})
                    f_out.write(json.dumps(rec1) + "\n")

                    # Task 2
                    rec2 = composite_field_metrics(G)
                    rec2.update({"topology": topo, "n": n})
                    f_out.write(json.dumps(rec2) + "\n")

                    # Task 3 (intensity sweep)
                    for rec3 in force_regime_phase_diagram(topo, n, seed, intensities):
                        f_out.write(json.dumps(rec3) + "\n")

                    # Task 4 (temporal orchestration)
                    G_t = make_graph(topo, n, seed + 99)
                    set_initial_state(G_t, nu_f=1.0, seed=seed + 123)
                    rec4 = temporal_orchestration_series(G_t, seed=seed + 777, intensity=args.op_intensity)
                    rec4.update({"topology": topo, "n": n})
                    f_out.write(json.dumps(rec4) + "\n")

                    # Task 5 (operator-field coupling)
                    for rec5 in operator_field_coupling(topo, n, seed, repeats=args.coupling_repeats, intensity=args.op_intensity):
                        f_out.write(json.dumps(rec5) + "\n")

                # Task 6 (cross-domain unification done once per size)
                rec6 = cross_domain_unification(topologies, n, args.seed + 2025)
                f_out.write(json.dumps(rec6) + "\n")

    print(f"Integrated study complete. Output: {out_path}")
    return 0


if __name__ == '__main__':  # pragma: no cover
    import sys
    raise SystemExit(main(sys.argv[1:]))
