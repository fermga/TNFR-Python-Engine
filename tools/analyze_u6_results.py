"""Analyze U6 simulation results (JSONL) and print summary statistics.

Usage (PowerShell):
    python tools/analyze_u6_results.py --file u6_results_battery.jsonl

Outputs:
- Fragmentation rates by sequence_type
- Mean recovery_steps and coherence_min by sequence_type
- Pearson correlation (point-biserial) between fragmentation and:
    - min_spacing_steps
    - tau_relax (when available)
- Alpha empirical distribution by topology and nu_f (mean ± std)

No external deps required (pure Python).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze U6 results JSONL")
    p.add_argument('--file', type=str, required=True, help='Path to results JSONL')
    return p.parse_args(list(argv))


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return m, sqrt(v)


def pearson(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = sqrt(sum((yi - my) ** 2 for yi in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def analyze(rows: List[dict]) -> None:
    # Basic splits
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_type[r.get('sequence_type', 'unknown')].append(r)

    # Fragmentation and recovery stats
    print("\n=== Fragmentation & Recovery ===")
    for stype, lst in by_type.items():
        frag_rate = sum(1 for r in lst if r.get('fragmentation')) / max(1, len(lst))
        rec_vals = [r.get('recovery_steps', -1) for r in lst]
        coh_min_vals = [float(r.get('coherence_min', 0.0)) for r in lst]
        rec_mean = sum(v for v in rec_vals if v >= 0) / max(1, sum(1 for v in rec_vals if v >= 0))
        coh_min_mean = sum(coh_min_vals) / max(1, len(coh_min_vals))
        print(f"{stype:>12}: N={len(lst):4d} | frag={frag_rate*100:5.1f}% | rec_mean={rec_mean:6.2f} | coh_min={coh_min_mean:6.3f}")

    # Correlations (point-biserial) between fragmentation (0/1) and predictors
    print("\n=== Correlations (point-biserial) ===")
    frag_all: List[float] = [1.0 if r.get('fragmentation') else 0.0 for r in rows]
    spacing_all: List[float] = [float(r.get('min_spacing_steps', 0)) for r in rows]
    tau_all: List[float] = [float(r['tau_relax']) for r in rows if r.get('tau_relax') is not None]
    frag_for_tau: List[float] = [1.0 if r.get('fragmentation') else 0.0 for r in rows if r.get('tau_relax') is not None]

    corr_spacing = pearson(spacing_all, frag_all)
    corr_tau = pearson(tau_all, frag_for_tau) if tau_all else 0.0
    print(f"corr(fragmentation, min_spacing_steps) = {corr_spacing: .3f}")
    print(f"corr(fragmentation, tau_relax)         = {corr_tau: .3f}")

    # Structural fields correlations
    print("\n=== Structural Fields Correlations ===")
    # ΔC(t) = coherence_final - coherence_initial
    delta_c: List[float] = []
    curv_max_final: List[float] = []
    grad_final: List[float] = []
    xi_c_final: List[float] = []
    for r in rows:
        if r.get('curv_phi_max_final') is not None:
            delta_c.append(float(r.get('coherence_final', 0.0)) - float(r.get('coherence_initial', 0.0)))
            curv_max_final.append(abs(float(r['curv_phi_max_final'])))
            if r.get('grad_phi_mean_final') is not None:
                grad_final.append(float(r['grad_phi_mean_final']))
            if r.get('xi_c_final') is not None:
                xi_c_final.append(float(r['xi_c_final']))
    
    if delta_c and curv_max_final:
        corr_dc_curv = pearson(delta_c, curv_max_final)
        print(f"corr(ΔC(t), |K_φ|_max_final)           = {corr_dc_curv: .3f}")
    
    if delta_c and grad_final and len(delta_c) == len(grad_final):
        corr_dc_grad = pearson(delta_c, grad_final)
        print(f"corr(ΔC(t), |∇φ|_mean_final)          = {corr_dc_grad: .3f}")
    
    if delta_c and xi_c_final and len(delta_c) == len(xi_c_final):
        corr_dc_xi = pearson(delta_c, xi_c_final)
        print(f"corr(ΔC(t), ξ_C_final)                = {corr_dc_xi: .3f}")

    # Coherence length by topology
    print("\n=== Coherence Length ξ_C by Topology ===")
    by_topo: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        xi = r.get('xi_c_final')
        if xi is not None:
            by_topo[r.get('topology', 'unknown')].append(float(xi))
    for topo in sorted(by_topo.keys()):
        vals = by_topo[topo]
        m, s = mean_std(vals)
        print(f"{topo:>11}: ξ_C = {m:6.3f} ± {s:5.3f} (N={len(vals)})")

    # Alpha empirical by topology and nu_f
    print("\n=== Alpha Empirical by Topology and νf ===")
    by_topo_nuf: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for r in rows:
        a = r.get('alpha_empirical')
        if a is None:
            continue
        topo = r.get('topology', 'unknown')
        nuf = float(r.get('nu_f', 0.0))
        by_topo_nuf[(topo, nuf)].append(float(a))
    keys = sorted(by_topo_nuf.keys(), key=lambda k: (k[0], k[1]))
    for k in keys:
        vals = by_topo_nuf[k]
        m, s = mean_std(vals)
        print(f"{k[0]:>11} νf={k[1]:4.1f} : α_emp = {m:6.3f} ± {s:5.3f} (N={len(vals)})")
    
    # Curvature variance by sequence type
    print("\n=== Phase Curvature |K_φ| Variance by Sequence Type ===")
    for stype, lst in by_type.items():
        curv_vals = [abs(float(r['curv_phi_max_final'])) for r in lst if r.get('curv_phi_max_final') is not None]
        if curv_vals:
            m, s = mean_std(curv_vals)
            print(f"{stype:>12}: |K_φ|_max = {m:6.3f} ± {s:5.3f} (N={len(curv_vals)})")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    rows = load_jsonl(path)
    if not rows:
        print("No rows loaded.")
        return 1
    analyze(rows)
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
