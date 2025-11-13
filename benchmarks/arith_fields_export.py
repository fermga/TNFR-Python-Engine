"""
Arithmetic TNFR structural fields export and telemetry (Φ_s, |∇φ|, K_φ, ξ_C).

Generates:
- Histogram of Φ_s
- Heatmap-like line of Φ_s vs n (colored by primality)
- C(r) curves with exponential fit and ξ_C for both distance modes
- K_φ safety metrics: fraction |K_φ|>=3.0 and multiscale decay fit
- JSONL telemetry with per-node and global metrics

Usage (optional): adjust MAX_N and OUTPUT_DIR below or pass via env.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

from tnfr.mathematics import ArithmeticTNFRNetwork


MAX_N = int(os.environ.get("TNFR_ARITH_MAX_N", "5000"))
OUTPUT_DIR = os.environ.get("TNFR_ARITH_OUT", "benchmarks/results")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_plots(net: ArithmeticTNFRNetwork, distance_mode: str, out_prefix: str) -> Dict[str, Any]:
    # Compute fields (includes kphi safety metrics)
    net.compute_structural_fields(phase_method="spectral")
    phi_s = net.compute_structural_potential(distance_mode=distance_mode)
    nodes = sorted(net.graph.nodes())
    # For large-N arithmetic, skip heavy ξ_C to keep runs tractable
    if distance_mode == 'arithmetic' and len(nodes) > 2000:
        xi = {"skipped": True}
    else:
        xi = net.estimate_coherence_length(distance_mode=distance_mode)

    phi_s_arr = np.array([phi_s[n] for n in nodes], dtype=float)
    is_prime = np.array([net.graph.nodes[n]['is_prime'] for n in nodes], dtype=bool)

    # 1) Histogram Φ_s
    plt.figure(figsize=(8, 4))
    plt.hist(phi_s_arr, bins=60, alpha=0.8, color="#4e79a7")
    plt.title(f"Φ_s histogram (distance={distance_mode}, n≤{net.max_number})")
    plt.xlabel("Φ_s")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_phi_s_hist.png")
    plt.close()

    # 2) Heatmap-like Φ_s vs n (line colored by primality)
    plt.figure(figsize=(12, 3))
    cmap_vals = np.where(is_prime, 1.0, 0.0)
    plt.scatter(nodes, phi_s_arr, c=cmap_vals, cmap="coolwarm", s=6, marker='o', linewidths=0)
    plt.title(f"Φ_s vs n (distance={distance_mode}, red=prime, blue=composite)")
    plt.xlabel("n")
    plt.ylabel("Φ_s")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_phi_s_vs_n.png")
    plt.close()

    # 3) C(r) curves and fit
    if isinstance(xi, dict) and xi.get('r') and xi.get('C_r'):
        r = np.array(xi['r'])
        C_r = np.array(xi['C_r'])
        plt.figure(figsize=(6, 4))
        plt.plot(r, C_r, 'o', label='C(r)')
        if xi.get('fit_slope') is not None and xi.get('fit_intercept') is not None:
            slope = xi['fit_slope']
            intercept = xi['fit_intercept']
            r_fit = np.linspace(r.min(), r.max(), 200)
            C_fit = np.exp(intercept + slope * r_fit)
            xi_val = xi.get('xi_c')
            r2_val = xi.get('R2')
            if xi_val is not None and r2_val is not None:
                label = f"fit ξ_C={xi_val:.2f}, R²={r2_val:.2f}"
            else:
                label = "fit"
            plt.plot(r_fit, C_fit, '-', label=label)
        plt.title(f"C(r) and fit (distance={distance_mode})")
        plt.xlabel("r")
        plt.ylabel("C(r)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_C_r_fit.png")
        plt.close()

    # 4) K_phi safety metrics and multiscale via class methods
    kphi_safety = net.compute_kphi_safety(threshold=3.0)
    kphi_ms = net.k_phi_multiscale_safety(distance_mode=distance_mode)
    r_list = kphi_ms.get('r_list', []) or []
    var_list = kphi_ms.get('var_list', []) or []
    alpha_est = kphi_ms.get('alpha_fit')
    R2_est = kphi_ms.get('R2_fit')
    slope = kphi_ms.get('slope_loglog')
    intercept = kphi_ms.get('intercept_loglog')

    # Plot multiscale decay if available
    if r_list and var_list:
        plt.figure(figsize=(6, 4))
        plt.loglog(r_list, var_list, 'o', label='var(K_φ neighborhood-mean)')
        if alpha_est is not None and slope is not None and intercept is not None:
            fit_line = np.exp(intercept) * (np.array(r_list, dtype=float) ** float(slope))
            plt.loglog(r_list, fit_line, '-', label=f"fit α≈{alpha_est:.2f}, R²={R2_est:.2f}")
        title_suffix = "arithmetic windows" if distance_mode == 'arithmetic' else "topological balls"
        plt.title(f"Multiscale decay of K_φ variance ({title_suffix})")
        plt.xlabel("r")
        plt.ylabel("var(K_φ mean)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_kphi_multiscale.png")
        plt.close()

    return {
        'kphi_frac_abs_ge_3': kphi_safety.get('frac_abs_ge_threshold'),
        'kphi_multiscale_alpha': alpha_est,
        'kphi_multiscale_R2': R2_est,
        'xi_c': xi,
    }


def export_jsonl(net: ArithmeticTNFRNetwork, meta: Dict[str, Any], out_path: str, globals_records: List[Dict[str, Any]] | None = None) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        # Header meta
        f.write(json.dumps({"type": "meta", **meta}) + "\n")
        # Global summary lines (optional)
        if globals_records:
            for g in globals_records:
                f.write(json.dumps({"type": "global", **g}) + "\n")
        # Per-node telemetry
        for n in sorted(net.graph.nodes()):
            nd = net.graph.nodes[n]
            rec = {
                "type": "node",
                "n": n,
                "is_prime": bool(nd.get('is_prime', False)),
                "EPI": float(nd.get('EPI', float('nan'))),
                "nu_f": float(nd.get('nu_f', float('nan'))),
                "DELTA_NFR": float(nd.get('DELTA_NFR', float('nan'))),
                "phi_s": float(nd.get('phi_s', float('nan'))),
                "phi_grad": float(nd.get('phi_grad', float('nan'))),
                "k_phi": float(nd.get('k_phi', float('nan'))),
                "coherence_local": float(nd.get('coherence_local', float('nan'))),
            }
            f.write(json.dumps(rec) + "\n")


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"Building arithmetic TNFR network up to n<={MAX_N}...")
    net = ArithmeticTNFRNetwork(MAX_N)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Export for arithmetic distance
    print("Computing fields (arithmetic distance)...")
    res_arith = export_plots(net, distance_mode="arithmetic", out_prefix=os.path.join(OUTPUT_DIR, f"arith_{MAX_N}_arith"))

    # Export for topological distance (may be slow for very large n)
    if MAX_N <= 2000:
        print("Computing fields (topological distance)...")
        res_topo = export_plots(net, distance_mode="topological", out_prefix=os.path.join(OUTPUT_DIR, f"arith_{MAX_N}_topo"))
    else:
        res_topo = {"skipped": True}

    # JSONL telemetry
    meta = {
        "timestamp": timestamp,
        "max_n": MAX_N,
        "distance_modes": ["arithmetic", "topological" if MAX_N <= 2000 else "topological_skipped"],
        "kphi_threshold": 3.0,
    }
    globals_records: List[Dict[str, Any]] = []
    globals_records.append({"distance_mode": "arithmetic", **{k: v for k, v in res_arith.items() if k != 'xi_c' or isinstance(v, dict)}})
    if not res_topo.get('skipped'):
        globals_records.append({"distance_mode": "topological", **{k: v for k, v in res_topo.items() if k != 'xi_c' or isinstance(v, dict)}})
    export_jsonl(net, meta, os.path.join(OUTPUT_DIR, f"arith_{MAX_N}_telemetry.jsonl"), globals_records=globals_records)

    # Print summary
    print("Summary:")
    print({
        "arith": res_arith,
        "topo": res_topo,
    })


if __name__ == "__main__":
    main()
