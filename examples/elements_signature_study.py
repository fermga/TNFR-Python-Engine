"""
TNFR Elements Signature Study (H, C, N, O, Au-like) — Telemetry-only

Computes Structural Field Tetrad summaries for element-like radial graphs
Z in {1 (H), 6 (C), 7 (N), 8 (O), 79 (Au-like)} and an optional Au-network
(composition of multiple Au-like subgraphs connected topologically).

Outputs: examples/output/elements_signature_study.{html,csv,jsonl}
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from tnfr.examples_utils import build_element_radial_graph, apply_synthetic_activation_sequence
from tnfr.physics.fields import (
    compute_structural_potential,
    estimate_coherence_length,
)
from tnfr.operators.grammar import (
    warn_phase_gradient_telemetry,
    warn_phase_curvature_telemetry,
    warn_coherence_length_telemetry,
    validate_structural_potential_confinement,
)
from tnfr.telemetry.constants import (
    PHASE_GRADIENT_THRESHOLD,
    PHASE_CURVATURE_ABS_THRESHOLD,
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
)
from tnfr.examples_utils.html import render_safety_triad_panel

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

Z_LIST = [1, 6, 7, 8, 79]

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 79: "Au"}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _symbol(Z: int) -> str:
    return SYMBOLS.get(Z, str(Z))


def _row_for_graph(G: Any, label: str) -> Dict[str, Any]:
    # Tetrad telemetry
    phi_before = compute_structural_potential(G)
    _, stats_g, msg_g, _ = warn_phase_gradient_telemetry(G, threshold=PHASE_GRADIENT_THRESHOLD)
    _, stats_k, msg_k, _ = warn_phase_curvature_telemetry(G, abs_threshold=PHASE_CURVATURE_ABS_THRESHOLD, multiscale_check=True, alpha_hint=2.76)
    _, stats_x, msg_x = warn_coherence_length_telemetry(G)
    xi_c = float(estimate_coherence_length(G))

    # Sequential ΔΦ_s (telemetry-only)
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)
    ok, drift, msg_u6 = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_POTENTIAL_DELTA_THRESHOLD, strict=False
    )

    return {
        "label": label,
        "xi_c": xi_c,
        "mean_grad": float(stats_g.get("mean_abs", 0.0)),
        "mean_kphi": float(stats_k.get("mean_abs", 0.0)),
        "mean_path_length": float(stats_x.get("mean_path_length", 0.0)),
        "u6_ok": bool(ok),
        "u6_drift": float(drift),
        "telemetry_msgs": [msg_g, msg_k, msg_x, msg_u6],
    }


def build_au_network(n_subgraphs: int = 4, *, seed: int = 123) -> Any:
    """Compose multiple Au-like element graphs and connect their nuclei to form a simple network.

    Telemetry-only: creates a combined graph to emulate metallic connectivity.
    """
    if nx is None:
        raise RuntimeError("networkx required")

    # Build subgraphs
    subs = []
    for i in range(n_subgraphs):
        subs.append(build_element_radial_graph(79, seed=seed + i))

    # Relabel and merge
    G = nx.Graph()
    current_max = -1
    nuclei = []
    for gi, SG in enumerate(subs):
        offset = current_max + 1
        rel = nx.relabel_nodes(SG, {n: n + offset for n in SG.nodes()})
        G.update(rel)
        # Record nucleus id (assume 0 in each SG)
        nuclei.append(offset + 0)
        current_max = max(G.nodes())

    # Connect nuclei in a ring (metal-like network)
    for i in range(len(nuclei)):
        a = nuclei[i]
        b = nuclei[(i + 1) % len(nuclei)]
        G.add_edge(a, b)

    # Tag cluster id for visualization
    for gi, SG in enumerate(subs):
        base = nuclei[gi]  # nucleus index in merged
        # Collect nodes of this cluster range
        # Conservative approach: color by proximity to nucleus in hop distance
        lengths = nx.single_source_shortest_path_length(G, base, cutoff=2)
        for n in lengths.keys():
            G.nodes[n]["cluster"] = gi

    G.graph["network"] = "Au-like"
    return G


def run() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Single-element graphs
    for Z in Z_LIST:
        G = build_element_radial_graph(Z, seed=100 + Z)
        rows.append({"Z": Z, "symbol": _symbol(Z), **_row_for_graph(G, f"{_symbol(Z)} (single)")})

    # Au-network (optional comparative)
    try:
        G_net = build_au_network(n_subgraphs=4, seed=200)
        rows.append({"Z": 79, "symbol": "Au-net", **_row_for_graph(G_net, "Au-network (n=4)")})
    except Exception:
        pass

    return rows


def _write_html(rows: List[Dict[str, Any]], path: str) -> None:
    # Safety Triad summary (dataset-level)
    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0
    mean_g = _mean([float(r.get("mean_grad", 0.0)) for r in rows])
    mean_k = _mean([float(r.get("mean_kphi", 0.0)) for r in rows])
    max_drift = max([abs(float(r.get("u6_drift", 0.0))) for r in rows] or [0.0])

    html = [
        "<html><head><meta charset='utf-8'><title>TNFR Elements Signature Study</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head><body>",
        "<h1>TNFR Elements Signature Study</h1>",
        render_safety_triad_panel(
            thresholds={
                "phi_delta": STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
                "grad": PHASE_GRADIENT_THRESHOLD,
                "kphi": PHASE_CURVATURE_ABS_THRESHOLD,
            },
            summary={
                "mean_grad": mean_g,
                "mean_kphi": mean_k,
                "max_drift": max_drift,
            },
        ),
        "<p>Telemetry-only Structural Field Tetrad and ΔΦ_s sequential check for H, C, N, O, and Au-like. Includes an Au-network composition.</p>",
        "<table><thead><tr><th>Symbol</th><th>Label</th><th>ξ_C</th><th>mean|∇φ|</th><th>mean|K_φ|</th><th>mean path len</th><th>ΔΦ_s</th><th>ΔΦ_s ok</th></tr></thead><tbody>",
    ]
    for r in rows:
        html.append(
            f"<tr><td>{r.get('symbol', '')}</td><td>{r.get('label', '')}</td><td>{float(r.get('xi_c', 0.0)):.2f}</td>"
            f"<td>{float(r.get('mean_grad', 0.0)):.3f}</td><td>{float(r.get('mean_kphi', 0.0)):.3f}</td><td>{float(r.get('mean_path_length', 0.0)):.2f}</td>"
            f"<td>{float(r.get('u6_drift', 0.0)):.3f}</td><td>{'PASS' if bool(r.get('u6_ok', False)) else 'FAIL'}</td></tr>"
        )
    html += ["</tbody></table>", "</body></html>"]
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    cols = ["symbol", "label", "xi_c", "mean_grad", "mean_kphi", "mean_path_length", "u6_drift", "u6_ok"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(
                ",".join([
                    str(r.get("symbol", "")),
                    str(r.get("label", "")),
                    f"{float(r.get('xi_c', 0.0)):.6f}",
                    f"{float(r.get('mean_grad', 0.0)):.6f}",
                    f"{float(r.get('mean_kphi', 0.0)):.6f}",
                    f"{float(r.get('mean_path_length', 0.0)):.6f}",
                    f"{float(r.get('u6_drift', 0.0)):.6f}",
                    "PASS" if bool(r.get("u6_ok", False)) else "FAIL",
                ]) + "\n"
            )


def _write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    from tnfr.utils import json_dumps
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json_dumps(r) + "\n")


def main() -> str:
    rows = run()
    base = os.path.join(os.path.dirname(__file__), "output")
    out_html = os.path.join(base, "elements_signature_study.html")
    out_csv = os.path.join(base, "elements_signature_study.csv")
    out_jsonl = os.path.join(base, "elements_signature_study.jsonl")
    _write_html(rows, out_html)
    _write_csv(rows, out_csv)
    _write_jsonl(rows, out_jsonl)
    return out_html


if __name__ == "__main__":
    path = main()
    print("Wrote elements signature study to:", path)
