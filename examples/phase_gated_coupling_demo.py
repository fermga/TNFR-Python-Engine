"""
TNFR Phase-Gated Coupling Demo (U3) — Telemetry-only

This demo shows explicit phase verification (U3) gating for coupling between two
atom-like graphs. We construct two scenarios:
  1) In-phase terminals (compatible) → edge added
  2) Anti-phase terminals (incompatible) → edge blocked

We report telemetry (|∇φ|, K_φ, ξ_C) and U6 ΔΦ_s drift before/after a synthetic step.
Outputs are written to examples/output/phase_gated_coupling_demo.{html,jsonl}.
"""
from __future__ import annotations

import os
import math
from typing import Any, Dict, Tuple

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
from tnfr.metrics.phase_compatibility import is_phase_compatible, compute_phase_coupling_strength

import networkx as nx  # type: ignore

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_terminal(G: nx.Graph) -> int:
    # Prefer a shell1 node; fallback to any non-nucleus
    cands = [n for n, d in G.nodes(data=True) if d.get("role") == "shell1"]
    if not cands:
        cands = [n for n, d in G.nodes(data=True) if d.get("role") != "nucleus"]
    return sorted(cands)[0]


def _telemetry_row(G: nx.Graph, label: str) -> Dict[str, Any]:
    phi_before = compute_structural_potential(G)
    xi_c = float(estimate_coherence_length(G))

    _, stats_g, msg_g, _ = warn_phase_gradient_telemetry(G, threshold=PHASE_GRADIENT_THRESHOLD)
    _, stats_k, msg_k, _ = warn_phase_curvature_telemetry(G, abs_threshold=PHASE_CURVATURE_ABS_THRESHOLD, multiscale_check=True, alpha_hint=2.76)
    _, stats_x, msg_x = warn_coherence_length_telemetry(G)

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


def run_demo(threshold: float = 0.9) -> Tuple[str, str]:
    """Run a simple two-scenario U3 gating demo and export HTML + JSONL.

    Scenarios:
      1) In-phase terminals (compatible) → edge added
      2) Anti-phase terminals (incompatible) → edge blocked
    """
    # Build two element-like graphs (e.g., H and O for variety)
    GA = build_element_radial_graph(1, seed=123)  # H-like
    GB = build_element_radial_graph(8, seed=124)  # O-like

    # Relabel and merge into a single graph for convenience
    offB = max(GA.nodes()) + 1
    GB_rel = nx.relabel_nodes(GB, {n: n + offB for n in GB.nodes()})
    G = nx.Graph()
    G.update(GA)
    G.update(GB_rel)

    # Tag atoms for visualization
    for n in GA.nodes():
        G.nodes[n]["atom"] = "A"
    for n in GB_rel.nodes():
        G.nodes[n]["atom"] = "B"

    a = _pick_terminal(GA)
    b = _pick_terminal(GB_rel)

    rows: list[Dict[str, Any]] = []

    # Scenario 1: In-phase → compatible → edge added
    G.nodes[a]["phase"] = 0.0
    G.nodes[b]["phase"] = 0.0
    cs = compute_phase_coupling_strength(G.nodes[a]["phase"], G.nodes[b]["phase"])
    compat = is_phase_compatible(G.nodes[a]["phase"], G.nodes[b]["phase"], threshold=threshold)
    if compat:
        G.add_edge(a, b, gated_by="U3", coupling_strength=cs)
    r1 = _telemetry_row(G, label=f"in-phase (cs={cs:.2f}, compat={compat})")
    r1.update({"phase_compatible": compat, "coupling_strength": cs, "edge_added": bool(G.has_edge(a, b))})
    rows.append(r1)
    if G.has_edge(a, b):
        G.remove_edge(a, b)

    # Scenario 2: Anti-phase → incompatible → edge blocked
    G.nodes[a]["phase"] = 0.0
    G.nodes[b]["phase"] = math.pi
    cs = compute_phase_coupling_strength(G.nodes[a]["phase"], G.nodes[b]["phase"])
    compat = is_phase_compatible(G.nodes[a]["phase"], G.nodes[b]["phase"], threshold=threshold)
    if compat:
        G.add_edge(a, b, gated_by="U3", coupling_strength=cs)
    r2 = _telemetry_row(G, label=f"anti-phase (cs={cs:.2f}, compat={compat})")
    r2.update({"phase_compatible": compat, "coupling_strength": cs, "edge_added": bool(G.has_edge(a, b))})
    rows.append(r2)
    if G.has_edge(a, b):
        G.remove_edge(a, b)

    # Write outputs
    _ensure_dir(OUTPUT_DIR)
    html_path = os.path.join(OUTPUT_DIR, "phase_gated_coupling_demo.html")
    jsonl_path = os.path.join(OUTPUT_DIR, "phase_gated_coupling_demo.jsonl")

    # Minimal HTML summary
    html = [
        "<html><head><meta charset='utf-8'><title>TNFR Phase-Gated Coupling Demo (U3)</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head><body>",
        "<h1>TNFR Phase-Gated Coupling Demo (U3)</h1>",
        render_safety_triad_panel(
            thresholds={
                "phi_delta": STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
                "grad": PHASE_GRADIENT_THRESHOLD,
                "kphi": PHASE_CURVATURE_ABS_THRESHOLD,
            }
        ),
        f"<p>Threshold: {threshold:.2f}. Edge is added only if is_phase_compatible ≥ threshold.</p>",
        "<table><thead><tr><th>Scenario</th><th>Compat</th><th>Coupling</th><th>Edge added</th><th>ξ_C</th><th>mean|∇φ|</th><th>mean|K_φ|</th><th>ΔΦ_s</th><th>ΔΦ_s ok</th></tr></thead><tbody>",
    ]
    for r in rows:
        html.append(
            f"<tr><td>{r['label']}</td><td>{r['phase_compatible']}</td><td>{r['coupling_strength']:.2f}</td><td>{r['edge_added']}</td>"
            f"<td>{r['xi_c']:.2f}</td><td>{r['mean_grad']:.3f}</td><td>{r['mean_kphi']:.3f}</td><td>{r['u6_drift']:.3f}</td><td>{'PASS' if r['u6_ok'] else 'FAIL'}</td></tr>"
        )
    html += ["</tbody></table>", "</body></html>"]
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    # JSONL
    from tnfr.utils import json_dumps
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json_dumps(r) + "\n")

    return html_path, jsonl_path


if __name__ == "__main__":
    h, j = run_demo()
    print("Wrote:", h)
    print("Wrote:", j)
