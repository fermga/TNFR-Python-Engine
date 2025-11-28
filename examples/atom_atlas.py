"""
Atom-like Pattern Atlas (HTML) — Hydrogen-like Minimal Demo

Constructs a simple radial network (nucleus + ring shell), computes the
Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C), highlights loci satisfying
particle-like local criteria, and runs a short synthetic [AL, RA, IL]-like
step to validate U6 (ΔΦ_s) sequential confinement.

Output: examples/output/atom_atlas.html

Notes:
- Telemetry is read-only; no EPI mutations occur in this demo.
- Thresholds/messages are sourced from tnfr.telemetry.constants and
  validators in tnfr.operators.grammar.
"""
from __future__ import annotations

import os
from typing import Any, Dict

try:
    import networkx as nx  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError("networkx is required for this example") from e

from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)
from tnfr.operators.grammar import (
    warn_phase_gradient_telemetry,
    warn_phase_curvature_telemetry,
    warn_coherence_length_telemetry,
    validate_structural_potential_confinement,
)
from tnfr.examples_utils import (
    apply_synthetic_activation_sequence,
    build_radial_atom_graph,
)
from tnfr.telemetry.constants import (
    PHASE_GRADIENT_THRESHOLD,
    PHASE_CURVATURE_ABS_THRESHOLD,
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
)
from tnfr.examples_utils.html import render_safety_triad_panel


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def classify_loci(
    G: Any,
    *,
    grad_threshold: float | None = None,
    kphi_threshold: float | None = None,
    require_local_regime: bool = True,
) -> Dict[str, Any]:
    """Compute tetrad, warnings, and mark loci that satisfy local criteria."""
    grad_threshold = PHASE_GRADIENT_THRESHOLD if grad_threshold is None else grad_threshold
    kphi_threshold = PHASE_CURVATURE_ABS_THRESHOLD if kphi_threshold is None else kphi_threshold

    phi_s = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    kphi = compute_phase_curvature(G)
    xi_c = float(estimate_coherence_length(G))

    safe_g, stats_g, msg_g, _ = warn_phase_gradient_telemetry(G, threshold=grad_threshold)
    safe_k, stats_k, msg_k, _ = warn_phase_curvature_telemetry(
        G, abs_threshold=kphi_threshold, multiscale_check=True, alpha_hint=2.76
    )
    safe_x, stats_x, msg_x = warn_coherence_length_telemetry(G)

    # Optional regime gate based on locality (ξ_C vs mean path length)
    allow_locality = True
    if require_local_regime:
        mpl = float(stats_x.get("mean_path_length", 0.0))
        allow_locality = (xi_c < mpl) if mpl > 0 else True

    candidates = [
        n
        for n in G.nodes()
        if abs(float(grad.get(n, 0.0))) < grad_threshold
        and abs(float(kphi.get(n, 0.0))) < kphi_threshold
    ]

    phi_vals = list(phi_s.values())
    phi_summary = {
        "mean": float(sum(phi_vals) / max(len(phi_vals), 1)) if phi_vals else 0.0,
        "min": float(min(phi_vals)) if phi_vals else 0.0,
        "max": float(max(phi_vals)) if phi_vals else 0.0,
    }

    rows = []
    for n in G.nodes():
        rows.append(
            (
                n,
                float(grad.get(n, 0.0)),
                float(kphi.get(n, 0.0)),
                G.nodes[n].get("role", ""),
                "yes" if n in candidates else "",
            )
        )

    return {
        "phi_summary": phi_summary,
        "xi_c": xi_c,
        "telemetry_msgs": [msg_g, msg_k, msg_x],
        "candidates": candidates,
        "rows": rows,
        "regime_local": allow_locality,
    }


def render_html(result: Dict[str, Any], *, out_path: str) -> None:
    phi = result["phi_summary"]
    xi_c = result["xi_c"]
    msgs = result["telemetry_msgs"]
    seq_msg = result.get("u6_sequence_msg")
    seq_drift = result.get("u6_sequence_drift")
    rows = result["rows"]
    regime_local = result["regime_local"]

    html = [
        "<html>",
        "<head><meta charset='utf-8'><title>TNFR Atom Atlas</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head>",
        "<body>",
        "<h1>TNFR Hydrogen-like Atom — Minimal Atlas</h1>",
        render_safety_triad_panel(
            thresholds={
                "phi_delta": STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
                "grad": PHASE_GRADIENT_THRESHOLD,
                "kphi": PHASE_CURVATURE_ABS_THRESHOLD,
            }
        ),
        "<p>Read-only telemetry demo on a radial topology (nucleus + shell). Thresholds: |∇φ|<0.38, |K_φ|<3.0; locality gate uses ξ_C vs mean path length.</p>",
        "<h2>Global Summaries</h2>",
        f"<p>Φ_s snapshot — mean: {phi['mean']:.3f}, min: {phi['min']:.3f}, max: {phi['max']:.3f}</p>",
        f"<p>ξ_C (coherence length): {xi_c:.2f} — regime: {'local' if regime_local else 'non-local'}</p>",
        "<h3>Telemetry Messages</h3>",
        "<ul>" + "".join(f"<li>{m}</li>" for m in msgs) + "</ul>",
        "<h2>Loci Table</h2>",
        "<table><thead><tr><th>Node</th><th>|∇φ|</th><th>|K_φ|</th><th>Role</th><th>Atom-like (local)</th></tr></thead><tbody>",
    ]

    for n, g, k, role, tag in rows:
        html.append(
            f"<tr><td>{n}</td><td>{abs(g):.3f}</td><td>{abs(k):.3f}</td><td>{role}</td><td>{tag}</td></tr>"
        )

    if seq_msg is not None and seq_drift is not None:
        html += [
            "</tbody></table>",
            "<h2>U6 sequence check (synthetic [AL, RA, IL])</h2>",
            f"<p>{seq_msg} (ΔΦ_s = {seq_drift:.3f})</p>",
        ]

    html += [
        "<p style='margin-top:16px;color:#666'>This demo computes ΔΦ_s using a lightweight synthetic sequence to emulate [AL, RA, IL] without mutating EPI.</p>",
        "</body></html>",
    ]

    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def main() -> str:
    G = build_radial_atom_graph(n_shell=24, seed=42)

    # Φ_s before sequence
    phi_before = compute_structural_potential(G)

    # Snapshot classification
    result = classify_loci(G)

    # Synthetic sequence and ΔΦ_s validation
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)
    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_POTENTIAL_DELTA_THRESHOLD, strict=False
    )
    result["u6_sequence_msg"] = msg
    result["u6_sequence_drift"] = drift

    out = os.path.join(os.path.dirname(__file__), "output", "atom_atlas.html")
    render_html(result, out_path=out)
    return out


if __name__ == "__main__":
    path = main()
    print(f"Wrote atom atlas to: {path}")
