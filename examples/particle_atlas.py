"""
Particle-like Loci Atlas (HTML) — Minimal Example

Creates a small network, computes the Structural Field Tetrad
(Φ_s, |∇φ|, K_φ, ξ_C), identifies particle-like loci using
canonical telemetry thresholds, and exports a simple HTML report.

This is a read-only diagnostic example; it does not mutate EPI.

Criteria (telemetry-only, non-prescriptive):
- Local synchrony: |∇φ| < 0.2904
- Curvature safety: |K_φ| < 3.0 (multiscale safety is summarized globally)
- Global regime gate (optional): ξ_C < mean path length (strict locality)

Output:
- examples/output/particle_atlas.html (created if missing)

Note:
- Φ_s drift (ΔΦ_s) is sequence-based. Here we report Φ_s summary stats
  for the snapshot, not drift, to keep the demo single-frame.
"""

from __future__ import annotations

import math
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
from tnfr.examples_utils import apply_synthetic_activation_sequence
from tnfr.telemetry.constants import (
    PHASE_GRADIENT_THRESHOLD,
    PHASE_CURVATURE_ABS_THRESHOLD,
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
)
from tnfr.examples_utils.html import render_safety_triad_panel


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_graph(n: int = 50, k: int = 4, p: float = 0.1, seed: int = 42) -> Any:
    """Create a small-world graph with seeded attributes for reproducibility."""
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    # Assign phases uniformly on [0, 2π) and moderate ΔNFR
    # Seeded RNG (Python's stdlib for reliability across environments)
    import random as _random
    rng = _random.Random(seed)

    def _rand() -> float:
        return float(rng.random())
    for n in G.nodes():
        G.nodes[n]["phase"] = float(2.0 * math.pi * _rand())
        # Moderate structural pressure in [0.2, 0.8)
        G.nodes[n]["delta_nfr"] = float(0.2 + 0.6 * _rand())

    return G


def classify_particle_like(
    G: Any,
    *,
    grad_threshold: float = 0.38,
    kphi_threshold: float = 3.0,
    require_local_regime: bool = True,
) -> Dict[str, Any]:
    """Identify particle-like loci via local thresholds and global regime gate.

    Returns a dict with fields, candidates, telemetry summaries, and HTML-safe rows.
    """
    # Compute fields
    phi_s = compute_structural_potential(G)  # snapshot (no drift)
    grad = compute_phase_gradient(G)
    kphi = compute_phase_curvature(G)
    xi_c = float(estimate_coherence_length(G))

    # Warnings / summaries
    # Use centralized default threshold if not overridden
    if grad_threshold is None:
        grad_threshold = PHASE_GRADIENT_THRESHOLD
    safe_g, stats_g, msg_g, flagged_g = warn_phase_gradient_telemetry(G, threshold=grad_threshold)
    if kphi_threshold is None:
        kphi_threshold = PHASE_CURVATURE_ABS_THRESHOLD
    safe_k, stats_k, msg_k, hotspots_k = warn_phase_curvature_telemetry(G, abs_threshold=kphi_threshold, multiscale_check=True, alpha_hint=2.76)
    safe_x, stats_x, msg_x = warn_coherence_length_telemetry(G)

    # Optional global regime gate (strict locality)
    allow_locality = True
    if require_local_regime:
        mpl = float(stats_x.get("mean_path_length", 0.0))
        allow_locality = (xi_c < mpl) if mpl > 0 else True

    # Candidate nodes: satisfy both local thresholds
    candidates = [
        n
        for n in G.nodes()
        if abs(float(grad.get(n, 0.0))) < grad_threshold
        and abs(float(kphi.get(n, 0.0))) < kphi_threshold
    ]
    if not allow_locality:
        # If not in local regime, mark candidates but note regime warning
        pass

    # Summaries for Φ_s (snapshot stats)
    phi_vals = list(phi_s.values())
    phi_summary = {
        "mean": float(sum(phi_vals) / max(len(phi_vals), 1)) if phi_vals else 0.0,
        "min": float(min(phi_vals)) if phi_vals else 0.0,
        "max": float(max(phi_vals)) if phi_vals else 0.0,
    }

    # Build HTML rows
    rows = []
    for n in G.nodes():
        rows.append(
            (
                n,
                float(grad.get(n, 0.0)),
                float(kphi.get(n, 0.0)),
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


# Synthetic sequence moved to tnfr.examples_utils.apply_synthetic_activation_sequence


def render_html(result: Dict[str, Any], *, out_path: str) -> None:
    """Write a simple static HTML report from the analysis result."""
    phi = result["phi_summary"]
    xi_c = result["xi_c"]
    msgs = result["telemetry_msgs"]
    seq_msg = result.get("u6_sequence_msg")
    seq_drift = result.get("u6_sequence_drift")
    rows = result["rows"]
    regime_local = result["regime_local"]

    html = [
        "<html>",
        "<head><meta charset='utf-8'><title>TNFR Particle Atlas</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head>",
        "<body>",
        "<h1>TNFR Structural Quanta — Minimal Atlas</h1>",
        render_safety_triad_panel(
            thresholds={
                "phi_delta": STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
                "grad": PHASE_GRADIENT_THRESHOLD,
                "kphi": PHASE_CURVATURE_ABS_THRESHOLD,
            }
        ),
        "<p>Read-only telemetry demo. Thresholds: |∇φ|<0.2904, |K_φ|<3.0; locality gate uses ξ_C vs mean path length.</p>",
        "<h2>Global Summaries</h2>",
        f"<p>Φ_s snapshot — mean: {phi['mean']:.3f}, min: {phi['min']:.3f}, max: {phi['max']:.3f}</p>",
        f"<p>ξ_C (coherence length): {xi_c:.2f} — regime: {'local' if regime_local else 'non-local'}</p>",
        "<h3>Telemetry Messages</h3>",
        "<ul>" + "".join(f"<li>{m}</li>" for m in msgs) + "</ul>",
        "<h2>Loci Table</h2>",
        "<table><thead><tr><th>Node</th><th>|∇φ|</th><th>|K_φ|</th><th>Particle-like</th></tr></thead><tbody>",
    ]

    for n, g, k, tag in rows:
        html.append(
            f"<tr><td>{n}</td><td>{abs(g):.3f}</td><td>{abs(k):.3f}</td><td>{tag}</td></tr>"
        )

    # Optional U6 sequence validation section
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
    G = build_graph()
    # Compute Φ_s before
    phi_before = compute_structural_potential(G)

    # Classify snapshot
    result = classify_particle_like(G)

    # Apply synthetic minimal sequence and validate ΔΦ_s confinement
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)
    ok, drift, msg = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_POTENTIAL_DELTA_THRESHOLD, strict=False
    )
    result["u6_sequence_msg"] = msg
    result["u6_sequence_drift"] = drift
    out = os.path.join(os.path.dirname(__file__), "output", "particle_atlas.html")
    render_html(result, out_path=out)
    return out


if __name__ == "__main__":
    path = main()
    print(f"Wrote particle atlas to: {path}")
