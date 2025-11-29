"""
TNFR Molecule Atlas (Early Diatomic Demo)

Builds simple diatomic molecules by composing element-like radial graphs and adding
coupling edges (telemetry-only). Computes the Structural Field Tetrad and a U6 ΔΦ_s
sequential check per molecule and writes HTML/CSV/JSONL summaries.

Output: examples/output/molecule_atlas.html
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from tnfr.examples_utils import (
    build_diatomic_molecule_graph,
    apply_synthetic_activation_sequence,
)
from tnfr.utils import json_dumps
from tnfr.examples_utils.html import render_safety_triad_panel
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
from tnfr.telemetry.constants import (
    PHASE_GRADIENT_THRESHOLD,
    PHASE_CURVATURE_ABS_THRESHOLD,
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _symbol(Z: int) -> str:
    base = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    }
    return base.get(Z, str(Z))


PAIRS: List[Tuple[int, int]] = [
    (1, 1),   # H2
    (9, 9),   # F2
    (3, 9),   # LiF
]


def _derive_signature(u6_ok: bool, mean_grad: float, mean_kphi: float, xi_c_val: float, mean_dist: float, loc_frac: float) -> str:
    if not u6_ok:
        return "runaway"
    if mean_dist > 0.0 and xi_c_val > 3.1416 * mean_dist:
        return "critical"
    if (
        mean_grad < 0.8 * PHASE_GRADIENT_THRESHOLD
        and mean_kphi < 0.8 * PHASE_CURVATURE_ABS_THRESHOLD
        and (mean_dist == 0.0 or xi_c_val < mean_dist)
        and loc_frac >= 0.8
    ):
        return "localized-confined"
    if (
        mean_grad >= PHASE_GRADIENT_THRESHOLD
        or mean_kphi >= PHASE_CURVATURE_ABS_THRESHOLD
        or loc_frac < 0.6
    ):
        return "stressed"
    return "confined"


def analyze_molecule(Z1: int, Z2: int, *, seed: int = 123) -> Dict[str, Any]:
    G = build_diatomic_molecule_graph(Z1, Z2, seed=seed, bond_links=1)

    phi_before = compute_structural_potential(G)
    grad = compute_phase_gradient(G)
    kphi = compute_phase_curvature(G)
    xi_c = float(estimate_coherence_length(G))

    _, stats_g, msg_g, _ = warn_phase_gradient_telemetry(G, threshold=PHASE_GRADIENT_THRESHOLD)
    _, stats_k, msg_k, _ = warn_phase_curvature_telemetry(
        G, abs_threshold=PHASE_CURVATURE_ABS_THRESHOLD, multiscale_check=True, alpha_hint=2.76
    )
    _, stats_x, msg_x = warn_coherence_length_telemetry(G)

    # U6 sequential ΔΦ_s check
    apply_synthetic_activation_sequence(G, alpha=0.25, dnfr_factor=0.9)
    phi_after = compute_structural_potential(G)
    ok, drift, msg_u6 = validate_structural_potential_confinement(
        G, phi_before, phi_after, threshold=STRUCTURAL_POTENTIAL_DELTA_THRESHOLD, strict=False
    )

    # Local criteria fraction
    total = max(1, len(G.nodes()))
    local_count = sum(
        1 for n in G.nodes()
        if abs(float(grad.get(n, 0.0))) < PHASE_GRADIENT_THRESHOLD
        and abs(float(kphi.get(n, 0.0))) < PHASE_CURVATURE_ABS_THRESHOLD
    )

    row = {
        "Z1": Z1,
        "Z2": Z2,
        "formula": f"{_symbol(Z1)}{_symbol(Z2)}",
        "xi_c": xi_c,
        "mean_grad": float(stats_g.get("mean_abs", 0.0)),
        "mean_kphi": float(stats_k.get("mean_abs", 0.0)),
        "mean_path_length": float(stats_x.get("mean_path_length", 0.0)),
        "u6_ok": bool(ok),
        "u6_drift": float(drift),
        "u6_msg": msg_u6,
        "local_frac": float(local_count) / float(total),
        "telemetry_msgs": [msg_g, msg_k, msg_x],
    }
    row["signature"] = _derive_signature(
        row["u6_ok"], row["mean_grad"], row["mean_kphi"], row["xi_c"], row["mean_path_length"], row["local_frac"]
    )
    return row


def render_html(rows: List[Dict[str, Any]], *, out_path: str) -> None:
    # Safety Triad summary (dataset-level)
    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0
    mean_g = _mean([float(r.get("mean_grad", 0.0)) for r in rows])
    mean_k = _mean([float(r.get("mean_kphi", 0.0)) for r in rows])
    max_drift = max([abs(float(r.get("u6_drift", 0.0))) for r in rows] or [0.0])

    html = [
        "<html>",
        "<head><meta charset='utf-8'><title>TNFR Molecule Atlas — Diatomic Demo</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head>",
        "<body>",
        "<h1>TNFR Molecule Atlas — Diatomic Demo</h1>",
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
        "<p>We compose element-like networks and add coupling edges to emulate diatomic bonding (UM/RA at topology level). Read-only telemetry summarizes stability and ΔΦ_s checks.</p>",
        "<table><thead><tr><th>Formula</th><th>Signature</th><th>ξ_C</th><th>mean|∇φ|</th><th>mean|K_φ|</th><th>mean path len</th><th>local frac</th><th>ΔΦ_s ok</th><th>ΔΦ_s</th></tr></thead><tbody>",
    ]
    for r in rows:
        html.append(
            f"<tr><td>{r['formula']}</td><td>{r.get('signature', '')}</td><td>{r['xi_c']:.2f}</td>"
            f"<td>{r['mean_grad']:.3f}</td><td>{r['mean_kphi']:.3f}</td><td>{r['mean_path_length']:.2f}</td>"
            f"<td>{r['local_frac']:.2f}</td><td>{'PASS' if r['u6_ok'] else 'FAIL'}</td><td>{r['u6_drift']:.3f}</td></tr>"
        )
    html += [
        "</tbody></table>",
        "</body></html>",
    ]
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def _write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json_dumps(r) + "\n")


def _write_csv(rows: list[dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    cols = [
        "formula",
        "signature",
        "xi_c",
        "mean_grad",
        "mean_kphi",
        "mean_path_length",
        "local_frac",
        "u6_ok",
        "u6_drift",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(
                ",".join(
                    [
                        str(r.get("formula", "")),
                        str(r.get("signature", "")),
                        f"{float(r.get('xi_c', 0.0)):.6f}",
                        f"{float(r.get('mean_grad', 0.0)):.6f}",
                        f"{float(r.get('mean_kphi', 0.0)):.6f}",
                        f"{float(r.get('mean_path_length', 0.0)):.6f}",
                        f"{float(r.get('local_frac', 0.0)):.6f}",
                        "PASS" if bool(r.get("u6_ok", False)) else "FAIL",
                        f"{float(r.get('u6_drift', 0.0)):.6f}",
                    ]
                )
                + "\n"
            )


def main() -> str:
    rows = [analyze_molecule(a, b) for (a, b) in PAIRS]
    base = os.path.join(os.path.dirname(__file__), "output")
    out_html = os.path.join(base, "molecule_atlas.html")
    out_jsonl = os.path.join(base, "molecule_atlas.jsonl")
    out_csv = os.path.join(base, "molecule_atlas.csv")
    render_html(rows, out_path=out_html)
    _write_jsonl(rows, out_jsonl)
    _write_csv(rows, out_csv)
    return out_html


if __name__ == "__main__":
    path = main()
    print(f"Wrote molecule atlas to: {path}")
