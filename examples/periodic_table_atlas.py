"""
TNFR Periodic Table (Early Demo) — HTML Summary for Z=1..10

Builds simple element-like radial graphs using examples_utils.build_element_radial_graph,
computes the Structural Field Tetrad and a U6 ΔΦ_s sequential check per element,
and writes an HTML summary table with key telemetry and stability signals.

Output: examples/output/periodic_table_atlas.html

This is a telemetry/demo pipeline (read-only). No EPI mutations.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from tnfr.examples_utils import build_element_radial_graph, apply_synthetic_activation_sequence
from tnfr.cache import build_cache_manager
from tnfr.utils import json_dumps
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
from tnfr.examples_utils.html import render_safety_triad_panel


SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


_PT_CACHE = build_cache_manager()
_PT_CACHE.register("periodic_table_rows", lambda: {})


def analyze_element(Z: int, *, seed: int = 42) -> Dict[str, Any]:
    cache = _PT_CACHE.get("periodic_table_rows")
    cache_key = f"Z:{Z}:seed:{seed}"
    if cache_key in cache:
        return cache[cache_key]
    G = build_element_radial_graph(Z, seed=seed)

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
        1
        for n in G.nodes()
        if abs(float(grad.get(n, 0.0))) < PHASE_GRADIENT_THRESHOLD
        and abs(float(kphi.get(n, 0.0))) < PHASE_CURVATURE_ABS_THRESHOLD
    )

    def _derive_signature(u6_ok: bool, mean_grad: float, mean_kphi: float, xi_c_val: float, mean_dist: float, loc_frac: float) -> str:
        """Descriptive structural signature based on Tetrad telemetry.
        Categories are read-only descriptors, not prescriptive grammar labels.
        """
        if not u6_ok:
            return "runaway"
        # Critical if coherence length far exceeds typical distances
        if mean_dist > 0.0 and xi_c_val > 3.0 * mean_dist:
            return "critical"
        # Safe pocket: low gradients/curvature, localized correlations, high locality
        if (
            mean_grad < 0.8 * PHASE_GRADIENT_THRESHOLD
            and mean_kphi < 0.8 * PHASE_CURVATURE_ABS_THRESHOLD
            and (mean_dist == 0.0 or xi_c_val < mean_dist)
            and loc_frac >= 0.8
        ):
            return "localized-confined"
        # Stressed if any major threshold is challenged or locality is low
        if (
            mean_grad >= PHASE_GRADIENT_THRESHOLD
            or mean_kphi >= PHASE_CURVATURE_ABS_THRESHOLD
            or loc_frac < 0.6
        ):
            return "stressed"
        # Default confined regime
        return "confined"

    row = {
        "Z": Z,
        "symbol": SYMBOLS.get(Z, str(Z)),
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
    # Add descriptive signature/family derived from telemetry
    row_signature = _derive_signature(
        row["u6_ok"], row["mean_grad"], row["mean_kphi"], row["xi_c"], row["mean_path_length"], row["local_frac"]
    )
    row["signature"] = row_signature
    row["family"] = row_signature  # alias for grouping in reports
    cache[cache_key] = row
    _PT_CACHE.store("periodic_table_rows", cache)
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
        "<head><meta charset='utf-8'><title>TNFR Periodic Table — Early Demo</title>",
        "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px;} th{background:#fafafa} .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0} .panel b{display:block;margin-bottom:4px}</style>",
        "</head>",
        "<body>",
        "<h1>TNFR Periodic Table — Early Demo (Z=1..10)</h1>",
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
        "<p>For each Z, we build a radial element-like network and compute field telemetry and a U6 sequential ΔΦ_s check using a synthetic [AL, RA, IL]-like step. Read-only, telemetry-based.</p>",
        "<table><thead><tr><th>Z</th><th>Symbol</th><th>Signature</th><th>ξ_C</th><th>mean|∇φ|</th><th>mean|K_φ|</th><th>mean path len</th><th>local frac</th><th>ΔΦ_s ok</th><th>ΔΦ_s</th></tr></thead><tbody>",
    ]

    for r in rows:
        html.append(
            f"<tr><td>{r['Z']}</td><td>{r['symbol']}</td><td>{r.get('signature', '')}</td><td>{r['xi_c']:.2f}</td>"
            f"<td>{r['mean_grad']:.3f}</td><td>{r['mean_kphi']:.3f}</td><td>{r['mean_path_length']:.2f}</td>"
            f"<td>{r['local_frac']:.2f}</td><td>{'PASS' if r['u6_ok'] else 'FAIL'}</td><td>{r['u6_drift']:.3f}</td></tr>"
        )

    html += [
        "</tbody></table>",
    ]

    # Summary by signature (telemetry-only, descriptive)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sig = str(r.get("signature", ""))
        groups.setdefault(sig, []).append(r)

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    html += [
        "<h2 style='margin-top:20px'>Summary by Signature</h2>",
        "<table><thead><tr><th>Signature</th><th>Count</th><th>PASS rate</th><th>mean ξ_C</th><th>mean|∇φ|</th><th>mean|K_φ|</th><th>mean local frac</th></tr></thead><tbody>",
    ]
    for sig, items in groups.items():
        cnt = len(items)
        pass_rate = _mean([1.0 if it.get("u6_ok", False) else 0.0 for it in items])
        mean_xi = _mean([float(it.get("xi_c", 0.0)) for it in items])
        mean_g = _mean([float(it.get("mean_grad", 0.0)) for it in items])
        mean_k = _mean([float(it.get("mean_kphi", 0.0)) for it in items])
        mean_loc = _mean([float(it.get("local_frac", 0.0)) for it in items])
        html.append(
            f"<tr><td>{sig}</td><td>{cnt}</td><td>{pass_rate:.2f}</td><td>{mean_xi:.2f}</td><td>{mean_g:.3f}</td><td>{mean_k:.3f}</td><td>{mean_loc:.2f}</td></tr>"
        )
    html += [
        "</tbody></table>",
    ]

    # Legend with canonical thresholds and distribution
    total_count = sum(len(v) for v in groups.values()) or 1
    html += [
        "<h3 style='margin-top:16px'>Legend</h3>",
        "<ul>",
        f"<li>|∇φ| threshold (stable): {PHASE_GRADIENT_THRESHOLD}</li>",
        f"<li>|K_φ| threshold (local safety): {PHASE_CURVATURE_ABS_THRESHOLD}</li>",
        f"<li>ΔΦ_s threshold (confinement): {STRUCTURAL_POTENTIAL_DELTA_THRESHOLD}</li>",
        "</ul>",
        "<h3>Distribution by Signature</h3>",
        "<ul>",
    ]
    for sig, items in groups.items():
        pct = (100.0 * len(items)) / float(total_count)
        html.append(f"<li>{sig or '(none)'}: {len(items)} ({pct:.1f}%)</li>")
    html += [
        "</ul>",
        "<p style='margin-top:16px;color:#666'>Thresholds and messages are centralized in tnfr.telemetry.constants and tnfr.operators.grammar. This is an early demonstrator; it does not claim quantitative mapping to SI/QED values.</p>",
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
        "Z",
        "symbol",
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
                        str(r.get("Z", "")),
                        str(r.get("symbol", "")),
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


def _write_groups_csv(groups: Dict[str, List[dict[str, Any]]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    
    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0
    cols = [
        "signature",
        "count",
        "pass_rate",
        "mean_xi_c",
        "mean_grad",
        "mean_kphi",
        "mean_local_frac",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for sig, items in groups.items():
            cnt = len(items)
            pass_rate = _mean([1.0 if it.get("u6_ok", False) else 0.0 for it in items])
            mean_xi = _mean([float(it.get("xi_c", 0.0)) for it in items])
            mean_g = _mean([float(it.get("mean_grad", 0.0)) for it in items])
            mean_k = _mean([float(it.get("mean_kphi", 0.0)) for it in items])
            mean_loc = _mean([float(it.get("local_frac", 0.0)) for it in items])
            f.write(
                ",".join(
                    [
                        sig,
                        str(cnt),
                        f"{pass_rate:.6f}",
                        f"{mean_xi:.6f}",
                        f"{mean_g:.6f}",
                        f"{mean_k:.6f}",
                        f"{mean_loc:.6f}",
                    ]
                )
                + "\n"
            )


def _write_summary_json(groups: Dict[str, List[dict[str, Any]]], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    
    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0
    summary: Dict[str, Dict[str, float]] = {}
    for sig, items in groups.items():
        summary[sig] = {
            "count": float(len(items)),
            "pass_rate": _mean([1.0 if it.get("u6_ok", False) else 0.0 for it in items]),
            "mean_xi_c": _mean([float(it.get("xi_c", 0.0)) for it in items]),
            "mean_grad": _mean([float(it.get("mean_grad", 0.0)) for it in items]),
            "mean_kphi": _mean([float(it.get("mean_kphi", 0.0)) for it in items]),
            "mean_local_frac": _mean([float(it.get("local_frac", 0.0)) for it in items]),
        }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_dumps(summary))


def main() -> str:
    rows = [analyze_element(Z) for Z in range(1, 11)]
    base = os.path.join(os.path.dirname(__file__), "output")
    out_html = os.path.join(base, "periodic_table_atlas.html")
    out_jsonl = os.path.join(base, "periodic_table_atlas.jsonl")
    out_csv = os.path.join(base, "periodic_table_atlas.csv")
    render_html(rows, out_path=out_html)
    _write_jsonl(rows, out_jsonl)
    _write_csv(rows, out_csv)
    # Grouped outputs
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sig = str(r.get("signature", ""))
        groups.setdefault(sig, []).append(r)
    out_groups_csv = os.path.join(base, "periodic_table_atlas_by_signature.csv")
    out_summary_json = os.path.join(base, "periodic_table_atlas_summary.json")
    _write_groups_csv(groups, out_groups_csv)
    _write_summary_json(groups, out_summary_json)
    return out_html


if __name__ == "__main__":
    path = main()
    print(f"Wrote periodic table atlas to: {path}")
