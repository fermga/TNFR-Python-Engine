"""HTML helpers for TNFR examples (telemetry-only UI elements).

Currently provides a compact Safety Triad panel used by example exporters.
"""
from __future__ import annotations

from typing import Dict, Optional


def render_safety_triad_panel(
    *,
    thresholds: Dict[str, float],
    summary: Optional[Dict[str, float]] = None,
) -> str:
    """Return a compact HTML snippet for the Safety Triad panel.

    Parameters
    ----------
    thresholds : dict
        Keys expected:
          - 'phi_delta': ΔΦ_s confinement threshold (STRUCTURAL_POTENTIAL_DELTA_THRESHOLD)
          - 'grad': |∇φ| stability threshold (PHASE_GRADIENT_THRESHOLD)
          - 'kphi': |K_φ| curvature safety threshold (PHASE_CURVATURE_ABS_THRESHOLD)
    summary : dict, optional
        If provided, may include:
          - 'mean_grad': float
          - 'mean_kphi': float
          - 'max_drift': float (max |ΔΦ_s| observed)

    Notes
    -----
    - This is telemetry-only presentation; it must not gate behavior.
    - CSS for the class '.panel' should be provided by the caller page:
        .panel{border:1px solid #e0e0e0;background:#f9fbff;padding:8px 12px;
               border-radius:6px;display:inline-block;margin:8px 0}
        .panel b{display:block;margin-bottom:4px}
    """
    phi_delta = float(thresholds.get("phi_delta", 0.0))
    grad_th = float(thresholds.get("grad", 0.0))
    kphi_th = float(thresholds.get("kphi", 0.0))

    html = [
        "<div class='panel'>",
        "<b>Safety Triad (telemetry-only)</b>",
        f"<div>ΔΦ_s threshold: {phi_delta} (confinement)</div>",
        f"<div>|∇φ| threshold: {grad_th}</div>",
        f"<div>|K_φ| threshold: {kphi_th}</div>",
    ]

    if summary is not None:
        mean_g = float(summary.get("mean_grad", 0.0))
        mean_k = float(summary.get("mean_kphi", 0.0))
        max_drift = float(summary.get("max_drift", 0.0))
        html.append(
            f"<div style='margin-top:4px;color:#555'>"
            f"Report summary — mean|∇φ|={mean_g:.3f}, mean|K_φ|={mean_k:.3f}, max ΔΦ_s={max_drift:.3f}"
            f"</div>"
        )

    html.append(
        "<div style='margin-top:2px;color:#777;font-size:12px'>U6 is descriptive only; no control feedback into dynamics.</div>"
    )
    html.append("</div>")
    return "".join(html)
