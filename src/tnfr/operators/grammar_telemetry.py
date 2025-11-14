"""TNFR Grammar: U6 Telemetry Functions

Phase gradient, phase curvature, and coherence length telemetry for U6 validation.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from typing import Any

def warn_phase_gradient_telemetry(
    G: Any,
    *,
    threshold: float = 0.38,
) -> tuple[bool, dict[str, float], str, list[Any]]:
    """Emit non-blocking telemetry warning for |∇φ| (phase gradient).

    Read-only safety check: computes |∇φ| per node and summarizes:
    - max, mean across nodes
    - fraction of nodes above threshold

    Returns (safe, stats, message, flagged_nodes) where safe indicates
    mean and max are below threshold (stable regime). Always non-blocking.

    Safety criterion: |∇φ| < 0.38 (stable operation)

    References: AGENTS.md Structural Fields; fields.compute_phase_gradient
    """
    try:
        from ..physics.fields import compute_phase_gradient
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        # If dependencies missing, be conservative but non-blocking
        return True, {"max": 0.0, "mean": 0.0, "frac_over": 0.0}, (
            "U6 (|∇φ|): telemetry unavailable (skipping)"
        ), []

    grad = compute_phase_gradient(G)
    if not grad:
        return True, {"max": 0.0, "mean": 0.0, "frac_over": 0.0}, (
            "U6 (|∇φ|): no nodes (trivial)"
        ), []

    vals = np.array(list(grad.values()), dtype=float)
    max_v = float(np.max(vals))
    mean_v = float(np.mean(vals))
    flagged = [n for n, v in grad.items() if float(abs(v)) >= float(threshold)]
    frac_over = float(len(flagged) / max(len(grad), 1))

    safe = bool((max_v < threshold) and (mean_v < threshold))
    if safe:
        msg = (
            f"U6 (|∇φ|): PASS - mean={mean_v:.3f}, max={max_v:.3f} < {threshold:.2f} "
            f"(stable)."
        )
    else:
        msg = (
            f"U6 (|∇φ|): WARN - mean={mean_v:.3f}, max={max_v:.3f} ≥ {threshold:.2f}. "
            f"Flagged {len(flagged)}/{len(grad)} loci (frac={frac_over:.2f})."
        )

    stats = {"max": max_v, "mean": mean_v, "frac_over": frac_over}
    return safe, stats, msg, flagged


def warn_phase_curvature_telemetry(
    G: Any,
    *,
    abs_threshold: float = 3.0,
    multiscale_check: bool = True,
    alpha_hint: float | None = 2.76,
    tolerance_factor: float = 2.0,
    fit_min_r2: float = 0.5,
) -> tuple[bool, dict[str, float | int | bool], str, list[Any]]:
    """Emit non-blocking telemetry warning for K_φ (phase curvature).

    Checks two safety aspects:
    - Local hotspots: count of nodes with |K_φ| ≥ abs_threshold (default 3.0)
    - Multiscale safety: var(K_φ) ~ 1/r^α behavior via k_phi_multiscale_safety

    Returns (safe, stats, message, hotspots).
    Safe if no local hotspots and multiscale safety passes. Non-blocking.
    """
    try:
        from ..physics.fields import (
            compute_phase_curvature,
            k_phi_multiscale_safety,
        )
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        return True, {"hotspots": 0, "max_abs": 0.0, "multiscale_safe": True}, (
            "U6 (K_φ): telemetry unavailable (skipping)"
        ), []

    kphi = compute_phase_curvature(G)
    if not kphi:
        return True, {"hotspots": 0, "max_abs": 0.0, "multiscale_safe": True}, (
            "U6 (K_φ): no nodes (trivial)"
        ), []

    vals = [abs(float(v)) for v in kphi.values()]
    max_abs = float(max(vals)) if vals else 0.0
    hotspots = [n for n, v in kphi.items() if abs(float(v)) >= float(abs_threshold)]

    multiscale_safe = True
    multiscale_info: dict[str, Any] | None = None
    if multiscale_check:
        multiscale_info = k_phi_multiscale_safety(
            G,
            alpha_hint=alpha_hint,
            tolerance_factor=tolerance_factor,
            fit_min_r2=fit_min_r2,
        )
        multiscale_safe = bool(multiscale_info.get("safe", True))

    safe = bool((len(hotspots) == 0) and multiscale_safe)
    if safe:
        msg = (
            f"U6 (K_φ): PASS - max|K_φ|={max_abs:.3f} < {abs_threshold:.2f} "
            f"and multiscale_safe={multiscale_safe}."
        )
    else:
        msg = (
            f"U6 (K_φ): WARN - hotspots={len(hotspots)} (|K_φ|≥{abs_threshold:.2f}), "
            f"max|K_φ|={max_abs:.3f}, multiscale_safe={multiscale_safe}."
        )

    stats: dict[str, float | int | bool] = {
        "hotspots": int(len(hotspots)),
        "max_abs": max_abs,
        "multiscale_safe": bool(multiscale_safe),
    }
    # Optionally attach multiscale fit details (non-breaking)
    if multiscale_info is not None:
        fit = multiscale_info.get("fit", {})
        stats.update(
            {
                "alpha": float(fit.get("alpha", 0.0)),
                "r_squared": float(fit.get("r_squared", 0.0)),
            }
        )

    return safe, stats, msg, hotspots


def warn_coherence_length_telemetry(
    G: Any,
    *,
    regime_multipliers: tuple[float, float] = (1.0, 3.0),
) -> tuple[bool, dict[str, float | str], str]:
    """Emit non-blocking telemetry warning for ξ_C (coherence length).

    Classifies regimes based on ξ_C relative to graph distances:
    - stable: ξ_C < mean_path_length
    - watch: mean_path_length ≤ ξ_C ≤ 3×mean_path_length
    - alert: ξ_C > 3×mean_path_length
    - critical: ξ_C ≥ system_diameter

    Returns (safe, stats, message). Always non-blocking.
    """
    try:
        from ..physics.fields import estimate_coherence_length
        import networkx as nx  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        return True, {"xi_c": 0.0, "severity": "unknown"}, (
            "U6 (ξ_C): telemetry unavailable (skipping)"
        )

    xi_c = float(estimate_coherence_length(G))

    # Compute mean shortest path length (by component) and system diameter
    def _mean_path_length(H: Any) -> float:
        try:
            if nx.is_connected(H):  # type: ignore[attr-defined]
                return float(nx.average_shortest_path_length(H))  # type: ignore[attr-defined]
        except Exception:
            pass
        # For disconnected graphs: weighted average over components
        m = 0.0
        total = 0
        for comp in nx.connected_components(H):  # type: ignore[attr-defined]
            CC = H.subgraph(comp)
            n = CC.number_of_nodes()
            if n >= 2:
                try:
                    m_comp = float(nx.average_shortest_path_length(CC))  # type: ignore[attr-defined]
                except Exception:
                    m_comp = 0.0
                m += m_comp * n
                total += n
        return float(m / total) if total > 0 else 0.0

    def _diameter(H: Any) -> float:
        try:
            if nx.is_connected(H):  # type: ignore[attr-defined]
                return float(nx.diameter(H))  # type: ignore[attr-defined]
        except Exception:
            pass
        # For disconnected, take max of component diameters
        diam = 0.0
        for comp in nx.connected_components(H):  # type: ignore[attr-defined]
            CC = H.subgraph(comp)
            try:
                d_comp = float(nx.diameter(CC))  # type: ignore[attr-defined]
            except Exception:
                d_comp = 0.0
            diam = max(diam, d_comp)
        return diam

    mpl = _mean_path_length(G)
    diam = _diameter(G)

    # Regime multipliers
    base, watch_mult = regime_multipliers
    watch_thr = float(base * mpl)  # typically 1×
    alert_thr = float(watch_mult * mpl)  # typically 3×

    # Classify severity
    if xi_c >= max(diam, 0.0) and diam > 0.0:
        severity = "critical"
        safe = False
    elif xi_c > alert_thr and mpl > 0.0:
        severity = "alert"
        safe = False
    elif xi_c >= watch_thr and mpl > 0.0:
        severity = "watch"
        safe = False
    else:
        severity = "stable"
        safe = True

    if severity == "stable":
        msg = (
            f"U6 (ξ_C): PASS - ξ_C={xi_c:.2f} < mean_path_length≈{mpl:.2f} "
            f"(stable regime)."
        )
    elif severity == "watch":
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.2f} ≥ mean_path_length≈{mpl:.2f}. "
            f"Long-range correlations emerging. Monitor closely."
        )
    elif severity == "alert":
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.2f} > {watch_mult:.1f}×mean_path_length≈{mpl:.2f}. "
            f"Strong long-range correlations. Potential transition."
        )
    else:  # critical
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.2f} ≥ system_diameter≈{diam:.2f}. "
            f"Critical approach: system-wide reorganization imminent."
        )

    stats = {"xi_c": xi_c, "mean_path_length": mpl, "diameter": diam, "severity": severity}
    return safe, stats, msg


