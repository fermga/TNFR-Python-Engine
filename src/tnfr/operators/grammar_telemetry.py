"""TNFR Grammar: U6 Telemetry Functions

Phase gradient, phase curvature, and coherence length telemetry for U6 validation.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..config.defaults_core import (
    K_PHI_ASYMPTOTIC_ALPHA,
    K_PHI_CURVATURE_THRESHOLD,
    STATISTICAL_SIGNIFICANCE_THRESHOLD,
)
from ..config.parsing import parse_bool
from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,  # heuristic ≈ 0.196 (π/16)
)
from ..mathematics.unified_numerical import np


def _positive_policy_value(value: Any, label: str) -> float:
    """Admit a positive represented-real policy value before comparisons."""
    admitted, _ = finite_represented_real(value, label)
    if admitted <= 0.0:
        raise ValueError(f"{label} must be positive")
    return admitted


def warn_phase_gradient_telemetry(
    G: Any,
    *,
    threshold: float = GRAD_PHI_CANONICAL_THRESHOLD,  # heuristic early-warning (audit 2026: not derived)
) -> tuple[bool, dict[str, float | bool], str, list[Any]]:
    """Emit non-blocking telemetry warning for |∇φ| (phase gradient).

    Read-only safety check: computes |∇φ| per node and summarizes:
    - max, mean across nodes
    - fraction of nodes above threshold

    Returns (safe, stats, message, flagged_nodes) where safe indicates
    observed mean and max are below the selected threshold. Missing observations
    return ``safe=False`` and ``available=False``, not a passing zero.
    Invalid thresholds raise. Always non-blocking.

    Safety criterion (heuristic early-warning, audit 2026: NOT a derived bound;
    the kinematic |∇φ| bound is π (phase wrap), the same as K_φ).

    References: AGENTS.md Structural Fields; fields.compute_phase_gradient
    """
    threshold = _positive_policy_value(threshold, "phase-gradient threshold")
    try:
        from ..physics.fields import compute_phase_gradient
    except ImportError:
        return (
            False,
            {
                "max": math.nan,
                "mean": math.nan,
                "frac_over": math.nan,
                "available": False,
            },
            ("U6 (|∇φ|): telemetry unavailable (skipping)"),
            [],
        )

    grad = compute_phase_gradient(G)
    if not grad:
        return (
            False,
            {
                "max": math.nan,
                "mean": math.nan,
                "frac_over": math.nan,
                "available": False,
            },
            ("U6 (|∇φ|): unavailable; no node observations."),
            [],
        )

    vals = np.array(list(grad.values()), dtype=float)
    max_v = float(np.max(vals))
    mean_v = float(np.mean(vals))
    flagged = [n for n, v in grad.items() if float(abs(v)) >= float(threshold)]
    frac_over = float(len(flagged) / max(len(grad), 1))

    safe = bool((max_v < threshold) and (mean_v < threshold))
    if safe:
        msg = (
            f"U6 (|∇φ|): PASS - mean={mean_v:.3f}, max={max_v:.3f} < {threshold:.2f} "
            f"(below the selected warning cut)."
        )
    else:
        msg = (
            f"U6 (|∇φ|): WARN - mean={mean_v:.3f}, max={max_v:.3f} ≥ {threshold:.2f}. "
            f"Flagged {len(flagged)}/{len(grad)} loci (frac={frac_over:.2f})."
        )

    stats = {"max": max_v, "mean": mean_v, "frac_over": frac_over, "available": True}
    return safe, stats, msg, flagged


def warn_phase_curvature_telemetry(
    G: Any,
    *,
    abs_threshold: float = K_PHI_CURVATURE_THRESHOLD,
    multiscale_check: bool = True,
    alpha_hint: float | None = K_PHI_ASYMPTOTIC_ALPHA,
    tolerance_factor: float = 2.0,
    fit_min_r2: float = STATISTICAL_SIGNIFICANCE_THRESHOLD,
) -> tuple[bool, dict[str, float | int | bool], str, list[Any]]:
    """Emit non-blocking telemetry warning for K_φ (phase curvature).

    Checks two safety aspects:
    - Local hotspots: count of nodes with |K_φ| ≥ abs_threshold (default 0.9×π ≈ 2.827)
    - Multiscale safety: var(K_φ) ~ 1/r^α behavior via k_phi_multiscale_safety

    Returns (safe, stats, message, hotspots).
    Safe if no local hotspots and multiscale safety passes. Non-blocking.
    ``alpha_hint=None`` omits comparison to a selected exponent.
    ``tolerance_factor`` is an inert legacy argument retained for compatibility;
    it does not change a threshold or the multiscale verdict.
    Missing observations never pass; invalid consumed policy values raise.
    """
    abs_threshold = _positive_policy_value(abs_threshold, "curvature threshold")
    multiscale_check = parse_bool(multiscale_check)
    if multiscale_check:
        if alpha_hint is not None:
            alpha_hint = _positive_policy_value(alpha_hint, "reference decay exponent")
        fit_min_r2, _ = bounded_represented_real(
            fit_min_r2, "minimum fit R-squared", minimum=0.0, maximum=1.0
        )
    try:
        from ..physics.fields import compute_phase_curvature, k_phi_multiscale_safety
    except ImportError:
        return (
            False,
            {
                "hotspots": 0,
                "max_abs": math.nan,
                "multiscale_safe": False,
                "available": False,
            },
            ("U6 (K_φ): telemetry unavailable (skipping)"),
            [],
        )

    kphi = compute_phase_curvature(G)
    if not kphi:
        return (
            False,
            {
                "hotspots": 0,
                "max_abs": math.nan,
                "multiscale_safe": False,
                "available": False,
            },
            ("U6 (K_φ): unavailable; no node observations."),
            [],
        )

    vals = [abs(float(v)) for v in kphi.values()]
    max_abs = float(max(vals)) if vals else 0.0
    hotspots = [n for n, v in kphi.items() if abs(float(v)) >= float(abs_threshold)]

    multiscale_safe = True
    multiscale_info: dict[str, Any] | None = None
    if multiscale_check:
        multiscale_info = k_phi_multiscale_safety(
            G,
            alpha_hint=alpha_hint,
            fit_min_r2=fit_min_r2,
        )
        multiscale_safe = bool(multiscale_info["available"] and multiscale_info["safe"])
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
        "multiscale_checked": multiscale_check,
        "available": (
            bool(multiscale_info["available"]) if multiscale_info is not None else True
        ),
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
        if not stats["available"]:
            msg = "U6 (K_φ): unavailable; no admissible multiscale observations."

    return safe, stats, msg, hotspots


def warn_coherence_length_telemetry(
    G: Any,
    *,
    regime_multipliers: tuple[float, float] = (1.0, 3.0),
) -> tuple[bool, dict[str, float | str], str]:
    """Compare the fitted static ξ_C to its own structural path geometry.

    The ordered positive ``regime_multipliers=(base, alert)`` select advisory
    labels, not predictions of a transition:
    - stable: ξ_C < base×mean_path_length
    - watch: base×mean_path_length ≤ ξ_C ≤ alert×mean_path_length
    - alert: ξ_C > alert×mean_path_length
    - critical: ξ_C ≥ system_diameter

    The critical label takes precedence; the default multipliers are (1, 3).
    Uses positive reachable outgoing pair distances with ``length``, else
    ``weight``, else unit length, as in the fit. The spectral fallback is a
    dimensionless mode scale and is reported as not comparable to path length.
    Missing estimates are unavailable, never a passing zero. Returns
    (safe, stats, message); ``safe=False`` can mean unavailable evidence and
    does not certify physical instability. Always non-blocking.
    """
    base_source, watch_source = regime_multipliers
    base, _ = bounded_represented_real(
        base_source, "base regime multiplier", maximum=watch_source
    )
    base = _positive_policy_value(base, "base regime multiplier")
    watch_mult = _positive_policy_value(watch_source, "watch regime multiplier")
    try:
        from ..physics._coherence_fit import DISTANCE_DESCRIPTION, _graph_distance_rows
        from ..physics.canonical import estimate_coherence_length_with_provenance
    except ImportError:
        return (
            False,
            {"xi_c": math.nan, "severity": "unavailable"},
            ("U6 (ξ_C): telemetry unavailable (skipping)"),
        )

    estimate = estimate_coherence_length_with_provenance(G)
    xi_c = float(estimate.value)
    stats: dict[str, float | str] = {
        "xi_c": xi_c,
        "method": estimate.method,
        "distance_weighting": estimate.distance_weighting,
        "mean_path_length": math.nan,
        "diameter": math.nan,
    }
    if not math.isfinite(xi_c) or xi_c <= 0.0:
        stats["severity"] = "unavailable"
        return False, stats, "U6 (ξ_C): unavailable; no admissible estimate."
    if estimate.method != "autocorrelation_fit":
        stats["severity"] = "not_comparable"
        return (
            False,
            stats,
            "U6 (ξ_C): not assessed against path distances; "
            "spectral fallback is a dimensionless mode scale.",
        )

    # Stream the same outgoing shortest-path rows as the fitting owner. Online
    # averaging avoids an O(N²) retained distance matrix and overflowing sums.
    count = 0
    mpl = 0.0
    diam = 0.0
    for source, row in _graph_distance_rows(G, tuple(G)):
        for target, distance in row.items():
            if source == target or distance <= 0.0:
                continue
            count += 1
            mpl += (distance - mpl) / count
            diam = max(diam, distance)
    if not count:
        stats["severity"] = "unavailable"
        return False, stats, "U6 (ξ_C): unavailable; no positive reachable distances."
    stats.update(
        mean_path_length=mpl,
        diameter=diam,
        reference_geometry=DISTANCE_DESCRIPTION,
    )

    # Retain the represented inputs through multiplication so overflow or a
    # nonzero threshold lost to underflow cannot silently change the policy.
    watch_thr = _positive_policy_value(
        Fraction(base) * Fraction(mpl), "watch path threshold"
    )
    alert_thr = _positive_policy_value(
        Fraction(watch_mult) * Fraction(mpl), "alert path threshold"
    )

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
            f"U6 (ξ_C): PASS - ξ_C={xi_c:.6g} < watch_threshold={watch_thr:.6g} "
            f"({base:.6g}×mean_path_length) "
            f"(below the selected warning cut)."
        )
    elif severity == "watch":
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.6g} ≥ watch_threshold={watch_thr:.6g} "
            f"({base:.6g}×mean_path_length). "
            f"Selected static length warning."
        )
    elif severity == "alert":
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.6g} > alert_threshold={alert_thr:.6g} "
            f"({watch_mult:.6g}×mean_path_length). "
            f"Selected static length alert."
        )
    else:  # critical
        msg = (
            f"U6 (ξ_C): WARN - ξ_C={xi_c:.2f} ≥ system_diameter≈{diam:.2f}. "
            f"Selected finite-size flag; no transition prediction."
        )

    stats["severity"] = severity
    return safe, stats, msg
