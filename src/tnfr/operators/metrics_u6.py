"""Experimental U6 telemetry and metrics.

This module isolates U6 experimental functions to avoid impacting the
stable typing surface of tnfr.operators.metrics. Functions here may use
relaxed typing and conservative imports.
"""

from __future__ import annotations

from typing import Any, Dict

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_VF, ALIAS_D2EPI

__all__ = [
    "measure_tau_relax_observed",
    "measure_nonlinear_accumulation",
    "compute_bifurcation_index",
]


def _get_node_attr(G: Any, node: Any, aliases: tuple[str, ...], default: float = 0.0) -> float:
    value = get_attr(G.nodes[node], aliases, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def measure_tau_relax_observed(
    G: Any,
    node_id: Any,
    coherence_threshold: float = 0.95,
    epsilon_c: float = 0.05,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """Measure observed relaxation time τ_relax after destabilizer application.

    Returns a snapshot; actual monitoring loop is left to the caller.
    """
    dnfr_initial = abs(_get_node_attr(G, node_id, ALIAS_DNFR))
    vf = _get_node_attr(G, node_id, ALIAS_VF)

    # Approximate local coherence using neighborhood ΔNFR variability
    neighbors = (
        list(getattr(G, "neighbors", lambda n: [])(node_id)) if hasattr(G, "neighbors") else []
    )
    dnfr_vals = [abs(_get_node_attr(G, node_id, ALIAS_DNFR))]
    for nb in neighbors:
        try:
            dnfr_vals.append(abs(_get_node_attr(G, nb, ALIAS_DNFR)))
        except Exception:
            continue
    if len(dnfr_vals) <= 1:
        coherence_initial = 1.0
    else:
        mean_dnfr = sum(dnfr_vals) / len(dnfr_vals)
        var_dnfr = sum((x - mean_dnfr) ** 2 for x in dnfr_vals) / len(dnfr_vals)
        sigma = var_dnfr**0.5
        dnfr_max = max(dnfr_vals)
        coherence_initial = 1.0 if dnfr_max == 0 else max(0.0, min(1.0, 1.0 - (sigma / dnfr_max)))

    # Spectral topological estimate (existing proxy)
    try:
        from tnfr.utils.topology import compute_k_top_spectral  # type: ignore

        k_top = compute_k_top_spectral(G)
    except Exception:
        k_top = 1.0

    k_op = 1.0  # operator-specific factor (refine later with glyph mapping)
    spectral_tau = (k_top / max(vf, 0.01)) * k_op * 3.0

    # Attempt Liouvillian slow-mode relaxation time integration.
    # Graceful fallback if spectrum unavailable.
    liouv_tau = None
    slow_mode_real = None
    try:
        # Preferred: use proper Liouvillian spectrum computation
        from tnfr.mathematics.liouville import get_liouvillian_spectrum, get_slow_relaxation_mode  # type: ignore

        liouv_eigs = get_liouvillian_spectrum(G)
        if liouv_eigs is not None:
            slow_mode = get_slow_relaxation_mode(liouv_eigs)
            if slow_mode is not None:
                slow_mode_real = float(slow_mode.real)
                if abs(slow_mode_real) > 1e-12:
                    liouv_tau = 1.0 / abs(slow_mode_real)
    except Exception:
        liouv_tau = None

    # Final τ selection: prefer Liouvillian slow-mode if available
    tau_relax_estimated = liouv_tau if liouv_tau is not None else spectral_tau

    return {
        "metric_type": "u6_relaxation_time",
        "tau_relax_observed": None,
        "dnfr_initial": dnfr_initial,
        "dnfr_final": None,
        "coherence_initial": coherence_initial,
        "coherence_final": None,
        "coherence_threshold": coherence_threshold,
        "epsilon_c": epsilon_c,
        "recovery_complete": None,
        "steps_to_recovery": None,
        "vf": vf,
        "k_top": k_top,
        "estimated_tau_relax": tau_relax_estimated,
        "estimated_tau_relax_spectral": spectral_tau,
        "estimated_tau_relax_liouvillian": liouv_tau,
        "liouvillian_slow_mode_real": slow_mode_real,
        "max_steps": max_steps,
        "node_id": node_id,
        "requires_monitoring_infrastructure": True,
    }


def measure_nonlinear_accumulation(
    G: Any,
    node_id: Any,
    dnfr_before_first: float,
    dnfr_before_second: float,
    dt_separation: float,
) -> Dict[str, Any]:
    """Measure nonlinear accumulation factor α(Δt) for spacing validation."""
    dnfr_actual = abs(_get_node_attr(G, node_id, ALIAS_DNFR))
    dnfr_linear = dnfr_before_second + abs(dnfr_before_first)

    denominator = abs(dnfr_before_first * dnfr_before_second)
    if denominator < 1e-9:
        alpha = 1.0
    else:
        alpha = (dnfr_actual - dnfr_linear) / denominator

    if alpha <= 1.1:
        severity = "none"
    elif alpha <= 1.5:
        severity = "mild"
    elif alpha <= 2.0:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "metric_type": "u6_nonlinear_accumulation",
        "alpha": alpha,
        "dnfr_actual": dnfr_actual,
        "dnfr_linear": dnfr_linear,
        "dnfr_before_first": dnfr_before_first,
        "dnfr_before_second": dnfr_before_second,
        "dt_separation": dt_separation,
        "nonlinear_regime": alpha > 1.1,
        "amplification_severity": severity,
        "node_id": node_id,
    }


def compute_bifurcation_index(G: Any, node_id: Any) -> Dict[str, Any]:
    """Compute bifurcation index B = |d2EPI| / νf^2."""
    vf = _get_node_attr(G, node_id, ALIAS_VF)
    d2_epi = _get_node_attr(G, node_id, ALIAS_D2EPI)

    if vf < 0.01:
        B = 0.0
    else:
        B = abs(d2_epi) / (vf * vf)

    if B < 0.5:
        risk = "stable"
    elif B < 1.5:
        risk = "moderate"
    elif B < 3.0:
        risk = "high"
    else:
        risk = "critical"

    return {
        "metric_type": "u6_bifurcation_index",
        "B": B,
        "B_normalized": B / 3.0,
        "d2_epi_dt2": d2_epi,
        "vf": vf,
        "risk_level": risk,
        "node_id": node_id,
    }
