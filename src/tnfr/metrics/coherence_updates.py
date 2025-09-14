"""Coherence and observer-related metric updates."""

from __future__ import annotations

import math
from typing import Any

from ..alias import get_attr, set_attr
from ..constants import get_aliases, get_param
from ..glyph_history import append_metric
from ..logging_utils import get_logger
from ..metrics_utils import compute_coherence
from ..observers import glyph_load, kuramoto_order, phase_sync
from ..sense import sigma_vector

ALIAS_DNFR = get_aliases("DNFR")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_SI = get_aliases("SI")
ALIAS_DSI = get_aliases("DSI")
ALIAS_VF = get_aliases("VF")
ALIAS_DVF = get_aliases("DVF")
ALIAS_D2VF = get_aliases("D2VF")

logger = get_logger(__name__)

__all__ = [
    "_update_coherence",
    "_record_metrics",
    "_update_phase_sync",
    "_update_sigma",
    "_track_stability",
    "_aggregate_si",
]


# ---------------------------------------------------------------------------
# Legacy metrics from dynamics
# ---------------------------------------------------------------------------


def _update_coherence(G, hist) -> None:
    """Update network coherence and related means."""

    C, dnfr_mean, depi_mean = compute_coherence(G, return_means=True)
    _record_metrics(
        hist,
        (C, "C_steps"),
        (dnfr_mean, "dnfr_mean"),
        (depi_mean, "depi_mean"),
    )

    wbar_w = int(get_param(G, "WBAR_WINDOW"))
    cs = hist["C_steps"]
    if cs:
        w = min(len(cs), max(1, wbar_w))
        wbar = sum(cs[-w:]) / w
        _record_metrics(hist, (wbar, "W_bar"))


def _record_metrics(
    hist: dict[str, Any], *pairs: tuple[Any, str], evaluate: bool = False
) -> None:
    """Generic recorder for metric values."""

    for value, key in pairs:
        append_metric(hist, key, value() if evaluate else value)


def _update_phase_sync(G, hist) -> None:
    """Capture phase synchrony and Kuramoto order."""

    ps = phase_sync(G)
    ko = kuramoto_order(G)
    _record_metrics(
        hist,
        (ps, "phase_sync"),
        (ko, "kuramoto_R"),
    )


def _update_sigma(G, hist) -> None:
    """Record glyph load and associated Σ⃗ vector."""

    win = int(get_param(G, "GLYPH_LOAD_WINDOW"))
    gl = glyph_load(G, window=win)
    _record_metrics(
        hist,
        (gl.get("_estabilizadores", 0.0), "glyph_load_estab"),
        (gl.get("_disruptivos", 0.0), "glyph_load_disr"),
    )

    dist = {k: v for k, v in gl.items() if not k.startswith("_")}
    sig = sigma_vector(dist)
    _record_metrics(
        hist,
        (sig.get("x", 0.0), "sense_sigma_x"),
        (sig.get("y", 0.0), "sense_sigma_y"),
        (sig.get("mag", 0.0), "sense_sigma_mag"),
        (sig.get("angle", 0.0), "sense_sigma_angle"),
    )


def _track_stability(G, hist, dt, eps_dnfr, eps_depi):
    """Track per-node stability and derivative metrics."""

    stables = 0
    total = max(1, G.number_of_nodes())
    delta_si_sum = 0.0
    delta_si_count = 0
    B_sum = 0.0
    B_count = 0

    for _, nd in G.nodes(data=True):
        if (
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) <= eps_dnfr
            and abs(get_attr(nd, ALIAS_DEPI, 0.0)) <= eps_depi
        ):
            stables += 1

        Si_curr = get_attr(nd, ALIAS_SI, 0.0)
        Si_prev = nd.get("_prev_Si", Si_curr)
        dSi = Si_curr - Si_prev
        nd["_prev_Si"] = Si_curr
        set_attr(nd, ALIAS_DSI, dSi)
        delta_si_sum += dSi
        delta_si_count += 1

        vf_curr = get_attr(nd, ALIAS_VF, 0.0)
        vf_prev = nd.get("_prev_vf", vf_curr)
        dvf_dt = (vf_curr - vf_prev) / dt
        dvf_prev = nd.get("_prev_dvf", dvf_dt)
        B = (dvf_dt - dvf_prev) / dt
        nd["_prev_vf"] = vf_curr
        nd["_prev_dvf"] = dvf_dt
        set_attr(nd, ALIAS_DVF, dvf_dt)
        set_attr(nd, ALIAS_D2VF, B)
        B_sum += B
        B_count += 1

    hist["stable_frac"].append(stables / total)
    hist["delta_Si"].append(
        delta_si_sum / delta_si_count if delta_si_count else 0.0
    )
    hist["B"].append(B_sum / B_count if B_count else 0.0)


def _aggregate_si(G, hist):
    """Aggregate Si statistics across nodes."""

    try:
        thr_sel = get_param(G, "SELECTOR_THRESHOLDS")
        thr_def = get_param(G, "GLYPH_THRESHOLDS")
        si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
        si_lo = float(thr_sel.get("si_lo", thr_def.get("lo", 0.33)))

        sis = [
            s
            for _, nd in G.nodes(data=True)
            if not math.isnan(s := get_attr(nd, ALIAS_SI, float("nan")))
        ]

        total = 0.0
        hi_count = 0
        lo_count = 0
        for s in sis:
            total += s
            if s >= si_hi:
                hi_count += 1
            if s <= si_lo:
                lo_count += 1

        n = len(sis)
        if n:
            hist["Si_mean"].append(total / n)
            hist["Si_hi_frac"].append(hi_count / n)
            hist["Si_lo_frac"].append(lo_count / n)
        else:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("Si aggregation failed: %s", exc)
