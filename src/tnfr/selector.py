"""Selección de glifos."""
from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from .constants import DEFAULTS, ALIAS_DNFR, ALIAS_D2EPI
from .helpers import clamp01, get_attr, compute_dnfr_accel_max


HYSTERESIS_GLYPHS = ("IL", "OZ", "ZHIR", "THOL", "NAV", "RA")

__all__ = [
    "_selector_thresholds",
    "_norms_para_selector",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
]


def _selector_thresholds(G: nx.Graph) -> dict:
    """Retorna umbrales normalizados hi/lo para Si, ΔNFR y aceleración.

    Combina ``SELECTOR_THRESHOLDS`` con ``GLYPH_THRESHOLDS`` (legado) para
    los cortes de Si. Todos los valores se claman a ``[0, 1]``.
    """
    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    glyph_defaults = DEFAULTS.get("GLYPH_THRESHOLDS", {})
    thr_def = {**glyph_defaults, **G.graph.get("GLYPH_THRESHOLDS", {})}

    si_hi = clamp01(
        float(
            thr_sel.get(
                "si_hi",
                thr_def.get("hi", glyph_defaults.get("hi", 0.66)),
            )
        )
    )
    si_lo = clamp01(
        float(
            thr_sel.get(
                "si_lo",
                thr_def.get("lo", glyph_defaults.get("lo", 0.33)),
            )
        )
    )
    dnfr_hi = clamp01(
        float(
            thr_sel.get(
                "dnfr_hi", sel_defaults.get("dnfr_hi", 0.5)
            )
        )
    )
    dnfr_lo = clamp01(
        float(
            thr_sel.get(
                "dnfr_lo", sel_defaults.get("dnfr_lo", 0.1)
            )
        )
    )
    acc_hi = clamp01(
        float(
            thr_sel.get(
                "accel_hi", sel_defaults.get("accel_hi", 0.5)
            )
        )
    )
    acc_lo = clamp01(
        float(
            thr_sel.get(
                "accel_lo", sel_defaults.get("accel_lo", 0.1)
            )
        )
    )

    return {
        "si_hi": si_hi,
        "si_lo": si_lo,
        "dnfr_hi": dnfr_hi,
        "dnfr_lo": dnfr_lo,
        "accel_hi": acc_hi,
        "accel_lo": acc_lo,
    }


def _norms_para_selector(G: nx.Graph) -> dict:
    """Calcula y guarda en ``G.graph`` los máximos para normalizar |ΔNFR| y |d2EPI/dt2|."""
    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    return norms


def _calc_selector_score(Si: float, dnfr: float, accel: float, weights: Dict[str, float]) -> float:
    """Calcula un ``score`` ponderado normalizando los pesos suministrados."""
    w_si = float(weights.get("w_si", 0.5))
    w_dn = float(weights.get("w_dnfr", 0.3))
    w_ac = float(weights.get("w_accel", 0.2))
    s = max(1e-9, w_si + w_dn + w_ac)
    w_si, w_dn, w_ac = w_si / s, w_dn / s, w_ac / s
    return w_si * Si + w_dn * (1.0 - dnfr) + w_ac * (1.0 - accel)


def _apply_selector_hysteresis(
    nd: Dict[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    thr: Dict[str, float],
    margin: float,
) -> str | None:
    """Aplica histéresis devolviendo el glifo previo si se está cerca de umbrales."""
    d_si = min(abs(Si - thr["si_hi"]), abs(Si - thr["si_lo"]))
    d_dn = min(abs(dnfr - thr["dnfr_hi"]), abs(dnfr - thr["dnfr_lo"]))
    d_ac = min(abs(accel - thr["accel_hi"]), abs(accel - thr["accel_lo"]))
    certeza = min(d_si, d_dn, d_ac)
    if certeza < margin:
        hist = nd.get("hist_glifos")
        if hist:
            prev = hist[-1]
            if isinstance(prev, str) and prev in HYSTERESIS_GLYPHS:
                return prev
    return None

