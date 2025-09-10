"""Glyph selection."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx  # type: ignore[import-untyped]

from .constants import DEFAULTS
from .constants.core import SELECTOR_THRESHOLD_DEFAULTS
from .helpers.numeric import clamp01
from .metrics_utils import compute_dnfr_accel_max


HYSTERESIS_GLYPHS: set[str] = {"IL", "OZ", "ZHIR", "THOL", "NAV", "RA"}

__all__ = (
    "_selector_thresholds",
    "_norms_para_selector",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
)


def _selector_thresholds(G: "nx.Graph") -> dict:
    """Return normalised hi/lo thresholds for Si, ΔNFR and acceleration.

    Combines ``SELECTOR_THRESHOLDS`` with legacy ``GLYPH_THRESHOLDS`` for Si
    cutoffs. All values are clamped to ``[0, 1]``.
    """
    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    glyph_defaults = DEFAULTS.get("GLYPH_THRESHOLDS", {})
    thr_def = {**glyph_defaults, **G.graph.get("GLYPH_THRESHOLDS", {})}

    specs = {
        "si_hi": ("hi", SELECTOR_THRESHOLD_DEFAULTS["si_hi"]),
        "si_lo": ("lo", SELECTOR_THRESHOLD_DEFAULTS["si_lo"]),
        "dnfr_hi": (None, SELECTOR_THRESHOLD_DEFAULTS["dnfr_hi"]),
        "dnfr_lo": (None, SELECTOR_THRESHOLD_DEFAULTS["dnfr_lo"]),
        "accel_hi": (None, SELECTOR_THRESHOLD_DEFAULTS["accel_hi"]),
        "accel_lo": (None, SELECTOR_THRESHOLD_DEFAULTS["accel_lo"]),
    }

    out: dict[str, float] = {}
    for key, (legacy, default) in specs.items():
        if legacy is not None:
            val = thr_sel.get(key, thr_def.get(legacy, default))
        else:
            val = thr_sel.get(key, default)
        out[key] = clamp01(float(val))
    return out


def _norms_para_selector(G: "nx.Graph") -> dict:
    """Compute and store maxima in ``G.graph`` to normalise |ΔNFR| and
    |d²EPI/dt²|."""
    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    return norms


def _calc_selector_score(
    Si: float, dnfr: float, accel: float, weights: dict[str, float]
) -> float:
    """Compute a weighted score assuming normalised weights."""
    return (
        weights["w_si"] * Si
        + weights["w_dnfr"] * (1.0 - dnfr)
        + weights["w_accel"] * (1.0 - accel)
    )


def _dist_to_threshold(value: float, hi: float, lo: float) -> float:
    """Return distance from ``value`` to nearest of ``hi`` or ``lo``."""
    return min(abs(value - hi), abs(value - lo))


def _apply_selector_hysteresis(
    nd: dict[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    thr: dict[str, float],
    margin: float,
) -> str | None:
    """Apply hysteresis, returning the previous glyph when close to
    thresholds."""
    d_si = _dist_to_threshold(Si, thr["si_hi"], thr["si_lo"])
    d_dn = _dist_to_threshold(dnfr, thr["dnfr_hi"], thr["dnfr_lo"])
    d_ac = _dist_to_threshold(accel, thr["accel_hi"], thr["accel_lo"])
    certeza = min(d_si, d_dn, d_ac)
    if certeza < margin:
        hist = nd.get("glyph_history")
        if not isinstance(hist, Sequence) or not hist:
            return None
        prev = hist[-1]
        if isinstance(prev, str) and prev in HYSTERESIS_GLYPHS:
            return prev
    return None
