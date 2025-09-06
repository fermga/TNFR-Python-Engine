"""Glyph selection."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .constants import DEFAULTS
from .constants.core import SELECTOR_THRESHOLD_DEFAULTS
from .helpers import clamp01, compute_dnfr_accel_max


HYSTERESIS_GLYPHS = ("IL", "OZ", "ZHIR", "THOL", "NAV", "RA")

__all__ = [
    "_selector_thresholds",
    "_norms_para_selector",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
]


def _selector_thresholds(G: "nx.Graph") -> dict:
    """Return normalised hi/lo thresholds for Si, ΔNFR and acceleration.

    Combines ``SELECTOR_THRESHOLDS`` with legacy ``GLYPH_THRESHOLDS`` for Si
    cutoffs. All values are clamped to ``[0, 1]``.
    """
    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    glyph_defaults = DEFAULTS.get("GLYPH_THRESHOLDS", {})
    thr_def = {**glyph_defaults, **G.graph.get("GLYPH_THRESHOLDS", {})}

    def _get_threshold(
        key: str, default: float, legacy: str | None = None
    ) -> float:
        """Obtain ``key`` from ``thr_sel`` honouring legacy keys.

        When ``legacy`` is provided ``thr_def`` is used as fallback,
        allowing compatibility with ``GLYPH_THRESHOLDS`` from older
        versions.
        """

        if legacy is not None:
            val = thr_sel.get(key, thr_def.get(legacy, default))
        else:
            val = thr_sel.get(key, default)
        return clamp01(float(val))

    specs = [
        (
            "si_hi",
            thr_def.get(
                "hi",
                glyph_defaults.get("hi", SELECTOR_THRESHOLD_DEFAULTS["si_hi"]),
            ),
            "hi",
        ),
        (
            "si_lo",
            thr_def.get(
                "lo",
                glyph_defaults.get("lo", SELECTOR_THRESHOLD_DEFAULTS["si_lo"]),
            ),
            "lo",
        ),
        (
            "dnfr_hi",
            sel_defaults.get(
                "dnfr_hi", SELECTOR_THRESHOLD_DEFAULTS["dnfr_hi"]
            ),
            None,
        ),
        (
            "dnfr_lo",
            sel_defaults.get(
                "dnfr_lo", SELECTOR_THRESHOLD_DEFAULTS["dnfr_lo"]
            ),
            None,
        ),
        (
            "accel_hi",
            sel_defaults.get(
                "accel_hi", SELECTOR_THRESHOLD_DEFAULTS["accel_hi"]
            ),
            None,
        ),
        (
            "accel_lo",
            sel_defaults.get(
                "accel_lo", SELECTOR_THRESHOLD_DEFAULTS["accel_lo"]
            ),
            None,
        ),
    ]

    return {
        key: _get_threshold(key, default, legacy)
        for key, default, legacy in specs
    }


def _norms_para_selector(G: "nx.Graph") -> dict:
    """Compute and store maxima in ``G.graph`` to normalise |ΔNFR| and |d²EPI/dt²|."""
    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    return norms


def _calc_selector_score(
    Si: float, dnfr: float, accel: float, weights: Dict[str, float]
) -> float:
    """Compute a weighted score assuming normalised weights."""
    return (
        weights["w_si"] * Si
        + weights["w_dnfr"] * (1.0 - dnfr)
        + weights["w_accel"] * (1.0 - accel)
    )


def _apply_selector_hysteresis(
    nd: Dict[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    thr: Dict[str, float],
    margin: float,
) -> str | None:
    """Apply hysteresis, returning the previous glyph when close to thresholds."""
    d_si = min(abs(Si - thr["si_hi"]), abs(Si - thr["si_lo"]))
    d_dn = min(abs(dnfr - thr["dnfr_hi"]), abs(dnfr - thr["dnfr_lo"]))
    d_ac = min(abs(accel - thr["accel_hi"]), abs(accel - thr["accel_lo"]))
    certeza = min(d_si, d_dn, d_ac)
    if certeza < margin:
        hist = nd.get("glyph_history")
        if not isinstance(hist, Sequence) or not hist:
            return None
        prev = hist[-1]
        if isinstance(prev, str) and prev in HYSTERESIS_GLYPHS:
            return prev
    return None
