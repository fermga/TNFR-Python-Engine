"""Glyph selection."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx  # type: ignore[import-untyped]

from .constants import DEFAULTS
from .constants.core import SELECTOR_THRESHOLD_DEFAULTS
from .helpers.numeric import clamp01
from .metrics_utils import compute_dnfr_accel_max
from .collections_utils import is_non_string_sequence


HYSTERESIS_GLYPHS: set[str] = {"IL", "OZ", "ZHIR", "THOL", "NAV", "RA"}

__all__ = (
    "_selector_thresholds",
    "_norms_para_selector",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
)



@lru_cache(maxsize=None)
def _build_selector_thresholds(
    graph_id: int,
    thr_sel_items: tuple[tuple[str, float], ...],
    thr_def_items: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    """Construct threshold dict once per graph.

    Parameters are hashable representations of the selector and legacy
    thresholds, enabling memoisation via ``lru_cache``. The returned dictionary
    is reused for subsequent calls with the same arguments.
    """

    thr_sel = dict(thr_sel_items)
    thr_def = dict(thr_def_items)

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


def _selector_thresholds(G: "nx.Graph") -> dict[str, float]:
    """Return normalised hi/lo thresholds for Si, ΔNFR and acceleration.

    Combines ``SELECTOR_THRESHOLDS`` with legacy ``GLYPH_THRESHOLDS`` for Si
    cutoffs. All values are clamped to ``[0, 1]``. Results are memoised so that
    each graph builds the thresholds only once.
    """

    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    glyph_defaults = DEFAULTS.get("GLYPH_THRESHOLDS", {})
    thr_def = {**glyph_defaults, **G.graph.get("GLYPH_THRESHOLDS", {})}

    return _build_selector_thresholds(
        id(G),
        tuple(sorted(thr_sel.items())),
        tuple(sorted(thr_def.items())),
    )


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
    # Cache threshold lookups to avoid repeated dictionary access.
    si_hi = thr["si_hi"]
    si_lo = thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]
    dnfr_lo = thr["dnfr_lo"]
    accel_hi = thr["accel_hi"]
    accel_lo = thr["accel_lo"]

    d_si = min(abs(Si - si_hi), abs(Si - si_lo))
    d_dn = min(abs(dnfr - dnfr_hi), abs(dnfr - dnfr_lo))
    d_ac = min(abs(accel - accel_hi), abs(accel - accel_lo))
    certeza = min(d_si, d_dn, d_ac)
    if certeza < margin:
        hist = nd.get("glyph_history")
        if not is_non_string_sequence(hist) or not hist:
            return None
        prev = hist[-1]
        if isinstance(prev, str) and prev in HYSTERESIS_GLYPHS:
            return prev
    return None
