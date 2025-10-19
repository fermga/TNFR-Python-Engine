"""Utilities to select glyphs based on structural metrics.

This module normalises thresholds, computes selection scores and applies
hysteresis when assigning glyphs to nodes.
"""

from __future__ import annotations

import threading
from operator import itemgetter
from typing import Any, Mapping, TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .constants import DEFAULTS
from .constants.core import SELECTOR_THRESHOLD_DEFAULTS
from .helpers.numeric import clamp01
from .metrics.common import compute_dnfr_accel_max
from .utils import is_non_string_sequence


HYSTERESIS_GLYPHS: set[str] = {"IL", "OZ", "ZHIR", "THOL", "NAV", "RA"}

__all__ = (
    "_selector_thresholds",
    "_norms_para_selector",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
)


_SelectorThresholdCacheEntry = tuple[
    tuple[tuple[str, float], ...],
    dict[str, float],
]
_SELECTOR_THRESHOLD_CACHE: WeakKeyDictionary[
    "nx.Graph",
    _SelectorThresholdCacheEntry,
] = WeakKeyDictionary()
_SELECTOR_THRESHOLD_CACHE_LOCK = threading.Lock()


def _sorted_items(mapping: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Return mapping items sorted by key.

    Parameters
    ----------
    mapping : Mapping[str, float]
        Mapping whose items will be sorted.

    Returns
    -------
    tuple[tuple[str, float], ...]
        Key-sorted items providing a hashable representation for memoisation.
    """
    return tuple(sorted(mapping.items()))


def _compute_selector_thresholds(
    thr_sel_items: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    """Construct selector thresholds for a graph.

    Parameters
    ----------
    thr_sel_items : tuple[tuple[str, float], ...]
        Selector threshold items as ``(key, value)`` pairs.

    Returns
    -------
    dict[str, float]
        Normalised thresholds for selector metrics.
    """
    thr_sel = dict(thr_sel_items)

    out: dict[str, float] = {}
    for key, default in SELECTOR_THRESHOLD_DEFAULTS.items():
        val = thr_sel.get(key, default)
        out[key] = clamp01(float(val))
    return out


def _selector_thresholds(G: "nx.Graph") -> dict[str, float]:
    """Return normalised thresholds for Si, ΔNFR and acceleration.

    Parameters
    ----------
    G : nx.Graph
        Graph whose configuration stores selector thresholds.

    Returns
    -------
    dict[str, float]
        Dictionary with clamped hi/lo thresholds, memoised per graph.
    """
    sel_defaults = DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    thr_sel = {**sel_defaults, **G.graph.get("SELECTOR_THRESHOLDS", {})}
    thr_sel_items = _sorted_items(thr_sel)

    with _SELECTOR_THRESHOLD_CACHE_LOCK:
        cached = _SELECTOR_THRESHOLD_CACHE.get(G)
        if cached is not None and cached[0] == thr_sel_items:
            return cached[1]

    thresholds = _compute_selector_thresholds(thr_sel_items)

    with _SELECTOR_THRESHOLD_CACHE_LOCK:
        cached = _SELECTOR_THRESHOLD_CACHE.get(G)
        if cached is not None and cached[0] == thr_sel_items:
            return cached[1]
        _SELECTOR_THRESHOLD_CACHE[G] = (thr_sel_items, thresholds)
    return thresholds


def _norms_para_selector(G: "nx.Graph") -> dict:
    """Compute and cache norms for ΔNFR and acceleration.

    Parameters
    ----------
    G : nx.Graph
        Graph for which to compute maxima. Results are stored in ``G.graph``
        under ``"_sel_norms"``.

    Returns
    -------
    dict
        Mapping with normalisation maxima for ``dnfr`` and ``accel``.
    """
    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    return norms


def _calc_selector_score(
    Si: float, dnfr: float, accel: float, weights: dict[str, float]
) -> float:
    """Compute weighted selector score.

    Parameters
    ----------
    Si : float
        Normalised sense index.
    dnfr : float
        Normalised absolute ΔNFR value.
    accel : float
        Normalised acceleration (|d²EPI/dt²|).
    weights : dict[str, float]
        Normalised weights for ``"w_si"``, ``"w_dnfr"`` and ``"w_accel"``.

    Returns
    -------
    float
        Final weighted score.
    """
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
    """Apply hysteresis when values are near thresholds.

    Parameters
    ----------
    nd : dict[str, Any]
        Node attribute dictionary containing glyph history.
    Si : float
        Normalised sense index.
    dnfr : float
        Normalised absolute ΔNFR value.
    accel : float
        Normalised acceleration.
    thr : dict[str, float]
        Thresholds returned by :func:`_selector_thresholds`.
    margin : float
        Distance from thresholds below which the previous glyph is reused.

    Returns
    -------
    str or None
        Previous glyph if hysteresis applies, otherwise ``None``.
    """
    # Batch extraction reduces dictionary lookups inside loops.
    si_hi, si_lo, dnfr_hi, dnfr_lo, accel_hi, accel_lo = itemgetter(
        "si_hi", "si_lo", "dnfr_hi", "dnfr_lo", "accel_hi", "accel_lo"
    )(thr)

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
