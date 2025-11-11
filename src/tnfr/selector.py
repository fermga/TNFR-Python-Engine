"""Utilities to select structural operator symbols based on structural metrics.

This module normalises thresholds, computes selection scores and applies
hysteresis when assigning structural operator symbols (glyphs) to nodes.

Each structural operator (Emission, Reception, Coherence, etc.) is represented
by a glyph symbol (AL, EN, IL, etc.) that this module selects based on the
node's current structural state.
"""

from __future__ import annotations

import threading
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Mapping, cast
from weakref import WeakKeyDictionary

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from .constants import DEFAULTS
from .config.defaults_core import SELECTOR_THRESHOLD_DEFAULTS
from .utils import clamp01
from .metrics.common import compute_dnfr_accel_max
from .types import SelectorNorms, SelectorThresholds, SelectorWeights
from .utils import is_non_string_sequence

if TYPE_CHECKING:  # pragma: no cover
    from .types import TNFRGraph

HYSTERESIS_GLYPHS: set[str] = {"IL", "OZ", "ZHIR", "THOL", "NAV", "RA"}

__all__ = (
    "_selector_thresholds",
    "_selector_norms",
    "_calc_selector_score",
    "_apply_selector_hysteresis",
    "_selector_parallel_jobs",
)

_SelectorThresholdItems = tuple[tuple[str, float], ...]
_SelectorThresholdCacheEntry = tuple[
    _SelectorThresholdItems,
    SelectorThresholds,
]
_SELECTOR_THRESHOLD_CACHE: WeakKeyDictionary[
    "nx.Graph",
    _SelectorThresholdCacheEntry,
] = WeakKeyDictionary()
_SELECTOR_THRESHOLD_CACHE_LOCK = threading.Lock()


def _sorted_items(mapping: Mapping[str, float]) -> _SelectorThresholdItems:
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
    thr_sel_items: _SelectorThresholdItems,
) -> SelectorThresholds:
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
    return cast(SelectorThresholds, out)


def _selector_thresholds(G: "nx.Graph") -> SelectorThresholds:
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


def _selector_norms(G: "nx.Graph") -> SelectorNorms:
    """Compute and cache selector norms for ΔNFR and acceleration.

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


def _calc_selector_score(Si: float, dnfr: float, accel: float, weights: SelectorWeights) -> float:
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
        weights["w_si"] * Si + weights["w_dnfr"] * (1.0 - dnfr) + weights["w_accel"] * (1.0 - accel)
    )


def _apply_selector_hysteresis(
    nd: dict[str, Any],
    Si: float,
    dnfr: float,
    accel: float,
    thr: dict[str, float],
    margin: float | None,
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
    margin : float or None
        When positive, distance from thresholds below which the previous
        glyph is reused. Falsy margins disable hysteresis entirely, letting
        selectors bypass the reuse logic.

    Returns
    -------
    str or None
        Previous glyph if hysteresis applies, otherwise ``None``.
    """
    # Batch extraction reduces dictionary lookups inside loops.
    if not margin:
        return None

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


def _selector_parallel_jobs(G: "TNFRGraph") -> int | None:
    """Return worker count for selector helpers when parallelism is enabled.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing selector configuration.

    Returns
    -------
    int | None
        Number of parallel jobs to use, or None if parallelism is disabled
        or invalid configuration is provided.
    """
    raw_jobs = G.graph.get("GLYPH_SELECTOR_N_JOBS")
    try:
        n_jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        return None
    if n_jobs is None or n_jobs <= 1:
        return None
    return n_jobs
