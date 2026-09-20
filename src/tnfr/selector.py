"""Utilities to select structural operator symbols based on structural metrics.

This module resolves thresholds, computes selection scores and applies
hysteresis when assigning structural operator symbols (glyphs) to nodes.

Each structural operator (Emission, Reception, Coherence, etc.) is represented
by a glyph symbol (AL, EN, IL, etc.) that this module selects based on the
node's current structural state.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

from ._exact_time import finite_represented_real
from .config.selector_thresholds import resolve_selector_thresholds
from .constants import get_param
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
    "_selector_margin",
)


def _selector_thresholds(G: "nx.Graph") -> SelectorThresholds:
    """Return the shared validated policy for Si, pressure and acceleration."""
    return resolve_selector_thresholds(G)


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


def _validated_margin(value: Any) -> float | None:
    if value is None:
        return None
    try:
        margin = finite_represented_real(value, "GLYPH_SELECTOR_MARGIN")[0]
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    if margin < 0.0:
        raise ValueError("GLYPH_SELECTOR_MARGIN must be nonnegative")
    return margin


def _selector_margin(G: "TNFRGraph") -> float | None:
    """Read the configured nonnegative hysteresis distance without coercion."""
    return _validated_margin(get_param(G, "GLYPH_SELECTOR_MARGIN"))


def _calc_selector_score(
    Si: float, dnfr: float, accel: float, weights: SelectorWeights
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
        glyph is reused. None or zero disables hysteresis entirely, letting
        selectors bypass the reuse logic.

    Returns
    -------
    str or None
        Previous glyph if hysteresis applies, otherwise ``None``.
    """
    # Batch extraction reduces dictionary lookups inside loops.
    margin = _validated_margin(margin)
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
