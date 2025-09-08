"""Miscellaneous metrics helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Iterable, Mapping
from types import MappingProxyType
import math

from .constants import (
    DEFAULTS,
    ALIAS_DNFR,
    ALIAS_D2EPI,
    ALIAS_dEPI,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_SI,
)
from .alias import get_attr, set_attr, multi_recompute_abs_max
from .collections_utils import normalize_weights
from .helpers.numeric import (
    clamp01,
    angle_diff,
)
from .helpers.cache import edge_version_cache
from .import_utils import get_numpy


__all__ = [
    "TrigCache",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "ensure_neighbors_map",
    "get_Si_weights",
    "get_trig_cache",
    "compute_Si_node",
    "compute_Si",
    "min_max_range",
]


@dataclass
class TrigCache:
    cos: Dict[Any, float]
    sin: Dict[Any, float]
    theta: Dict[Any, float]


def compute_dnfr_accel_max(G) -> dict:
    """Compute absolute maxima of |ΔNFR| and |d²EPI/dt²|."""
    maxes = multi_recompute_abs_max(
        G, {"dnfr_max": ALIAS_DNFR, "accel_max": ALIAS_D2EPI}
    )
    return maxes


def compute_coherence(
    G, *, return_means: bool = False
) -> float | tuple[float, float, float]:
    """Compute global coherence ``C`` from ``ΔNFR`` and ``dEPI``.

    Parameters
    ----------
    G:
        Graph containing ``dnfr`` and ``dEPI`` attributes per node.
    return_means:
        If ``True``, also return the means of ``|ΔNFR|`` and ``|dEPI|``.

    Returns
    -------
    float or tuple
        ``C`` when ``return_means`` is ``False`` (default). When ``True``, a
        tuple ``(C, dnfr_mean, depi_mean)`` is returned.
    """
    count = G.number_of_nodes()
    if count:
        dnfr_sum = math.fsum(
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) for _, nd in G.nodes(data=True)
        )
        depi_sum = math.fsum(
            abs(get_attr(nd, ALIAS_dEPI, 0.0)) for _, nd in G.nodes(data=True)
        )
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count
    else:
        dnfr_mean = depi_mean = 0.0
    coherence = 1.0 / (1.0 + dnfr_mean + depi_mean)
    return (coherence, dnfr_mean, depi_mean) if return_means else coherence


def ensure_neighbors_map(G) -> Mapping[Any, Sequence[Any]]:
    """Return cached neighbors list keyed by node as a read-only mapping."""

    def builder() -> Mapping[Any, Sequence[Any]]:
        return MappingProxyType({n: list(G.neighbors(n)) for n in G})

    return edge_version_cache(G, "_neighbors", builder)


def get_Si_weights(G: Any) -> tuple[float, float, float]:
    """Obtain and normalise weights for the sense index."""
    w = {**DEFAULTS["SI_WEIGHTS"], **G.graph.get("SI_WEIGHTS", {})}
    weights = normalize_weights(w, ("alpha", "beta", "gamma"), default=0.0)
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    G.graph["_Si_weights"] = weights
    G.graph["_Si_sensitivity"] = {
        "dSi_dvf_norm": alpha,
        "dSi_ddisp_fase": -beta,
        "dSi_ddnfr_norm": -gamma,
    }
    return alpha, beta, gamma


def _build_trig_cache(G: Any) -> TrigCache:
    """Construct trigonometric cache for ``G``."""
    cos_th: Dict[Any, float] = {}
    sin_th: Dict[Any, float] = {}
    thetas: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thetas[n] = th
        cos_th[n] = math.cos(th)
        sin_th[n] = math.sin(th)
    return TrigCache(cos=cos_th, sin=sin_th, theta=thetas)


def get_trig_cache(G: Any) -> TrigCache:
    """Return cached cosines and sines of ``θ`` per node.

    The cache is invalidated not only when the edge set changes but also when
    node phases ``θ`` are updated. A per-graph ``_trig_version`` counter is
    incremented whenever phases change and forms part of the cache key, forcing
    a rebuild when the counter advances.
    """
    version = G.graph.setdefault("_trig_version", 0)
    key = ("_trig", version)
    return edge_version_cache(G, key, lambda: _build_trig_cache(G))


def compute_Si_node(
    n: Any,
    nd: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
    cos_vals: Sequence[float],
    sin_vals: Sequence[float],
    theta_i: float,
    inplace: bool,
    np=None,
) -> float:
    """Compute ``Si`` for a single node.

    Parameters
    ----------
    cos_vals, sin_vals:
        Pre-computed trigonometric values of the neighbouring nodes.
    theta_i:
        Phase of the current node.
    """
    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    deg = len(cos_vals)
    if deg == 0:
        th_bar = theta_i
    else:
        if np is not None and hasattr(cos_vals, "mean"):
            mean_cos = float(cos_vals.mean())
            mean_sin = float(sin_vals.mean())
            th_bar = float(np.arctan2(mean_sin, mean_cos))
        else:
            mean_cos = sum(cos_vals) / deg
            mean_sin = sum(sin_vals) / deg
            th_bar = math.atan2(mean_sin, mean_cos)
    disp_fase = abs(angle_diff(theta_i, th_bar)) / math.pi

    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

    Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
    Si = clamp01(Si)
    if inplace:
        set_attr(nd, ALIAS_SI, Si)
    return Si


def _get_vf_dnfr_max(G) -> tuple[float, float]:
    """Ensure and return absolute maxima for ``vf`` and ``ΔNFR``."""
    vfmax = G.graph.get("_vfmax")
    dnfrmax = G.graph.get("_dnfrmax")
    if vfmax is None or dnfrmax is None:
        maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF, "_dnfrmax": ALIAS_DNFR})
        if vfmax is None:
            vfmax = maxes["_vfmax"]
            G.graph.setdefault("_vfmax", vfmax)
        if dnfrmax is None:
            dnfrmax = maxes["_dnfrmax"]
            G.graph.setdefault("_dnfrmax", dnfrmax)
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax
    return vfmax, dnfrmax


def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Compute ``Si`` per node and write it to ``G.nodes[n]['Si']``."""
    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)

    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    trig = get_trig_cache(G)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta
    np = get_numpy()

    cos_cache: Dict[Any, Sequence[float]] = {}
    sin_cache: Dict[Any, Sequence[float]] = {}
    for n, neigh in neighbors.items():
        deg = len(neigh)
        if np is not None and deg:
            cos_cache[n] = np.fromiter((cos_th[v] for v in neigh), dtype=float, count=deg)
            sin_cache[n] = np.fromiter((sin_th[v] for v in neigh), dtype=float, count=deg)
        else:
            cos_cache[n] = [cos_th[v] for v in neigh]
            sin_cache[n] = [sin_th[v] for v in neigh]

    out: Dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        out[n] = compute_Si_node(
            n,
            nd,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            vfmax=vfmax,
            dnfrmax=dnfrmax,
            cos_vals=cos_cache[n],
            sin_vals=sin_cache[n],
            theta_i=thetas[n],
            inplace=inplace,
            np=np,
        )
    return out


def min_max_range(
    values: Iterable[float], *, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return default
    min_val = max_val = first
    for val in it:
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val
    return min_val, max_val
