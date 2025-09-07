"""Miscellaneous metrics helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Iterable
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
from .helpers import clamp01, angle_diff, edge_version_cache
from .import_utils import get_numpy


__all__ = [
    "TrigCache",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "ensure_neighbors_map",
    "get_Si_weights",
    "get_trig_cache",
    "precompute_trigonometry",
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


def compute_coherence(G) -> float:
    """Compute global coherence C(t) from ΔNFR and dEPI."""
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
    return 1.0 / (1.0 + dnfr_mean + depi_mean)


def ensure_neighbors_map(G) -> Dict[Any, Sequence[Any]]:
    """Return cached neighbors list keyed by node."""
    graph = G.graph
    edge_version = int(graph.get("_edge_version", 0))
    neighbors = graph.get("_neighbors")
    if graph.get("_neighbors_version") != edge_version or neighbors is None:
        neighbors = {n: list(G.neighbors(n)) for n in G}
        graph["_neighbors"] = neighbors
        graph["_neighbors_version"] = edge_version
    return neighbors


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
    """Return cached cosines and sines of ``θ`` per node."""
    return edge_version_cache(G, "_trig", lambda: _build_trig_cache(G))


def precompute_trigonometry(G: Any) -> TrigCache:
    """Precompute cosines and sines of ``θ`` per node.

    Alias for :func:`get_trig_cache` for backward compatibility.
    """
    return get_trig_cache(G)


def compute_Si_node(
    n: Any,
    nd: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
    cos_th: Dict[Any, float],
    sin_th: Dict[Any, float],
    thetas: Dict[Any, float],
    neighbors: Dict[Any, Sequence[Any]],
    inplace: bool,
) -> float:
    """Compute ``Si`` for a single node."""
    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    th_i = thetas[n]
    neigh = neighbors[n]
    deg = len(neigh)
    if deg:
        np = get_numpy()
        if np is not None:
            cos_vals = np.fromiter((cos_th[v] for v in neigh), float, count=deg)
            sin_vals = np.fromiter((sin_th[v] for v in neigh), float, count=deg)
            mean_cos = float(cos_vals.mean())
            mean_sin = float(sin_vals.mean())
            th_bar = float(np.arctan2(mean_sin, mean_cos))
        else:
            sum_cos = math.fsum(cos_th[v] for v in neigh)
            sum_sin = math.fsum(sin_th[v] for v in neigh)
            mean_cos = sum_cos / deg
            mean_sin = sum_sin / deg
            th_bar = math.atan2(mean_sin, mean_cos)
    else:
        th_bar = th_i
    disp_fase = abs(angle_diff(th_i, th_bar)) / math.pi

    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

    Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
    Si = clamp01(Si)
    if inplace:
        set_attr(nd, ALIAS_SI, Si)
    return Si


def compute_Si(G, *, inplace: bool = True) -> Dict[Any, float]:
    """Compute ``Si`` per node and write it to ``G.nodes[n]['Si']``."""
    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)

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

    trig = precompute_trigonometry(G)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta

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
            cos_th=cos_th,
            sin_th=sin_th,
            thetas=thetas,
            neighbors=neighbors,
            inplace=inplace,
        )
    return out


def min_max_range(
    values: Iterable[float], *, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    try:
        vmin = min(values)
        vmax = max(values)
    except ValueError:
        return default
    return vmin, vmax
