"""Miscellaneous metrics helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Sequence
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
from .alias import get_attr, set_attr, _recompute_abs_max
from .collections_utils import normalize_weights
from .helpers import clamp01, angle_diff, edge_version_cache


__all__ = [
    "TrigCache",
    "compute_dnfr_accel_max",
    "compute_coherence",
    "get_Si_weights",
    "precompute_trigonometry",
    "compute_Si_node",
    "compute_Si",
]


@dataclass
class TrigCache:
    cos: Dict[Any, float]
    sin: Dict[Any, float]
    theta: Dict[Any, float]


def compute_dnfr_accel_max(G) -> dict:
    """Compute absolute maxima of |ΔNFR| and |d²EPI/dt²|."""
    dnfr_max = max(
        (abs(get_attr(nd, ALIAS_DNFR, 0.0)) for _, nd in G.nodes(data=True)),
        default=0.0,
    )
    accel_max = max(
        (abs(get_attr(nd, ALIAS_D2EPI, 0.0)) for _, nd in G.nodes(data=True)),
        default=0.0,
    )
    return {"dnfr_max": float(dnfr_max), "accel_max": float(accel_max)}


def compute_coherence(G) -> float:
    """Compute global coherence C(t) from ΔNFR and dEPI."""
    nodes = list(G.nodes(data=True))
    count = len(nodes)
    if count:
        dnfr_mean = math.fsum(
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) for _, nd in nodes
        ) / count
        depi_mean = math.fsum(
            abs(get_attr(nd, ALIAS_dEPI, 0.0)) for _, nd in nodes
        ) / count
    else:
        dnfr_mean = depi_mean = 0.0
    return 1.0 / (1.0 + dnfr_mean + depi_mean)


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


def precompute_trigonometry(
    G: Any,
) -> TrigCache:
    """Precompute cosines and sines of ``θ`` per node."""

    def builder() -> TrigCache:
        cos_th: Dict[Any, float] = {}
        sin_th: Dict[Any, float] = {}
        thetas: Dict[Any, float] = {}
        for n, nd in G.nodes(data=True):
            th = get_attr(nd, ALIAS_THETA, 0.0)
            thetas[n] = th
            cos_th[n] = math.cos(th)
            sin_th[n] = math.sin(th)
        return TrigCache(cos=cos_th, sin=sin_th, theta=thetas)

    return edge_version_cache(G, "_trig", builder)


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
    graph = G.graph
    edge_version = int(graph.get("_edge_version", 0))

    neighbors = graph.get("_neighbors")
    if graph.get("_neighbors_version") != edge_version or neighbors is None:
        neighbors = {n: list(G.neighbors(n)) for n in G}
        graph["_neighbors"] = neighbors
        graph["_neighbors_version"] = edge_version

    alpha, beta, gamma = get_Si_weights(G)

    vfmax = G.graph.get("_vfmax")
    if vfmax is None:
        vfmax, vf_node = _recompute_abs_max(G, ALIAS_VF)
        G.graph.setdefault("_vfmax", vfmax)
        G.graph.setdefault("_vfmax_node", vf_node)
    dnfrmax = G.graph.get("_dnfrmax")
    if dnfrmax is None:
        dnfrmax, dnfr_node = _recompute_abs_max(G, ALIAS_DNFR)
        G.graph.setdefault("_dnfrmax", dnfrmax)
        G.graph.setdefault("_dnfrmax_node", dnfr_node)
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

