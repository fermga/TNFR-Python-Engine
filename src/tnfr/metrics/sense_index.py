"""Sense index helpers."""

from __future__ import annotations

import math
from functools import partial
from typing import Any

from ..alias import get_attr, set_attr
from ..collections_utils import normalize_weights
from ..constants import get_aliases
from ..helpers import edge_version_cache, stable_json
from ..helpers.numeric import angle_diff, clamp01
from .trig import neighbor_phase_mean_list
from ..import_utils import get_numpy

from .common import (
    GraphLike,
    ensure_neighbors_map,
    merge_graph_weights,
    _get_vf_dnfr_max,
)
from .trig_cache import get_trig_cache

ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_SI = get_aliases("SI")
ALIAS_THETA = get_aliases("THETA")

__all__ = ("get_Si_weights", "compute_Si_node", "compute_Si")


def _cache_weights(G: GraphLike) -> tuple[float, float, float]:
    """Normalise and cache Si weights, delegating persistence."""

    w = merge_graph_weights(G, "SI_WEIGHTS")
    cfg_key = stable_json(w)

    def builder() -> tuple[float, float, float]:
        weights = normalize_weights(w, ("alpha", "beta", "gamma"), default=0.0)
        alpha = weights["alpha"]
        beta = weights["beta"]
        gamma = weights["gamma"]
        G.graph["_Si_weights"] = weights
        G.graph["_Si_weights_key"] = cfg_key
        G.graph["_Si_sensitivity"] = {
            "dSi_dvf_norm": alpha,
            "dSi_ddisp_fase": -beta,
            "dSi_ddnfr_norm": -gamma,
        }
        return alpha, beta, gamma

    return edge_version_cache(G, ("_Si_weights", cfg_key), builder)


def get_Si_weights(G: GraphLike) -> tuple[float, float, float]:
    """Obtain and normalise weights for the sense index."""

    return _cache_weights(G)


def compute_Si_node(
    n: Any,
    nd: dict[str, Any],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
    disp_fase: float,
    inplace: bool,
) -> float:
    """Compute ``Si`` for a single node."""

    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dnfr_norm = clamp01(abs(dnfr) / dnfrmax)

    Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
    Si = clamp01(Si)
    if inplace:
        set_attr(nd, ALIAS_SI, Si)
    return Si


def compute_Si(G: GraphLike, *, inplace: bool = True) -> dict[Any, float]:
    """Compute ``Si`` per node and optionally store it on the graph."""

    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)
    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    np = get_numpy()
    trig = get_trig_cache(G, np=np)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta

    pm_fn = partial(
        neighbor_phase_mean_list, cos_th=cos_th, sin_th=sin_th, np=np
    )

    out: dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        neigh = neighbors[n]
        th_bar = pm_fn(neigh, fallback=thetas[n])
        disp_fase = abs(angle_diff(thetas[n], th_bar)) / math.pi
        out[n] = compute_Si_node(
            n,
            nd,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            vfmax=vfmax,
            dnfrmax=dnfrmax,
            disp_fase=disp_fase,
            inplace=inplace,
        )
    return out
