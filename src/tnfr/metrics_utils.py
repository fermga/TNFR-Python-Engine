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
from .helpers import (
    clamp01,
    angle_diff,
    edge_version_cache,
    neighbor_phase_mean_list,
)
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
        dnfr_vals = []
        depi_vals = []
        for _, nd in G.nodes(data=True):
            # single-pass accumulation of dnfr and depi values
            dnfr_vals.append(abs(get_attr(nd, ALIAS_DNFR, 0.0)))
            depi_vals.append(abs(get_attr(nd, ALIAS_dEPI, 0.0)))
        dnfr_mean = math.fsum(dnfr_vals) / count
        depi_mean = math.fsum(depi_vals) / count
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
    """Return cached cosines and sines of ``θ`` per node."""
    return edge_version_cache(G, "_trig", lambda: _build_trig_cache(G))


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
    np=None,
) -> float:
    """Compute ``Si`` for a single node."""
    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

    th_i = thetas[n]
    neigh = neighbors[n]
    th_bar = neighbor_phase_mean_list(neigh, cos_th, sin_th, np, th_i)
    disp_fase = abs(angle_diff(th_i, th_bar)) / math.pi

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
