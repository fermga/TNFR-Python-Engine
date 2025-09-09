"""Miscellaneous metrics helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence, Iterable, Mapping
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
    neighbor_phase_mean_list,
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


@dataclass(slots=True)
class TrigCache:
    cos: dict[Any, float]
    sin: dict[Any, float]
    theta: dict[Any, float]


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
        dnfr_mean = math.fsum(
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) for _, nd in G.nodes(data=True)
        ) / count
        depi_mean = math.fsum(
            abs(get_attr(nd, ALIAS_dEPI, 0.0)) for _, nd in G.nodes(data=True)
        ) / count
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


def _build_trig_cache(G: Any, np: Any | None = None) -> TrigCache:
    """Construct trigonometric cache for ``G``."""
    if np is None:
        np = get_numpy()

    cos_th: dict[Any, float] = {}
    sin_th: dict[Any, float] = {}
    thetas: dict[Any, float] = {}

    if (
        np is not None
        and all(hasattr(np, attr) for attr in ("asarray", "cos", "sin"))
    ):
        nodes = list(G.nodes())
        theta_arr = np.asarray(
            [get_attr(G.nodes[n], ALIAS_THETA, 0.0) for n in nodes], dtype=float
        )
        cos_arr = np.cos(theta_arr)
        sin_arr = np.sin(theta_arr)

        thetas = dict(zip(nodes, theta_arr.tolist()))
        cos_th = dict(zip(nodes, cos_arr.tolist()))
        sin_th = dict(zip(nodes, sin_arr.tolist()))
    else:
        for n, nd in G.nodes(data=True):
            th = get_attr(nd, ALIAS_THETA, 0.0)
            thetas[n] = th
            cos_th[n] = math.cos(th)
            sin_th[n] = math.sin(th)

    return TrigCache(cos=cos_th, sin=sin_th, theta=thetas)


def get_trig_cache(G: Any, *, np: Any | None = None) -> TrigCache:
    """Return cached cosines and sines of ``θ`` per node.

    The cache is invalidated not only when the edge set changes but also when
    node phases ``θ`` are updated. Calling :func:`tnfr.alias.set_theta`
    increments a per-graph ``_trig_version`` counter and purges any previously
    stored ``_cos_th``, ``_sin_th`` or ``_thetas`` entries in ``G.graph``.
    The counter forms part of the cache key, forcing a rebuild when it
    advances.
    """
    version = G.graph.setdefault("_trig_version", 0)
    key = ("_trig", version)
    return edge_version_cache(G, key, lambda: _build_trig_cache(G, np=np))


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
    """Compute ``Si`` for a single node.

    Parameters
    ----------
    disp_fase:
        Normalised phase displacement ``|θᵢ - \bar{θ}|/π``.
    """
    vf = get_attr(nd, ALIAS_VF, 0.0)
    vf_norm = clamp01(abs(vf) / vfmax)

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
        maxes = multi_recompute_abs_max(
            G, {"_vfmax": ALIAS_VF, "_dnfrmax": ALIAS_DNFR}
        )
        if vfmax is None:
            vfmax = maxes["_vfmax"]
            G.graph.setdefault("_vfmax", vfmax)
        if dnfrmax is None:
            dnfrmax = maxes["_dnfrmax"]
            G.graph.setdefault("_dnfrmax", dnfrmax)
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax
    return vfmax, dnfrmax


def compute_Si(G, *, inplace: bool = True) -> dict[Any, float]:
    """Compute ``Si`` per node and write it to ``G.nodes[n]['Si']``."""
    neighbors = ensure_neighbors_map(G)
    alpha, beta, gamma = get_Si_weights(G)

    vfmax, dnfrmax = _get_vf_dnfr_max(G)

    np_mod = get_numpy()
    trig = get_trig_cache(G, np=np_mod)
    cos_th, sin_th, thetas = trig.cos, trig.sin, trig.theta

    def phase_mean_fn(neigh, *, fallback):
        if np_mod is None:
            return neighbor_phase_mean_list(
                neigh, cos_th, sin_th, fallback=fallback
            )
        return neighbor_phase_mean_list(
            neigh, cos_th, sin_th, np=np_mod, fallback=fallback
        )

    out: dict[Any, float] = {}
    for n, nd in G.nodes(data=True):
        neigh = neighbors[n]
        th_bar = phase_mean_fn(neigh, fallback=thetas[n])
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
