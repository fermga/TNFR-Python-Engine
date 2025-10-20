"""Sense index helpers."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Iterable

from ..alias import get_attr, set_attr
from ..constants import get_aliases
from ..helpers.numeric import angle_diff, clamp01
from ..types import GraphLike
from ..utils import (
    edge_version_cache,
    get_numpy,
    normalize_weights,
    stable_json,
)
from .trig import neighbor_phase_mean_list

from .common import (
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


def _coerce_jobs(raw_jobs: Any | None) -> int | None:
    """Normalise ``n_jobs`` values coming from user configuration."""

    try:
        jobs = None if raw_jobs is None else int(raw_jobs)
    except (TypeError, ValueError):
        return None
    if jobs is not None and jobs <= 0:
        return None
    return jobs


def _compute_si_python_chunk(
    chunk: Iterable[tuple[Any, tuple[Any, ...], float, float, float]],
    *,
    cos_th: dict[Any, float],
    sin_th: dict[Any, float],
    alpha: float,
    beta: float,
    gamma: float,
    vfmax: float,
    dnfrmax: float,
) -> dict[Any, float]:
    """Compute Si values for a chunk of nodes using pure Python math."""

    results: dict[Any, float] = {}
    for n, neigh, theta, vf, dnfr in chunk:
        th_bar = neighbor_phase_mean_list(
            neigh, cos_th=cos_th, sin_th=sin_th, np=None, fallback=theta
        )
        disp_fase = abs(angle_diff(theta, th_bar)) / math.pi
        vf_norm = clamp01(abs(vf) / vfmax)
        dnfr_norm = clamp01(abs(dnfr) / dnfrmax)
        Si = alpha * vf_norm + beta * (1.0 - disp_fase) + gamma * (1.0 - dnfr_norm)
        results[n] = clamp01(Si)
    return results


def compute_Si(
    G: GraphLike,
    *,
    inplace: bool = True,
    n_jobs: int | None = None,
) -> dict[Any, float]:
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

    if n_jobs is None:
        n_jobs = _coerce_jobs(G.graph.get("SI_N_JOBS"))
    else:
        n_jobs = _coerce_jobs(n_jobs)

    supports_vector = (
        np is not None
        and hasattr(np, "ndarray")
        and all(hasattr(np, attr) for attr in ("fromiter", "abs", "clip", "remainder"))
    )

    nodes_data = list(G.nodes(data=True))
    if not nodes_data:
        return {}

    if supports_vector:
        node_ids: list[Any] = []
        theta_vals: list[float] = []
        mean_vals: list[float] = []
        vf_vals: list[float] = []
        dnfr_vals: list[float] = []
        for n, nd in nodes_data:
            theta = thetas.get(n, 0.0)
            neigh = neighbors[n]
            node_ids.append(n)
            theta_vals.append(theta)
            mean_vals.append(pm_fn(neigh, fallback=theta))
            vf_vals.append(get_attr(nd, ALIAS_VF, 0.0))
            dnfr_vals.append(get_attr(nd, ALIAS_DNFR, 0.0))

        count = len(node_ids)
        theta_arr = np.fromiter(theta_vals, dtype=float, count=count)
        mean_arr = np.fromiter(mean_vals, dtype=float, count=count)
        diff = np.remainder(theta_arr - mean_arr + math.pi, math.tau) - math.pi
        disp_fase_arr = np.abs(diff) / math.pi

        vf_arr = np.fromiter(vf_vals, dtype=float, count=count)
        dnfr_arr = np.fromiter(dnfr_vals, dtype=float, count=count)
        vf_norm = np.clip(np.abs(vf_arr) / vfmax, 0.0, 1.0)
        dnfr_norm = np.clip(np.abs(dnfr_arr) / dnfrmax, 0.0, 1.0)

        si_arr = np.clip(
            alpha * vf_norm + beta * (1.0 - disp_fase_arr)
            + gamma * (1.0 - dnfr_norm),
            0.0,
            1.0,
        )

        out = {node_ids[i]: float(si_arr[i]) for i in range(count)}
    else:
        out: dict[Any, float] = {}
        if n_jobs is not None and n_jobs > 1:
            node_payload: list[tuple[Any, tuple[Any, ...], float, float, float]] = []
            for n, nd in nodes_data:
                theta = thetas.get(n, 0.0)
                vf = float(get_attr(nd, ALIAS_VF, 0.0))
                dnfr = float(get_attr(nd, ALIAS_DNFR, 0.0))
                neigh = neighbors[n]
                node_payload.append((n, tuple(neigh), theta, vf, dnfr))

            if node_payload:
                chunk_size = math.ceil(len(node_payload) / n_jobs)
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(
                            _compute_si_python_chunk,
                            node_payload[idx:idx + chunk_size],
                            cos_th=cos_th,
                            sin_th=sin_th,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            vfmax=vfmax,
                            dnfrmax=dnfrmax,
                        )
                        for idx in range(0, len(node_payload), chunk_size)
                    ]
                    for future in futures:
                        out.update(future.result())
        else:
            for n, nd in nodes_data:
                theta = thetas.get(n, 0.0)
                neigh = neighbors[n]
                th_bar = pm_fn(neigh, fallback=theta)
                disp_fase = abs(angle_diff(theta, th_bar)) / math.pi
                out[n] = compute_Si_node(
                    n,
                    nd,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    vfmax=vfmax,
                    dnfrmax=dnfrmax,
                    disp_fase=disp_fase,
                    inplace=False,
                )

    if inplace:
        for n, value in out.items():
            set_attr(G.nodes[n], ALIAS_SI, value)
    return out
