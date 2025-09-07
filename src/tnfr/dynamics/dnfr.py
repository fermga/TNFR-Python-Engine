from __future__ import annotations

import math
from typing import Dict, Any, Callable

from ..collections_utils import normalize_weights
from ..constants import DEFAULTS, ALIAS_THETA, ALIAS_EPI, ALIAS_VF
from ..helpers import (
    angle_diff,
    neighbor_mean,
    neighbor_phase_mean,
    cached_nodes_and_A,
    _phase_mean_from_iter,
)
from ..alias import (
    get_attr,
    set_dnfr,
)
from ..metrics_utils import get_trig_cache
from ..import_utils import get_numpy

__all__ = [
    "default_compute_delta_nfr",
    "set_delta_nfr_hook",
    "dnfr_phase_only",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "apply_dnfr_field",
]
def _write_dnfr_metadata(
    G, *, weights: dict, hook_name: str, note: str | None = None
) -> None:
    """Write a ``_DNFR_META`` block in ``G.graph`` with the mix and hook name.

    ``weights`` may include arbitrary components (phase/epi/vf/topo/etc.).
    """
    weights_norm = normalize_weights(weights, weights.keys())
    meta = {
        "hook": hook_name,
        "weights_raw": dict(weights),
        "weights_norm": weights_norm,
        "components": [k for k, v in weights_norm.items() if v != 0.0],
        "doc": "ΔNFR = Σ w_i·g_i",
    }
    if note:
        meta["note"] = str(note)
    G.graph["_DNFR_META"] = meta
    G.graph["_dnfr_hook_name"] = hook_name  # string friendly


def _configure_dnfr_weights(G) -> dict:
    """Normalise and store ΔNFR weights in ``G.graph['_dnfr_weights']``.

    Uses ``G.graph['DNFR_WEIGHTS']`` or default values. The result is a
    dictionary of normalised components reused at each simulation step
    without recomputing the mix.
    """
    w = {**DEFAULTS["DNFR_WEIGHTS"], **G.graph.get("DNFR_WEIGHTS", {})}
    weights = normalize_weights(w, ("phase", "epi", "vf", "topo"), default=0.0)
    G.graph["_dnfr_weights"] = weights
    return weights


def _init_dnfr_cache(G, nodes, prev_cache, checksum, dirty):
    """Initialise or reuse cached ΔNFR arrays."""
    if prev_cache and prev_cache.get("checksum") == checksum and not dirty:
        idx = prev_cache["idx"]
        theta = prev_cache["theta"]
        epi = prev_cache["epi"]
        vf = prev_cache["vf"]
        cos_theta = prev_cache["cos_theta"]
        sin_theta = prev_cache["sin_theta"]
        return prev_cache, idx, theta, epi, vf, cos_theta, sin_theta, False

    idx = {n: i for i, n in enumerate(nodes)}
    theta = [0.0] * len(nodes)
    epi = [0.0] * len(nodes)
    vf = [0.0] * len(nodes)
    cos_theta = [1.0] * len(nodes)
    sin_theta = [0.0] * len(nodes)
    cache = {
        "checksum": checksum,
        "idx": idx,
        "theta": theta,
        "epi": epi,
        "vf": vf,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "degs": prev_cache.get("degs") if prev_cache else None,
    }
    G.graph["_dnfr_prep_cache"] = cache
    return cache, idx, theta, epi, vf, cos_theta, sin_theta, True


def _refresh_dnfr_vectors(G, nodes, theta, epi, vf, cos_theta, sin_theta):
    """Update cached angle and state vectors for ΔNFR."""
    trig = get_trig_cache(G)
    for i, n in enumerate(nodes):
        nd = G.nodes[n]
        th = get_attr(nd, ALIAS_THETA, 0.0)
        theta[i] = th
        epi[i] = get_attr(nd, ALIAS_EPI, 0.0)
        vf[i] = get_attr(nd, ALIAS_VF, 0.0)
        cos_theta[i] = trig.cos.get(n, math.cos(th))
        sin_theta[i] = trig.sin.get(n, math.sin(th))


def _prepare_dnfr_data(G, *, cache_size: int | None = 128) -> dict:
    """Precompute common data for ΔNFR strategies."""
    weights = G.graph.get("_dnfr_weights")
    if weights is None:
        weights = _configure_dnfr_weights(G)

    use_numpy = get_numpy() is not None and G.graph.get("vectorized_dnfr")

    nodes, A = cached_nodes_and_A(G, cache_size=cache_size)
    cache = G.graph.get("_dnfr_prep_cache")
    checksum = G.graph.get("_dnfr_nodes_checksum")
    dirty = bool(G.graph.pop("_dnfr_prep_dirty", False))
    cache, idx, theta, epi, vf, cos_theta, sin_theta, refreshed = _init_dnfr_cache(
        G, nodes, cache, checksum, dirty
    )
    if refreshed:
        _refresh_dnfr_vectors(G, nodes, theta, epi, vf, cos_theta, sin_theta)

    w_phase = float(weights.get("phase", 0.0))
    w_epi = float(weights.get("epi", 0.0))
    w_vf = float(weights.get("vf", 0.0))
    w_topo = float(weights.get("topo", 0.0))
    degs = cache.get("degs") if cache else None
    if w_topo != 0 and (dirty or degs is None):
        degs = dict(G.degree())
        cache["degs"] = degs
    elif w_topo == 0:
        degs = None
        if cache is not None:
            cache["degs"] = None

    if not use_numpy:
        A = None

    G.graph["_dnfr_prep_dirty"] = False

    return {
        "weights": weights,
        "nodes": nodes,
        "idx": idx,
        "theta": theta,
        "epi": epi,
        "vf": vf,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "w_phase": w_phase,
        "w_epi": w_epi,
        "w_vf": w_vf,
        "w_topo": w_topo,
        "degs": degs,
        "A": A,
        "cache_size": cache_size,
    }


def _apply_dnfr_gradients(
    G,
    data,
    th_bar,
    epi_bar,
    vf_bar,
    deg_bar=None,
    degs=None,
):
    """Combine precomputed gradients and write ΔNFR to each node."""
    nodes = data["nodes"]
    theta = data["theta"]
    epi = data["epi"]
    vf = data["vf"]
    w_phase = data["w_phase"]
    w_epi = data["w_epi"]
    w_vf = data["w_vf"]
    w_topo = data["w_topo"]
    if degs is None:
        degs = data.get("degs")

    for i, n in enumerate(nodes):
        g_phase = -angle_diff(theta[i], th_bar[i]) / math.pi
        g_epi = epi_bar[i] - epi[i]
        g_vf = vf_bar[i] - vf[i]
        if w_topo != 0.0 and deg_bar is not None and degs is not None:
            if isinstance(degs, dict):
                deg_i = float(degs.get(n, 0))
            else:
                deg_i = float(degs[i])
            g_topo = deg_bar[i] - deg_i
        else:
            g_topo = 0.0
        dnfr = (
            w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + w_topo * g_topo
        )
        set_dnfr(G, n, float(dnfr))


def _compute_neighbor_means(
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum=None,
    degs=None,
):
    """Return neighbour mean arrays for ΔNFR."""
    w_topo = data["w_topo"]
    theta = data["theta"]
    epi = data["epi"]
    vf = data["vf"]
    np = get_numpy()

    if np is not None and isinstance(count, np.ndarray):
        mask = count > 0
        th_bar = np.array(theta, dtype=float)
        epi_bar = np.array(epi, dtype=float)
        vf_bar = np.array(vf, dtype=float)
        deg_bar = None
        if w_topo != 0.0 and degs is not None:
            deg_bar = np.array(degs, dtype=float)
        if np.any(mask):
            th_bar[mask] = np.arctan2(
                y[mask] / count[mask], x[mask] / count[mask]
            )
            epi_bar[mask] = epi_sum[mask] / count[mask]
            vf_bar[mask] = vf_sum[mask] / count[mask]
            if w_topo != 0.0 and deg_bar is not None and deg_sum is not None:
                deg_bar[mask] = deg_sum[mask] / count[mask]
        return th_bar, epi_bar, vf_bar, deg_bar

    n = len(theta)
    th_bar = list(theta)
    epi_bar = list(epi)
    vf_bar = list(vf)
    deg_bar = list(degs) if w_topo != 0.0 and degs is not None else None
    cos_th = data["cos_theta"]
    sin_th = data["sin_theta"]
    idx = data["idx"]
    nodes = data["nodes"]
    for i in range(n):
        c = count[i]
        if c:
            node = nodes[i]
            th_bar[i] = _phase_mean_from_iter(
                ((cos_th[idx[v]], sin_th[idx[v]]) for v in G.neighbors(node)),
                theta[i],
            )
            epi_bar[i] = epi_sum[i] / c
            vf_bar[i] = vf_sum[i] / c
            if w_topo != 0.0 and deg_bar is not None and deg_sum is not None:
                deg_bar[i] = deg_sum[i] / c
    return th_bar, epi_bar, vf_bar, deg_bar


def _compute_dnfr_common(
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum=None,
    degs=None,
):
    """Compute neighbour means and apply ΔNFR gradients."""

    th_bar, epi_bar, vf_bar, deg_bar = _compute_neighbor_means(
        G,
        data,
        x=x,
        y=y,
        epi_sum=epi_sum,
        vf_sum=vf_sum,
        count=count,
        deg_sum=deg_sum,
        degs=degs,
    )
    _apply_dnfr_gradients(G, data, th_bar, epi_bar, vf_bar, deg_bar, degs)


def _build_neighbor_sums_common(G, data, *, use_numpy: bool):
    nodes = data["nodes"]
    w_topo = data["w_topo"]
    if use_numpy:
        np = get_numpy(warn=True)
        if np is None:  # pragma: no cover - runtime check
            raise RuntimeError("numpy no disponible para la versión vectorizada")
        if not nodes:
            return None
        A = data.get("A")
        if A is None:
            _, A = cached_nodes_and_A(G, cache_size=data.get("cache_size"))
            data["A"] = A
        epi = np.array(data["epi"], dtype=float)
        vf = np.array(data["vf"], dtype=float)
        cos_th = np.array(data["cos_theta"], dtype=float)
        sin_th = np.array(data["sin_theta"], dtype=float)
        x = A @ cos_th
        y = A @ sin_th
        epi_sum = A @ epi
        vf_sum = A @ vf
        count = A.sum(axis=1)
        if w_topo != 0.0:
            degs = count
            deg_sum = A @ degs
        else:
            degs = deg_sum = None
        return x, y, epi_sum, vf_sum, count, deg_sum, degs
    else:
        idx = data["idx"]
        epi = data["epi"]
        vf = data["vf"]
        degs = data["degs"]
        cos_th = data["cos_theta"]
        sin_th = data["sin_theta"]
        n = len(nodes)
        x = [0.0] * n
        y = [0.0] * n
        epi_sum = [0.0] * n
        vf_sum = [0.0] * n
        count = [0] * n
        if w_topo != 0 and degs is not None:
            deg_sum = [0.0] * n
            degs_list = [float(degs.get(node, 0)) for node in nodes]
        else:
            deg_sum = None
            degs_list = None
        for i, node in enumerate(nodes):
            deg_i = degs_list[i] if degs_list is not None else 0.0
            for v in G.neighbors(node):
                j = idx[v]
                x[i] += cos_th[j]
                y[i] += sin_th[j]
                epi_sum[i] += epi[j]
                vf_sum[i] += vf[j]
                count[i] += 1
                if deg_sum is not None:
                    deg_sum[i] += degs.get(v, deg_i)
        return x, y, epi_sum, vf_sum, count, deg_sum, degs_list


def _compute_dnfr(G, data, *, use_numpy: bool) -> None:
    """Helper for ΔNFR computation using neighbour sums."""
    res = _build_neighbor_sums_common(G, data, use_numpy=use_numpy)
    if res is None:
        return
    x, y, epi_sum, vf_sum, count, deg_sum, degs = res
    _compute_dnfr_common(
        G,
        data,
        x=x,
        y=y,
        epi_sum=epi_sum,
        vf_sum=vf_sum,
        count=count,
        deg_sum=deg_sum,
        degs=degs,
    )


def _compute_dnfr_numpy(G, data) -> None:
    """Vectorised strategy using ``numpy``."""
    _compute_dnfr(G, data, use_numpy=True)


def _compute_dnfr_loops(G, data) -> None:
    """Loop-based strategy."""
    _compute_dnfr(G, data, use_numpy=False)


def default_compute_delta_nfr(G, *, cache_size: int | None = 1) -> None:
    """Compute ΔNFR by mixing phase, EPI, νf and a topological term.

    Parameters
    ----------
    G : nx.Graph
        Graph on which the computation is performed.
    cache_size : int | None, optional
        Maximum number of edge configurations cached in ``G.graph``. Values
        ``None`` or <= 0 imply unlimited cache. Defaults to ``1`` to keep the
        previous behaviour.
    """
    data = _prepare_dnfr_data(G, cache_size=cache_size)
    _write_dnfr_metadata(
        G,
        weights=data["weights"],
        hook_name="default_compute_delta_nfr",
    )
    if get_numpy() is not None and G.graph.get("vectorized_dnfr"):
        _compute_dnfr_numpy(G, data)
    else:
        _compute_dnfr_loops(G, data)


def set_delta_nfr_hook(
    G, func, *, name: str | None = None, note: str | None = None
) -> None:
    """Set a stable hook to compute ΔNFR.
    Required signature: ``func(G) -> None`` and it must write ``ALIAS_DNFR``
    in each node. Basic metadata in ``G.graph`` is updated accordingly.
    """
    G.graph["compute_delta_nfr"] = func
    G.graph["_dnfr_hook_name"] = str(
        name or getattr(func, "__name__", "custom_dnfr")
    )
    if "_dnfr_weights" not in G.graph:
        _configure_dnfr_weights(G)
    if note:
        meta = G.graph.get("_DNFR_META", {})
        meta["note"] = str(note)
        G.graph["_DNFR_META"] = meta


def _apply_dnfr_hook(
    G,
    grads: Dict[str, Callable[[Any, Any], float]],
    *,
    weights: Dict[str, float],
    hook_name: str,
    note: str | None = None,
) -> None:
    """Generic helper to compute and store ΔNFR using ``grads``.

    ``grads`` maps component names to functions ``(G, n, nd) -> float``.
    Each gradient is multiplied by its corresponding weight from ``weights``.
    Metadata is recorded through :func:`_write_dnfr_metadata`.
    """

    for n, nd in G.nodes(data=True):
        total = 0.0
        for name, func in grads.items():
            w = weights.get(name, 0.0)
            if w:
                total += w * func(G, n, nd)
        set_dnfr(G, n, total)

    _write_dnfr_metadata(
        G, weights=weights, hook_name=hook_name, note=note
    )


# --- Hooks de ejemplo (opcionales) ---
def dnfr_phase_only(G) -> None:
    """Example: ΔNFR from phase only (Kuramoto-like)."""

    def g_phase(G, n, nd):
        th_i = get_attr(nd, ALIAS_THETA, 0.0)
        th_bar = neighbor_phase_mean(G, n)
        return -angle_diff(th_i, th_bar) / math.pi

    _apply_dnfr_hook(
        G,
        {"phase": g_phase},
        weights={"phase": 1.0},
        hook_name="dnfr_phase_only",
        note="Hook de ejemplo.",
    )


def dnfr_epi_vf_mixed(G) -> None:
    """Example: ΔNFR without phase, mixing EPI and νf."""

    def g_epi(G, n, nd):
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)
        epi_bar = neighbor_mean(G, n, ALIAS_EPI, default=epi_i)
        return epi_bar - epi_i

    def g_vf(G, n, nd):
        vf_i = get_attr(nd, ALIAS_VF, 0.0)
        vf_bar = neighbor_mean(G, n, ALIAS_VF, default=vf_i)
        return vf_bar - vf_i

    _apply_dnfr_hook(
        G,
        {"epi": g_epi, "vf": g_vf},
        weights={"phase": 0.0, "epi": 0.5, "vf": 0.5},
        hook_name="dnfr_epi_vf_mixed",
        note="Hook de ejemplo.",
    )


def dnfr_laplacian(G) -> None:
    """Explicit topological gradient using Laplacian over EPI and νf."""
    wE = float(G.graph.get("DNFR_WEIGHTS", {}).get("epi", 0.33))
    wV = float(G.graph.get("DNFR_WEIGHTS", {}).get("vf", 0.33))

    def g_epi(G, n, nd):
        epi = get_attr(nd, ALIAS_EPI, 0.0)
        neigh = list(G.neighbors(n))
        deg = len(neigh) or 1
        epi_bar = sum(get_attr(G.nodes[v], ALIAS_EPI, epi) for v in neigh) / deg
        return epi_bar - epi

    def g_vf(G, n, nd):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        neigh = list(G.neighbors(n))
        deg = len(neigh) or 1
        vf_bar = sum(get_attr(G.nodes[v], ALIAS_VF, vf) for v in neigh) / deg
        return vf_bar - vf

    _apply_dnfr_hook(
        G,
        {"epi": g_epi, "vf": g_vf},
        weights={"epi": wE, "vf": wV},
        hook_name="dnfr_laplacian",
        note="Gradiente topológico",
    )


def apply_dnfr_field(G, w_theta=None, w_epi=None, w_vf=None) -> None:
    if any(v is not None for v in (w_theta, w_epi, w_vf)):
        mix = G.graph.get("DNFR_WEIGHTS", DEFAULTS["DNFR_WEIGHTS"]).copy()
        if w_theta is not None:
            mix["phase"] = float(w_theta)
        if w_epi is not None:
            mix["epi"] = float(w_epi)
        if w_vf is not None:
            mix["vf"] = float(w_vf)
        G.graph["DNFR_WEIGHTS"] = mix
    default_compute_delta_nfr(G)
