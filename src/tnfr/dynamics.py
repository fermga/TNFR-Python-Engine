"""System dynamics."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Any, Literal
import networkx as nx

# Importar compute_Si y apply_glyph a nivel de módulo evita el coste de
# realizar la importación en cada paso de la dinámica. Como los módulos de
# origen no dependen de ``dynamics``, no se introducen ciclos.
from .operators import apply_remesh_if_globally_stable, apply_glyph
from .grammar import (
    enforce_canonical_grammar,
    on_applied_glyph,
    AL,
    EN,
)
from .constants import (
    DEFAULTS,
    REMESH_DEFAULTS,
    METRIC_DEFAULTS,
    ALIAS_VF,
    ALIAS_THETA,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_dEPI,
    ALIAS_D2EPI,
    ALIAS_dSI,
    ALIAS_EPI_KIND,
    get_param,
)
from .gamma import eval_gamma
from .observers import glyph_load, kuramoto_order

from .helpers import (
    clamp,
    clamp01,
    angle_diff,
    neighbor_mean,
    neighbor_phase_mean,
    cached_nodes_and_A,
    _cache_node_list,
    _phase_mean_from_iter,
)
from .alias import (
    get_attr,
    set_attr,
    get_attr_str,
    set_attr_str,
    set_vf,
    set_dnfr,
    multi_recompute_abs_max,
)
from .metrics_utils import compute_Si, compute_dnfr_accel_max, get_trig_cache
from .rng import _rng_for_step, base_seed
from .callback_utils import invoke_callbacks
from .glyph_history import recent_glyph, ensure_history, append_metric
from .collections_utils import normalize_weights
from .import_utils import get_numpy
from .selector import (
    _selector_thresholds,
    _norms_para_selector,
    _calc_selector_score,
    _apply_selector_hysteresis,
)

from .logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "default_compute_delta_nfr",
    "set_delta_nfr_hook",
    "dnfr_phase_only",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "prepare_integration_params",
    "update_epi_via_nodal_equation",
    "apply_dnfr_field",
    "integrar_epi_euler",
    "apply_canonical_clamps",
    "validate_canon",
    "coordinate_global_local_phase",
    "adapt_vf_by_coherence",
    "default_glyph_selector",
    "parametric_glyph_selector",
    "step",
    "run",
]


def _update_node_sample(G, *, step: int) -> None:
    """Refresh ``G.graph['_node_sample']`` with a random subset of nodes.

    The sample is limited by ``UM_CANDIDATE_COUNT`` and refreshed every
    simulation step. When the network is small (``< 50`` nodes) or the limit
    is non‑positive, the full node set is used and sampling is effectively
    disabled. A tuple snapshot of nodes is cached in
    ``G.graph['_node_list']`` and reused across steps; it is only refreshed
    when the graph size changes. Sampling operates directly on this cached
    tuple.
    """
    graph = G.graph
    limit = int(graph.get("UM_CANDIDATE_COUNT", 0))
    nodes = _cache_node_list(G)
    current_n = len(nodes)
    if limit <= 0 or current_n < 50 or limit >= current_n:
        graph["_node_sample"] = nodes
        return

    seed = base_seed(G)
    rng = _rng_for_step(seed, step)
    graph["_node_sample"] = rng.sample(nodes, limit)


# -------------------------
# ΔNFR por defecto (campo) + utilidades de hook/metadata
# -------------------------


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


# -------------------------
# Ecuación nodal
# -------------------------


def prepare_integration_params(
    G,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
):
    """Validate and normalise ``dt``, ``t`` and ``method`` for integration.

    Returns ``(dt_step, steps, t0, method)`` where ``dt_step`` is the
    effective step, ``steps`` the number of substeps and ``t0`` the prepared
    initial time.
    """
    if dt is None:
        dt = float(G.graph.get("DT", DEFAULTS["DT"]))
    else:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a number")
        if dt < 0:
            raise ValueError("dt must be non-negative")
        dt = float(dt)

    if t is None:
        t = float(G.graph.get("_t", 0.0))
    else:
        t = float(t)

    method = (
        method
        or G.graph.get(
            "INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler")
        )
    ).lower()
    if method not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")

    dt_min = float(G.graph.get("DT_MIN", DEFAULTS.get("DT_MIN", 0.0)))
    if dt_min > 0 and dt > dt_min:
        steps = int(math.ceil(dt / dt_min))
    else:
        steps = 1
    dt_step = dt / steps if steps else 0.0

    return dt_step, steps, t, method


def _integrate_euler(G, dt_step: float, t_local: float):
    """One explicit Euler integration step."""
    gamma_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
    new_states: Dict[Any, tuple[float, float, float]] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dEPI_dt_prev = get_attr(nd, ALIAS_dEPI, 0.0)
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)

        base = vf * dnfr
        dEPI_dt = base + gamma_map.get(n, 0.0)
        epi = epi_i + dt_step * dEPI_dt
        d2epi = (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0
        new_states[n] = (epi, dEPI_dt, d2epi)
    return new_states


def _integrate_rk4(G, dt_step: float, t_local: float):
    """One Runge–Kutta order-4 integration step."""
    t_mid = t_local + dt_step / 2.0
    t_end = t_local + dt_step
    g1_map = {n: eval_gamma(G, n, t_local) for n in G.nodes}
    g_mid_map = {n: eval_gamma(G, n, t_mid) for n in G.nodes}
    g4_map = {n: eval_gamma(G, n, t_end) for n in G.nodes}

    new_states: Dict[Any, tuple[float, float, float]] = {}
    for n, nd in G.nodes(data=True):
        vf = get_attr(nd, ALIAS_VF, 0.0)
        dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
        dEPI_dt_prev = get_attr(nd, ALIAS_dEPI, 0.0)
        epi_i = get_attr(nd, ALIAS_EPI, 0.0)

        base = vf * dnfr
        g1 = g1_map.get(n, 0.0)
        g_mid = g_mid_map.get(n, 0.0)
        g4 = g4_map.get(n, 0.0)
        k1 = base + g1
        k2 = base + g_mid
        k3 = base + g_mid
        k4 = base + g4
        epi = epi_i + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        dEPI_dt = k4
        d2epi = (dEPI_dt - dEPI_dt_prev) / dt_step if dt_step != 0 else 0.0
        new_states[n] = (epi, dEPI_dt, d2epi)
    return new_states


def update_epi_via_nodal_equation(
    G,
    *,
    dt: float = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
) -> None:
    """TNFR nodal equation.

    Implements the extended nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)

    Where:
      - EPI is the node's Primary Information Structure.
      - νf is the node's structural frequency (Hz_str).
      - ΔNFR(t) is the nodal gradient (reorganisation need), typically a mix
        of components (e.g. phase θ, EPI, νf).
      - Γi(R) is the optional network coupling as a function of Kuramoto order
        ``R`` (see :mod:`gamma`), used to modulate network integration.

    TNFR references: nodal equation (manual), νf/ΔNFR/EPI glossary, Γ operator.
    Side effects: caches dEPI and updates EPI via explicit integration.
    """
    if not isinstance(
        G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    ):
        raise TypeError("G must be a networkx graph instance")

    dt_step, steps, t0, method = prepare_integration_params(G, dt, t, method)

    t_local = t0
    for _ in range(steps):
        if method == "rk4":
            updates = _integrate_rk4(G, dt_step, t_local)
        else:
            updates = _integrate_euler(G, dt_step, t_local)

        for n, (epi, dEPI_dt, d2epi) in updates.items():
            nd = G.nodes[n]
            epi_kind = get_attr_str(nd, ALIAS_EPI_KIND, "")
            set_attr(nd, ALIAS_EPI, epi)
            if epi_kind:
                set_attr_str(nd, ALIAS_EPI_KIND, epi_kind)
            set_attr(nd, ALIAS_dEPI, dEPI_dt)
            set_attr(nd, ALIAS_D2EPI, d2epi)

        t_local += dt_step

    G.graph["_t"] = t_local


# -------------------------
# Wrappers nombrados (compatibilidad)
# -------------------------


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


def integrar_epi_euler(G, dt: float | None = None) -> None:
    update_epi_via_nodal_equation(G, dt=dt, method="euler")


def _param(G, name):
    return float(G.graph.get(name) if G is not None else DEFAULTS[name])


def _log_clamp(hist, node, attr, value, lo, hi):
    if value < lo or value > hi:
        hist.append({"node": node, "attr": attr, "value": float(value)})


def apply_canonical_clamps(nd: Dict[str, Any], G=None, node=None) -> None:
    eps_min = _param(G, "EPI_MIN")
    eps_max = _param(G, "EPI_MAX")
    vf_min = _param(G, "VF_MIN")
    vf_max = _param(G, "VF_MAX")

    epi = get_attr(nd, ALIAS_EPI, 0.0)
    vf = get_attr(nd, ALIAS_VF, 0.0)
    th = get_attr(nd, ALIAS_THETA, 0.0)

    strict = bool(
        (
            G.graph.get("VALIDATORS_STRICT")
            if G is not None
            else DEFAULTS.get("VALIDATORS_STRICT", False)
        )
    )
    if strict and G is not None:
        hist = G.graph.setdefault("history", {}).setdefault("clamp_alerts", [])
        _log_clamp(hist, node, "EPI", epi, eps_min, eps_max)
        _log_clamp(hist, node, "VF", vf, vf_min, vf_max)

    set_attr(nd, ALIAS_EPI, clamp(epi, eps_min, eps_max))
    if G is not None and node is not None:
        set_vf(G, node, clamp(vf, vf_min, vf_max), update_max=False)
    else:
        set_attr(nd, ALIAS_VF, clamp(vf, vf_min, vf_max))
    if G.graph.get("THETA_WRAP") if G is not None else DEFAULTS["THETA_WRAP"]:
        set_attr(nd, ALIAS_THETA, ((th + math.pi) % (2 * math.pi) - math.pi))


def validate_canon(G) -> None:
    """Apply canonical clamps to all nodes of ``G``.

    Wrap phase and constrain ``EPI`` and ``νf`` to the ranges in ``G.graph``.
    If ``VALIDATORS_STRICT`` is active, alerts are logged in ``history``.
    """
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(nd, G, n)
    maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF})
    G.graph.update(maxes)
    return G


def _read_adaptive_params(
    g: Dict[str, Any],
) -> tuple[Dict[str, Any], float, float]:
    """Obtain configuration and current values for phase adaptation."""
    cfg = g.get("PHASE_ADAPT", DEFAULTS.get("PHASE_ADAPT", {}))
    kG = float(g.get("PHASE_K_GLOBAL", DEFAULTS["PHASE_K_GLOBAL"]))
    kL = float(g.get("PHASE_K_LOCAL", DEFAULTS["PHASE_K_LOCAL"]))
    return cfg, kG, kL


def _compute_state(G, cfg: Dict[str, Any]) -> tuple[str, float, float]:
    """Return current state (stable/dissonant/transition) and metrics."""
    R = kuramoto_order(G)
    win = int(
        G.graph.get("GLYPH_LOAD_WINDOW", METRIC_DEFAULTS["GLYPH_LOAD_WINDOW"])
    )
    dist = glyph_load(G, window=win)
    disr = float(dist.get("_disruptivos", 0.0)) if dist else 0.0

    R_hi = float(cfg.get("R_hi", 0.90))
    R_lo = float(cfg.get("R_lo", 0.60))
    disr_hi = float(cfg.get("disr_hi", 0.50))
    disr_lo = float(cfg.get("disr_lo", 0.25))
    if (R >= R_hi) and (disr <= disr_lo):
        state = "estable"
    elif (R <= R_lo) or (disr >= disr_hi):
        state = "disonante"
    else:
        state = "transicion"
    return state, float(R), disr


def _smooth_adjust_k(
    kG: float, kL: float, state: str, cfg: Dict[str, Any]
) -> tuple[float, float]:
    """Smoothly update kG/kL toward targets according to state."""
    kG_min = float(cfg.get("kG_min", 0.01))
    kG_max = float(cfg.get("kG_max", 0.20))
    kL_min = float(cfg.get("kL_min", 0.05))
    kL_max = float(cfg.get("kL_max", 0.25))

    if state == "disonante":
        kG_t = kG_max
        kL_t = 0.5 * (
            kL_min + kL_max
        )  # local medio para no perder plasticidad
    elif state == "estable":
        kG_t = kG_min
        kL_t = kL_min
    else:
        kG_t = 0.5 * (kG_min + kG_max)
        kL_t = 0.5 * (kL_min + kL_max)

    up = float(cfg.get("up", 0.10))
    down = float(cfg.get("down", 0.07))

    def _step(curr: float, target: float, mn: float, mx: float) -> float:
        gain = up if target > curr else down
        nxt = curr + gain * (target - curr)
        return max(mn, min(mx, nxt))

    return _step(kG, kG_t, kG_min, kG_max), _step(kL, kL_t, kL_min, kL_max)


def coordinate_global_local_phase(
    G, global_force: float | None = None, local_force: float | None = None
) -> None:
    """
    Ajusta fase con mezcla GLOBAL+VECINAL.
    Si no se pasan fuerzas explícitas, adapta kG/kL según estado
    (disonante / transición / estable).
    Estado se decide por R (Kuramoto) y carga glífica disruptiva reciente.
    """
    g = G.graph
    defaults = DEFAULTS
    hist = g.setdefault("history", {})
    maxlen = int(
        g.get("PHASE_HISTORY_MAXLEN", METRIC_DEFAULTS["PHASE_HISTORY_MAXLEN"])
    )
    hist_state = hist.setdefault("phase_state", deque(maxlen=maxlen))
    if not isinstance(hist_state, deque):
        hist_state = deque(hist_state, maxlen=maxlen)
        hist["phase_state"] = hist_state
    hist_R = hist.setdefault("phase_R", deque(maxlen=maxlen))
    if not isinstance(hist_R, deque):
        hist_R = deque(hist_R, maxlen=maxlen)
        hist["phase_R"] = hist_R
    hist_disr = hist.setdefault("phase_disr", deque(maxlen=maxlen))
    if not isinstance(hist_disr, deque):
        hist_disr = deque(hist_disr, maxlen=maxlen)
        hist["phase_disr"] = hist_disr
    # 0) Si hay fuerzas explícitas, usar y salir del modo adaptativo
    if (global_force is not None) or (local_force is not None):
        kG = float(
            global_force
            if global_force is not None
            else g.get("PHASE_K_GLOBAL", defaults["PHASE_K_GLOBAL"])
        )
        kL = float(
            local_force
            if local_force is not None
            else g.get("PHASE_K_LOCAL", defaults["PHASE_K_LOCAL"])
        )
    else:
        cfg, kG, kL = _read_adaptive_params(g)

        if bool(cfg.get("enabled", False)):
            state, R, disr = _compute_state(G, cfg)
            kG, kL = _smooth_adjust_k(kG, kL, state, cfg)

            hist_state.append(state)
            hist_R.append(float(R))
            hist_disr.append(float(disr))

    g["PHASE_K_GLOBAL"] = kG
    g["PHASE_K_LOCAL"] = kL
    append_metric(hist, "phase_kG", float(kG))
    append_metric(hist, "phase_kL", float(kL))

    # 6) Fase GLOBAL (centroide) para empuje
    x_sum = y_sum = 0.0
    for _, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        x_sum += math.cos(th)
        y_sum += math.sin(th)
    num_nodes = G.number_of_nodes()
    if num_nodes:
        thG = math.atan2(y_sum / num_nodes, x_sum / num_nodes)
    else:
        thG = 0.0

    # 7) Aplicar corrección global+vecinal
    for n, nd in G.nodes(data=True):
        th = get_attr(nd, ALIAS_THETA, 0.0)
        thL = neighbor_phase_mean(G, n)
        dG = angle_diff(thG, th)
        dL = angle_diff(thL, th)
        set_attr(nd, ALIAS_THETA, th + kG * dG + kL * dL)


# -------------------------
# Adaptación de νf por coherencia
# -------------------------


def adapt_vf_by_coherence(G) -> None:
    """Adjust νf toward neighbour mean in nodes with sustained stability."""
    tau = int(G.graph.get("VF_ADAPT_TAU", DEFAULTS.get("VF_ADAPT_TAU", 5)))
    mu = float(G.graph.get("VF_ADAPT_MU", DEFAULTS.get("VF_ADAPT_MU", 0.1)))
    eps_dnfr = float(
        G.graph.get("EPS_DNFR_STABLE", REMESH_DEFAULTS["EPS_DNFR_STABLE"])
    )
    thr_sel = G.graph.get(
        "SELECTOR_THRESHOLDS", DEFAULTS.get("SELECTOR_THRESHOLDS", {})
    )
    thr_def = G.graph.get(
        "GLYPH_THRESHOLDS", DEFAULTS.get("GLYPH_THRESHOLDS", {"hi": 0.66})
    )
    si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
    vf_min = float(G.graph.get("VF_MIN", DEFAULTS["VF_MIN"]))
    vf_max = float(G.graph.get("VF_MAX", DEFAULTS["VF_MAX"]))

    updates = {}
    for n, nd in G.nodes(data=True):
        Si = get_attr(nd, ALIAS_SI, 0.0)
        dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0))
        if Si >= si_hi and dnfr <= eps_dnfr:
            nd["stable_count"] = nd.get("stable_count", 0) + 1
        else:
            nd["stable_count"] = 0
            continue

        if nd["stable_count"] >= tau:
            vf = get_attr(nd, ALIAS_VF, 0.0)
            vf_bar = neighbor_mean(G, n, ALIAS_VF, default=vf)
            updates[n] = vf + mu * (vf_bar - vf)

    for n, vf_new in updates.items():
        set_vf(G, n, clamp(vf_new, vf_min, vf_max))


# -------------------------
# Selector glífico por defecto
# -------------------------
def default_glyph_selector(G, n) -> str:
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    hi, lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]

    norms = G.graph.get("_sel_norms")
    if norms is None:
        norms = compute_dnfr_accel_max(G)
        G.graph["_sel_norms"] = norms
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0

    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max

    if Si >= hi:
        return "IL"
    if Si <= lo:
        return "OZ" if dnfr > dnfr_hi else "ZHIR"
    return "NAV" if dnfr > dnfr_hi else "RA"


# -------------------------
# Selector glífico multiobjetivo (paramétrico)
# -------------------------
def _soft_grammar_prefilter(G, n, cand, dnfr, accel):
    """Soft grammar: avoid repetitions before the canonical one."""
    gram = G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {}))
    gwin = int(gram.get("window", 3))
    avoid = set(gram.get("avoid_repeats", []))
    force_dn = float(gram.get("force_dnfr", 0.60))
    force_ac = float(gram.get("force_accel", 0.60))
    fallbacks = gram.get("fallbacks", {})
    nd = G.nodes[n]
    if cand in avoid and recent_glyph(nd, cand, gwin):
        if not (dnfr >= force_dn or accel >= force_ac):
            cand = fallbacks.get(cand, cand)
    return cand


def _selector_normalized_metrics(nd, norms):
    """Extract and normalise Si, ΔNFR and acceleration for the selector."""
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    acc_max = float(norms.get("accel_max", 1.0)) or 1.0
    Si = clamp01(get_attr(nd, ALIAS_SI, 0.5))
    dnfr = abs(get_attr(nd, ALIAS_DNFR, 0.0)) / dnfr_max
    accel = abs(get_attr(nd, ALIAS_D2EPI, 0.0)) / acc_max
    return Si, dnfr, accel


def _selector_base_choice(Si, dnfr, accel, thr):
    """Base decision according to thresholds of Si, ΔNFR and acceleration."""
    si_hi, si_lo = thr["si_hi"], thr["si_lo"]
    dnfr_hi = thr["dnfr_hi"]
    acc_hi = thr["accel_hi"]
    if Si >= si_hi:
        return "IL"
    if Si <= si_lo:
        if accel >= acc_hi:
            return "THOL"
        return "OZ" if dnfr >= dnfr_hi else "ZHIR"
    if dnfr >= dnfr_hi or accel >= acc_hi:
        return "NAV"
    return "RA"


def _configure_selector_weights(G) -> dict:
    """Normalise and store selector weights in ``G.graph``."""
    w = {**DEFAULTS["SELECTOR_WEIGHTS"], **G.graph.get("SELECTOR_WEIGHTS", {})}
    weights = normalize_weights(w, ("w_si", "w_dnfr", "w_accel"))
    G.graph["_selector_weights"] = weights
    return weights


def _compute_selector_score(G, nd, Si, dnfr, accel, cand):
    """Compute score and apply stagnation penalties."""
    W = G.graph.get("_selector_weights")
    if W is None:
        W = _configure_selector_weights(G)
    score = _calc_selector_score(Si, dnfr, accel, W)
    hist_prev = nd.get("glyph_history")
    if hist_prev and hist_prev[-1] == cand:
        delta_si = get_attr(nd, ALIAS_dSI, 0.0)
        h = G.graph.get("history", {})
        sig = h.get("sense_sigma_mag", [])
        delta_sigma = sig[-1] - sig[-2] if len(sig) >= 2 else 0.0
        if delta_si <= 0.0 and delta_sigma <= 0.0:
            score -= 0.05
    return score


def _apply_score_override(cand, score, dnfr, dnfr_lo):
    """Adjust final candidate smoothly according to the score."""
    if score >= 0.66 and cand in ("NAV", "RA", "ZHIR", "OZ"):
        cand = "IL"
    elif score <= 0.33 and cand in ("NAV", "RA", "IL"):
        cand = "OZ" if dnfr >= dnfr_lo else "ZHIR"
    return cand


def parametric_glyph_selector(G, n) -> str:
    """Multiobjective: combine Si, |ΔNFR|_norm and |accel|_norm with hysteresis.
    Base rules:
      - High Si  ⇒ IL
      - Low Si   ⇒ OZ if |ΔNFR| high; ZHIR if |ΔNFR| low; THOL if acceleration is high
      - Medium Si ⇒ NAV if |ΔNFR| high (or acceleration high), otherwise RA
    """
    nd = G.nodes[n]
    thr = _selector_thresholds(G)
    margin = float(
        G.graph.get("GLYPH_SELECTOR_MARGIN", DEFAULTS["GLYPH_SELECTOR_MARGIN"])
    )

    norms = G.graph.get("_sel_norms") or _norms_para_selector(G)
    Si, dnfr, accel = _selector_normalized_metrics(nd, norms)

    cand = _selector_base_choice(Si, dnfr, accel, thr)

    hist_cand = _apply_selector_hysteresis(nd, Si, dnfr, accel, thr, margin)
    if hist_cand is not None:
        return hist_cand

    score = _compute_selector_score(G, nd, Si, dnfr, accel, cand)

    cand = _apply_score_override(cand, score, dnfr, thr["dnfr_lo"])

    return _soft_grammar_prefilter(G, n, cand, dnfr, accel)


def _choose_glyph(G, n, selector, use_canon, h_al, h_en, al_max, en_max):
    """Select the glyph to apply on node ``n``."""
    if h_al[n] > al_max:
        return AL
    if h_en[n] > en_max:
        return EN
    g = selector(G, n)
    if use_canon:
        g = enforce_canonical_grammar(G, n, g)
    return g


# -------------------------
# Step / run
# -------------------------


def _run_before_callbacks(
    G, *, step_idx: int, dt: float | None, use_Si: bool, apply_glyphs: bool
) -> None:
    invoke_callbacks(
        G,
        "before_step",
        {
            "step": step_idx,
            "dt": dt,
            "use_Si": use_Si,
            "apply_glyphs": apply_glyphs,
        },
    )


def _prepare_dnfr(G, *, use_Si: bool) -> None:
    """Compute ΔNFR and optionally Si for the current graph state."""
    compute_dnfr_cb = G.graph.get(
        "compute_delta_nfr", default_compute_delta_nfr
    )
    compute_dnfr_cb(G)
    if use_Si:
        compute_Si(G, inplace=True)


def _apply_selector(G):
    """Configure and return the glyph selector for this step."""
    selector = G.graph.get("glyph_selector", default_glyph_selector)
    if selector is parametric_glyph_selector:
        _norms_para_selector(G)
        _configure_selector_weights(G)
    return selector


def _apply_glyphs(G, selector, hist) -> None:
    """Apply glyphs to nodes using ``selector`` and update history."""
    window = int(get_param(G, "GLYPH_HYSTERESIS_WINDOW"))
    use_canon = bool(
        G.graph.get("GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})).get(
            "enabled", False
        )
    )
    al_max = int(G.graph.get("AL_MAX_LAG", DEFAULTS["AL_MAX_LAG"]))
    en_max = int(G.graph.get("EN_MAX_LAG", DEFAULTS["EN_MAX_LAG"]))
    h_al = hist.setdefault("since_AL", {})
    h_en = hist.setdefault("since_EN", {})
    for n, _ in G.nodes(data=True):
        h_al[n] = int(h_al.get(n, 0)) + 1
        h_en[n] = int(h_en.get(n, 0)) + 1
        g = _choose_glyph(G, n, selector, use_canon, h_al, h_en, al_max, en_max)
        apply_glyph(G, n, g, window=window)
        if use_canon:
            on_applied_glyph(G, n, g)
        if g == AL:
            h_al[n] = 0
            h_en[n] = min(h_en[n], en_max)
        elif g == EN:
            h_en[n] = 0


def _update_nodes(
    G,
    *,
    dt: float | None,
    use_Si: bool,
    apply_glyphs: bool,
    step_idx: int,
    hist,
) -> None:
    _update_node_sample(G, step=step_idx)
    _prepare_dnfr(G, use_Si=use_Si)
    selector = _apply_selector(G)
    if apply_glyphs:
        _apply_glyphs(G, selector, hist)
    _dt = float(G.graph.get("DT", DEFAULTS["DT"])) if dt is None else float(dt)
    method = G.graph.get(
        "INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler")
    )
    update_epi_via_nodal_equation(G, dt=_dt, method=method)
    for n, nd in G.nodes(data=True):
        apply_canonical_clamps(nd, G, n)
    coordinate_global_local_phase(G, None, None)
    adapt_vf_by_coherence(G)


def _update_epi_hist(G) -> None:
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau = max(tau_g, tau_l)
    maxlen = max(2 * tau + 5, 64)
    epi_hist = G.graph.get("_epi_hist")
    if not isinstance(epi_hist, deque) or epi_hist.maxlen != maxlen:
        epi_hist = deque(list(epi_hist or [])[-maxlen:], maxlen=maxlen)
        G.graph["_epi_hist"] = epi_hist
    epi_hist.append(
        {n: get_attr(nd, ALIAS_EPI, 0.0) for n, nd in G.nodes(data=True)}
    )


def _maybe_remesh(G) -> None:
    apply_remesh_if_globally_stable(G)


def _run_validators(G) -> None:
    from .validators import run_validators

    run_validators(G)


def _run_after_callbacks(G, *, step_idx: int) -> None:
    h = G.graph.get("history", {})
    ctx = {"step": step_idx}
    metric_pairs = [
        ("C", "C_steps"),
        ("stable_frac", "stable_frac"),
        ("phase_sync", "phase_sync"),
        ("glyph_disr", "glyph_load_disr"),
        ("Si_mean", "Si_mean"),
    ]
    for dst, src in metric_pairs:
        values = h.get(src)
        if values:
            ctx[dst] = values[-1]
    invoke_callbacks(G, "after_step", ctx)


def step(
    G,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    hist = ensure_history(G)
    step_idx = len(hist.setdefault("C_steps", []))
    _run_before_callbacks(
        G, step_idx=step_idx, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs
    )
    _update_nodes(
        G,
        dt=dt,
        use_Si=use_Si,
        apply_glyphs=apply_glyphs,
        step_idx=step_idx,
        hist=hist,
    )
    _update_epi_hist(G)
    _maybe_remesh(G)
    _run_validators(G)
    _run_after_callbacks(G, step_idx=step_idx)


def run(
    G,
    steps: int,
    *,
    dt: float | None = None,
    use_Si: bool = True,
    apply_glyphs: bool = True,
) -> None:
    for _ in range(int(steps)):
        step(G, dt=dt, use_Si=use_Si, apply_glyphs=apply_glyphs)
        # Early-stop opcional
        stop_cfg = G.graph.get(
            "STOP_EARLY", METRIC_DEFAULTS.get("STOP_EARLY", {"enabled": False})
        )
        if stop_cfg and stop_cfg.get("enabled", False):
            w = int(stop_cfg.get("window", 25))
            frac = float(stop_cfg.get("fraction", 0.90))
            hist = G.graph.setdefault("history", {"stable_frac": []})
            series = hist.get("stable_frac", [])
            if len(series) >= w and all(v >= frac for v in series[-w:]):
                break
