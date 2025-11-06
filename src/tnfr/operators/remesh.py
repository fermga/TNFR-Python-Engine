"""Adaptive remeshing operators preserving TNFR structural coherence."""

from __future__ import annotations

import hashlib
import heapq
import random
from collections import deque
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from functools import cache
from io import StringIO
from itertools import combinations
from operator import ge, le
from statistics import StatisticsError, fmean
from types import ModuleType
from typing import Any, cast

from .._compat import TypeAlias
from ..alias import get_attr, set_attr
from ..constants import DEFAULTS, REMESH_DEFAULTS, get_param
from ..constants.aliases import ALIAS_EPI
from ..rng import make_rng
from ..types import RemeshMeta
from ..utils import cached_import, edge_version_update, kahan_sum_nd

CommunityGraph: TypeAlias = Any
NetworkxModule: TypeAlias = ModuleType
CommunityModule: TypeAlias = ModuleType
RemeshEdge: TypeAlias = tuple[Hashable, Hashable]
NetworkxModules: TypeAlias = tuple[NetworkxModule, CommunityModule]
RemeshConfigValue: TypeAlias = bool | float | int


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to ``float`` returning ``default`` on failure."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ordered_edge(u: Hashable, v: Hashable) -> RemeshEdge:
    """Return a deterministic ordering for an undirected edge."""

    return (u, v) if repr(u) <= repr(v) else (v, u)


COOLDOWN_KEY = "REMESH_COOLDOWN_WINDOW"


@cache
def _get_networkx_modules() -> NetworkxModules:
    nx = cached_import("networkx")
    if nx is None:
        raise ImportError(
            "networkx is required for network operators; install 'networkx' "
            "to enable this feature"
        )
    nx_comm = cached_import("networkx.algorithms", "community")
    if nx_comm is None:
        raise ImportError(
            "networkx.algorithms.community is required for community-based "
            "operations; install 'networkx' to enable this feature"
        )
    return cast(NetworkxModule, nx), cast(CommunityModule, nx_comm)


def _remesh_alpha_info(G: CommunityGraph) -> tuple[float, str]:
    """Return ``(alpha, source)`` with explicit precedence."""
    if bool(G.graph.get("REMESH_ALPHA_HARD", REMESH_DEFAULTS["REMESH_ALPHA_HARD"])):
        val = _as_float(
            G.graph.get("REMESH_ALPHA", REMESH_DEFAULTS["REMESH_ALPHA"]),
            float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        )
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return _as_float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return _as_float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return (
        float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        "REMESH_DEFAULTS.REMESH_ALPHA",
    )


def _snapshot_topology(G: CommunityGraph, nx: NetworkxModule) -> str | None:
    """Return a hash representing the current graph topology."""
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        return hashlib.blake2b(topo_str.encode(), digest_size=6).hexdigest()
    except (AttributeError, TypeError, nx.NetworkXError):
        return None


def _snapshot_epi(G: CommunityGraph) -> tuple[float, str]:
    """Return ``(mean, checksum)`` of the node EPI values."""
    buf = StringIO()
    values = []
    for n, data in G.nodes(data=True):
        v = _as_float(get_attr(data, ALIAS_EPI, 0.0))
        values.append(v)
        buf.write(f"{str(n)}:{round(v, 6)};")
    total = kahan_sum_nd(((v,) for v in values), dims=1)[0]
    mean_val = total / len(values) if values else 0.0
    checksum = hashlib.blake2b(buf.getvalue().encode(), digest_size=6).hexdigest()
    return float(mean_val), checksum


def _log_remesh_event(G: CommunityGraph, meta: RemeshMeta) -> None:
    """Store remesh metadata and optionally log and trigger callbacks."""
    from ..utils import CallbackEvent, callback_manager
    from ..glyph_history import append_metric

    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        append_metric(hist, "remesh_events", dict(meta))
    callback_manager.invoke_callbacks(G, CallbackEvent.ON_REMESH.value, dict(meta))


def apply_network_remesh(G: CommunityGraph) -> None:
    """Network-scale REMESH using ``_epi_hist`` with multi-scale memory."""
    from ..glyph_history import current_step_idx, ensure_history
    from ..dynamics.structural_clip import structural_clip

    nx, _ = _get_networkx_modules()
    tau_g = int(get_param(G, "REMESH_TAU_GLOBAL"))
    tau_l = int(get_param(G, "REMESH_TAU_LOCAL"))
    tau_req = max(tau_g, tau_l)
    alpha, alpha_src = _remesh_alpha_info(G)
    G.graph["_REMESH_ALPHA_SRC"] = alpha_src
    hist = G.graph.get("_epi_hist", deque())
    if len(hist) < tau_req + 1:
        return

    past_g = hist[-(tau_g + 1)]
    past_l = hist[-(tau_l + 1)]

    topo_hash = _snapshot_topology(G, nx)
    epi_mean_before, epi_checksum_before = _snapshot_epi(G)
    
    # Get EPI bounds for structural preservation
    epi_min = float(G.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
    epi_max = float(G.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
    clip_mode_str = str(G.graph.get("CLIP_MODE", "hard"))
    if clip_mode_str not in ("hard", "soft"):
        clip_mode_str = "hard"
    clip_mode = clip_mode_str  # type: ignore[assignment]

    for n, nd in G.nodes(data=True):
        epi_now = _as_float(get_attr(nd, ALIAS_EPI, 0.0))
        epi_old_l = _as_float(
            past_l.get(n) if isinstance(past_l, Mapping) else None, epi_now
        )
        epi_old_g = _as_float(
            past_g.get(n) if isinstance(past_g, Mapping) else None, epi_now
        )
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        
        # Apply structural boundary preservation to prevent overflow
        mixed_clipped = structural_clip(mixed, lo=epi_min, hi=epi_max, mode=clip_mode)
        set_attr(nd, ALIAS_EPI, mixed_clipped)

    epi_mean_after, epi_checksum_after = _snapshot_epi(G)

    step_idx = current_step_idx(G)
    meta: RemeshMeta = {
        "alpha": alpha,
        "alpha_source": alpha_src,
        "tau_global": tau_g,
        "tau_local": tau_l,
        "step": step_idx,
        "topo_hash": topo_hash,
        "epi_mean_before": float(epi_mean_before),
        "epi_mean_after": float(epi_mean_after),
        "epi_checksum_before": epi_checksum_before,
        "epi_checksum_after": epi_checksum_after,
    }

    h = ensure_history(G)
    if h:
        if h.get("stable_frac"):
            meta["stable_frac_last"] = h["stable_frac"][-1]
        if h.get("phase_sync"):
            meta["phase_sync_last"] = h["phase_sync"][-1]
        if h.get("glyph_load_disr"):
            meta["glyph_disr_last"] = h["glyph_load_disr"][-1]

    _log_remesh_event(G, meta)


def _mst_edges_from_epi(
    nx: NetworkxModule,
    nodes: Sequence[Hashable],
    epi: Mapping[Hashable, float],
) -> set[RemeshEdge]:
    """Return MST edges based on absolute EPI distance."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from(
        (u, v, abs(epi[u] - epi[v])) for u, v in combinations(nodes, 2)
    )
    return {_ordered_edge(u, v) for u, v in nx.minimum_spanning_edges(H, data=False)}


def _knn_edges(
    nodes: Sequence[Hashable],
    epi: Mapping[Hashable, float],
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
) -> set[RemeshEdge]:
    """Edges linking each node to its ``k`` nearest neighbours in EPI."""
    new_edges = set()
    node_set = set(nodes)
    for u in nodes:
        epi_u = epi[u]
        neighbours = [
            v
            for _, v in heapq.nsmallest(
                k_val,
                ((abs(epi_u - epi[v]), v) for v in nodes if v != u),
            )
        ]
        for v in neighbours:
            if rnd.random() < p_rewire:
                choices = list(node_set - {u, v})
                if choices:
                    v = rnd.choice(choices)
            new_edges.add(_ordered_edge(u, v))
    return new_edges


def _community_graph(
    comms: Iterable[Iterable[Hashable]],
    epi: Mapping[Hashable, float],
    nx: NetworkxModule,
) -> CommunityGraph:
    """Return community graph ``C`` with mean EPI per community."""
    C = nx.Graph()
    for idx, comm in enumerate(comms):
        members = list(comm)
        try:
            epi_mean = fmean(_as_float(epi.get(n)) for n in members)
        except StatisticsError:
            epi_mean = 0.0
        C.add_node(idx)
        set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
        C.nodes[idx]["members"] = members
    for i, j in combinations(C.nodes(), 2):
        w = abs(
            _as_float(get_attr(C.nodes[i], ALIAS_EPI, 0.0))
            - _as_float(get_attr(C.nodes[j], ALIAS_EPI, 0.0))
        )
        C.add_edge(i, j, weight=w)
    return cast(CommunityGraph, C)


def _community_k_neighbor_edges(
    C: CommunityGraph,
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
) -> tuple[set[RemeshEdge], dict[int, int], list[tuple[int, int, int]]]:
    """Edges linking each community to its ``k`` nearest neighbours."""
    epi_vals = {n: _as_float(get_attr(C.nodes[n], ALIAS_EPI, 0.0)) for n in C.nodes()}
    ordered = sorted(C.nodes(), key=lambda v: epi_vals[v])
    new_edges = set()
    attempts = {n: 0 for n in C.nodes()}
    rewired = []
    node_set = set(C.nodes())
    for idx, u in enumerate(ordered):
        epi_u = epi_vals[u]
        left = idx - 1
        right = idx + 1
        added = 0
        while added < k_val and (left >= 0 or right < len(ordered)):
            if left < 0:
                v = ordered[right]
                right += 1
            elif right >= len(ordered):
                v = ordered[left]
                left -= 1
            else:
                if abs(epi_u - epi_vals[ordered[left]]) <= abs(
                    epi_vals[ordered[right]] - epi_u
                ):
                    v = ordered[left]
                    left -= 1
                else:
                    v = ordered[right]
                    right += 1
            original_v = v
            rewired_now = False
            if rnd.random() < p_rewire:
                choices = list(node_set - {u, original_v})
                if choices:
                    v = rnd.choice(choices)
                    rewired_now = True
            new_edges.add(_ordered_edge(u, v))
            attempts[u] += 1
            if rewired_now:
                rewired.append((u, original_v, v))
            added += 1
    return new_edges, attempts, rewired


def _community_remesh(
    G: CommunityGraph,
    epi: Mapping[Hashable, float],
    k_val: int,
    p_rewire: float,
    rnd: random.Random,
    nx: NetworkxModule,
    nx_comm: CommunityModule,
    mst_edges: Iterable[RemeshEdge],
    n_before: int,
) -> None:
    """Remesh ``G`` replacing nodes by modular communities."""
    from ..glyph_history import append_metric

    comms = list(nx_comm.greedy_modularity_communities(G))
    if len(comms) <= 1:
        with edge_version_update(G):
            G.clear_edges()
            G.add_edges_from(mst_edges)
        return
    C = _community_graph(comms, epi, nx)
    mst_c = nx.minimum_spanning_tree(C, weight="weight")
    new_edges: set[RemeshEdge] = {_ordered_edge(u, v) for u, v in mst_c.edges()}
    extra_edges, attempts, rewired_edges = _community_k_neighbor_edges(
        C, k_val, p_rewire, rnd
    )
    new_edges |= extra_edges

    extra_degrees = {idx: 0 for idx in C.nodes()}
    for u, v in extra_edges:
        extra_degrees[u] += 1
        extra_degrees[v] += 1

    with edge_version_update(G):
        G.clear_edges()
        G.remove_nodes_from(list(G.nodes()))
        for idx in C.nodes():
            data = dict(C.nodes[idx])
            G.add_node(idx, **data)
        G.add_edges_from(new_edges)

    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        mapping = {idx: C.nodes[idx].get("members", []) for idx in C.nodes()}
        append_metric(
            hist,
            "remesh_events",
            {
                "mode": "community",
                "n_before": n_before,
                "n_after": G.number_of_nodes(),
                "mapping": mapping,
                "k": int(k_val),
                "p_rewire": float(p_rewire),
                "extra_edges_added": len(extra_edges),
                "extra_edge_attempts": attempts,
                "extra_edge_degrees": extra_degrees,
                "rewired_edges": [
                    {"source": int(u), "from": int(v0), "to": int(v1)}
                    for u, v0, v1 in rewired_edges
                ],
            },
        )


def apply_topological_remesh(
    G: CommunityGraph,
    mode: str | None = None,
    *,
    k: int | None = None,
    p_rewire: float = 0.2,
    seed: int | None = None,
) -> None:
    """Approximate topological remeshing.

    When ``seed`` is ``None`` the RNG draws its base seed from
    ``G.graph['RANDOM_SEED']`` to keep runs reproducible.
    """
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    if seed is None:
        base_seed = int(G.graph.get("RANDOM_SEED", 0))
    else:
        base_seed = int(seed)
    rnd = make_rng(base_seed, -2, G)

    if mode is None:
        mode = str(
            G.graph.get("REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn"))
        )
    mode = str(mode)
    nx, nx_comm = _get_networkx_modules()
    epi = {n: _as_float(get_attr(G.nodes[n], ALIAS_EPI, 0.0)) for n in nodes}
    mst_edges = _mst_edges_from_epi(nx, nodes, epi)
    default_k = int(
        G.graph.get("REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2))
    )
    k_val = max(1, int(k) if k is not None else default_k)

    if mode == "community":
        _community_remesh(
            G,
            epi,
            k_val,
            p_rewire,
            rnd,
            nx,
            nx_comm,
            mst_edges,
            n_before,
        )
        return

    new_edges = set(mst_edges)
    if mode == "knn":
        new_edges |= _knn_edges(nodes, epi, k_val, p_rewire, rnd)

    with edge_version_update(G):
        G.clear_edges()
        G.add_edges_from(new_edges)


def _extra_gating_ok(
    hist: MutableMapping[str, Sequence[float]],
    cfg: Mapping[str, RemeshConfigValue],
    w_estab: int,
) -> bool:
    """Check additional stability gating conditions."""
    checks = [
        ("phase_sync", "REMESH_MIN_PHASE_SYNC", ge),
        ("glyph_load_disr", "REMESH_MAX_GLYPH_DISR", le),
        ("sense_sigma_mag", "REMESH_MIN_SIGMA_MAG", ge),
        ("kuramoto_R", "REMESH_MIN_KURAMOTO_R", ge),
        ("Si_hi_frac", "REMESH_MIN_SI_HI_FRAC", ge),
    ]
    for hist_key, cfg_key, op in checks:
        series = hist.get(hist_key)
        if series is not None and len(series) >= w_estab:
            win = series[-w_estab:]
            avg = sum(win) / len(win)
            threshold = _as_float(cfg[cfg_key])
            if not op(avg, threshold):
                return False
    return True


def apply_remesh_if_globally_stable(
    G: CommunityGraph,
    stable_step_window: int | None = None,
    **kwargs: Any,
) -> None:
    """Trigger remeshing when global stability indicators satisfy thresholds."""

    from ..glyph_history import ensure_history

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "apply_remesh_if_globally_stable() got unexpected keyword argument(s): "
            f"{unexpected}"
        )

    params = [
        (
            "REMESH_STABILITY_WINDOW",
            int,
            REMESH_DEFAULTS["REMESH_STABILITY_WINDOW"],
        ),
        (
            "REMESH_REQUIRE_STABILITY",
            bool,
            REMESH_DEFAULTS["REMESH_REQUIRE_STABILITY"],
        ),
        (
            "REMESH_MIN_PHASE_SYNC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_PHASE_SYNC"],
        ),
        (
            "REMESH_MAX_GLYPH_DISR",
            float,
            REMESH_DEFAULTS["REMESH_MAX_GLYPH_DISR"],
        ),
        (
            "REMESH_MIN_SIGMA_MAG",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SIGMA_MAG"],
        ),
        (
            "REMESH_MIN_KURAMOTO_R",
            float,
            REMESH_DEFAULTS["REMESH_MIN_KURAMOTO_R"],
        ),
        (
            "REMESH_MIN_SI_HI_FRAC",
            float,
            REMESH_DEFAULTS["REMESH_MIN_SI_HI_FRAC"],
        ),
        (COOLDOWN_KEY, int, REMESH_DEFAULTS[COOLDOWN_KEY]),
        ("REMESH_COOLDOWN_TS", float, REMESH_DEFAULTS["REMESH_COOLDOWN_TS"]),
    ]
    cfg = {}
    for key, conv, _default in params:
        cfg[key] = conv(get_param(G, key))
    frac_req = _as_float(get_param(G, "FRACTION_STABLE_REMESH"))
    w_estab = (
        stable_step_window
        if stable_step_window is not None
        else cfg["REMESH_STABILITY_WINDOW"]
    )

    hist = ensure_history(G)
    sf = hist.setdefault("stable_frac", [])
    if len(sf) < w_estab:
        return
    win_sf = sf[-w_estab:]
    if not all(v >= frac_req for v in win_sf):
        return
    if cfg["REMESH_REQUIRE_STABILITY"] and not _extra_gating_ok(hist, cfg, w_estab):
        return

    last = G.graph.get("_last_remesh_step", -(10**9))
    step_idx = len(sf)
    if step_idx - last < cfg[COOLDOWN_KEY]:
        return
    t_now = _as_float(G.graph.get("_t", 0.0))
    last_ts = _as_float(G.graph.get("_last_remesh_ts", -1e12))
    if cfg["REMESH_COOLDOWN_TS"] > 0 and (t_now - last_ts) < cfg["REMESH_COOLDOWN_TS"]:
        return

    apply_network_remesh(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now


__all__ = [
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
]
