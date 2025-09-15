from __future__ import annotations
import hashlib
import heapq
from operator import ge, le
from functools import cache
from itertools import combinations
from io import StringIO
from collections import deque
from statistics import fmean, StatisticsError

from ..constants import DEFAULTS, REMESH_DEFAULTS, get_aliases, get_param
from ..helpers.numeric import kahan_sum_nd
from ..helpers import edge_version_update
from ..alias import get_attr, set_attr
from ..rng import make_rng
from ..callback_utils import callback_manager
from ..glyph_history import append_metric, ensure_history, current_step_idx
from ..import_utils import cached_import

ALIAS_EPI = get_aliases("EPI")


@cache
def _get_networkx_modules():
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
    return nx, nx_comm


def _remesh_alpha_info(G):
    """Return ``(alpha, source)`` with explicit precedence."""
    if bool(
        G.graph.get("REMESH_ALPHA_HARD", REMESH_DEFAULTS["REMESH_ALPHA_HARD"])
    ):
        val = float(
            G.graph.get("REMESH_ALPHA", REMESH_DEFAULTS["REMESH_ALPHA"])
        )
        return val, "REMESH_ALPHA"
    gf = G.graph.get("GLYPH_FACTORS", DEFAULTS.get("GLYPH_FACTORS", {}))
    if "REMESH_alpha" in gf:
        return float(gf["REMESH_alpha"]), "GLYPH_FACTORS.REMESH_alpha"
    if "REMESH_ALPHA" in G.graph:
        return float(G.graph["REMESH_ALPHA"]), "REMESH_ALPHA"
    return (
        float(REMESH_DEFAULTS["REMESH_ALPHA"]),
        "REMESH_DEFAULTS.REMESH_ALPHA",
    )


def _snapshot_topology(G, nx):
    """Return a hash representing the current graph topology."""
    try:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degs = sorted(d for _, d in G.degree())
        topo_str = f"n={n_nodes};m={n_edges};deg=" + ",".join(map(str, degs))
        return hashlib.sha1(topo_str.encode()).hexdigest()[:12]
    except (AttributeError, TypeError, nx.NetworkXError):
        return None


def _snapshot_epi(G):
    """Return ``(mean, checksum)`` of the node EPI values."""
    buf = StringIO()
    values = []
    for n, data in G.nodes(data=True):
        v = float(get_attr(data, ALIAS_EPI, 0.0))
        values.append(v)
        buf.write(f"{str(n)}:{round(v, 6)};")
    total = kahan_sum_nd(((v,) for v in values), dims=1)[0]
    mean_val = total / len(values) if values else 0.0
    checksum = hashlib.sha1(buf.getvalue().encode()).hexdigest()[:12]
    return float(mean_val), checksum


def _log_remesh_event(G, meta):
    """Store remesh metadata and optionally log and trigger callbacks."""
    G.graph["_REMESH_META"] = meta
    if G.graph.get("REMESH_LOG_EVENTS", REMESH_DEFAULTS["REMESH_LOG_EVENTS"]):
        hist = G.graph.setdefault("history", {})
        append_metric(hist, "remesh_events", dict(meta))
    callback_manager.invoke_callbacks(G, "on_remesh", dict(meta))


def apply_network_remesh(G) -> None:
    """Network-scale REMESH using ``_epi_hist`` with multi-scale memory."""
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

    for n, nd in G.nodes(data=True):
        epi_now = get_attr(nd, ALIAS_EPI, 0.0)
        epi_old_l = float(past_l.get(n, epi_now))
        epi_old_g = float(past_g.get(n, epi_now))
        mixed = (1 - alpha) * epi_now + alpha * epi_old_l
        mixed = (1 - alpha) * mixed + alpha * epi_old_g
        set_attr(nd, ALIAS_EPI, mixed)

    epi_mean_after, epi_checksum_after = _snapshot_epi(G)

    step_idx = current_step_idx(G)
    meta = {
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


def _mst_edges_from_epi(nx, nodes, epi):
    """Return MST edges based on absolute EPI distance."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from(
        (u, v, abs(epi[u] - epi[v])) for u, v in combinations(nodes, 2)
    )
    return {
        tuple(sorted((u, v)))
        for u, v in nx.minimum_spanning_edges(H, data=False)
    }


def _knn_edges(nodes, epi, k_val, p_rewire, rnd):
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
            new_edges.add(tuple(sorted((u, v))))
    return new_edges


def _community_graph(comms, epi, nx):
    """Return community graph ``C`` with mean EPI per community."""
    C = nx.Graph()
    for idx, comm in enumerate(comms):
        members = list(comm)
        try:
            epi_mean = fmean(epi[n] for n in members)
        except StatisticsError:
            epi_mean = 0.0
        C.add_node(idx)
        set_attr(C.nodes[idx], ALIAS_EPI, epi_mean)
        C.nodes[idx]["members"] = members
    for i, j in combinations(C.nodes(), 2):
        w = abs(
            get_attr(C.nodes[i], ALIAS_EPI, 0.0)
            - get_attr(C.nodes[j], ALIAS_EPI, 0.0)
        )
        C.add_edge(i, j, weight=w)
    return C


def _community_k_neighbor_edges(C, k_val, p_rewire, rnd):
    """Edges linking each community to its ``k`` nearest neighbours."""
    epi_vals = {n: get_attr(C.nodes[n], ALIAS_EPI, 0.0) for n in C.nodes()}
    ordered = sorted(C.nodes(), key=lambda v: epi_vals[v])
    new_edges = set()
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
            if rnd.random() < p_rewire:
                new_edges.add(tuple(sorted((u, v))))
            added += 1
    return new_edges


def _community_remesh(
    G,
    epi,
    k_val,
    p_rewire,
    rnd,
    nx,
    nx_comm,
    mst_edges,
    n_before,
):
    """Remesh ``G`` replacing nodes by modular communities."""
    comms = list(nx_comm.greedy_modularity_communities(G))
    if len(comms) <= 1:
        with edge_version_update(G):
            G.clear_edges()
            G.add_edges_from(mst_edges)
        return
    C = _community_graph(comms, epi, nx)
    mst_c = nx.minimum_spanning_tree(C, weight="weight")
    new_edges = set(mst_c.edges())
    new_edges |= _community_k_neighbor_edges(C, k_val, p_rewire, rnd)

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
            },
        )


def apply_topological_remesh(
    G,
    mode: str | None = None,
    *,
    k: int | None = None,
    p_rewire: float = 0.2,
    seed: int | None = None,
) -> None:
    """Approximate topological remeshing."""
    nodes = list(G.nodes())
    n_before = len(nodes)
    if n_before <= 1:
        return
    base_seed = 0 if seed is None else int(seed)
    rnd = make_rng(base_seed, -2, G)
    rnd.seed(base_seed)

    if mode is None:
        mode = str(
            G.graph.get(
                "REMESH_MODE", REMESH_DEFAULTS.get("REMESH_MODE", "knn")
            )
        )
    mode = str(mode)
    nx, nx_comm = _get_networkx_modules()
    epi = {n: get_attr(G.nodes[n], ALIAS_EPI, 0.0) for n in nodes}
    mst_edges = _mst_edges_from_epi(nx, nodes, epi)
    default_k = int(
        G.graph.get(
            "REMESH_COMMUNITY_K", REMESH_DEFAULTS.get("REMESH_COMMUNITY_K", 2)
        )
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


def _extra_gating_ok(hist, cfg, w_estab):
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
            if not op(avg, cfg[cfg_key]):
                return False
    return True


def apply_remesh_if_globally_stable(
    G, pasos_estables_consecutivos: int | None = None
) -> None:
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
        (
            "REMESH_COOLDOWN_VENTANA",
            int,
            REMESH_DEFAULTS["REMESH_COOLDOWN_VENTANA"],
        ),
        ("REMESH_COOLDOWN_TS", float, REMESH_DEFAULTS["REMESH_COOLDOWN_TS"]),
    ]
    cfg = {}
    for key, conv, _default in params:
        cfg[key] = conv(get_param(G, key))
    frac_req = float(get_param(G, "FRACTION_STABLE_REMESH"))
    w_estab = (
        pasos_estables_consecutivos
        if pasos_estables_consecutivos is not None
        else cfg["REMESH_STABILITY_WINDOW"]
    )

    hist = ensure_history(G)
    sf = hist.setdefault("stable_frac", [])
    if len(sf) < w_estab:
        return
    win_sf = sf[-w_estab:]
    if not all(v >= frac_req for v in win_sf):
        return
    if cfg["REMESH_REQUIRE_STABILITY"] and not _extra_gating_ok(
        hist, cfg, w_estab
    ):
        return

    last = G.graph.get("_last_remesh_step", -(10**9))
    step_idx = len(sf)
    if step_idx - last < cfg["REMESH_COOLDOWN_VENTANA"]:
        return
    t_now = float(G.graph.get("_t", 0.0))
    last_ts = float(G.graph.get("_last_remesh_ts", -1e12))
    if (
        cfg["REMESH_COOLDOWN_TS"] > 0
        and (t_now - last_ts) < cfg["REMESH_COOLDOWN_TS"]
    ):
        return

    apply_network_remesh(G)
    G.graph["_last_remesh_step"] = step_idx
    G.graph["_last_remesh_ts"] = t_now


__all__ = [
    "apply_network_remesh",
    "apply_topological_remesh",
    "apply_remesh_if_globally_stable",
]
