"""Métricas de coherencia."""
from __future__ import annotations

from math import cos
import cmath
from typing import Dict, List, Tuple

from ..constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_SI, COHERENCE
from ..helpers import register_callback, ensure_history, get_attr, clamp01


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    v = (float(x) - float(lo)) / (float(hi) - float(lo))
    return 0.0 if v < 0 else (1.0 if v > 1.0 else v)


def _similarity_abs(a, b, lo, hi):
    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, float(hi - lo) if hi > lo else 1.0)


def _coherence_components(G, ni, nj, epi_min, epi_max, vf_min, vf_max):
    ndi = G.nodes[ni]
    ndj = G.nodes[nj]
    th_i = get_attr(ndi, ALIAS_THETA, 0.0)
    th_j = get_attr(ndj, ALIAS_THETA, 0.0)
    s_phase = 0.5 * (1.0 + cos(th_i - th_j))
    epi_i = get_attr(ndi, ALIAS_EPI, 0.0)
    epi_j = get_attr(ndj, ALIAS_EPI, 0.0)
    s_epi = _similarity_abs(epi_i, epi_j, epi_min, epi_max)
    vf_i = get_attr(ndi, ALIAS_VF, 0.0)
    vf_j = get_attr(ndj, ALIAS_VF, 0.0)
    s_vf = _similarity_abs(vf_i, vf_j, vf_min, vf_max)
    si_i = clamp01(get_attr(ndi, ALIAS_SI, 0.0))
    si_j = clamp01(get_attr(ndj, ALIAS_SI, 0.0))
    s_si = 1.0 - abs(si_i - si_j)
    return s_phase, s_epi, s_vf, s_si


def _combine_components(
    wnorm: Dict[str, float],
    G,
    ni,
    nj,
    epi_min,
    epi_max,
    vf_min,
    vf_max,
):
    """Calcula la coherencia combinando componentes con sus pesos."""
    s_phase, s_epi, s_vf, s_si = _coherence_components(
        G, ni, nj, epi_min, epi_max, vf_min, vf_max
    )
    wij = (
        wnorm["phase"] * s_phase
        + wnorm["epi"] * s_epi
        + wnorm["vf"] * s_vf
        + wnorm["si"] * s_si
    )
    return clamp01(wij)


def coherence_matrix(G):
    cfg = G.graph.get("COHERENCE", COHERENCE)
    if not cfg.get("enabled", True):
        return None, None

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return nodes, []

    # Precompute indices to avoid repeated list.index calls within loops
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    epi_vals = [float(get_attr(G.nodes[v], ALIAS_EPI, 0.0)) for v in nodes]
    vf_vals = [get_attr(G.nodes[v], ALIAS_VF, 0.0) for v in nodes]
    epi_min, epi_max = min(epi_vals), max(epi_vals)
    vf_min, vf_max = min(vf_vals), max(vf_vals)

    wdict = dict(cfg.get("weights", {}))
    for k in ("phase", "epi", "vf", "si"):
        wdict.setdefault(k, 0.0)
    wsum = sum(float(v) for v in wdict.values()) or 1.0
    wnorm = {k: float(v) / wsum for k, v in wdict.items()}

    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"
    self_diag = bool(cfg.get("self_on_diag", True))
    mode = str(cfg.get("store_mode", "sparse")).lower()
    thr = float(cfg.get("threshold", 0.0))
    if mode not in ("sparse", "dense"):
        mode = "sparse"

    if mode == "dense":
        W = [[0.0] * n for _ in range(n)]
    else:
        W = []

    row_sum = [0.0] * n
    row_count = [0] * n

    def add_entry(i: int, j: int, w: float) -> None:
        """Añade un valor a la matriz y acumula sumas/contadores."""
        if mode == "dense":
            W[i][j] = w
        else:
            if w >= thr:
                W.append((i, j, w))
        row_sum[i] += w
        row_count[i] += 1

    # Diagonal de unos si corresponde
    if self_diag:
        for i in range(n):
            add_entry(i, i, 1.0)

    if neighbors_only:
        seen: set[tuple[int, int]] = set()
        for u, v in G.edges():
            i = node_to_index[u]
            j = node_to_index[v]
            if i == j:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            wij = _combine_components(
                wnorm, G, u, v, epi_min, epi_max, vf_min, vf_max
            )
            add_entry(i, j, wij)
            add_entry(j, i, wij)
    else:
        for i in range(n):
            ni = nodes[i]
            for j in range(i + 1, n):
                nj = nodes[j]
                wij = _combine_components(
                    wnorm, G, ni, nj, epi_min, epi_max, vf_min, vf_max
                )
                add_entry(i, j, wij)
                add_entry(j, i, wij)

    Wi = [row_sum[i] / max(1, row_count[i]) for i in range(n)]
    vals = []
    if mode == "dense":
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vals.append(W[i][j])
    else:
        for (i, j, w) in W:
            if i == j:
                continue
            vals.append(w)

    stats = {
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
        "mean": (sum(vals) / len(vals)) if vals else 0.0,
        "n_edges": len(vals),
        "mode": mode,
        "scope": scope,
    }

    hist = ensure_history(G)
    hist.setdefault(cfg.get("history_key", "W_sparse"), []).append(W)
    hist.setdefault(cfg.get("Wi_history_key", "W_i"), []).append(Wi)
    hist.setdefault(cfg.get("stats_history_key", "W_stats"), []).append(stats)

    return nodes, W


def local_phase_sync_weighted(G, n, nodes_order=None, W_row=None, node_to_index=None):
    cfg = G.graph.get("COHERENCE", COHERENCE)
    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"

    # --- Caso sin pesos ---
    if W_row is None or nodes_order is None:
        vec = [
            cmath.exp(1j * float(get_attr(G.nodes[v], ALIAS_THETA, 0.0)))
            for v in (G.neighbors(n) if neighbors_only else (set(G.nodes()) - {n}))
        ]
        if not vec:
            return 0.0
        mean = sum(vec) / len(vec)
        return abs(mean)

    # --- Mapeo nodo → índice ---
    if node_to_index is None:
        cache_nodes = getattr(local_phase_sync_weighted, "_cache_nodes", None)
        cache_map = getattr(local_phase_sync_weighted, "_cache_map", None)
        if cache_nodes is not nodes_order:
            node_to_index = {v: i for i, v in enumerate(nodes_order)}
            local_phase_sync_weighted._cache_nodes = nodes_order
            local_phase_sync_weighted._cache_map = node_to_index
        else:
            node_to_index = cache_map

    i = node_to_index.get(n, None)
    if i is None:
        i = nodes_order.index(n)

    if isinstance(W_row, list) and W_row and isinstance(W_row[0], (int, float)):
        weights = W_row
    else:
        weights = [0.0] * len(nodes_order)
        for (ii, jj, w) in W_row:
            if ii == i:
                weights[jj] = w

    num = 0 + 0j
    den = 0.0
    for j, nj in enumerate(nodes_order):
        if nj == n:
            continue
        w = weights[j]
        den += w
        th_j = float(get_attr(G.nodes[nj], ALIAS_THETA, 0.0))
        num += w * cmath.exp(1j * th_j)
    return abs(num / den) if den else 0.0


def _coherence_step(G, ctx=None):
    if not G.graph.get("COHERENCE", COHERENCE).get("enabled", True):
        return
    coherence_matrix(G)


def register_coherence_callbacks(G) -> None:
    register_callback(G, event="after_step", func=_coherence_step, name="coherence_step")
