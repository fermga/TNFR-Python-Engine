"""Coherence metrics."""

from __future__ import annotations

import math
from typing import Dict

from ..constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_SI, COHERENCE
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, append_metric
from ..alias import get_attr
from ..collections_utils import normalize_weights
from ..helpers import clamp01, ensure_node_index_map
from ..import_utils import get_numpy


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def _similarity_abs(a, b, lo, hi):
    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def _coherence_components(
    th_vals,
    epi_vals,
    vf_vals,
    si_vals,
    ni,
    nj,
    epi_min,
    epi_max,
    vf_min,
    vf_max,
):
    th_i = th_vals[ni]
    th_j = th_vals[nj]
    s_phase = 0.5 * (1.0 + math.cos(th_i - th_j))
    s_epi = _similarity_abs(epi_vals[ni], epi_vals[nj], epi_min, epi_max)
    s_vf = _similarity_abs(vf_vals[ni], vf_vals[nj], vf_min, vf_max)
    s_si = 1.0 - abs(si_vals[ni] - si_vals[nj])
    return s_phase, s_epi, s_vf, s_si


def _combine_components(
    wnorm: Dict[str, float],
    th_vals,
    epi_vals,
    vf_vals,
    si_vals,
    ni,
    nj,
    epi_min,
    epi_max,
    vf_min,
    vf_max,
):
    """Compute coherence by combining components with their weights."""
    s_phase, s_epi, s_vf, s_si = _coherence_components(
        th_vals,
        epi_vals,
        vf_vals,
        si_vals,
        ni,
        nj,
        epi_min,
        epi_max,
        vf_min,
        vf_max,
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
    node_to_index = ensure_node_index_map(G)

    th_vals = [get_attr(G.nodes[v], ALIAS_THETA, 0.0) for v in nodes]
    epi_vals = [get_attr(G.nodes[v], ALIAS_EPI, 0.0) for v in nodes]
    vf_vals = [get_attr(G.nodes[v], ALIAS_VF, 0.0) for v in nodes]
    si_vals = [clamp01(get_attr(G.nodes[v], ALIAS_SI, 0.0)) for v in nodes]
    epi_min, epi_max = min(epi_vals), max(epi_vals)
    vf_min, vf_max = min(vf_vals), max(vf_vals)

    wdict = dict(cfg.get("weights", {}))
    for k in ("phase", "epi", "vf", "si"):
        wdict.setdefault(k, 0.0)
    wnorm = normalize_weights(wdict, ("phase", "epi", "vf", "si"), default=0.0)

    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"
    self_diag = bool(cfg.get("self_on_diag", True))
    mode = str(cfg.get("store_mode", "sparse")).lower()
    thr = float(cfg.get("threshold", 0.0))
    if mode not in ("sparse", "dense"):
        mode = "sparse"

    np = get_numpy()
    if np is not None and not neighbors_only:
        th = np.array(th_vals)
        epi = np.array(epi_vals)
        vf = np.array(vf_vals)
        si = np.array(si_vals)
        th_diff = th[:, None] - th[None, :]
        s_phase = 0.5 * (1.0 + np.cos(th_diff))
        epi_range = epi_max - epi_min if epi_max > epi_min else 1.0
        vf_range = vf_max - vf_min if vf_max > vf_min else 1.0
        s_epi = 1.0 - np.abs(epi[:, None] - epi[None, :]) / epi_range
        s_vf = 1.0 - np.abs(vf[:, None] - vf[None, :]) / vf_range
        s_si = 1.0 - np.abs(si[:, None] - si[None, :])
        wij = (
            wnorm["phase"] * s_phase
            + wnorm["epi"] * s_epi
            + wnorm["vf"] * s_vf
            + wnorm["si"] * s_si
        )
        wij = np.clip(wij, 0.0, 1.0)
        if self_diag:
            np.fill_diagonal(wij, 1.0)
        else:
            np.fill_diagonal(wij, 0.0)

        row_sum = wij.sum(axis=1)
        row_count = np.full(n, n if self_diag else n - 1)

        if mode == "dense":
            W = wij.tolist()
            mask = ~np.eye(n, dtype=bool)
        else:
            mask = (wij >= thr) & (~np.eye(n, dtype=bool))
            idx = np.where(wij >= thr)
            W = [
                (int(i), int(j), float(wij[i, j]))
                for i, j in zip(idx[0], idx[1])
            ]

        values = wij[mask]
        min_val = float(values.min()) if values.size else 0.0
        max_val = float(values.max()) if values.size else 0.0
        sum_val = float(values.sum())
        count_val = int(values.size)

        Wi = [float(row_sum[i] / max(1, row_count[i])) for i in range(n)]
        stats = {
            "min": min_val,
            "max": max_val,
            "mean": (sum_val / count_val) if count_val else 0.0,
            "n_edges": count_val,
            "mode": mode,
            "scope": scope,
        }

        hist = ensure_history(G)
        append_metric(hist, cfg.get("history_key", "W_sparse"), W)
        append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
        append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)
        return nodes, W

    if mode == "dense":
        W = [[0.0] * n for _ in range(n)]
    else:
        W = []

    row_sum = [0.0] * n
    row_count = [0] * n

    # Accumulators for statistics over non-diagonal stored entries
    min_val = float("inf")
    max_val = float("-inf")
    sum_val = 0.0
    count_val = 0

    def _accumulate(i: int, j: int, w: float) -> None:
        nonlocal min_val, max_val, sum_val, count_val
        if i == j:
            return
        if w < min_val:
            min_val = w
        if w > max_val:
            max_val = w
        sum_val += w
        count_val += 1

    def add_entry(i: int, j: int, w: float) -> None:
        """Add a value to the matrix and accumulate sums/counters."""
        if mode == "dense":
            W[i][j] = w
            _accumulate(i, j, w)
        else:
            if w >= thr:
                W.append((i, j, w))
                _accumulate(i, j, w)
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
                wnorm,
                th_vals,
                epi_vals,
                vf_vals,
                si_vals,
                i,
                j,
                epi_min,
                epi_max,
                vf_min,
                vf_max,
            )
            add_entry(i, j, wij)
            add_entry(j, i, wij)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                wij = _combine_components(
                    wnorm,
                    th_vals,
                    epi_vals,
                    vf_vals,
                    si_vals,
                    i,
                    j,
                    epi_min,
                    epi_max,
                    vf_min,
                    vf_max,
                )
                add_entry(i, j, wij)
                add_entry(j, i, wij)
    Wi = [row_sum[i] / max(1, row_count[i]) for i in range(n)]

    stats = {
        "min": min_val if count_val else 0.0,
        "max": max_val if count_val else 0.0,
        "mean": (sum_val / count_val) if count_val else 0.0,
        "n_edges": count_val,
        "mode": mode,
        "scope": scope,
    }

    hist = ensure_history(G)
    append_metric(hist, cfg.get("history_key", "W_sparse"), W)
    append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
    append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)

    return nodes, W


def local_phase_sync_weighted(
    G, n, nodes_order=None, W_row=None, node_to_index=None
):
    cfg = G.graph.get("COHERENCE", COHERENCE)
    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"

    # --- Caso sin pesos ---
    if W_row is None or nodes_order is None:
        vec = [
            complex(
                math.cos(get_attr(G.nodes[v], ALIAS_THETA, 0.0)),
                math.sin(get_attr(G.nodes[v], ALIAS_THETA, 0.0)),
            )
            for v in (
                G.neighbors(n) if neighbors_only else (set(G.nodes()) - {n})
            )
        ]
        if not vec:
            return 0.0
        mean = sum(vec) / len(vec)
        return abs(mean)

    # --- Mapeo nodo → índice ---
    if node_to_index is None:
        node_to_index = ensure_node_index_map(G)

    i = node_to_index.get(n, None)
    if i is None:
        i = nodes_order.index(n)

    if (
        isinstance(W_row, list)
        and W_row
        and isinstance(W_row[0], (int, float))
    ):
        weights = W_row
    else:
        weights = [0.0] * len(nodes_order)
        for ii, jj, w in W_row:
            if ii == i:
                weights[jj] = w

    num = 0 + 0j
    den = 0.0
    for j, nj in enumerate(nodes_order):
        if nj == n:
            continue
        w = weights[j]
        den += w
        th_j = get_attr(G.nodes[nj], ALIAS_THETA, 0.0)
        num += w * complex(math.cos(th_j), math.sin(th_j))
    return abs(num / den) if den else 0.0


def _coherence_step(G, ctx=None):
    if not G.graph.get("COHERENCE", COHERENCE).get("enabled", True):
        return
    coherence_matrix(G)


def register_coherence_callbacks(G) -> None:
    register_callback(
        G, event="after_step", func=_coherence_step, name="coherence_step"
    )
