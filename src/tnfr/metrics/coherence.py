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
from ..metrics_utils import min_max_range
from ..import_utils import get_numpy


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def _similarity_abs(a, b, lo, hi):
    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def compute_components(
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
) -> Dict[str, float]:
    th_i = th_vals[ni]
    th_j = th_vals[nj]
    s_phase = 0.5 * (1.0 + math.cos(th_i - th_j))
    s_epi = _similarity_abs(epi_vals[ni], epi_vals[nj], epi_min, epi_max)
    s_vf = _similarity_abs(vf_vals[ni], vf_vals[nj], vf_min, vf_max)
    s_si = 1.0 - abs(si_vals[ni] - si_vals[nj])
    return {
        "s_phase": s_phase,
        "s_epi": s_epi,
        "s_vf": s_vf,
        "s_si": s_si,
    }


def _wij_vectorized(
    th_vals,
    epi_vals,
    vf_vals,
    si_vals,
    wnorm,
    epi_min,
    epi_max,
    vf_min,
    vf_max,
    self_diag,
    np,
):
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
    return wij


def _wij_loops(
    G,
    nodes,
    node_to_index,
    th_vals,
    epi_vals,
    vf_vals,
    si_vals,
    wnorm,
    epi_min,
    epi_max,
    vf_min,
    vf_max,
    neighbors_only,
    self_diag,
):
    n = len(nodes)
    wij = [[1.0 if (self_diag and i == j) else 0.0 for j in range(n)] for i in range(n)]
    phase_w = wnorm["phase"]
    epi_w = wnorm["epi"]
    vf_w = wnorm["vf"]
    si_w = wnorm["si"]
    def assign_wij(i: int, j: int) -> None:
        comps = compute_components(
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
        s_phase = comps["s_phase"]
        s_epi = comps["s_epi"]
        s_vf = comps["s_vf"]
        s_si = comps["s_si"]
        wij_ij = clamp01(
            phase_w * s_phase + epi_w * s_epi + vf_w * s_vf + si_w * s_si
        )
        wij[i][j] = wij[j][i] = wij_ij

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
            assign_wij(i, j)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                assign_wij(i, j)
    return wij


def _stats_numpy(wij, self_diag, np):
    n = wij.shape[0]
    mask = ~np.eye(n, dtype=bool)
    values = wij[mask]
    min_val = float(values.min()) if values.size else 0.0
    max_val = float(values.max()) if values.size else 0.0
    mean_val = float(values.mean()) if values.size else 0.0
    row_sum = wij.sum(axis=1)
    row_count = np.full(n, n if self_diag else n - 1)
    Wi = [float(row_sum[i] / max(1, row_count[i])) for i in range(n)]
    count_val = int(values.size)
    return min_val, max_val, mean_val, Wi, count_val, mask


def _stats_from_lists(values, row_sum, n, self_diag):
    min_val = min(values) if values else 0.0
    max_val = max(values) if values else 0.0
    sum_val = sum(values)
    count_val = len(values)
    mean_val = (sum_val / count_val) if count_val else 0.0
    row_count = n if self_diag else n - 1
    Wi = [row_sum[i] / max(1, row_count) for i in range(n)]
    return min_val, max_val, mean_val, Wi, count_val


def _finalize_wij_numpy(wij, mode, thr, self_diag, np):
    min_val, max_val, mean_val, Wi, count_val, mask = _stats_numpy(
        wij, self_diag, np
    )
    if mode == "dense":
        W = wij.tolist()
    else:
        idx = np.where((wij >= thr) & mask)
        W = [
            (int(i), int(j), float(wij[i, j]))
            for i, j in zip(idx[0], idx[1])
        ]
    stats = {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "n_edges": count_val,
    }
    return W, Wi, stats


def _finalize_wij_python(wij, mode, thr, self_diag):
    n = len(wij)
    values: list[float] = []
    row_sum = [0.0] * n
    if mode == "dense":
        W = [row[:] for row in wij]
        for i in range(n):
            for j in range(n):
                w = W[i][j]
                if i != j:
                    values.append(w)
                row_sum[i] += w
    else:
        W = []
        for i in range(n):
            row_i = wij[i]
            for j in range(n):
                w = row_i[j]
                if i != j:
                    values.append(w)
                    if w >= thr:
                        W.append((i, j, w))
                row_sum[i] += w
    min_val, max_val, mean_val, Wi, count_val = _stats_from_lists(
        values, row_sum, n, self_diag
    )
    stats = {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "n_edges": count_val,
    }
    return W, Wi, stats


def _finalize_wij(G, nodes, wij, mode, thr, scope, self_diag, np):
    if np is not None and isinstance(wij, np.ndarray):
        W, Wi, stats = _finalize_wij_numpy(wij, mode, thr, self_diag, np)
    else:
        W, Wi, stats = _finalize_wij_python(wij, mode, thr, self_diag)
    stats["mode"] = mode
    stats["scope"] = scope

    hist = ensure_history(G)
    cfg = G.graph.get("COHERENCE", COHERENCE)
    append_metric(hist, cfg.get("history_key", "W_sparse"), W)
    append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
    append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)
    return nodes, W


def coherence_matrix(G):
    cfg = G.graph.get("COHERENCE", COHERENCE)
    if not cfg.get("enabled", True):
        return None, None

    node_to_index = ensure_node_index_map(G)
    nodes = list(node_to_index.keys())
    n = len(nodes)
    if n == 0:
        return nodes, []

    # Precompute indices to avoid repeated list.index calls within loops

    th_vals = [get_attr(G.nodes[v], ALIAS_THETA, 0.0) for v in nodes]
    epi_vals = [get_attr(G.nodes[v], ALIAS_EPI, 0.0) for v in nodes]
    vf_vals = [get_attr(G.nodes[v], ALIAS_VF, 0.0) for v in nodes]
    si_vals = [clamp01(get_attr(G.nodes[v], ALIAS_SI, 0.0)) for v in nodes]
    epi_min, epi_max = min_max_range(epi_vals)
    vf_min, vf_max = min_max_range(vf_vals)

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
        wij = _wij_vectorized(
            th_vals,
            epi_vals,
            vf_vals,
            si_vals,
            wnorm,
            epi_min,
            epi_max,
            vf_min,
            vf_max,
            self_diag,
            np,
        )
    else:
        wij = _wij_loops(
            G,
            nodes,
            node_to_index,
            th_vals,
            epi_vals,
            vf_vals,
            si_vals,
            wnorm,
            epi_min,
            epi_max,
            vf_min,
            vf_max,
            neighbors_only,
            self_diag,
        )

    return _finalize_wij(G, nodes, wij, mode, thr, scope, self_diag, np)


def local_phase_sync(G, n):
    cfg = G.graph.get("COHERENCE", COHERENCE)
    scope = str(cfg.get("scope", "neighbors")).lower()
    neighbors_only = scope != "all"
    targets = (
        G.neighbors(n)
        if neighbors_only
        else (v for v in G.nodes() if v != n)
    )
    vec = [
        complex(
            math.cos(get_attr(G.nodes[v], ALIAS_THETA, 0.0)),
            math.sin(get_attr(G.nodes[v], ALIAS_THETA, 0.0)),
        )
        for v in targets
    ]
    if not vec:
        return 0.0
    mean = sum(vec) / len(vec)
    return abs(mean)

def local_phase_sync_weighted(G, n, nodes_order=None, W_row=None, node_to_index=None):
    """Compute local phase synchrony using explicit weights.

    ``nodes_order`` is the node ordering used to build the coherence matrix and
    ``W_row`` contains either the dense row corresponding to ``n`` or the sparse
    list of ``(i, j, w)`` tuples for the whole matrix.
    """
    if W_row is None or nodes_order is None:
        raise ValueError(
            "nodes_order and W_row are required for weighted phase synchrony"
        )

    if node_to_index is None:
        node_to_index = ensure_node_index_map(G)
    i = node_to_index.get(n)
    if i is None:
        i = nodes_order.index(n)

    num = 0 + 0j
    den = 0.0

    if isinstance(W_row, list) and W_row and isinstance(W_row[0], (int, float)):
        for w, nj in zip(W_row, nodes_order):
            if nj == n:
                continue
            den += w
            th_j = get_attr(G.nodes[nj], ALIAS_THETA, 0.0)
            num += w * complex(math.cos(th_j), math.sin(th_j))
    else:
        for ii, jj, w in W_row:
            if ii != i:
                continue
            nj = nodes_order[jj]
            if nj == n:
                continue
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
