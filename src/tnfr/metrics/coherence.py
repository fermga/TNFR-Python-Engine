"""Coherence metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence


from ..constants import ALIAS_THETA, ALIAS_EPI, ALIAS_VF, ALIAS_SI, COHERENCE
from ..callback_utils import register_callback
from ..glyph_history import ensure_history, append_metric
from ..alias import get_attr
from ..collections_utils import normalize_weights
from ..helpers.numeric import clamp01
from ..helpers.cache import ensure_node_index_map
from ..metrics_utils import min_max_range
from ..import_utils import get_numpy


def _norm01(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((float(x) - float(lo)) / (float(hi) - float(lo)))


def _similarity_abs(a, b, lo, hi):
    return 1.0 - _norm01(abs(float(a) - float(b)), 0.0, hi - lo)


def compute_wij_phase_epi_vf_si(
    th_vals,
    epi_vals,
    vf_vals,
    si_vals,
    i: int | None = None,
    j: int | None = None,
    cos_th=None,
    sin_th=None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np=None,
):
    """Return similarity components for nodes ``i`` and ``j``.

    When ``np`` is provided and ``i`` and ``j`` are ``None`` the computation is
    vectorized returning full matrices for all node pairs.
    """

    if np is not None and i is None and j is None:
        th = np.asarray(th_vals)
        epi = np.asarray(epi_vals)
        vf = np.asarray(vf_vals)
        si = np.asarray(si_vals)
        cos_th = np.cos(th)
        sin_th = np.sin(th)
        epi_range = epi_range if epi_range > 0 else 1.0
        vf_range = vf_range if vf_range > 0 else 1.0
        s_phase = 0.5 * (
            1.0
            + cos_th[:, None] * cos_th[None, :]
            + sin_th[:, None] * sin_th[None, :]
        )
        s_epi = 1.0 - np.abs(epi[:, None] - epi[None, :]) / epi_range
        s_vf = 1.0 - np.abs(vf[:, None] - vf[None, :]) / vf_range
        s_si = 1.0 - np.abs(si[:, None] - si[None, :])
        return s_phase, s_epi, s_vf, s_si

    if i is None or j is None:
        raise ValueError("i and j are required for non-vectorized computation")
    epi_range = epi_range if epi_range > 0 else 1.0
    vf_range = vf_range if vf_range > 0 else 1.0
    cos_th = cos_th or [math.cos(t) for t in th_vals]
    sin_th = sin_th or [math.sin(t) for t in th_vals]
    s_phase = 0.5 * (1.0 + (cos_th[i] * cos_th[j] + sin_th[i] * sin_th[j]))
    s_epi = 1.0 - abs(epi_vals[i] - epi_vals[j]) / epi_range
    s_vf = 1.0 - abs(vf_vals[i] - vf_vals[j]) / vf_range
    s_si = 1.0 - abs(si_vals[i] - si_vals[j])
    return s_phase, s_epi, s_vf, s_si


def _combine_similarity(
    s_phase,
    s_epi,
    s_vf,
    s_si,
    phase_w,
    epi_w,
    vf_w,
    si_w,
    np=None,
):
    wij = phase_w * s_phase + epi_w * s_epi + vf_w * s_vf + si_w * s_si
    if np is not None:
        return np.clip(wij, 0.0, 1.0)
    return clamp01(wij)


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
    epi_range = epi_max - epi_min if epi_max > epi_min else 1.0
    vf_range = vf_max - vf_min if vf_max > vf_min else 1.0
    s_phase, s_epi, s_vf, s_si = compute_wij_phase_epi_vf_si(
        th_vals,
        epi_vals,
        vf_vals,
        si_vals,
        epi_range=epi_range,
        vf_range=vf_range,
        np=np,
    )
    wij = _combine_similarity(
        s_phase,
        s_epi,
        s_vf,
        s_si,
        wnorm["phase"],
        wnorm["epi"],
        wnorm["vf"],
        wnorm["si"],
        np=np,
    )
    if self_diag:
        np.fill_diagonal(wij, 1.0)
    else:
        np.fill_diagonal(wij, 0.0)
    return wij


def _assign_wij(
    wij: list[list[float]],
    i: int,
    j: int,
    th_vals: Sequence[float],
    epi_vals: Sequence[float],
    vf_vals: Sequence[float],
    si_vals: Sequence[float],
    cos_th: Sequence[float],
    sin_th: Sequence[float],
    epi_range: float,
    vf_range: float,
    phase_w: float,
    epi_w: float,
    vf_w: float,
    si_w: float,
) -> None:
    s_phase, s_epi, s_vf, s_si = compute_wij_phase_epi_vf_si(
        th_vals,
        epi_vals,
        vf_vals,
        si_vals,
        i,
        j,
        cos_th,
        sin_th,
        epi_range,
        vf_range,
    )
    wij_ij = _combine_similarity(
        s_phase, s_epi, s_vf, s_si, phase_w, epi_w, vf_w, si_w
    )
    wij[i][j] = wij[j][i] = wij_ij


def _wij_loops(
    G,
    nodes: Sequence[Any],
    node_to_index: dict[Any, int],
    th_vals: Sequence[float],
    epi_vals: Sequence[float],
    vf_vals: Sequence[float],
    si_vals: Sequence[float],
    wnorm: dict[str, float],
    epi_min: float,
    epi_max: float,
    vf_min: float,
    vf_max: float,
    neighbors_only: bool,
    self_diag: bool,
) -> list[list[float]]:
    n = len(nodes)
    wij = [
        [1.0 if (self_diag and i == j) else 0.0 for j in range(n)]
        for i in range(n)
    ]
    phase_w = wnorm["phase"]
    epi_w = wnorm["epi"]
    vf_w = wnorm["vf"]
    si_w = wnorm["si"]

    cos_th = [math.cos(t) for t in th_vals]
    sin_th = [math.sin(t) for t in th_vals]
    epi_range = epi_max - epi_min if epi_max > epi_min else 1.0
    vf_range = vf_max - vf_min if vf_max > vf_min else 1.0
    if neighbors_only:
        for u, v in G.edges():
            i = node_to_index[u]
            j = node_to_index[v]
            if i == j:
                continue
            _assign_wij(
                wij,
                i,
                j,
                th_vals,
                epi_vals,
                vf_vals,
                si_vals,
                cos_th,
                sin_th,
                epi_range,
                vf_range,
                phase_w,
                epi_w,
                vf_w,
                si_w,
            )
    else:
        for i in range(n):
            for j in range(i + 1, n):
                _assign_wij(
                    wij,
                    i,
                    j,
                    th_vals,
                    epi_vals,
                    vf_vals,
                    si_vals,
                    cos_th,
                    sin_th,
                    epi_range,
                    vf_range,
                    phase_w,
                    epi_w,
                    vf_w,
                    si_w,
                )
    return wij


def _compute_stats(values, row_sum, n, self_diag, np=None):
    """Return aggregate statistics for ``values`` and normalized row sums.

    ``values`` and ``row_sum`` can be any iterables. They are normalized to
    either NumPy arrays or Python lists depending on the availability of
    NumPy. The computation then delegates to the appropriate numerical
    functions with minimal branching.
    """

    if np is not None:
        # Normalize inputs to NumPy arrays
        if not isinstance(values, np.ndarray):
            values = np.asarray(list(values), dtype=float)
        else:
            values = values.astype(float)
        if not isinstance(row_sum, np.ndarray):
            row_sum = np.asarray(list(row_sum), dtype=float)
        else:
            row_sum = row_sum.astype(float)

        def size_fn(v):
            return int(v.size)

        def min_fn(v):
            return float(v.min()) if v.size else 0.0

        def max_fn(v):
            return float(v.max()) if v.size else 0.0

        def mean_fn(v):
            return float(v.mean()) if v.size else 0.0

        def wi_fn(r, d):
            return (r / d).astype(float).tolist()
    else:
        # Fall back to pure Python lists
        values = list(values)
        row_sum = list(row_sum)

        def size_fn(v):
            return len(v)

        def min_fn(v):
            return min(v) if v else 0.0

        def max_fn(v):
            return max(v) if v else 0.0

        def mean_fn(v):
            return sum(v) / len(v) if v else 0.0

        def wi_fn(r, d):
            return [float(r[i]) / d for i in range(n)]

    count_val = size_fn(values)
    min_val = min_fn(values)
    max_val = max_fn(values)
    mean_val = mean_fn(values)
    row_count = n if self_diag else n - 1
    denom = max(1, row_count)
    Wi = wi_fn(row_sum, denom)
    return min_val, max_val, mean_val, Wi, count_val


def _finalize_wij(G, nodes, wij, mode, thr, scope, self_diag, np=None):
    """Finalize the coherence matrix ``wij`` and store results in history.

    When ``np`` is provided and ``wij`` is a NumPy array, the computation is
    performed using vectorized operations. Otherwise a pure Python loop-based
    approach is used.
    """

    use_np = np is not None and isinstance(wij, np.ndarray)
    if use_np:
        n = wij.shape[0]
        mask = ~np.eye(n, dtype=bool)
        values = wij[mask]
        row_sum = wij.sum(axis=1)
        if mode == "dense":
            W = wij.tolist()
        else:
            idx = np.where((wij >= thr) & mask)
            W = [
                (int(i), int(j), float(wij[i, j]))
                for i, j in zip(idx[0], idx[1])
            ]
    else:
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

    min_val, max_val, mean_val, Wi, count_val = _compute_stats(
        values, row_sum, n, self_diag, np if use_np else None
    )
    stats = {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "n_edges": count_val,
        "mode": mode,
        "scope": scope,
    }

    hist = ensure_history(G)
    cfg = G.graph.get("COHERENCE", COHERENCE)
    append_metric(hist, cfg.get("history_key", "W_sparse"), W)
    append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
    append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)
    return nodes, W


def coherence_matrix(G, use_numpy: bool | None = None):
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
    use_np = (
        np is not None if use_numpy is None else (use_numpy and np is not None)
    )
    if use_np:
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
        if neighbors_only:
            adj = np.eye(n, dtype=bool)
            for u, v in G.edges():
                i = node_to_index[u]
                j = node_to_index[v]
                adj[i, j] = True
                adj[j, i] = True
            wij = np.where(adj, wij, 0.0)
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


def local_phase_sync_weighted(
    G, n, nodes_order=None, W_row=None, node_to_index=None
):
    """Compute local phase synchrony using explicit weights.

    ``nodes_order`` is the node ordering used to build the coherence matrix
    and ``W_row`` contains either the dense row corresponding to ``n`` or the
    sparse list of ``(i, j, w)`` tuples for the whole matrix.
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

    if (
        isinstance(W_row, list)
        and W_row
        and isinstance(W_row[0], (int, float))
    ):
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


def local_phase_sync(G, n):
    """Compute unweighted local phase synchronization for node ``n``."""
    nodes, W = coherence_matrix(G)
    if nodes is None:
        return 0.0
    return local_phase_sync_weighted(G, n, nodes_order=nodes, W_row=W)


def _coherence_step(G, ctx=None):
    if not G.graph.get("COHERENCE", COHERENCE).get("enabled", True):
        return
    coherence_matrix(G)


def register_coherence_callbacks(G) -> None:
    register_callback(
        G, event="after_step", func=_coherence_step, name="coherence_step"
    )
