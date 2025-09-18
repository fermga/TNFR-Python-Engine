"""Coherence metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


from ..constants import (
    get_aliases,
    get_param,
)
from ..callback_utils import CallbackEvent, callback_manager
from ..glyph_history import ensure_history, append_metric
from ..alias import collect_attr, get_attr, set_attr
from ..collections_utils import normalize_weights
from ..helpers.numeric import clamp01
from ..cache import ensure_node_index_map
from .common import compute_coherence, min_max_range
from .trig_cache import compute_theta_trig, get_trig_cache
from ..observers import (
    DEFAULT_GLYPH_LOAD_SPAN,
    DEFAULT_WBAR_SPAN,
    glyph_load,
    kuramoto_order,
    phase_sync,
)
from ..sense import sigma_vector
from ..import_utils import get_numpy
from ..logging_utils import get_logger

logger = get_logger(__name__)

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_SI = get_aliases("SI")
ALIAS_DNFR = get_aliases("DNFR")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_DSI = get_aliases("DSI")
ALIAS_DVF = get_aliases("DVF")
ALIAS_D2VF = get_aliases("D2VF")


@dataclass
class SimilarityInputs:
    """Similarity inputs and optional trigonometric caches."""

    th_vals: Sequence[float]
    epi_vals: Sequence[float]
    vf_vals: Sequence[float]
    si_vals: Sequence[float]
    cos_vals: Sequence[float] | None = None
    sin_vals: Sequence[float] | None = None


def _compute_wij_phase_epi_vf_si_vectorized(
    epi,
    vf,
    si,
    cos_th,
    sin_th,
    epi_range,
    vf_range,
    np,
):
    """Vectorized computation of similarity components.

    All parameters are expected to be NumPy arrays already cast to ``float``
    when appropriate. ``epi_range`` and ``vf_range`` are normalized inside the
    function to avoid division by zero.
    """

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


def compute_wij_phase_epi_vf_si(
    inputs: SimilarityInputs,
    i: int | None = None,
    j: int | None = None,
    *,
    trig=None,
    G: Any | None = None,
    nodes: Sequence[Any] | None = None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np=None,
):
    """Return similarity components for nodes ``i`` and ``j``.

    When ``np`` is provided and ``i`` and ``j`` are ``None`` the computation is
    vectorized returning full matrices for all node pairs.
    """

    trig = trig or (get_trig_cache(G, np=np) if G is not None else None)
    cos_vals = inputs.cos_vals
    sin_vals = inputs.sin_vals
    if cos_vals is None or sin_vals is None:
        th_vals = inputs.th_vals
        pairs = zip(nodes or range(len(th_vals)), th_vals)
        trig_local = compute_theta_trig(pairs, np=np)
        index_iter = nodes if nodes is not None else range(len(th_vals))
        if trig is not None and nodes is not None:
            cos_vals = [trig.cos.get(n, trig_local.cos[n]) for n in nodes]
            sin_vals = [trig.sin.get(n, trig_local.sin[n]) for n in nodes]
        else:
            cos_vals = [trig_local.cos[i] for i in index_iter]
            sin_vals = [trig_local.sin[i] for i in index_iter]
        inputs.cos_vals = cos_vals
        inputs.sin_vals = sin_vals

    th_vals = inputs.th_vals
    epi_vals = inputs.epi_vals
    vf_vals = inputs.vf_vals
    si_vals = inputs.si_vals

    if np is not None and i is None and j is None:
        epi = np.asarray(epi_vals)
        vf = np.asarray(vf_vals)
        si = np.asarray(si_vals)
        cos_th = np.asarray(cos_vals, dtype=float)
        sin_th = np.asarray(sin_vals, dtype=float)
        return _compute_wij_phase_epi_vf_si_vectorized(
            epi,
            vf,
            si,
            cos_th,
            sin_th,
            epi_range,
            vf_range,
            np,
        )

    if i is None or j is None:
        raise ValueError("i and j are required for non-vectorized computation")
    epi_range = epi_range if epi_range > 0 else 1.0
    vf_range = vf_range if vf_range > 0 else 1.0
    cos_i = cos_vals[i]
    sin_i = sin_vals[i]
    cos_j = cos_vals[j]
    sin_j = sin_vals[j]
    s_phase = 0.5 * (1.0 + (cos_i * cos_j + sin_i * sin_j))
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


def _wij_components_weights(
    G,
    nodes,
    inputs: SimilarityInputs,
    wnorm,
    i: int | None = None,
    j: int | None = None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np=None,
):
    """Return similarity components together with their weights.

    This consolidates repeated computations ensuring that both the
    similarity components and the corresponding weights are derived once and
    consistently across different implementations.
    """

    s_phase, s_epi, s_vf, s_si = compute_wij_phase_epi_vf_si(
        inputs,
        i,
        j,
        G=G,
        nodes=nodes,
        epi_range=epi_range,
        vf_range=vf_range,
        np=np,
    )
    phase_w = wnorm["phase"]
    epi_w = wnorm["epi"]
    vf_w = wnorm["vf"]
    si_w = wnorm["si"]
    return s_phase, s_epi, s_vf, s_si, phase_w, epi_w, vf_w, si_w


def _wij_vectorized(
    G,
    nodes,
    inputs: SimilarityInputs,
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
    (
        s_phase,
        s_epi,
        s_vf,
        s_si,
        phase_w,
        epi_w,
        vf_w,
        si_w,
    ) = _wij_components_weights(
        G,
        nodes,
        inputs,
        wnorm,
        epi_range=epi_range,
        vf_range=vf_range,
        np=np,
    )
    wij = _combine_similarity(
        s_phase, s_epi, s_vf, s_si, phase_w, epi_w, vf_w, si_w, np=np
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
    G: Any,
    nodes: Sequence[Any],
    inputs: SimilarityInputs,
    epi_range: float,
    vf_range: float,
    wnorm: dict[str, float],
) -> None:
    (
        s_phase,
        s_epi,
        s_vf,
        s_si,
        phase_w,
        epi_w,
        vf_w,
        si_w,
    ) = _wij_components_weights(
        G,
        nodes,
        inputs,
        wnorm,
        i,
        j,
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
    inputs: SimilarityInputs,
    wnorm: dict[str, float],
    epi_min: float,
    epi_max: float,
    vf_min: float,
    vf_max: float,
    neighbors_only: bool,
    self_diag: bool,
) -> list[list[float]]:
    n = len(nodes)
    cos_vals = inputs.cos_vals
    sin_vals = inputs.sin_vals
    if cos_vals is None or sin_vals is None:
        th_vals = inputs.th_vals
        trig_local = compute_theta_trig(zip(nodes, th_vals))
        cos_vals = [trig_local.cos[n] for n in nodes]
        sin_vals = [trig_local.sin[n] for n in nodes]
        inputs.cos_vals = cos_vals
        inputs.sin_vals = sin_vals
    wij = [
        [1.0 if (self_diag and i == j) else 0.0 for j in range(n)]
        for i in range(n)
    ]
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
                G,
                nodes,
                inputs,
                epi_range,
                vf_range,
                wnorm,
            )
    else:
        for i in range(n):
            for j in range(i + 1, n):
                _assign_wij(
                    wij,
                    i,
                    j,
                    G,
                    nodes,
                    inputs,
                    epi_range,
                    vf_range,
                    wnorm,
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


def _coherence_numpy(wij, mode, thr, np):
    """Aggregate coherence weights using vectorized operations.

    Produces the structural weight matrix ``W`` along with the list of off
    diagonal values and row sums ready for statistical analysis.
    """

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
    return n, values, row_sum, W


def _coherence_python(wij, mode, thr):
    """Aggregate coherence weights using pure Python loops."""

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
        W: list[tuple[int, int, float]] = []
        for i in range(n):
            row_i = wij[i]
            for j in range(n):
                w = row_i[j]
                if i != j:
                    values.append(w)
                    if w >= thr:
                        W.append((i, j, w))
                row_sum[i] += w
    return n, values, row_sum, W


def _finalize_wij(G, nodes, wij, mode, thr, scope, self_diag, np=None):
    """Finalize the coherence matrix ``wij`` and store results in history.

    When ``np`` is provided and ``wij`` is a NumPy array, the computation is
    performed using vectorized operations. Otherwise a pure Python loop-based
    approach is used.
    """

    use_np = np is not None and isinstance(wij, np.ndarray)
    n, values, row_sum, W = (
        _coherence_numpy(wij, mode, thr, np)
        if use_np
        else _coherence_python(wij, mode, thr)
    )

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
    cfg = get_param(G, "COHERENCE")
    append_metric(hist, cfg.get("history_key", "W_sparse"), W)
    append_metric(hist, cfg.get("Wi_history_key", "W_i"), Wi)
    append_metric(hist, cfg.get("stats_history_key", "W_stats"), stats)
    return nodes, W


def coherence_matrix(G, use_numpy: bool | None = None):
    cfg = get_param(G, "COHERENCE")
    if not cfg.get("enabled", True):
        return None, None

    node_to_index = ensure_node_index_map(G)
    nodes = list(node_to_index.keys())
    n = len(nodes)
    if n == 0:
        return nodes, []

    # NumPy handling for optional vectorized operations
    np = get_numpy()
    use_np = (
        np is not None if use_numpy is None else (use_numpy and np is not None)
    )

    # Precompute indices to avoid repeated list.index calls within loops

    th_vals = collect_attr(G, nodes, ALIAS_THETA, 0.0, np=np if use_np else None)
    epi_vals = collect_attr(G, nodes, ALIAS_EPI, 0.0, np=np if use_np else None)
    vf_vals = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np if use_np else None)
    si_vals = collect_attr(G, nodes, ALIAS_SI, 0.0, np=np if use_np else None)
    si_vals = (
        np.clip(si_vals, 0.0, 1.0)
        if use_np
        else [clamp01(v) for v in si_vals]
    )
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
    trig = get_trig_cache(G, np=np)
    cos_map, sin_map = trig.cos, trig.sin
    trig_local = compute_theta_trig(zip(nodes, th_vals), np=np)
    cos_vals = [cos_map.get(n, trig_local.cos[n]) for n in nodes]
    sin_vals = [sin_map.get(n, trig_local.sin[n]) for n in nodes]
    inputs = SimilarityInputs(
        th_vals=th_vals,
        epi_vals=epi_vals,
        vf_vals=vf_vals,
        si_vals=si_vals,
        cos_vals=cos_vals,
        sin_vals=sin_vals,
    )
    if use_np:
        wij = _wij_vectorized(
            G,
            nodes,
            inputs,
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
            inputs,
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

    trig = get_trig_cache(G)
    cos_map, sin_map = trig.cos, trig.sin

    if (
        isinstance(W_row, list)
        and W_row
        and isinstance(W_row[0], (int, float))
    ):
        for w, nj in zip(W_row, nodes_order):
            if nj == n:
                continue
            den += w
            cos_j = cos_map.get(nj)
            sin_j = sin_map.get(nj)
            if cos_j is None or sin_j is None:
                trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
                cos_j = trig_j.cos[nj]
                sin_j = trig_j.sin[nj]
            num += w * complex(cos_j, sin_j)
    else:
        for ii, jj, w in W_row:
            if ii != i:
                continue
            nj = nodes_order[jj]
            if nj == n:
                continue
            den += w
            cos_j = cos_map.get(nj)
            sin_j = sin_map.get(nj)
            if cos_j is None or sin_j is None:
                trig_j = compute_theta_trig(((nj, G.nodes[nj]),))
                cos_j = trig_j.cos[nj]
                sin_j = trig_j.sin[nj]
            num += w * complex(cos_j, sin_j)

    return abs(num / den) if den else 0.0


def local_phase_sync(G, n):
    """Compute unweighted local phase synchronization for node ``n``."""
    nodes, W = coherence_matrix(G)
    if nodes is None:
        return 0.0
    return local_phase_sync_weighted(G, n, nodes_order=nodes, W_row=W)


def _coherence_step(G, ctx: dict[str, Any] | None = None):
    del ctx

    if not get_param(G, "COHERENCE").get("enabled", True):
        return
    coherence_matrix(G)


def register_coherence_callbacks(G) -> None:
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=_coherence_step,
        name="coherence_step",
    )


# ---------------------------------------------------------------------------
# Coherence and observer-related metric updates
# ---------------------------------------------------------------------------


def _record_metrics(
    hist: dict[str, Any], *pairs: tuple[Any, str], evaluate: bool = False
) -> None:
    """Generic recorder for metric values."""

    for value, key in pairs:
        append_metric(hist, key, value() if evaluate else value)


def _update_coherence(G, hist) -> None:
    """Update network coherence and related means."""

    C, dnfr_mean, depi_mean = compute_coherence(G, return_means=True)
    _record_metrics(
        hist,
        (C, "C_steps"),
        (dnfr_mean, "dnfr_mean"),
        (depi_mean, "depi_mean"),
    )

    cs = hist["C_steps"]
    if cs:
        window = min(len(cs), DEFAULT_WBAR_SPAN)
        w = max(1, window)
        wbar = sum(cs[-w:]) / w
        _record_metrics(hist, (wbar, "W_bar"))


def _update_phase_sync(G, hist) -> None:
    """Capture phase synchrony and Kuramoto order."""

    ps = phase_sync(G)
    ko = kuramoto_order(G)
    _record_metrics(
        hist,
        (ps, "phase_sync"),
        (ko, "kuramoto_R"),
    )


def _update_sigma(G, hist) -> None:
    """Record glyph load and associated Σ⃗ vector."""

    gl = glyph_load(G, window=DEFAULT_GLYPH_LOAD_SPAN)
    _record_metrics(
        hist,
        (gl.get("_estabilizadores", 0.0), "glyph_load_estab"),
        (gl.get("_disruptivos", 0.0), "glyph_load_disr"),
    )

    dist = {k: v for k, v in gl.items() if not k.startswith("_")}
    sig = sigma_vector(dist)
    _record_metrics(
        hist,
        (sig.get("x", 0.0), "sense_sigma_x"),
        (sig.get("y", 0.0), "sense_sigma_y"),
        (sig.get("mag", 0.0), "sense_sigma_mag"),
        (sig.get("angle", 0.0), "sense_sigma_angle"),
    )


def _track_stability(G, hist, dt, eps_dnfr, eps_depi):
    """Track per-node stability and derivative metrics."""

    stables = 0
    total = max(1, G.number_of_nodes())
    delta_si_sum = 0.0
    delta_si_count = 0
    B_sum = 0.0
    B_count = 0

    for _, nd in G.nodes(data=True):
        if (
            abs(get_attr(nd, ALIAS_DNFR, 0.0)) <= eps_dnfr
            and abs(get_attr(nd, ALIAS_DEPI, 0.0)) <= eps_depi
        ):
            stables += 1

        Si_curr = get_attr(nd, ALIAS_SI, 0.0)
        Si_prev = nd.get("_prev_Si", Si_curr)
        dSi = Si_curr - Si_prev
        nd["_prev_Si"] = Si_curr
        set_attr(nd, ALIAS_DSI, dSi)
        delta_si_sum += dSi
        delta_si_count += 1

        vf_curr = get_attr(nd, ALIAS_VF, 0.0)
        vf_prev = nd.get("_prev_vf", vf_curr)
        dvf_dt = (vf_curr - vf_prev) / dt
        dvf_prev = nd.get("_prev_dvf", dvf_dt)
        B = (dvf_dt - dvf_prev) / dt
        nd["_prev_vf"] = vf_curr
        nd["_prev_dvf"] = dvf_dt
        set_attr(nd, ALIAS_DVF, dvf_dt)
        set_attr(nd, ALIAS_D2VF, B)
        B_sum += B
        B_count += 1

    hist["stable_frac"].append(stables / total)
    hist["delta_Si"].append(
        delta_si_sum / delta_si_count if delta_si_count else 0.0
    )
    hist["B"].append(B_sum / B_count if B_count else 0.0)


def _aggregate_si(G, hist):
    """Aggregate Si statistics across nodes."""

    try:
        thr_sel = get_param(G, "SELECTOR_THRESHOLDS")
        thr_def = get_param(G, "GLYPH_THRESHOLDS")
        si_hi = float(thr_sel.get("si_hi", thr_def.get("hi", 0.66)))
        si_lo = float(thr_sel.get("si_lo", thr_def.get("lo", 0.33)))

        sis = [
            s
            for _, nd in G.nodes(data=True)
            if not math.isnan(s := get_attr(nd, ALIAS_SI, float("nan")))
        ]

        total = 0.0
        hi_count = 0
        lo_count = 0
        for s in sis:
            total += s
            if s >= si_hi:
                hi_count += 1
            if s <= si_lo:
                lo_count += 1

        n = len(sis)
        if n:
            hist["Si_mean"].append(total / n)
            hist["Si_hi_frac"].append(hi_count / n)
            hist["Si_lo_frac"].append(lo_count / n)
        else:
            hist["Si_mean"].append(0.0)
            hist["Si_hi_frac"].append(0.0)
            hist["Si_lo_frac"].append(0.0)
    except (KeyError, AttributeError, TypeError) as exc:
        logger.debug("Si aggregation failed: %s", exc)
