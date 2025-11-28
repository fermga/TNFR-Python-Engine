"""Diagnostic metrics."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from operator import ge, le
from statistics import StatisticsError, fmean
from typing import Any, Callable, Iterable, cast

from ..alias import get_attr
from ..utils import CallbackEvent, callback_manager
from ..constants import (
    STATE_DISSONANT,
    STATE_STABLE,
    STATE_TRANSITION,
    VF_KEY,
    get_param,
    normalise_state_token,
)
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_VF
from ..glyph_history import append_metric, ensure_history
from ..utils import clamp01, resolve_chunk_size, similarity_abs
from ..types import (
    DiagnosisNodeData,
    DiagnosisPayload,
    DiagnosisPayloadChunk,
    DiagnosisResult,
    DiagnosisResultList,
    DiagnosisSharedState,
    NodeId,
    TNFRGraph,
)
from ..utils import get_numpy
from .coherence import CoherenceMatrixPayload, coherence_matrix, local_phase_sync
from .common import (
    _coerce_jobs,
    compute_dnfr_accel_max,
    min_max_range,
    normalize_dnfr,
)
from .trig_cache import compute_theta_trig, get_trig_cache

CoherenceSeries = Sequence[CoherenceMatrixPayload | None]
CoherenceHistory = Mapping[str, CoherenceSeries]


def _coherence_matrix_to_numpy(
    weight_matrix: Any,
    size: int,
    np_mod: Any,
) -> Any:
    """Convert stored coherence weights into a dense NumPy array."""

    if weight_matrix is None or np_mod is None or size <= 0:
        return None

    ndarray_type: Any = getattr(np_mod, "ndarray", tuple())
    if ndarray_type and isinstance(weight_matrix, ndarray_type):
        matrix = weight_matrix.astype(float, copy=True)
    elif isinstance(weight_matrix, (list, tuple)):
        weight_seq = list(weight_matrix)
        if not weight_seq:
            matrix = np_mod.zeros((size, size), dtype=float)
        else:
            first = weight_seq[0]
            if isinstance(first, (list, tuple)) and len(first) == size:
                matrix = np_mod.array(weight_seq, dtype=float)
            elif (
                isinstance(first, (list, tuple))
                and len(first) == 3
                and not isinstance(first[0], (list, tuple))
            ):
                matrix = np_mod.zeros((size, size), dtype=float)
                for i, j, weight in weight_seq:
                    matrix[int(i), int(j)] = float(weight)
            else:
                return None
    else:
        return None

    if matrix.shape != (size, size):
        return None
    np_mod.fill_diagonal(matrix, 0.0)
    return matrix


def _weighted_phase_sync_vectorized(
    matrix: Any,
    cos_vals: Any,
    sin_vals: Any,
    np_mod: Any,
) -> Any:
    """Vectorised computation of weighted local phase synchrony."""

    denom = np_mod.sum(matrix, axis=1)
    if np_mod.all(denom == 0.0):
        return np_mod.zeros_like(denom, dtype=float)
    real = matrix @ cos_vals
    imag = matrix @ sin_vals
    magnitude = np_mod.hypot(real, imag)
    safe_denom = np_mod.where(denom == 0.0, 1.0, denom)
    return magnitude / safe_denom


def _unweighted_phase_sync_vectorized(
    nodes: Sequence[Any],
    neighbors_map: Mapping[Any, tuple[Any, ...]],
    cos_arr: Any,
    sin_arr: Any,
    index_map: Mapping[Any, int],
    np_mod: Any,
) -> list[float]:
    """Compute unweighted phase synchrony using NumPy helpers."""

    results: list[float] = []
    for node in nodes:
        neighbors = neighbors_map.get(node, ())
        if not neighbors:
            results.append(0.0)
            continue
        indices = [index_map[nb] for nb in neighbors if nb in index_map]
        if not indices:
            results.append(0.0)
            continue
        cos_vals = np_mod.take(cos_arr, indices)
        sin_vals = np_mod.take(sin_arr, indices)
        real = np_mod.sum(cos_vals)
        imag = np_mod.sum(sin_vals)
        denom = float(len(indices))
        if denom == 0.0:
            results.append(0.0)
        else:
            results.append(float(np_mod.hypot(real, imag) / denom))
    return results


def _neighbor_means_vectorized(
    nodes: Sequence[Any],
    neighbors_map: Mapping[Any, tuple[Any, ...]],
    epi_arr: Any,
    index_map: Mapping[Any, int],
    np_mod: Any,
) -> list[float | None]:
    """Vectorized helper to compute neighbour EPI means."""

    results: list[float | None] = []
    for node in nodes:
        neighbors = neighbors_map.get(node, ())
        if not neighbors:
            results.append(None)
            continue
        indices = [index_map[nb] for nb in neighbors if nb in index_map]
        if not indices:
            results.append(None)
            continue
        values = np_mod.take(epi_arr, indices)
        results.append(float(np_mod.mean(values)))
    return results


@dataclass(frozen=True)
class RLocalWorkerArgs:
    """Typed payload passed to :func:`_rlocal_worker`."""

    chunk: Sequence[Any]
    coherence_nodes: Sequence[Any]
    weight_matrix: Any
    weight_index: Mapping[Any, int]
    neighbors_map: Mapping[Any, tuple[Any, ...]]
    cos_map: Mapping[Any, float]
    sin_map: Mapping[Any, float]


@dataclass(frozen=True)
class NeighborMeanWorkerArgs:
    """Typed payload passed to :func:`_neighbor_mean_worker`."""

    chunk: Sequence[Any]
    neighbors_map: Mapping[Any, tuple[Any, ...]]
    epi_map: Mapping[Any, float]


def _rlocal_worker(args: RLocalWorkerArgs) -> list[float]:
    """Worker used to compute ``R_local`` in Python fallbacks."""

    results: list[float] = []
    for node in args.chunk:
        if args.coherence_nodes and args.weight_matrix is not None:
            idx = args.weight_index.get(node)
            if idx is None:
                rloc = 0.0
            else:
                rloc = _weighted_phase_sync_from_matrix(
                    idx,
                    node,
                    args.coherence_nodes,
                    args.weight_matrix,
                    args.cos_map,
                    args.sin_map,
                )
        else:
            rloc = _local_phase_sync_unweighted(
                args.neighbors_map.get(node, ()),
                args.cos_map,
                args.sin_map,
            )
        results.append(float(rloc))
    return results


def _neighbor_mean_worker(args: NeighborMeanWorkerArgs) -> list[float | None]:
    """Worker used to compute neighbour EPI means in Python mode."""

    results: list[float | None] = []
    for node in args.chunk:
        neighbors = args.neighbors_map.get(node, ())
        if not neighbors:
            results.append(None)
            continue
        try:
            results.append(fmean(args.epi_map[nb] for nb in neighbors))
        except StatisticsError:
            results.append(None)
    return results


def _weighted_phase_sync_from_matrix(
    node_index: int,
    node: Any,
    nodes_order: Sequence[Any],
    matrix: Any,
    cos_map: Mapping[Any, float],
    sin_map: Mapping[Any, float],
) -> float:
    """Compute weighted phase synchrony using a cached matrix."""

    if matrix is None or not nodes_order:
        return 0.0

    num = 0.0 + 0.0j
    den = 0.0

    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        row = matrix[node_index]
        for weight, neighbor in zip(row, nodes_order):
            if neighbor == node:
                continue
            w = float(weight)
            if w == 0.0:
                continue
            cos_j = cos_map.get(neighbor)
            sin_j = sin_map.get(neighbor)
            if cos_j is None or sin_j is None:
                continue
            den += w
            num += w * complex(cos_j, sin_j)
    else:
        for ii, jj, weight in matrix:
            if ii != node_index:
                continue
            neighbor = nodes_order[jj]
            if neighbor == node:
                continue
            w = float(weight)
            if w == 0.0:
                continue
            cos_j = cos_map.get(neighbor)
            sin_j = sin_map.get(neighbor)
            if cos_j is None or sin_j is None:
                continue
            den += w
            num += w * complex(cos_j, sin_j)

    return abs(num / den) if den else 0.0


def _local_phase_sync_unweighted(
    neighbors: Iterable[Any],
    cos_map: Mapping[Any, float],
    sin_map: Mapping[Any, float],
) -> float:
    """Fallback unweighted phase synchrony based on neighbours."""

    num = 0.0 + 0.0j
    den = 0.0
    for neighbor in neighbors:
        cos_j = cos_map.get(neighbor)
        sin_j = sin_map.get(neighbor)
        if cos_j is None or sin_j is None:
            continue
        num += complex(cos_j, sin_j)
        den += 1.0
    return abs(num / den) if den else 0.0


def _state_from_thresholds(
    Rloc: float,
    dnfr_n: float,
    cfg: Mapping[str, Any],
) -> str:
    stb = cfg.get("stable", {"Rloc_hi": 0.8, "dnfr_lo": 0.2, "persist": 3})
    dsr = cfg.get("dissonance", {"Rloc_lo": 0.4, "dnfr_hi": 0.5, "persist": 3})

    stable_checks = {
        "Rloc": (Rloc, float(stb["Rloc_hi"]), ge),
        "dnfr": (dnfr_n, float(stb["dnfr_lo"]), le),
    }
    if all(comp(val, thr) for val, thr, comp in stable_checks.values()):
        return STATE_STABLE

    dissonant_checks = {
        "Rloc": (Rloc, float(dsr["Rloc_lo"]), le),
        "dnfr": (dnfr_n, float(dsr["dnfr_hi"]), ge),
    }
    if all(comp(val, thr) for val, thr, comp in dissonant_checks.values()):
        return STATE_DISSONANT

    return STATE_TRANSITION


def _recommendation(state: str, cfg: Mapping[str, Any]) -> list[Any]:
    adv = cfg.get("advice", {})
    canonical_state = normalise_state_token(state)
    return list(adv.get(canonical_state, []))


def _get_last_weights(
    G: TNFRGraph,
    hist: CoherenceHistory,
) -> tuple[CoherenceMatrixPayload | None, CoherenceMatrixPayload | None]:
    """Return last Wi and Wm matrices from history."""
    CfgW = get_param(G, "COHERENCE")
    Wkey = CfgW.get("Wi_history_key", "W_i")
    Wm_key = CfgW.get("history_key", "W_sparse")
    Wi_series = hist.get(Wkey, [])
    Wm_series = hist.get(Wm_key, [])
    Wi_last = Wi_series[-1] if Wi_series else None
    Wm_last = Wm_series[-1] if Wm_series else None
    return Wi_last, Wm_last


def _node_diagnostics(
    node_data: DiagnosisNodeData,
    shared: DiagnosisSharedState,
) -> DiagnosisResult:
    """Compute diagnostic payload for a single node."""

    dcfg = shared["dcfg"]
    compute_symmetry = shared["compute_symmetry"]
    epi_min = shared["epi_min"]
    epi_max = shared["epi_max"]

    node = node_data["node"]
    Si = clamp01(float(node_data["Si"]))
    EPI = float(node_data["EPI"])
    vf = float(node_data["VF"])
    dnfr_n = clamp01(float(node_data["dnfr_norm"]))
    Rloc = float(node_data["R_local"])

    if compute_symmetry:
        epi_bar = node_data.get("neighbor_epi_mean")
        symm = 1.0 if epi_bar is None else similarity_abs(EPI, epi_bar, epi_min, epi_max)
    else:
        symm = None

    state = _state_from_thresholds(Rloc, dnfr_n, dcfg)
    canonical_state = normalise_state_token(state)

    alerts = []
    if canonical_state == STATE_DISSONANT and dnfr_n >= shared["dissonance_hi"]:
        alerts.append("high structural tension")

    advice = _recommendation(canonical_state, dcfg)

    payload: DiagnosisPayload = {
        "node": node,
        "Si": Si,
        "EPI": EPI,
        VF_KEY: vf,
        "dnfr_norm": dnfr_n,
        "W_i": node_data.get("W_i"),
        "R_local": Rloc,
        "symmetry": symm,
        "state": canonical_state,
        "advice": advice,
        "alerts": alerts,
    }

    return node, payload


def _diagnosis_worker_chunk(
    chunk: DiagnosisPayloadChunk,
    shared: DiagnosisSharedState,
) -> DiagnosisResultList:
    """Evaluate diagnostics for a chunk of nodes."""

    return [_node_diagnostics(item, shared) for item in chunk]


def _diagnosis_step(
    G: TNFRGraph,
    ctx: DiagnosisSharedState | None = None,
    *,
    n_jobs: int | None = None,
) -> None:
    del ctx

    if n_jobs is None:
        n_jobs = _coerce_jobs(G.graph.get("DIAGNOSIS_N_JOBS"))
    else:
        n_jobs = _coerce_jobs(n_jobs)

    dcfg = get_param(G, "DIAGNOSIS")
    if not dcfg.get("enabled", True):
        return

    hist = ensure_history(G)
    coherence_hist = cast(CoherenceHistory, hist)
    key = dcfg.get("history_key", "nodal_diag")

    existing_diag_history = hist.get(key, [])
    if isinstance(existing_diag_history, deque):
        snapshots = list(existing_diag_history)
    elif isinstance(existing_diag_history, list):
        snapshots = existing_diag_history
    else:
        snapshots = []

    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        for node, payload in snapshot.items():
            if not isinstance(payload, Mapping):
                continue
            state_value = payload.get("state")
            if not isinstance(state_value, str):
                continue
            canonical = normalise_state_token(state_value)
            if canonical == state_value:
                continue
            if isinstance(payload, MutableMapping):
                payload["state"] = canonical
            elif isinstance(snapshot, MutableMapping):
                new_payload = dict(payload)
                new_payload["state"] = canonical
                snapshot[node] = new_payload

    norms = compute_dnfr_accel_max(G)
    G.graph["_sel_norms"] = norms
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0

    nodes_data: list[tuple[NodeId, dict[str, Any]]] = list(G.nodes(data=True))
    nodes: list[NodeId] = [n for n, _ in nodes_data]

    Wi_last, Wm_last = _get_last_weights(G, coherence_hist)

    np_mod = get_numpy()
    supports_vector = bool(
        np_mod is not None
        and all(
            hasattr(np_mod, attr)
            for attr in (
                "fromiter",
                "clip",
                "abs",
                "maximum",
                "minimum",
                "array",
                "zeros",
                "zeros_like",
                "sum",
                "hypot",
                "where",
                "take",
                "mean",
                "fill_diagonal",
                "all",
            )
        )
    )

    if not nodes:
        append_metric(hist, key, {})
        return

    rloc_values: list[float]

    if supports_vector:
        epi_arr = np_mod.fromiter(
            (cast(float, get_attr(nd, ALIAS_EPI, 0.0)) for _, nd in nodes_data),
            dtype=float,
            count=len(nodes_data),
        )
        epi_min = float(np_mod.min(epi_arr))
        epi_max = float(np_mod.max(epi_arr))
        epi_vals = epi_arr.tolist()

        si_arr = np_mod.clip(
            np_mod.fromiter(
                (cast(float, get_attr(nd, ALIAS_SI, 0.0)) for _, nd in nodes_data),
                dtype=float,
                count=len(nodes_data),
            ),
            0.0,
            1.0,
        )
        si_vals = si_arr.tolist()

        vf_arr = np_mod.fromiter(
            (cast(float, get_attr(nd, ALIAS_VF, 0.0)) for _, nd in nodes_data),
            dtype=float,
            count=len(nodes_data),
        )
        vf_vals = vf_arr.tolist()

        if dnfr_max > 0:
            dnfr_arr = np_mod.clip(
                np_mod.fromiter(
                    (abs(cast(float, get_attr(nd, ALIAS_DNFR, 0.0))) for _, nd in nodes_data),
                    dtype=float,
                    count=len(nodes_data),
                )
                / dnfr_max,
                0.0,
                1.0,
            )
            dnfr_norms = dnfr_arr.tolist()
        else:
            dnfr_norms = [0.0] * len(nodes)
    else:
        epi_vals = [cast(float, get_attr(nd, ALIAS_EPI, 0.0)) for _, nd in nodes_data]
        epi_min, epi_max = min_max_range(epi_vals, default=(0.0, 1.0))
        si_vals = [clamp01(get_attr(nd, ALIAS_SI, 0.0)) for _, nd in nodes_data]
        vf_vals = [cast(float, get_attr(nd, ALIAS_VF, 0.0)) for _, nd in nodes_data]
        dnfr_norms = [normalize_dnfr(nd, dnfr_max) if dnfr_max > 0 else 0.0 for _, nd in nodes_data]

    epi_map = {node: epi_vals[idx] for idx, node in enumerate(nodes)}

    trig_cache = get_trig_cache(G, np=np_mod)
    trig_local = compute_theta_trig(nodes_data, np=np_mod)
    cos_map = dict(trig_cache.cos)
    sin_map = dict(trig_cache.sin)
    cos_map.update(trig_local.cos)
    sin_map.update(trig_local.sin)

    neighbors_map = {n: tuple(G.neighbors(n)) for n in nodes}

    if Wm_last is None:
        coherence_nodes, weight_matrix = coherence_matrix(G)
        if coherence_nodes is None:
            coherence_nodes = []
            weight_matrix = None
    else:
        coherence_nodes = list(nodes)
        weight_matrix = Wm_last

    coherence_nodes = list(coherence_nodes)
    weight_index = {node: idx for idx, node in enumerate(coherence_nodes)}

    node_index_map: dict[Any, int] | None = None

    if supports_vector:
        size = len(coherence_nodes)
        matrix_np = _coherence_matrix_to_numpy(weight_matrix, size, np_mod) if size else None
        if matrix_np is not None and size:
            cos_weight = np_mod.fromiter(
                (float(cos_map.get(node, 0.0)) for node in coherence_nodes),
                dtype=float,
                count=size,
            )
            sin_weight = np_mod.fromiter(
                (float(sin_map.get(node, 0.0)) for node in coherence_nodes),
                dtype=float,
                count=size,
            )
            weighted_sync = _weighted_phase_sync_vectorized(
                matrix_np,
                cos_weight,
                sin_weight,
                np_mod,
            )
            rloc_map = {coherence_nodes[idx]: float(weighted_sync[idx]) for idx in range(size)}
        else:
            rloc_map = {}

        node_index_map = {node: idx for idx, node in enumerate(nodes)}
        if not rloc_map:
            cos_arr = np_mod.fromiter(
                (float(cos_map.get(node, 0.0)) for node in nodes),
                dtype=float,
                count=len(nodes),
            )
            sin_arr = np_mod.fromiter(
                (float(sin_map.get(node, 0.0)) for node in nodes),
                dtype=float,
                count=len(nodes),
            )
            rloc_values = _unweighted_phase_sync_vectorized(
                nodes,
                neighbors_map,
                cos_arr,
                sin_arr,
                node_index_map,
                np_mod,
            )
        else:
            rloc_values = [rloc_map.get(node, 0.0) for node in nodes]
    else:
        if n_jobs and n_jobs > 1 and len(nodes) > 1:
            approx_chunk = math.ceil(len(nodes) / n_jobs) if n_jobs else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(nodes),
                minimum=1,
            )
            rloc_values = []
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        _rlocal_worker,
                        RLocalWorkerArgs(
                            chunk=nodes[idx : idx + chunk_size],
                            coherence_nodes=coherence_nodes,
                            weight_matrix=weight_matrix,
                            weight_index=weight_index,
                            neighbors_map=neighbors_map,
                            cos_map=cos_map,
                            sin_map=sin_map,
                        ),
                    )
                    for idx in range(0, len(nodes), chunk_size)
                ]
                for fut in futures:
                    rloc_values.extend(fut.result())
        else:
            rloc_values = _rlocal_worker(
                RLocalWorkerArgs(
                    chunk=nodes,
                    coherence_nodes=coherence_nodes,
                    weight_matrix=weight_matrix,
                    weight_index=weight_index,
                    neighbors_map=neighbors_map,
                    cos_map=cos_map,
                    sin_map=sin_map,
                )
            )

    if isinstance(Wi_last, (list, tuple)) and Wi_last:
        wi_values = [Wi_last[i] if i < len(Wi_last) else None for i in range(len(nodes))]
    else:
        wi_values = [None] * len(nodes)

    compute_symmetry = bool(dcfg.get("compute_symmetry", True))
    neighbor_means: list[float | None]
    if compute_symmetry:
        if supports_vector and node_index_map is not None and len(nodes):
            neighbor_means = _neighbor_means_vectorized(
                nodes,
                neighbors_map,
                epi_arr,
                node_index_map,
                np_mod,
            )
        elif n_jobs and n_jobs > 1 and len(nodes) > 1:
            approx_chunk = math.ceil(len(nodes) / n_jobs) if n_jobs else None
            chunk_size = resolve_chunk_size(
                approx_chunk,
                len(nodes),
                minimum=1,
            )
            neighbor_means = cast(list[float | None], [])
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                submit = cast(Callable[..., Any], executor.submit)
                futures = [
                    submit(
                        cast(
                            Callable[[NeighborMeanWorkerArgs], list[float | None]],
                            _neighbor_mean_worker,
                        ),
                        NeighborMeanWorkerArgs(
                            chunk=nodes[idx : idx + chunk_size],
                            neighbors_map=neighbors_map,
                            epi_map=epi_map,
                        ),
                    )
                    for idx in range(0, len(nodes), chunk_size)
                ]
                for fut in futures:
                    neighbor_means.extend(cast(list[float | None], fut.result()))
        else:
            neighbor_means = _neighbor_mean_worker(
                NeighborMeanWorkerArgs(
                    chunk=nodes,
                    neighbors_map=neighbors_map,
                    epi_map=epi_map,
                )
            )
    else:
        neighbor_means = [None] * len(nodes)

    node_payload: DiagnosisPayloadChunk = []
    for idx, node in enumerate(nodes):
        node_payload.append(
            {
                "node": node,
                "Si": si_vals[idx],
                "EPI": epi_vals[idx],
                "VF": vf_vals[idx],
                "dnfr_norm": dnfr_norms[idx],
                "R_local": rloc_values[idx],
                "W_i": wi_values[idx],
                "neighbor_epi_mean": neighbor_means[idx],
            }
        )

    shared = {
        "dcfg": dcfg,
        "compute_symmetry": compute_symmetry,
        "epi_min": float(epi_min),
        "epi_max": float(epi_max),
        "dissonance_hi": float(dcfg.get("dissonance", {}).get("dnfr_hi", 0.5)),
    }

    if n_jobs and n_jobs > 1 and len(node_payload) > 1:
        approx_chunk = math.ceil(len(node_payload) / n_jobs) if n_jobs else None
        chunk_size = resolve_chunk_size(
            approx_chunk,
            len(node_payload),
            minimum=1,
        )
        diag_pairs: DiagnosisResultList = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            submit = cast(Callable[..., Any], executor.submit)
            futures = [
                submit(
                    cast(
                        Callable[
                            [list[dict[str, Any]], dict[str, Any]],
                            list[tuple[Any, dict[str, Any]]],
                        ],
                        _diagnosis_worker_chunk,
                    ),
                    node_payload[idx : idx + chunk_size],
                    shared,
                )
                for idx in range(0, len(node_payload), chunk_size)
            ]
            for fut in futures:
                diag_pairs.extend(cast(DiagnosisResultList, fut.result()))
    else:
        diag_pairs = [_node_diagnostics(item, shared) for item in node_payload]

    diag_map = dict(diag_pairs)
    diag: dict[NodeId, DiagnosisPayload] = {node: diag_map.get(node, {}) for node in nodes}

    append_metric(hist, key, diag)


def dissonance_events(G: TNFRGraph, ctx: DiagnosisSharedState | None = None) -> None:
    """Emit per-node structural dissonance start/end events.

    Events are recorded as ``"dissonance_start"`` and ``"dissonance_end"``.
    """

    del ctx

    hist = ensure_history(G)
    # Dissonance events are recorded in ``history['events']``
    norms = G.graph.get("_sel_norms", {})
    dnfr_max = float(norms.get("dnfr_max", 1.0)) or 1.0
    step_idx = len(hist.get("C_steps", []))
    nodes: list[NodeId] = list(G.nodes())
    for n in nodes:
        nd = G.nodes[n]
        dn = normalize_dnfr(nd, dnfr_max)
        Rloc = local_phase_sync(G, n)
        st = bool(nd.get("_disr_state", False))
        if (not st) and dn >= 0.5 and Rloc <= 0.4:
            nd["_disr_state"] = True
            append_metric(
                hist,
                "events",
                ("dissonance_start", {"node": n, "step": step_idx}),
            )
        elif st and dn <= 0.2 and Rloc >= 0.7:
            nd["_disr_state"] = False
            append_metric(
                hist,
                "events",
                ("dissonance_end", {"node": n, "step": step_idx}),
            )


def register_diagnosis_callbacks(G: TNFRGraph) -> None:
    """Attach diagnosis observers (Si/dissonance tracking) to ``G``."""

    raw_jobs = G.graph.get("DIAGNOSIS_N_JOBS")
    n_jobs = _coerce_jobs(raw_jobs)

    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=partial(_diagnosis_step, n_jobs=n_jobs),
        name="diagnosis_step",
    )
    callback_manager.register_callback(
        G,
        event=CallbackEvent.AFTER_STEP.value,
        func=dissonance_events,
        name="dissonance_events",
    )
