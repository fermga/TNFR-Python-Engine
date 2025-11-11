"""ΔNFR (dynamic network field response) utilities and strategies.

This module provides helper functions to configure, cache and apply ΔNFR
components such as phase, epidemiological state and vortex fields during
simulations.  The neighbour accumulation helpers reuse cached edge indices
and NumPy workspaces whenever available so cosine, sine, EPI, νf and topology
means remain faithful to the canonical ΔNFR reorganisation without redundant
allocations.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from time import perf_counter

from ..alias import get_attr, get_theta_attr, set_dnfr
from ..constants import DEFAULTS, get_param
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
from ..metrics.common import merge_and_normalize_weights
from ..metrics.trig import neighbor_phase_mean_list
from ..metrics.trig_cache import compute_theta_trig
from ..types import (
    DeltaNFRHook,
    DnfrCacheVectors,
    DnfrVectorMap,
    NeighborStats,
    NodeId,
    TNFRGraph,
)
from ..utils import (
    DNFR_PREP_STATE_KEY,
    DnfrPrepState,
    DnfrCache,
    CacheManager,
    _graph_cache_manager,
    angle_diff,
    angle_diff_array,
    cached_node_list,
    cached_nodes_and_A,
    get_numpy,
    normalize_weights,
    resolve_chunk_size,
    new_dnfr_cache,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing hook
    import numpy as np

_MEAN_VECTOR_EPS = 1e-12
_SPARSE_DENSITY_THRESHOLD = 0.25
_DNFR_APPROX_BYTES_PER_EDGE = 48


def _should_vectorize(G: TNFRGraph, np_module: ModuleType | None) -> bool:
    """Return ``True`` when NumPy is available unless the graph disables it."""

    if np_module is None:
        return False
    flag = G.graph.get("vectorized_dnfr")
    if flag is None:
        return True
    return bool(flag)


_NUMPY_CACHE_ATTRS = (
    "theta_np",
    "epi_np",
    "vf_np",
    "cos_theta_np",
    "sin_theta_np",
    "deg_array",
    "neighbor_x_np",
    "neighbor_y_np",
    "neighbor_epi_sum_np",
    "neighbor_vf_sum_np",
    "neighbor_count_np",
    "neighbor_deg_sum_np",
    "neighbor_inv_count_np",
    "neighbor_cos_avg_np",
    "neighbor_sin_avg_np",
    "neighbor_mean_tmp_np",
    "neighbor_mean_length_np",
    "neighbor_accum_np",
    "neighbor_edge_values_np",
    "dense_components_np",
    "dense_accum_np",
    "dense_degree_np",
)


def _profile_start_stop(
    profile: MutableMapping[str, float] | None,
    *,
    keys: Sequence[str] = (),
) -> tuple[Callable[[], float], Callable[[str, float], None]]:
    """Return helpers to measure wall-clock durations for ``profile`` keys."""

    if profile is not None:
        for key in keys:
            profile.setdefault(key, 0.0)

        def _start() -> float:
            return perf_counter()

        def _stop(metric: str, start: float) -> None:
            profile[metric] = float(profile.get(metric, 0.0)) + (perf_counter() - start)

    else:

        def _start() -> float:
            return 0.0

        def _stop(metric: str, start: float) -> None:  # noqa: ARG001 - uniform signature
            return None

    return _start, _stop


def _iter_chunk_offsets(total: int, jobs: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` offsets splitting ``total`` items across ``jobs``."""

    if total <= 0 or jobs <= 1:
        return

    jobs = max(1, min(int(jobs), total))
    base, extra = divmod(total, jobs)
    start = 0
    for i in range(jobs):
        size = base + (1 if i < extra else 0)
        if size <= 0:
            continue
        end = start + size
        yield start, end
        start = end


def _neighbor_sums_worker(
    start: int,
    end: int,
    neighbor_indices: Sequence[Sequence[int]],
    cos_th: Sequence[float],
    sin_th: Sequence[float],
    epi: Sequence[float],
    vf: Sequence[float],
    x_base: Sequence[float],
    y_base: Sequence[float],
    epi_base: Sequence[float],
    vf_base: Sequence[float],
    count_base: Sequence[float],
    deg_base: Sequence[float] | None,
    deg_list: Sequence[float] | None,
    degs_list: Sequence[float] | None,
) -> tuple[
    int,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float] | None,
]:
    """Return partial neighbour sums for the ``[start, end)`` range."""

    chunk_x: list[float] = []
    chunk_y: list[float] = []
    chunk_epi: list[float] = []
    chunk_vf: list[float] = []
    chunk_count: list[float] = []
    chunk_deg: list[float] | None = [] if deg_base is not None else None

    for offset, idx in enumerate(range(start, end)):
        neighbors = neighbor_indices[idx]
        x_i = float(x_base[offset])
        y_i = float(y_base[offset])
        epi_i = float(epi_base[offset])
        vf_i = float(vf_base[offset])
        count_i = float(count_base[offset])
        if deg_base is not None and chunk_deg is not None:
            deg_i_acc = float(deg_base[offset])
        else:
            deg_i_acc = 0.0
        deg_i = float(degs_list[idx]) if degs_list is not None else 0.0

        for neighbor_idx in neighbors:
            x_i += float(cos_th[neighbor_idx])
            y_i += float(sin_th[neighbor_idx])
            epi_i += float(epi[neighbor_idx])
            vf_i += float(vf[neighbor_idx])
            count_i += 1.0
            if chunk_deg is not None:
                if deg_list is not None:
                    deg_i_acc += float(deg_list[neighbor_idx])
                else:
                    deg_i_acc += deg_i

        chunk_x.append(x_i)
        chunk_y.append(y_i)
        chunk_epi.append(epi_i)
        chunk_vf.append(vf_i)
        chunk_count.append(count_i)
        if chunk_deg is not None:
            chunk_deg.append(deg_i_acc)

    return (
        start,
        chunk_x,
        chunk_y,
        chunk_epi,
        chunk_vf,
        chunk_count,
        chunk_deg,
    )


def _dnfr_gradients_worker(
    start: int,
    end: int,
    nodes: Sequence[NodeId],
    theta: list[float],
    epi: list[float],
    vf: list[float],
    th_bar: list[float],
    epi_bar: list[float],
    vf_bar: list[float],
    deg_bar: list[float] | None,
    degs: Mapping[Any, float] | Sequence[float] | None,
    w_phase: float,
    w_epi: float,
    w_vf: float,
    w_topo: float,
) -> tuple[int, list[float]]:
    """Return partial ΔNFR gradients for the ``[start, end)`` range."""

    chunk: list[float] = []
    for idx in range(start, end):
        n = nodes[idx]
        g_phase = -angle_diff(theta[idx], th_bar[idx]) / math.pi
        g_epi = epi_bar[idx] - epi[idx]
        g_vf = vf_bar[idx] - vf[idx]
        if w_topo != 0.0 and deg_bar is not None and degs is not None:
            if isinstance(degs, dict):
                deg_i = float(degs.get(n, 0))
            else:
                deg_i = float(degs[idx])
            g_topo = deg_bar[idx] - deg_i
        else:
            g_topo = 0.0
        chunk.append(w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + w_topo * g_topo)
    return start, chunk


def _resolve_parallel_jobs(n_jobs: int | None, total: int) -> int | None:
    """Return an effective worker count for ``total`` items or ``None``."""

    if n_jobs is None:
        return None
    try:
        jobs = int(n_jobs)
    except (TypeError, ValueError):
        return None
    if jobs <= 1 or total <= 1:
        return None
    return max(1, min(jobs, total))


def _is_numpy_like(obj) -> bool:
    return getattr(obj, "dtype", None) is not None and getattr(obj, "shape", None) is not None


def _has_cached_numpy_buffers(data: dict, cache: DnfrCache | None) -> bool:
    for attr in _NUMPY_CACHE_ATTRS:
        arr = data.get(attr)
        if _is_numpy_like(arr):
            return True
    if cache is not None:
        for attr in _NUMPY_CACHE_ATTRS:
            arr = getattr(cache, attr, None)
            if _is_numpy_like(arr):
                return True
    A = data.get("A")
    if _is_numpy_like(A):
        return True
    return False


__all__ = (
    "default_compute_delta_nfr",
    "set_delta_nfr_hook",
    "dnfr_phase_only",
    "dnfr_epi_vf_mixed",
    "dnfr_laplacian",
    "compute_delta_nfr_hamiltonian",
)


def _write_dnfr_metadata(G, *, weights: dict, hook_name: str, note: str | None = None) -> None:
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
    weights = merge_and_normalize_weights(
        G, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )
    G.graph["_dnfr_weights"] = weights
    return weights


def _init_dnfr_cache(
    G: TNFRGraph,
    nodes: Sequence[NodeId],
    cache_or_manager: CacheManager | DnfrCache | None = None,
    checksum: Any | None = None,
    force_refresh: bool = False,
    *,
    manager: CacheManager | None = None,
) -> tuple[
    DnfrCache,
    dict[NodeId, int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    bool,
]:
    """Initialise or reuse cached ΔNFR arrays.

    ``manager`` telemetry became mandatory in TNFR 9.0 to expose cache hits,
    misses and timings. Older callers still pass a ``cache`` instance as the
    third positional argument; this helper supports both signatures by seeding
    the manager-backed state with the provided cache when necessary.
    """

    if manager is None and isinstance(cache_or_manager, CacheManager):
        manager = cache_or_manager
        cache_or_manager = None

    if manager is None:
        manager = _graph_cache_manager(G.graph)

    graph = G.graph
    state = manager.get(DNFR_PREP_STATE_KEY)
    if not isinstance(state, DnfrPrepState):
        manager.clear(DNFR_PREP_STATE_KEY)
        state = manager.get(DNFR_PREP_STATE_KEY)

    if isinstance(cache_or_manager, DnfrCache):
        state.cache = cache_or_manager
        if checksum is None:
            checksum = cache_or_manager.checksum

    cache = state.cache
    reuse = (
        not force_refresh
        and isinstance(cache, DnfrCache)
        and cache.checksum == checksum
        and len(cache.theta) == len(nodes)
    )
    if reuse:
        manager.increment_hit(DNFR_PREP_STATE_KEY)
        graph["_dnfr_prep_cache"] = cache
        return (
            cache,
            cache.idx,
            cache.theta,
            cache.epi,
            cache.vf,
            cache.cos_theta,
            cache.sin_theta,
            False,
        )

    def _rebuild(current: DnfrPrepState | Any) -> DnfrPrepState:
        if not isinstance(current, DnfrPrepState):
            raise RuntimeError("ΔNFR prep state unavailable during rebuild")
        prev_cache = current.cache if isinstance(current.cache, DnfrCache) else None
        idx_local = {n: i for i, n in enumerate(nodes)}
        size = len(nodes)
        zeros = [0.0] * size
        cache_new = prev_cache if prev_cache is not None else new_dnfr_cache()
        cache_new.idx = idx_local
        cache_new.theta = zeros.copy()
        cache_new.epi = zeros.copy()
        cache_new.vf = zeros.copy()
        cache_new.cos_theta = [1.0] * size
        cache_new.sin_theta = [0.0] * size
        cache_new.neighbor_x = zeros.copy()
        cache_new.neighbor_y = zeros.copy()
        cache_new.neighbor_epi_sum = zeros.copy()
        cache_new.neighbor_vf_sum = zeros.copy()
        cache_new.neighbor_count = zeros.copy()
        cache_new.neighbor_deg_sum = zeros.copy() if size else []
        cache_new.degs = None
        cache_new.edge_src = None
        cache_new.edge_dst = None
        cache_new.checksum = checksum

        # Reset any numpy mirrors or aggregated buffers to avoid leaking
        # state across refresh cycles (e.g. switching between vectorised
        # and Python paths or reusing legacy caches).
        if prev_cache is not None:
            for attr in _NUMPY_CACHE_ATTRS:
                setattr(cache_new, attr, None)
            for attr in (
                "th_bar_np",
                "epi_bar_np",
                "vf_bar_np",
                "deg_bar_np",
                "grad_phase_np",
                "grad_epi_np",
                "grad_vf_np",
                "grad_topo_np",
                "grad_total_np",
            ):
                setattr(cache_new, attr, None)
            cache_new.edge_src = None
            cache_new.edge_dst = None
            cache_new.edge_signature = None
            cache_new.neighbor_accum_signature = None
        cache_new.degs = prev_cache.degs if prev_cache else None
        cache_new.checksum = checksum
        current.cache = cache_new
        graph["_dnfr_prep_cache"] = cache_new
        return current

    with manager.timer(DNFR_PREP_STATE_KEY):
        state = manager.update(DNFR_PREP_STATE_KEY, _rebuild)
    manager.increment_miss(DNFR_PREP_STATE_KEY)
    cache = state.cache
    if not isinstance(cache, DnfrCache):  # pragma: no cover - defensive guard
        raise RuntimeError("ΔNFR cache initialisation failed")
    return (
        cache,
        cache.idx,
        cache.theta,
        cache.epi,
        cache.vf,
        cache.cos_theta,
        cache.sin_theta,
        True,
    )


def _ensure_numpy_vectors(cache: DnfrCache, np: ModuleType) -> DnfrCacheVectors:
    """Ensure NumPy copies of cached vectors are initialised and up to date."""

    if cache is None:
        return (None, None, None, None, None)

    arrays: list[Any | None] = []
    size = len(cache.theta)
    for attr_np, source_attr in (
        ("theta_np", "theta"),
        ("epi_np", "epi"),
        ("vf_np", "vf"),
        ("cos_theta_np", "cos_theta"),
        ("sin_theta_np", "sin_theta"),
    ):
        arr = getattr(cache, attr_np)
        if arr is not None and getattr(arr, "shape", None) == (size,):
            arrays.append(arr)
            continue
        src = getattr(cache, source_attr)
        if src is None:
            setattr(cache, attr_np, None)
            arrays.append(None)
            continue
        arr = np.asarray(src, dtype=float)
        if getattr(arr, "shape", None) != (size,):
            arr = np.array(src, dtype=float)
        setattr(cache, attr_np, arr)
        arrays.append(arr)
    return tuple(arrays)


def _ensure_numpy_degrees(
    cache: DnfrCache,
    deg_list: Sequence[float] | None,
    np: ModuleType,
) -> np.ndarray | None:
    """Initialise/update NumPy array mirroring ``deg_list``.

    Deg_array reuse pattern:
    -------------------------
    The degree array (deg_array) is a cached NumPy buffer that stores node
    degrees for topology-based ΔNFR computations. The reuse pattern follows:

    1. **Allocation**: Created once when topology weight (w_topo) > 0 or when
       caching is enabled, sized to match the node count.

    2. **Reuse across steps**: When the graph topology is stable (no edge
       additions/removals), the same deg_array buffer is reused across
       multiple ΔNFR computation steps by updating in-place via np.copyto.

    3. **Count buffer optimization**: For undirected graphs where node degree
       equals neighbor count, deg_array can serve double duty as the count
       buffer (see _accumulate_neighbors_numpy lines 2185-2194), eliminating
       the need for an extra accumulator row.

    4. **Invalidation**: Cache is cleared when graph.edges changes or when
       _dnfr_prep_dirty flag is set, ensuring fresh allocation on next use.

    This pattern maintains ΔNFR computational accuracy (Invariant #8) while
    minimizing allocations for stable topologies.
    """

    if deg_list is None:
        if cache is not None:
            cache.deg_array = None
        return None
    if cache is None:
        return np.array(deg_list, dtype=float)
    arr = cache.deg_array
    if arr is None or len(arr) != len(deg_list):
        arr = np.array(deg_list, dtype=float)
    else:
        np.copyto(arr, deg_list, casting="unsafe")
    cache.deg_array = arr
    return arr


def _resolve_numpy_degree_array(
    data: MutableMapping[str, Any],
    count: np.ndarray | None,
    *,
    cache: DnfrCache | None,
    np: ModuleType,
) -> np.ndarray | None:
    """Return the vector of node degrees required for topology gradients."""

    if data["w_topo"] == 0.0:
        return None
    deg_array = data.get("deg_array")
    if deg_array is not None:
        return deg_array
    deg_list = data.get("deg_list")
    if deg_list is not None:
        deg_array = np.array(deg_list, dtype=float)
        data["deg_array"] = deg_array
        if cache is not None:
            cache.deg_array = deg_array
        return deg_array
    return count


def _ensure_cached_array(
    cache: DnfrCache | None,
    attr: str,
    shape: tuple[int, ...],
    np: ModuleType,
) -> np.ndarray:
    """Return a cached NumPy buffer with ``shape`` creating/reusing it."""

    if np is None:
        raise RuntimeError("NumPy is required to build cached arrays")
    arr = getattr(cache, attr) if cache is not None else None
    if arr is None or getattr(arr, "shape", None) != shape:
        arr = np.empty(shape, dtype=float)
        if cache is not None:
            setattr(cache, attr, arr)
    return arr


def _ensure_numpy_state_vectors(data: MutableMapping[str, Any], np: ModuleType) -> DnfrVectorMap:
    """Synchronise list-based state vectors with their NumPy counterparts."""

    nodes = data.get("nodes") or ()
    size = len(nodes)
    cache: DnfrCache | None = data.get("cache")

    cache_arrays: DnfrCacheVectors = (None, None, None, None, None)
    if cache is not None:
        cache_arrays = _ensure_numpy_vectors(cache, np)

    result: dict[str, Any | None] = {}
    for plain_key, np_key, cached_arr, result_key in (
        ("theta", "theta_np", cache_arrays[0], "theta"),
        ("epi", "epi_np", cache_arrays[1], "epi"),
        ("vf", "vf_np", cache_arrays[2], "vf"),
        ("cos_theta", "cos_theta_np", cache_arrays[3], "cos"),
        ("sin_theta", "sin_theta_np", cache_arrays[4], "sin"),
    ):
        arr = data.get(np_key)
        if arr is None:
            arr = cached_arr
        if arr is None or getattr(arr, "shape", None) != (size,):
            src = data.get(plain_key)
            if src is None and cache is not None:
                src = getattr(cache, plain_key)
            if src is None:
                arr = None
            else:
                arr = np.asarray(src, dtype=float)
                if getattr(arr, "shape", None) != (size,):
                    arr = np.array(src, dtype=float)
        if arr is not None:
            data[np_key] = arr
            data[plain_key] = arr
            if cache is not None:
                setattr(cache, np_key, arr)
        else:
            data[np_key] = None
        result[result_key] = arr

    return result


def _build_edge_index_arrays(
    G: TNFRGraph,
    nodes: Sequence[NodeId],
    idx: Mapping[NodeId, int],
    np: ModuleType,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (src, dst) index arrays for ``G`` respecting ``nodes`` order."""

    if np is None:
        return None, None
    if not nodes:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    src = []
    dst = []
    append_src = src.append
    append_dst = dst.append
    for node in nodes:
        i = idx.get(node)
        if i is None:
            continue
        for neighbor in G.neighbors(node):
            j = idx.get(neighbor)
            if j is None:
                continue
            append_src(i)
            append_dst(j)
    if not src:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    edge_src = np.asarray(src, dtype=np.intp)
    edge_dst = np.asarray(dst, dtype=np.intp)
    return edge_src, edge_dst


def _refresh_dnfr_vectors(G: TNFRGraph, nodes: Sequence[NodeId], cache: DnfrCache) -> None:
    """Update cached angle and state vectors for ΔNFR."""
    np_module = get_numpy()
    trig = compute_theta_trig(((n, G.nodes[n]) for n in nodes), np=np_module)
    use_numpy = _should_vectorize(G, np_module)
    node_count = len(nodes)
    trig_theta = getattr(trig, "theta_values", None)
    trig_cos = getattr(trig, "cos_values", None)
    trig_sin = getattr(trig, "sin_values", None)
    np_ready = (
        use_numpy
        and np_module is not None
        and isinstance(trig_theta, getattr(np_module, "ndarray", tuple()))
        and isinstance(trig_cos, getattr(np_module, "ndarray", tuple()))
        and isinstance(trig_sin, getattr(np_module, "ndarray", tuple()))
        and getattr(trig_theta, "shape", None) == getattr(trig_cos, "shape", None)
        and getattr(trig_theta, "shape", None) == getattr(trig_sin, "shape", None)
        and (trig_theta.shape[0] if getattr(trig_theta, "ndim", 0) else 0) == node_count
    )

    if np_ready:
        if node_count:
            epi_arr = np_module.fromiter(
                (get_attr(G.nodes[node], ALIAS_EPI, 0.0) for node in nodes),
                dtype=float,
                count=node_count,
            )
            vf_arr = np_module.fromiter(
                (get_attr(G.nodes[node], ALIAS_VF, 0.0) for node in nodes),
                dtype=float,
                count=node_count,
            )
        else:
            epi_arr = np_module.empty(0, dtype=float)
            vf_arr = np_module.empty(0, dtype=float)

        theta_arr = np_module.asarray(trig_theta, dtype=float)
        cos_arr = np_module.asarray(trig_cos, dtype=float)
        sin_arr = np_module.asarray(trig_sin, dtype=float)

        def _sync_numpy(attr: str, source: Any) -> Any:
            dest = getattr(cache, attr)
            if dest is None or getattr(dest, "shape", None) != source.shape:
                dest = np_module.array(source, dtype=float)
            else:
                np_module.copyto(dest, source, casting="unsafe")
            setattr(cache, attr, dest)
            return dest

        _sync_numpy("theta_np", theta_arr)
        _sync_numpy("epi_np", epi_arr)
        _sync_numpy("vf_np", vf_arr)
        _sync_numpy("cos_theta_np", cos_arr)
        _sync_numpy("sin_theta_np", sin_arr)

        # Python mirrors remain untouched while the vectorised path is active.
        # They will be rebuilt the next time the runtime falls back to lists.
        if cache.theta is not None and len(cache.theta) != node_count:
            cache.theta = [0.0] * node_count
        if cache.epi is not None and len(cache.epi) != node_count:
            cache.epi = [0.0] * node_count
        if cache.vf is not None and len(cache.vf) != node_count:
            cache.vf = [0.0] * node_count
        if cache.cos_theta is not None and len(cache.cos_theta) != node_count:
            cache.cos_theta = [1.0] * node_count
        if cache.sin_theta is not None and len(cache.sin_theta) != node_count:
            cache.sin_theta = [0.0] * node_count
    else:
        for index, node in enumerate(nodes):
            i: int = int(index)
            node_id: NodeId = node
            nd = G.nodes[node_id]
            cache.theta[i] = trig.theta[node_id]
            cache.epi[i] = get_attr(nd, ALIAS_EPI, 0.0)
            cache.vf[i] = get_attr(nd, ALIAS_VF, 0.0)
            cache.cos_theta[i] = trig.cos[node_id]
            cache.sin_theta[i] = trig.sin[node_id]
        if use_numpy and np_module is not None:
            _ensure_numpy_vectors(cache, np_module)
        else:
            cache.theta_np = None
            cache.epi_np = None
            cache.vf_np = None
            cache.cos_theta_np = None
            cache.sin_theta_np = None


def _prepare_dnfr_data(
    G: TNFRGraph,
    *,
    cache_size: int | None = 128,
    profile: MutableMapping[str, float] | None = None,
) -> dict[str, Any]:
    """Precompute common data for ΔNFR strategies.

    The helper decides between edge-wise and dense adjacency accumulation
    heuristically.  Graphs whose edge density exceeds
    ``_SPARSE_DENSITY_THRESHOLD`` receive a cached adjacency matrix so the
    dense path can be exercised; callers may also force the dense mode by
    setting ``G.graph['dnfr_force_dense']`` to a truthy value.

    Parameters
    ----------
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping that accumulates wall-clock timings for ΔNFR
        preparation. When provided the helper increases the
        ``"dnfr_cache_rebuild"`` bucket with the time spent refreshing cached
        node vectors and associated NumPy workspaces.
    """
    start_timer, stop_timer = _profile_start_stop(
        profile,
        keys=("dnfr_cache_rebuild",),
    )

    graph = G.graph
    weights = graph.get("_dnfr_weights")
    if weights is None:
        weights = _configure_dnfr_weights(G)

    result: dict[str, Any] = {
        "weights": weights,
        "cache_size": cache_size,
    }

    np_module = get_numpy()
    use_numpy = _should_vectorize(G, np_module)

    nodes = cast(tuple[NodeId, ...], cached_node_list(G))
    edge_count = G.number_of_edges()

    # Centralized decision logic for sparse vs dense accumulation path.
    # This decision affects which accumulation strategy will be used:
    #   - "sparse": edge-based accumulation (_accumulate_neighbors_broadcasted)
    #   - "dense": matrix multiplication with adjacency matrix (_accumulate_neighbors_dense)
    # The decision is stored in dnfr_path_decision for telemetry and debugging.
    prefer_sparse = False
    dense_override = bool(G.graph.get("dnfr_force_dense"))
    dnfr_path_decision = "fallback"  # Default when numpy unavailable

    if use_numpy:
        # Heuristic: use sparse path when density <= _SPARSE_DENSITY_THRESHOLD (0.25)
        prefer_sparse = _prefer_sparse_accumulation(len(nodes), edge_count)

        if dense_override:
            # User explicitly requested dense mode
            prefer_sparse = False
            dnfr_path_decision = "dense_forced"
        elif not prefer_sparse:
            # Heuristic chose dense path (high density graph)
            dnfr_path_decision = "dense_auto"
        else:
            # Heuristic chose sparse path (low density graph)
            dnfr_path_decision = "sparse"

    nodes_cached, A_untyped = cached_nodes_and_A(
        G,
        cache_size=cache_size,
        require_numpy=False,
        prefer_sparse=prefer_sparse,
        nodes=nodes,
    )
    nodes = cast(tuple[NodeId, ...], nodes_cached)
    A: np.ndarray | None = A_untyped
    result["nodes"] = nodes
    result["A"] = A
    manager = _graph_cache_manager(G.graph)
    checksum = G.graph.get("_dnfr_nodes_checksum")
    dirty_flag = bool(G.graph.pop("_dnfr_prep_dirty", False))
    existing_cache = cast(DnfrCache | None, graph.get("_dnfr_prep_cache"))
    cache_timer = start_timer()
    cache, idx, theta, epi, vf, cos_theta, sin_theta, refreshed = _init_dnfr_cache(
        G,
        nodes,
        existing_cache,
        checksum,
        force_refresh=dirty_flag,
        manager=manager,
    )
    stop_timer("dnfr_cache_rebuild", cache_timer)
    dirty = dirty_flag or refreshed
    caching_enabled = cache is not None and (cache_size is None or cache_size > 0)
    result["cache"] = cache
    result["idx"] = idx
    result["theta"] = theta
    result["epi"] = epi
    result["vf"] = vf
    result["cos_theta"] = cos_theta
    result["sin_theta"] = sin_theta
    if cache is not None:
        _refresh_dnfr_vectors(G, nodes, cache)
        if np_module is None and not caching_enabled:
            for attr in (
                "neighbor_x_np",
                "neighbor_y_np",
                "neighbor_epi_sum_np",
                "neighbor_vf_sum_np",
                "neighbor_count_np",
                "neighbor_deg_sum_np",
                "neighbor_inv_count_np",
                "neighbor_cos_avg_np",
                "neighbor_sin_avg_np",
                "neighbor_mean_tmp_np",
                "neighbor_mean_length_np",
                "neighbor_accum_np",
                "neighbor_edge_values_np",
            ):
                setattr(cache, attr, None)
            cache.neighbor_accum_signature = None
            for attr in (
                "th_bar_np",
                "epi_bar_np",
                "vf_bar_np",
                "deg_bar_np",
                "grad_phase_np",
                "grad_epi_np",
                "grad_vf_np",
                "grad_topo_np",
                "grad_total_np",
            ):
                setattr(cache, attr, None)

    w_phase = float(weights.get("phase", 0.0))
    w_epi = float(weights.get("epi", 0.0))
    w_vf = float(weights.get("vf", 0.0))
    w_topo = float(weights.get("topo", 0.0))
    result["w_phase"] = w_phase
    result["w_epi"] = w_epi
    result["w_vf"] = w_vf
    result["w_topo"] = w_topo
    degree_map = cast(dict[NodeId, float] | None, cache.degs if cache else None)
    if cache is not None and dirty:
        cache.degs = None
        cache.deg_list = None
        cache.deg_array = None
        cache.edge_src = None
        cache.edge_dst = None
        cache.edge_signature = None
        cache.neighbor_accum_signature = None
        cache.neighbor_accum_np = None
        cache.neighbor_edge_values_np = None
        degree_map = None

    deg_list: list[float] | None = None
    degs: dict[NodeId, float] | None = None
    deg_array: np.ndarray | None = None

    if w_topo != 0.0 or caching_enabled:
        if degree_map is None or len(degree_map) != len(G):
            degree_map = {cast(NodeId, node): float(deg) for node, deg in G.degree()}
            if cache is not None:
                cache.degs = degree_map

        if (
            cache is not None
            and cache.deg_list is not None
            and not dirty
            and len(cache.deg_list) == len(nodes)
        ):
            deg_list = cache.deg_list
        else:
            deg_list = [float(degree_map.get(node, 0.0)) for node in nodes]
            if cache is not None:
                cache.deg_list = deg_list

        degs = degree_map

        if np_module is not None and deg_list is not None:
            if cache is not None:
                deg_array = _ensure_numpy_degrees(cache, deg_list, np_module)
            else:
                deg_array = np_module.array(deg_list, dtype=float)
        elif cache is not None:
            cache.deg_array = None
    elif cache is not None and dirty:
        cache.deg_list = None
        cache.deg_array = None

    G.graph["_dnfr_prep_dirty"] = False

    result["degs"] = degs
    result["deg_list"] = deg_list

    theta_np: np.ndarray | None
    epi_np: np.ndarray | None
    vf_np: np.ndarray | None
    cos_theta_np: np.ndarray | None
    sin_theta_np: np.ndarray | None
    edge_src: np.ndarray | None
    edge_dst: np.ndarray | None
    if use_numpy:
        theta_np, epi_np, vf_np, cos_theta_np, sin_theta_np = _ensure_numpy_vectors(
            cache, np_module
        )
        edge_src = None
        edge_dst = None
        if cache is not None:
            edge_src = cache.edge_src
            edge_dst = cache.edge_dst
            if edge_src is None or edge_dst is None or dirty:
                edge_src, edge_dst = _build_edge_index_arrays(G, nodes, idx, np_module)
                cache.edge_src = edge_src
                cache.edge_dst = edge_dst
        else:
            edge_src, edge_dst = _build_edge_index_arrays(G, nodes, idx, np_module)

        if cache is not None:
            for attr in ("neighbor_accum_np", "neighbor_edge_values_np"):
                arr = getattr(cache, attr, None)
                if arr is not None:
                    result[attr] = arr
        if edge_src is not None and edge_dst is not None:
            signature = (id(edge_src), id(edge_dst), len(nodes))
            result["edge_signature"] = signature
            if cache is not None:
                cache.edge_signature = signature
    else:
        theta_np = None
        epi_np = None
        vf_np = None
        cos_theta_np = None
        sin_theta_np = None
        edge_src = None
        edge_dst = None
        if cache is not None:
            cache.edge_src = None
            cache.edge_dst = None

    result.setdefault("neighbor_edge_values_np", None)
    if cache is not None and "edge_signature" not in result:
        result["edge_signature"] = cache.edge_signature

    result["theta_np"] = theta_np
    result["epi_np"] = epi_np
    result["vf_np"] = vf_np
    result["cos_theta_np"] = cos_theta_np
    result["sin_theta_np"] = sin_theta_np
    if theta_np is not None and getattr(theta_np, "shape", None) == (len(nodes),):
        result["theta"] = theta_np
    if epi_np is not None and getattr(epi_np, "shape", None) == (len(nodes),):
        result["epi"] = epi_np
    if vf_np is not None and getattr(vf_np, "shape", None) == (len(nodes),):
        result["vf"] = vf_np
    if cos_theta_np is not None and getattr(cos_theta_np, "shape", None) == (len(nodes),):
        result["cos_theta"] = cos_theta_np
    if sin_theta_np is not None and getattr(sin_theta_np, "shape", None) == (len(nodes),):
        result["sin_theta"] = sin_theta_np
    result["deg_array"] = deg_array
    result["edge_src"] = edge_src
    result["edge_dst"] = edge_dst
    result["edge_count"] = edge_count
    result["prefer_sparse"] = prefer_sparse
    result["dense_override"] = dense_override
    result["dnfr_path_decision"] = dnfr_path_decision
    result.setdefault("neighbor_accum_np", None)
    result.setdefault("neighbor_accum_signature", None)

    return result


def _apply_dnfr_gradients(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    th_bar: Sequence[float] | np.ndarray,
    epi_bar: Sequence[float] | np.ndarray,
    vf_bar: Sequence[float] | np.ndarray,
    deg_bar: Sequence[float] | np.ndarray | None = None,
    degs: Mapping[Any, float] | Sequence[float] | np.ndarray | None = None,
    *,
    n_jobs: int | None = None,
    profile: MutableMapping[str, float] | None = None,
) -> None:
    """Combine precomputed gradients and write ΔNFR to each node.

    Parameters
    ----------
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping receiving aggregated timings for the gradient assembly
        (``"dnfr_gradient_assembly"``) and in-place writes
        (``"dnfr_inplace_write"``).
    """
    start_timer, stop_timer = _profile_start_stop(
        profile,
        keys=("dnfr_gradient_assembly", "dnfr_inplace_write"),
    )

    np = get_numpy()
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

    cache: DnfrCache | None = data.get("cache")

    theta_np = data.get("theta_np")
    epi_np = data.get("epi_np")
    vf_np = data.get("vf_np")
    deg_array = data.get("deg_array") if w_topo != 0.0 else None

    use_vector = (
        np is not None
        and theta_np is not None
        and epi_np is not None
        and vf_np is not None
        and isinstance(th_bar, np.ndarray)
        and isinstance(epi_bar, np.ndarray)
        and isinstance(vf_bar, np.ndarray)
    )
    if use_vector and w_topo != 0.0:
        use_vector = (
            deg_bar is not None
            and isinstance(deg_bar, np.ndarray)
            and isinstance(deg_array, np.ndarray)
        )

    grad_timer = start_timer()

    if use_vector:
        grad_phase = _ensure_cached_array(cache, "grad_phase_np", theta_np.shape, np)
        grad_epi = _ensure_cached_array(cache, "grad_epi_np", epi_np.shape, np)
        grad_vf = _ensure_cached_array(cache, "grad_vf_np", vf_np.shape, np)
        grad_total = _ensure_cached_array(cache, "grad_total_np", theta_np.shape, np)
        grad_topo = None
        if w_topo != 0.0:
            grad_topo = _ensure_cached_array(cache, "grad_topo_np", deg_array.shape, np)

        angle_diff_array(theta_np, th_bar, np=np, out=grad_phase)
        np.multiply(grad_phase, -1.0 / math.pi, out=grad_phase)

        np.copyto(grad_epi, epi_bar, casting="unsafe")
        grad_epi -= epi_np

        np.copyto(grad_vf, vf_bar, casting="unsafe")
        grad_vf -= vf_np

        if grad_topo is not None and deg_bar is not None:
            np.copyto(grad_topo, deg_bar, casting="unsafe")
            grad_topo -= deg_array

        if w_phase != 0.0:
            np.multiply(grad_phase, w_phase, out=grad_total)
        else:
            grad_total.fill(0.0)
        if w_epi != 0.0:
            if w_epi != 1.0:
                np.multiply(grad_epi, w_epi, out=grad_epi)
            np.add(grad_total, grad_epi, out=grad_total)
        if w_vf != 0.0:
            if w_vf != 1.0:
                np.multiply(grad_vf, w_vf, out=grad_vf)
            np.add(grad_total, grad_vf, out=grad_total)
        if w_topo != 0.0 and grad_topo is not None:
            if w_topo != 1.0:
                np.multiply(grad_topo, w_topo, out=grad_topo)
            np.add(grad_total, grad_topo, out=grad_total)

        dnfr_values = grad_total
    else:
        effective_jobs = _resolve_parallel_jobs(n_jobs, len(nodes))
        if effective_jobs:
            chunk_results = []
            with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
                futures = []
                for start, end in _iter_chunk_offsets(len(nodes), effective_jobs):
                    if start == end:
                        continue
                    futures.append(
                        executor.submit(
                            _dnfr_gradients_worker,
                            start,
                            end,
                            nodes,
                            theta,
                            epi,
                            vf,
                            th_bar,
                            epi_bar,
                            vf_bar,
                            deg_bar,
                            degs,
                            w_phase,
                            w_epi,
                            w_vf,
                            w_topo,
                        )
                    )
                for future in futures:
                    chunk_results.append(future.result())

            dnfr_values = [0.0] * len(nodes)
            for start, chunk in sorted(chunk_results, key=lambda item: item[0]):
                end = start + len(chunk)
                dnfr_values[start:end] = chunk
        else:
            dnfr_values = []
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
                dnfr_values.append(
                    w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + w_topo * g_topo
                )

        if cache is not None:
            cache.grad_phase_np = None
            cache.grad_epi_np = None
            cache.grad_vf_np = None
            cache.grad_topo_np = None
            cache.grad_total_np = None

    stop_timer("dnfr_gradient_assembly", grad_timer)

    write_timer = start_timer()
    for i, n in enumerate(nodes):
        set_dnfr(G, n, float(dnfr_values[i]))
    stop_timer("dnfr_inplace_write", write_timer)


def _init_bar_arrays(
    data: MutableMapping[str, Any],
    *,
    degs: Mapping[Any, float] | Sequence[float] | None = None,
    np: ModuleType | None = None,
) -> tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float] | None]:
    """Prepare containers for neighbour means.

    If ``np`` is provided, NumPy arrays are created; otherwise lists are used.
    ``degs`` is optional and only initialised when the topological term is
    active.
    """

    nodes = data["nodes"]
    theta = data["theta"]
    epi = data["epi"]
    vf = data["vf"]
    w_topo = data["w_topo"]
    cache: DnfrCache | None = data.get("cache")
    if np is None:
        np = get_numpy()
    if np is not None:
        size = len(theta)
        if cache is not None:
            th_bar = cache.th_bar_np
            if th_bar is None or getattr(th_bar, "shape", None) != (size,):
                th_bar = np.array(theta, dtype=float)
            else:
                np.copyto(th_bar, theta, casting="unsafe")
            cache.th_bar_np = th_bar

            epi_bar = cache.epi_bar_np
            if epi_bar is None or getattr(epi_bar, "shape", None) != (size,):
                epi_bar = np.array(epi, dtype=float)
            else:
                np.copyto(epi_bar, epi, casting="unsafe")
            cache.epi_bar_np = epi_bar

            vf_bar = cache.vf_bar_np
            if vf_bar is None or getattr(vf_bar, "shape", None) != (size,):
                vf_bar = np.array(vf, dtype=float)
            else:
                np.copyto(vf_bar, vf, casting="unsafe")
            cache.vf_bar_np = vf_bar

            if w_topo != 0.0 and degs is not None:
                if isinstance(degs, dict):
                    deg_size = len(nodes)
                else:
                    deg_size = len(degs)
                deg_bar = cache.deg_bar_np
                if deg_bar is None or getattr(deg_bar, "shape", None) != (deg_size,):
                    if isinstance(degs, dict):
                        deg_bar = np.array(
                            [float(degs.get(node, 0.0)) for node in nodes],
                            dtype=float,
                        )
                    else:
                        deg_bar = np.array(degs, dtype=float)
                else:
                    if isinstance(degs, dict):
                        for i, node in enumerate(nodes):
                            deg_bar[i] = float(degs.get(node, 0.0))
                    else:
                        np.copyto(deg_bar, degs, casting="unsafe")
                cache.deg_bar_np = deg_bar
            else:
                deg_bar = None
                if cache is not None:
                    cache.deg_bar_np = None
        else:
            th_bar = np.array(theta, dtype=float)
            epi_bar = np.array(epi, dtype=float)
            vf_bar = np.array(vf, dtype=float)
            deg_bar = np.array(degs, dtype=float) if w_topo != 0.0 and degs is not None else None
    else:
        size = len(theta)
        if cache is not None:
            th_bar = cache.th_bar
            if th_bar is None or len(th_bar) != size:
                th_bar = [0.0] * size
            th_bar[:] = theta
            cache.th_bar = th_bar

            epi_bar = cache.epi_bar
            if epi_bar is None or len(epi_bar) != size:
                epi_bar = [0.0] * size
            epi_bar[:] = epi
            cache.epi_bar = epi_bar

            vf_bar = cache.vf_bar
            if vf_bar is None or len(vf_bar) != size:
                vf_bar = [0.0] * size
            vf_bar[:] = vf
            cache.vf_bar = vf_bar

            if w_topo != 0.0 and degs is not None:
                if isinstance(degs, dict):
                    deg_size = len(nodes)
                else:
                    deg_size = len(degs)
                deg_bar = cache.deg_bar
                if deg_bar is None or len(deg_bar) != deg_size:
                    deg_bar = [0.0] * deg_size
                if isinstance(degs, dict):
                    for i, node in enumerate(nodes):
                        deg_bar[i] = float(degs.get(node, 0.0))
                else:
                    for i, value in enumerate(degs):
                        deg_bar[i] = float(value)
                cache.deg_bar = deg_bar
            else:
                deg_bar = None
                cache.deg_bar = None
        else:
            th_bar = list(theta)
            epi_bar = list(epi)
            vf_bar = list(vf)
            deg_bar = list(degs) if w_topo != 0.0 and degs is not None else None
    return th_bar, epi_bar, vf_bar, deg_bar


def _compute_neighbor_means(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    x: Sequence[float],
    y: Sequence[float],
    epi_sum: Sequence[float],
    vf_sum: Sequence[float],
    count: Sequence[float] | np.ndarray,
    deg_sum: Sequence[float] | None = None,
    degs: Mapping[Any, float] | Sequence[float] | None = None,
    np: ModuleType | None = None,
) -> tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float] | None]:
    """Return neighbour mean arrays for ΔNFR."""
    w_topo = data["w_topo"]
    theta = data["theta"]
    cache: DnfrCache | None = data.get("cache")
    is_numpy = np is not None and isinstance(count, np.ndarray)
    th_bar, epi_bar, vf_bar, deg_bar = _init_bar_arrays(
        data, degs=degs, np=np if is_numpy else None
    )

    if is_numpy:
        n = count.shape[0]
        mask = count > 0
        if not np.any(mask):
            return th_bar, epi_bar, vf_bar, deg_bar

        inv = _ensure_cached_array(cache, "neighbor_inv_count_np", (n,), np)
        inv.fill(0.0)
        np.divide(1.0, count, out=inv, where=mask)

        cos_avg = _ensure_cached_array(cache, "neighbor_cos_avg_np", (n,), np)
        cos_avg.fill(0.0)
        np.multiply(x, inv, out=cos_avg, where=mask)

        sin_avg = _ensure_cached_array(cache, "neighbor_sin_avg_np", (n,), np)
        sin_avg.fill(0.0)
        np.multiply(y, inv, out=sin_avg, where=mask)

        lengths = _ensure_cached_array(cache, "neighbor_mean_length_np", (n,), np)
        np.hypot(cos_avg, sin_avg, out=lengths)

        temp = _ensure_cached_array(cache, "neighbor_mean_tmp_np", (n,), np)
        np.arctan2(sin_avg, cos_avg, out=temp)

        theta_src = data.get("theta_np")
        if theta_src is None:
            theta_src = np.asarray(theta, dtype=float)
        zero_mask = lengths <= _MEAN_VECTOR_EPS
        np.copyto(temp, theta_src, where=zero_mask)
        np.copyto(th_bar, temp, where=mask, casting="unsafe")

        np.divide(epi_sum, count, out=epi_bar, where=mask)
        np.divide(vf_sum, count, out=vf_bar, where=mask)
        if w_topo != 0.0 and deg_bar is not None and deg_sum is not None:
            np.divide(deg_sum, count, out=deg_bar, where=mask)
        return th_bar, epi_bar, vf_bar, deg_bar

    n = len(theta)
    for i in range(n):
        c = count[i]
        if not c:
            continue
        inv = 1.0 / float(c)
        cos_avg = x[i] * inv
        sin_avg = y[i] * inv
        if math.hypot(cos_avg, sin_avg) <= _MEAN_VECTOR_EPS:
            th_bar[i] = theta[i]
        else:
            th_bar[i] = math.atan2(sin_avg, cos_avg)
        epi_bar[i] = epi_sum[i] * inv
        vf_bar[i] = vf_sum[i] * inv
        if w_topo != 0.0 and deg_bar is not None and deg_sum is not None:
            deg_bar[i] = deg_sum[i] * inv
    return th_bar, epi_bar, vf_bar, deg_bar


def _compute_dnfr_common(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    x: Sequence[float],
    y: Sequence[float],
    epi_sum: Sequence[float],
    vf_sum: Sequence[float],
    count: Sequence[float] | None,
    deg_sum: Sequence[float] | None = None,
    degs: Sequence[float] | None = None,
    n_jobs: int | None = None,
    profile: MutableMapping[str, float] | None = None,
) -> None:
    """Compute neighbour means and apply ΔNFR gradients.

    Parameters
    ----------
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping that records wall-clock durations for the neighbour
        mean computation (``"dnfr_neighbor_means"``), the gradient assembly
        (``"dnfr_gradient_assembly"``) and the final in-place writes to the
        graph (``"dnfr_inplace_write"``).
    """
    start_timer, stop_timer = _profile_start_stop(
        profile,
        keys=("dnfr_neighbor_means", "dnfr_gradient_assembly", "dnfr_inplace_write"),
    )

    np_module = get_numpy()
    if np_module is not None and isinstance(count, getattr(np_module, "ndarray", tuple)):
        np_arg = np_module
    else:
        np_arg = None
    neighbor_timer = start_timer()
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
        np=np_arg,
    )
    stop_timer("dnfr_neighbor_means", neighbor_timer)
    _apply_dnfr_gradients(
        G,
        data,
        th_bar,
        epi_bar,
        vf_bar,
        deg_bar,
        degs,
        n_jobs=n_jobs,
        profile=profile,
    )


def _reset_numpy_buffer(
    buffer: np.ndarray | None,
    size: int,
    np: ModuleType,
) -> np.ndarray:
    if buffer is None or getattr(buffer, "shape", None) is None or buffer.shape[0] != size:
        return np.zeros(size, dtype=float)
    buffer.fill(0.0)
    return buffer


def _init_neighbor_sums(
    data: MutableMapping[str, Any],
    *,
    np: ModuleType | None = None,
) -> NeighborStats:
    """Initialise containers for neighbour sums."""
    nodes = data["nodes"]
    n = len(nodes)
    w_topo = data["w_topo"]
    cache: DnfrCache | None = data.get("cache")

    def _reset_list(buffer: list[float] | None, value: float = 0.0) -> list[float]:
        if buffer is None or len(buffer) != n:
            return [value] * n
        for i in range(n):
            buffer[i] = value
        return buffer

    if np is not None:
        if cache is not None:
            x = cache.neighbor_x_np
            y = cache.neighbor_y_np
            epi_sum = cache.neighbor_epi_sum_np
            vf_sum = cache.neighbor_vf_sum_np
            count = cache.neighbor_count_np
            x = _reset_numpy_buffer(x, n, np)
            y = _reset_numpy_buffer(y, n, np)
            epi_sum = _reset_numpy_buffer(epi_sum, n, np)
            vf_sum = _reset_numpy_buffer(vf_sum, n, np)
            count = _reset_numpy_buffer(count, n, np)
            cache.neighbor_x_np = x
            cache.neighbor_y_np = y
            cache.neighbor_epi_sum_np = epi_sum
            cache.neighbor_vf_sum_np = vf_sum
            cache.neighbor_count_np = count
            cache.neighbor_x = _reset_list(cache.neighbor_x)
            cache.neighbor_y = _reset_list(cache.neighbor_y)
            cache.neighbor_epi_sum = _reset_list(cache.neighbor_epi_sum)
            cache.neighbor_vf_sum = _reset_list(cache.neighbor_vf_sum)
            cache.neighbor_count = _reset_list(cache.neighbor_count)
            if w_topo != 0.0:
                deg_sum = _reset_numpy_buffer(cache.neighbor_deg_sum_np, n, np)
                cache.neighbor_deg_sum_np = deg_sum
                cache.neighbor_deg_sum = _reset_list(cache.neighbor_deg_sum)
            else:
                cache.neighbor_deg_sum_np = None
                cache.neighbor_deg_sum = None
                deg_sum = None
        else:
            x = np.zeros(n, dtype=float)
            y = np.zeros(n, dtype=float)
            epi_sum = np.zeros(n, dtype=float)
            vf_sum = np.zeros(n, dtype=float)
            count = np.zeros(n, dtype=float)
            deg_sum = np.zeros(n, dtype=float) if w_topo != 0.0 else None
        degs = None
    else:
        if cache is not None:
            x = _reset_list(cache.neighbor_x)
            y = _reset_list(cache.neighbor_y)
            epi_sum = _reset_list(cache.neighbor_epi_sum)
            vf_sum = _reset_list(cache.neighbor_vf_sum)
            count = _reset_list(cache.neighbor_count)
            cache.neighbor_x = x
            cache.neighbor_y = y
            cache.neighbor_epi_sum = epi_sum
            cache.neighbor_vf_sum = vf_sum
            cache.neighbor_count = count
            if w_topo != 0.0:
                deg_sum = _reset_list(cache.neighbor_deg_sum)
                cache.neighbor_deg_sum = deg_sum
            else:
                cache.neighbor_deg_sum = None
                deg_sum = None
        else:
            x = [0.0] * n
            y = [0.0] * n
            epi_sum = [0.0] * n
            vf_sum = [0.0] * n
            count = [0.0] * n
            deg_sum = [0.0] * n if w_topo != 0.0 else None
        deg_list = data.get("deg_list")
        if w_topo != 0.0 and deg_list is not None:
            degs = deg_list
        else:
            degs = None
    return x, y, epi_sum, vf_sum, count, deg_sum, degs


def _prefer_sparse_accumulation(n: int, edge_count: int | None) -> bool:
    """Return ``True`` when neighbour sums should use edge accumulation."""

    if n <= 1 or not edge_count:
        return False
    possible_edges = n * (n - 1)
    if possible_edges <= 0:
        return False
    density = edge_count / possible_edges
    return density <= _SPARSE_DENSITY_THRESHOLD


def _accumulate_neighbors_dense(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    epi_sum: np.ndarray,
    vf_sum: np.ndarray,
    count: np.ndarray,
    deg_sum: np.ndarray | None,
    np: ModuleType,
) -> NeighborStats:
    """Vectorised neighbour accumulation using a dense adjacency matrix."""

    nodes = data["nodes"]
    if not nodes:
        return x, y, epi_sum, vf_sum, count, deg_sum, None

    A = data.get("A")
    if A is None:
        return _accumulate_neighbors_numpy(
            G,
            data,
            x=x,
            y=y,
            epi_sum=epi_sum,
            vf_sum=vf_sum,
            count=count,
            deg_sum=deg_sum,
            np=np,
        )

    cache: DnfrCache | None = data.get("cache")
    n = len(nodes)

    state = _ensure_numpy_state_vectors(data, np)
    vectors = [state["cos"], state["sin"], state["epi"], state["vf"]]

    components = _ensure_cached_array(cache, "dense_components_np", (n, 4), np)
    accum = _ensure_cached_array(cache, "dense_accum_np", (n, 4), np)

    # ``components`` retains the last source copies so callers relying on
    # cached buffers (e.g. diagnostics) still observe meaningful values.
    np.copyto(components, np.column_stack(vectors), casting="unsafe")

    np.matmul(A, components, out=accum)

    np.copyto(x, accum[:, 0], casting="unsafe")
    np.copyto(y, accum[:, 1], casting="unsafe")
    np.copyto(epi_sum, accum[:, 2], casting="unsafe")
    np.copyto(vf_sum, accum[:, 3], casting="unsafe")

    degree_counts = data.get("dense_degree_np")
    if degree_counts is None or getattr(degree_counts, "shape", (0,))[0] != n:
        degree_counts = None
    if degree_counts is None and cache is not None:
        cached_counts = cache.dense_degree_np
        if cached_counts is not None and getattr(cached_counts, "shape", (0,))[0] == n:
            degree_counts = cached_counts
    if degree_counts is None:
        degree_counts = A.sum(axis=1)
        if cache is not None:
            cache.dense_degree_np = degree_counts
    data["dense_degree_np"] = degree_counts
    np.copyto(count, degree_counts, casting="unsafe")

    degs = None
    if deg_sum is not None:
        deg_array = data.get("deg_array")
        if deg_array is None:
            deg_array = _resolve_numpy_degree_array(
                data,
                count,
                cache=cache,
                np=np,
            )
        if deg_array is None:
            deg_sum.fill(0.0)
        else:
            np.matmul(A, deg_array, out=deg_sum)
            degs = deg_array

    return x, y, epi_sum, vf_sum, count, deg_sum, degs


def _accumulate_neighbors_broadcasted(
    *,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    epi: np.ndarray,
    vf: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epi_sum: np.ndarray,
    vf_sum: np.ndarray,
    count: np.ndarray | None,
    deg_sum: np.ndarray | None,
    deg_array: np.ndarray | None,
    cache: DnfrCache | None,
    np: ModuleType,
    chunk_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Accumulate neighbour contributions using direct indexed reductions.

    Array reuse strategy for non-chunked blocks:
    --------------------------------------------
    This function optimizes memory usage by reusing cached destination arrays:

    1. **Accumulator reuse**: The `accum` matrix (component_rows × n) is cached
       across invocations when signature remains stable. For non-chunked paths,
       it's zero-filled (accum.fill(0.0)) rather than reallocated.

    2. **Workspace reuse**: The `workspace` buffer (component_rows × edge_count)
       stores intermediate edge values. In non-chunked mode with sufficient
       workspace size, edge values are extracted once into workspace rows
       via np.take(..., out=workspace[row, :]) to avoid repeated allocations.

    3. **Destination array writes**: np.bincount results are written to accum
       rows via np.copyto(..., casting="unsafe"), reusing the same memory
       across all components (cos, sin, epi, vf, count, deg).

    4. **Deg_array optimization**: When deg_array is provided and topology
       weight is active, degree values are extracted into workspace and
       accumulated via bincount, maintaining the reuse pattern.

    The non-chunked path achieves minimal temporary allocations by:
    - Reusing cached accum and workspace buffers
    - Extracting all edge values into workspace in a single pass
    - Writing bincount results directly to destination rows

    Note: np.bincount does not support an `out` parameter, so its return
    value must be copied to the destination. The workspace pattern minimizes
    the number of temporary arrays created during edge value extraction.

    This approach maintains ΔNFR computational accuracy (Invariant #8) while
    reducing memory footprint for repeated accumulations with stable topology.
    """

    n = x.shape[0]
    edge_count = int(edge_src.size)

    include_count = count is not None
    use_topology = deg_sum is not None and deg_array is not None

    component_rows = 4 + (1 if include_count else 0) + (1 if use_topology else 0)

    if edge_count:
        if chunk_size is None:
            resolved_chunk = edge_count
        else:
            try:
                resolved_chunk = int(chunk_size)
            except (TypeError, ValueError):
                resolved_chunk = edge_count
            else:
                if resolved_chunk <= 0:
                    resolved_chunk = edge_count
        resolved_chunk = max(1, min(edge_count, resolved_chunk))
    else:
        resolved_chunk = 0

    use_chunks = bool(edge_count and resolved_chunk < edge_count)

    if cache is not None:
        base_signature = (id(edge_src), id(edge_dst), n, edge_count)
        cache.edge_signature = base_signature
        signature = (base_signature, component_rows)
        previous_signature = cache.neighbor_accum_signature

        accum = cache.neighbor_accum_np
        if (
            accum is None
            or getattr(accum, "shape", None) != (component_rows, n)
            or previous_signature != signature
        ):
            accum = np.zeros((component_rows, n), dtype=float)
            cache.neighbor_accum_np = accum
        else:
            accum.fill(0.0)

        workspace = cache.neighbor_edge_values_np
        if use_chunks:
            workspace_length = resolved_chunk
        else:
            # For non-chunked path, allocate workspace to hold edge_count values
            # so we can extract edge values without temporary allocations
            workspace_length = edge_count if edge_count else component_rows
        if workspace_length:
            expected_shape = (component_rows, workspace_length)
            if workspace is None or getattr(workspace, "shape", None) != expected_shape:
                workspace = np.empty(expected_shape, dtype=float)
        else:
            workspace = None
        cache.neighbor_edge_values_np = workspace

        cache.neighbor_accum_signature = signature
    else:
        accum = np.zeros((component_rows, n), dtype=float)
        # For non-chunked path without cache, allocate workspace for edge values
        workspace_length = (
            edge_count
            if (not use_chunks and edge_count)
            else (resolved_chunk if use_chunks else component_rows)
        )
        workspace = (
            np.empty((component_rows, workspace_length), dtype=float) if workspace_length else None
        )

    if edge_count:
        row = 0
        cos_row = row
        row += 1
        sin_row = row
        row += 1
        epi_row = row
        row += 1
        vf_row = row
        row += 1
        count_row = row if include_count and count is not None else None
        if count_row is not None:
            row += 1
        deg_row = row if use_topology and deg_array is not None else None

        edge_src_int = edge_src.astype(np.intp, copy=False)
        edge_dst_int = edge_dst.astype(np.intp, copy=False)

        if use_chunks:
            chunk_step = resolved_chunk if resolved_chunk else edge_count
            chunk_indices = range(0, edge_count, chunk_step)

            for start in chunk_indices:
                end = min(start + chunk_step, edge_count)
                if start >= end:
                    continue
                src_slice = edge_src_int[start:end]
                dst_slice = edge_dst_int[start:end]
                slice_len = end - start
                if slice_len <= 0:
                    continue

                if workspace is not None:
                    chunk_matrix = workspace[:, :slice_len]
                else:
                    chunk_matrix = np.empty((component_rows, slice_len), dtype=float)

                np.take(cos, dst_slice, out=chunk_matrix[cos_row, :slice_len])
                np.take(sin, dst_slice, out=chunk_matrix[sin_row, :slice_len])
                np.take(epi, dst_slice, out=chunk_matrix[epi_row, :slice_len])
                np.take(vf, dst_slice, out=chunk_matrix[vf_row, :slice_len])

                if count_row is not None:
                    chunk_matrix[count_row, :slice_len].fill(1.0)
                if deg_row is not None and deg_array is not None:
                    np.take(deg_array, dst_slice, out=chunk_matrix[deg_row, :slice_len])

                def _accumulate_into(
                    target_row: int | None,
                    values: np.ndarray | None = None,
                    *,
                    unit_weight: bool = False,
                ) -> None:
                    if target_row is None:
                        return
                    row_view = accum[target_row]
                    if unit_weight:
                        np.add.at(row_view, src_slice, 1.0)
                    else:
                        if values is None:
                            return
                        np.add.at(row_view, src_slice, values)

                _accumulate_into(cos_row, chunk_matrix[cos_row, :slice_len])
                _accumulate_into(sin_row, chunk_matrix[sin_row, :slice_len])
                _accumulate_into(epi_row, chunk_matrix[epi_row, :slice_len])
                _accumulate_into(vf_row, chunk_matrix[vf_row, :slice_len])

                if count_row is not None:
                    _accumulate_into(count_row, unit_weight=True)

                if deg_row is not None and deg_array is not None:
                    _accumulate_into(deg_row, chunk_matrix[deg_row, :slice_len])
        else:
            # Non-chunked path: reuse workspace to minimize temporary allocations.
            # When workspace is available with sufficient size, extract edge values
            # into workspace rows before passing to bincount.
            if workspace is not None and workspace.shape[1] >= edge_count:
                # Verify workspace has enough rows for all components
                # workspace has shape (component_rows, edge_count)
                required_rows = max(
                    cos_row + 1,
                    sin_row + 1,
                    epi_row + 1,
                    vf_row + 1,
                    (count_row + 1) if count_row is not None else 0,
                    (deg_row + 1) if deg_row is not None else 0,
                )
                if workspace.shape[0] >= required_rows:
                    # Reuse workspace rows for edge value extraction
                    np.take(cos, edge_dst_int, out=workspace[cos_row, :edge_count])
                    np.take(sin, edge_dst_int, out=workspace[sin_row, :edge_count])
                    np.take(epi, edge_dst_int, out=workspace[epi_row, :edge_count])
                    np.take(vf, edge_dst_int, out=workspace[vf_row, :edge_count])

                    def _apply_full_bincount(
                        target_row: int | None,
                        values: np.ndarray | None = None,
                        *,
                        unit_weight: bool = False,
                    ) -> None:
                        if target_row is None:
                            return
                        if values is None and not unit_weight:
                            return
                        if unit_weight:
                            component_accum = np.bincount(
                                edge_src_int,
                                minlength=n,
                            )
                        else:
                            component_accum = np.bincount(
                                edge_src_int,
                                weights=values,
                                minlength=n,
                            )
                        np.copyto(
                            accum[target_row, :n],
                            component_accum[:n],
                            casting="unsafe",
                        )

                    _apply_full_bincount(cos_row, workspace[cos_row, :edge_count])
                    _apply_full_bincount(sin_row, workspace[sin_row, :edge_count])
                    _apply_full_bincount(epi_row, workspace[epi_row, :edge_count])
                    _apply_full_bincount(vf_row, workspace[vf_row, :edge_count])

                    if count_row is not None:
                        _apply_full_bincount(count_row, unit_weight=True)

                    if deg_row is not None and deg_array is not None:
                        np.take(deg_array, edge_dst_int, out=workspace[deg_row, :edge_count])
                        _apply_full_bincount(deg_row, workspace[deg_row, :edge_count])
                else:
                    # Workspace doesn't have enough rows, fall back to temporary arrays
                    def _apply_full_bincount(
                        target_row: int | None,
                        values: np.ndarray | None = None,
                        *,
                        unit_weight: bool = False,
                    ) -> None:
                        if target_row is None:
                            return
                        if values is None and not unit_weight:
                            return
                        if unit_weight:
                            component_accum = np.bincount(
                                edge_src_int,
                                minlength=n,
                            )
                        else:
                            component_accum = np.bincount(
                                edge_src_int,
                                weights=values,
                                minlength=n,
                            )
                        np.copyto(
                            accum[target_row, :n],
                            component_accum[:n],
                            casting="unsafe",
                        )

                    _apply_full_bincount(cos_row, np.take(cos, edge_dst_int))
                    _apply_full_bincount(sin_row, np.take(sin, edge_dst_int))
                    _apply_full_bincount(epi_row, np.take(epi, edge_dst_int))
                    _apply_full_bincount(vf_row, np.take(vf, edge_dst_int))

                    if count_row is not None:
                        _apply_full_bincount(count_row, unit_weight=True)

                    if deg_row is not None and deg_array is not None:
                        _apply_full_bincount(deg_row, np.take(deg_array, edge_dst_int))
            else:
                # Fallback: no workspace or insufficient width, use temporary arrays
                def _apply_full_bincount(
                    target_row: int | None,
                    values: np.ndarray | None = None,
                    *,
                    unit_weight: bool = False,
                ) -> None:
                    if target_row is None:
                        return
                    if values is None and not unit_weight:
                        return
                    if unit_weight:
                        component_accum = np.bincount(
                            edge_src_int,
                            minlength=n,
                        )
                    else:
                        component_accum = np.bincount(
                            edge_src_int,
                            weights=values,
                            minlength=n,
                        )
                    np.copyto(
                        accum[target_row, :n],
                        component_accum[:n],
                        casting="unsafe",
                    )

                _apply_full_bincount(cos_row, np.take(cos, edge_dst_int))
                _apply_full_bincount(sin_row, np.take(sin, edge_dst_int))
                _apply_full_bincount(epi_row, np.take(epi, edge_dst_int))
                _apply_full_bincount(vf_row, np.take(vf, edge_dst_int))

                if count_row is not None:
                    _apply_full_bincount(count_row, unit_weight=True)

                if deg_row is not None and deg_array is not None:
                    _apply_full_bincount(deg_row, np.take(deg_array, edge_dst_int))
    else:
        accum.fill(0.0)
        if workspace is not None:
            workspace.fill(0.0)

    row = 0
    np.copyto(x, accum[row], casting="unsafe")
    row += 1
    np.copyto(y, accum[row], casting="unsafe")
    row += 1
    np.copyto(epi_sum, accum[row], casting="unsafe")
    row += 1
    np.copyto(vf_sum, accum[row], casting="unsafe")
    row += 1

    if include_count and count is not None:
        np.copyto(count, accum[row], casting="unsafe")
        row += 1

    if use_topology and deg_sum is not None:
        np.copyto(deg_sum, accum[row], casting="unsafe")

    return {
        "accumulator": accum,
        "edge_values": workspace,
    }


def _build_neighbor_sums_common(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    use_numpy: bool,
    n_jobs: int | None = None,
) -> NeighborStats:
    """Build neighbour accumulators honouring cached NumPy buffers when possible."""

    nodes = data["nodes"]
    cache: DnfrCache | None = data.get("cache")
    np_module = get_numpy()
    has_numpy_buffers = _has_cached_numpy_buffers(data, cache)

    # Fallback: when get_numpy() returns None but we have cached NumPy buffers,
    # attempt to retrieve NumPy from sys.modules to avoid losing vectorization.
    # This preserves ΔNFR semantics (Invariant #3) and maintains performance.
    if use_numpy and np_module is None and has_numpy_buffers:
        candidate = sys.modules.get("numpy")
        # Validate the candidate module has required NumPy attributes
        if candidate is not None and hasattr(candidate, "ndarray") and hasattr(candidate, "empty"):
            np_module = candidate

    if np_module is not None:
        if not nodes:
            return _init_neighbor_sums(data, np=np_module)

        x, y, epi_sum, vf_sum, count, deg_sum, degs = _init_neighbor_sums(data, np=np_module)

        # Reuse centralized sparse/dense decision from _prepare_dnfr_data.
        # The decision logic at lines 785-807 already computed prefer_sparse
        # and dense_override based on graph density and user flags.
        prefer_sparse = data.get("prefer_sparse")
        if prefer_sparse is None:
            # Fallback: recompute if not set (defensive, should be rare)
            prefer_sparse = _prefer_sparse_accumulation(len(nodes), data.get("edge_count"))
            data["prefer_sparse"] = prefer_sparse

        use_dense = False
        A = data.get("A")
        dense_override = data.get("dense_override", False)

        # Apply centralized decision: dense path requires adjacency matrix
        # and either high graph density or explicit dense_override flag.
        if use_numpy and A is not None:
            shape = getattr(A, "shape", (0, 0))
            matrix_valid = shape[0] == len(nodes) and shape[1] == len(nodes)
            if matrix_valid and (dense_override or not prefer_sparse):
                use_dense = True

        if use_dense:
            accumulator = _accumulate_neighbors_dense
        else:
            _ensure_numpy_state_vectors(data, np_module)
            accumulator = _accumulate_neighbors_numpy
        return accumulator(
            G,
            data,
            x=x,
            y=y,
            epi_sum=epi_sum,
            vf_sum=vf_sum,
            count=count,
            deg_sum=deg_sum,
            np=np_module,
        )

    if not nodes:
        return _init_neighbor_sums(data)

    x, y, epi_sum, vf_sum, count, deg_sum, degs_list = _init_neighbor_sums(data)
    idx = data["idx"]
    epi = data["epi"]
    vf = data["vf"]
    cos_th = data["cos_theta"]
    sin_th = data["sin_theta"]
    deg_list = data.get("deg_list")

    effective_jobs = _resolve_parallel_jobs(n_jobs, len(nodes))
    if effective_jobs:
        neighbor_indices: list[list[int]] = []
        for node in nodes:
            indices: list[int] = []
            for v in G.neighbors(node):
                indices.append(idx[v])
            neighbor_indices.append(indices)

        chunk_results = []
        with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            futures = []
            for start, end in _iter_chunk_offsets(len(nodes), effective_jobs):
                if start == end:
                    continue
                futures.append(
                    executor.submit(
                        _neighbor_sums_worker,
                        start,
                        end,
                        neighbor_indices,
                        cos_th,
                        sin_th,
                        epi,
                        vf,
                        x[start:end],
                        y[start:end],
                        epi_sum[start:end],
                        vf_sum[start:end],
                        count[start:end],
                        deg_sum[start:end] if deg_sum is not None else None,
                        deg_list,
                        degs_list,
                    )
                )
            for future in futures:
                chunk_results.append(future.result())

        for (
            start,
            chunk_x,
            chunk_y,
            chunk_epi,
            chunk_vf,
            chunk_count,
            chunk_deg,
        ) in sorted(chunk_results, key=lambda item: item[0]):
            end = start + len(chunk_x)
            x[start:end] = chunk_x
            y[start:end] = chunk_y
            epi_sum[start:end] = chunk_epi
            vf_sum[start:end] = chunk_vf
            count[start:end] = chunk_count
            if deg_sum is not None and chunk_deg is not None:
                deg_sum[start:end] = chunk_deg
        return x, y, epi_sum, vf_sum, count, deg_sum, degs_list

    for i, node in enumerate(nodes):
        deg_i = degs_list[i] if degs_list is not None else 0.0
        x_i = x[i]
        y_i = y[i]
        epi_i = epi_sum[i]
        vf_i = vf_sum[i]
        count_i = count[i]
        deg_acc = deg_sum[i] if deg_sum is not None else 0.0
        for v in G.neighbors(node):
            j = idx[v]
            cos_j = cos_th[j]
            sin_j = sin_th[j]
            epi_j = epi[j]
            vf_j = vf[j]
            x_i += cos_j
            y_i += sin_j
            epi_i += epi_j
            vf_i += vf_j
            count_i += 1
            if deg_sum is not None:
                deg_acc += deg_list[j] if deg_list is not None else deg_i
        x[i] = x_i
        y[i] = y_i
        epi_sum[i] = epi_i
        vf_sum[i] = vf_i
        count[i] = count_i
        if deg_sum is not None:
            deg_sum[i] = deg_acc
    return x, y, epi_sum, vf_sum, count, deg_sum, degs_list


def _accumulate_neighbors_numpy(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    epi_sum: np.ndarray,
    vf_sum: np.ndarray,
    count: np.ndarray | None,
    deg_sum: np.ndarray | None,
    np: ModuleType,
) -> NeighborStats:
    """Vectorised neighbour accumulation reusing cached NumPy buffers."""

    nodes = data["nodes"]
    if not nodes:
        return x, y, epi_sum, vf_sum, count, deg_sum, None

    cache: DnfrCache | None = data.get("cache")

    state = _ensure_numpy_state_vectors(data, np)
    cos_th = state["cos"]
    sin_th = state["sin"]
    epi = state["epi"]
    vf = state["vf"]

    edge_src = data.get("edge_src")
    edge_dst = data.get("edge_dst")
    if edge_src is None or edge_dst is None:
        edge_src, edge_dst = _build_edge_index_arrays(G, nodes, data["idx"], np)
        data["edge_src"] = edge_src
        data["edge_dst"] = edge_dst
        if cache is not None:
            cache.edge_src = edge_src
            cache.edge_dst = edge_dst
    if edge_src is not None:
        data["edge_count"] = int(edge_src.size)

    cached_deg_array = data.get("deg_array")

    # Memory optimization: When we have a cached degree array and need a count
    # buffer, we can reuse the degree array buffer as the destination for counts.
    # This works because:
    #   1. For undirected graphs, node degree equals neighbor count
    #   2. The degree array is already allocated and the right size
    #   3. We avoid allocating an extra row in the accumulator matrix
    # When reuse_count_from_deg is True:
    #   - We copy cached_deg_array into the count buffer before accumulation
    #   - We pass count_for_accum=None to _accumulate_neighbors_broadcasted
    #   - After accumulation, we restore count = cached_deg_array (line 2121)
    reuse_count_from_deg = bool(count is not None and cached_deg_array is not None)
    count_for_accum = count
    if count is not None:
        if reuse_count_from_deg:
            # Pre-fill count with degrees (will be returned as-is since accumulator
            # skips the count row when count_for_accum=None)
            np.copyto(count, cached_deg_array, casting="unsafe")
            count_for_accum = None
        else:
            count.fill(0.0)

    deg_array = None
    if deg_sum is not None:
        deg_sum.fill(0.0)
        deg_array = _resolve_numpy_degree_array(
            data, count if count is not None else None, cache=cache, np=np
        )
    elif cached_deg_array is not None:
        deg_array = cached_deg_array

    edge_count = int(edge_src.size) if edge_src is not None else 0
    chunk_hint = data.get("neighbor_chunk_hint")
    if chunk_hint is None:
        chunk_hint = G.graph.get("DNFR_CHUNK_SIZE")
    resolved_neighbor_chunk = (
        resolve_chunk_size(
            chunk_hint,
            edge_count,
            minimum=1,
            approx_bytes_per_item=_DNFR_APPROX_BYTES_PER_EDGE,
            clamp_to=None,
        )
        if edge_count
        else 0
    )
    data["neighbor_chunk_hint"] = chunk_hint
    data["neighbor_chunk_size"] = resolved_neighbor_chunk

    accum = _accumulate_neighbors_broadcasted(
        edge_src=edge_src,
        edge_dst=edge_dst,
        cos=cos_th,
        sin=sin_th,
        epi=epi,
        vf=vf,
        x=x,
        y=y,
        epi_sum=epi_sum,
        vf_sum=vf_sum,
        count=count_for_accum,
        deg_sum=deg_sum,
        deg_array=deg_array,
        cache=cache,
        np=np,
        chunk_size=resolved_neighbor_chunk,
    )

    data["neighbor_accum_np"] = accum.get("accumulator")
    edge_values = accum.get("edge_values")
    data["neighbor_edge_values_np"] = edge_values
    if edge_values is not None:
        width = getattr(edge_values, "shape", (0, 0))[1]
        data["neighbor_chunk_size"] = int(width)
    else:
        data["neighbor_chunk_size"] = resolved_neighbor_chunk
    if cache is not None:
        data["neighbor_accum_signature"] = cache.neighbor_accum_signature
    if reuse_count_from_deg and cached_deg_array is not None:
        count = cached_deg_array
    degs = deg_array if deg_sum is not None and deg_array is not None else None
    return x, y, epi_sum, vf_sum, count, deg_sum, degs


def _compute_dnfr(
    G: TNFRGraph,
    data: MutableMapping[str, Any],
    *,
    use_numpy: bool | None = None,
    n_jobs: int | None = None,
    profile: MutableMapping[str, float] | None = None,
) -> None:
    """Compute ΔNFR using neighbour sums.

    Parameters
    ----------
    G : nx.Graph
        Graph on which the computation is performed.
    data : dict
        Precomputed ΔNFR data as returned by :func:`_prepare_dnfr_data`.
    use_numpy : bool | None, optional
        Backwards compatibility flag. When ``True`` the function eagerly
        prepares NumPy buffers (if available). When ``False`` the engine still
        prefers the vectorised path whenever :func:`get_numpy` returns a module
        and the graph does not set ``vectorized_dnfr`` to ``False``.
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping that aggregates wall-clock durations for neighbour
        accumulation and records which execution branch was used. The
        ``"dnfr_neighbor_accumulation"`` bucket gathers the time spent inside
        :func:`_build_neighbor_sums_common`, while ``"dnfr_path"`` stores the
        string ``"vectorized"`` or ``"fallback"`` describing the active
        implementation.
    """
    start_timer, stop_timer = _profile_start_stop(
        profile,
        keys=("dnfr_neighbor_accumulation",),
    )

    np_module = get_numpy()
    data["dnfr_numpy_available"] = bool(np_module)
    vector_disabled = G.graph.get("vectorized_dnfr") is False
    prefer_dense = np_module is not None and not vector_disabled
    if use_numpy is True and np_module is not None:
        prefer_dense = True
    if use_numpy is False or vector_disabled:
        prefer_dense = False
    data["dnfr_used_numpy"] = bool(prefer_dense and np_module is not None)
    if profile is not None:
        profile["dnfr_path"] = "vectorized" if data["dnfr_used_numpy"] else "fallback"

    data["n_jobs"] = n_jobs
    try:
        neighbor_timer = start_timer()
        res = _build_neighbor_sums_common(
            G,
            data,
            use_numpy=prefer_dense,
            n_jobs=n_jobs,
        )
        stop_timer("dnfr_neighbor_accumulation", neighbor_timer)
    except TypeError as exc:
        if "n_jobs" not in str(exc):
            raise
        neighbor_timer = start_timer()
        res = _build_neighbor_sums_common(
            G,
            data,
            use_numpy=prefer_dense,
        )
        stop_timer("dnfr_neighbor_accumulation", neighbor_timer)
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
        n_jobs=n_jobs,
        profile=profile,
    )


def default_compute_delta_nfr(
    G: TNFRGraph,
    *,
    cache_size: int | None = 1,
    n_jobs: int | None = None,
    profile: MutableMapping[str, float] | None = None,
) -> None:
    """Compute ΔNFR by mixing phase, EPI, νf and a topological term.

    Parameters
    ----------
    G : nx.Graph
        Graph on which the computation is performed.
    cache_size : int | None, optional
        Maximum number of edge configurations cached in ``G.graph``. Values
        ``None`` or <= 0 imply unlimited cache. Defaults to ``1`` to keep the
        previous behaviour.
    n_jobs : int | None, optional
        Parallel worker count for the pure-Python accumulation path. ``None``
        or values <= 1 preserve the serial behaviour. The vectorised NumPy
        branch ignores this parameter as it already operates in bulk.
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping that aggregates the wall-clock timings captured during
        the ΔNFR computation. The mapping receives the buckets documented in
        :func:`_prepare_dnfr_data` and :func:`_compute_dnfr`, plus
        ``"dnfr_neighbor_means"``, ``"dnfr_gradient_assembly"`` and
        ``"dnfr_inplace_write"`` describing the internal stages of
        :func:`_compute_dnfr_common`. ``"dnfr_path"`` reflects whether the
        vectorised or fallback implementation executed the call.
    """
    if profile is not None:
        for key in (
            "dnfr_cache_rebuild",
            "dnfr_neighbor_accumulation",
            "dnfr_neighbor_means",
            "dnfr_gradient_assembly",
            "dnfr_inplace_write",
        ):
            profile.setdefault(key, 0.0)

    data = _prepare_dnfr_data(G, cache_size=cache_size, profile=profile)
    _write_dnfr_metadata(
        G,
        weights=data["weights"],
        hook_name="default_compute_delta_nfr",
    )
    _compute_dnfr(G, data, n_jobs=n_jobs, profile=profile)
    if not data.get("dnfr_numpy_available"):
        cache = data.get("cache")
        cache_size = data.get("cache_size")
        caching_enabled = isinstance(cache, DnfrCache) and (
            cache_size is None or int(cache_size) > 0
        )
        if isinstance(cache, DnfrCache) and not caching_enabled:
            for attr in (
                "neighbor_x_np",
                "neighbor_y_np",
                "neighbor_epi_sum_np",
                "neighbor_vf_sum_np",
                "neighbor_count_np",
                "neighbor_deg_sum_np",
                "neighbor_inv_count_np",
                "neighbor_cos_avg_np",
                "neighbor_sin_avg_np",
                "neighbor_mean_tmp_np",
                "neighbor_mean_length_np",
                "neighbor_accum_np",
                "neighbor_edge_values_np",
            ):
                setattr(cache, attr, None)
            cache.neighbor_accum_signature = None


def set_delta_nfr_hook(
    G: TNFRGraph,
    func: DeltaNFRHook,
    *,
    name: str | None = None,
    note: str | None = None,
) -> None:
    """Set a stable hook to compute ΔNFR.

    The callable should accept ``(G, *[, n_jobs])`` and is responsible for
    writing ``ALIAS_DNFR`` in each node. ``n_jobs`` is optional and ignored by
    hooks that do not support parallel execution. Basic metadata in
    ``G.graph`` is updated accordingly.
    """

    def _wrapped(graph: TNFRGraph, *args: Any, **kwargs: Any) -> None:
        if "n_jobs" in kwargs:
            try:
                func(graph, *args, **kwargs)
                return
            except TypeError as exc:
                if "n_jobs" not in str(exc):
                    raise
                kwargs = dict(kwargs)
                kwargs.pop("n_jobs", None)
                func(graph, *args, **kwargs)
                return
        func(graph, *args, **kwargs)

    _wrapped.__name__ = getattr(func, "__name__", "custom_dnfr")
    _wrapped.__doc__ = getattr(func, "__doc__", _wrapped.__doc__)

    G.graph["compute_delta_nfr"] = _wrapped
    G.graph["_dnfr_hook_name"] = str(name or getattr(func, "__name__", "custom_dnfr"))
    if "_dnfr_weights" not in G.graph:
        _configure_dnfr_weights(G)
    if note:
        meta = G.graph.get("_DNFR_META", {})
        meta["note"] = str(note)
        G.graph["_DNFR_META"] = meta


def _dnfr_hook_chunk_worker(
    G: TNFRGraph,
    node_ids: Sequence[NodeId],
    grad_items: tuple[
        tuple[str, Callable[[TNFRGraph, NodeId, Mapping[str, Any]], float]],
        ...,
    ],
    weights: Mapping[str, float],
) -> list[tuple[NodeId, float]]:
    """Compute weighted gradients for ``node_ids``.

    The helper is defined at module level so it can be pickled by
    :class:`concurrent.futures.ProcessPoolExecutor`.
    """

    results: list[tuple[NodeId, float]] = []
    for node in node_ids:
        nd = G.nodes[node]
        total = 0.0
        for name, func in grad_items:
            w = weights.get(name, 0.0)
            if w:
                total += w * float(func(G, node, nd))
        results.append((node, total))
    return results


def _apply_dnfr_hook(
    G: TNFRGraph,
    grads: Mapping[str, Callable[[TNFRGraph, NodeId, Mapping[str, Any]], float]],
    *,
    weights: Mapping[str, float],
    hook_name: str,
    note: str | None = None,
    n_jobs: int | None = None,
) -> None:
    """Compute and store ΔNFR using ``grads``.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes will receive the ΔNFR update.
    grads : dict
        Mapping from component names to callables with signature
        ``(G, node, data) -> float`` returning the gradient contribution.
    weights : dict
        Weight per component; missing entries default to ``0``.
    hook_name : str
        Friendly identifier stored in ``G.graph`` metadata.
    note : str | None, optional
        Additional documentation recorded next to the hook metadata.
    n_jobs : int | None, optional
        Optional worker count for the pure-Python execution path. When NumPy
        is available the helper always prefers the vectorised implementation
        and ignores ``n_jobs`` because the computation already happens in
        bulk.
    """

    nodes_data: list[tuple[NodeId, Mapping[str, Any]]] = list(G.nodes(data=True))
    if not nodes_data:
        _write_dnfr_metadata(G, weights=weights, hook_name=hook_name, note=note)
        return

    np_module = cast(ModuleType | None, get_numpy())
    if np_module is not None:
        totals = np_module.zeros(len(nodes_data), dtype=float)
        for name, func in grads.items():
            w = float(weights.get(name, 0.0))
            if w == 0.0:
                continue
            values = np_module.fromiter(
                (float(func(G, n, nd)) for n, nd in nodes_data),
                dtype=float,
                count=len(nodes_data),
            )
            if w == 1.0:
                np_module.add(totals, values, out=totals)
            else:
                np_module.add(totals, values * w, out=totals)
        for idx, (n, _) in enumerate(nodes_data):
            set_dnfr(G, n, float(totals[idx]))
        _write_dnfr_metadata(G, weights=weights, hook_name=hook_name, note=note)
        return

    effective_jobs = _resolve_parallel_jobs(n_jobs, len(nodes_data))
    results: list[tuple[NodeId, float]] | None = None
    if effective_jobs:
        grad_items = tuple(grads.items())
        # ProcessPoolExecutor requires picklable arguments. Instead of explicitly
        # testing with pickle.dumps (which poses security risks), we attempt
        # parallelization and gracefully fall back to serial on any failure.
        try:
            chunk_results: list[tuple[NodeId, float]] = []
            with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
                futures = []
                node_ids: list[NodeId] = [n for n, _ in nodes_data]
                for start, end in _iter_chunk_offsets(len(node_ids), effective_jobs):
                    if start == end:
                        continue
                    futures.append(
                        executor.submit(
                            _dnfr_hook_chunk_worker,
                            G,
                            node_ids[start:end],
                            grad_items,
                            weights,
                        )
                    )
                for future in futures:
                    chunk_results.extend(future.result())
            results = chunk_results
        except Exception:
            # Parallel execution failed (pickle, executor, or worker error)
            # Fall back to serial processing
            results = None

    if results is None:
        results = []
        for n, nd in nodes_data:
            total = 0.0
            for name, func in grads.items():
                w = weights.get(name, 0.0)
                if w:
                    total += w * float(func(G, n, nd))
            results.append((n, total))

    for node, value in results:
        set_dnfr(G, node, float(value))

    _write_dnfr_metadata(G, weights=weights, hook_name=hook_name, note=note)


# --- Example hooks (optional) ---


class _PhaseGradient:
    """Callable computing the phase contribution using cached trig values."""

    __slots__ = ("cos", "sin")

    def __init__(
        self,
        cos_map: Mapping[NodeId, float],
        sin_map: Mapping[NodeId, float],
    ) -> None:
        self.cos: Mapping[NodeId, float] = cos_map
        self.sin: Mapping[NodeId, float] = sin_map

    def __call__(
        self,
        G: TNFRGraph,
        n: NodeId,
        nd: Mapping[str, Any],
    ) -> float:
        theta_val = get_theta_attr(nd, 0.0)
        th_i = float(theta_val if theta_val is not None else 0.0)
        neighbors = list(G.neighbors(n))
        if neighbors:
            th_bar = neighbor_phase_mean_list(
                neighbors,
                cos_th=self.cos,
                sin_th=self.sin,
                fallback=th_i,
            )
        else:
            th_bar = th_i
        return -angle_diff(th_i, th_bar) / math.pi


class _NeighborAverageGradient:
    """Callable computing neighbour averages for scalar attributes."""

    __slots__ = ("alias", "values")

    def __init__(
        self,
        alias: tuple[str, ...],
        values: MutableMapping[NodeId, float],
    ) -> None:
        self.alias: tuple[str, ...] = alias
        self.values: MutableMapping[NodeId, float] = values

    def __call__(
        self,
        G: TNFRGraph,
        n: NodeId,
        nd: Mapping[str, Any],
    ) -> float:
        val = self.values.get(n)
        if val is None:
            val = float(get_attr(nd, self.alias, 0.0))
            self.values[n] = val
        neighbors = list(G.neighbors(n))
        if not neighbors:
            return 0.0
        total = 0.0
        for neigh in neighbors:
            neigh_val = self.values.get(neigh)
            if neigh_val is None:
                neigh_val = float(get_attr(G.nodes[neigh], self.alias, val))
                self.values[neigh] = neigh_val
            total += neigh_val
        return total / len(neighbors) - val


def dnfr_phase_only(G: TNFRGraph, *, n_jobs: int | None = None) -> None:
    """Compute ΔNFR from phase only (Kuramoto-like).

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes receive the ΔNFR assignment.
    n_jobs : int | None, optional
        Parallel worker hint used when NumPy is unavailable. Defaults to
        serial execution.
    """

    trig = compute_theta_trig(G.nodes(data=True))
    g_phase = _PhaseGradient(trig.cos, trig.sin)
    _apply_dnfr_hook(
        G,
        {"phase": g_phase},
        weights={"phase": 1.0},
        hook_name="dnfr_phase_only",
        note="Example hook.",
        n_jobs=n_jobs,
    )


def dnfr_epi_vf_mixed(G: TNFRGraph, *, n_jobs: int | None = None) -> None:
    """Compute ΔNFR without phase, mixing EPI and νf.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes receive the ΔNFR assignment.
    n_jobs : int | None, optional
        Parallel worker hint used when NumPy is unavailable. Defaults to
        serial execution.
    """

    epi_values = {n: float(get_attr(nd, ALIAS_EPI, 0.0)) for n, nd in G.nodes(data=True)}
    vf_values = {n: float(get_attr(nd, ALIAS_VF, 0.0)) for n, nd in G.nodes(data=True)}
    grads = {
        "epi": _NeighborAverageGradient(ALIAS_EPI, epi_values),
        "vf": _NeighborAverageGradient(ALIAS_VF, vf_values),
    }
    _apply_dnfr_hook(
        G,
        grads,
        weights={"phase": 0.0, "epi": 0.5, "vf": 0.5},
        hook_name="dnfr_epi_vf_mixed",
        note="Example hook.",
        n_jobs=n_jobs,
    )


def dnfr_laplacian(G: TNFRGraph, *, n_jobs: int | None = None) -> None:
    """Explicit topological gradient using Laplacian over EPI and νf.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes receive the ΔNFR assignment.
    n_jobs : int | None, optional
        Parallel worker hint used when NumPy is unavailable. Defaults to
        serial execution.
    """

    weights_cfg = get_param(G, "DNFR_WEIGHTS")
    wE = float(weights_cfg.get("epi", DEFAULTS["DNFR_WEIGHTS"]["epi"]))
    wV = float(weights_cfg.get("vf", DEFAULTS["DNFR_WEIGHTS"]["vf"]))

    epi_values = {n: float(get_attr(nd, ALIAS_EPI, 0.0)) for n, nd in G.nodes(data=True)}
    vf_values = {n: float(get_attr(nd, ALIAS_VF, 0.0)) for n, nd in G.nodes(data=True)}
    grads = {
        "epi": _NeighborAverageGradient(ALIAS_EPI, epi_values),
        "vf": _NeighborAverageGradient(ALIAS_VF, vf_values),
    }
    _apply_dnfr_hook(
        G,
        grads,
        weights={"epi": wE, "vf": wV},
        hook_name="dnfr_laplacian",
        note="Topological gradient",
        n_jobs=n_jobs,
    )


def compute_delta_nfr_hamiltonian(
    G: TNFRGraph,
    *,
    hbar_str: float | None = None,
    cache_hamiltonian: bool = True,
    profile: MutableMapping[str, float] | None = None,
) -> None:
    r"""Compute ΔNFR using rigorous Hamiltonian commutator formulation.

    This is the **canonical** TNFR method that constructs the internal
    Hamiltonian H_int = H_coh + H_freq + H_coupling explicitly and computes
    ΔNFR from the quantum commutator:

    .. math::
        \Delta\text{NFR}_n = \frac{i}{\hbar_{str}} \langle n | [\hat{H}_{int}, \rho_n] | n \rangle

    where \rho_n = |n\rangle\langle n| is the density matrix for node n.

    Theory
    ------

    The internal Hamiltonian governs structural evolution through:

    .. math::
        \frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)

    with the reorganization operator defined as:

    .. math::
        \Delta\text{NFR} = \frac{d}{dt} + \frac{i[\hat{H}_{int}, \cdot]}{\hbar_{str}}

    **Components**:

    1. **H_coh**: Coherence potential from structural similarity
    2. **H_freq**: Diagonal frequency operator (νf per node)
    3. **H_coupling**: Network topology-induced interactions

    Parameters
    ----------
    G : TNFRGraph
        Graph with nodes containing 'nu_f', 'phase', 'epi', 'si' attributes
    hbar_str : float, optional
        Structural Planck constant (ℏ_str). If None, uses
        ``G.graph.get('HBAR_STR', 1.0)``. Natural units (1.0) make the
        Hamiltonian directly represent structural energy scales.
    cache_hamiltonian : bool, default=True
        If True, caches the Hamiltonian in ``G.graph['_hamiltonian_cache']``
        for reuse in subsequent calls. Set to False for dynamic networks
        where topology changes frequently.
    profile : MutableMapping[str, float] or None, optional
        Mutable mapping that accumulates wall-clock timings:

        - ``"hamiltonian_construction"``: Time to build H_int
        - ``"hamiltonian_computation"``: Time to compute all ΔNFR values
        - ``"hamiltonian_write"``: Time to write results to nodes

    Notes
    -----

    **Advantages over heuristic methods**:

    - **Rigorous**: Directly implements TNFR mathematical formalization
    - **Hermitian**: Guarantees real eigenvalues and unitary evolution
    - **Verifiable**: Can compute energy spectrum and eigenstates
    - **Complete**: Accounts for all structural correlations via coherence matrix

    **Performance considerations**:

    - Complexity: O(N²) for matrix construction, O(N³) for eigendecomposition
    - Recommended for networks with N < 1000 nodes
    - For larger networks, use default_compute_delta_nfr (heuristic, O(E))

    **Cache behavior**:

    - Hamiltonian is cached if ``cache_hamiltonian=True``
    - Cache is invalidated when node attributes or topology change
    - Uses ``CacheManager`` for consistency with other TNFR computations

    Examples
    --------

    **Basic usage**:

    >>> import networkx as nx
    >>> from tnfr.dynamics.dnfr import compute_delta_nfr_hamiltonian
    >>> G = nx.cycle_graph(10)
    >>> for node in G.nodes:
    ...     G.nodes[node].update({
    ...         'nu_f': 1.0, 'phase': 0.0, 'epi': 1.0, 'si': 0.8
    ...     })
    >>> compute_delta_nfr_hamiltonian(G)
    >>> # ΔNFR values now stored in G.nodes[n]['delta_nfr']

    **With profiling**:

    >>> profile = {}
    >>> compute_delta_nfr_hamiltonian(G, profile=profile)
    >>> print(f"Construction: {profile['hamiltonian_construction']:.3f}s")
    >>> print(f"Computation: {profile['hamiltonian_computation']:.3f}s")

    **Integration with dynamics**:

    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> set_delta_nfr_hook(G, compute_delta_nfr_hamiltonian, name="hamiltonian")
    >>> # Now simulate() will use Hamiltonian-based ΔNFR

    See Also
    --------
    tnfr.operators.hamiltonian.InternalHamiltonian : Core Hamiltonian class
    default_compute_delta_nfr : Heuristic O(E) method for large networks
    set_delta_nfr_hook : Register custom ΔNFR computation

    References
    ----------

    - Mathematical formalization: ``Formalizacion-Matematica-TNFR-Unificada.pdf`` §2.4
    - ΔNFR development: ``Desarrollo-Exhaustivo_-Formalizacion-Matematica-Ri-3.pdf``
    """
    from ..operators.hamiltonian import InternalHamiltonian

    # Initialize profiling
    start_timer, stop_timer = _profile_start_stop(
        profile,
        keys=(
            "hamiltonian_construction",
            "hamiltonian_computation",
            "hamiltonian_write",
        ),
    )

    # Get structural Planck constant
    if hbar_str is None:
        hbar_str = G.graph.get("HBAR_STR", 1.0)

    # Check cache for existing Hamiltonian
    cache_key = "_hamiltonian_cache"
    ham = None

    if cache_hamiltonian:
        cached_ham = G.graph.get(cache_key)
        # Verify cache validity (node count and checksum)
        if cached_ham is not None:
            current_checksum = G.graph.get("_dnfr_nodes_checksum")
            cached_checksum = getattr(cached_ham, "_cache_checksum", None)
            if (
                isinstance(cached_ham, InternalHamiltonian)
                and cached_ham.N == G.number_of_nodes()
                and current_checksum == cached_checksum
            ):
                ham = cached_ham

    # Construct Hamiltonian if not cached or invalid
    if ham is None:
        timer = start_timer()

        # Get cache manager for integration with existing infrastructure
        manager = _graph_cache_manager(G.graph)

        # Build Hamiltonian
        ham = InternalHamiltonian(G, hbar_str=float(hbar_str), cache_manager=manager)

        # Cache for reuse
        if cache_hamiltonian:
            ham._cache_checksum = G.graph.get("_dnfr_nodes_checksum")
            G.graph[cache_key] = ham

        stop_timer("hamiltonian_construction", timer)

    # Compute ΔNFR for all nodes
    timer = start_timer()

    delta_nfr_values = {}
    for node in ham.nodes:
        delta_nfr = ham.compute_node_delta_nfr(node)
        delta_nfr_values[node] = delta_nfr

    stop_timer("hamiltonian_computation", timer)

    # Write results to graph nodes
    timer = start_timer()

    for node, delta_val in delta_nfr_values.items():
        set_dnfr(G, node, delta_val)

    stop_timer("hamiltonian_write", timer)

    # Write metadata
    _write_dnfr_metadata(
        G,
        weights={"hamiltonian": 1.0},
        hook_name="compute_delta_nfr_hamiltonian",
        note="Canonical Hamiltonian commutator formulation",
    )
