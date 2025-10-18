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
from dataclasses import dataclass
from typing import Any, Callable

from ..alias import (
    get_attr,
    set_dnfr,
)
from ..constants import DEFAULTS, get_aliases, get_param
from ..helpers.numeric import angle_diff
from ..metrics.common import merge_and_normalize_weights
from ..metrics.trig import neighbor_phase_mean
from ..metrics.trig_cache import compute_theta_trig
from ..utils import cached_node_list, cached_nodes_and_A, get_numpy, normalize_weights
ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


_MEAN_VECTOR_EPS = 1e-12
_SPARSE_DENSITY_THRESHOLD = 0.25



def _should_vectorize(G, np_module) -> bool:
    """Return ``True`` when NumPy is available unless the graph disables it."""

    if np_module is None:
        return False
    flag = G.graph.get("vectorized_dnfr")
    if flag is None:
        return True
    return bool(flag)




@dataclass
class DnfrCache:
    idx: dict[Any, int]
    theta: list[float]
    epi: list[float]
    vf: list[float]
    cos_theta: list[float]
    sin_theta: list[float]
    neighbor_x: list[float]
    neighbor_y: list[float]
    neighbor_epi_sum: list[float]
    neighbor_vf_sum: list[float]
    neighbor_count: list[float]
    neighbor_deg_sum: list[float] | None
    th_bar: list[float] | None = None
    epi_bar: list[float] | None = None
    vf_bar: list[float] | None = None
    deg_bar: list[float] | None = None
    degs: dict[Any, float] | None = None
    deg_list: list[float] | None = None
    theta_np: Any | None = None
    epi_np: Any | None = None
    vf_np: Any | None = None
    cos_theta_np: Any | None = None
    sin_theta_np: Any | None = None
    deg_array: Any | None = None
    edge_src: Any | None = None
    edge_dst: Any | None = None
    checksum: Any | None = None
    neighbor_x_np: Any | None = None
    neighbor_y_np: Any | None = None
    neighbor_epi_sum_np: Any | None = None
    neighbor_vf_sum_np: Any | None = None
    neighbor_count_np: Any | None = None
    neighbor_deg_sum_np: Any | None = None
    th_bar_np: Any | None = None
    epi_bar_np: Any | None = None
    vf_bar_np: Any | None = None
    deg_bar_np: Any | None = None
    grad_phase_np: Any | None = None
    grad_epi_np: Any | None = None
    grad_vf_np: Any | None = None
    grad_topo_np: Any | None = None
    grad_total_np: Any | None = None
    dense_components_np: Any | None = None
    dense_accum_np: Any | None = None
    dense_degree_np: Any | None = None
    neighbor_contrib_np: Any | None = None
    neighbor_workspace_np: Any | None = None
    neighbor_inv_count_np: Any | None = None
    neighbor_cos_avg_np: Any | None = None
    neighbor_sin_avg_np: Any | None = None
    neighbor_mean_tmp_np: Any | None = None
    neighbor_mean_length_np: Any | None = None
    edge_signature: Any | None = None
    neighbor_accum_signature: Any | None = None


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
    "neighbor_contrib_np",
    "neighbor_workspace_np",
    "dense_components_np",
    "dense_accum_np",
    "dense_degree_np",
)

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
)


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
    weights = merge_and_normalize_weights(
        G, "DNFR_WEIGHTS", ("phase", "epi", "vf", "topo"), default=0.0
    )
    G.graph["_dnfr_weights"] = weights
    return weights


def _init_dnfr_cache(G, nodes, prev_cache: DnfrCache | None, checksum, dirty):
    """Initialise or reuse cached ΔNFR arrays."""
    if prev_cache and prev_cache.checksum == checksum and not dirty:
        return (
            prev_cache,
            prev_cache.idx,
            prev_cache.theta,
            prev_cache.epi,
            prev_cache.vf,
            prev_cache.cos_theta,
            prev_cache.sin_theta,
            False,
        )

    idx = {n: i for i, n in enumerate(nodes)}
    theta = [0.0] * len(nodes)
    epi = [0.0] * len(nodes)
    vf = [0.0] * len(nodes)
    cos_theta = [1.0] * len(nodes)
    sin_theta = [0.0] * len(nodes)
    neighbor_x = [0.0] * len(nodes)
    neighbor_y = [0.0] * len(nodes)
    neighbor_epi_sum = [0.0] * len(nodes)
    neighbor_vf_sum = [0.0] * len(nodes)
    neighbor_count = [0.0] * len(nodes)
    neighbor_deg_sum = [0.0] * len(nodes) if len(nodes) else []
    cache = DnfrCache(
        idx=idx,
        theta=theta,
        epi=epi,
        vf=vf,
        cos_theta=cos_theta,
        sin_theta=sin_theta,
        neighbor_x=neighbor_x,
        neighbor_y=neighbor_y,
        neighbor_epi_sum=neighbor_epi_sum,
        neighbor_vf_sum=neighbor_vf_sum,
        neighbor_count=neighbor_count,
        neighbor_deg_sum=neighbor_deg_sum,
        degs=prev_cache.degs if prev_cache else None,
        edge_src=None,
        edge_dst=None,
        checksum=checksum,
    )
    G.graph["_dnfr_prep_cache"] = cache
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


def _ensure_numpy_vectors(cache: DnfrCache, np):
    """Ensure NumPy copies of cached vectors are initialised and up to date."""

    if cache is None:
        return (None, None, None, None, None)

    arrays = []
    for attr_np, source_attr in (
        ("theta_np", "theta"),
        ("epi_np", "epi"),
        ("vf_np", "vf"),
        ("cos_theta_np", "cos_theta"),
        ("sin_theta_np", "sin_theta"),
    ):
        src = getattr(cache, source_attr)
        arr = getattr(cache, attr_np)
        if src is None:
            setattr(cache, attr_np, None)
            arrays.append(None)
            continue
        if arr is None or len(arr) != len(src):
            arr = np.array(src, dtype=float)
        else:
            np.copyto(arr, src, casting="unsafe")
        setattr(cache, attr_np, arr)
        arrays.append(arr)
    return tuple(arrays)


def _ensure_numpy_degrees(cache: DnfrCache, deg_list, np):
    """Initialise/update NumPy array mirroring ``deg_list``."""

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


def _resolve_numpy_degree_array(data, count, *, cache: DnfrCache | None, np):
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


def _ensure_cached_array(cache: DnfrCache | None, attr: str, shape, np):
    """Return a cached NumPy buffer with ``shape`` creating/reusing it."""

    if np is None:
        raise RuntimeError("NumPy is required to build cached arrays")
    arr = getattr(cache, attr) if cache is not None else None
    if arr is None or getattr(arr, "shape", None) != shape:
        arr = np.empty(shape, dtype=float)
        if cache is not None:
            setattr(cache, attr, arr)
    return arr


def _ensure_numpy_state_vectors(data: dict, np):
    """Synchronise list-based state vectors with their NumPy counterparts."""

    nodes = data.get("nodes") or ()
    size = len(nodes)
    cache: DnfrCache | None = data.get("cache")

    if cache is not None:
        theta_np, epi_np, vf_np, cos_np, sin_np = _ensure_numpy_vectors(cache, np)
        for key, arr in (
            ("theta_np", theta_np),
            ("epi_np", epi_np),
            ("vf_np", vf_np),
            ("cos_theta_np", cos_np),
            ("sin_theta_np", sin_np),
        ):
            if arr is not None and getattr(arr, "shape", None) == (size,):
                data[key] = arr

    mapping = (
        ("theta_np", "theta"),
        ("epi_np", "epi"),
        ("vf_np", "vf"),
        ("cos_theta_np", "cos_theta"),
        ("sin_theta_np", "sin_theta"),
    )
    for np_key, src_key in mapping:
        src = data.get(src_key)
        if src is None:
            continue
        arr = data.get(np_key)
        if arr is None or getattr(arr, "shape", None) != (size,):
            arr = np.array(src, dtype=float)
        elif cache is None:
            np.copyto(arr, src, casting="unsafe")
        data[np_key] = arr
        if cache is not None:
            setattr(cache, np_key, arr)

    return {
        "theta": data.get("theta_np"),
        "epi": data.get("epi_np"),
        "vf": data.get("vf_np"),
        "cos": data.get("cos_theta_np"),
        "sin": data.get("sin_theta_np"),
    }


def _build_edge_index_arrays(G, nodes, idx, np):
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


def _refresh_dnfr_vectors(G, nodes, cache: DnfrCache):
    """Update cached angle and state vectors for ΔNFR."""
    np_module = get_numpy()
    trig = compute_theta_trig(((n, G.nodes[n]) for n in nodes), np=np_module)
    use_numpy = _should_vectorize(G, np_module)
    for i, n in enumerate(nodes):
        nd = G.nodes[n]
        cache.theta[i] = trig.theta[n]
        cache.epi[i] = get_attr(nd, ALIAS_EPI, 0.0)
        cache.vf[i] = get_attr(nd, ALIAS_VF, 0.0)
        cache.cos_theta[i] = trig.cos[n]
        cache.sin_theta[i] = trig.sin[n]
    if use_numpy:
        _ensure_numpy_vectors(cache, np_module)
    else:
        cache.theta_np = None
        cache.epi_np = None
        cache.vf_np = None
        cache.cos_theta_np = None
        cache.sin_theta_np = None


def _prepare_dnfr_data(G, *, cache_size: int | None = 128) -> dict:
    """Precompute common data for ΔNFR strategies.

    The helper decides between edge-wise and dense adjacency accumulation
    heuristically.  Graphs whose edge density exceeds
    ``_SPARSE_DENSITY_THRESHOLD`` receive a cached adjacency matrix so the
    dense path can be exercised; callers may also force the dense mode by
    setting ``G.graph['dnfr_force_dense']`` to a truthy value.
    """
    weights = G.graph.get("_dnfr_weights")
    if weights is None:
        weights = _configure_dnfr_weights(G)

    np_module = get_numpy()
    use_numpy = _should_vectorize(G, np_module)

    nodes = cached_node_list(G)
    edge_count = G.number_of_edges()
    prefer_sparse = False
    dense_override = bool(G.graph.get("dnfr_force_dense"))
    if use_numpy:
        prefer_sparse = _prefer_sparse_accumulation(len(nodes), edge_count)
        if dense_override:
            prefer_sparse = False
    nodes, A = cached_nodes_and_A(
        G,
        cache_size=cache_size,
        require_numpy=False,
        prefer_sparse=prefer_sparse,
        nodes=nodes,
    )
    cache: DnfrCache | None = G.graph.get("_dnfr_prep_cache")
    checksum = G.graph.get("_dnfr_nodes_checksum")
    dirty = bool(G.graph.pop("_dnfr_prep_dirty", False))
    cache, idx, theta, epi, vf, cos_theta, sin_theta, refreshed = (
        _init_dnfr_cache(G, nodes, cache, checksum, dirty)
    )
    if cache is not None:
        _refresh_dnfr_vectors(G, nodes, cache)

    w_phase = float(weights.get("phase", 0.0))
    w_epi = float(weights.get("epi", 0.0))
    w_vf = float(weights.get("vf", 0.0))
    w_topo = float(weights.get("topo", 0.0))
    degree_map: dict[Any, float] | None = cache.degs if cache else None
    if cache is not None and dirty:
        cache.degs = None
        cache.deg_list = None
        cache.deg_array = None
        cache.edge_src = None
        cache.edge_dst = None
        cache.edge_signature = None
        cache.neighbor_accum_signature = None
        degree_map = None
    if degree_map is None or len(degree_map) != len(G):
        degree_map = dict(G.degree())
        if cache is not None:
            cache.degs = degree_map

    G.graph["_dnfr_prep_dirty"] = False

    if cache is not None and cache.deg_list is not None and not dirty and len(cache.deg_list) == len(nodes):
        deg_list: list[float] | None = cache.deg_list
    else:
        deg_list = [float(degree_map.get(node, 0.0)) for node in nodes]
        if cache is not None:
            cache.deg_list = deg_list

    if w_topo != 0.0:
        degs = degree_map
    else:
        degs = None

    deg_array = None
    if np_module is not None and deg_list is not None:
        if cache is not None:
            deg_array = _ensure_numpy_degrees(cache, deg_list, np_module)
        else:
            deg_array = np_module.array(deg_list, dtype=float)
    elif cache is not None:
        cache.deg_array = None

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
                edge_src, edge_dst = _build_edge_index_arrays(
                    G, nodes, idx, np_module
                )
                cache.edge_src = edge_src
                cache.edge_dst = edge_dst
        else:
            edge_src, edge_dst = _build_edge_index_arrays(G, nodes, idx, np_module)

        if cache is not None:
            for attr in (
                "neighbor_workspace_np",
                "neighbor_contrib_np",
            ):
                arr = getattr(cache, attr, None)
                if arr is not None:
                    data[attr] = arr
        if edge_src is not None and edge_dst is not None:
            signature = (id(edge_src), id(edge_dst), len(nodes))
            data["edge_signature"] = signature
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

    return {
        "weights": weights,
        "nodes": nodes,
        "idx": idx,
        "theta": theta,
        "epi": epi,
        "vf": vf,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "theta_np": theta_np,
        "epi_np": epi_np,
        "vf_np": vf_np,
        "cos_theta_np": cos_theta_np,
        "sin_theta_np": sin_theta_np,
        "w_phase": w_phase,
        "w_epi": w_epi,
        "w_vf": w_vf,
        "w_topo": w_topo,
        "degs": degs,
        "deg_list": deg_list,
        "deg_array": deg_array,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "A": A,
        "cache_size": cache_size,
        "cache": cache,
        "edge_count": edge_count,
        "prefer_sparse": prefer_sparse,
        "dense_override": dense_override,
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

    if use_vector:
        grad_phase = _ensure_cached_array(cache, "grad_phase_np", theta_np.shape, np)
        grad_epi = _ensure_cached_array(cache, "grad_epi_np", epi_np.shape, np)
        grad_vf = _ensure_cached_array(cache, "grad_vf_np", vf_np.shape, np)
        grad_total = _ensure_cached_array(cache, "grad_total_np", theta_np.shape, np)
        grad_topo = None
        if w_topo != 0.0:
            grad_topo = _ensure_cached_array(
                cache, "grad_topo_np", deg_array.shape, np
            )

        np.copyto(grad_phase, theta_np, casting="unsafe")
        grad_phase -= th_bar
        grad_phase += math.pi
        np.mod(grad_phase, math.tau, out=grad_phase)
        grad_phase -= math.pi
        grad_phase *= -1.0 / math.pi

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

    for i, n in enumerate(nodes):
        set_dnfr(G, n, float(dnfr_values[i]))


def _init_bar_arrays(data, *, degs=None, np=None):
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
                if (
                    deg_bar is None
                    or getattr(deg_bar, "shape", None) != (deg_size,)
                ):
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
            deg_bar = (
                np.array(degs, dtype=float)
                if w_topo != 0.0 and degs is not None
                else None
            )
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
            deg_bar = (
                list(degs) if w_topo != 0.0 and degs is not None else None
            )
    return th_bar, epi_bar, vf_bar, deg_bar


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
    np=None,
):
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
        np.where(lengths <= _MEAN_VECTOR_EPS, theta_src, temp, out=temp)
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
    np_module = get_numpy()
    if np_module is not None and isinstance(count, getattr(np_module, "ndarray", tuple)):
        np_arg = np_module
    else:
        np_arg = None
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
    _apply_dnfr_gradients(G, data, th_bar, epi_bar, vf_bar, deg_bar, degs)


def _reset_numpy_buffer(buffer, size, np):
    if buffer is None or getattr(buffer, "shape", None) is None or buffer.shape[0] != size:
        return np.zeros(size, dtype=float)
    buffer.fill(0.0)
    return buffer


def _init_neighbor_sums(data, *, np=None):
    """Initialise containers for neighbour sums."""
    nodes = data["nodes"]
    n = len(nodes)
    w_topo = data["w_topo"]
    cache: DnfrCache | None = data.get("cache")

    def _reset_list(buffer, value=0.0):
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
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    np,
):
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
    for col, src_vec in enumerate(vectors):
        np.copyto(components[:, col], src_vec, casting="unsafe")

    accum = _ensure_cached_array(cache, "dense_accum_np", (n, 4), np)
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
    edge_src,
    edge_dst,
    cos,
    sin,
    epi,
    vf,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    deg_array,
    cache: DnfrCache | None,
    np,
):
    """Accumulate neighbour contributions across all columns in one pass."""

    n = x.shape[0]
    edge_count = int(edge_src.size)

    columns = 5  # cos, sin, epi, vf, count
    deg_column = None
    if deg_sum is not None and deg_array is not None:
        deg_column = columns
        columns += 1

    contrib = _ensure_cached_array(cache, "neighbor_contrib_np", (edge_count, columns), np)
    if edge_count:
        np.take(cos, edge_dst, out=contrib[:, 0])
        np.take(sin, edge_dst, out=contrib[:, 1])
        np.take(epi, edge_dst, out=contrib[:, 2])
        np.take(vf, edge_dst, out=contrib[:, 3])
        contrib[:, 4].fill(1.0)
        if deg_column is not None:
            np.take(deg_array, edge_dst, out=contrib[:, deg_column])
    elif getattr(contrib, "size", 0):
        contrib.fill(0.0)

    workspace = _ensure_cached_array(cache, "neighbor_workspace_np", (n, columns), np)
    workspace.fill(0.0)

    if cache is not None:
        base_signature = (id(edge_src), id(edge_dst), n)
        cache.edge_signature = base_signature
        cache.neighbor_accum_signature = (base_signature, columns)

    if edge_count:
        np.add.at(workspace, edge_src, contrib)

    np.copyto(x, workspace[:, 0], casting="unsafe")
    np.copyto(y, workspace[:, 1], casting="unsafe")
    np.copyto(epi_sum, workspace[:, 2], casting="unsafe")
    np.copyto(vf_sum, workspace[:, 3], casting="unsafe")
    if count is not None:
        np.copyto(count, workspace[:, 4], casting="unsafe")
    if deg_column is not None and deg_sum is not None:
        np.copyto(deg_sum, workspace[:, deg_column], casting="unsafe")

    return {
        "workspace": workspace,
        "contrib": contrib,
    }


def _build_neighbor_sums_common(G, data, *, use_numpy: bool):
    """Build neighbour accumulators honouring the requested NumPy path."""

    nodes = data["nodes"]
    np_module = get_numpy() if use_numpy else None
    if np_module is None:
        if not nodes:
            return _init_neighbor_sums(data)

    if np_module is not None:
        if not nodes:
            return _init_neighbor_sums(data, np=np_module)

        x, y, epi_sum, vf_sum, count, deg_sum, degs = _init_neighbor_sums(
            data, np=np_module
        )
        prefer_sparse = data.get("prefer_sparse")
        if prefer_sparse is None:
            prefer_sparse = _prefer_sparse_accumulation(
                len(nodes), data.get("edge_count")
            )
            data["prefer_sparse"] = prefer_sparse

        use_dense = False
        A = data.get("A")
        if not prefer_sparse and A is not None:
            shape = getattr(A, "shape", (0, 0))
            use_dense = shape[0] == len(nodes) and shape[1] == len(nodes)
        if data.get("dense_override") and A is not None:
            shape = getattr(A, "shape", (0, 0))
            if shape[0] == len(nodes) and shape[1] == len(nodes):
                use_dense = True

        accumulator = (
            _accumulate_neighbors_dense if use_dense else _accumulate_neighbors_numpy
        )
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
    np_module = get_numpy()
    if np_module is not None and isinstance(count, getattr(np_module, "ndarray", tuple)):
        np_arg = np_module
    else:
        np_arg = None
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
    _apply_dnfr_gradients(G, data, th_bar, epi_bar, vf_bar, deg_bar, degs)


def _reset_numpy_buffer(buffer, size, np):
    if buffer is None or getattr(buffer, "shape", None) is None or buffer.shape[0] != size:
        return np.zeros(size, dtype=float)
    buffer.fill(0.0)
    return buffer


def _init_neighbor_sums(data, *, np=None):
    """Initialise containers for neighbour sums."""
    nodes = data["nodes"]
    n = len(nodes)
    w_topo = data["w_topo"]
    cache: DnfrCache | None = data.get("cache")

    def _reset_list(buffer, value=0.0):
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
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    np,
):
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
    for col, src_vec in enumerate(vectors):
        np.copyto(components[:, col], src_vec, casting="unsafe")

    accum = _ensure_cached_array(cache, "dense_accum_np", (n, 4), np)
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


def _accumulate_neighbors_numpy(
    G,
    data,
    *,
    x,
    y,
    epi_sum,
    vf_sum,
    count,
    deg_sum,
    np,
):
    """Vectorised neighbour accumulation reusing cached NumPy buffers."""

    nodes = data["nodes"]
    n = len(nodes)
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

    edge_count = int(edge_src.size)

    if count is not None:
        count.fill(0.0)

    deg_array = None
    if deg_sum is not None:
        deg_sum.fill(0.0)
        deg_array = _resolve_numpy_degree_array(
            data, count if count is not None else None, cache=cache, np=np
        )

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
        count=count,
        deg_sum=deg_sum,
        deg_array=deg_array,
        cache=cache,
        np=np,
    )

    data["neighbor_workspace_np"] = accum["workspace"]
    data["neighbor_contrib_np"] = accum["contrib"]
    degs = deg_array if deg_sum is not None and deg_array is not None else None
    return x, y, epi_sum, vf_sum, count, deg_sum, degs


def _compute_dnfr(G, data, *, use_numpy: bool | None = None) -> None:
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
    """
    np_module = get_numpy()
    cache: DnfrCache | None = data.get("cache")

    vector_disabled = G.graph.get("vectorized_dnfr") is False
    vector_flag = np_module is not None and not vector_disabled
    if use_numpy is True and np_module is not None and not vector_disabled:
        vector_flag = True

    if vector_flag:
        _ensure_numpy_state_vectors(data, np_module)

    res = _build_neighbor_sums_common(G, data, use_numpy=vector_flag)
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
    _compute_dnfr(G, data)


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
    grads: dict[str, Callable[[Any, Any], float]],
    *,
    weights: dict[str, float],
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

    _write_dnfr_metadata(G, weights=weights, hook_name=hook_name, note=note)


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
        neighbors = list(G.neighbors(n))
        if neighbors:
            total = 0.0
            for v in neighbors:
                total += float(get_attr(G.nodes[v], ALIAS_EPI, epi_i))
            epi_bar = total / len(neighbors)
        else:
            epi_bar = float(epi_i)
        return epi_bar - epi_i

    def g_vf(G, n, nd):
        vf_i = get_attr(nd, ALIAS_VF, 0.0)
        neighbors = list(G.neighbors(n))
        if neighbors:
            total = 0.0
            for v in neighbors:
                total += float(get_attr(G.nodes[v], ALIAS_VF, vf_i))
            vf_bar = total / len(neighbors)
        else:
            vf_bar = float(vf_i)
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
    weights_cfg = get_param(G, "DNFR_WEIGHTS")
    wE = float(weights_cfg.get("epi", DEFAULTS["DNFR_WEIGHTS"]["epi"]))
    wV = float(weights_cfg.get("vf", DEFAULTS["DNFR_WEIGHTS"]["vf"]))

    def g_epi(G, n, nd):
        epi = get_attr(nd, ALIAS_EPI, 0.0)
        neigh = list(G.neighbors(n))
        deg = len(neigh) or 1
        epi_bar = (
            sum(get_attr(G.nodes[v], ALIAS_EPI, epi) for v in neigh) / deg
        )
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


