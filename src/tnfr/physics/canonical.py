"""TNFR Canonical Structural Fields - Core Implementation

The four CANONICAL structural fields that provide complete multi-scale
characterization of TNFR network state:

- Φ_s: Global structural potential (field theory dimension)
- |∇φ|: Local phase desynchronization (gradient dimension)
- K_φ: Phase curvature / geometric confinement (real part of unified Ψ = K_φ + i·J_φ)
- ξ_C: Coherence length / spatial correlations (correlation dimension)

All fields are read-only telemetry that never mutate EPI.

PRECISION MODE INTEGRATION (Nov 2025):
--------------------------------------
Fields respect global precision_mode from tnfr.config:
- "standard": float64, standard algorithms (default, production)
- "high": float64 + refined quadrature, tighter tolerances
- "research": longdouble where available, publication-grade numerics

**Physics Invariant**: Precision changes affect ONLY numeric details,
NEVER grammar (U1-U6), operator contracts, or coherence semantics.
U6 decisions must be invariant across precision modes.

CACHE INVALIDATION (root cause corrected + fixed, May 2026):
------------------------------------------------------------
compute_structural_potential (and estimate_coherence_length, J_ΔNFR) is
cached via @cache_tnfr_computation with dependencies
{graph_topology, node_dnfr}. The cache key embeds a dependency hash of the
node fields, so changing ΔNFR on a fixed topology MUST invalidate the entry.

**Historical bug (now fixed)**: the dependency hash
(tnfr.utils.cache._compute_dependency_hash) read node values by hardcoded
English keys ('delta_nfr', 'vf', 'epi'), but the canonical writer
(tnfr.alias.set_attr) stores each field under its FIRST alias — the
Greek/canonical key ('ΔNFR', 'νf', 'EPI'). The mismatch made the hash read
None for every node, so the cache key was BLIND to ΔNFR: Φ_s returned stale
values after ANY ΔNFR change (uniform or not), and two distinct graphs with
identical topology but different ΔNFR collided.

**Earlier misdiagnosis (superseded)**: this was previously attributed to
"uniform ΔNFR scaling producing no spatial gradient", with an
alpha-variation (2.0→2.001) workaround to force cache misses. That analysis
was incorrect — Φ_s is linear in ΔNFR (Φ_s(k·ΔNFR) = k·Φ_s), so uniform
scaling DOES change Φ_s and DOES yield a non-zero drift (k−1)·Φ_s; the
zero-drift symptom was entirely the cache bug, not the physics.

**Fix**: tnfr.utils.cache._compute_dependency_hash now resolves
dependencies through the canonical alias tuples (_dependency_alias_keys),
so ΔNFR/νf/EPI changes correctly invalidate dependent caches. No
alpha-variation workaround is needed.

See: tests/physics/test_field_cache_invalidation.py for regression coverage
"""

from __future__ import annotations

import math
from typing import Any

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import precision mode configuration
from ..config import get_precision_mode

# Import TNFR cache system
from ..mathematics.unified_cache import CacheLevel, cache_tnfr_computation

_CACHE_AVAILABLE = True

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]

# Import vectorized operations
try:
    from .vectorized_ops import (
        compute_coherence_length_vectorized,
        compute_phase_gradient_and_curvature_vectorized,
        compute_phi_s_exact_vectorized,
        compute_phi_s_landmarks_vectorized,
    )

    _VECTORIZATION_AVAILABLE = True
except ImportError:
    _VECTORIZATION_AVAILABLE = False

# Import GPU-aware mathematics backend
try:
    from ..mathematics.backend import get_backend

    _GPU_BACKENDS_AVAILABLE = True
except ImportError:
    _GPU_BACKENDS_AVAILABLE = False


def _use_gpu_acceleration(n_nodes: int) -> bool:
    """Determine if GPU acceleration should be used based on problem size.

    Args:
        n_nodes: Number of nodes in the graph

    Returns:
        True if GPU acceleration is beneficial and available
    """
    if not _GPU_BACKENDS_AVAILABLE or n_nodes < 200:
        return False

    try:
        backend = get_backend()
        return backend.supports_autodiff
    except Exception:
        return False


def _gpu_distance_matrix(positions: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """Compute distance matrix on GPU for large graphs.

    Args:
        positions: Node positions array (N, d)
        alpha: Distance exponent

    Returns:
        Distance matrix with 1/d^alpha entries
    """
    if not _GPU_BACKENDS_AVAILABLE:
        raise RuntimeError("GPU backends not available")

    backend = get_backend()

    # Convert to backend tensors
    pos_tensor = backend.as_array(positions)

    # Compute pairwise distances: ||x_i - x_j||^2
    # Using broadcasting: (N,1,d) - (1,N,d) -> (N,N,d)
    pos_i = pos_tensor[:, None, :]  # (N, 1, d)
    pos_j = pos_tensor[None, :, :]  # (1, N, d)
    diff = pos_i - pos_j  # (N, N, d)

    # Squared distances
    dist_sq = backend.einsum("ijd,ijd->ij", diff, diff)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-12
    dist_sq = dist_sq + epsilon

    # Compute 1/d^alpha
    if alpha == 2.0:
        inv_dist = 1.0 / dist_sq
    else:
        dist = backend.einsum("ij->ij", dist_sq**0.5)  # sqrt for distance
        inv_dist = 1.0 / (dist**alpha)

    # set diagonal to zero (self-distances)
    n = positions.shape[0]
    eye = backend.as_array(np.eye(n))
    inv_dist = inv_dist * (1 - eye)

    return backend.to_numpy(inv_dist)


def _get_precision_dtype() -> type:
    """Return numpy dtype based on current precision mode.

    Returns
    -------
    type
        np.float64 (standard/high) or np.longdouble (research)

    Notes
    -----
    Physics invariant: dtype affects numeric accuracy, never semantics.
    Grammar (U1-U6) decisions must be identical across all dtypes.
    """
    mode = get_precision_mode()
    if mode == "research":
        # Use extended precision if available (typically 80-bit on x86)
        return np.longdouble
    else:
        # Standard and high both use float64
        # High mode uses refined algorithms, not different dtype
        return np.float64


# Centralised helpers — single source of truth in _helpers.py
from ._helpers import get_dnfr as _get_dnfr  # noqa: E402
from ._helpers import get_phase as _get_phase  # noqa: E402
from ._helpers import wrap_angle as _wrap_angle  # noqa: E402

_PHI_S_DISTANCE_CACHE: dict[tuple, dict[Any, dict[Any, float]]] = {}


def _graph_topology_hash(G: Any) -> int:
    """Return lightweight topology hash (nodes, edges, degree multiset).

    Hash changes on structural reorganization affecting distances; phase-only
    changes do not alter shortest-path distances and should keep cache valid.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = sorted([d for _, d in G.degree()])
    return hash((num_nodes, num_edges, tuple(degrees)))


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr"},
)
def compute_structural_potential(
    G: Any,
    alpha: float = 2.0,
    *,
    landmark_ratio: float | None = None,
    validate: bool = False,
    error_epsilon: float = 0.05,
    max_refinements: int = 3,
    sample_size: int = 32,
) -> dict[Any, float]:
    """Compute structural potential Φ_s for each locus [CANONICAL].

    Parameters
    ----------
    G : Graph
        TNFR graph with ΔNFR node attributes.
    alpha : float, default 2.0
        Distance exponent (inverse-square analog).
    landmark_ratio : float | None
        Optional override for landmark sampling ratio (0 < r ≤ 0.5). If None,
        canonical size-based heuristic is used.
    validate : bool, default False
        If True, performs adaptive refinement: compares landmark approximation
        against exact potentials on a random node subset (size = sample_size)
        and increases landmark_ratio until relative mean absolute error < ε.
    error_epsilon : float, default 0.05
        Relative mean absolute error (RMAE) threshold for acceptance.
    max_refinements : int, default 3
        Maximum number of landmark_ratio doublings during validation.
    sample_size : int, default 32
        Number of nodes sampled for exact comparison.

    Returns
    -------
    dict[node, float]
        Mapping of node to Φ_s value.

    Canonical Integrity
    -------------------
    - Preserves physical definition: Σ ΔNFR_j / d(i,j)^α.
    - Landmark approximation is a controlled sampling strategy; validation
      enforces bounded error (U6 safety—confinement metrics remain meaningful).
    - Distance cache keyed on topology hash + ratio enables reuse across phase
      changes (phase does not affect shortest-path distances).
    """
    if nx is None:
        raise RuntimeError("networkx required for structural potential computation")

    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Precompute ΔNFR values using TNFR alias system
    delta_nfr = {n: _get_dnfr(G, n) for n in nodes}

    # Choose computation path
    use_landmarks = False
    effective_ratio: float | None = None
    if landmark_ratio is not None:
        effective_ratio = max(0.001, min(0.5, landmark_ratio))
        use_landmarks = True
    else:
        # Heuristic selection based on size bands
        if num_nodes <= 50:
            return _compute_phi_s_exact(G, nodes, delta_nfr, alpha)
        elif num_nodes <= 500:
            return _compute_phi_s_optimized(G, nodes, delta_nfr, alpha)
        else:
            effective_ratio = min(0.1, 50.0 / num_nodes)
            use_landmarks = True

    if not use_landmarks:
        return _compute_phi_s_exact(G, nodes, delta_nfr, alpha)

    # Landmark computation with optional caching and validation
    import random

    topo_hash = _graph_topology_hash(G)
    cache_key = (topo_hash, effective_ratio)
    cached = _PHI_S_DISTANCE_CACHE.get(cache_key)

    def compute_with_ratio(ratio: float) -> dict[Any, float]:
        """Inner landmark pass (rebuild distances only if ratio changed)."""
        nonlocal cached
        if cached is None or cache_key[1] != ratio:
            # Rebuild landmarks & distances
            num_landmarks = max(3, int(len(nodes) * ratio))
            node_scores = []
            for node in nodes:
                degree = G.degree(node)
                dnfr_contrib = abs(delta_nfr[node])
                score = degree * (1.0 + dnfr_contrib)
                node_scores.append((score, node))
            node_scores.sort(reverse=True)
            top_candidates = [n for _, n in node_scores[: num_landmarks * 2]]
            landmarks = random.sample(
                top_candidates, min(num_landmarks, len(top_candidates))
            )
            landmark_distances: dict[Any, dict[Any, float]] = {}
            for landmark in landmarks:
                if G.number_of_edges() > 0:
                    distances = nx.single_source_dijkstra_path_length(
                        G, landmark, weight="weight"
                    )
                else:
                    distances = {landmark: 0.0}
                landmark_distances[landmark] = distances
            cached = landmark_distances
            _PHI_S_DISTANCE_CACHE[(topo_hash, ratio)] = cached
        landmark_distances = cached

        # Use vectorized implementation if available
        if _VECTORIZATION_AVAILABLE:
            landmarks = list(landmark_distances.keys())
            return compute_phi_s_landmarks_vectorized(
                G,
                nodes,
                delta_nfr,
                alpha,
                landmarks,
                landmark_distances,
                dtype=_get_precision_dtype(),
            )

        # Approximate potentials (Python fallback)
        potential: dict[Any, float] = {}
        landmarks = list(landmark_distances.keys())
        for src in nodes:
            total = 0.0
            # Exact contributions from landmarks
            for landmark in landmarks:
                if landmark == src:
                    continue
                d = landmark_distances[landmark].get(src, math.inf)
                if math.isfinite(d) and d > 0.0:
                    total += delta_nfr[landmark] / (d**alpha)
            # Approximate remaining nodes
            for dst in nodes:
                if dst == src or dst in landmarks:
                    continue
                min_approx_dist = math.inf
                for landmark in landmarks:
                    d_land_src = landmark_distances[landmark].get(src, math.inf)
                    d_land_dst = landmark_distances[landmark].get(dst, math.inf)
                    if math.isfinite(d_land_src) and math.isfinite(d_land_dst):
                        approx_dist = abs(d_land_src - d_land_dst)
                        if approx_dist <= 0.0:
                            approx_dist = 1.0
                        if approx_dist < min_approx_dist:
                            min_approx_dist = approx_dist
                if math.isfinite(min_approx_dist) and min_approx_dist > 0.0:
                    total += delta_nfr[dst] / (min_approx_dist**alpha)
            potential[src] = total
        return potential

    current_ratio = effective_ratio if effective_ratio is not None else 0.01
    potential = compute_with_ratio(current_ratio)

    if validate and num_nodes >= 100:
        # Sample subset for exact computation
        import random as _r

        subset = nodes if len(nodes) <= sample_size else _r.sample(nodes, sample_size)
        exact_subset: dict[Any, float] = {}
        dtype = _get_precision_dtype()
        mode = get_precision_mode()
        for src in subset:
            if G.number_of_edges() > 0:
                lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
            else:
                lengths = {src: 0.0}
            total = dtype(0.0)
            for dst in nodes:
                if dst == src:
                    continue
                d = lengths.get(dst, math.inf)
                if not math.isfinite(d) or d <= 0.0:
                    continue
                if mode in ("high", "research"):
                    log_contrib = np.log(abs(delta_nfr[dst]) + 1e-100) - alpha * np.log(
                        d
                    )
                    contrib = dtype(np.exp(log_contrib))
                    if delta_nfr[dst] < 0:
                        contrib = -contrib
                else:
                    contrib = dtype(delta_nfr[dst] / (d**alpha))
                total += contrib
            exact_subset[src] = float(total)

        # Compute relative mean absolute error (RMAE)
        abs_errors = []
        exact_vals = []
        for n in subset:
            e_val = exact_subset[n]
            a_val = potential[n]
            exact_vals.append(abs(e_val))
            abs_errors.append(abs(e_val - a_val))
        denom = (sum(exact_vals) / len(exact_vals)) if exact_vals else 1.0
        rmae = (sum(abs_errors) / len(abs_errors)) / denom if denom else 0.0
        refinements = 0
        while rmae > error_epsilon and refinements < max_refinements:
            current_ratio = min(current_ratio * 2.0, 0.5)
            potential = compute_with_ratio(current_ratio)
            abs_errors = []
            exact_vals = []
            for n in subset:
                e_val = exact_subset[n]
                a_val = potential[n]
                exact_vals.append(abs(e_val))
                abs_errors.append(abs(e_val - a_val))
            denom = (sum(exact_vals) / len(exact_vals)) if exact_vals else 1.0
            rmae = (sum(abs_errors) / len(abs_errors)) / denom if denom else 0.0
            refinements += 1
        # (Optional) embed metadata for downstream telemetry introspection
        # Embed approximation metadata (prefixed with __)
        potential["__phi_s_landmark_ratio__"] = current_ratio  # type: ignore[index]
        potential["__phi_s_rmae__"] = rmae  # type: ignore[index]

    return potential


def _compute_phi_s_exact(
    G: Any, nodes: list[Any], delta_nfr: dict[Any, float], alpha: float
) -> dict[Any, float]:
    """Exact Φ_s computation using all-pairs shortest paths.

    Precision-aware: uses dtype from get_precision_mode().
    """
    # Use vectorized implementation if available and appropriate
    # Vectorized is faster for N < 500 (approx)
    # For larger N, memory might be an issue if dense matrix is created
    if _VECTORIZATION_AVAILABLE and len(nodes) <= 1000:
        return compute_phi_s_exact_vectorized(
            G, nodes, delta_nfr, alpha, dtype=_get_precision_dtype()
        )

    potential: dict[Any, float] = {}
    dtype = _get_precision_dtype()
    mode = get_precision_mode()

    for src in nodes:
        lengths = (
            nx.single_source_dijkstra_path_length(G, src, weight="weight")
            if G.number_of_edges() > 0
            else {src: 0.0}
        )
        total = dtype(0.0)
        for dst in nodes:
            if dst == src:
                continue
            d = lengths.get(dst, math.inf)
            if not math.isfinite(d) or d <= 0.0:
                continue

            # High/research modes: use more stable exponentiation
            if mode in ("high", "research"):
                # log-space computation for better numerical stability
                log_contrib = np.log(abs(delta_nfr[dst]) + 1e-100) - alpha * np.log(d)
                contrib = dtype(np.exp(log_contrib))
                if delta_nfr[dst] < 0:
                    contrib = -contrib
            else:
                # Standard mode: direct computation
                contrib = dtype(delta_nfr[dst] / (d**alpha))

            total += contrib
        potential[src] = float(total)

    return potential


def _compute_phi_s_optimized(
    G: Any, nodes: list[Any], delta_nfr: dict[Any, float], alpha: float
) -> dict[Any, float]:
    """Optimized Φ_s computation using BFS for unweighted graphs."""
    potential: dict[Any, float] = {}

    # Check if graph is unweighted
    has_weights = any("weight" in G[u][v] for u, v in G.edges())

    if not has_weights:
        # Use BFS for unweighted graphs (more efficient)
        for src in nodes:
            total = 0.0
            visited = {src}
            queue = [(src, 0)]

            while queue:
                node, dist = queue.pop(0)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = dist + 1
                        if new_dist > 0:
                            contrib = delta_nfr[neighbor] / (new_dist**alpha)
                            total += contrib
                        queue.append((neighbor, new_dist))

            potential[src] = total
    else:
        # Fall back to exact method for weighted graphs
        return _compute_phi_s_exact(G, nodes, delta_nfr, alpha)

    return potential


def _compute_phi_s_landmarks(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    alpha: float,
    landmark_ratio: float = 0.1,
) -> dict[Any, float]:
    """Approximate Φ_s computation using landmark sampling."""
    import random

    num_landmarks = max(3, int(len(nodes) * landmark_ratio))

    # Select landmarks: prefer high-degree nodes and nodes with high |ΔNFR|
    node_scores = []
    for node in nodes:
        degree = G.degree(node)
        dnfr_contrib = abs(delta_nfr[node])
        score = degree * (1.0 + dnfr_contrib)
        node_scores.append((score, node))

    # Select top nodes by score, with some randomization
    node_scores.sort(reverse=True)
    top_candidates = [node for _, node in node_scores[: num_landmarks * 2]]
    landmarks = random.sample(top_candidates, min(num_landmarks, len(top_candidates)))

    # Compute exact distances from landmarks
    landmark_distances = {}
    for landmark in landmarks:
        if G.number_of_edges() > 0:
            distances = nx.single_source_dijkstra_path_length(
                G, landmark, weight="weight"
            )
        else:
            distances = {landmark: 0.0}
        landmark_distances[landmark] = distances

    # Approximate potential for each node
    potential: dict[Any, float] = {}

    for src in nodes:
        total = 0.0

        # Exact contribution from landmarks
        for landmark in landmarks:
            if landmark == src:
                continue
            d = landmark_distances[landmark].get(src, math.inf)
            if math.isfinite(d) and d > 0.0:
                contrib = delta_nfr[landmark] / (d**alpha)
                total += contrib

        # Approximate contribution from non-landmarks
        for dst in nodes:
            if dst == src or dst in landmarks:
                continue

            # Find nearest landmark to dst and approximate distance
            min_approx_dist = math.inf
            for landmark in landmarks:
                d_landmark_src = landmark_distances[landmark].get(src, math.inf)
                d_landmark_dst = landmark_distances[landmark].get(dst, math.inf)

                if math.isfinite(d_landmark_src) and math.isfinite(d_landmark_dst):
                    # Triangle approximation
                    approx_dist = abs(d_landmark_src - d_landmark_dst)
                    approx_dist = max(approx_dist, 1.0)  # Avoid zero distance
                    min_approx_dist = min(min_approx_dist, approx_dist)

            if math.isfinite(min_approx_dist) and min_approx_dist > 0.0:
                contrib = delta_nfr[dst] / (min_approx_dist**alpha)
                total += contrib

        potential[src] = total

    return potential


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase"},
)
def compute_phase_gradient(G: Any) -> dict[Any, float]:
    r"""Compute magnitude of discrete phase gradient |∇φ| per locus [CANONICAL].

    |∇φ|(i) = mean_{j∈N(i)} |wrap(φ_j − φ_i)|

    **Dual interpretation** (both consistent):

    1. **As potential energy component** (variational formulation):
       V(i) = ½[Φ_s² + |∇φ|² + K_φ²].  Here |∇φ| is a configuration
       degree of freedom — the system evolves to minimise V, rolling
       downhill toward |∇φ| = 0 (synchronisation).

    2. **As local disorder metric** (telemetry):
       High |∇φ| indicates poor local phase synchronisation and correlates
       with bifurcation risk.  The system naturally minimises |∇φ| through
       coherence (IL) attraction.

    These are not contradictory: the potential well's minimum *is* the
    synchronized state (|∇φ| = 0), and high |∇φ| = high potential energy
    = high stress.

    Safety threshold (telemetry): |∇φ| < γ/π ≈ 0.1837 is the *claimed* Kuramoto
    critical coupling (Universal Tetrahedral Correspondence γ ↔ |∇φ|). NOTE
    (audit 2026-06): a fair test finds |∇φ| at the synchronization onset is
    ≈ 0.29 and σ-dependent, **not** the constant γ/π. The genuine kinematic
    bound is |∇φ| ≤ π (a mean of wrapped angles); γ/π is a *dynamical transition*
    value, not a universal constant. Treat it as an organizing heuristic, not a
    derived threshold.
    """
    grad, _ = _compute_phase_gradient_and_curvature(G)
    return grad


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase"},
)
def compute_phase_curvature(G: Any) -> dict[Any, float]:
    """Compute discrete Laplacian curvature K_φ of the phase field [CANONICAL]."""
    _, curvature = _compute_phase_gradient_and_curvature(G)
    return curvature


def _compute_phase_gradient_and_curvature(
    G: Any,
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Compute |∇φ| and K_φ in a single neighborhood pass.

    Precision-aware: uses dtype from get_precision_mode().
    """
    dtype = _get_precision_dtype()

    nodes = list(G.nodes())
    if not nodes:
        return {}, {}

    # Vectorized path
    if _VECTORIZATION_AVAILABLE:
        try:
            node_to_idx = {node: i for i, node in enumerate(nodes)}

            # Phase array
            phases = np.array([_get_phase(G, node) for node in nodes], dtype=np.float64)

            # Degree array
            degrees = np.array([G.degree[node] for node in nodes], dtype=np.float64)

            # Edge lists
            edge_src_list = []
            edge_dst_list = []

            is_directed = G.is_directed()

            for u, v in G.edges():
                if u not in node_to_idx or v not in node_to_idx:
                    continue

                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]

                # If u is center, v is neighbor: src=v, dst=u
                edge_src_list.append(v_idx)
                edge_dst_list.append(u_idx)

                if not is_directed:
                    # If v is center, u is neighbor: src=u, dst=v
                    edge_src_list.append(u_idx)
                    edge_dst_list.append(v_idx)

            edge_src = np.array(edge_src_list, dtype=np.intp)
            edge_dst = np.array(edge_dst_list, dtype=np.intp)

            grad_arr, curv_arr = compute_phase_gradient_and_curvature_vectorized(
                phases, edge_src, edge_dst, degrees, dtype=dtype
            )

            grad = {node: float(grad_arr[i]) for i, node in enumerate(nodes)}
            curvature = {node: float(curv_arr[i]) for i, node in enumerate(nodes)}
            return grad, curvature

        except Exception:
            # Fallback
            pass

    grad: dict[Any, float] = {}
    curvature: dict[Any, float] = {}

    phases = {node: _get_phase(G, node) for node in nodes}

    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            grad[i] = 0.0
            curvature[i] = 0.0
            continue

        phi_i = dtype(phases[i])
        neigh_phases = np.array([phases[j] for j in neighbors], dtype=dtype)

        if neigh_phases.size == 0:
            grad[i] = 0.0
            curvature[i] = 0.0
            continue

        # Gradient: mean absolute wrapped difference
        diffs = phi_i - neigh_phases
        pi_typed = dtype(np.pi)
        wrapped_diffs = (diffs + pi_typed) % (2 * pi_typed) - pi_typed
        grad[i] = float(np.mean(np.abs(wrapped_diffs)))

        # Curvature: deviation from circular mean of neighbor phases
        cos_vals = np.cos(neigh_phases)
        sin_vals = np.sin(neigh_phases)
        mean_cos = dtype(np.mean(cos_vals))
        mean_sin = dtype(np.mean(sin_vals))

        mean_vec_length = math.hypot(mean_cos, mean_sin)
        if mean_vec_length < 1e-9:
            mean_phase = float(np.mean(neigh_phases))
        else:
            mean_phase = math.atan2(mean_sin, mean_cos)

        curvature[i] = float(_wrap_angle(phi_i - mean_phase))

    return grad, curvature


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr"},
)
def estimate_coherence_length(G: Any) -> float:
    """Estimate coherence length ξ_C from spatial autocorrelation [CANONICAL].

    Precision-aware: uses dtype from get_precision_mode().
    High/research modes use more distance samples for better fit.
    """
    dtype = _get_precision_dtype()
    mode = get_precision_mode()

    # Adjust sampling based on precision mode
    if mode == "research":
        sample_threshold = 100  # More samples for research
        min_samples = 30
    elif mode == "high":
        sample_threshold = 75
        min_samples = 20
    else:  # standard
        sample_threshold = 50
        min_samples = 20

    nodes = list(G.nodes())
    if len(nodes) < 3:
        return float("nan")

    # Vectorized path
    if _VECTORIZATION_AVAILABLE:
        try:
            # Collect ΔNFR map
            dnfr_map = {node: _get_dnfr(G, node) for node in nodes}

            # Use vectorized implementation
            # Note: This uses full distance matrix, so it's O(N^3) or O(N^2) depending on algo.
            # For very large graphs, we might want to stick to the sampling approach below.
            # Let's use a heuristic: if N < 1000, use vectorized.
            if len(nodes) < 1000:
                return compute_coherence_length_vectorized(
                    G, nodes, dnfr_map, dtype=dtype
                )
        except Exception:
            # Fallback to Python implementation
            pass

    # Compute per-node local coherence
    coherences = {}
    for node in nodes:
        dnfr = dtype(abs(_get_dnfr(G, node)))
        coherences[node] = dtype(1.0) / (dtype(1.0) + dnfr)

    # Compute distance matrix (precision-aware sampling)
    if len(nodes) <= sample_threshold:
        distances = dict(nx.all_pairs_shortest_path_length(G))
    else:
        # Sample approach for large graphs
        distances = {}
        num_samples = max(min_samples, len(nodes) // 20)
        sample_nodes = nodes[:: max(1, len(nodes) // num_samples)]
        for node in sample_nodes:
            distances[node] = dict(nx.single_source_shortest_path_length(G, node))

    # Build distance-coherence correlation pairs
    corr_pairs = []
    for src in distances:
        for dst, dist in distances[src].items():
            if src != dst and dist > 0:
                corr = coherences[src] * coherences[dst]
                corr_pairs.append((dist, corr))

    if len(corr_pairs) < 10:
        return float("nan")

    # Group by distance and compute mean correlation
    distance_bins: dict[int, list[float]] = {}
    for dist, corr in corr_pairs:
        if dist not in distance_bins:
            distance_bins[dist] = []
        distance_bins[dist].append(corr)

    dist_corr_pairs = [
        (d, np.mean(corrs)) for d, corrs in distance_bins.items() if len(corrs) >= 2
    ]

    if len(dist_corr_pairs) < 3:
        return float("nan")

    # Fit exponential decay: C(r) ~ exp(-r/ξ_C)
    dist_corr_pairs.sort()
    distances_arr = np.array([d for d, _ in dist_corr_pairs])
    corrs_arr = np.array([c for _, c in dist_corr_pairs])

    # Avoid log of negative/zero values
    positive_corrs = corrs_arr > 1e-9
    if np.sum(positive_corrs) < 3:
        return float("nan")

    distances_fit = distances_arr[positive_corrs]
    log_corrs_fit = np.log(corrs_arr[positive_corrs])

    # Linear fit to log(C) vs r
    try:
        slope, _ = np.polyfit(distances_fit, log_corrs_fit, 1)
        if slope >= 0:  # Should be negative for decay
            return float("nan")
        xi_c = -1.0 / slope
        return float(xi_c) if xi_c > 0 else float("nan")
    except np.linalg.LinAlgError:
        return float("nan")


__all__ = [
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
]
