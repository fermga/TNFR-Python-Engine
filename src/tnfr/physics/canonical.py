"""TNFR structural-field diagnostics - core implementation.

The four canonical public read-outs organize complementary aggregation, phase
and correlation information. They do not reconstruct a complete network state:

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
cached via @cache_tnfr_computation with graph_topology and node_dnfr
dependencies. Precision-aware canonical fields additionally declare
precision_mode, so changing either the source or numerical mode cannot reuse
an incompatible cached field. The cache key embeds a dependency hash of the
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
from dataclasses import dataclass
from typing import Any

from ..mathematics.unified_numerical import np
from ._edge_semantics import (
    has_explicit_edge_lengths,
    has_nonpositive_edge_length,
    structural_path_weight,
)

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
from ._helpers import compensated_sum  # noqa: E402
from ._helpers import get_dnfr as _get_dnfr  # noqa: E402
from ._helpers import get_phase as _get_phase  # noqa: E402
from ._helpers import neighborhood_arrays  # noqa: E402
from ._helpers import wrap_angle as _wrap_angle  # noqa: E402

_PHI_S_DISTANCE_CACHE: dict[tuple, dict[Any, dict[Any, float]]] = {}


def _graph_topology_hash(G: Any) -> int:
    """Hash the labelled, weighted topology used by shortest-path distances.

    Hash changes on structural reorganization affecting distances; phase-only
    changes do not alter shortest-path distances and should keep cache valid.
    """
    from ..utils.cache import _compute_dependency_hash

    return hash(_compute_dependency_hash(G, {"graph_topology"}))


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr", "precision_mode"},
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
    """Compute the structural potential ``sum_j ΔNFR_j / d(i,j)**alpha``.

    The default is exact at every graph size. Distances follow outgoing arcs
    on directed graphs. An explicit ``length`` edge attribute defines the
    metric; an edge without it falls back to ``weight`` for compatibility and
    then to unit length. ``weight`` remains the EPI transport-conductance
    channel, so new weighted graphs should set both attributes whenever those
    quantities differ. Zero-length and unreachable source-target pairs
    contribute zero. Parallel edges use the minimum effective path length.

    Parameters
    ----------
    G : Graph
        TNFR graph with ΔNFR node attributes.
    alpha : float, default 2.0
        Distance exponent. The canonical inverse-square field uses 2.
    landmark_ratio : float | None
        Explicitly opt into a landmark approximation with a ratio clamped to
        [0.001, 0.5]. ``None`` selects exact evaluation. Landmark distances are
        lengths of paths through a landmark, never lower distance bounds.
        With signed pressure, these distances do not bound relative potential
        error. Use the exact default for U6 decisions.
    validate : bool, default False
        For explicit approximations, compare every returned node to the exact
        field. Refine up to ``max_refinements``; return the exact field if the
        requested global RMAE is still unmet. Thus this option also incurs the
        cost of exact evaluation.
    error_epsilon : float, default 0.05
        Nonnegative finite tolerance for ``sum(abs(approx-exact)) /
        sum(abs(exact))`` over all nodes. A zero exact denominator yields zero
        only when the absolute error is zero, otherwise infinity.
    max_refinements : int, default 3
        Maximum number of ratio doublings during validation.
    sample_size : int, default 32
        Retained for call compatibility. Verification covers all nodes;
        sampling cannot certify the global error of a signed field.

    Returns
    -------
    dict[node, float]
        Node potentials. Explicit ``landmark_ratio`` plus ``validate=True``
        retains the legacy diagnostic keys ``__phi_s_landmark_ratio__`` and
        ``__phi_s_rmae__``. ``__phi_s_fallback_exact__`` is 1.0 after an exact
        fallback and 0.0 otherwise; RMAE describes the returned field.

    Notes
    -----
    This read-only computation does not modify the graph or consume random
    state. Exactness means exact graph distances with floating-point sums.
    """
    if nx is None:
        raise RuntimeError("networkx required for structural potential computation")
    nodes = list(G.nodes())
    delta_nfr = {node: _get_dnfr(G, node) for node in nodes}
    if landmark_ratio is None or not nodes:
        return _compute_phi_s_exact(G, nodes, delta_nfr, alpha)

    ratio = float(landmark_ratio)
    if not math.isfinite(ratio):
        raise ValueError("landmark_ratio must be finite")
    ratio = max(0.001, min(0.5, ratio))
    if validate and (not math.isfinite(error_epsilon) or error_epsilon < 0.0):
        raise ValueError("error_epsilon must be finite and nonnegative")
    if validate and max_refinements < 0:
        raise ValueError("max_refinements must be nonnegative")
    if validate and not all(math.isfinite(value) for value in delta_nfr.values()):
        raise ValueError("potential validation requires finite ΔNFR values")

    # The path-bound interpretation requires positive edge lengths. Preserve
    # the exact kernel's historical exclusion of zero-distance pairs.
    has_nonpositive_edge = has_nonpositive_edge_length(G)
    if has_nonpositive_edge:
        potential = _compute_phi_s_exact(G, nodes, delta_nfr, alpha)
        if validate:
            _require_finite_exact_potential(potential)
            potential.update({
                "__phi_s_landmark_ratio__": ratio,
                "__phi_s_rmae__": 0.0,
                "__phi_s_fallback_exact__": 1.0,
            })
        return potential

    potential = _compute_phi_s_landmarks(G, nodes, delta_nfr, alpha, ratio)
    if not validate:
        return potential

    exact = _compute_phi_s_exact(G, nodes, delta_nfr, alpha)
    _require_finite_exact_potential(exact)
    denominator = math.fsum(abs(value) for value in exact.values())

    def relative_error(candidate: dict[Any, float]) -> float:
        if not all(math.isfinite(candidate[node]) for node in nodes):
            return math.inf
        error = math.fsum(abs(candidate[node] - exact[node]) for node in nodes)
        return error / denominator if denominator else (0.0 if error == 0.0 else math.inf)

    rmae = relative_error(potential)
    for _ in range(max_refinements):
        if rmae <= error_epsilon or ratio >= 0.5:
            break
        ratio = min(ratio * 2.0, 0.5)
        potential = _compute_phi_s_landmarks(G, nodes, delta_nfr, alpha, ratio)
        rmae = relative_error(potential)
    fallback_exact = rmae > error_epsilon
    if fallback_exact:
        potential, rmae = exact, 0.0
    potential.update({
        "__phi_s_landmark_ratio__": ratio,
        "__phi_s_rmae__": rmae,
        "__phi_s_fallback_exact__": float(fallback_exact),
    })
    return potential


def _require_finite_exact_potential(potential: dict[Any, float]) -> None:
    """Reject nonfinite reference fields before certifying approximation error."""
    if not all(math.isfinite(value) for value in potential.values()):
        raise ValueError("potential validation requires finite exact potentials")


def _compute_phi_s_exact(
    G: Any, nodes: list[Any], delta_nfr: dict[Any, float], alpha: float
) -> dict[Any, float]:
    """Exact distances with a small dense path or streamed BFS/Dijkstra."""
    if (
        _VECTORIZATION_AVAILABLE
        and len(nodes) <= 50
        and not has_explicit_edge_lengths(G)
    ):
        return compute_phi_s_exact_vectorized(
            G, nodes, delta_nfr, alpha, dtype=_get_precision_dtype()
        )
    return _compute_phi_s_optimized(G, nodes, delta_nfr, alpha)


def _compute_phi_s_optimized(
    G: Any, nodes: list[Any], delta_nfr: dict[Any, float], alpha: float
) -> dict[Any, float]:
    """Exact per-source sums without an O(N²) resident distance matrix.

    Unweighted graphs use BFS, weighted graphs Dijkstra, including parallel
    edge attributes. Compensated float sums reduce cancellation for signed
    pressure; research mode retains the configured extended scalar dtype.
    """
    has_lengths = any(
        "length" in data or "weight" in data
        for _, _, data in G.edges(data=True)
    )
    mode = get_precision_mode()
    dtype = _get_precision_dtype() if mode == "research" else float
    potential: dict[Any, float] = {}
    for source in nodes:
        lengths = (
            nx.single_source_dijkstra_path_length(
                G, source, weight=structural_path_weight(G)
            )
            if has_lengths
            else nx.single_source_shortest_path_length(G, source)
        )
        contributions = (
            dtype(delta_nfr[target]) / dtype(distance) ** alpha
            for target, distance in lengths.items()
            if target != source and math.isfinite(distance) and distance > 0.0
        )
        potential[source] = compensated_sum(contributions, dtype=dtype)
    return potential


def _landmark_distance_maps(
    G: Any, nodes: list[Any], ratio: float
) -> tuple[list[Any], dict[Any, dict[Any, float]], dict[Any, dict[Any, float]]]:
    """Deterministic topology-only landmarks and both directed distance legs."""
    count = min(len(nodes), max(3, int(len(nodes) * ratio)))
    landmarks = sorted(nodes, key=lambda node: (-G.degree(node), repr(node)))[:count]
    topology = _graph_topology_hash(G)
    outward_key = (topology, tuple(landmarks), "outward")
    outward = _PHI_S_DISTANCE_CACHE.get(outward_key)
    if outward is None:
        path_weight = structural_path_weight(G)
        outward = {
            node: nx.single_source_dijkstra_path_length(
                G, node, weight=path_weight
            )
            for node in landmarks
        }
        _PHI_S_DISTANCE_CACHE[outward_key] = outward
    if not G.is_directed():
        return landmarks, outward, outward
    inward_key = (topology, tuple(landmarks), "inward")
    inward = _PHI_S_DISTANCE_CACHE.get(inward_key)
    if inward is None:
        reverse = G.reverse(copy=False)
        reverse_weight = structural_path_weight(reverse)
        inward = {
            node: nx.single_source_dijkstra_path_length(
                reverse, node, weight=reverse_weight
            )
            for node in landmarks
        }
        _PHI_S_DISTANCE_CACHE[inward_key] = inward
    return landmarks, outward, inward


def _compute_phi_s_landmarks(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    alpha: float,
    landmark_ratio: float = 0.1,
) -> dict[Any, float]:
    """Opt-in paths-via-landmarks approximation; no relative-error guarantee."""
    if not nodes:
        return {}
    landmarks, outward, inward = _landmark_distance_maps(G, nodes, landmark_ratio)
    if _VECTORIZATION_AVAILABLE:
        return compute_phi_s_landmarks_vectorized(
            G, nodes, delta_nfr, alpha, landmarks, outward,
            dtype=_get_precision_dtype(), reverse_landmark_distances=inward,
        )
    potential: dict[Any, float] = {}
    for source in nodes:
        contributions = []
        for target in nodes:
            if source == target:
                continue
            distance = min(
                inward[landmark].get(source, math.inf)
                + outward[landmark].get(target, math.inf)
                for landmark in landmarks
            )
            if math.isfinite(distance) and distance > 0.0:
                contributions.append(delta_nfr[target] / distance**alpha)
        potential[source] = math.fsum(contributions)
    return potential


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase", "precision_mode"},
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

    Telemetry uses π/16 ≈ 0.196 as a selected early-warning policy. It is
    neither the Kuramoto critical coupling nor an exact phase-gradient bound.
    The measured synchronization onset is ≈0.29 and σ-dependent. The exact
    kinematic bound is |∇φ| ≤ π because this field averages wrapped angles.
    """
    grad, _ = _compute_phase_gradient_and_curvature(G)
    return grad


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase", "precision_mode"},
)
def compute_phase_curvature(G: Any) -> dict[Any, float]:
    """Compute discrete Laplacian curvature K_φ of the phase field [CANONICAL]."""
    _, curvature = _compute_phase_gradient_and_curvature(G)
    return curvature


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase", "precision_mode"},
)
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
            # Phase array
            phases = np.array([_get_phase(G, node) for node in nodes], dtype=np.float64)
            edge_src, edge_dst, degrees = neighborhood_arrays(G, nodes, dtype=dtype)

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
    dependencies={"graph_topology", "node_dnfr", "precision_mode"},
)
def _estimate_coherence_length_autocorr(G: Any) -> float:
    """Coherence length ξ_C from the spatial-autocorrelation exp-decay fit.

    Precision-aware: uses dtype from get_precision_mode().  Returns ``nan`` when
    the fit degenerates -- a uniform/coherent field (all per-node ``C ≈ 1`` ⇒
    flat correlation ⇒ non-negative slope) or a graph too small for the fit;
    the public :func:`estimate_coherence_length` then falls back to the emergent
    spectral gap.
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

    # Compute the same static local coherence field as the vectorized path.
    from ..metrics.common import structural_coherence

    coherences = {}
    for node in nodes:
        dnfr = dtype(_get_dnfr(G, node))
        coherences[node] = dtype(structural_coherence(dnfr, 0.0))

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


def _spectral_gap_coherence_length(G: Any) -> float:
    """Graph-spectral fallback scale ``1/√λ_gap`` from ``L_rw``.

    This topology-only value is used when the state-dependent autocorrelation
    fit degenerates. On a connected undirected graph the smallest positive
    mode is λ₂. On disconnected or degenerate graphs it does not describe
    correlations across components and may be unavailable.
    """
    from .structural_diffusion import (  # local import: avoid module cycle
        structural_eigenvalues,
    )

    try:
        eigvals = structural_eigenvalues(G)
        nonzero = [float(v) for v in np.asarray(eigvals) if float(v) > 1e-9]
        if not nonzero:
            return float("nan")
        lam2 = min(nonzero)
        return float(1.0 / np.sqrt(lam2)) if lam2 > 0.0 else float("nan")
    except Exception:  # pragma: no cover - degenerate graph guard
        return float("nan")


def estimate_coherence_length(G: Any) -> float:
    """Estimate state-dependent coherence length with a spectral fallback.

    Primary: the exponential-decay fit ``C(r) ~ exp(-r/ξ_C)`` of the coherence
    autocorrelation vs graph distance (:func:`_estimate_coherence_length_autocorr`).
    When that fit degenerates -- a uniformly coherent / near-equilibrium field
    (all per-node ``C ≈ 1`` ⇒ flat correlation ⇒ non-negative slope) or a graph
    too small -- fall back to the topology-only scale ``1/√λ₂`` on a valid
    connected graph. The returned provenance distinguishes the fit from this
    fallback; the two are not asserted to be identical observables.
    """
    return estimate_coherence_length_with_provenance(G).value


def estimate_coherence_length_with_provenance(
    G: Any,
) -> CoherenceLengthEstimate:
    """Return ``ξ_C`` together with the fit/fallback method used."""
    fit = _estimate_coherence_length_autocorr(G)
    if fit == fit and fit > 0.0:
        return CoherenceLengthEstimate(
            float(fit), "autocorrelation_fit", True,
            distance_weighting="graph shortest-path distance",
            sample_selection=_coherence_sample_selection(G),
            fit_quality="negative slope; at least three positive distance bins",
            positive_mode_selection="not applicable",
            graph_regime=_coherence_graph_regime(G),
        )
    spectral = _spectral_gap_coherence_length(G)
    if spectral == spectral and spectral > 0.0:
        return CoherenceLengthEstimate(
            float(spectral), "spectral_gap", False,
            distance_weighting="not applicable",
            sample_selection="not applicable",
            fit_quality="autocorrelation fit unavailable",
            positive_mode_selection="smallest eigenvalue above 1e-9",
            graph_regime=_coherence_graph_regime(G),
        )
    return CoherenceLengthEstimate(
        float("nan"), "unavailable", False,
        distance_weighting="graph shortest-path distance",
        sample_selection=_coherence_sample_selection(G),
        fit_quality="no admissible decay fit or positive symmetric mode",
        positive_mode_selection="smallest eigenvalue above 1e-9",
        graph_regime=_coherence_graph_regime(G),
    )


def _coherence_sample_selection(G: Any) -> str:
    """Describe the active autocorrelation sampling policy."""
    size = len(G)
    if size < 1000 and _VECTORIZATION_AVAILABLE:
        return "all unordered node pairs"
    mode = get_precision_mode()
    threshold = 100 if mode == "research" else 75 if mode == "high" else 50
    return (
        "all source nodes" if size <= threshold
        else "deterministic evenly-spaced source-node sample"
    )


def _coherence_graph_regime(G: Any) -> str:
    """Describe graph assumptions visible at the estimator boundary."""
    directed = bool(G.is_directed())
    connected = False
    try:
        connected = bool(
            nx.is_weakly_connected(G) if directed else nx.is_connected(G)
        )
    except nx.NetworkXPointlessConcept:
        pass
    return (
        f"{'directed' if directed else 'undirected'}; "
        f"{'connected' if connected else 'empty_or_disconnected'}"
    )


@dataclass(frozen=True)
class CoherenceLengthEstimate:
    """Coherence-length value with estimator provenance."""

    value: float
    method: str
    fit_available: bool
    distance_weighting: str = "unspecified"
    sample_selection: str = "unspecified"
    fit_quality: str = "unspecified"
    positive_mode_selection: str = "unspecified"
    graph_regime: str = "unspecified"


__all__ = [
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
    "estimate_coherence_length_with_provenance",
    "CoherenceLengthEstimate",
]
