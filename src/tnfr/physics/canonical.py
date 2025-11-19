"""TNFR Canonical Structural Fields - Core Implementation

The four CANONICAL structural fields that provide complete multi-scale
characterization of TNFR network state:

- Φ_s: Global structural potential (field theory dimension)
- |∇φ|: Local phase desynchronization (gradient dimension)
- K_φ: Phase curvature / geometric confinement (curvature dimension)
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

CRITICAL TECHNICAL NOTE (Cache Invalidation Issue - Nov 2025):
--------------------------------------------------------------
compute_structural_potential is cached via @cache_tnfr_computation.
Cache key depends on: graph_topology + node_dnfr distribution.

**Issue**: Uniform ΔNFR scaling (e.g., all nodes 0.5→3.0) preserves Φ_s
ratios because Φ_s is linear in ΔNFR. This produces zero drift detection:
  Φ_s(ΔNFR) = Σ_j ΔNFR_j/d_ij^α
  Φ_s(k·ΔNFR) = k·Φ_s(ΔNFR)
  → Drift = Φ_s_after - Φ_s_before = k·Φ - Φ = (k-1)·Φ scales uniformly
  → No spatial gradient created, U6 validation fails

**Solution**: Use non-uniform ΔNFR patterns (e.g., alternating 5.0/0.1)
to create spatial gradients. For cache workarounds in tests, vary alpha
(e.g., 2.0 → 2.001) to force cache miss via different function arguments.

**Physics**: U6 structural potential confinement (Δ Φ_s < 2.0 threshold)
requires spatial ΔNFR gradients to produce measurable drift for passive
equilibrium validation. Uniform scaling defeats gradient detection.

See: tests/unit/operators/test_unified_grammar.py TestU6 for examples
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import precision mode configuration
from ..config import get_precision_mode

# Import TNFR cache system
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

    def cache_tnfr_computation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    class CacheLevel:
        DERIVED_METRICS = None

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_THETA, ALIAS_DNFR
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]


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


def _wrap_angle(angle: float) -> float:
    """Map angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for a node (radians in [0, 2π))."""
    node_data = G.nodes[node]
    for alias in ALIAS_THETA:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def _get_dnfr(G: Any, node: Any) -> float:
    """Retrieve ΔNFR value for a node."""
    node_data = G.nodes[node]
    for alias in ALIAS_DNFR:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


_PHI_S_DISTANCE_CACHE: Dict[tuple, Dict[Any, Dict[Any, float]]] = {}


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
    dependencies={'graph_topology', 'node_dnfr'},
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
) -> Dict[Any, float]:
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
    Dict[node, float]
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
        raise RuntimeError(
            "networkx required for structural potential computation"
        )

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

    def compute_with_ratio(ratio: float) -> Dict[Any, float]:
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
            landmark_distances: Dict[Any, Dict[Any, float]] = {}
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

        # Approximate potentials
        potential: Dict[Any, float] = {}
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
                    d_land_src = landmark_distances[landmark].get(
                        src, math.inf
                    )
                    d_land_dst = landmark_distances[landmark].get(
                        dst, math.inf
                    )
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
        subset = (
            nodes
            if len(nodes) <= sample_size
            else _r.sample(nodes, sample_size)
        )
        exact_subset: Dict[Any, float] = {}
        dtype = _get_precision_dtype()
        mode = get_precision_mode()
        for src in subset:
            if G.number_of_edges() > 0:
                lengths = nx.single_source_dijkstra_path_length(
                    G, src, weight="weight"
                )
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
                    log_contrib = (
                        np.log(abs(delta_nfr[dst]) + 1e-100)
                        - alpha * np.log(d)
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
            rmae = (
                (sum(abs_errors) / len(abs_errors)) / denom if denom else 0.0
            )
            refinements += 1
        # (Optional) embed metadata for downstream telemetry introspection
        # Embed approximation metadata (prefixed with __)
        potential['__phi_s_landmark_ratio__'] = current_ratio  # type: ignore[index]
        potential['__phi_s_rmae__'] = rmae  # type: ignore[index]

    return potential


def _compute_phi_s_exact(
    G: Any,
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    alpha: float
) -> Dict[Any, float]:
    """Exact Φ_s computation using all-pairs shortest paths.
    
    Precision-aware: uses dtype from get_precision_mode().
    """
    potential: Dict[Any, float] = {}
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
                log_contrib = (
                    np.log(abs(delta_nfr[dst]) + 1e-100)
                    - alpha * np.log(d)
                )
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
    G: Any,
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    alpha: float
) -> Dict[Any, float]:
    """Optimized Φ_s computation using BFS for unweighted graphs."""
    potential: Dict[Any, float] = {}

    # Check if graph is unweighted
    has_weights = any('weight' in G[u][v] for u, v in G.edges())

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
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    alpha: float,
    landmark_ratio: float = 0.1
) -> Dict[Any, float]:
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
    top_candidates = [node for _, node in node_scores[:num_landmarks * 2]]
    landmarks = random.sample(
        top_candidates, min(num_landmarks, len(top_candidates))
    )

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
    potential: Dict[Any, float] = {}

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
                d_landmark_src = landmark_distances[landmark].get(
                    src, math.inf
                )
                d_landmark_dst = landmark_distances[landmark].get(
                    dst, math.inf
                )

                if (
                    math.isfinite(d_landmark_src)
                    and math.isfinite(d_landmark_dst)
                ):
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
def compute_phase_gradient(G: Any) -> Dict[Any, float]:
    """Compute magnitude of discrete phase gradient |∇φ| per locus [CANONICAL]."""
    grad, _ = _compute_phase_gradient_and_curvature(G)
    return grad


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase"},
)
def compute_phase_curvature(G: Any) -> Dict[Any, float]:
    """Compute discrete Laplacian curvature K_φ of the phase field [CANONICAL]."""
    _, curvature = _compute_phase_gradient_and_curvature(G)
    return curvature


def _compute_phase_gradient_and_curvature(
    G: Any,
) -> tuple[Dict[Any, float], Dict[Any, float]]:
    """Compute |∇φ| and K_φ in a single neighborhood pass.
    
    Precision-aware: uses dtype from get_precision_mode().
    """
    dtype = _get_precision_dtype()

    grad: Dict[Any, float] = {}
    curvature: Dict[Any, float] = {}

    nodes = list(G.nodes())
    phases = {node: _get_phase(G, node) for node in nodes}

    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            grad[i] = 0.0
            curvature[i] = 0.0
            continue

        phi_i = dtype(phases[i])
        neigh_phases = np.array(
            [phases[j] for j in neighbors], dtype=dtype
        )

        if neigh_phases.size == 0:
            grad[i] = 0.0
            curvature[i] = 0.0
            continue

        # Gradient: mean absolute wrapped difference
        diffs = phi_i - neigh_phases
        pi_typed = dtype(np.pi)
        wrapped_diffs = (
            (diffs + pi_typed) % (2 * pi_typed) - pi_typed
        )
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
    dependencies={'graph_topology', 'node_dnfr'},
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
        return float('nan')

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
        sample_nodes = nodes[::max(1, len(nodes) // num_samples)]
        for node in sample_nodes:
            distances[node] = dict(
                nx.single_source_shortest_path_length(G, node)
            )

    # Build distance-coherence correlation pairs
    corr_pairs = []
    for src in distances:
        for dst, dist in distances[src].items():
            if src != dst and dist > 0:
                corr = coherences[src] * coherences[dst]
                corr_pairs.append((dist, corr))

    if len(corr_pairs) < 10:
        return float('nan')

    # Group by distance and compute mean correlation
    distance_bins: Dict[int, List[float]] = {}
    for dist, corr in corr_pairs:
        if dist not in distance_bins:
            distance_bins[dist] = []
        distance_bins[dist].append(corr)

    dist_corr_pairs = [
        (d, np.mean(corrs)) for d, corrs in distance_bins.items()
        if len(corrs) >= 2
    ]

    if len(dist_corr_pairs) < 3:
        return float('nan')

    # Fit exponential decay: C(r) ~ exp(-r/ξ_C)
    dist_corr_pairs.sort()
    distances_arr = np.array([d for d, _ in dist_corr_pairs])
    corrs_arr = np.array([c for _, c in dist_corr_pairs])

    # Avoid log of negative/zero values
    positive_corrs = corrs_arr > 1e-9
    if np.sum(positive_corrs) < 3:
        return float('nan')

    distances_fit = distances_arr[positive_corrs]
    log_corrs_fit = np.log(corrs_arr[positive_corrs])

    # Linear fit to log(C) vs r
    try:
        slope, _ = np.polyfit(distances_fit, log_corrs_fit, 1)
        if slope >= 0:  # Should be negative for decay
            return float('nan')
        xi_c = -1.0 / slope
        return float(xi_c) if xi_c > 0 else float('nan')
    except np.linalg.LinAlgError:
        return float('nan')


__all__ = [
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length"
]
