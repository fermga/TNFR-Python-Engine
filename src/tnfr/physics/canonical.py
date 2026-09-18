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
Fields record global precision_mode from tnfr.config. Some accumulators use
longdouble in research mode where available, but circular-curvature components
always use explicitly materialized NumPy binary64 trigonometry. Exactness there
concerns the component sums only; the angle and wrapped curvature remain
approximations. Changing precision does not change a policy definition, but a
finite decision near a threshold can change with numerical error.

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
from .phase_curvature import (
    PhaseCurvatureNodeObservation,
    PhaseCurvatureObservation,
    UndefinedPhaseCurvatureError,
    _materialize_phases,
    _observe_neighborhoods,
    _require_defined_curvature,
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
from ._helpers import get_phase as _get_phase  # noqa: E402,F401 - compatibility import
from ._helpers import (  # noqa: E402,F401 - compatibility import
    wrap_angle as _wrap_angle,
)

_PHI_S_DISTANCE_CACHE: dict[tuple, dict[Any, dict[Any, float]]] = {}


def _graph_topology_hash(G: Any) -> int:
    """Hash the labelled, weighted topology used by shortest-path distances.

    Hash changes on structural reorganization affecting distances; phase-only
    changes do not alter shortest-path distances and should keep cache valid.
    """
    from ..utils.cache import _compute_dependency_hash

    return hash(_compute_dependency_hash(G, {"graph_topology"}))


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
    Authoritative pressure aliases must be finite represented real scalars;
    validation precedes every cache lookup. Returned maps are detached from
    the cache and may be modified by the caller.
    """
    if nx is None:
        raise RuntimeError("networkx required for structural potential computation")
    nodes = tuple(G.nodes())
    pressure = tuple(_get_dnfr(G, node) for node in nodes)
    return dict(
        _structural_potential_cached(
            G,
            nodes,
            pressure,
            bool(_VECTORIZATION_AVAILABLE),
            alpha,
            landmark_ratio=landmark_ratio,
            validate=validate,
            error_epsilon=error_epsilon,
            max_refinements=max_refinements,
            sample_size=sample_size,
        )
    )


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr", "precision_mode"},
)
def _structural_potential_cached(
    G,
    node_order,
    pressure_values,
    vectorized,
    alpha,
    *,
    landmark_ratio,
    validate,
    error_epsilon,
    max_refinements,
    sample_size,
):
    """Cache the potential from already admitted ordered pressure values."""
    nodes = list(node_order)
    delta_nfr = dict(zip(nodes, pressure_values))
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
            potential.update(
                {
                    "__phi_s_landmark_ratio__": ratio,
                    "__phi_s_rmae__": 0.0,
                    "__phi_s_fallback_exact__": 1.0,
                }
            )
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
        return (
            error / denominator if denominator else (0.0 if error == 0.0 else math.inf)
        )

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
    potential.update(
        {
            "__phi_s_landmark_ratio__": ratio,
            "__phi_s_rmae__": rmae,
            "__phi_s_fallback_exact__": float(fallback_exact),
        }
    )
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
        "length" in data or "weight" in data for _, _, data in G.edges(data=True)
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
            node: nx.single_source_dijkstra_path_length(G, node, weight=path_weight)
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
            G,
            nodes,
            delta_nfr,
            alpha,
            landmarks,
            outward,
            dtype=_get_precision_dtype(),
            reverse_landmark_distances=inward,
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


def observe_phase_curvature(G: Any) -> PhaseCurvatureObservation:
    """Observe circular curvature without inventing a direction at cancellation.

    The immutable evidence retains ordered phases, neighbors, materialized
    binary64 components and their exact rational sums. A nonempty represented
    joint-zero resultant has ``curvature=None``; isolates use a separate zero
    convention. Neither small nonzero sums nor represented cancellation prove
    a corresponding exact-real trigonometric statement. See the result's scope.
    """
    return _phase_readout_bundle(G)[0]


def _phase_readout_bundle(G):
    # Validate before entering the cache: raw invalid aliases must not be
    # hidden by a dependency hash or by a previously computed valid result.
    nodes = tuple(G.nodes())
    neighbors = tuple(tuple(G.neighbors(node)) for node in nodes)
    phases = _materialize_phases(
        next(
            (G.nodes[node][alias] for alias in ALIAS_THETA if alias in G.nodes[node]),
            0.0,
        )
        for node in nodes
    )
    observation = _phase_readout_cached(
        G, nodes, neighbors, phases, get_precision_mode()
    )
    # Numeric dictionaries are detached projections, never cached mutable
    # evidence. A caller may edit its result without poisoning later readers.
    return (
        observation,
        {row.node: row.gradient for row in observation.rows},
        {row.node: row.curvature for row in observation.rows},
    )


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase", "precision_mode"},
)
def _phase_readout_cached(G, nodes, neighbors, phases, precision_mode):
    indices = {node: i for i, node in enumerate(nodes)}
    return _observe_neighborhoods(
        nodes,
        tuple(tuple(indices[node] for node in row) for row in neighbors),
        phases,
        dtype=_get_precision_dtype(),
        precision_mode=precision_mode,
    )


def compute_phase_gradient(G: Any) -> dict[Any, float]:
    r"""Compute ``mean_j |wrap(phi_i - phi_j)|`` over unique neighbors.

    The wrapped-angle bound is pi; a warning threshold is a separate policy.
    This read-out remains available when a neighbor circular mean is undefined.
    No monotonicity or evolution law follows from this diagnostic definition.
    """
    return _phase_readout_bundle(G)[1]


def compute_phase_curvature(G: Any) -> dict[Any, float]:
    """Return wrapped deviation from the represented neighbor-phasor direction.

    Raises ``UndefinedPhaseCurvatureError`` for an exact represented joint-zero
    nonempty neighborhood. Use ``observe_phase_curvature`` to retain that
    unavailable value with its evidence; it is not replaced by an angle mean.
    Defined results retain the public node-to-float schema and pi bound.
    """
    observation, _, curvature = _phase_readout_bundle(G)
    _require_defined_curvature(observation)
    return curvature


def _compute_phase_gradient_and_curvature(G):
    """Strict numeric adapter; the independent gradient does not call this."""
    observation, gradient, curvature = _phase_readout_bundle(G)
    _require_defined_curvature(observation)
    return gradient, curvature


def _estimate_coherence_length_autocorr(G: Any) -> float:
    """Fit static uncentered coherence products using the shared distance contract."""
    from ._coherence_fit import coherence_sources

    nodes = tuple(G.nodes())
    sources = coherence_sources(nodes, get_precision_mode())
    pressure = tuple(_get_dnfr(G, node) for node in nodes)
    # The topology hash is order-independent, but a large-graph source sample
    # is not. Bind both orders and the selected numerical path in this cache.
    return _coherence_fit_cached(
        G, nodes, sources, pressure, bool(_VECTORIZATION_AVAILABLE)
    )


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr", "precision_mode"},
)
def _coherence_fit_cached(G, nodes, sources, pressure_values, vectorized):
    from ._coherence_fit import fit_coherence_length

    pressure = dict(zip(nodes, pressure_values))
    if vectorized:
        return compute_coherence_length_vectorized(
            G,
            list(nodes),
            pressure,
            dtype=_get_precision_dtype(),
        )
    return fit_coherence_length(
        G,
        nodes,
        pressure,
        sources=sources,
        dtype=_get_precision_dtype(),
    )


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

    Primary: the exponential-decay fit ``q(r) ~ a*exp(-r/ξ_C)``, where
    ``q(r)=mean(C_i*C_j | d(i,j)=r)`` and ``C_i=1/(1+|ΔNFR_i|)``. This is an
    uncentered static product with dEPI set to zero, not connected covariance.
    Distances use explicit length, else weight, else unit length; parallel
    edges use their minimum and directed graphs use outgoing paths.
    When that fit degenerates -- a uniformly coherent / near-equilibrium field
    (all per-node ``C ≈ 1`` ⇒ flat correlation ⇒ non-negative slope) or a graph
    too small -- fall back to the topology-only scale ``1/√λ₂`` on a valid
    connected graph. The returned provenance distinguishes the fit from this
    fallback; the fitted value has path-length units, while the fallback is a
    dimensionless normalized-generator mode scale. They are not interchangeable
    observables. Invalid edge lengths raise rather than selecting a fallback.
    """
    return estimate_coherence_length_with_provenance(G).value


def estimate_coherence_length_with_provenance(
    G: Any,
) -> CoherenceLengthEstimate:
    """Return ``ξ_C`` together with the fit/fallback method used."""
    from ._coherence_fit import DISTANCE_DESCRIPTION, FIT_DESCRIPTION

    fit = _estimate_coherence_length_autocorr(G)
    if fit == fit and fit > 0.0:
        return CoherenceLengthEstimate(
            float(fit),
            "autocorrelation_fit",
            True,
            distance_weighting=DISTANCE_DESCRIPTION
            + "; fitted value in path-length units",
            sample_selection=_coherence_sample_selection(G),
            fit_quality=FIT_DESCRIPTION,
            positive_mode_selection="not applicable",
            graph_regime=_coherence_graph_regime(G),
        )
    spectral = _spectral_gap_coherence_length(G)
    if spectral == spectral and spectral > 0.0:
        return CoherenceLengthEstimate(
            float(spectral),
            "spectral_gap",
            False,
            distance_weighting="not applicable; dimensionless normalized-generator mode scale",
            sample_selection="not applicable",
            fit_quality="autocorrelation fit unavailable",
            positive_mode_selection="smallest eigenvalue above 1e-9",
            graph_regime=_coherence_graph_regime(G),
        )
    return CoherenceLengthEstimate(
        float("nan"),
        "unavailable",
        False,
        distance_weighting=DISTANCE_DESCRIPTION,
        sample_selection=_coherence_sample_selection(G),
        fit_quality="no admissible decay fit or positive symmetric mode",
        positive_mode_selection="smallest eigenvalue above 1e-9",
        graph_regime=_coherence_graph_regime(G),
    )


def _coherence_sample_selection(G: Any) -> str:
    """Describe the active autocorrelation sampling policy."""
    from ._coherence_fit import coherence_sample_description, coherence_sources

    nodes = tuple(G)
    return coherence_sample_description(
        G, nodes, coherence_sources(nodes, get_precision_mode())
    )


def _coherence_graph_regime(G: Any) -> str:
    """Describe graph assumptions visible at the estimator boundary."""
    directed = bool(G.is_directed())
    connected = False
    try:
        connected = bool(nx.is_weakly_connected(G) if directed else nx.is_connected(G))
    except nx.NetworkXPointlessConcept:
        pass
    return (
        f"{'directed' if directed else 'undirected'}; "
        f"{'connected' if connected else 'empty_or_disconnected'}"
    )


@dataclass(frozen=True)
class CoherenceLengthEstimate:
    """Fit length or distinct spectral scale, identified by method and units."""

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
    "observe_phase_curvature",
    "PhaseCurvatureObservation",
    "PhaseCurvatureNodeObservation",
    "UndefinedPhaseCurvatureError",
    "estimate_coherence_length",
    "estimate_coherence_length_with_provenance",
    "CoherenceLengthEstimate",
]
