"""Vectorized operations for TNFR physics.

This module provides optimized NumPy implementations of structural field computations
to replace slow Python loops in canonical.py.
"""

from typing import Any

import networkx as nx

from ..mathematics._neighbor_differences import edge_mean_differences
from ..mathematics.unified_numerical import compute_phase_difference, np
from ._edge_semantics import structural_distance_rows, validate_structural_graph
from ._helpers import finite_real_scalar
from ._potential_kernel import potential_row_sum


def compute_phi_s_exact_vectorized(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    alpha: float,
    dtype: type = np.float64,
    distance_matrix: np.ndarray | None = None,
) -> dict[Any, float]:
    """Exact-distance Φ_s, with compensated rows for signed pressure."""
    from ._coherence_fit import _distance_array

    validate_structural_graph(G, nodes)
    alpha = finite_real_scalar(alpha, "potential alpha")
    for node in nodes:
        finite_real_scalar(delta_nfr[node], "potential pressure")
    if distance_matrix is not None:
        D = _distance_array(distance_matrix, len(nodes), G.is_directed()).copy()
    elif any("length" in data or "weight" in data for _, _, data in G.edges(data=True)):
        # The shared path owner validates metric and reachable path overflow.
        return _compute_phi_s_exact_python_fallback(G, nodes, delta_nfr, alpha, dtype)
    else:
        try:
            D = nx.floyd_warshall_numpy(G, nodelist=nodes)
        except Exception:
            return _compute_phi_s_exact_python_fallback(
                G, nodes, delta_nfr, alpha, dtype
            )
    D = np.asarray(D, dtype=dtype)
    np.fill_diagonal(D, np.inf)
    valid_distances = np.isfinite(D) & (D > 0.0)
    return {
        node: potential_row_sum(
            (
                (delta_nfr[target], D[i, j])
                for j, target in enumerate(nodes)
                if valid_distances[i, j]
            ),
            alpha,
            dtype=dtype,
        )
        for i, node in enumerate(nodes)
    }


def _compute_phi_s_exact_python_fallback(G, nodes, delta_nfr, alpha, dtype):
    """Use the same distance and reduction owners when dense paths are unsuitable."""
    potential = {}
    for source, lengths in structural_distance_rows(G, nodes):
        potential[source] = potential_row_sum(
            (
                (delta_nfr[target], distance)
                for target, distance in lengths.items()
                if target != source and distance > 0.0
            ),
            alpha,
            dtype=dtype,
        )
    return potential


def compute_phi_s_landmarks_vectorized(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    alpha: float,
    landmarks: list[Any],
    landmark_distances: dict[Any, dict[Any, float]],
    dtype: type = np.float64,
    *,
    reverse_landmark_distances: dict[Any, dict[Any, float]] | None = None,
) -> dict[Any, float]:
    """Approximate Φ_s with paths ``d(i, landmark) + d(landmark, j)``.

    For positive edge lengths these are upper bounds on shortest-path
    distances. They do not certify relative potential error for signed ΔNFR.
    A pair without a path through any selected landmark contributes zero;
    disconnected components never exchange an artificial source.

    ``landmark_distances`` contains outgoing distances from each landmark.
    On directed graphs the reverse maps supply distances to each landmark;
    they are computed here when the optional maps are omitted. Scalar and
    vectorized callers therefore use the same outgoing-path convention.
    """
    validate_structural_graph(G, nodes)
    alpha = finite_real_scalar(alpha, "potential alpha")
    for node in nodes:
        finite_real_scalar(delta_nfr[node], "potential pressure")
    num_nodes = len(nodes)
    if not nodes:
        return {}
    if not landmarks:
        return {node: 0.0 for node in nodes}
    if reverse_landmark_distances is None:
        if G.is_directed():
            reverse = G.reverse(copy=False)
            reverse_landmark_distances = dict(
                structural_distance_rows(reverse, landmarks)
            )
        else:
            reverse_landmark_distances = landmark_distances

    outward = np.asarray(
        [
            [landmark_distances[landmark].get(node, np.inf) for node in nodes]
            for landmark in landmarks
        ],
        dtype=dtype,
    )
    inward = np.asarray(
        [
            [reverse_landmark_distances[landmark].get(node, np.inf) for node in nodes]
            for landmark in landmarks
        ],
        dtype=dtype,
    )
    pressure = np.asarray([delta_nfr[node] for node in nodes], dtype=dtype)
    potential = np.zeros(num_nodes, dtype=dtype)

    # A two-dimensional working block avoids the old L x batch x N tensor.
    # Infinity remains infinity throughout, including for unreachable pairs.
    batch_size = 200
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        distances = np.full((end - start, num_nodes), np.inf, dtype=dtype)
        reachable = np.zeros(distances.shape, dtype=bool)
        for index in range(len(landmarks)):
            first = inward[index, start:end, None]
            second = outward[index, None, :]
            reachable |= np.isfinite(first) & np.isfinite(second)
            with np.errstate(over="ignore"):
                through_landmark = first + second
            np.minimum(distances, through_landmark, out=distances)
        if np.any(reachable & ~np.isfinite(distances)):
            raise ValueError(
                "reachable landmark path distance exceeds the finite represented range"
            )
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        valid = np.isfinite(distances) & (distances > 0.0)
        for i in range(end - start):
            potential[start + i] = potential_row_sum(
                zip(pressure[valid[i]], distances[i, valid[i]]), alpha, dtype=dtype
            )
    return {node: float(potential[index]) for index, node in enumerate(nodes)}


def compute_vf_variance_vectorized(
    G: Any, vf_attr: str = "νf", radius: int = 1
) -> dict[Any, float]:
    """Vectorized computation of νf variance."""
    if radius != 1:
        # Fallback for radius > 1 (complex neighborhood)
        return None

    nodes = list(G.nodes())
    n = len(nodes)

    # Get Adjacency Matrix A
    try:
        A = nx.to_numpy_array(G, nodelist=nodes)
    except Exception:
        return None

    # Add self-loops (neighborhood includes self)
    np.fill_diagonal(A, 1.0)

    # Get vf vector
    vf_values = np.array([G.nodes[node].get(vf_attr, 0.0) for node in nodes])

    # Sum of values in neighborhood: S1 = A @ V
    S1 = A @ vf_values

    # Sum of squared values: S2 = A @ V^2
    S2 = A @ (vf_values**2)

    # Count of neighbors: N = A @ 1
    N_counts = np.sum(A, axis=1)

    # Avoid division by zero (should not happen with self-loops)
    N_counts[N_counts == 0] = 1.0

    # Mean: mu = S1 / N
    mu = S1 / N_counts

    # Population Variance: sigma^2 = (S2 / N) - mu^2
    var_pop = (S2 / N_counts) - (mu**2)

    # Sample Variance: var_sample = var_pop * (N / (N - 1))
    # If N=1, var=0
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = N_counts / (N_counts - 1.0)
        var_sample = var_pop * correction

    # Fix N=1 case (correction is inf/nan)
    var_sample[N_counts <= 1] = 0.0

    # Ensure non-negative (numerical noise)
    var_sample = np.maximum(var_sample, 0.0)

    return {node: float(var_sample[i]) for i, node in enumerate(nodes)}


def compute_spectral_kurtosis_vectorized(G: Any, normalized: bool = True) -> float:
    """Vectorized Spectral Kurtosis using Trace(A^4)."""
    try:
        A = nx.to_numpy_array(G)
    except Exception:
        return 0.0

    n = A.shape[0]
    if n == 0:
        return 0.0

    # Compute A^2
    A2 = A @ A

    # Trace(A^4) = ||A^2||_F^2 (sum of squared elements of A^2)
    # This avoids full eigendecomposition
    mu_4 = np.sum(A2**2) / n

    if normalized:
        return mu_4 / (n**2)
    return mu_4


def compute_phase_current_vectorized(
    theta_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64,
) -> np.ndarray:
    """Vectorized computation of Phase Current J_φ.

    J_φ(i) = mean(sin(θ_j - θ_i)) for j in neighbors(i)

    Parameters
    ----------
    theta_arr : np.ndarray
        Array of phase values for all nodes.
    edge_src : np.ndarray
        Indices of neighbor nodes (j).
    edge_dst : np.ndarray
        Indices of center nodes (i).
        Must include both (u,v) and (v,u) for undirected graphs to cover all neighbors.
    degrees : np.ndarray
        Degree of each node (number of neighbors).

    Returns
    -------
    np.ndarray
        Phase current for each node.
    """
    # Retain the signed branch and tiny displacements; adding pi first
    # erases sub-ULP gaps and flips the represented positive half-turn.
    wrapped_diffs = compute_phase_difference(theta_arr[edge_src], theta_arr[edge_dst])

    # sin(Δθ)
    sines = np.sin(wrapped_diffs)

    # Sum over neighbors
    # We use a larger type for accumulation to avoid overflow/precision issues
    sums = np.zeros(len(theta_arr), dtype=dtype)
    np.add.at(sums, edge_dst, sines)

    # Divide by degree to get mean
    # Handle division by zero for isolated nodes
    with np.errstate(divide="ignore", invalid="ignore"):
        result = sums / degrees

    # Fix isolated nodes (degree 0 -> result NaN/Inf -> 0)
    result[degrees == 0] = 0.0

    return result


def compute_dnfr_flux_vectorized(
    dnfr_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64,
) -> np.ndarray:
    """Read binary64 mean-neighbor pressure contrast through the shared reducer.

    J_ΔNFR(i) = mean(ΔNFR_j - ΔNFR_i) for j in neighbors(i)
              = mean(ΔNFR_j) - ΔNFR_i

    Parameters
    ----------
    dnfr_arr : np.ndarray
        Array of ΔNFR values for all nodes.
    edge_src : np.ndarray
        Indices of neighbor nodes (j).
    edge_dst : np.ndarray
        Indices of center nodes (i).
    degrees : np.ndarray
        Unique outgoing-neighbor counts, including a self-loop once. Must
        agree with the supplied incidences; conductance is not a weight here.

    Returns
    -------
    np.ndarray
        ΔNFR flux for each node, stored in ``dtype``. The shared binary64
        reduction retains finite differences and signed cancellation; a
        mathematically unrepresentable final flux is rejected.
    """
    counts = np.bincount(edge_dst, minlength=len(dnfr_arr))
    if not np.array_equal(counts, degrees):
        raise ValueError("neighbor counts must match pressure-flux incidence")
    # This API names neighbors 'source' and centers 'destination'; the shared
    # transport reducer uses the opposite names for outgoing incidence rows.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(
            edge_mean_differences(dnfr_arr, edge_dst, edge_src), dtype=dtype
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("pressure flux exceeds the selected output dtype range")
    return result


def compute_phase_gradient_and_curvature_vectorized(
    theta_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient/curvature arrays using the shared represented read-out.

    Sources are unique neighbor indices and destinations are center indices;
    counts must match those incidences (loops once, directed successors).
    NumPy binary64 trigonometric components are summed exactly regardless of
    accumulator ``dtype``. The angle is approximate. An exact represented
    joint-zero nonempty neighborhood raises ``UndefinedPhaseCurvatureError``.
    """
    from ..config import get_precision_mode
    from .phase_curvature import _observe_phase_arrays, _require_defined_curvature

    observation = _observe_phase_arrays(
        theta_arr,
        edge_src,
        edge_dst,
        degrees,
        dtype=dtype,
        precision_mode=get_precision_mode(),
    )
    _require_defined_curvature(observation)
    return (
        np.asarray([row.gradient for row in observation.rows], dtype=dtype),
        np.asarray([row.curvature for row in observation.rows], dtype=dtype),
    )


def compute_coherence_length_vectorized(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    dtype: type = np.float64,
    distance_matrix: np.ndarray | None = None,
) -> float:
    """Fit uncentered static products using the shared structural distances.

    Uses per-edge length, else weight, else unit length, with parallel minimum
    and outgoing distances on directed graphs. This shares pair selection and
    fitting with the streamed canonical path. The result has the chosen path
    length units; it is not a connected covariance or spectral-gap identity.

    A supplied matrix must have the declared node order, zero diagonal,
    nonnegative entries (positive infinity for missing pairs), and symmetry
    for an undirected graph. Invalid matrices raise; they are caller-declared
    distances, not a certificate that shortest paths were computed correctly.
    """
    from ..config import get_precision_mode
    from ._coherence_fit import coherence_sources, fit_coherence_length

    return fit_coherence_length(
        G,
        nodes,
        delta_nfr,
        sources=coherence_sources(nodes, get_precision_mode()),
        dtype=dtype,
        materialize=len(nodes) < 1000,
        distance_matrix=distance_matrix,
    )
