"""Vectorized operations for TNFR physics.

This module provides optimized NumPy implementations of structural field computations
to replace slow Python loops in canonical.py.
"""

import math
from typing import Any

from ..mathematics.unified_numerical import np
from ._edge_semantics import has_explicit_edge_lengths, structural_path_weight
from ._helpers import compensated_sum

try:
    import networkx as nx
except ImportError:
    nx = None


def compute_phi_s_exact_vectorized(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    alpha: float,
    dtype: type = np.float64,
    distance_matrix: np.ndarray | None = None,
) -> dict[Any, float]:
    """Exact-distance Φ_s, with compensated rows for signed pressure."""

    # Get adjacency matrix or distance matrix
    # For small N, Floyd-Warshall is fine
    # For larger N, we might need Johnson's or repeated Dijkstra
    # But if we are here, N is likely small (< 500)

    try:
        if distance_matrix is not None:
            D = distance_matrix.copy()
        elif has_explicit_edge_lengths(G):
            # Floyd-Warshall's public weight argument is an attribute name, so
            # use the shared callable fallback when per-edge ``length`` must
            # take precedence over the legacy ``weight`` channel.
            return _compute_phi_s_exact_python_fallback(
                G, nodes, delta_nfr, alpha, dtype
            )
        else:
            # Use networkx floyd_warshall_numpy if available
            # It returns a matrix of distances
            D = nx.floyd_warshall_numpy(G, nodelist=nodes)
    except Exception:
        # Fallback if graph is disconnected or other issue
        # Or if we want to support weighted graphs explicitly
        # Construct manually via Dijkstra if FW fails or is too slow?
        # For N < 500, FW is fast.
        return _compute_phi_s_exact_python_fallback(G, nodes, delta_nfr, alpha, dtype)

    # Cast before exponentiation so a genuinely extended dtype retains its
    # intermediate range as well as its accumulator precision.
    D = np.asarray(D, dtype=dtype)
    # Mask diagonal (self-interaction)
    np.fill_diagonal(D, np.inf)

    # Compute potential
    # Φ_i = Σ_j ΔNFR_j / D_ij^α

    # Only reachable, positive distances contribute, as in the scalar kernel.
    # A finite substitute for infinity invents cross-component interaction.
    valid_distances = np.isfinite(D) & (D > 0.0)

    # ΔNFR vector
    dnfr_vec = np.array([delta_nfr[node] for node in nodes], dtype=dtype)

    if np.any(dnfr_vec < 0.0) and np.any(dnfr_vec > 0.0):
        # A dot product may lose a residual such as 1e30 + 1 - 1e30.
        # Use the same compensated reduction as streamed BFS/Dijkstra.
        return {
            node: compensated_sum(
                dnfr_vec[valid_distances[i]] / D[i, valid_distances[i]] ** alpha,
                dtype=dtype,
            )
            for i, node in enumerate(nodes)
        }

    # Matrix-vector product
    inv_D = np.zeros_like(D, dtype=dtype)
    inv_D[valid_distances] = 1.0 / (D[valid_distances] ** alpha)
    phi_vec = inv_D @ dnfr_vec

    return {node: float(phi_vec[i]) for i, node in enumerate(nodes)}


def _compute_phi_s_exact_python_fallback(G, nodes, delta_nfr, alpha, dtype):
    """Fallback for when vectorization fails."""
    potential = {}
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(
            G, src, weight=structural_path_weight(G)
        )
        contributions = (
            dtype(delta_nfr[dst]) / dtype(distance) ** alpha
            for dst, distance in lengths.items()
            if dst != src and math.isfinite(distance) and distance > 0.0
        )
        potential[src] = compensated_sum(contributions, dtype=dtype)
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
    num_nodes = len(nodes)
    if not nodes:
        return {}
    if not landmarks:
        return {node: 0.0 for node in nodes}
    if reverse_landmark_distances is None:
        if G.is_directed():
            reverse = G.reverse(copy=False)
            reverse_weight = structural_path_weight(reverse)
            reverse_landmark_distances = {
                node: nx.single_source_dijkstra_path_length(
                    reverse, node, weight=reverse_weight
                )
                for node in landmarks
            }
        else:
            reverse_landmark_distances = landmark_distances

    outward = np.asarray([
        [landmark_distances[landmark].get(node, np.inf) for node in nodes]
        for landmark in landmarks
    ], dtype=dtype)
    inward = np.asarray([
        [reverse_landmark_distances[landmark].get(node, np.inf) for node in nodes]
        for landmark in landmarks
    ], dtype=dtype)
    pressure = np.asarray([delta_nfr[node] for node in nodes], dtype=dtype)
    potential = np.zeros(num_nodes, dtype=dtype)

    # A two-dimensional working block avoids the old L x batch x N tensor.
    # Infinity remains infinity throughout, including for unreachable pairs.
    batch_size = 200
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        distances = np.full((end - start, num_nodes), np.inf, dtype=dtype)
        for index in range(len(landmarks)):
            through_landmark = inward[index, start:end, None] + outward[index, None, :]
            np.minimum(distances, through_landmark, out=distances)
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        valid = np.isfinite(distances) & (distances > 0.0)
        inverse = np.zeros_like(distances)
        inverse[valid] = 1.0 / distances[valid] ** alpha
        potential[start:end] = inverse @ pressure
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
    # θ_j - θ_i
    diffs = theta_arr[edge_src] - theta_arr[edge_dst]

    # Wrap to [-π, π]
    wrapped_diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

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
    """Vectorized computation of ΔNFR Flux J_ΔNFR.

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
        Degree of each node.

    Returns
    -------
    np.ndarray
        ΔNFR flux for each node.
    """
    # Sum ΔNFR_j for all neighbors
    neighbor_sums = np.zeros(len(dnfr_arr), dtype=dtype)
    np.add.at(neighbor_sums, edge_dst, dnfr_arr[edge_src])

    # Mean neighbor ΔNFR
    with np.errstate(divide="ignore", invalid="ignore"):
        neighbor_means = neighbor_sums / degrees

    # Fix isolated nodes
    neighbor_means[degrees == 0] = 0.0

    # J = Mean(Neighbors) - Self
    # For isolated nodes, neighbor_means is 0, so result is -Self.
    # However, the original code says: "if not neighbors: flux[i] = 0.0"
    # So we must mask isolated nodes explicitly.
    result = neighbor_means - dnfr_arr
    result[degrees == 0] = 0.0

    return result


def compute_phase_gradient_and_curvature_vectorized(
    theta_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized computation of |∇φ| and K_φ.

    |∇φ|_i = mean(|wrap(θ_i - θ_j)|)
    K_φ_i = wrap(θ_i - circular_mean(θ_neighbors))

    Parameters
    ----------
    theta_arr : np.ndarray
        Array of phase values.
    edge_src : np.ndarray
        Indices of neighbor nodes (j).
    edge_dst : np.ndarray
        Indices of center nodes (i).
    degrees : np.ndarray
        Degree of each node.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (gradient_arr, curvature_arr)
    """
    n = len(theta_arr)

    # --- Gradient Calculation ---
    # θ_i - θ_j
    diffs = theta_arr[edge_dst] - theta_arr[edge_src]

    # Wrap to [-π, π]
    wrapped_diffs = (diffs + np.pi) % (2 * np.pi) - np.pi

    # Abs diffs
    abs_diffs = np.abs(wrapped_diffs)

    # Sum over neighbors
    grad_sums = np.zeros(n, dtype=dtype)
    np.add.at(grad_sums, edge_dst, abs_diffs)

    # Mean
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_arr = grad_sums / degrees
    grad_arr[degrees == 0] = 0.0

    # --- Curvature Calculation ---
    # Circular mean of neighbors
    # sum(cos(θ_j)), sum(sin(θ_j))
    cos_vals = np.cos(theta_arr[edge_src])
    sin_vals = np.sin(theta_arr[edge_src])

    cos_sums = np.zeros(n, dtype=dtype)
    sin_sums = np.zeros(n, dtype=dtype)

    np.add.at(cos_sums, edge_dst, cos_vals)
    np.add.at(sin_sums, edge_dst, sin_vals)

    # Mean vector (C, S)
    # We don't strictly need to divide by N for atan2, but let's do it for correctness of "mean vector length" check
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cos = cos_sums / degrees
        mean_sin = sin_sums / degrees

    mean_cos[degrees == 0] = 0.0
    mean_sin[degrees == 0] = 0.0

    # Circular mean phase
    mean_phases = np.arctan2(mean_sin, mean_cos)

    # Handle case where mean vector length is near zero (undefined mean phase)
    # In that case, fallback to arithmetic mean (as per original code)
    # Or just 0? Original code: "if mean_vec_length < 1e-9: mean_phase = float(np.mean(neigh_phases))"
    # Vectorized fallback is tricky.
    # Let's compute arithmetic mean as fallback.

    mean_vec_len = np.hypot(mean_cos, mean_sin)
    unstable_mask = mean_vec_len < 1e-9

    if np.any(unstable_mask):
        # Compute arithmetic mean for unstable nodes
        # We need sum(θ_j)
        theta_sums = np.zeros(n, dtype=dtype)
        np.add.at(theta_sums, edge_dst, theta_arr[edge_src])
        with np.errstate(divide="ignore", invalid="ignore"):
            arith_means = theta_sums / degrees
        mean_phases[unstable_mask] = arith_means[unstable_mask]

    # Curvature = wrap(θ_i - mean_phase)
    curv_diffs = theta_arr - mean_phases
    curv_arr = (curv_diffs + np.pi) % (2 * np.pi) - np.pi

    # Fix isolated nodes
    curv_arr[degrees == 0] = 0.0

    return grad_arr, curv_arr


def compute_coherence_length_vectorized(
    G: Any,
    nodes: list[Any],
    delta_nfr: dict[Any, float],
    dtype: type = np.float64,
    distance_matrix: np.ndarray | None = None,
) -> float:
    """Vectorized estimation of coherence length ξ_C.

    Computes spatial autocorrelation of local coherence C_i = 1/(1+|ΔNFR_i|).
    Fits C(r) ~ exp(-r/ξ_C).
    """
    n = len(nodes)
    if n < 3:
        return float("nan")

    # 1. Compute local coherence array via the canonical kernel (numpy-broadcast)
    # C_i = structural_coherence(ΔNFR_i) = 1 / (1 + |ΔNFR_i|)
    from ..metrics.common import structural_coherence

    dnfr_arr = np.array([abs(delta_nfr.get(node, 0.0)) for node in nodes], dtype=dtype)
    coherence_arr = structural_coherence(dnfr_arr)

    # 2. Compute Distance Matrix
    # For N < 1000, full matrix is fine (1M entries = 8MB)
    # For larger N, we should probably fallback to sampling or sparse methods
    # But here we assume we are in the vectorized path which implies reasonable N
    try:
        if distance_matrix is not None:
            D = distance_matrix
        else:
            # Returns matrix of distances
            D = nx.floyd_warshall_numpy(G, nodelist=nodes)
    except Exception:
        return float("nan")

    # 3. Compute Correlation Matrix C_i * C_j
    # Outer product
    Corr_matrix = np.outer(coherence_arr, coherence_arr)

    # 4. Flatten and Filter
    # We only care about upper triangle (symmetric) and non-zero distances
    # Mask for upper triangle, k=1 excludes diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    valid_dists = D[mask]
    valid_corrs = Corr_matrix[mask]

    # Filter out infinity (disconnected), NaN, and negative sentinels
    # (e.g. -1 used by some callers to mark "no path"). Negative or non-finite
    # distances would crash np.bincount after the int cast below.
    finite_mask = np.isfinite(valid_dists) & (valid_dists >= 0)
    valid_dists = valid_dists[finite_mask]
    valid_corrs = valid_corrs[finite_mask]

    if len(valid_dists) < 10:
        return float("nan")

    # 5. Group by distance
    # Since graph is unweighted, distances are integers.
    # We can use bincount for fast grouping if we cast to int.
    # Check if distances are effectively integers
    is_integer_dist = np.all(np.mod(valid_dists, 1) == 0)

    if is_integer_dist:
        d_ints = valid_dists.astype(np.intp)

        # Defensive guard: reject overflow from oversized float distances or
        # any residual negative entries that slipped past the finite/>=0 mask
        # (e.g. caller-supplied distance matrices with custom sentinels).
        if d_ints.size == 0 or np.any(d_ints < 0):
            return float("nan")

        # Sum of correlations per distance
        corr_sums = np.bincount(d_ints, weights=valid_corrs)
        # Count of pairs per distance
        counts = np.bincount(d_ints)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_corrs = corr_sums / counts

        # Extract valid bins (count >= 2 for statistical relevance)
        valid_bins = counts >= 2
        # Also skip distance 0 (shouldn't be there due to triu(k=1) but just in case)
        valid_bins[0] = False

        distances_fit = np.where(valid_bins)[0]
        corrs_fit = mean_corrs[valid_bins]

    else:
        # Fallback for weighted graphs: sort and unique
        # This is slower but general
        unique_dists, inverse_indices = np.unique(valid_dists, return_inverse=True)

        corr_sums = np.zeros_like(unique_dists, dtype=dtype)
        np.add.at(corr_sums, inverse_indices, valid_corrs)

        counts = np.zeros_like(unique_dists, dtype=int)
        np.add.at(counts, inverse_indices, 1)

        mean_corrs = corr_sums / counts

        valid_bins = counts >= 2
        distances_fit = unique_dists[valid_bins]
        corrs_fit = mean_corrs[valid_bins]

    if len(distances_fit) < 3:
        return float("nan")

    # 6. Fit exponential decay
    # ln(C(r)) = -1/ξ_C * r + b

    # Filter positive correlations for log
    pos_mask = corrs_fit > 1e-9
    if np.sum(pos_mask) < 3:
        return float("nan")

    x = distances_fit[pos_mask]
    y = np.log(corrs_fit[pos_mask])

    try:
        # Linear regression
        # slope = (NΣxy - ΣxΣy) / (NΣx² - (Σx)²)
        # or just use polyfit
        slope, _ = np.polyfit(x, y, 1)

        if slope >= 0:
            return float("nan")

        xi_c = -1.0 / slope
        return float(xi_c)
    except Exception:
        return float("nan")
