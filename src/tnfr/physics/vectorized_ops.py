"""Vectorized operations for TNFR physics.

This module provides optimized NumPy implementations of structural field computations
to replace slow Python loops in canonical.py.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    import networkx as nx
except ImportError:
    nx = None

def compute_phi_s_exact_vectorized(
    G: Any,
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    alpha: float,
    dtype: type = np.float64,
    distance_matrix: Optional[np.ndarray] = None
) -> Dict[Any, float]:
    """Vectorized exact Φ_s computation."""
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Get adjacency matrix or distance matrix
    # For small N, Floyd-Warshall is fine
    # For larger N, we might need Johnson's or repeated Dijkstra
    # But if we are here, N is likely small (< 500)
    
    try:
        if distance_matrix is not None:
            D = distance_matrix.copy()
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

    # Handle infinity (disconnected)
    D[np.isinf(D)] = 1e9 # Large number to make potential ~0
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(D, np.inf)
    
    # Compute potential
    # Φ_i = Σ_j ΔNFR_j / D_ij^α
    
    # Inverse distance matrix
    # Avoid division by zero (diagonal is inf)
    with np.errstate(divide='ignore'):
        inv_D = 1.0 / (D ** alpha)
    
    # Fix diagonal (1/inf = 0)
    inv_D[np.isinf(D)] = 0.0
    
    # ΔNFR vector
    dnfr_vec = np.array([delta_nfr[node] for node in nodes], dtype=dtype)
    
    # Matrix-vector product
    phi_vec = inv_D @ dnfr_vec
    
    return {node: float(phi_vec[i]) for i, node in enumerate(nodes)}

def _compute_phi_s_exact_python_fallback(G, nodes, delta_nfr, alpha, dtype):
    """Fallback for when vectorization fails."""
    potential = {}
    for src in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        total = dtype(0.0)
        for dst, d in lengths.items():
            if dst == src or d <= 0:
                continue
            total += dtype(delta_nfr[dst] / (d**alpha))
        potential[src] = float(total)
    return potential

def compute_phi_s_landmarks_vectorized(
    G: Any,
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    alpha: float,
    landmarks: List[Any],
    landmark_distances: Dict[Any, Dict[Any, float]],
    dtype: type = np.float64
) -> Dict[Any, float]:
    """Vectorized landmark approximation for Φ_s."""
    num_nodes = len(nodes)
    num_landmarks = len(landmarks)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # 1. Build Landmark Distance Matrix D_L (L x N)
    D_L = np.zeros((num_landmarks, num_nodes), dtype=dtype)
    
    for i, l in enumerate(landmarks):
        dists = landmark_distances[l]
        # Fill row
        # We iterate nodes to ensure order
        # Optimization: use map/array creation if dists is complete
        # But dists might be sparse if disconnected?
        # Assuming connected component or handling inf
        
        # Vectorized fill
        # Create a temporary array filled with inf
        row = np.full(num_nodes, np.inf, dtype=dtype)
        
        # We need to map node IDs to indices efficiently
        # Doing this in a loop is slow.
        # Better: iterate over dists items
        for n, d in dists.items():
            if n in node_to_idx:
                row[node_to_idx[n]] = d
        
        D_L[i, :] = row

    # 2. Prepare vectors
    dnfr_vec = np.array([delta_nfr[n] for n in nodes], dtype=dtype)
    phi_vec = np.zeros(num_nodes, dtype=dtype)
    
    # 3. Chunked Computation
    batch_size = 200 # Tunable
    
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batch_len = end_idx - start_idx
        
        # D_L_src: (L, B)
        D_L_src = D_L[:, start_idx:end_idx]
        
        # D_L_dst: (L, N)
        D_L_dst = D_L
        
        # Approx dist: |D_L_src - D_L_dst|
        # Shape: (L, B, N)
        # Broadcasting: (L, B, 1) - (L, 1, N)
        # Note: D_L contains infs. inf - inf = nan.
        # We need to handle this.
        
        # Mask infs
        # If either is inf, approx dist is inf (disconnected)
        # We can replace inf with a large number for subtraction?
        # No, |inf - 5| = inf. |inf - inf| = nan.
        # Let's use 1e9 for inf.
        
        D_L_src_safe = np.where(np.isinf(D_L_src), 1e9, D_L_src)
        D_L_dst_safe = np.where(np.isinf(D_L_dst), 1e9, D_L_dst)
        
        diff = np.abs(D_L_src_safe[:, :, None] - D_L_dst_safe[:, None, :])
        
        # Min over landmarks: (B, N)
        approx_dist = np.min(diff, axis=0)
        
        # Restore infs where approx_dist is large (meaning disconnected)
        # If approx_dist > 1e8, treat as inf
        # But wait, if both are 1e9, diff is 0.
        # If one is 1e9, diff is ~1e9.
        # If both are 1e9 (disconnected from landmark), diff is 0.
        # This implies distance 0 between two disconnected nodes? WRONG.
        # If both are disconnected from landmark, we have NO INFO from that landmark.
        # We should ignore that landmark.
        # But we take MIN over landmarks.
        # If all landmarks are disconnected, we have a problem.
        # Assuming graph is connected for now (standard TNFR assumption).
        
        # Clamp to 1.0 to avoid zero division
        approx_dist = np.maximum(approx_dist, 1.0)
        
        # Inverse power
        inv_dist = 1.0 / (approx_dist ** alpha)
        
        # Mask self-interaction
        # Global indices for batch rows: start_idx + i
        # We want inv_dist[i, start_idx + i] = 0
        rows = np.arange(batch_len)
        cols = np.arange(start_idx, end_idx)
        inv_dist[rows, cols] = 0.0
        
        # Compute approximate potential
        phi_batch = inv_dist @ dnfr_vec
        
        # 4. Landmark Corrections
        for l_idx, l_node in enumerate(landmarks):
            global_l_idx = node_to_idx[l_node]
            
            # Subtract approx term
            approx_term = inv_dist[:, global_l_idx] * delta_nfr[l_node]
            phi_batch -= approx_term
            
            # Add exact term
            d_exact = D_L_src[l_idx, :] # (B,)
            
            # Handle self-interaction (if src is the landmark)
            # d_exact is 0.0 there.
            # We want term to be 0.0.
            
            # Safe inverse
            with np.errstate(divide='ignore'):
                term_exact = delta_nfr[l_node] / (d_exact ** alpha)
            
            # Fix infs (div by zero or disconnected)
            term_exact[np.isinf(term_exact)] = 0.0
            
            phi_batch += term_exact
            
        phi_vec[start_idx:end_idx] = phi_batch
        
    return {n: float(phi_vec[i]) for i, n in enumerate(nodes)}

def compute_vf_variance_vectorized(
    G: Any,
    vf_attr: str = 'νf',
    radius: int = 1
) -> Dict[Any, float]:
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
    S2 = A @ (vf_values ** 2)
    
    # Count of neighbors: N = A @ 1
    N_counts = np.sum(A, axis=1)
    
    # Avoid division by zero (should not happen with self-loops)
    N_counts[N_counts == 0] = 1.0
    
    # Mean: mu = S1 / N
    mu = S1 / N_counts
    
    # Population Variance: sigma^2 = (S2 / N) - mu^2
    var_pop = (S2 / N_counts) - (mu ** 2)
    
    # Sample Variance: var_sample = var_pop * (N / (N - 1))
    # If N=1, var=0
    with np.errstate(divide='ignore', invalid='ignore'):
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
    mu_4 = np.sum(A2 ** 2) / n
    
    if normalized:
        return mu_4 / (n**2)
    return mu_4

def compute_phase_current_vectorized(
    theta_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64
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
    with np.errstate(divide='ignore', invalid='ignore'):
        result = sums / degrees
        
    # Fix isolated nodes (degree 0 -> result NaN/Inf -> 0)
    result[degrees == 0] = 0.0
    
    return result

def compute_dnfr_flux_vectorized(
    dnfr_arr: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    degrees: np.ndarray,
    dtype: type = np.float64
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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    dtype: type = np.float64
) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.ndarray, np.ndarray]
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
    with np.errstate(divide='ignore', invalid='ignore'):
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
    with np.errstate(divide='ignore', invalid='ignore'):
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
    nodes: List[Any],
    delta_nfr: Dict[Any, float],
    dtype: type = np.float64,
    distance_matrix: Optional[np.ndarray] = None
) -> float:
    """Vectorized estimation of coherence length ξ_C.
    
    Computes spatial autocorrelation of local coherence C_i = 1/(1+|ΔNFR_i|).
    Fits C(r) ~ exp(-r/ξ_C).
    """
    n = len(nodes)
    if n < 3:
        return float('nan')
        
    # 1. Compute local coherence array
    # C_i = 1 / (1 + |ΔNFR_i|)
    dnfr_arr = np.array([abs(delta_nfr.get(node, 0.0)) for node in nodes], dtype=dtype)
    coherence_arr = 1.0 / (1.0 + dnfr_arr)
    
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
        return float('nan')
        
    # 3. Compute Correlation Matrix C_i * C_j
    # Outer product
    Corr_matrix = np.outer(coherence_arr, coherence_arr)
    
    # 4. Flatten and Filter
    # We only care about upper triangle (symmetric) and non-zero distances
    # Mask for upper triangle, k=1 excludes diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    valid_dists = D[mask]
    valid_corrs = Corr_matrix[mask]
    
    # Filter out infinity (disconnected)
    finite_mask = np.isfinite(valid_dists)
    valid_dists = valid_dists[finite_mask]
    valid_corrs = valid_corrs[finite_mask]
    
    if len(valid_dists) < 10:
        return float('nan')
        
    # 5. Group by distance
    # Since graph is unweighted, distances are integers.
    # We can use bincount for fast grouping if we cast to int.
    # Check if distances are effectively integers
    is_integer_dist = np.all(np.mod(valid_dists, 1) == 0)
    
    if is_integer_dist:
        d_ints = valid_dists.astype(np.intp)
        
        # Sum of correlations per distance
        corr_sums = np.bincount(d_ints, weights=valid_corrs)
        # Count of pairs per distance
        counts = np.bincount(d_ints)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
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
        return float('nan')
        
    # 6. Fit exponential decay
    # ln(C(r)) = -1/ξ_C * r + b
    
    # Filter positive correlations for log
    pos_mask = corrs_fit > 1e-9
    if np.sum(pos_mask) < 3:
        return float('nan')
        
    x = distances_fit[pos_mask]
    y = np.log(corrs_fit[pos_mask])
    
    try:
        # Linear regression
        # slope = (NΣxy - ΣxΣy) / (NΣx² - (Σx)²)
        # or just use polyfit
        slope, _ = np.polyfit(x, y, 1)
        
        if slope >= 0:
            return float('nan')
            
        xi_c = -1.0 / slope
        return float(xi_c)
    except Exception:
        return float('nan')


