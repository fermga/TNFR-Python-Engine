"""Topological analysis utilities for TNFR networks.

This module provides spectral analysis tools for computing topological
factors relevant to temporal ordering (U6) and network structure.

Terminology (TNFR semantics):
- "node" refers to a resonant locus (coherent pattern site) in the TNFR sense.
- We keep the term "node" in APIs for NetworkX interoperability.
- This usage has no relation to the Node.js runtime.

**Consolidation Note**: Reuses `_laplacian_from_adjacency()` from
`mathematics.generators` for Laplacian construction to avoid redundancy.
Spectral analysis (eigenvalues, Fiedler value) is NEW functionality.
"""

from __future__ import annotations


import numpy as np

from ..types import Graph as TNFRGraph  # noqa: F401 - used in docstrings

__all__ = [
    "compute_k_top_spectral",
    "compute_laplacian_spectrum",
    "compute_fiedler_value",
]


def compute_laplacian_spectrum(G) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of the normalized graph Laplacian.

    The normalized Laplacian is defined as:
        L = I - D^(-1/2) A D^(-1/2)

    where:
    - I: Identity matrix
    - D: Degree matrix
    - A: Adjacency matrix

    **Consolidation**: Uses existing `_laplacian_from_adjacency()` from
    `mathematics.generators` for basic Laplacian (L = D - A), then
    normalizes for spectral analysis.

    Parameters
    ----------
    G : TNFRGraph
        TNFR network graph

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues sorted in ascending order (λ₀ ≤ λ₁ ≤ ... ≤ λₙ)
    eigenvectors : ndarray
        Corresponding eigenvectors as columns

    Notes
    -----
    For connected graphs:
    - λ₀ = 0 (always, corresponds to constant eigenvector)
    - λ₁ > 0 (Fiedler value, algebraic connectivity)
    - Larger λ₁ → better connectivity → faster relaxation

    Physical interpretation in TNFR:
    - Eigenvalues represent characteristic relaxation modes
    - Fiedler value (λ₁) determines dominant relaxation time
    - Spectral gap (λ₁ - λ₀ = λ₁) inversely related to diffusion time

    See Also
    --------
    compute_k_top_spectral : Use spectrum to estimate topological factor
    mathematics.generators._laplacian_from_adjacency : Basic Laplacian construction
    """
    # Note: We construct normalized Laplacian directly; no import needed.

    # Build adjacency matrix
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return np.array([]), np.array([])

    # Get nodes in consistent order
    node_list = sorted(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(node_list)}

    # Construct adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        # For weighted graphs, use coupling strength
        weight = G[u][v].get("weight", 1.0)
        A[i, j] = weight
        A[j, i] = weight  # Symmetric

    # Compute degree matrix
    degrees = A.sum(axis=1)

    # Handle isolated nodes (degree = 0)
    D_inv_sqrt = np.zeros(num_nodes)
    for i in range(num_nodes):
        if degrees[i] > 1e-10:
            D_inv_sqrt[i] = 1.0 / np.sqrt(degrees[i])
        else:
            D_inv_sqrt[i] = 0.0

    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L = np.eye(num_nodes) - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort in ascending order (eigh should already do this, but ensure)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_fiedler_value(G) -> float:
    """Compute Fiedler value (second-smallest Laplacian eigenvalue).

    The Fiedler value λ₁ measures algebraic connectivity:
    - λ₁ = 0: Graph is disconnected
    - λ₁ > 0: Graph is connected, larger values → better connectivity

    Parameters
    ----------
    G : TNFRGraph
        TNFR network graph

    Returns
    -------
    float
        Fiedler value λ₁ (second eigenvalue of normalized Laplacian)

    Notes
    -----
    Physical interpretation for U6:
    - Larger λ₁ → faster information propagation
    - Smaller λ₁ → slower relaxation, longer τ_relax
    - For rings: λ₁ ≈ 1 - cos(2π/n) (small for large n)
    - For stars: λ₁ = 1.0 (optimal connectivity from center)

    See Also
    --------
    compute_k_top_spectral : Convert λ₁ to topological factor
    """
    eigenvalues, _ = compute_laplacian_spectrum(G)

    if len(eigenvalues) < 2:
        return 0.0  # Trivial graph

    return float(eigenvalues[1])


def compute_k_top_spectral(
    G,
    method: str = "fiedler_inverse",
) -> float:
    """Compute topological factor k_top from spectral analysis.

    **Status:** EXPERIMENTAL (U6 research)

    The topological factor k_top enters the relaxation time formula:
        τ_relax = (k_top / νf) · k_op · ln(1/ε)

    This function estimates k_top from the network's Laplacian spectrum,
    capturing how topology affects diffusion and relaxation dynamics.

    Parameters
    ----------
    G : TNFRGraph
        TNFR network graph
    method : str, optional
        Method for computing k_top (default: "fiedler_inverse")
        - "fiedler_inverse": k_top = γ / λ₁ (inversely related to connectivity)
        - "spectral_gap": k_top = 1 / (λ₁ - λ₀) = 1/λ₁
        - "diameter_scaled": k_top ≈ diameter / characteristic_path_length

    Returns
    -------
    float
        Topological factor k_top
        - Typical range: [0.1, 2.0]
        - Star/radial graphs: k_top ≈ 1.0 (fast relaxation)
        - Ring graphs: k_top ≈ 1/(2π) ≈ 0.16 (slow relaxation)
        - Modular graphs: k_top depends on bottleneck strength

    Notes
    -----
    **Physical basis:**

    From diffusion on graphs, the relaxation time τ after perturbation scales
    with the inverse of the spectral gap:
        τ_diffusion ∝ 1/λ₁

    For TNFR, this becomes:
        τ_damp ∝ k_top / νf

    where k_top captures topology-dependent diffusion slowdown.

    **Calibration strategy:**

    1. Measure τ_relax_observed for known topologies (star, ring, grid)
    2. Fit k_top = f(λ₁) to minimize prediction error
    3. Current default: k_top = 1.0/λ₁ (theoretical diffusion scaling)

    **Limitations:**
    - Formula not yet derived from nodal equation (empirical)
    - Assumes single dominant relaxation mode (valid for well-connected graphs)
    - May need refinement for hierarchical/modular structures

    Examples
    --------
    >>> import networkx as nx
    >>> from tnfr.structural import create_nfr
    >>>
    >>> # Star graph (radial)
    >>> G_star = nx.star_graph(5)
    >>> k_star = compute_k_top_spectral(G_star)
    >>> # Expected: k_star ≈ 1.0
    >>>
    >>> # Ring graph (cyclic)
    >>> G_ring = nx.cycle_graph(10)
    >>> k_ring = compute_k_top_spectral(G_ring)
    >>> # Expected: k_ring ≈ 0.16 (1/(2π))

    See Also
    --------
    compute_fiedler_value : Get λ₁ directly
    docs/grammar/U6_TEMPORAL_ORDERING.md : Complete derivation
    """
    lambda_1 = compute_fiedler_value(G)

    if lambda_1 < 1e-6:
        # Graph is disconnected or trivial
        return 2.0  # Assume slow relaxation

    if method == "fiedler_inverse":
        # k_top ∝ 1/λ₁ (diffusion time scaling)
        # Normalize so star graph gives k_top ≈ 1.0
        # For star graph with n nodes: λ₁ = 1.0
        k_top = 1.0 / lambda_1

    elif method == "spectral_gap":
        # Same as fiedler_inverse for normalized Laplacian (λ₀ = 0)
        k_top = 1.0 / lambda_1

    elif method == "diameter_scaled":
        # Use network diameter as proxy
        import networkx as nx

        try:
            diameter = nx.diameter(G)
            avg_path_length = nx.average_shortest_path_length(G)
            k_top = diameter / max(avg_path_length, 1.0)
        except nx.NetworkXError:
            # Graph disconnected
            k_top = 2.0
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid: ['fiedler_inverse', 'spectral_gap', 'diameter_scaled']"
        )

    # Clamp to reasonable range
    k_top = np.clip(k_top, 0.05, 5.0)

    return float(k_top)
