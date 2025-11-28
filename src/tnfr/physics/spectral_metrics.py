"""
TNFR Standard Spectral Metrics - Production Implementation

This module implements the newly promoted standard spectral metric:
- νf_variance: Second moment of reorganization rate distribution

Status: STANDARD_SPECTRAL (promoted 2025-11-12)
Validation: r(νf_variance, Φ_s) = +0.478

Integration with Extended Canonical Hexad for comprehensive TNFR analysis.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any

# Import vectorized operations
try:
    from .vectorized_ops import (
        compute_vf_variance_vectorized,
        compute_spectral_kurtosis_vectorized
    )
    _VECTORIZATION_AVAILABLE = True
except ImportError:
    _VECTORIZATION_AVAILABLE = False

__all__ = [
    'compute_vf_variance',
    'compute_standard_spectral_suite',
    'STANDARD_SPECTRAL_METRICS'
]

STANDARD_SPECTRAL_METRICS = [
    'νf_variance',              # Local reorganization rate dispersion
    'spectral_gap_sensitivity', # Low-frequency mode participation
    'laplacian_centrality',     # Spectral influence via eigenvectors
    'spectral_kurtosis'         # Hypothesis O: Structural Obstruction
]


def compute_vf_variance(G: nx.Graph, vf_attr: str = 'νf', 
                       radius: int = 1) -> Dict[Any, float]:
    """
    Compute νf Variance (Second Moment) - STANDARD SPECTRAL metric.
    
    Measures local variability in reorganization rates within neighborhood.
    Physical interpretation: Structural gradient strength via rate dispersion.
    
    Status: STANDARD_SPECTRAL (promoted Nov 2025)
    Validation: r(νf_variance, Φ_s) = +0.478
    
    Args:
        G: NetworkX graph with νf attributes
        vf_attr: Node attribute containing reorganization rates (Hz_str)
        radius: Neighborhood radius for variance computation
    
    Returns:
        Dict mapping node_id -> vf_variance_value (float)
    
    Physics:
        νf_var(i) = Var[νf_j : j ∈ N_r(i)]
        
        Where:
        - νf_j: Reorganization frequency of node j
        - N_r(i): r-neighborhood of node i  
        - Higher variance indicates stronger local reorganization gradients
        
    Correlation with Φ_s (Structural Potential):
        Positive correlation indicates regions with high structural potential
        exhibit greater variability in reorganization rates, consistent
        with gradient-driven dynamics.
    """
    if not G.nodes():
        return {}
    
    # Use vectorized implementation if available and radius=1
    if _VECTORIZATION_AVAILABLE and radius == 1:
        result = compute_vf_variance_vectorized(G, vf_attr, radius)
        if result is not None:
            return result
    
    variance = {}
    
    for node in G.nodes():
        # Get neighborhood within radius
        if radius == 1:
            neighborhood = [node] + list(G.neighbors(node))
        else:
            try:
                # Use BFS to get nodes within radius
                lengths = nx.single_source_shortest_path_length(
                    G, node, cutoff=radius)
                neighborhood = list(lengths.keys())
            except Exception:
                # Fallback to immediate neighbors
                neighborhood = [node] + list(G.neighbors(node))
        
        # Collect νf values in neighborhood
        vf_values = []
        for neighbor in neighborhood:
            vf = G.nodes[neighbor].get(vf_attr, 0.0)
            vf_values.append(vf)
        
        # Compute variance
        if len(vf_values) > 1:
            # Sample variance (ddof=1)
            variance[node] = float(np.var(vf_values, ddof=1))
        else:
            variance[node] = 0.0
    
    return variance


def compute_spectral_kurtosis(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute Spectral Kurtosis (4th Moment) - Hypothesis O Metric.
    
    Measures the deviation of the graph spectrum from the "quasi-random" 
    baseline (Paley graphs). High kurtosis indicates "grid-like" or 
    tensor-product structures (Obstruction).
    
    Args:
        G: NetworkX graph
        normalized: If True, divide by N^2 (standard scaling for adjacency)
        
    Returns:
        float: The 4th spectral moment mu_4
    """
    # Use vectorized implementation if available (avoids eigendecomposition)
    if _VECTORIZATION_AVAILABLE:
        return compute_spectral_kurtosis_vectorized(G, normalized)

    try:
        from scipy.linalg import eigvalsh
        adj = nx.to_numpy_array(G)
        eigenvalues = eigvalsh(adj)
        n = len(G)
        
        # 4th Moment: sum(lambda^4) / N
        mu_4 = np.sum(eigenvalues**4) / n
        
        if normalized:
            return mu_4 / (n**2)
        return mu_4
        
    except ImportError:
        return 0.0
    except Exception:
        return 0.0

def compute_standard_spectral_suite(G: nx.Graph, **kwargs) -> Dict[str, Any]:
    """
    Compute standard spectral metrics suite.
    
    Includes:
    - vf_variance (Local reorganization rate dispersion)
    - spectral_kurtosis (Hypothesis O obstruction)
    
    Returns:
        Dictionary containing all computed metrics.
    """
    results = {}
    
    # 1. VF Variance (Standard)
    results['vf_variance'] = compute_vf_variance(G, **kwargs)
    
    # 2. Spectral Kurtosis (Hypothesis O)
    # This returns a single float, not a dict per node
    results['spectral_kurtosis'] = compute_spectral_kurtosis(G)
    
    return results


# Promotion metadata
SPECTRAL_PROMOTION_METADATA = {
    'νf_variance': {
        'promotion_date': '2025-11-12',
        'validation_correlation': 'r(νf_variance, Φ_s) = +0.478',
        'physical_interpretation': 'Local reorganization rate dispersion',
        'computational_complexity': 'O(N·k·r) where k=avg_degree, r=radius',
        'integration_priority': 'MEDIUM',
        'recommended_use': 'Structural gradient detection'
    }
}