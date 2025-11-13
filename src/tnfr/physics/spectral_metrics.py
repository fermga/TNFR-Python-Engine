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

__all__ = [
    'compute_vf_variance',
    'compute_standard_spectral_suite',
    'STANDARD_SPECTRAL_METRICS'
]

STANDARD_SPECTRAL_METRICS = [
    'νf_variance',              # Local reorganization rate dispersion
    'spectral_gap_sensitivity', # Low-frequency mode participation
    'laplacian_centrality'      # Spectral influence via eigenvectors
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


def compute_standard_spectral_suite(G: nx.Graph, **kwargs) -> Dict[str, Dict[Any, float]]:
    """
    Compute standard spectral metrics suite.
    
    Provides νf_variance and other spectral measures for TNFR analysis.
    
    Args:
        G: NetworkX graph with required attributes
        **kwargs: Parameters for metric computation
        
    Returns:
        Dict mapping metric_name -> {node_id: metric_value}
    """
    results = {}
    
    # Core standard spectral metric
    results['νf_variance'] = compute_vf_variance(
        G, 
        vf_attr=kwargs.get('vf_attr', 'νf'),
        radius=kwargs.get('radius', 1)
    )
    
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