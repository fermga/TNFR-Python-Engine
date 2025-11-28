"""Unified telemetry for TNFR structural fields.

This module provides a centralized, optimized pass for computing the
Canonical Structural Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and other metrics.
It minimizes redundant data extraction and distance matrix computations.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    import networkx as nx
except ImportError:
    nx = None

import numpy as np

from ..utils.cache import cache_tnfr_computation, CacheLevel
from .canonical import _get_precision_dtype, _get_phase, _get_dnfr
from .vectorized_ops import (
    compute_phi_s_exact_vectorized,
    compute_phase_gradient_and_curvature_vectorized,
    compute_coherence_length_vectorized,
    compute_phase_current_vectorized,
    compute_dnfr_flux_vectorized
)

@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={'graph_topology', 'node_phase', 'node_dnfr'}
)
def compute_structural_telemetry(G: Any) -> Dict[str, Any]:
    """Compute the full Canonical Structural Suite in a single optimized pass.
    
    Includes the Canonical Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and the
    Extended Canonical Fluxes (J_φ, J_ΔNFR).
    
    Returns
    -------
    Dict[str, Any]
        {
            'phi_s': Dict[NodeId, float],      # Structural Potential
            'grad_phi': Dict[NodeId, float],   # Phase Gradient
            'curv_phi': Dict[NodeId, float],   # Phase Curvature
            'xi_c': float,                     # Coherence Length
            'j_phi': Dict[NodeId, float],      # Phase Current
            'j_dnfr': Dict[NodeId, float]      # ΔNFR Flux
        }
    """
    dtype = _get_precision_dtype()
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return {
            'phi_s': {},
            'grad_phi': {},
            'curv_phi': {},
            'xi_c': float('nan'),
            'j_phi': {},
            'j_dnfr': {}
        }

    # 1. Extract Arrays (O(N))
    # We do this once for all fields
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    phases = np.array([_get_phase(G, node) for node in nodes], dtype=dtype)
    dnfr_map = {node: _get_dnfr(G, node) for node in nodes}
    dnfr_arr = np.array([dnfr_map[node] for node in nodes], dtype=dtype)
    degrees = np.array([G.degree[node] for node in nodes], dtype=dtype)
    
    # 2. Compute Distance Matrix (O(N^3) or O(N*E))
    # Only if N is reasonable for dense matrix operations
    distance_matrix = None
    if n < 1000 and nx is not None:
        try:
            distance_matrix = nx.floyd_warshall_numpy(G, nodelist=nodes)
            # Fix infinity for potential calculation (handled inside ops, but good to be safe)
        except Exception:
            pass
            
    # 3. Compute Gradient & Curvature (O(E))
    # Build edge lists for vectorized op
    edge_src_list = []
    edge_dst_list = []
    is_directed = G.is_directed()
    
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            # If u is center, v is neighbor: src=v, dst=u
            edge_src_list.append(v_idx)
            edge_dst_list.append(u_idx)
            
            if not is_directed:
                edge_src_list.append(u_idx)
                edge_dst_list.append(v_idx)
                
    edge_src = np.array(edge_src_list, dtype=np.intp)
    edge_dst = np.array(edge_dst_list, dtype=np.intp)
    
    grad_arr, curv_arr = compute_phase_gradient_and_curvature_vectorized(
        phases, edge_src, edge_dst, degrees, dtype=dtype
    )
    
    grad_phi = {node: float(grad_arr[i]) for i, node in enumerate(nodes)}
    curv_phi = {node: float(curv_arr[i]) for i, node in enumerate(nodes)}
    
    # 4. Compute Structural Potential (O(N^2) with precomputed D)
    phi_s = compute_phi_s_exact_vectorized(
        G, nodes, dnfr_map, alpha=2.0, dtype=dtype, distance_matrix=distance_matrix
    )
    
    # 5. Compute Coherence Length (O(N^2) with precomputed D)
    xi_c = compute_coherence_length_vectorized(
        G, nodes, dnfr_map, dtype=dtype, distance_matrix=distance_matrix
    )
    
    # 6. Compute Extended Fluxes (O(E))
    # Phase Current
    j_phi_arr = compute_phase_current_vectorized(
        phases, edge_src, edge_dst, degrees, dtype=dtype
    )
    j_phi = {node: float(j_phi_arr[i]) for i, node in enumerate(nodes)}
    
    # ΔNFR Flux
    j_dnfr_arr = compute_dnfr_flux_vectorized(
        dnfr_arr, edge_src, edge_dst, degrees, dtype=dtype
    )
    j_dnfr = {node: float(j_dnfr_arr[i]) for i, node in enumerate(nodes)}
    
    return {
        'phi_s': phi_s,
        'grad_phi': grad_phi,
        'curv_phi': curv_phi,
        'xi_c': xi_c,
        'j_phi': j_phi,
        'j_dnfr': j_dnfr
    }
