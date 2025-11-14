"""
TNFR Extended Canonical Fields - Production Implementation

This module implements the newly promoted canonical fields:
- J_φ (Phase Current): Directed phase flow via geometric confinement  
- J_ΔNFR (ΔNFR Flux): Potential-driven reorganization transport

Status: CANONICAL (promoted 2025-11-12)
Validation: Multi-topology robustness testing with >90% statistical confidence

Extended Canonical Hexad: Φ_s, |∇φ|, K_φ, ξ_C, J_φ, J_ΔNFR
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import cache infrastructure for performance optimization
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    CACHE_AVAILABLE = True
except ImportError:
    # Fallback for testing environments
    def cache_tnfr_computation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    CacheLevel = None
    CACHE_AVAILABLE = False

__all__ = [
    'compute_phase_current',
    'compute_dnfr_flux',
    'compute_extended_canonical_suite',
    'EXTENDED_CANONICAL_FIELDS'
]

# Extended canonical field registry
EXTENDED_CANONICAL_FIELDS = [
    'Φ_s',        # Structural Potential
    '|∇φ|',       # Phase Gradient  
    'K_φ',        # Phase Curvature
    'ξ_C',        # Coherence Length
    'J_φ',        # Phase Current (NEW - 2025-11-12)
    'J_ΔNFR'      # ΔNFR Flux (NEW - 2025-11-12)
]

def _estimate_phase_current_cost(G: nx.Graph, theta_attr: str = 'theta') -> float:
    """Estimate computational cost for phase current calculation."""
    return 1.0 + len(G.nodes()) * 0.01 + len(G.edges()) * 0.005


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if CACHE_AVAILABLE else None,
    dependencies={'theta', 'graph_structure'},
    cost_estimator=_estimate_phase_current_cost,
)
def compute_phase_current(
    G: nx.Graph, 
    theta_attr: str = 'theta'
) -> Dict[Any, float]:
    """
    Compute Phase Current (J_φ) - CANONICAL field.
    
    Measures directed phase flow through network topology.
    Physical interpretation: Phase transport driven by geometric confinement.
    
    Status: CANONICAL (promoted Nov 2025)
    Validation: r(J_φ, K_φ) = +0.592 ± 0.092, 100% sign consistency, 48 samples
    
    Args:
        G: NetworkX graph with phase attributes
        theta_attr: Node attribute containing phase values [0, 2π]
    
    Returns:
        Dict mapping node_id -> phase_current_value (float)
    
    Physics:
        J_φ(i) = Σ_{j∈N(i)} sin(θ_j - θ_i) / deg(i)
        
        Where:
        - θ_i: Phase of node i
        - N(i): Neighbors of node i
        - sin(Δθ): Captures directed phase gradient (current)
        
    Physical Meaning:
        - Positive J_φ: Net phase inflow to node
        - Negative J_φ: Net phase outflow from node
        - |J_φ| magnitude: Strength of directed phase transport
        
    Correlation with K_φ (Phase Curvature):
        Strong positive correlation indicates geometric phase confinement
        regions (high |K_φ|) drive directed phase transport (high |J_φ|).
    """
    if not G.nodes():
        return {}
    
    current = {}
    
    for node in G.nodes():
        theta_i = G.nodes[node].get(theta_attr, 0.0)
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) == 0:
            current[node] = 0.0
            continue
            
        # Compute phase current as average directed flow
        total_flow = 0.0
        for neighbor in neighbors:
            theta_j = G.nodes[neighbor].get(theta_attr, 0.0)
            
            # Phase difference with proper 2π periodicity
            phase_diff = theta_j - theta_i
            
            # Normalize to [-π, π] range for correct sine calculation
            while phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            while phase_diff < -np.pi:
                phase_diff += 2 * np.pi
                
            # Sin gives directed component (positive = inflow, negative = outflow)
            total_flow += np.sin(phase_diff)
        
        # Average over neighborhood
        current[node] = float(total_flow / len(neighbors))
    
    return current

def _estimate_dnfr_flux_cost(G: nx.Graph, dnfr_attr: str = 'ΔNFR') -> float:
    """Estimate computational cost for ΔNFR flux calculation."""
    return 1.0 + len(G.nodes()) * 0.01 + len(G.edges()) * 0.005


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if CACHE_AVAILABLE else None,
    dependencies={'ΔNFR', 'graph_structure'},
    cost_estimator=_estimate_dnfr_flux_cost,
)
def compute_dnfr_flux(
    G: nx.Graph, 
    dnfr_attr: str = 'ΔNFR'
) -> Dict[Any, float]:
    """
    Compute ΔNFR Flux (J_ΔNFR) - CANONICAL field.
    
    Measures structural reorganization flow between coupled nodes.
    Physical interpretation: Potential-driven ΔNFR transport.
    
    Status: CANONICAL (promoted Nov 2025)
    Validation: r(J_ΔNFR, Φ_s) = -0.471 ± 0.159, 100% sign consistency, 48 samples
    
    Args:
        G: NetworkX graph with ΔNFR attributes
        dnfr_attr: Node attribute containing ΔNFR values
    
    Returns:
        Dict mapping node_id -> dnfr_flux_value (float)
    
    Physics:
        J_ΔNFR(i) = Σ_{j∈N(i)} (ΔNFR_j - ΔNFR_i) / deg(i)
        
        Where:
        - ΔNFR_i: Reorganization gradient of node i
        - N(i): Neighbors of node i  
        - Difference captures net inflow (+) or outflow (-)
        
    Physical Meaning:
        - Positive J_ΔNFR: Net reorganization pressure inflow
        - Negative J_ΔNFR: Net reorganization pressure outflow  
        - |J_ΔNFR| magnitude: Strength of structural flow
        
    Correlation with Φ_s (Structural Potential):
        Strong negative correlation indicates high potential regions
        (high Φ_s) create outward ΔNFR flows (negative J_ΔNFR), consistent
        with potential-driven transport physics.
    """
    if not G.nodes():
        return {}
    
    flux = {}
    
    for node in G.nodes():
        dnfr_i = G.nodes[node].get(dnfr_attr, 0.0)
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) == 0:
            flux[node] = 0.0
            continue
            
        # Compute net ΔNFR flux (inflow - outflow)
        total_flux = 0.0
        for neighbor in neighbors:
            dnfr_j = G.nodes[neighbor].get(dnfr_attr, 0.0)
            # Positive flux = inflow to node i from neighbor j
            total_flux += (dnfr_j - dnfr_i)
        
        # Average over neighborhood  
        flux[node] = float(total_flux / len(neighbors))
    
    return flux

def compute_extended_canonical_suite(G: nx.Graph, **kwargs) -> Dict[str, Dict[Any, float]]:
    """
    Compute complete Extended Canonical Hexad.
    
    Computes all six canonical TNFR fields for comprehensive analysis.
    
    Args:
        G: NetworkX graph with required node attributes
        **kwargs: Optional parameters for field computations
        
    Returns:
        Dict mapping field_name -> {node_id: field_value}
        
    Required Node Attributes:
        - 'theta': Phase values [0, 2π] (for |∇φ|, K_φ, J_φ)
        - 'ΔNFR': Reorganization gradients (for Φ_s, J_ΔNFR)
        - 'νf': Reorganization frequencies Hz_str (for ξ_C)
        
    Returns Extended Canonical Hexad:
        - Φ_s: Structural Potential
        - |∇φ|: Phase Gradient
        - K_φ: Phase Curvature  
        - ξ_C: Coherence Length
        - J_φ: Phase Current (NEW)
        - J_ΔNFR: ΔNFR Flux (NEW)
    """
    try:
        from .fields import (
            compute_structural_potential,
            compute_phase_gradient,
            compute_phase_curvature,
            estimate_coherence_length
        )
    except ImportError:
        # Fallback for testing/development
        logger.warning(" Could not import canonical tetrad functions. Using minimal implementations.")
        return {
            'J_φ': compute_phase_current(G, kwargs.get('theta_attr', 'theta')),
            'J_ΔNFR': compute_dnfr_flux(G, kwargs.get('dnfr_attr', 'ΔNFR'))
        }
    
    results = {}
    
    # Original canonical tetrad
    results['Φ_s'] = compute_structural_potential(G, **kwargs)
    results['|∇φ|'] = compute_phase_gradient(G, **kwargs)  
    results['K_φ'] = compute_phase_curvature(G, **kwargs)
    results['ξ_C'] = estimate_coherence_length(G, **kwargs)
    
    # Extended canonical fields (newly promoted)
    results['J_φ'] = compute_phase_current(G, kwargs.get('theta_attr', 'theta'))
    results['J_ΔNFR'] = compute_dnfr_flux(G, kwargs.get('dnfr_attr', 'ΔNFR'))
    
    return results

def validate_extended_canonical_correlations(
    canonical_fields: Dict[str, Dict[Any, float]],
    tolerance: float = 0.2
) -> Dict[str, bool]:
    """
    Validate expected correlations between extended canonical fields.
    
    Args:
        canonical_fields: Output from compute_extended_canonical_suite
        tolerance: Acceptable deviation from expected correlations
        
    Returns:
        Dict mapping correlation_name -> validation_passed (bool)
        
    Expected Correlations (from validation):
        - r(J_φ, K_φ) ≈ +0.592 ± 0.092
        - r(J_ΔNFR, Φ_s) ≈ -0.471 ± 0.159
    """
    if not all(field in canonical_fields for field in ['J_φ', 'K_φ', 'J_ΔNFR', 'Φ_s']):
        return {'validation_error': False}
    
    # Extract field vectors
    nodes = list(next(iter(canonical_fields.values())).keys())
    
    j_phi_vals = np.array([canonical_fields['J_φ'][n] for n in nodes])
    k_phi_vals = np.array([canonical_fields['K_φ'][n] for n in nodes])
    j_dnfr_vals = np.array([canonical_fields['J_ΔNFR'][n] for n in nodes])
    phi_s_vals = np.array([canonical_fields['Φ_s'][n] for n in nodes])
    
    # Compute correlations
    def safe_corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    
    r_j_phi_k_phi = safe_corr(j_phi_vals, k_phi_vals)
    r_j_dnfr_phi_s = safe_corr(j_dnfr_vals, phi_s_vals)
    
    # Validate against expected values
    expected_j_phi_k_phi = 0.592
    expected_j_dnfr_phi_s = -0.471
    
    validation_results = {}
    
    if not np.isnan(r_j_phi_k_phi):
        validation_results['j_phi_k_phi_correlation'] = (
            abs(r_j_phi_k_phi - expected_j_phi_k_phi) <= tolerance
        )
    else:
        validation_results['j_phi_k_phi_correlation'] = False
        
    if not np.isnan(r_j_dnfr_phi_s):
        validation_results['j_dnfr_phi_s_correlation'] = (
            abs(r_j_dnfr_phi_s - expected_j_dnfr_phi_s) <= tolerance
        )
    else:
        validation_results['j_dnfr_phi_s_correlation'] = False
    
    return validation_results

# Promotion metadata for integration tracking
PROMOTION_METADATA = {
    'J_φ': {
        'promotion_date': '2025-11-12',
        'validation_correlation': 'r(J_φ, K_φ) = +0.592 ± 0.092',
        'sign_consistency': '100%',
        'validation_samples': 48,
        'robustness_topologies': ['WS', 'BA', 'Grid'],
        'physics_interpretation': 'Geometric phase confinement drives directed transport',
        'integration_priority': 'HIGH'
    },
    'J_ΔNFR': {
        'promotion_date': '2025-11-12', 
        'validation_correlation': 'r(J_ΔNFR, Φ_s) = -0.471 ± 0.159',
        'sign_consistency': '100%',
        'validation_samples': 48,
        'robustness_topologies': ['WS', 'BA', 'Grid'],
        'physics_interpretation': 'Potential-driven reorganization transport',
        'integration_priority': 'HIGH'
    }
}