
# =============================================================================
# TNFR EXTENDED CANONICAL FIELDS - PRODUCTION DEPLOYMENT
# Generated: 2025-11-12
# Status: CANONICAL (J_φ, J_ΔNFR), STANDARD_SPECTRAL (νf_variance)
# =============================================================================

"""
Extended canonical fields for TNFR physics engine.

This module provides canonical implementations of J_φ (Phase Current) and 
J_ΔNFR (ΔNFR Flux) fields, promoted from research status based on robust
multi-topology validation.

Validation Evidence:
- J_φ ↔ K_φ: r̄ = +0.592 ± 0.092, 100% sign consistency (48 samples)
- J_ΔNFR ↔ Φ_s: r̄ = -0.471 ± 0.159, 100% sign consistency (48 samples)
"""

import numpy as np
import networkx as nx
from typing import Dict, Any

# Core canonical field computations
from .canonical_j_phi import compute_phase_current
from .canonical_j_dnfr import compute_dnfr_flux  
from .spectral_metrics import compute_vf_variance_canonical
from .calibration import calibrate_tc_xi_correlation

# Extended canonical hexad
EXTENDED_CANONICAL_FIELDS = [
    'Φ_s',        # Structural Potential
    '|∇φ|',       # Phase Gradient  
    'K_φ',        # Phase Curvature
    'ξ_C',        # Coherence Length
    'J_φ',        # Phase Current (NEW)
    'J_ΔNFR'      # ΔNFR Flux (NEW)
]

def compute_extended_canonical_suite(G: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """
    Compute complete extended canonical field suite.

    Returns all six canonical fields for comprehensive TNFR analysis.
    """

    from tnfr.physics.fields import (
        compute_structural_potential,
        compute_phase_gradient, 
        compute_phase_curvature,
        estimate_coherence_length
    )

    results = {}

    # Original canonical tetrad
    results['Φ_s'] = compute_structural_potential(G)
    results['|∇φ|'] = compute_phase_gradient(G)  
    results['K_φ'] = compute_phase_curvature(G)
    results['ξ_C'] = estimate_coherence_length(G)

    # Extended canonical fields (newly promoted)
    results['J_φ'] = compute_phase_current(G, 'theta')
    results['J_ΔNFR'] = compute_dnfr_flux(G, 'ΔNFR')

    return results

# Export functions for integration
__all__ = [
    'compute_phase_current',
    'compute_dnfr_flux', 
    'compute_vf_variance_canonical',
    'calibrate_tc_xi_correlation',
    'compute_extended_canonical_suite',
    'EXTENDED_CANONICAL_FIELDS'
]
