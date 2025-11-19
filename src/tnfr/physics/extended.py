"""TNFR Extended Canonical Fields - Flux and Transport

The two newly-promoted CANONICAL flux fields that capture directed transport:

- J_φ: Phase current (geometric phase confinement drives directed transport)
- J_ΔNFR: ΔNFR flux (potential-driven reorganization transport)

These complement the core tetrad (Φ_s, |∇φ|, K_φ, ξ_C) by adding transport
dynamics while maintaining read-only telemetry semantics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import canonical fields for interdependence
try:
    from .canonical import (
        _get_phase, _get_dnfr, _wrap_angle,
        compute_phase_gradient,
        compute_phase_curvature, compute_structural_potential
    )
except ImportError:
    # Fallback definitions if canonical module not available
    def _get_phase(G: Any, node: Any) -> float:
        return G.nodes[node].get('phase', G.nodes[node].get('theta', 0.0))

    def _get_dnfr(G: Any, node: Any) -> float:
        return G.nodes[node].get('delta_nfr', G.nodes[node].get('dnfr', 0.0))

    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

# Import TNFR cache system
try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

    def cache_tnfr_computation(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    class CacheLevel:
        DERIVED_METRICS = None


# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_THETA, ALIAS_DNFR
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={'graph_topology', 'node_phase'},
)
def compute_phase_current(G: Any) -> Dict[Any, float]:
    """Compute phase current J_φ for each locus [CANONICAL - PROMOTED Nov 12, 2025].

    **Canonical Status**: Promoted November 12, 2025 after robust multi-topology
    validation (48 samples, r(J_φ, K_φ) = +0.592 ± 0.092, 100% sign consistency).

    **Physics**: Geometric phase confinement drives directed transport.
    Phase current captures the local "flow" of phase information through the
    network, complementing static curvature K_φ with transport dynamics.

    **Definition**:
        J_φ(i) = Σ_{j∈neighbors(i)} sin(φ_j - φ_i) / |neighbors(i)|

    **Validation Evidence**:
    - 48 samples across WS, BA, Grid topologies
    - Ultra-robust correlation: r(J_φ, K_φ) = +0.592 ± 0.092
    - 100% sign consistency across parameter sweeps
    - Integration priority: HIGH

    **Usage as Telemetry**:
    - Read-only field computation (never mutates EPI)
    - Complements K_φ by adding directed transport dimension
    - High |J_φ| indicates active phase transport vs static confinement

    Parameters
    ----------
    G : TNFRGraph
        Graph with node phase attributes

    Returns
    -------
    Dict[NodeId, float]
        Phase current per node. Positive = net inward flow,
        negative = net outward flow, zero = equilibrium.

    References
    ----------
    - AGENTS.md § Extended Canonical Fields (Nov 12, 2025 promotion)
    - Validation data: 48-sample multi-topology experiment
    - Physics: Geometric transport from phase field gradients
    """
    current: Dict[Any, float] = {}

    nodes = list(G.nodes())
    phases = {node: _get_phase(G, node) for node in nodes}

    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            current[i] = 0.0
            continue

        phi_i = phases[i]

        # Phase current as mean of sine differences (captures flow direction)
        neighbor_phases = np.array([phases[j] for j in neighbors])
        phase_diffs = neighbor_phases - phi_i

        # Wrap differences to [-π, π] for proper sine calculation
        wrapped_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi

        # Current = mean sine (positive = inward flow, negative = outward)
        current[i] = float(np.mean(np.sin(wrapped_diffs)))

    return current


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={'graph_topology', 'node_dnfr'},
)
def compute_dnfr_flux(G: Any) -> Dict[Any, float]:
    """Compute ΔNFR flux J_ΔNFR for each locus [CANONICAL - PROMOTED Nov 12, 2025].

    **Canonical Status**: Promoted November 12, 2025 after robust multi-topology
    validation (48 samples, r(J_ΔNFR, Φ_s) = -0.471 ± 0.159, 100% sign consistency).

    **Physics**: Potential-driven reorganization transport. ΔNFR flux captures
    the local "flow" of structural reorganization pressure, analogous to current
    flow in potential fields.

    **Definition**:
        J_ΔNFR(i) = Σ_{j∈neighbors(i)} (ΔNFR_j - ΔNFR_i) / |neighbors(i)|

    **Validation Evidence**:
    - 48 samples across WS, BA, Grid topologies
    - Ultra-robust correlation: r(J_ΔNFR, Φ_s) = -0.471 ± 0.159
    - 100% sign consistency across parameter sweeps
    - Integration priority: HIGH

    **Usage as Telemetry**:
    - Read-only field computation (never mutates EPI)
    - Complements Φ_s by adding directed transport dimension
    - Positive J_ΔNFR = net inward pressure, negative = net outward

    Parameters
    ----------
    G : TNFRGraph
        Graph with node ΔNFR attributes

    Returns
    -------
    Dict[NodeId, float]
        ΔNFR flux per node. Positive = net inward reorganization pressure,
        negative = net outward pressure, zero = equilibrium.

    References
    ----------
    - AGENTS.md § Extended Canonical Fields (Nov 12, 2025 promotion)
    - Validation data: 48-sample multi-topology experiment
    - Physics: Transport from ΔNFR gradients (potential-driven flow)
    """
    flux: Dict[Any, float] = {}

    nodes = list(G.nodes())
    dnfr_values = {node: _get_dnfr(G, node) for node in nodes}

    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            flux[i] = 0.0
            continue

        dnfr_i = dnfr_values[i]

        # ΔNFR flux as mean difference (captures pressure gradients)
        neighbor_dnfr = np.array([dnfr_values[j] for j in neighbors])
        dnfr_diffs = neighbor_dnfr - dnfr_i

        # Flux = mean difference (positive = inward pressure, negative = outward)
        flux[i] = float(np.mean(dnfr_diffs))

    return flux


def compute_extended_canonical_suite(G: Any) -> Dict[str, Dict[Any, float]]:
    """Compute all extended canonical fields in optimized fashion.

    Returns
    -------
    Dict[str, Dict[Any, float]]
        Dictionary with keys 'phase_current' and 'dnfr_flux' containing
        the respective field values per node.
    """
    return {
        'phase_current': compute_phase_current(G),
        'dnfr_flux': compute_dnfr_flux(G)
    }


# ============================================================================
# RESEARCH-PHASE EXTENDED FIELDS (Not in canonical tetrad)
# ============================================================================
# Additional transport and deformation fields for advanced analysis.


def compute_phase_strain(G, scale=1):
    """Compute spatial phase strain rate (research phase).

    **Status**: RESEARCH (structural deformation analysis)

    Definition
    ----------
    Local deformation rate from phase gradients:
        σ_φ(i) = variance of phase gradients at neighbors

    Physical Interpretation
    -----------------------
    Measures "stretching" or "compression" of phase field locally.
    """
    grad_phi = compute_phase_gradient(G)
    nodes = list(G.nodes())
    strain = {}

    for node in nodes:
        neighbor_grads = []
        for neighbor in G.neighbors(node):
            if neighbor in grad_phi:
                neighbor_grads.append(grad_phi[neighbor])

        if neighbor_grads:
            strain[node] = float(np.var(neighbor_grads))
        else:
            strain[node] = 0.0

    return strain


def compute_phase_vorticity(G):
    """Compute phase vorticity (rotational circulation).

    **Status**: RESEARCH (topological defect detection)

    Definition
    ----------
    Detects phase vortices (spinning patterns):
        ω_φ(i) = weighted sum of phase differences around node

    Physical Interpretation
    -----------------------
    Non-zero vorticity indicates phase singularities/defects.
    """
    nodes = list(G.nodes())
    vorticity = {}

    for node in nodes:
        total_curl = 0.0
        neighbor_count = 0

        for neighbor in G.neighbors(node):
            phi_i = _get_phase(G, node)
            phi_j = _get_phase(G, neighbor)
            d_phi = _wrap_angle(phi_j - phi_i)
            weight = G[node][neighbor].get('weight', 1.0)
            dist_inv = 1.0 / weight
            total_curl += d_phi * dist_inv
            neighbor_count += 1

        if neighbor_count > 0:
            vorticity[node] = total_curl / neighbor_count
        else:
            vorticity[node] = 0.0

    return vorticity


def compute_reorganization_strain(G):
    """Compute ΔNFR-based reorganization strain.

    **Status**: RESEARCH (structural pressure deformation)

    Definition
    ----------
    Spatial variation in reorganization gradients:
        s_Δ(i) = std of ΔNFR at neighbors

    Physical Interpretation
    -----------------------
    High strain indicates unbalanced forces on node.
    """
    nodes = list(G.nodes())
    strain = {}

    for node in nodes:
        neighbor_dnfrs = []
        for neighbor in G.neighbors(node):
            neighbor_data = G.nodes[neighbor]
            for alias in ALIAS_DNFR:
                if alias in neighbor_data:
                    neighbor_dnfrs.append(float(neighbor_data[alias]))
                    break

        if neighbor_dnfrs:
            strain[node] = float(np.std(neighbor_dnfrs))
        else:
            strain[node] = 0.0

    return strain


def compute_extended_dynamics_suite(G):
    """Compute all research-phase extended fields together.

    **Status**: RESEARCH (comprehensive structural analysis)

    Returns
    -------
    Dict[str, Dict]
        All extended field values keyed by field name
    """
    return {
        'phase_strain': compute_phase_strain(G),
        'phase_vorticity': compute_phase_vorticity(G),
        'reorganization_strain': compute_reorganization_strain(G),
    }


__all__ = [
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    "compute_phase_strain",
    "compute_phase_vorticity",
    "compute_reorganization_strain",
    "compute_extended_dynamics_suite"
]
