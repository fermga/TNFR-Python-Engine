"""TNFR Extended Canonical Fields - Flux and Transport

Two canonical diagnostic fields read directed neighbor contrasts:

- J_φ: Mean sine of neighbor phase displacement
- J_ΔNFR: Mean neighbor pressure difference

These complement the core tetrad (Φ_s, |∇φ|, K_φ, ξ_C) as read-only
statistics. They are not measured time derivatives or evolution laws.
"""

from __future__ import annotations

from typing import Any

from ..constants.aliases import ALIAS_DNFR
from ..mathematics.unified_numerical import np
from ._helpers import get_dnfr as _get_dnfr
from ._helpers import get_phase as _get_phase
from ._helpers import neighborhood_arrays
from ._helpers import wrap_angle as _wrap_angle

try:
    import networkx as nx
except ImportError:
    nx = None

# Import canonical fields for interdependence
try:
    from .canonical import compute_phase_gradient
    from .vectorized_ops import (
        compute_dnfr_flux_vectorized,
        compute_phase_current_vectorized,
    )
except ImportError:
    # Shared scalar readers remain authoritative in the loop fallback.
    pass


# Import TNFR cache system
from ..mathematics.unified_cache import CacheLevel, cache_tnfr_computation

_CACHE_AVAILABLE = True

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_THETA
except ImportError:
    ALIAS_THETA = ["phase", "theta"]


def compute_phase_current(G: Any) -> dict[Any, float]:
    """Read the signed mean-sine phase statistic over unique neighbors.

    ``J_phi(i)=mean_j sin(theta_j-theta_i)`` uses successors on directed
    graphs, counts parallel neighbors once and includes a self-loop once.
    Isolates have the explicit value zero. Edge conductance does not weight
    this diagnostic, including on zero-conductance support edges.

    In exact arithmetic, with nonzero neighbor resultant S and regular
    curvature K=wrap(theta_i-Arg(S)), ``J_phi=-|S|*sin(K)/degree``.
    Nonzero current and curvature therefore have opposite signs away from
    the half-turn cut. Their numeric readers use separately rounded
    trigonometric and angular operations; no exact binary64 identity is
    asserted. See TNFR_VARIATIONAL_PRINCIPLE section 13.6 for the reciprocal
    support pair cost and its state-dependent phase-pressure metric.

    This directional statistic selects no phase evolution or actual transport
    rate. Zero mean sine may arise from antipodal phases or cancellation;
    it does not establish a defined circular mean, zero canonical phase
    pressure, or equilibrium of the full nodal state.

    Parameters
    ----------
    G : TNFRGraph
        Graph with node phase attributes

    Returns
    -------
    dict[NodeId, float]
        Detached per-node signed mean sine, a read-only diagnostic.
    """
    nodes = tuple(G.nodes())
    phases = tuple(_get_phase(G, node) for node in nodes)
    neighbors = tuple(tuple(G.neighbors(node)) for node in nodes)
    return dict(_phase_current_cached(G, nodes, neighbors, phases))


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_phase"},
)
def _phase_current_cached(G, node_order, neighbor_order, phase_values):
    """Cache current only after raw phase admission; public maps are detached."""
    current: dict[Any, float] = {}

    nodes = list(node_order)
    if not nodes:
        return {}

    # Check for vectorization support
    try:
        # Phase array
        phases = np.array(phase_values, dtype=np.float64)
        edge_src, edge_dst, degrees = neighborhood_arrays(G, nodes)

        # Vectorized computation
        current_arr = compute_phase_current_vectorized(
            phases, edge_src, edge_dst, degrees
        )

        return {node: float(current_arr[i]) for i, node in enumerate(nodes)}

    except Exception:
        # Fallback to loop if vectorization fails (e.g. memory issue)
        pass

    phases_dict = dict(zip(nodes, phase_values))

    for i in nodes:
        neighbors = list(G.neighbors(i))
        if not neighbors:
            current[i] = 0.0
            continue

        phi_i = phases_dict[i]

        # Phase current as mean of sine differences (captures flow direction)
        neighbor_phases = np.array([phases_dict[j] for j in neighbors])
        phase_diffs = neighbor_phases - phi_i

        # Wrap differences to [-π, π] for proper sine calculation
        wrapped_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi

        # Current = mean sine (positive = inward flow, negative = outward)
        current[i] = float(np.mean(np.sin(wrapped_diffs)))

    return current


def compute_dnfr_flux(G: Any) -> dict[Any, float]:
    """Read the mean-neighbor pressure contrast, named ΔNFR flux J_ΔNFR.

    **Definition**:
        J_ΔNFR(i) = Σ_{j∈neighbors(i)} (ΔNFR_j - ΔNFR_i) / |neighbors(i)|

    This is a read-only diagnostic of stored pressure on unique support
    neighbors. It contains no capacity or time factor and does not specify
    pressure evolution, physical transport or a sustaining interaction.
    Correlation with structural potential cannot establish those laws.

    Parameters
    ----------
    G : TNFRGraph
        Graph with node ΔNFR attributes

    Returns
    -------
    dict[NodeId, float]
        Positive means neighbors have higher mean stored pressure than the
        node; negative means lower. Zero is zero mean neighbor pressure
        contrast, not necessarily zero pressure or nodal equilibrium.

    References
    ----------
    - theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md (field definitions)
    - theory/TNFR_VARIATIONAL_PRINCIPLE.md (scoped graph-field dynamics)
    """
    nodes = tuple(G.nodes())
    pressure = tuple(_get_dnfr(G, node) for node in nodes)
    neighbors = tuple(tuple(G.neighbors(node)) for node in nodes)
    return dict(_dnfr_flux_cached(G, nodes, neighbors, pressure))


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS if _CACHE_AVAILABLE else None,
    dependencies={"graph_topology", "node_dnfr"},
)
def _dnfr_flux_cached(G, node_order, neighbor_order, pressure_values):
    """Cache flux only after raw pressure admission; public maps are detached."""
    flux: dict[Any, float] = {}

    nodes = list(node_order)
    if not nodes:
        return {}

    # Check for vectorization support
    try:
        # ΔNFR array
        dnfr_arr = np.array(pressure_values, dtype=np.float64)
        edge_src, edge_dst, degrees = neighborhood_arrays(G, nodes)

        # Vectorized computation
        flux_arr = compute_dnfr_flux_vectorized(dnfr_arr, edge_src, edge_dst, degrees)

        return {node: float(flux_arr[i]) for i, node in enumerate(nodes)}

    except Exception:
        pass

    dnfr_values = dict(zip(nodes, pressure_values))

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


def compute_extended_canonical_suite(G: Any) -> dict[str, dict[Any, float]]:
    """Compute all extended canonical fields in optimized fashion.

    Returns
    -------
    dict[str, dict[Any, float]]
        Dictionary with keys 'phase_current' and 'dnfr_flux' containing
        the respective field values per node.
    """
    return {
        "phase_current": compute_phase_current(G),
        "dnfr_flux": compute_dnfr_flux(G),
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
            weight = G[node][neighbor].get("weight", 1.0)
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
    dict[str, dict]
        All extended field values keyed by field name
    """
    return {
        "phase_strain": compute_phase_strain(G),
        "phase_vorticity": compute_phase_vorticity(G),
        "reorganization_strain": compute_reorganization_strain(G),
    }


__all__ = [
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    "compute_phase_strain",
    "compute_phase_vorticity",
    "compute_reorganization_strain",
    "compute_extended_dynamics_suite",
]
