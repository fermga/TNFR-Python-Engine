"""TNFR Extended Canonical Fields - Flux and Transport

Two canonical diagnostic fields read directed neighbor contrasts:

- J_φ: Mean sine of neighbor phase displacement
- J_ΔNFR: Mean neighbor pressure difference

These complement the core tetrad (Φ_s, |∇φ|, K_φ, ξ_C) as read-only
statistics. They are not measured time derivatives or evolution laws.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from ..mathematics._neighbor_differences import mean_neighbor_difference
from ..mathematics.unified_numerical import compute_phase_difference, np
from ..metrics.common import finite_population_std
from ._edge_semantics import structural_path_weight
from ._helpers import finite_real_scalar
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
        wrapped_diffs = compute_phase_difference(neighbor_phases, phi_i)

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
    The stable binary64 neighbor-difference owner is shared with linear
    pressure arithmetic. An unrepresentable final contrast raises instead of
    caching infinity or NaN; reusing arithmetic does not install a pressure law.

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

    for i, neighbors in zip(nodes, neighbor_order):
        flux[i] = mean_neighbor_difference(
            dnfr_values[i], [dnfr_values[j] for j in neighbors]
        )

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
    """Read the population variance of neighboring phase-gradient magnitudes.

    This research snapshot contains no time interval and is not a deformation
    rate. Unique outgoing neighbors are counted once; isolates return zero.
    Only the implemented one-hop ``scale=1`` is supported. The compatibility
    parameter does not silently select an unimplemented coarse-graining.
    """
    if finite_real_scalar(scale, "scale") != 1.0:
        raise ValueError("phase strain supports only scale=1")
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
    """Read the legacy inverse-length mean wrapped neighbor displacement.

    The historical name is retained for compatibility. The formula is
    ``mean_j wrap(theta_j-theta_i)/length(i,j)`` over unique outgoing
    neighbors, with zero at isolates. It can be nonzero on a two-node tree;
    it is neither a curl nor circulation, winding or defect evidence. Use
    ``winding_certificates.certify_phase_winding`` for an actual declared cycle.

    Length uses the shared explicit ``length`` channel, then legacy ``weight``
    fallback, then one. Parallel edges use their minimum structural length.
    Every consumed length must be strictly positive, and wrapped differences
    and final results must be representable and finite.
    """
    nodes = list(G.nodes())
    phases = {node: _get_phase(G, node) for node in nodes}
    edge_length = structural_path_weight(G)
    vorticity = {}

    for node in nodes:
        terms = []
        for neighbor in G.neighbors(node):
            length = edge_length(node, neighbor, G[node][neighbor])
            if length <= 0.0:
                raise ValueError("phase vorticity requires strictly positive lengths")
            difference = _wrap_angle(phases[neighbor] - phases[node])
            if not math.isfinite(difference):
                raise ValueError("wrapped phase displacement must be finite")
            terms.append((difference, length))

        if not terms:
            vorticity[node] = 0.0
            continue

        try:
            result = math.fsum(gap / length for gap, length in terms) / len(terms)
            if not math.isfinite(result):
                raise OverflowError
        except (OverflowError, ValueError):
            # Keep finite cancellation/means even when individual represented
            # quotients or their sum overflow; the wrapped gaps stay unchanged.
            exact = sum(
                (Fraction(gap) / Fraction(length) for gap, length in terms),
                Fraction(),
            ) / len(terms)
            try:
                result = float(exact)
            except OverflowError as exc:
                raise ValueError("phase vorticity is outside finite range") from exc
        vorticity[node] = result

    return vorticity


def compute_reorganization_strain(G):
    """Read the population standard deviation of neighboring stored pressure.

    This research snapshot is spatial dispersion, not a force or a measured
    deformation rate. The shared pressure reader validates the first present
    alias and uses zero for missing pressure. Unique outgoing neighbors count
    once, and isolates return zero. Scaling avoids overflow in the variance
    of finite large pressures.
    """
    nodes = list(G.nodes())
    pressure = {node: _get_dnfr(G, node) for node in nodes}
    strain = {}

    for node in nodes:
        strain[node] = finite_population_std(
            (pressure[neighbor] for neighbor in G.neighbors(node)), name="pressure"
        )

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
