"""Unified telemetry for TNFR structural fields.

This module provides a centralized, optimized pass for computing the
canonical structural tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and derived currents.
It minimizes redundant data extraction and distance matrix computations.
"""

from __future__ import annotations

from typing import Any

from ..mathematics.unified_cache import CacheLevel, cache_tnfr_computation
from ..mathematics.unified_numerical import np
from ._helpers import neighborhood_arrays
from .canonical import (
    _get_dnfr,
    _phase_readout_bundle,
    _get_precision_dtype,
    compute_structural_potential,
    estimate_coherence_length,
)
from .phase_curvature import _require_defined_curvature
from .vectorized_ops import (
    compute_dnfr_flux_vectorized,
    compute_phase_current_vectorized,
)


def compute_structural_telemetry(G: Any) -> dict[str, Any]:
    """Compute the full Canonical Structural Suite in a single optimized pass.

    Includes the diagnostic tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and derived currents
    (J_φ, J_ΔNFR). The nodal state triad remains EPI, capacity and phase.

    Returns
    -------
    dict[str, Any]
        {
            'phi_s': dict[NodeId, float],      # Structural Potential
            'grad_phi': dict[NodeId, float],   # Phase Gradient
            'curv_phi': dict[NodeId, float],   # Phase Curvature
            'xi_c': float,                     # Coherence Length
            'j_phi': dict[NodeId, float],      # Phase Current
            'j_dnfr': dict[NodeId, float]      # ΔNFR Flux
        }

    Node and neighbor iteration orders are part of the cache input, because
    finite reductions and the large-graph coherence sample consume them.
    Curvature uses binary64 trigonometric components with exact represented
    reduction. An exact represented joint-zero neighborhood raises
    ``UndefinedPhaseCurvatureError``; independent gradient remains available.
    """
    from .canonical import _VECTORIZATION_AVAILABLE

    # Raw pressure and phase admission precede this outer telemetry cache.
    nodes = tuple(G.nodes())
    pressure = tuple(_get_dnfr(G, node) for node in nodes)
    observation, gradient, curvature = _phase_readout_bundle(G)
    _require_defined_curvature(observation)
    neighbors = tuple(tuple(G.neighbors(node)) for node in nodes)
    cached = _structural_telemetry_cached(
        G, nodes, neighbors, pressure, bool(_VECTORIZATION_AVAILABLE),
    )
    # Every public field map is detached, including the pressure/current maps
    # stored by the outer cache. Mutating one read-out must not change another.
    detached = {key: dict(value) if isinstance(value, dict) else value
                for key, value in cached.items()}
    detached.update(grad_phi=gradient, curv_phi=curvature)
    return detached


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={"graph_topology", "node_phase", "node_dnfr", "precision_mode"},
)
def _structural_telemetry_cached(
    G, node_order, neighbor_order, pressure_values, coherence_vectorized,
):
    """Cache the numerical paths and ordered inputs consumed by this read-out."""
    dtype = _get_precision_dtype()
    nodes = list(node_order)
    n = len(nodes)

    if n == 0:
        return {
            "phi_s": {},
            "grad_phi": {},
            "curv_phi": {},
            "xi_c": float("nan"),
            "j_phi": {},
            "j_dnfr": {},
        }

    # 1. Extract Arrays (O(N))
    # We do this once for all fields
    observation, grad_phi, curv_phi = _phase_readout_bundle(G)
    _require_defined_curvature(observation)
    phases = np.array(observation.primitive_phases, dtype=dtype)
    dnfr_map = dict(zip(nodes, pressure_values))
    dnfr_arr = np.array([dnfr_map[node] for node in nodes], dtype=dtype)

    # 3. Compute Gradient & Curvature (O(E))
    edge_src, edge_dst, degrees = neighborhood_arrays(G, nodes, dtype=dtype)

    # 4. Use the exact canonical kernel at every size, sharing its cache and
    # sparse shortest-path strategy with direct field and U6 callers.
    phi_s = compute_structural_potential(G)

    # 5. Compute Coherence Length ξ_C via the single canonical kernel
    #    (:func:`estimate_coherence_length`) — the same one ``tetrad()`` uses,
    #    with its separately identified spectral-gap fallback. Directed or
    #    degenerate domains may legitimately have no available estimate.
    xi_c = estimate_coherence_length(G)

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
        "phi_s": phi_s,
        "grad_phi": grad_phi,
        "curv_phi": curv_phi,
        "xi_c": xi_c,
        "j_phi": j_phi,
        "j_dnfr": j_dnfr,
    }
