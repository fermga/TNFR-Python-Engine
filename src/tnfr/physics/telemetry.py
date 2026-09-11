"""Unified telemetry for TNFR structural fields.

This module provides a centralized, optimized pass for computing the
Canonical Structural Triad (Φ_s, |∇φ|, K_φ) and experimental ξ_C correlation analysis.
It minimizes redundant data extraction and distance matrix computations.
"""

from __future__ import annotations

from typing import Any

from ..mathematics.unified_cache import CacheLevel, cache_tnfr_computation
from ..mathematics.unified_numerical import np
from ._helpers import neighborhood_arrays
from .canonical import (
    _get_dnfr,
    _get_phase,
    _get_precision_dtype,
    compute_structural_potential,
    estimate_coherence_length,
)
from .vectorized_ops import (
    compute_dnfr_flux_vectorized,
    compute_phase_current_vectorized,
    compute_phase_gradient_and_curvature_vectorized,
)


@cache_tnfr_computation(
    level=CacheLevel.DERIVED_METRICS,
    dependencies={"graph_topology", "node_phase", "node_dnfr", "precision_mode"},
)
def compute_structural_telemetry(G: Any) -> dict[str, Any]:
    """Compute the full Canonical Structural Suite in a single optimized pass.

    Includes the Canonical Structural Triad (Φ_s, |∇φ|, K_φ) and the
    Extended Canonical Fluxes (J_φ, J_ΔNFR).

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
    """
    dtype = _get_precision_dtype()
    nodes = list(G.nodes())
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
    phases = np.array([_get_phase(G, node) for node in nodes], dtype=dtype)
    dnfr_map = {node: _get_dnfr(G, node) for node in nodes}
    dnfr_arr = np.array([dnfr_map[node] for node in nodes], dtype=dtype)

    # 3. Compute Gradient & Curvature (O(E))
    edge_src, edge_dst, degrees = neighborhood_arrays(G, nodes, dtype=dtype)

    grad_arr, curv_arr = compute_phase_gradient_and_curvature_vectorized(
        phases, edge_src, edge_dst, degrees, dtype=dtype
    )

    grad_phi = {node: float(grad_arr[i]) for i, node in enumerate(nodes)}
    curv_phi = {node: float(curv_arr[i]) for i, node in enumerate(nodes)}

    # 4. Use the exact canonical kernel at every size, sharing its cache and
    # sparse shortest-path strategy with direct field and U6 callers.
    phi_s = compute_structural_potential(G)

    # 5. Compute Coherence Length ξ_C via the single canonical kernel
    #    (:func:`estimate_coherence_length`) — the same one ``tetrad()`` uses,
    #    with the spectral-gap fallback, so telemetry() and tetrad() agree and
    #    never return NaN on a connected graph (ADR-005, invariant #5).
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
