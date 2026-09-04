# -*- coding: utf-8 -*-
"""Fused ΔNFR computation kernels for optimized backend.

This module provides optimized implementations of ΔNFR computation that fuse
multiple operations to reduce memory traffic and improve cache locality.

The fused kernels maintain exact TNFR semantics while achieving better
performance through:

1. Combined gradient computation (phase + EPI + topology in single pass)
2. Reduced intermediate array allocations
3. Better cache locality through sequential memory access
4. Optional Numba JIT compilation for critical loops

All implementations preserve TNFR structural invariants:
- ΔNFR = w_phase·g_phase + w_epi·g_epi + w_vf·g_vf + w_topo·g_topo
- Isolated nodes receive ΔNFR = 0
- Deterministic results with fixed topology
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..mathematics.unified_numerical import np
from ..utils import get_logger

logger = get_logger(__name__)

# Try to import Numba for JIT acceleration
_NUMBA_AVAILABLE = False
_numba = None
try:
    import numba

    _numba = numba
    _NUMBA_AVAILABLE = True
    logger.debug("Numba JIT available for fused gradient acceleration")
except ImportError:
    logger.debug("Numba not available, using pure NumPy implementation")


def _compute_canonical_gradients_jit_kernel(
    edge_src,
    edge_dst,
    edge_weight,
    phase,
    epi,
    vf,
    w_phase: float,
    w_epi: float,
    w_vf: float,
    w_topo: float,
    n_nodes: int,
    symmetric: bool,
    delta_nfr,
):
    """Numba JIT kernel for canonical TNFR gradient computation.

    Realizes the canonical **outgoing** random-walk operator L_out = I - D^-1 W
    (the operator of :func:`tnfr.physics.structural_diffusion.
    structural_diffusion_operator`): for edge ``u -> v`` node ``u`` receives
    neighbour ``v``.  The EPI channel is edge-weighted (the channel with the
    proven L_rw identity); phase, vf and topology use the unweighted
    neighbourhood (ADR-002).  Undirected graphs carry both (u, v) and (v, u),
    so the result is symmetric and reproduces the legacy path exactly.
    """
    # Allocations (Numba handles these efficiently in nopython mode)
    cos_sum = np.zeros(n_nodes, dtype=np.float64)
    sin_sum = np.zeros(n_nodes, dtype=np.float64)
    epi_wsum = np.zeros(n_nodes, dtype=np.float64)
    epi_weight = np.zeros(n_nodes, dtype=np.float64)
    vf_sum = np.zeros(n_nodes, dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)

    n_edges = edge_src.shape[0]

    # Pass 1: Accumulate neighbour statistics (outgoing orientation)
    for i in range(n_edges):
        u = edge_src[i]
        v = edge_dst[i]
        w = edge_weight[i]

        # u -> v: node u receives neighbour v
        cos_sum[u] += math.cos(phase[v])
        sin_sum[u] += math.sin(phase[v])
        epi_wsum[u] += w * epi[v]
        epi_weight[u] += w
        vf_sum[u] += vf[v]
        count[u] += 1.0

        if symmetric:
            # v -> u: node v receives neighbour u
            cos_sum[v] += math.cos(phase[u])
            sin_sum[v] += math.sin(phase[u])
            epi_wsum[v] += w * epi[u]
            epi_weight[v] += w
            vf_sum[v] += vf[u]
            count[v] += 1.0

    # Pass 2: Compute gradients
    for i in range(n_nodes):
        if count[i] > 0:
            # Phase: g = (theta_mean - theta_node) / pi
            # theta_mean = atan2(sum_sin, sum_cos)
            theta_mean = math.atan2(sin_sum[i], cos_sum[i])

            # angle_diff(a, b) = (a - b + pi) % 2pi - pi
            # We want (theta_mean - phase[i])
            diff = (theta_mean - phase[i] + math.pi) % (2 * math.pi) - math.pi
            g_phase = diff / math.pi

            # EPI: weighted neighbour mean (L_out with edge weights)
            g_epi = 0.0
            if epi_weight[i] > 0.0:
                g_epi = (epi_wsum[i] / epi_weight[i]) - epi[i]
            # VF: unweighted neighbour mean
            g_vf = (vf_sum[i] / count[i]) - vf[i]

            delta_nfr[i] = w_phase * g_phase + w_epi * g_epi + w_vf * g_vf

    # Pass 3: Topology (if needed) — unweighted out-degree neighbourhood
    if w_topo != 0.0:
        deg_sum = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_edges):
            u = edge_src[i]
            v = edge_dst[i]

            # Accumulate neighbour degrees (degree = count)
            deg_sum[u] += count[v]
            if symmetric:
                deg_sum[v] += count[u]

        for i in range(n_nodes):
            if count[i] > 0:
                deg_mean = deg_sum[i] / count[i]
                g_topo = deg_mean - count[i]
                delta_nfr[i] += w_topo * g_topo


# Create JIT-compiled version if Numba is available
if _NUMBA_AVAILABLE:
    try:
        _compute_canonical_gradients_jit = _numba.njit(
            _compute_canonical_gradients_jit_kernel,
            parallel=False,
            fastmath=True,
            cache=True,
        )
        logger.debug("Numba JIT compilation successful for fused gradients")
    except Exception as e:
        logger.warning(f"Numba JIT compilation failed: {e}")
        _compute_canonical_gradients_jit = _compute_canonical_gradients_jit_kernel
else:
    _compute_canonical_gradients_jit = _compute_canonical_gradients_jit_kernel


def compute_fused_gradients(
    *,
    edge_src: Any,
    edge_dst: Any,
    phase: Any,
    epi: Any,
    vf: Any,
    weights: Mapping[str, float],
    edge_weight: Any | None = None,
    use_jit: bool = True,
) -> Any:
    """Compute all ΔNFR gradients in a fused kernel (Directed).

    This function delegates to `compute_fused_gradients_symmetric` with
    `accumulate_both_directions=False` to support directed graphs using
    the canonical TNFR formula (neighbor means).

    Parameters
    ----------
    edge_src : array-like
        Source node indices for each edge (shape: [E])
    edge_dst : array-like
        Destination node indices for each edge (shape: [E])
    phase : array-like
        Phase values for each node (shape: [N])
    epi : array-like
        EPI values for each node (shape: [N])
    vf : array-like
        Structural frequency νf for each node (shape: [N])
    weights : Mapping[str, float]
        ΔNFR component weights (w_phase, w_epi, w_vf, w_topo)
    np : module
        NumPy module
    use_jit : bool, default=True
        Whether to use JIT compilation if available

    Returns
    -------
    ndarray
        Fused gradient vector (shape: [N])
    """
    return compute_fused_gradients_symmetric(
        edge_src=edge_src,
        edge_dst=edge_dst,
        phase=phase,
        epi=epi,
        vf=vf,
        weights=weights,
        edge_weight=edge_weight,
        accumulate_both_directions=False,
        use_jit=use_jit,
    )


def compute_fused_gradients_symmetric(
    *,
    edge_src: Any,
    edge_dst: Any,
    phase: Any,
    epi: Any,
    vf: Any,
    weights: Mapping[str, float],
    edge_weight: Any | None = None,
    accumulate_both_directions: bool = True,
    use_jit: bool = True,
) -> Any:
    """Compute ΔNFR gradients for undirected graphs using fused operations.

    This kernel fuses neighbor accumulation, mean computation, and gradient
    assembly into a single optimized pass. It is specifically designed for
    undirected graphs where interactions are symmetric.

    Parameters
    ----------
    edge_src : ndarray
        Source node indices (shape: [E])
    edge_dst : ndarray
        Destination node indices (shape: [E])
    phase : ndarray
        Phase values (shape: [N])
    epi : ndarray
        EPI values (shape: [N])
    vf : ndarray
        νf values (shape: [N])
    weights : Mapping[str, float]
        Component weights (w_phase, w_epi, w_vf, w_topo)
    accumulate_both_directions : bool, optional
        If True (default), each edge (u, v) contributes to both u and v.
        If False, each edge (u, v) only contributes to v (dst).
        set to False if edge_src/edge_dst already contain both (u,v) and (v,u).
    use_jit : bool, default=True
        Whether to use JIT compilation if available

    Returns
    -------
    ndarray
        ΔNFR gradient vector (shape: [N])
    """
    n_nodes = phase.shape[0]
    n_edges = edge_src.shape[0]

    w_phase = float(weights.get("w_phase", 0.0))
    w_epi = float(weights.get("w_epi", 0.0))
    w_vf = float(weights.get("w_vf", 0.0))
    w_topo = float(weights.get("w_topo", 0.0))

    delta_nfr = np.zeros(n_nodes, dtype=float)

    if n_edges == 0:
        return delta_nfr

    # Edge weights for the EPI channel (canonical L_rw = I - D^-1 W).  Absent
    # weights default to unity, reproducing the unweighted path exactly.
    if edge_weight is None:
        w_edge = np.ones(n_edges, dtype=float)
    else:
        w_edge = np.asarray(edge_weight, dtype=float)
        if w_edge.shape[0] != n_edges:
            raise ValueError(
                "edge_weight length does not match edge_src/edge_dst"
            )

    # JIT Path
    if use_jit and _NUMBA_AVAILABLE and n_edges > 100:
        _compute_canonical_gradients_jit(
            edge_src,
            edge_dst,
            w_edge,
            phase,
            epi,
            vf,
            w_phase,
            w_epi,
            w_vf,
            w_topo,
            n_nodes,
            accumulate_both_directions,
            delta_nfr,
        )
        return delta_nfr

    # Pass 1: Accumulate neighbour statistics for computing means, under the
    # canonical outgoing orientation L_out = I - D^-1 W (node i receives the
    # nodes it points to).  For edge i->j, node i (src) accumulates neighbour
    # j (dst).  Undirected graphs carry both (i,j) and (j,i), so the result
    # is symmetric and identical to the legacy computation.
    #   phase: cos/sin sums for the circular mean
    #   EPI:   weighted value sum + weighted degree (edge-weighted channel)
    #   vf:    unweighted value sum
    neighbor_cos_sum = np.zeros(n_nodes, dtype=float)
    neighbor_sin_sum = np.zeros(n_nodes, dtype=float)
    neighbor_epi_sum = np.zeros(n_nodes, dtype=float)
    neighbor_epi_weight = np.zeros(n_nodes, dtype=float)
    neighbor_vf_sum = np.zeros(n_nodes, dtype=float)
    neighbor_count = np.zeros(n_nodes, dtype=float)

    # Extract neighbour values
    phase_src_vals = phase[edge_src]
    phase_dst_vals = phase[edge_dst]
    epi_src_vals = epi[edge_src]
    epi_dst_vals = epi[edge_dst]
    vf_src_vals = vf[edge_src]
    vf_dst_vals = vf[edge_dst]

    # Outgoing: node src receives neighbour dst
    np.add.at(neighbor_cos_sum, edge_src, np.cos(phase_dst_vals))
    np.add.at(neighbor_sin_sum, edge_src, np.sin(phase_dst_vals))
    np.add.at(neighbor_epi_sum, edge_src, w_edge * epi_dst_vals)
    np.add.at(neighbor_epi_weight, edge_src, w_edge)
    np.add.at(neighbor_vf_sum, edge_src, vf_dst_vals)
    np.add.at(neighbor_count, edge_src, 1.0)

    if accumulate_both_directions:
        # Reverse edge: node dst receives neighbour src
        np.add.at(neighbor_cos_sum, edge_dst, np.cos(phase_src_vals))
        np.add.at(neighbor_sin_sum, edge_dst, np.sin(phase_src_vals))
        np.add.at(neighbor_epi_sum, edge_dst, w_edge * epi_src_vals)
        np.add.at(neighbor_epi_weight, edge_dst, w_edge)
        np.add.at(neighbor_vf_sum, edge_dst, vf_src_vals)
        np.add.at(neighbor_count, edge_dst, 1.0)

    # Pass 2: Compute gradients from means
    # Avoid division by zero for isolated nodes
    has_neighbors = neighbor_count > 0
    # EPI uses the weighted degree (positive whenever the node has neighbours
    # of positive total weight).
    has_epi_weight = neighbor_epi_weight > 0

    # Compute circular mean phase for nodes with neighbors
    phase_mean = np.zeros(n_nodes, dtype=float)
    phase_mean[has_neighbors] = np.arctan2(
        neighbor_sin_sum[has_neighbors], neighbor_cos_sum[has_neighbors]
    )

    # Compute means: EPI weighted, νf arithmetic
    epi_mean = np.zeros(n_nodes, dtype=float)
    vf_mean = np.zeros(n_nodes, dtype=float)
    epi_mean[has_epi_weight] = (
        neighbor_epi_sum[has_epi_weight] / neighbor_epi_weight[has_epi_weight]
    )
    vf_mean[has_neighbors] = (
        neighbor_vf_sum[has_neighbors] / neighbor_count[has_neighbors]
    )

    # Compute gradients using TNFR canonical formula
    # Phase: g_phase = -angle_diff(θ_node, θ_mean) / π
    # angle_diff(a, b) = (a - b + π) % 2π - π (minimal angular difference)
    phase_diff = (phase_mean - phase + np.pi) % (2 * np.pi) - np.pi
    g_phase = phase_diff / np.pi
    g_phase[~has_neighbors] = 0.0  # Isolated nodes have no gradient

    # EPI/νf: g = mean - node_value
    g_epi = epi_mean - epi
    g_epi[~has_epi_weight] = 0.0

    g_vf = vf_mean - vf
    g_vf[~has_neighbors] = 0.0

    # Topology: Canonical form is (mean_neighbor_degree - node_degree)
    if w_topo != 0.0:
        # For undirected graphs, degree is exactly neighbor_count
        degrees = neighbor_count
        neighbor_deg_sum = np.zeros(n_nodes, dtype=float)

        # We need a second pass over edges to accumulate neighbor degrees
        # because degrees are only known after the first pass.
        deg_src_vals = degrees[edge_src]
        deg_dst_vals = degrees[edge_dst]

        np.add.at(neighbor_deg_sum, edge_src, deg_dst_vals)
        if accumulate_both_directions:
            np.add.at(neighbor_deg_sum, edge_dst, deg_src_vals)

        deg_mean = np.zeros(n_nodes, dtype=float)
        deg_mean[has_neighbors] = (
            neighbor_deg_sum[has_neighbors] / neighbor_count[has_neighbors]
        )

        g_topo = deg_mean - degrees
        g_topo[~has_neighbors] = 0.0

        # Scale by weight
        g_topo *= w_topo
    else:
        g_topo = 0.0

    # Combine gradients
    delta_nfr = w_phase * g_phase + w_epi * g_epi + w_vf * g_vf + g_topo

    return delta_nfr


def apply_vf_scaling(
    *,
    delta_nfr: Any,
    vf: Any,
    np: Any,
) -> None:
    """Apply structural frequency scaling to ΔNFR in-place.

    Applies the fundamental TNFR transformation:
    Delta_NFR_final = nu_f * Delta_NFR_gradient

    This completes the nodal equation: dEPI/dt = nu_f * Delta_NFR(t)

    Parameters
    ----------
    delta_nfr : array-like
        Gradient vector to scale in-place (shape: [N])
    vf : array-like
        Structural frequency for each node (shape: [N])
    np : module
        NumPy module

    Notes
    -----
    Modified in-place to avoid allocating result array.

    Examples
    --------
    >>> import numpy as np
    >>> delta_nfr = np.array([1.0, 2.0, 3.0])
    >>> vf = np.array([0.5, 1.0, 1.5])
    >>> apply_vf_scaling(delta_nfr=delta_nfr, vf=vf, np=np)
    >>> delta_nfr
    array([0.5, 2. , 4.5])
    """
    np.multiply(delta_nfr, vf, out=delta_nfr)


__all__ = [
    "compute_fused_gradients",
    "compute_fused_gradients_symmetric",
    "apply_vf_scaling",
]
