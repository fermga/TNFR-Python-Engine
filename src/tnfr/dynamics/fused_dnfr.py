# -*- coding: utf-8 -*-
"""Fused ΔNFR computation kernels for optimized backend.

This module provides optimized implementations of ΔNFR computation that fuse
multiple operations to reduce memory traffic and improve cache locality.

The fused kernels implement the same declared pressure channels through:

1. Combined gradient computation (phase + EPI + topology in single pass)
2. Reduced intermediate array allocations
3. Better cache locality through sequential memory access
4. Optional Numba JIT compilation for critical loops

All implementations preserve TNFR structural invariants:
- ΔNFR = w_phase·g_phase + w_epi·g_epi + w_vf·g_vf + w_topo·g_topo
- Isolated nodes receive ΔNFR = 0
- Reproducible results with fixed inputs, contribution order and backend

Transcendental materialization and accumulation can differ by backend;
near a vanishing resultant, small component errors can change its direction.
Outside the certified two-neighbor branch, an exactly joint-zero accumulated
pair uses the center phase as an explicit computational extension, not a
derived phase direction or physical law.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..mathematics._neighbor_differences import (
    _require_finite_pressure,
    edge_mean_differences,
)
from ..mathematics._phase_midpoint import certified_two_neighbor_phase
from ..mathematics.unified_numerical import compute_phase_difference, np
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
    g_epi,
    g_vf,
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
    so the result has the same support convention as the graph dispatcher.
    """
    # Allocations (Numba handles these efficiently in nopython mode)
    cos_sum = np.zeros(n_nodes, dtype=np.float64)
    sin_sum = np.zeros(n_nodes, dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.float64)
    phase_cos = np.zeros(n_nodes, dtype=np.float64)
    phase_sin = np.zeros(n_nodes, dtype=np.float64)
    if w_phase != 0.0:
        for i in range(n_nodes):
            phase_cos[i] = math.cos(phase[i])
            phase_sin[i] = math.sin(phase[i])

    n_edges = edge_src.shape[0]

    # Pass 1: Accumulate neighbour statistics (outgoing orientation)
    for i in range(n_edges):
        u = edge_src[i]
        v = edge_dst[i]
        # u -> v: node u receives neighbour v
        if w_phase != 0.0:
            cos_sum[u] += phase_cos[v]
            sin_sum[u] += phase_sin[v]
        count[u] += 1.0

        if symmetric:
            # v -> u: node v receives neighbour u
            if w_phase != 0.0:
                cos_sum[v] += phase_cos[u]
                sin_sum[v] += phase_sin[u]
            count[v] += 1.0

    # Pass 2: Compute gradients
    for i in range(n_nodes):
        if count[i] > 0:
            g_phase = 0.0
            if w_phase != 0.0 and (sin_sum[i] != 0.0 or cos_sum[i] != 0.0):
                theta_mean = math.atan2(sin_sum[i], cos_sum[i])
                center = phase[i]
                if abs(center) > math.pi:
                    center = math.atan2(phase_sin[i], phase_cos[i])
                raw = theta_mean - center
                # Numba-compatible form of the shared signed atan2 wrap.
                # Adding pi first loses small gaps and changes antipodal ties.
                diff = math.atan2(math.sin(raw), math.cos(raw))
                g_phase = diff / math.pi

            # Linear channels arrive from the shared stable edge reducer.
            delta_nfr[i] = w_phase * g_phase + g_epi[i] + g_vf[i]

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
            fastmath=False,
            cache=True,
        )
        logger.debug("Numba JIT compilation successful for fused gradients")
    except Exception as e:
        logger.warning(f"Numba JIT compilation failed: {e}")
        _compute_canonical_gradients_jit = _compute_canonical_gradients_jit_kernel
else:
    _compute_canonical_gradients_jit = _compute_canonical_gradients_jit_kernel


def _two_neighbor_phase_gradients(phase, source, target):
    """Read eligible two-contribution rows without changing support semantics."""
    counts = np.bincount(source, minlength=len(phase))
    selected = np.flatnonzero(counts[source] == 2)
    rows, gradients = [], []
    if len(selected):
        selected_source = source[selected]
        if np.any(selected_source[1:] < selected_source[:-1]):
            selected = selected[np.argsort(selected_source, kind="stable")]
        for first, second in selected.reshape((-1, 2)):
            row = int(source[first])
            result = certified_two_neighbor_phase(
                float(phase[row]),
                float(phase[target[first]]),
                float(phase[target[second]]),
            )
            if result is not None:
                rows.append(row)
                gradients.append(result.delta / math.pi)
    return np.asarray(rows, dtype=np.intp), np.asarray(gradients, dtype=float), counts


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
        If False, each edge (u, v) contributes neighbor v to row u (src).
        Set to False if edge_src/edge_dst already contain both (u,v) and (v,u).
        These arrays describe contributions, not a deduplicated graph. True
        also doubles self-loop contributions; the graph-facing dispatcher
        supplies unique outgoing neighbors and always uses False.
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
            raise ValueError("edge_weight length does not match edge_src/edge_dst")

    if accumulate_both_directions:
        linear_src = np.concatenate((edge_src, edge_dst))
        linear_dst = np.concatenate((edge_dst, edge_src))
        linear_weight = np.concatenate((w_edge, w_edge))
    else:
        linear_src, linear_dst, linear_weight = edge_src, edge_dst, w_edge
    # Disabled channels never evaluate nodal differences, including at extremes.
    g_epi = (
        edge_mean_differences(
            epi, linear_src, linear_dst, linear_weight, coefficient=w_epi
        )
        if w_epi != 0.0
        else np.zeros(n_nodes, dtype=float)
    )
    g_vf = (
        edge_mean_differences(vf, linear_src, linear_dst, coefficient=w_vf)
        if w_vf != 0.0
        else np.zeros(n_nodes, dtype=float)
    )
    phase_rows, phase_values, contribution_counts = (
        _two_neighbor_phase_gradients(phase, linear_src, linear_dst)
        if w_phase != 0.0
        else (np.empty(0, dtype=np.intp), np.empty(0, dtype=float), None)
    )

    # JIT Path
    if use_jit and _NUMBA_AVAILABLE and n_edges > 100:
        _compute_canonical_gradients_jit(
            edge_src,
            edge_dst,
            w_edge,
            phase,
            g_epi,
            g_vf,
            w_phase,
            w_epi,
            w_vf,
            w_topo,
            n_nodes,
            accumulate_both_directions,
            delta_nfr,
        )
        if len(phase_rows):
            # Reassemble eligible rows without subtracting an already rounded
            # phasor term or relying on the kernel's evaluation order.
            topology = 0.0
            if w_topo != 0.0:
                degree_sum = np.bincount(
                    linear_src,
                    weights=contribution_counts[linear_dst],
                    minlength=n_nodes,
                )
                topology = w_topo * (
                    degree_sum[phase_rows] / contribution_counts[phase_rows]
                    - contribution_counts[phase_rows]
                )
            delta_nfr[phase_rows] = (
                w_phase * phase_values + g_epi[phase_rows] + g_vf[phase_rows] + topology
            )
        _require_finite_pressure(delta_nfr)
        return delta_nfr

    # Pass 1: Accumulate neighbour statistics for computing means, under the
    # canonical outgoing orientation L_out = I - D^-1 W (node i receives the
    # nodes it points to).  For edge i->j, node i (src) accumulates neighbour
    # j (dst).  Undirected graphs carry both (i,j) and (j,i), so the result
    # is symmetric and identical to the legacy computation.
    #   phase: cos/sin sums for the circular mean
    # Linear EPI/frequency gradients were reduced above without common offsets.
    neighbor_cos_sum = np.zeros(n_nodes, dtype=float)
    neighbor_sin_sum = np.zeros(n_nodes, dtype=float)
    neighbor_count = np.zeros(n_nodes, dtype=float)
    if w_phase != 0.0:
        phase_cos, phase_sin = np.cos(phase), np.sin(phase)

    # Outgoing: node src receives neighbour dst
    if w_phase != 0.0:
        np.add.at(neighbor_cos_sum, edge_src, phase_cos[edge_dst])
        np.add.at(neighbor_sin_sum, edge_src, phase_sin[edge_dst])
    np.add.at(neighbor_count, edge_src, 1.0)

    if accumulate_both_directions:
        # Reverse edge: node dst receives neighbour src
        if w_phase != 0.0:
            np.add.at(neighbor_cos_sum, edge_dst, phase_cos[edge_src])
            np.add.at(neighbor_sin_sum, edge_dst, phase_sin[edge_src])
        np.add.at(neighbor_count, edge_dst, 1.0)

    # Pass 2: Compute gradients from means
    # Avoid division by zero for isolated nodes
    has_neighbors = neighbor_count > 0

    g_phase = np.zeros(n_nodes, dtype=float)
    if w_phase != 0.0:
        # Zero is a property of these accumulated floating components. It is
        # not a certificate that the exact phasor resultant is zero.
        defined = has_neighbors & (
            (neighbor_sin_sum != 0.0) | (neighbor_cos_sum != 0.0)
        )
        phase_mean = np.arctan2(neighbor_sin_sum[defined], neighbor_cos_sum[defined])
        # Compare with the same represented center phasor. Subtracting a huge
        # raw angle loses the O(1) target and creates pressure even at uniform
        # phase. Keep already centered values to retain tiny displacements.
        centers = np.array(phase[defined], dtype=float, copy=True)
        outside = np.abs(centers) > math.pi
        centers[outside] = np.arctan2(
            phase_sin[defined][outside], phase_cos[defined][outside]
        )
        g_phase[defined] = compute_phase_difference(phase_mean, centers) / math.pi
        # Isolates and joint-zero rows retain zero phase pressure; admitted
        # two-neighbor rows retain their certified midpoint realization.
        g_phase[phase_rows] = phase_values

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
    with np.errstate(over="ignore", invalid="ignore"):
        delta_nfr = w_phase * g_phase + g_epi + g_vf + g_topo
    _require_finite_pressure(delta_nfr)

    return delta_nfr


def apply_vf_scaling(
    *,
    delta_nfr: Any,
    vf: Any,
    np: Any,
) -> None:
    """Replace a pressure buffer by the nodal rate in-place.

    The resulting buffer contains dEPI/dt = nu_f * Delta_NFR, not a second
    pressure. This legacy utility has no pressure-pipeline caller; the shared
    nodal integrator applies capacity once when advancing EPI.

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
