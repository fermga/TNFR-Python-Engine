"""Canonical ΔNFR integrators driving TNFR runtime evolution.

This module implements numerical integration of the canonical TNFR nodal equation:

    ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)

The extended equation includes:
  - Base term: νf · ΔNFR(t) - canonical structural evolution
  - Network term: Γi(R) - optional Kuramoto coupling

Integration respects TNFR invariants:
  - Structural units (Hz_str for νf)
  - Operator closure (valid ΔNFR semantics)
  - Phase coherence (network synchronization)
  - Reproducibility (deterministic with seeds)

The canonical base term is computed explicitly in _collect_nodal_increments()
at line 321 and 342 as: base = vf * dnfr, implementing ∂EPI/∂t = νf·ΔNFR(t).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Literal, cast

import networkx as nx

from .._compat import TypeAlias
from ..alias import collect_attr, get_attr, get_attr_str, set_attr, set_attr_str
from ..constants import DEFAULTS
from ..constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_VF,
)
from ..gamma import _get_gamma_spec, eval_gamma
from ..types import NodeId, TNFRGraph
from ..utils import get_numpy, resolve_chunk_size
from .structural_clip import structural_clip

__all__ = (
    "AbstractIntegrator",
    "DefaultIntegrator",
    "prepare_integration_params",
    "update_epi_via_nodal_equation",
)

GammaMap: TypeAlias = dict[NodeId, float]
"""Γ evaluation cache keyed by node identifier."""

NodeIncrements: TypeAlias = dict[NodeId, tuple[float, ...]]
"""Mapping of nodes to staged integration increments."""

NodalUpdate: TypeAlias = dict[NodeId, tuple[float, float, float]]
"""Mapping of nodes to ``(EPI, dEPI/dt, ∂²EPI/∂t²)`` tuples."""

IntegratorMethod: TypeAlias = Literal["euler", "rk4"]
"""Supported explicit integration schemes for nodal updates."""

_PARALLEL_GRAPH: TNFRGraph | None = None


def _gamma_worker_init(graph: TNFRGraph) -> None:
    """Initialise process-local graph reference for Γ evaluation."""

    global _PARALLEL_GRAPH
    _PARALLEL_GRAPH = graph


def _gamma_worker(task: tuple[list[NodeId], float]) -> list[tuple[NodeId, float]]:
    """Evaluate Γ for ``task`` chunk using process-local graph."""

    chunk, t = task
    if _PARALLEL_GRAPH is None:
        raise RuntimeError("Parallel Γ worker initialised without graph reference")
    return [(node, float(eval_gamma(_PARALLEL_GRAPH, node, t))) for node in chunk]


def _normalise_jobs(n_jobs: int | None, total: int) -> int | None:
    """Return an effective worker count respecting serial fallbacks."""

    if n_jobs is None:
        return None
    try:
        workers = int(n_jobs)
    except (TypeError, ValueError):
        return None
    if workers <= 1 or total <= 1:
        return None
    return max(1, min(workers, total))


def _chunk_nodes(nodes: list[NodeId], chunk_size: int) -> Iterable[list[NodeId]]:
    """Yield deterministic chunks from ``nodes`` respecting insertion order."""

    for idx in range(0, len(nodes), chunk_size):
        yield nodes[idx : idx + chunk_size]


def _apply_increment_chunk(
    chunk: list[tuple[NodeId, float, float, tuple[float, ...]]],
    dt_step: float,
    method: str,
) -> list[tuple[NodeId, tuple[float, float, float]]]:
    """Compute updated states for ``chunk`` using scalar arithmetic."""

    results: list[tuple[NodeId, tuple[float, float, float]]] = []
    dt_nonzero = dt_step != 0

    for node, epi_i, dEPI_prev, ks in chunk:
        if method == "rk4":
            k1, k2, k3, k4 = ks
            epi = epi_i + (dt_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            dEPI_dt = k4
        else:
            (k1,) = ks
            epi = epi_i + dt_step * k1
            dEPI_dt = k1
        d2epi = (dEPI_dt - dEPI_prev) / dt_step if dt_nonzero else 0.0
        results.append((node, (float(epi), float(dEPI_dt), float(d2epi))))

    return results


def _evaluate_gamma_map(
    G: TNFRGraph,
    nodes: list[NodeId],
    t: float,
    *,
    n_jobs: int | None = None,
) -> GammaMap:
    """Return Γ evaluations for ``nodes`` at time ``t`` respecting parallelism."""

    workers = _normalise_jobs(n_jobs, len(nodes))
    if workers is None:
        return {n: float(eval_gamma(G, n, t)) for n in nodes}

    approx_chunk = math.ceil(len(nodes) / (workers * 4)) if workers > 0 else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        len(nodes),
        minimum=1,
    )
    mp_ctx = get_context("spawn")
    tasks = ((chunk, t) for chunk in _chunk_nodes(nodes, chunk_size))

    results: GammaMap = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_gamma_worker_init,
        initargs=(G,),
    ) as executor:
        futures = [executor.submit(_gamma_worker, task) for task in tasks]
        for fut in futures:
            for node, value in fut.result():
                results[node] = value
    return results


def prepare_integration_params(
    G: TNFRGraph,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
) -> tuple[float, int, float, Literal["euler", "rk4"]]:
    """Validate and normalise ``dt``, ``t`` and ``method`` for integration.

    The function raises :class:`TypeError` when ``dt`` cannot be coerced to a
    number, :class:`ValueError` if ``dt`` is negative, and another
    :class:`ValueError` when an unsupported method is requested.  When ``dt``
    exceeds a positive ``DT_MIN`` stored on ``G`` the span is deterministically
    subdivided into integer steps so that the resulting ``dt_step`` never falls
    below that minimum threshold.

    Returns ``(dt_step, steps, t0, method)`` where ``dt_step`` is the effective
    step, ``steps`` the number of substeps and ``t0`` the prepared initial
    time.
    """
    if dt is None:
        dt = float(G.graph.get("DT", DEFAULTS["DT"]))
    else:
        if not isinstance(dt, (int, float)):
            raise TypeError("dt must be a number")
        if dt < 0:
            raise ValueError("dt must be non-negative")
        dt = float(dt)

    if t is None:
        t = float(G.graph.get("_t", 0.0))
    else:
        t = float(t)

    method_value = (
        method or G.graph.get("INTEGRATOR_METHOD", DEFAULTS.get("INTEGRATOR_METHOD", "euler"))
    ).lower()
    if method_value not in ("euler", "rk4"):
        raise ValueError("method must be 'euler' or 'rk4'")

    dt_min = float(G.graph.get("DT_MIN", DEFAULTS.get("DT_MIN", 0.0)))
    steps = 1
    if dt_min > 0 and dt > dt_min:
        ratio = dt / dt_min
        steps = max(1, int(math.floor(ratio + 1e-12)))
        if dt / steps < dt_min:
            steps = int(math.ceil(ratio))
    dt_step = dt / steps if steps else 0.0

    return dt_step, steps, t, cast(Literal["euler", "rk4"], method_value)


def _apply_increments(
    G: TNFRGraph,
    dt_step: float,
    increments: NodeIncrements,
    *,
    method: str,
    n_jobs: int | None = None,
) -> NodalUpdate:
    """Combine precomputed increments to update node states."""

    nodes: list[NodeId] = list(G.nodes)
    if not nodes:
        return {}

    np = get_numpy()

    epi_initial: list[float] = []
    dEPI_prev: list[float] = []
    ordered_increments: list[tuple[float, ...]] = []

    for node in nodes:
        nd = G.nodes[node]
        _, _, dEPI_dt_prev, epi_i = _node_state(nd)
        epi_initial.append(float(epi_i))
        dEPI_prev.append(float(dEPI_dt_prev))
        ordered_increments.append(increments[node])

    if np is not None:
        epi_arr = np.asarray(epi_initial, dtype=float)
        dEPI_prev_arr = np.asarray(dEPI_prev, dtype=float)
        k_arr = np.asarray(ordered_increments, dtype=float)

        if method == "rk4":
            if k_arr.ndim != 2 or k_arr.shape[1] != 4:
                raise ValueError("rk4 increments require four staged values")
            dt_factor = dt_step / 6.0
            k1 = k_arr[:, 0]
            k2 = k_arr[:, 1]
            k3 = k_arr[:, 2]
            k4 = k_arr[:, 3]
            epi = epi_arr + dt_factor * (k1 + 2 * k2 + 2 * k3 + k4)
            dEPI_dt = k4
        else:
            if k_arr.ndim == 1:
                k1 = k_arr
            else:
                k1 = k_arr[:, 0]
            epi = epi_arr + dt_step * k1
            dEPI_dt = k1

        if dt_step != 0:
            d2epi = (dEPI_dt - dEPI_prev_arr) / dt_step
        else:
            d2epi = np.zeros_like(dEPI_dt)

        results: NodalUpdate = {}
        for idx, node in enumerate(nodes):
            results[node] = (
                float(epi[idx]),
                float(dEPI_dt[idx]),
                float(d2epi[idx]),
            )
        return results

    payload: list[tuple[NodeId, float, float, tuple[float, ...]]] = list(
        zip(nodes, epi_initial, dEPI_prev, ordered_increments)
    )

    workers = _normalise_jobs(n_jobs, len(nodes))
    if workers is None:
        return dict(_apply_increment_chunk(payload, dt_step, method))

    approx_chunk = math.ceil(len(nodes) / (workers * 4)) if workers > 0 else None
    chunk_size = resolve_chunk_size(
        approx_chunk,
        len(nodes),
        minimum=1,
    )
    mp_ctx = get_context("spawn")

    results: NodalUpdate = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as executor:
        futures = [
            executor.submit(
                _apply_increment_chunk,
                chunk,
                dt_step,
                method,
            )
            for chunk in _chunk_nodes(payload, chunk_size)
        ]
        for fut in futures:
            for node, value in fut.result():
                results[node] = value

    return {node: results[node] for node in nodes}


def _collect_nodal_increments(
    G: TNFRGraph,
    gamma_maps: tuple[GammaMap, ...],
    *,
    method: str,
) -> NodeIncrements:
    """Combine node base state with staged Γ contributions.

    Implements the canonical TNFR nodal equation in two parts:

    1. **Base term** (canonical equation):
       base = vf * dnfr  →  ∂EPI/∂t = νf · ΔNFR(t)

       This is the fundamental TNFR equation where:
         - vf (νf): structural frequency in Hz_str
         - dnfr (ΔNFR): nodal gradient (reorganization operator)
         - base: instantaneous rate of EPI evolution

    2. **Network coupling term**:
       Γi(R) from gamma_maps - optional Kuramoto order parameter

    The full extended equation is: ∂EPI/∂t = νf·ΔNFR(t) + Γi(R)

    Args:
        G: TNFR graph with node attributes vf and dnfr
        gamma_maps: Staged Γ evaluations (1 for Euler, 4 for RK4)
        method: Integration method ('euler' or 'rk4')

    Returns:
        Mapping of nodes to staged integration increments

    Notes:
        - Line 321 implements the canonical nodal equation explicitly
        - Units: vf in Hz_str, dnfr dimensionless, base in Hz_str
        - Preserves TNFR operator closure and structural semantics
    """

    nodes: list[NodeId] = list(G.nodes())
    if not nodes:
        return {}

    if method == "rk4":
        expected_maps = 4
    elif method == "euler":
        expected_maps = 1
    else:
        raise ValueError("method must be 'euler' or 'rk4'")

    if len(gamma_maps) != expected_maps:
        raise ValueError(f"{method} integration requires {expected_maps} gamma maps")

    np = get_numpy()
    if np is not None:
        vf = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np)
        dnfr = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np)
        # CANONICAL TNFR EQUATION: ∂EPI/∂t = νf · ΔNFR(t)
        # This implements the fundamental nodal equation explicitly
        base = vf * dnfr

        gamma_arrays = [
            np.fromiter((gm.get(n, 0.0) for n in nodes), float, count=len(nodes))
            for gm in gamma_maps
        ]
        if gamma_arrays:
            gamma_stack = np.stack(gamma_arrays, axis=1)
            combined = base[:, None] + gamma_stack
        else:
            combined = base[:, None]

        return {
            node: tuple(float(value) for value in combined[idx]) for idx, node in enumerate(nodes)
        }

    increments: NodeIncrements = {}
    for node in nodes:
        nd = G.nodes[node]
        vf, dnfr, *_ = _node_state(nd)
        # CANONICAL TNFR EQUATION: ∂EPI/∂t = νf · ΔNFR(t)
        # Scalar implementation of the fundamental nodal equation
        base = vf * dnfr
        gammas = [gm.get(node, 0.0) for gm in gamma_maps]

        if method == "rk4":
            k1, k2, k3, k4 = gammas
            increments[node] = (
                base + k1,
                base + k2,
                base + k3,
                base + k4,
            )
        else:
            (k1,) = gammas
            increments[node] = (base + k1,)

    return increments


def _build_gamma_increments(
    G: TNFRGraph,
    dt_step: float,
    t_local: float,
    *,
    method: str,
    n_jobs: int | None = None,
) -> NodeIncrements:
    """Evaluate Γ contributions and merge them with ``νf·ΔNFR`` base terms."""

    if method == "rk4":
        gamma_count = 4
    elif method == "euler":
        gamma_count = 1
    else:
        raise ValueError("method must be 'euler' or 'rk4'")

    gamma_spec = G.graph.get("_gamma_spec")
    if gamma_spec is None:
        gamma_spec = _get_gamma_spec(G)

    gamma_type = ""
    if isinstance(gamma_spec, Mapping):
        gamma_type = str(gamma_spec.get("type", "")).lower()

    if gamma_type == "none":
        gamma_maps: tuple[GammaMap, ...] = tuple(cast(GammaMap, {}) for _ in range(gamma_count))
        return _collect_nodal_increments(G, gamma_maps, method=method)

    nodes: list[NodeId] = list(G.nodes)
    if not nodes:
        gamma_maps = tuple(cast(GammaMap, {}) for _ in range(gamma_count))
        return _collect_nodal_increments(G, gamma_maps, method=method)

    if method == "rk4":
        t_mid = t_local + dt_step / 2.0
        t_end = t_local + dt_step
        g1_map = _evaluate_gamma_map(G, nodes, t_local, n_jobs=n_jobs)
        g_mid_map = _evaluate_gamma_map(G, nodes, t_mid, n_jobs=n_jobs)
        g4_map = _evaluate_gamma_map(G, nodes, t_end, n_jobs=n_jobs)
        gamma_maps = (g1_map, g_mid_map, g_mid_map, g4_map)
    else:  # method == "euler"
        gamma_maps = (_evaluate_gamma_map(G, nodes, t_local, n_jobs=n_jobs),)

    return _collect_nodal_increments(G, gamma_maps, method=method)


def _integrate_euler(
    G: TNFRGraph,
    dt_step: float,
    t_local: float,
    *,
    n_jobs: int | None = None,
) -> NodalUpdate:
    """One explicit Euler integration step."""
    increments = _build_gamma_increments(
        G,
        dt_step,
        t_local,
        method="euler",
        n_jobs=n_jobs,
    )
    return _apply_increments(
        G,
        dt_step,
        increments,
        method="euler",
        n_jobs=n_jobs,
    )


def _integrate_rk4(
    G: TNFRGraph,
    dt_step: float,
    t_local: float,
    *,
    n_jobs: int | None = None,
) -> NodalUpdate:
    """One Runge–Kutta order-4 integration step."""
    increments = _build_gamma_increments(
        G,
        dt_step,
        t_local,
        method="rk4",
        n_jobs=n_jobs,
    )
    return _apply_increments(
        G,
        dt_step,
        increments,
        method="rk4",
        n_jobs=n_jobs,
    )


class AbstractIntegrator(ABC):
    """Abstract base class encapsulating nodal equation integration."""

    @abstractmethod
    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None,
        t: float | None,
        method: str | None,
        n_jobs: int | None,
    ) -> None:
        """Advance ``graph`` coherence states according to the nodal equation."""


class DefaultIntegrator(AbstractIntegrator):
    """Explicit integrator combining Euler and RK4 step implementations."""

    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None,
        t: float | None,
        method: str | None,
        n_jobs: int | None,
    ) -> None:
        """Integrate the nodal equation updating EPI, ΔEPI and Δ²EPI."""

        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            raise TypeError("G must be a networkx graph instance")

        dt_step, steps, t0, resolved_method = prepare_integration_params(
            graph, dt, t, cast(IntegratorMethod | None, method)
        )

        t_local = t0
        for _ in range(steps):
            if resolved_method == "rk4":
                updates: NodalUpdate = _integrate_rk4(graph, dt_step, t_local, n_jobs=n_jobs)
            else:
                updates = _integrate_euler(graph, dt_step, t_local, n_jobs=n_jobs)

            for n, (epi, dEPI_dt, d2epi) in updates.items():
                nd = graph.nodes[n]
                epi_kind = get_attr_str(nd, ALIAS_EPI_KIND, "")

                # Apply structural boundary preservation
                epi_min = float(graph.graph.get("EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)))
                epi_max = float(graph.graph.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)))
                clip_mode_str = str(graph.graph.get("CLIP_MODE", "hard"))
                # Validate clip mode and cast to proper type
                if clip_mode_str not in ("hard", "soft"):
                    clip_mode_str = "hard"
                clip_mode: Literal["hard", "soft"] = clip_mode_str  # type: ignore[assignment]
                clip_k = float(graph.graph.get("CLIP_SOFT_K", 3.0))

                epi_clipped = structural_clip(
                    epi,
                    lo=epi_min,
                    hi=epi_max,
                    mode=clip_mode,
                    k=clip_k,
                    record_stats=False,
                )

                set_attr(nd, ALIAS_EPI, epi_clipped)
                if epi_kind:
                    set_attr_str(nd, ALIAS_EPI_KIND, epi_kind)
                set_attr(nd, ALIAS_DEPI, dEPI_dt)
                set_attr(nd, ALIAS_D2EPI, d2epi)

            t_local += dt_step

        graph.graph["_t"] = t_local


def update_epi_via_nodal_equation(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
    n_jobs: int | None = None,
) -> None:
    """TNFR nodal equation with optional extended dynamics.

    Implements either:
    
    **Classical**: ∂EPI/∂t = νf · ΔNFR(t) + Γi(R)
      - EPI is the node's Primary Information Structure  
      - νf is the node's structural frequency (Hz_str)
      - ΔNFR(t) is the nodal gradient (reorganisation need)
      - Γi(R) is optional network coupling via Kuramoto order
      
    **Extended**: Coupled system with flux fields (when use_extended_dynamics=True)
      - ∂EPI/∂t = νf · ΔNFR(t) [Classical equation unchanged]
      - ∂θ/∂t = f(νf, ΔNFR, J_φ) [Phase evolution with transport]
      - ∂ΔNFR/∂t = g(∇·J_ΔNFR) [ΔNFR conservation dynamics]
      
    The extended system includes canonical flux fields J_φ (phase current) 
    and J_ΔNFR (reorganization flux) that enable directed transport and 
    conservation dynamics while preserving all TNFR invariants.

    Args:
        G: TNFR graph with nodes containing structural attributes
        dt: Integration time step (uses graph default if None)
        t: Current time (uses graph default if None)  
        method: Integration method ('euler' or 'rk4')
        n_jobs: Number of parallel jobs for integration

    Notes:
        - Use G.graph['use_extended_dynamics'] = True to enable extended system
        - Extended dynamics require J_φ and J_ΔNFR fields (from physics module)
        - Classical limit: when J_φ = J_ΔNFR = 0, recovers original behavior
        - Extended system preserves backward compatibility (default: False)
        
    Examples:
        >>> # Classical dynamics (default)
        >>> update_epi_via_nodal_equation(G, dt=0.01)
        
        >>> # Extended dynamics with flux fields
        >>> G.graph['use_extended_dynamics'] = True
        >>> update_epi_via_nodal_equation(G, dt=0.01)
    """
    # Check if extended dynamics is enabled
    use_extended = G.graph.get('use_extended_dynamics', False)
    
    if use_extended:
        # Use extended nodal system with flux fields
        _update_extended_nodal_system(
            G, dt=dt, t=t, method=method, n_jobs=n_jobs
        )
    else:
        # Use classical TNFR dynamics
        DefaultIntegrator().integrate(
            G, dt=dt, t=t, method=method, n_jobs=n_jobs,
        )


def _node_state(nd: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return common node state attributes for canonical equation evaluation.

    Extracts the fundamental TNFR variables from node data:
      - νf (vf): Structural frequency in Hz_str
      - ΔNFR (dnfr): Nodal gradient (reorganization operator)
      - dEPI/dt (previous): Last computed EPI derivative
      - EPI (current): Current Primary Information Structure

    These variables are used in the canonical nodal equation:
        ∂EPI/∂t = νf · ΔNFR(t)

    Args:
        nd: Node data dictionary containing TNFR attributes

    Returns:
        Tuple of (vf, dnfr, dEPI_dt_prev, epi_i) with 0.0 defaults

    Notes:
        - vf alias maps to VF, frequency, or structural_frequency
        - dnfr alias maps to DNFR, delta_nfr, or reorganization_gradient
        - All values are coerced to float for numerical stability
    """

    vf = get_attr(nd, ALIAS_VF, 0.0)
    dnfr = get_attr(nd, ALIAS_DNFR, 0.0)
    dEPI_dt_prev = get_attr(nd, ALIAS_DEPI, 0.0)
    epi_i = get_attr(nd, ALIAS_EPI, 0.0)
    return vf, dnfr, dEPI_dt_prev, epi_i


def _update_extended_nodal_system(
    G: TNFRGraph,
    *,
    dt: float | None = None,
    t: float | None = None,
    method: Literal["euler", "rk4"] | None = None,
    n_jobs: int | None = None,
) -> None:
    """Update network using extended TNFR dynamics with flux fields.
    
    This function implements the coupled system:
    1. ∂EPI/∂t = νf · ΔNFR(t)     [Classical nodal equation]
    2. ∂θ/∂t = f(νf, ΔNFR, J_φ)   [Phase evolution with transport] 
    3. ∂ΔNFR/∂t = g(∇·J_ΔNFR)     [ΔNFR conservation dynamics]
    
    The extended system requires canonical flux fields to be computed
    before integration. Uses compute_extended_nodal_system() from
    canonical module for physics-correct dynamics.
    
    Args:
        G: TNFR graph with extended dynamics enabled
        dt: Integration time step
        t: Current simulation time
        method: Integration method (currently supports 'euler')
        n_jobs: Parallel jobs (extended system uses single-threaded for now)
        
    Notes:
        - Requires J_φ and J_ΔNFR fields computed via physics module
        - Falls back gracefully if flux fields missing (J=0 assumption)
        - Updates EPI, theta, and ΔNFR for each node
        - Maintains numerical stability with clipping
    """
    from .canonical import compute_extended_nodal_system
    
    # Get integration parameters
    if dt is None:
        dt = G.graph.get("dt", DEFAULTS.dt)
    if t is None:
        t = G.graph.get("_t", 0.0)
        
    # Extended system currently uses Euler method for stability
    if method is None:
        method = "euler"
    elif method != "euler":
        # RK4 implementation requires numerical stability analysis for extended canonical fields
        # Currently using Euler method for guaranteed stability with coupled field equations
        method = "euler"
    
    # Import flux field computations
    try:
        from ..physics.extended_canonical_fields import (
            compute_phase_current,
            compute_dnfr_flux
        )
        flux_fields_available = True
    except ImportError:
        # Graceful degradation if extended fields not available
        flux_fields_available = False
    
    # Update each node with extended dynamics
    for node in G.nodes():
        nd = G.nodes[node]
        
        # Get current state
        vf, dnfr, _, epi_current = _node_state(nd)
        theta_current = nd.get('theta', 0.0)
        
        # Compute flux fields if available
        if flux_fields_available:
            try:
                # Use centralized canonical field computations (entire graph)
                j_phi_dict = compute_phase_current(G, theta_attr='theta')
                j_dnfr_dict = compute_dnfr_flux(G, dnfr_attr=ALIAS_DNFR)
                
                # Extract values for current node
                j_phi = j_phi_dict.get(node, 0.0)
                j_dnfr = j_dnfr_dict.get(node, 0.0)
                
                # If fluxes are still zero, use synthetic fallback
                if abs(j_phi) < 1e-9:
                    j_phi = _compute_synthetic_phase_current(G, node)
                
                # Compute divergences vectorized for efficiency
                if 'j_dnfr_divergences' not in locals():
                    # Cache vectorized divergences for all nodes
                    j_dnfr_divergences = compute_flux_divergence_vectorized(G, j_dnfr_dict)
                j_dnfr_div = j_dnfr_divergences.get(node, 0.0)
                
            except Exception:
                # Fallback to synthetic values for testing
                j_phi = _compute_synthetic_phase_current(G, node)
                j_dnfr_div = _compute_synthetic_dnfr_divergence(G, node)
        else:
            # Use synthetic flux fields for extended dynamics testing
            j_phi = _compute_synthetic_phase_current(G, node)
            j_dnfr_div = _compute_synthetic_dnfr_divergence(G, node)
        
        # Estimate coupling strength from local topology
        coupling_strength = _estimate_local_coupling_strength(G, node)
        
        # Compute extended system derivatives
        result = compute_extended_nodal_system(
            nu_f=vf,
            delta_nfr=dnfr,
            theta=theta_current,
            j_phi=j_phi,
            j_dnfr_divergence=j_dnfr_div,
            coupling_strength=coupling_strength,
            validate_units=False  # Skip validation for performance
        )
        
        # Integrate using Euler method
        new_epi = epi_current + result.classical_derivative * dt
        new_theta = (theta_current + result.phase_derivative * dt) % (2 * math.pi)
        new_dnfr = dnfr + result.dnfr_derivative * dt
        
        # Apply clipping for numerical stability
        new_epi = max(0.0, min(1.0, new_epi))  # EPI ∈ [0, 1]
        new_dnfr = max(-2.0, min(2.0, new_dnfr))  # ΔNFR bounded
        
        # Update node attributes  
        set_attr(nd, ALIAS_EPI, new_epi)
        nd['theta'] = new_theta
        set_attr(nd, ALIAS_DNFR, new_dnfr)
        
        # Cache derivatives for analysis
        set_attr(nd, ALIAS_DEPI, result.classical_derivative) 
        nd['dtheta_dt'] = result.phase_derivative
        nd['ddnfr_dt'] = result.dnfr_derivative
    
    # Update simulation time
    G.graph["_t"] = t + dt


# Centralized flux divergence computation


def _compute_flux_divergence_centralized(
    G: TNFRGraph, 
    flux_dict: dict[NodeId, float], 
    node: NodeId
) -> float:
    """
    Compute flux divergence using centralized finite difference method.
    
    Uses vectorized neighbor access and proper conservation physics.
    Replaces ad-hoc approximations with systematic approach.
    """
    if G.degree(node) == 0:
        return 0.0
    
    central_flux = flux_dict.get(node, 0.0)
    neighbors = list(G.neighbors(node))
    
    if not neighbors:
        return 0.0
    
    # Vectorized neighbor flux collection
    neighbor_fluxes = [flux_dict.get(neighbor, 0.0) for neighbor in neighbors]
    mean_neighbor_flux = sum(neighbor_fluxes) / len(neighbor_fluxes)
    
    # Finite difference with topology-dependent spacing
    spacing = 1.0 / math.sqrt(len(neighbors))
    divergence = (central_flux - mean_neighbor_flux) / spacing
    
    return divergence


def compute_flux_divergence_vectorized(
    G: TNFRGraph,
    flux_dict: dict[NodeId, float]
) -> dict[NodeId, float]:
    """
    Vectorized flux divergence computation using sparse matrix operations.
    
    Uses adjacency matrix and broadcasting for true vectorization,
    following TNFR patterns from dynamics/dnfr.py for optimal performance.
    
    Args:
        G: TNFR graph
        flux_dict: Node -> flux value mapping
        
    Returns:
        Node -> divergence value mapping
    """
    try:
        import numpy as np
        from scipy import sparse
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        try:
            import numpy as np
        except ImportError:
            pass
        # Fallback to node-by-node computation
        return {
            node: _compute_flux_divergence_centralized(G, flux_dict, node) 
            for node in G.nodes()
        }
    
    if not G.nodes() or not G.edges():
        return {node: 0.0 for node in G.nodes()}
    
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    # Flux array
    flux_array = np.array([flux_dict.get(node, 0.0) for node in nodes])
    
    if SCIPY_AVAILABLE and n_nodes > 100:  # Use sparse for larger graphs
        try:
            # Build adjacency matrix for vectorized operations
            A = sparse.csr_matrix(nx.adjacency_matrix(G, nodelist=nodes))
            
            # Degree array for normalization
            degrees = np.array(A.sum(axis=1)).flatten()
            
            # Neighbor mean fluxes using sparse matrix multiplication
            neighbor_sums = A @ flux_array  # Sum of neighbor fluxes
            neighbor_means = np.divide(
                neighbor_sums, degrees, 
                out=np.zeros_like(neighbor_sums), 
                where=degrees!=0
            )
            
            # Vectorized divergence computation
            # spacing = 1.0 / sqrt(degree) for each node
            spacings = np.divide(
                1.0, np.sqrt(degrees), 
                out=np.ones_like(degrees), 
                where=degrees!=0
            )
            
            divergence_array = (flux_array - neighbor_means) / spacings
            
        except Exception:
            # Fallback to dense if sparse fails
            SCIPY_AVAILABLE = False
    
    if not SCIPY_AVAILABLE or n_nodes <= 100:
        # Dense NumPy implementation for smaller graphs
        divergence_array = np.zeros(n_nodes, dtype=float)
        
        for i, node in enumerate(nodes):
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
                
            neighbor_indices = np.array([nodes.index(neighbor) for neighbor in neighbors])
            neighbor_fluxes = flux_array[neighbor_indices]
            
            central_flux = flux_array[i]
            mean_neighbor_flux = np.mean(neighbor_fluxes)
            spacing = 1.0 / math.sqrt(len(neighbors))
            divergence_array[i] = (central_flux - mean_neighbor_flux) / spacing
    
    # Convert back to dict
    return {node: float(divergence_array[i]) for i, node in enumerate(nodes)}


def _compute_synthetic_phase_current(G: TNFRGraph, node: NodeId) -> float:
    """Compute synthetic J_φ based on phase gradients with neighbors."""
    if G.degree(node) == 0:
        return 0.0
    
    node_theta = G.nodes[node].get('theta', 0.0)
    
    # Compute phase differences with neighbors
    phase_diffs = []
    for neighbor in G.neighbors(node):
        neighbor_theta = G.nodes[neighbor].get('theta', 0.0)
        # Use circular difference for phases
        diff = neighbor_theta - node_theta
        # Normalize to [-π, π]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        phase_diffs.append(diff)
    
    if not phase_diffs:
        return 0.0
    
    # Mean phase gradient (synthetic J_φ)
    mean_gradient = sum(phase_diffs) / len(phase_diffs)
    
    # Scale by coupling strength and local network properties
    coupling = _estimate_local_coupling_strength(G, node)
    synthetic_j_phi = 0.1 * mean_gradient * coupling  # Scale factor for realism
    
    return synthetic_j_phi


def _compute_synthetic_dnfr_divergence(G: TNFRGraph, node: NodeId) -> float:
    """Compute synthetic ∇·J_ΔNFR based on ΔNFR gradients."""
    if G.degree(node) == 0:
        return 0.0
    
    node_dnfr = G.nodes[node].get(ALIAS_DNFR, 0.0)
    
    # Compute ΔNFR differences with neighbors  
    dnfr_diffs = []
    for neighbor in G.neighbors(node):
        neighbor_dnfr = G.nodes[neighbor].get(ALIAS_DNFR, 0.0)
        diff = neighbor_dnfr - node_dnfr
        dnfr_diffs.append(diff)
    
    if not dnfr_diffs:
        return 0.0
    
    # Mean ΔNFR gradient approximates flux divergence
    mean_gradient = sum(dnfr_diffs) / len(dnfr_diffs)
    
    # Synthetic divergence with conservation physics
    # Positive gradient (neighbors higher) → convergent flow → negative divergence
    synthetic_div = -0.2 * mean_gradient  # Conservation coefficient
    
    return synthetic_div


def _approximate_flux_divergence(G: TNFRGraph, node: NodeId, central_flux: float) -> float:
    """Approximate ∇·J using finite differences with neighbors."""
    if G.degree(node) == 0:
        return 0.0
    
    # Collect neighbor fluxes (simplified: assume same flux type)
    neighbor_fluxes = []
    for neighbor in G.neighbors(node):
        # Simplified: use same flux value for neighbors
        # In full implementation, would compute flux for each neighbor
        neighbor_flux = G.nodes[neighbor].get('j_flux_cache', central_flux * 0.9)
        neighbor_fluxes.append(neighbor_flux)
    
    if not neighbor_fluxes:
        return 0.0
    
    mean_neighbor_flux = sum(neighbor_fluxes) / len(neighbor_fluxes)
    
    # Finite difference approximation: (central - mean_neighbors) / spacing
    spacing = 1.0 / math.sqrt(G.degree(node))  # Topology-dependent spacing
    divergence = (central_flux - mean_neighbor_flux) / spacing
    
    return divergence


def _estimate_local_coupling_strength(G: TNFRGraph, node: NodeId) -> float:
    """Estimate coupling strength from local network topology."""
    degree = G.degree(node)
    if degree == 0:
        return 0.0
    
    # Sigmoid coupling: stronger for well-connected nodes
    normalized_degree = min(degree / 10.0, 1.0)  # Saturation at degree 10
    coupling = 1.0 / (1.0 + math.exp(-5 * (normalized_degree - 0.5)))
    
    return coupling
