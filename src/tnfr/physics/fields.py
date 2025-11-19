"""Structural field computations for TNFR physics.

REORGANIZED (Nov 14, 2025): Canonical field implementations moved to modular
submódules (canonical.py, extended.py) to reduce coupling and improve
maintainability. This module now acts as the public API, re-exporting all
canonical fields and containing only research-phase utilities.

This module computes emergent structural "fields" from TNFR graph state,
grounding a pathway from the nodal equation to macroscopic interaction
patterns.

CANONICAL FIELDS (Read-Only Telemetry)
---------------------------------------
All four structural fields have CANONICAL status as of November 12, 2025:

- Φ_s (Structural Potential): Global field from ΔNFR distribution
- |∇φ| (Phase Gradient): Local phase desynchronization metric
- K_φ (Phase Curvature): Geometric phase confinement indicator
- ξ_C (Coherence Length): Spatial correlation scale

EXTENDED CANONICAL FIELDS (Promoted Nov 12, 2025)
-------------------------------------------------
Two flux fields capturing directed transport:

- J_φ (Phase Current): Geometric phase-driven transport
- J_ΔNFR (ΔNFR Flux): Potential-driven reorganization transport

RESEARCH-PHASE UTILITIES
------------------------
Additional functions for analysis and advanced validation:

- compute_k_phi_multiscale_variance(): Coarse-grained curvature variance
- fit_k_phi_asymptotic_alpha(): Power-law fitting for multiscale K_φ
- k_phi_multiscale_safety(): Safety check for multiscale curvature
- path_integrated_gradient(): Path-integrated phase gradient
- compute_phase_winding(): Topological charge (winding number)
- fit_correlation_length_exponent(): Critical exponent extraction

Physics Foundation
------------------
From the nodal equation:
    ∂EPI/∂t = νf · ΔNFR(t)

ΔNFR represents structural pressure driving reorganization. Aggregating
ΔNFR across the network with distance weighting creates the structural
potential field Φ_s, analogous to gravitational potential from mass
distribution.

References
----------
- UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
- docs/TNFR_FORCES_EMERGENCE.md § 14-15: Complete validation
- docs/XI_C_CANONICAL_PROMOTION.md: ξ_C experimental validation
- AGENTS.md § Structural Fields: Canonical tetrad documentation
- TNFR.pdf § 2.1: Nodal equation foundation
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

# ============================================================================
# PUBLIC API: Import all canonical and extended canonical fields
# ============================================================================

# Canonical tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

# Extended canonical fields (J_φ, J_ΔNFR) - Promoted Nov 12, 2025
from .extended import (
    compute_phase_current,
    compute_dnfr_flux,
    compute_extended_canonical_suite,
)

# Import TNFR cache system for research functions
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

__all__ = [
    # Canonical Tetrad
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
    # Extended Canonical Fields (NEWLY PROMOTED Nov 12, 2025)
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    # Research-phase utilities
    "path_integrated_gradient",
    "compute_phase_winding",
    "compute_k_phi_multiscale_variance",
    "fit_k_phi_asymptotic_alpha",
    "k_phi_multiscale_safety",
    "fit_correlation_length_exponent",
    "measure_phase_symmetry",
]


# ============================================================================
# RESEARCH-PHASE UTILITIES (Not in modular implementations)
# ============================================================================

def _get_phase(G: Any, node: Any) -> float:
    """Retrieve phase value φ for a node (radians in [0, 2π))."""
    node_data = G.nodes[node]
    for alias in ALIAS_THETA:
        if alias in node_data:
            return float(node_data[alias])
    return 0.0


def _wrap_angle(angle: float) -> float:
    """Map angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def path_integrated_gradient(
    G: Any, source: Any, target: Any
) -> float:
    """Compute path-integrated phase gradient along a shortest path.

    **Status**: RESEARCH (telemetry support for custom analyses)

    Definition
    ----------
    Given a path P = [v_0, v_1, ..., v_k] from source to target:
        PIG = Σ_{i=0}^{k-1} |∇φ|(v_i)

    where |∇φ|(v) is the phase gradient at node v.

    Physical Interpretation
    -----------------------
    Cumulative phase desynchronization along a path. High PIG indicates
    that the path traverses regions with significant local phase disorder.

    Parameters
    ----------
    G : TNFRGraph
        Graph with node phase attributes
    source : NodeId
        Start node
    target : NodeId
        End node

    Returns
    -------
    float
        Path-integrated gradient (sum of node gradients along shortest path).
        Returns 0.0 if no path exists or nodes are isolated.

    Notes
    -----
    - Telemetry-only; does not mutate graph state.
    - Uses shortest path from networkx.
    - If multiple shortest paths exist, uses lexicographically first one
      (arbitrary but deterministic).
    """
    if nx is None:
        raise RuntimeError("networkx required for path operations")

    try:
        path = nx.shortest_path(G, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.0

    # Compute phase gradient if not cached
    grad = compute_phase_gradient(G)

    # Sum gradients along path
    total = 0.0
    for node in path:
        if node in grad:
            total += grad[node]

    return float(total)


def measure_phase_symmetry(G: Any) -> float:
    """Compute a phase symmetry metric in [0, 1].

    **Status**: RESEARCH (telemetry-only compatibility function)

    Definition
    ----------
    Let {φ_i} be phases for all nodes with a phase attribute.
    Compute circular mean μ = Arg( Σ_i e^{j φ_i} ). Symmetry metric:

        S = 1 - mean( |sin(φ_i - μ)| )

    Interpretation
    --------------
    S ≈ 1  : Highly clustered / symmetric phase distribution.
    S → 0  : Broad / antisymmetric distribution (desynchronization).

    Returns 0.0 if no phases are available.

    Notes
    -----
    - Read-only; does not mutate graph state (grammar safe).
    - Provides backward compatibility for benchmarks expecting this symbol.
    - Invariant #5 respected (phase verification external to this metric).
    """
    phases: List[float] = []
    # Collect phases from node attributes using alias list
    for node, data in G.nodes(data=True):  # type: ignore[attr-defined]
        for alias in ALIAS_THETA:
            if alias in data:
                try:
                    phases.append(float(data[alias]))
                except (TypeError, ValueError):
                    pass
                break
    if not phases:
        return 0.0
    arr = np.array(phases, dtype=float)
    # Wrap into [0, 2π)
    arr = np.mod(arr, 2 * math.pi)
    vec = np.exp(1j * arr)
    mean_angle = float(np.angle(np.mean(vec)))
    diffs = np.abs(np.sin(arr - mean_angle))
    return float(1.0 - min(1.0, float(np.mean(diffs))))


def compute_phase_winding(G: Any, cycle_nodes: List[Any]) -> int:
    """Compute winding number (topological charge) for a closed cycle.

    **Status**: RESEARCH (topological analysis support)

    Definition
    ----------
    For a closed loop of nodes, count full rotations of phase:
        q = (1/2π) Σ_{edges in cycle} Δφ_wrapped

    Returns
    -------
    int
        Winding number q. Non-zero values indicate phase vortices/defects
        enclosed by the loop.

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with per-node phase attribute.
    cycle_nodes : list
        Ordered list of node IDs forming a closed cycle. Function will
        connect the last node back to the first to complete the loop.

    Returns
    -------
    int
        Integer winding number (topological charge). Values != 0 indicate
        a phase vortex/defect enclosed by the loop.

    Notes
    -----
    - Telemetry-only; does not mutate EPI.
    - Robust to local reparameterizations of phase due to circular wrapping.
    - If fewer than 2 nodes are provided, returns 0.
    """
    if not cycle_nodes or len(cycle_nodes) < 2:
        return 0

    total = 0.0
    seq = list(cycle_nodes)
    # Ensure closure by including last->first
    for i, j in zip(seq, seq[1:] + [seq[0]]):
        phi_i = _get_phase(G, i)
        phi_j = _get_phase(G, j)
        total += _wrap_angle(phi_j - phi_i)

    q = int(round(total / (2.0 * math.pi)))
    return q


def _ego_mean(values: Dict[Any, float], nodes: list) -> float:
    """Mean of values restricted to given nodes; returns 0.0 if empty."""
    if not nodes:
        return 0.0
    arr = [values[n] for n in nodes if n in values]
    if not arr:
        return 0.0
    return float(sum(arr) / len(arr))


def compute_k_phi_multiscale_variance(
    G: Any,
    *,
    scales: tuple = (1, 2, 3, 5),
    k_phi_field: Optional[Dict[Any, float]] = None,
) -> Dict[int, float]:
    """Compute variance of coarse-grained K_φ across scales [RESEARCH].

    Definition (coarse-graining by r-hop ego neighborhoods):
        K_φ^r(i) = mean_{j in ego_r(i)} K_φ(j)
        var_r = Var_i [ K_φ^r(i) ]

    Parameters
    ----------
    G : TNFRGraph
        NetworkX-like graph with phase attributes accessible via aliases.
    scales : tuple[int, ...]
        Radii (in hops) at which to compute coarse-grained variance.
    k_phi_field : Optional[Dict]
        Precomputed K_φ per node. If None, computed via
        compute_phase_curvature.

    Returns
    -------
    Dict[int, float]
        Mapping from radius r to variance of coarse-grained K_φ at scale.

    Notes
    -----
    - Read-only telemetry; does not mutate graph state.
    - Intended to support asymptotic freedom assessments.
    """
    if k_phi_field is None:
        k_phi_field = compute_phase_curvature(G)

    nodes = list(G.nodes())
    variance_by_scale = {}

    for scale in scales:
        coarse_k_phi = {}
        for src in nodes:
            # BFS ego-graph of radius scale
            ego_nodes = set([src])
            frontier = set([src])
            for _ in range(scale):
                next_frontier = set()
                for node in frontier:
                    for neighbor in G.neighbors(node):
                        if neighbor not in ego_nodes:
                            ego_nodes.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier

            # Coarse-grained K_φ as mean over ego-graph
            coarse_k_phi[src] = _ego_mean(k_phi_field, list(ego_nodes))

        # Variance across all nodes
        vals = np.array(list(coarse_k_phi.values()))
        variance_by_scale[scale] = float(np.var(vals))

    return variance_by_scale


def fit_k_phi_asymptotic_alpha(
    variance_by_scale: Dict[int, float], alpha_hint: float = 2.76
) -> Dict[str, Any]:
    """Fit power-law exponent α for multiscale K_φ variance decay.

    **Status**: RESEARCH (multiscale analysis support)

    Model
    -----
    var(K_φ) at scale r ~ C / r^α

    Taking logarithms:
        log(var) = log(C) - α * log(r)

    Parameters
    ----------
    variance_by_scale : Dict[int, float]
        Mapping from scale r to variance of coarse-grained K_φ
    alpha_hint : float
        Expected value of α for comparison (default 2.76 from research)

    Returns
    -------
    Dict[str, Any]
        - alpha: Fitted exponent α
        - c: Fitted constant C (pre-factor)
        - r_squared: Goodness of fit
        - residuals: Per-scale residuals
        - prediction_error: Relative error vs alpha_hint
    """
    if len(variance_by_scale) < 3:
        return {
            "alpha": 0.0,
            "c": 0.0,
            "r_squared": 0.0,
            "residuals": {},
            "prediction_error": 0.0,
        }

    scales = np.array(sorted(variance_by_scale.keys()))
    variances = np.array([variance_by_scale[s] for s in scales])

    # Fit log(var) = log(C) - alpha * log(scale)
    log_scales = np.log(scales.astype(float))
    log_vars = np.log(variances + 1e-12)  # Avoid log(0)

    try:
        coeffs = np.polyfit(log_scales, log_vars, 1)
        alpha = -coeffs[0]
        log_c = coeffs[1]
        c = np.exp(log_c)

        # Compute R^2
        fitted = log_c - alpha * log_scales
        ss_res = np.sum((log_vars - fitted) ** 2)
        ss_tot = np.sum((log_vars - np.mean(log_vars)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        # Residuals per scale
        residuals = {
            s: float(v - np.exp(fitted[i]))
            for i, (s, v) in enumerate(variance_by_scale.items())
        }

        # Error vs hint
        pred_error = abs(alpha - alpha_hint) / (alpha_hint + 1e-9)

        return {
            "alpha": float(alpha),
            "c": float(c),
            "r_squared": float(r2),
            "residuals": residuals,
            "prediction_error": float(pred_error),
        }
    except (np.linalg.LinAlgError, ValueError):
        return {
            "alpha": 0.0,
            "c": 0.0,
            "r_squared": 0.0,
            "residuals": {},
            "prediction_error": 0.0,
        }


def k_phi_multiscale_safety(
    G: Any,
    alpha_hint: float = 2.76,
    fit_min_r2: float = 0.5,
) -> Dict[str, Any]:
    """Assess multiscale safety of K_φ field [RESEARCH].

    **Status**: RESEARCH (safety analysis support)

    Computes coarse-grained K_φ variance across scales, fits power-law
    decay, and returns safety verdict based on fit quality and threshold
    violations.

    Returns
    -------
    Dict[str, Any]
        - variance_by_scale: Dict[int, float] - computed variances
        - fit: Dict - power-law fitting results
        - violations: List[int] - scales with |K_φ| >= 3.0
        - safe: bool - overall safety status
    """
    # Compute multiscale variance
    variance_by_scale = compute_k_phi_multiscale_variance(G)

    # Fit power-law
    fit = fit_k_phi_asymptotic_alpha(variance_by_scale, alpha_hint)

    # Check for threshold violations
    # (Removed unused local k_phi_field assignment to satisfy lint)
    violations = [
        r
        for r, var in variance_by_scale.items()
        if var > 3.0 ** 2
    ]  # Approx threshold

    # Assess safety
    safe_by_fit = (
        fit.get("alpha", 0.0) > 0.0
        and fit.get("r_squared", 0.0) >= fit_min_r2
    )
    safe_by_tolerance = (alpha_hint is not None) and (len(violations) == 0)
    safe = bool(safe_by_fit or safe_by_tolerance)

    return {
        "variance_by_scale": {
            int(k): float(v) for k, v in variance_by_scale.items()
        },
        "fit": fit,
        "violations": violations,
        "safe": safe,
    }


def fit_correlation_length_exponent(
    intensities: np.ndarray,
    xi_c_values: np.ndarray,
    I_c: float = 2.015,
    min_distance: float = 0.01,
) -> Dict[str, Any]:
    """Fit critical exponent nu from xi_C ~ |I - I_c|^(-nu) [RESEARCH].

    **Status**: RESEARCH (critical phenomena analysis support)

    Theory
    ------
    At continuous phase transitions, correlation length diverges:
        xi_C ~ |I - I_c|^(-nu)

    Taking logarithms:
        log(xi_C) = log(A) - nu * log(|I - I_c|)

    Parameters
    ----------
    intensities : np.ndarray
        Array of intensity values I
    xi_c_values : np.ndarray
        Corresponding coherence lengths xi_C
    I_c : float, default=2.015
        Critical intensity (from prior evidence)
    min_distance : float, default=0.01
        Minimum |I - I_c| to avoid divergence noise

    Returns
    -------
    Dict[str, Any]
        - nu_below: Critical exponent for I < I_c
        - nu_above: Critical exponent for I > I_c
        - r_squared_below: Fit quality below I_c
        - r_squared_above: Fit quality above I_c
        - universality_class: 'mean-field' | 'ising-3d' | 'ising-2d' |
          'unknown'
        - n_points_below: Number of data points I < I_c
        - n_points_above: Number of data points I > I_c

    Notes
    -----
    Expected critical exponents:
    - Mean-field: nu = 0.5
    - 3D Ising: nu = 0.63
    - 2D Ising: nu = 1.0
    """
    results = {
        "nu_below": 0.0,
        "nu_above": 0.0,
        "r_squared_below": 0.0,
        "r_squared_above": 0.0,
        "universality_class": "unknown",
        "n_points_below": 0,
        "n_points_above": 0,
    }

    # Split data at critical point
    below_mask = (
        (intensities < I_c)
        & (np.abs(intensities - I_c) > min_distance)
    )
    above_mask = (
        (intensities > I_c)
        & (np.abs(intensities - I_c) > min_distance)
    )

    # Fit below I_c
    if np.sum(below_mask) >= 3:
        I_below = intensities[below_mask]
        xi_below = xi_c_values[below_mask]

        x = np.log(np.abs(I_below - I_c))
        y = np.log(xi_below)

        # Linear regression: y = a - nu * x
        coeffs = np.polyfit(x, y, 1)
        nu_below = -coeffs[0]  # Negative slope

        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_below = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results["nu_below"] = float(nu_below)
        results["r_squared_below"] = float(r2_below)
        results["n_points_below"] = int(np.sum(below_mask))

    # Fit above I_c
    if np.sum(above_mask) >= 3:
        I_above = intensities[above_mask]
        xi_above = xi_c_values[above_mask]

        x = np.log(np.abs(I_above - I_c))
        y = np.log(xi_above)

        coeffs = np.polyfit(x, y, 1)
        nu_above = -coeffs[0]

        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_above = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results["nu_above"] = float(nu_above)
        results["r_squared_above"] = float(r2_above)
        results["n_points_above"] = int(np.sum(above_mask))

    # Classify universality
    if results["n_points_below"] >= 3 and results["n_points_above"] >= 3:
        nu_avg = (results["nu_below"] + results["nu_above"]) / 2.0
        if abs(nu_avg - 0.5) < 0.1:
            results["universality_class"] = "mean-field"
        elif abs(nu_avg - 0.63) < 0.1:
            results["universality_class"] = "ising-3d"
        elif abs(nu_avg - 1.0) < 0.15:
            results["universality_class"] = "ising-2d"

    return results


# Import extended canonical fields (NEWLY PROMOTED Nov 12, 2025)
# as fallback for development/testing environments
# Redundant import block removed (extended canonical already imported)


# End of physics field computations.
#
# CANONICAL fields (Φ_s, |∇φ|, K_φ, ξ_C) are validated telemetry
# for operator safety/diagnosis (read-only; never mutate EPI).
# RESEARCH fields (e.g., PIG) are telemetry-only.

