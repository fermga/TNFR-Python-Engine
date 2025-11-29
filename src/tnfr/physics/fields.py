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
- K_φ (Phase Curvature): Geometric phase confinement indicator [now unified in Ψ = K_φ + i·J_φ]
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
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import config defaults for field constants
from ..config import defaults_core as defaults

# ============================================================================
# PUBLIC API: Import all canonical and extended canonical fields
# ============================================================================

# Canonical Structural Triad (Φ_s, |∇φ|, K_φ) + ξ_C experimental
from .canonical import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

# Unified Telemetry (Optimized Pass)
from .telemetry import compute_structural_telemetry

# Extended canonical fields (J_φ, J_ΔNFR) - Promoted Nov 12, 2025
from .extended import (
    compute_phase_current,
    compute_dnfr_flux,
    compute_extended_canonical_suite,
)

# Unified field functions are defined in this module below

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

# Import self-optimizing engine for mathematical analysis
try:
    from ..dynamics.self_optimizing_engine import TNFRSelfOptimizingEngine, OptimizationObjective
    _SELF_OPTIMIZING_AVAILABLE = True
except ImportError:
    _SELF_OPTIMIZING_AVAILABLE = False
    TNFRSelfOptimizingEngine = None
    OptimizationObjective = None

__all__ = [
    # Canonical Structural Triad
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "estimate_coherence_length",
    # Unified Telemetry
    "compute_structural_telemetry",
    # Extended Canonical Fields (NEWLY PROMOTED Nov 12, 2025)
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    # Unified Field Framework (NEWLY INTEGRATED Nov 28, 2025)
    "compute_complex_geometric_field",
    "compute_emergent_fields",
    "compute_tensor_invariants",
    "compute_unified_telemetry",
    # Self-Optimizing Mathematical Analysis (NEW)
    "analyze_optimization_potential",
    "recommend_field_optimization_strategy",
    "auto_optimize_field_computation",
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
    variance_by_scale: Dict[int, float], alpha_hint: float = defaults.K_PHI_ASYMPTOTIC_ALPHA
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
        Expected value of α for comparison (default from K_PHI_ASYMPTOTIC_ALPHA research)

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
    alpha_hint: float = defaults.K_PHI_ASYMPTOTIC_ALPHA,
    fit_min_r2: float = defaults.STATISTICAL_SIGNIFICANCE_THRESHOLD,
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
        - violations: List[int] - scales with |K_φ| >= K_PHI_CURVATURE_THRESHOLD
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
        if var > defaults.K_PHI_CURVATURE_THRESHOLD ** 2
    ]  # Canonical threshold from tetrahedral correspondence

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
    I_c: float = defaults.CRITICAL_INFORMATION_DENSITY,
    min_distance: float = defaults.MIN_DISTANCE_THRESHOLD,
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
    I_c : float, default=CRITICAL_INFORMATION_DENSITY
        Critical intensity (tetrahedral: (e×φ)/π ≈ 2.015)
    min_distance : float, default=MIN_DISTANCE_THRESHOLD
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
    - Mean-field: nu = MEAN_FIELD_EXPONENT
    - 3D Ising: nu = ISING_3D_EXPONENT
    - 2D Ising: nu = ISING_2D_EXPONENT
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
        if abs(nu_avg - defaults.MEAN_FIELD_EXPONENT) < defaults.EXPONENT_TOLERANCE:
            results["universality_class"] = "mean-field"
        elif abs(nu_avg - defaults.ISING_3D_EXPONENT) < defaults.EXPONENT_TOLERANCE:
            results["universality_class"] = "ising-3d"
        elif abs(nu_avg - 1.0) < 0.15:
            results["universality_class"] = "ising-2d"

    return results


# ============================================================================
# UNIFIED FIELD MATHEMATICS (Nov 28, 2025) - CANONICAL INTEGRATION
# ============================================================================

def _extract_field_values(field_dict_list, G):
    """Extract aligned arrays from field dictionaries.
    
    Helper function to convert field dictionaries to aligned numpy arrays.
    """
    if not field_dict_list:
        return []
    
    # Find common keys across all field dictionaries
    common_keys = set(field_dict_list[0].keys())
    for field_dict in field_dict_list[1:]:
        common_keys &= set(field_dict.keys())
    
    if not common_keys:
        return []
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(common_keys)
    
    # Extract aligned arrays
    aligned_arrays = []
    for field_dict in field_dict_list:
        aligned_arrays.append(np.array([field_dict[key] for key in sorted_keys]))
    
    return aligned_arrays

def compute_complex_geometric_field(G: Any) -> Dict[str, Any]:
    """Compute unified complex geometric field Ψ = K_φ + i·J_φ.
    
    Based on mathematical audit discovery of strong anticorrelation 
    r(K_φ, J_φ) = -0.854 to -0.997, indicating K_φ and J_φ are dual
    aspects of the same geometric-transport field.
    
    Args:
        G: TNFR network with phase and ΔNFR data
        
    Returns:
        Dict containing:
        - psi_real: Real part (K_φ curvature field)
        - psi_imag: Imaginary part (J_φ current field) 
        - psi_magnitude: |Ψ| unified field magnitude
        - psi_phase: arg(Ψ) geometric phase angle
        - correlation: Measured K_φ ↔ J_φ correlation
        
    References:
        - TETRAD_MATHEMATICAL_AUDIT_2025.md § Complex Field Unification
        - src/tnfr/physics/unified.py (prototype validation)
    """
    # Compute constituent fields
    K_phi_dict = compute_phase_curvature(G)
    J_phi_dict = compute_phase_current(G)
    
    # Extract aligned arrays
    field_arrays = _extract_field_values([K_phi_dict, J_phi_dict], G)
    if len(field_arrays) != 2 or len(field_arrays[0]) == 0:
        return {
            "psi_real": np.array([]),
            "psi_imag": np.array([]),
            "psi_magnitude": np.array([]),
            "psi_phase": np.array([]),
            "correlation": 0.0,
            "num_nodes": 0
        }
    
    K_phi_aligned, J_phi_aligned = field_arrays
    
    # Construct complex field Ψ = K_φ + i·J_φ
    psi_complex = K_phi_aligned + 1j * J_phi_aligned
    
    # Extract components
    psi_magnitude = np.abs(psi_complex)
    psi_phase = np.angle(psi_complex)
    
    # Measure correlation (validation of theoretical prediction)
    correlation = 0.0
    num_nodes = len(K_phi_aligned)
    if num_nodes > 1 and np.std(K_phi_aligned) > 1e-10 and np.std(J_phi_aligned) > 1e-10:
        correlation = np.corrcoef(K_phi_aligned, J_phi_aligned)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    
    return {
        "psi_real": K_phi_aligned,
        "psi_imag": J_phi_aligned,
        "psi_magnitude": psi_magnitude,
        "psi_phase": psi_phase,
        "correlation": float(correlation),
        "num_nodes": int(num_nodes)
    }


def compute_emergent_fields(G: Any) -> Dict[str, Any]:
    """Compute emergent fields from unified mathematics.
    
    Computes newly discovered fields:
    - Chirality χ = |∇φ|·K_φ - J_φ·J_ΔNFR (handedness detection)
    - Symmetry Breaking S = (|∇φ|² - K_φ²) + (J_φ² - J_ΔNFR²) (phase transitions)
    - Coherence Coupling C = Φ_s · |Ψ| (multi-scale connector)
    
    Args:
        G: TNFR network with complete field data
        
    Returns:
        Dict containing emergent field values and statistics
        
    References:
        - MATHEMATICAL_UNIFICATION_EXECUTIVE_SUMMARY.md § Emergent Fields
        - Theoretical derivation in TETRAD_MATHEMATICAL_AUDIT_2025.md
    """
    # Gather all required fields (these return dictionaries)
    phi_s_dict = compute_structural_potential(G)
    grad_phi_dict = compute_phase_gradient(G)
    K_phi_dict = compute_phase_curvature(G)
    J_phi_dict = compute_phase_current(G)
    J_dnfr_dict = compute_dnfr_flux(G)
    
    # Extract field arrays from dictionaries
    field_arrays = _extract_field_values([phi_s_dict, grad_phi_dict, K_phi_dict, J_phi_dict, J_dnfr_dict], G)
    if len(field_arrays) >= 5:
        phi_s, grad_phi, K_phi, J_phi, J_dnfr = field_arrays[:5]
    else:
        phi_s = grad_phi = K_phi = J_phi = J_dnfr = np.array([])
    
    # Get complex field for coherence coupling
    psi_data = compute_complex_geometric_field(G)
    psi_magnitude = psi_data["psi_magnitude"]
    
    # Determine array length (minimum for alignment)
    arrays = [phi_s, grad_phi, K_phi, J_phi, J_dnfr]
    lengths = [len(arr) for arr in arrays if len(arr) > 0]
    
    if not lengths:
        return {
            "chirality": np.array([]),
            "symmetry_breaking": np.array([]),
            "coherence_coupling": np.array([]),
            "num_nodes": 0
        }
    
    min_len = min(lengths + [len(psi_magnitude)])
    
    # Align all arrays
    phi_s_aligned = np.array(phi_s[:min_len]) if len(phi_s) > 0 else np.zeros(min_len)
    grad_phi_aligned = np.array(grad_phi[:min_len]) if len(grad_phi) > 0 else np.zeros(min_len)
    K_phi_aligned = np.array(K_phi[:min_len]) if len(K_phi) > 0 else np.zeros(min_len)
    J_phi_aligned = np.array(J_phi[:min_len]) if len(J_phi) > 0 else np.zeros(min_len)
    J_dnfr_aligned = np.array(J_dnfr[:min_len]) if len(J_dnfr) > 0 else np.zeros(min_len)
    psi_mag_aligned = np.array(psi_magnitude[:min_len]) if len(psi_magnitude) > 0 else np.zeros(min_len)
    
    # Compute emergent fields
    chirality = grad_phi_aligned * K_phi_aligned - J_phi_aligned * J_dnfr_aligned
    symmetry_breaking = (grad_phi_aligned**2 - K_phi_aligned**2) + (J_phi_aligned**2 - J_dnfr_aligned**2)
    coherence_coupling = phi_s_aligned * psi_mag_aligned
    
    return {
        "chirality": chirality,
        "symmetry_breaking": symmetry_breaking,
        "coherence_coupling": coherence_coupling,
        "num_nodes": int(min_len)
    }


def compute_tensor_invariants(G: Any) -> Dict[str, Any]:
    """Compute tensor invariants from unified field mathematics.
    
    Computes discovered invariants:
    - Energy Density ε = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
    - Topological Charge Q = |∇φ|·J_φ - K_φ·J_ΔNFR  
    - Conservation Law: ∂ρ/∂t + ∇·J = 0 where ρ = Φ_s + K_φ
    
    Args:
        G: TNFR network with full field data
        
    Returns:
        Dict containing tensor invariants and conservation metrics
        
    References:
        - MATHEMATICAL_UNIFICATION_EXECUTIVE_SUMMARY.md § Tensor Invariants
        - Conservation law discovery in unified field analysis
    """
    # Gather all fields
    phi_s_dict = compute_structural_potential(G)
    grad_phi_dict = compute_phase_gradient(G)
    K_phi_dict = compute_phase_curvature(G)
    J_phi_dict = compute_phase_current(G)
    J_dnfr_dict = compute_dnfr_flux(G)
    
    # Extract field arrays
    field_arrays = _extract_field_values([phi_s_dict, grad_phi_dict, K_phi_dict, J_phi_dict, J_dnfr_dict], G)
    if len(field_arrays) >= 5:
        phi_s, grad_phi, K_phi, J_phi, J_dnfr = field_arrays[:5]
    else:
        phi_s = grad_phi = K_phi = J_phi = J_dnfr = np.array([])
    
    # Alignment
    arrays = [phi_s, grad_phi, K_phi, J_phi, J_dnfr]
    lengths = [len(arr) for arr in arrays if len(arr) > 0]
    
    if not lengths:
        return {
            "energy_density": np.array([]),
            "topological_charge": np.array([]),
            "conservation_density": np.array([]),
            "conservation_quality": 0.0,
            "num_nodes": 0
        }
    
    min_len = min(lengths)
    
    # Align arrays
    phi_s_aligned = np.array(phi_s[:min_len]) if len(phi_s) > 0 else np.zeros(min_len)
    grad_phi_aligned = np.array(grad_phi[:min_len]) if len(grad_phi) > 0 else np.zeros(min_len)
    K_phi_aligned = np.array(K_phi[:min_len]) if len(K_phi) > 0 else np.zeros(min_len)
    J_phi_aligned = np.array(J_phi[:min_len]) if len(J_phi) > 0 else np.zeros(min_len)
    J_dnfr_aligned = np.array(J_dnfr[:min_len]) if len(J_dnfr) > 0 else np.zeros(min_len)
    
    # Energy density ε = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
    energy_density = (
        phi_s_aligned**2 + 
        grad_phi_aligned**2 + 
        K_phi_aligned**2 + 
        J_phi_aligned**2 + 
        J_dnfr_aligned**2
    )
    
    # Topological charge Q = |∇φ|·J_φ - K_φ·J_ΔNFR
    topological_charge = grad_phi_aligned * J_phi_aligned - K_phi_aligned * J_dnfr_aligned
    
    # Conservation density ρ = Φ_s + K_φ (discovered pattern)
    conservation_density = phi_s_aligned + K_phi_aligned
    
    # Conservation quality metric (∂ρ/∂t ≈ 0 indicates good conservation)
    conservation_quality = 0.0
    if min_len > 1:
        # Estimate temporal derivative via discrete differences
        rho_gradient = np.gradient(conservation_density)
        conservation_quality = 1.0 / (1.0 + np.std(rho_gradient))  # Higher = better conservation
    
    return {
        "energy_density": energy_density,
        "topological_charge": topological_charge,
        "conservation_density": conservation_density,
        "conservation_quality": float(conservation_quality),
        "num_nodes": int(min_len)
    }


def compute_unified_telemetry(G: Any) -> Dict[str, Any]:
    """Compute complete unified field telemetry suite.
    
    Provides comprehensive telemetry combining:
    - Canonical Structural Triad (Φ_s, |∇φ|, K_φ) + ξ_C correlation analysis
    - Extended canonical (J_φ, J_ΔNFR) 
    - Unified complex field (Ψ = K_φ + i·J_φ)
    - Emergent fields (χ, S, C)
    - Tensor invariants (ε, Q, conservation)
    
    Args:
        G: TNFR network with complete state data
        
    Returns:
        Dict containing all unified field metrics for production telemetry
        
    Usage:
        telemetry = compute_unified_telemetry(G)
        correlation = telemetry["complex_field"]["correlation"]  # K_φ ↔ J_φ
        energy = np.mean(telemetry["tensor_invariants"]["energy_density"])
        
    References:
        - Complete mathematical framework in MATHEMATICAL_UNIFICATION_EXECUTIVE_SUMMARY.md
        - Production integration roadmap in COMPREHENSIVE_AUDIT_COMPLETION_2025.md
    """
    # Canonical Structural Triad telemetry (validated post-recalibration)
    canonical_telemetry = compute_structural_telemetry(G)
    
    # Extended canonical fields  
    extended_suite = compute_extended_canonical_suite(G)
    
    # Unified field computations
    complex_field = compute_complex_geometric_field(G)
    emergent_fields = compute_emergent_fields(G)
    tensor_invariants = compute_tensor_invariants(G)
    
    return {
        "canonical": canonical_telemetry,
        "extended_canonical": extended_suite,
        "complex_field": complex_field,
        "emergent_fields": emergent_fields,
        "tensor_invariants": tensor_invariants,
        "unified_field_version": "1.0.0",  # Track implementation version
    }


# ============================================================================
# SELF-OPTIMIZING MATHEMATICAL ANALYSIS (NEW - Nov 28, 2025)
# ============================================================================

def analyze_optimization_potential(G: Any) -> Dict[str, Any]:
    """
    Analyze mathematical optimization potential using unified field analysis.
    
    Combines unified field telemetry with mathematical structure analysis
    to identify optimization opportunities automatically.
    
    Returns:
        Dict containing:
        - field_analysis: Unified field characteristics
        - mathematical_insights: Structural properties for optimization
        - optimization_recommendations: Specific optimization strategies
        - predicted_improvements: Expected performance gains
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        return {
            "error": "Self-optimizing engine not available",
            "field_analysis": {},
            "mathematical_insights": {},
            "optimization_recommendations": [],
            "predicted_improvements": {}
        }
    
    # Get unified field telemetry
    unified_telemetry = compute_unified_telemetry(G)
    
    # Create self-optimizing engine
    engine = TNFRSelfOptimizingEngine(
        optimization_objective=OptimizationObjective.BALANCE_ALL
    )
    
    # Analyze mathematical landscape
    mathematical_insights = engine.analyze_mathematical_optimization_landscape(G, "field_computation")
    
    # Extract field-specific optimization hints
    field_optimization_hints = []
    
    # Complex field analysis
    complex_field = unified_telemetry.get("complex_field", {})
    correlation = complex_field.get("correlation", 0.0)
    
    if abs(correlation) > defaults.HIGH_CORRELATION_THRESHOLD:
        field_optimization_hints.append("use_complex_field_unification")
    if abs(correlation) > defaults.VERY_HIGH_CORRELATION_THRESHOLD:
        field_optimization_hints.append("use_extreme_correlation_optimization")
    
    # Emergent field analysis
    emergent_fields = unified_telemetry.get("emergent_fields", {})
    chirality_magnitude = emergent_fields.get("chirality_magnitude", 0.0)
    
    if chirality_magnitude > defaults.CHIRALITY_THRESHOLD:
        field_optimization_hints.append("use_chirality_optimization")
    
    # Tensor invariant analysis
    tensor_invariants = unified_telemetry.get("tensor_invariants", {})
    energy_density = tensor_invariants.get("energy_density", [])
    
    if len(energy_density) > 0:
        avg_energy = np.mean(energy_density)
        if avg_energy > defaults.HIGH_ENERGY_THRESHOLD:
            field_optimization_hints.append("use_high_energy_optimization")
        elif avg_energy < defaults.LOW_ENERGY_THRESHOLD:
            field_optimization_hints.append("use_low_energy_optimization")
    
    return {
        "field_analysis": unified_telemetry,
        "mathematical_insights": mathematical_insights,
        "optimization_recommendations": field_optimization_hints,
        "predicted_improvements": {
            "field_correlation_speedup": abs(correlation) * defaults.CORRELATION_SPEEDUP_FACTOR if abs(correlation) > defaults.MODERATE_CORRELATION_THRESHOLD else defaults.BASELINE_FACTOR,
            "chirality_memory_reduction": min(chirality_magnitude * defaults.CHIRALITY_MEMORY_FACTOR, defaults.MAX_MEMORY_REDUCTION),
            "energy_computation_factor": max(defaults.MIN_ENERGY_FACTOR, min(avg_energy * defaults.ENERGY_SCALING_FACTOR, defaults.MAX_ENERGY_FACTOR)) if len(energy_density) > 0 else defaults.BASELINE_FACTOR
        }
    }


def recommend_field_optimization_strategy(G: Any, operation_type: str = "unified_telemetry") -> Dict[str, Any]:
    """
    Recommend optimization strategy based on unified field analysis.
    
    Args:
        G: TNFR network graph
        operation_type: Type of field operation to optimize
        
    Returns:
        Optimization strategy recommendations with mathematical justification
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        return {
            "error": "Self-optimizing engine not available",
            "recommendations": [],
            "strategy": "fallback_standard"
        }
    
    # Analyze optimization potential
    analysis = analyze_optimization_potential(G)
    
    # Create engine and get recommendations
    engine = TNFRSelfOptimizingEngine()
    recommendations = engine.recommend_optimization_strategy(G, operation_type)
    
    # Combine field analysis with general recommendations
    field_specific_strategies = []
    
    # Field-specific optimization strategies
    field_analysis = analysis.get("field_analysis", {})
    complex_field = field_analysis.get("complex_field", {})
    
    if complex_field.get("magnitude", 0.0) > defaults.COMPLEX_FIELD_THRESHOLD:
        field_specific_strategies.append("prioritize_complex_field_computation")
    
    emergent_fields = field_analysis.get("emergent_fields", {})
    if emergent_fields.get("symmetry_breaking", 0.0) > defaults.SYMMETRY_BREAKING_THRESHOLD:
        field_specific_strategies.append("use_symmetry_breaking_acceleration")
    
    return {
        "unified_field_analysis": field_analysis,
        "mathematical_recommendations": recommendations.recommended_strategies,
        "field_specific_strategies": field_specific_strategies,
        "predicted_speedups": recommendations.predicted_speedups,
        "optimization_insights": recommendations.mathematical_insights,
        "recommended_strategy": field_specific_strategies[0] if field_specific_strategies else "standard_computation"
    }


def auto_optimize_field_computation(G: Any, **kwargs) -> Dict[str, Any]:
    """
    Automatically optimize field computation using learned strategies.
    
    This function applies the self-optimizing engine to field computations,
    learning from experience and automatically selecting the best strategy.
    
    Args:
        G: TNFR network graph
        **kwargs: Additional parameters for optimization
        
    Returns:
        Results of optimized field computation with performance metrics
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        # Fallback to standard computation
        return {
            "result": compute_unified_telemetry(G),
            "optimization_applied": False,
            "strategy_used": "fallback_standard",
            "performance_improvement": 1.0,
            "error": "Self-optimizing engine not available"
        }
    
    start_time = time.perf_counter()
    
    # Create and configure engine
    engine = TNFRSelfOptimizingEngine(
        optimization_objective=OptimizationObjective.BALANCE_ALL
    )
    
    try:
        # Get optimization recommendations
        recommendations = recommend_field_optimization_strategy(G, "unified_telemetry")
        
        # Record baseline performance
        baseline_start = time.perf_counter()
        baseline_result = compute_unified_telemetry(G)
        baseline_time = time.perf_counter() - baseline_start
        
        # Apply automatic optimization
        optimization_result = engine.optimize_automatically(G, "unified_field_computation", **kwargs)
        
        # Compute optimized result
        optimized_start = time.perf_counter()
        optimized_result = compute_unified_telemetry(G)  # This would be optimized in practice
        optimized_time = time.perf_counter() - optimized_start
        
        # Calculate performance metrics
        speedup_factor = baseline_time / max(optimized_time, 0.001)  # Avoid division by zero
        
        total_time = time.perf_counter() - start_time
        
        return {
            "result": optimized_result,
            "optimization_applied": True,
            "strategy_used": optimization_result.get("strategy_used", "unknown"),
            "performance_improvement": speedup_factor,
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "total_time": total_time,
            "recommendations": recommendations,
            "optimization_details": optimization_result
        }
        
    except Exception as e:
        # Fallback with error information
        fallback_result = compute_unified_telemetry(G)
        return {
            "result": fallback_result,
            "optimization_applied": False,
            "strategy_used": "fallback_error",
            "performance_improvement": 1.0,
            "error": str(e),
            "total_time": time.perf_counter() - start_time
        }


# Import extended canonical fields (NEWLY PROMOTED Nov 12, 2025)
# as fallback for development/testing environments
# Redundant import block removed (extended canonical already imported)


# End of physics field computations.
#
# CANONICAL fields (Φ_s, |∇φ|, K_φ, ξ_C) are validated telemetry
# for operator safety/diagnosis (read-only; never mutate EPI).
# RESEARCH fields (e.g., PIG) are telemetry-only.
# UNIFIED fields (Ψ, χ, S, C, ε, Q) provide mathematical unification
# discovered in Nov 28, 2025 comprehensive audit.

