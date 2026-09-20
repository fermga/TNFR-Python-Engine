"""Structural field computations for TNFR physics.

REORGANIZED (Nov 14, 2025): Canonical field implementations moved to modular
submodules (canonical.py, extended.py) to reduce coupling and improve
maintainability. This module now acts as the public API, re-exporting all
canonical fields and containing only research-phase utilities.

This module computes emergent structural "fields" from TNFR graph state,
grounding a pathway from the nodal equation to macroscopic interaction
patterns.

CANONICAL FIELDS (Read-Only Telemetry)
---------------------------------------
All four structural fields form the canonical diagnostic interface:

- Φ_s (Structural Potential): Global field from ΔNFR distribution
- |∇φ| (Phase Gradient): Local phase desynchronization metric
- K_φ (Phase Curvature): Geometric phase confinement indicator [now unified in Ψ = K_φ + i·J_φ]
- ξ_C (Coherence Length): State- and topology-dependent correlation estimate

Canonical status specifies the required read-outs and their implementations.
It does not prove that four lossy summaries reconstruct the graph state or its
dynamics; minimal complete observability remains open. In particular, ξ_C is
nonlinear in the sampled field and can use a spectral-gap fallback on supported
symmetric graphs. The static product fit uses declared structural path-length
units; the normalized-generator spectral fallback is dimensionless. Its
provenance must be retained when interpretations differ.

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

ΔNFR represents structural pressure driving reorganization. The defined
distance-weighted aggregation of ΔNFR produces Φ_s. This read-out does not
derive a physical interaction or an evolution law for the graph metric.

References
----------
- UNIFIED_GRAMMAR_RULES.md § U6: STRUCTURAL POTENTIAL CONFINEMENT
- docs/STRUCTURAL_FIELDS_TETRAD.md: Field API and validation scope
- docs/XI_C_CANONICAL_PROMOTION.md: ξ_C experimental validation
- AGENTS.md § Structural Fields: Canonical tetrad documentation
- TNFR.pdf § 2.1: Nodal equation foundation
"""

from __future__ import annotations

import math
import time
from numbers import Real
from typing import Any

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:
    nx = None

# Import config defaults for field constants
from ..config import defaults_core as defaults

# ---------------------------------------------------------------------------
# Universality classification tolerance
# ---------------------------------------------------------------------------
_ISING_2D_EXPONENT_TOLERANCE = 0.15

# ============================================================================
# PUBLIC API: Import all canonical and extended canonical fields
# ============================================================================

# Canonical diagnostic tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
from .canonical import (
    CoherenceLengthEstimate,
    PhaseCurvatureNodeObservation,
    PhaseCurvatureObservation,
    UndefinedPhaseCurvatureError,
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
    estimate_coherence_length_with_provenance,
    observe_phase_curvature,
)

# Backward-compatible alias (used by pattern_discovery and parallel modules)
compute_structural_potential_field = compute_structural_potential

# Extended canonical fields (J_φ, J_ΔNFR) - Promoted Nov 12, 2025
from .extended import (
    compute_dnfr_flux,
    compute_extended_canonical_suite,
    compute_phase_current,
)

# Unified Telemetry (Optimized Pass)
from .telemetry import compute_structural_telemetry

# Unified field functions are defined in this module below

# Import TNFR cache system for research functions
_CACHE_AVAILABLE = True

# Import TNFR aliases
try:
    from ..constants.aliases import ALIAS_DNFR, ALIAS_THETA
except ImportError:
    ALIAS_THETA = ["phase", "theta"]
    ALIAS_DNFR = ["delta_nfr", "dnfr"]

# Import self-optimizing engine for mathematical analysis
try:
    from ..dynamics.self_optimizing_engine import (
        OptimizationObjective,
        TNFRSelfOptimizingEngine,
    )

    _SELF_OPTIMIZING_AVAILABLE = True
except ImportError:
    _SELF_OPTIMIZING_AVAILABLE = False
    TNFRSelfOptimizingEngine = None
    OptimizationObjective = None

__all__ = [
    # Canonical diagnostic tetrad
    "compute_structural_potential",
    "compute_phase_gradient",
    "compute_phase_curvature",
    "observe_phase_curvature",
    "PhaseCurvatureObservation",
    "PhaseCurvatureNodeObservation",
    "UndefinedPhaseCurvatureError",
    "estimate_coherence_length",
    "estimate_coherence_length_with_provenance",
    "CoherenceLengthEstimate",
    # Unified Telemetry
    "compute_structural_telemetry",
    # Extended Canonical Fields (NEWLY PROMOTED Nov 12, 2025)
    "compute_phase_current",
    "compute_dnfr_flux",
    "compute_extended_canonical_suite",
    # Unified Field Framework (NEWLY INTEGRATED Nov 28, 2025)
    "compute_complex_geometric_field_arrays",
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
    "classify_nodal_topology",
    "compute_k_phi_multiscale_variance",
    "fit_k_phi_asymptotic_alpha",
    "k_phi_multiscale_safety",
    "fit_correlation_length_exponent",
    "measure_phase_symmetry",
]

# ============================================================================
# RESEARCH-PHASE UTILITIES (Not in modular implementations)
# ============================================================================


def path_integrated_gradient(G: Any, source: Any, target: Any) -> float:
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
    - If multiple shortest paths exist, NetworkX resolves ties using graph
      traversal order. The sum has one term per edge, excluding the target.
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
    for node in path[:-1]:
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
    - Invariant #2 respected (phase verification external to this metric).
    """
    phases: list[float] = []
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


def compute_phase_winding(G: Any, cycle_nodes: list[Any]) -> int:
    """Read integer phase winding on a declared non-ambiguous graph cycle.

    Delegate support, phase and branch admission to the shared winding owner.
    Missing edges, repeated nodes, incomplete cycles and undefined wrap-boundary
    cases raise ``ValueError``; a nonexistent loop is not a zero-winding loop.
    This read-only integer is not a physical particle or defect classification.
    """
    from .emergent_particles import winding_number

    return winding_number(G, order=cycle_nodes)[0]


# Nodal-topology classification thresholds (TNFR.pdf §1.4.1: radial / annular /
# multinodal). MEASURED and validated on canonical topologies (star, ring,
# complete, barbell, two-hub, build_element_radial_pattern); these are
# calibration cuts, NOT physical constants.
_NFR_TOPOLOGY_ANNULAR_CONC_MAX = 1.10  # max/mean centrality below this = no
#   distinguished center (ring/complete ~1.00; center-bearing forms >= 1.17).
_NFR_TOPOLOGY_CENTER_TIER = 0.85  # centrality >= tier*max = a "center"
#   (radial -> exactly 1 center; multinodal -> >= 2).
_CANONICAL_NODAL_TOPOLOGY_ALPHA = 2.0


def _validate_nodal_topology_alpha(alpha: float) -> float:
    """Return the sole exponent covered by the calibrated topology labels."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("canonical nodal-topology classification requires alpha=2.0")
    try:
        value = float(alpha)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "canonical nodal-topology classification requires alpha=2.0"
        ) from exc
    if not math.isfinite(value) or value != _CANONICAL_NODAL_TOPOLOGY_ALPHA:
        raise ValueError("canonical nodal-topology classification requires alpha=2.0")
    return value


def classify_nodal_topology(G: Any, *, alpha: float = 2.0) -> dict[str, Any]:
    r"""Classify a network's nodal topology: radial / annular / multinodal.

    Per TNFR.pdf §1.4.1, every Fractal-Resonant Node (NFR) is a *region of
    structural coherence* with an internal **nodal topology**: radial (one
    central nucleus), annular (passive center, peripheral ring) or multinodal
    (several connected centers). This reads that topology from the canonical
    structural-potential geometry -- the Green's function of :math:`\Phi_s`
    under a *unit* structural source,

    .. math::
        c(i) = \sum_{j \neq i} \frac{1}{d(i,j)^\alpha}, \qquad \alpha = 2,

    i.e. the same inverse-square kernel as :func:`compute_structural_potential`
    (grammar U6) but sourced uniformly. It therefore reads the *structural*
    form of the region even when the dynamical :math:`\Phi_s` vanishes. This
    geometry-only read-out does not imply that every state channel is uniform
    at a general :math:`\Delta\mathrm{NFR}=0` snapshot.

    The classification is emergent and threshold-light: the concentration
    ``max(c)/mean(c)`` separates the rotationally-symmetric annular form
    (:math:`\approx 1`) from center-bearing forms; among the latter, the count
    of near-maximal centers (``c >= 0.85*max``) is 1 for radial and >= 2 for
    multinodal. Both cuts are measured/validated on canonical topologies, not
    physical constants.  ``alpha`` remains in the signature for compatibility
    but must equal the calibrated canonical value ``2.0``.

    Returns
    -------
    dict
        ``topology`` ("radial"/"annular"/"multinodal"), ``centers`` (center
        node ids), ``concentration`` (max/mean), ``dispersion`` (coefficient
        of variation), ``centrality`` (per-node geometric centrality) and
        ``n_nodes``.
    """
    exponent = _validate_nodal_topology_alpha(alpha)
    if nx is None:
        raise RuntimeError("networkx is required for nodal-topology classification")
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {
            "topology": "annular",
            "centers": [],
            "concentration": 0.0,
            "dispersion": 0.0,
            "centrality": {},
            "n_nodes": 0,
        }
    from .canonical import _compute_phi_s_exact

    # Evaluate the documented unit-source geometry with exactly the same
    # weighted, outgoing distance kernel as the dynamical potential.
    centrality = _compute_phi_s_exact(G, nodes, {node: 1.0 for node in nodes}, exponent)
    vals = np.asarray([centrality[i] for i in nodes], dtype=float)
    mean = float(vals.mean())
    vmax = float(vals.max())
    if mean <= 0.0:
        return {
            "topology": "annular",
            "centers": [],
            "concentration": 0.0,
            "dispersion": 0.0,
            "centrality": centrality,
            "n_nodes": n,
        }
    conc = vmax / mean
    cv = float(vals.std()) / mean
    if conc < _NFR_TOPOLOGY_ANNULAR_CONC_MAX:
        topology = "annular"
        centers: list[Any] = []
    else:
        centers = [
            i for i in nodes if centrality[i] >= _NFR_TOPOLOGY_CENTER_TIER * vmax
        ]
        topology = "radial" if len(centers) == 1 else "multinodal"
    return {
        "topology": topology,
        "centers": centers,
        "concentration": conc,
        "dispersion": cv,
        "centrality": centrality,
        "n_nodes": n,
    }


def _ego_mean(values: dict[Any, float], nodes: list) -> float:
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
    k_phi_field: dict[Any, float] | None = None,
) -> dict[int, float]:
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
    k_phi_field : dict | None
        Precomputed K_φ per node. If None, computed via
        compute_phase_curvature.

    Returns
    -------
    dict[int, float]
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
    variance_by_scale: dict[int, float],
    alpha_hint: float = defaults.K_PHI_ASYMPTOTIC_ALPHA,
) -> dict[str, Any]:
    """Fit power-law exponent α for multiscale K_φ variance decay.

    **Status**: RESEARCH (multiscale analysis support)

    Model
    -----
    var(K_φ) at scale r ~ C / r^α

    Taking logarithms:
        log(var) = log(C) - α * log(r)

    Parameters
    ----------
    variance_by_scale : dict[int, float]
        Mapping from scale r to variance of coarse-grained K_φ
    alpha_hint : float
        Expected value of α for comparison (default from K_PHI_ASYMPTOTIC_ALPHA research)

    Returns
    -------
    dict[str, Any]
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
) -> dict[str, Any]:
    """Assess multiscale safety of K_φ field [RESEARCH].

    **Status**: RESEARCH (safety analysis support)

    Computes coarse-grained K_φ variance across scales, fits power-law
    decay, and returns safety verdict based on fit quality and threshold
    violations.

    Returns
    -------
    dict[str, Any]
        - variance_by_scale: dict[int, float] - computed variances
        - fit: dict - power-law fitting results
        - violations: list[int] - scales with |K_φ| >= K_PHI_CURVATURE_THRESHOLD
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
        if var > defaults.K_PHI_CURVATURE_THRESHOLD**2
    ]  # Phase-wrap threshold: K_φ ≤ π (audit 2026: π is the genuine scale)

    # Assess safety
    safe_by_fit = (
        fit.get("alpha", 0.0) > 0.0 and fit.get("r_squared", 0.0) >= fit_min_r2
    )
    safe_by_tolerance = (alpha_hint is not None) and (len(violations) == 0)
    safe = bool(safe_by_fit or safe_by_tolerance)

    return {
        "variance_by_scale": {int(k): float(v) for k, v in variance_by_scale.items()},
        "fit": fit,
        "violations": violations,
        "safe": safe,
    }


def fit_correlation_length_exponent(
    intensities: np.ndarray,
    xi_c_values: np.ndarray,
    I_c: float = defaults.CRITICAL_INFORMATION_DENSITY,
    min_distance: float = defaults.MIN_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
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
        Critical intensity (calibrated operational scale ≈ 2.015)
    min_distance : float, default=MIN_DISTANCE_THRESHOLD
        Minimum |I - I_c| to avoid divergence noise

    Returns
    -------
    dict[str, Any]
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
    below_mask = (intensities < I_c) & (np.abs(intensities - I_c) > min_distance)
    above_mask = (intensities > I_c) & (np.abs(intensities - I_c) > min_distance)

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
        elif abs(nu_avg - 1.0) < _ISING_2D_EXPONENT_TOLERANCE:
            results["universality_class"] = "ising-2d"

    return results


# ============================================================================
# UNIFIED FIELD MATHEMATICS (Nov 28, 2025) - CANONICAL INTEGRATION
# ============================================================================


def _aligned_field_arrays(nodes, *field_maps):
    """Align complete field maps with the declared graph iteration order."""
    return tuple(
        np.asarray([field[node] for node in nodes], dtype=float) for field in field_maps
    )


def _complex_field_array_view(nodes, psi):
    k_phi = np.asarray([psi[node].real for node in nodes], dtype=float)
    j_phi = np.asarray([psi[node].imag for node in nodes], dtype=float)
    psi_complex = k_phi + 1j * j_phi
    correlation = 0.0
    if len(nodes) > 1 and np.std(k_phi) > 1e-10 and np.std(j_phi) > 1e-10:
        correlation = float(np.corrcoef(k_phi, j_phi)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
    return {
        "nodes": nodes,
        "psi_real": k_phi,
        "psi_imag": j_phi,
        "psi_magnitude": np.abs(psi_complex),
        "psi_phase": np.angle(psi_complex),
        "correlation": correlation,
        "num_nodes": len(nodes),
    }


def compute_complex_geometric_field_arrays(G: Any) -> dict[str, Any]:
    """Read Psi = K_phi + i J_phi as arrays aligned with returned ``nodes``.

    ``nodes`` follows graph iteration order, including incomparable labels.
    The underlying per-node field remains owned by ``physics.unified``.
    Correlation retains the historical zero convention for constant samples.
    """
    from .unified import compute_complex_geometric_field as _psi_dict

    return _complex_field_array_view(tuple(G), _psi_dict(G))


# Backward-compatible alias (prefer compute_complex_geometric_field_arrays)
compute_complex_geometric_field = compute_complex_geometric_field_arrays


def _emergent_field_array_view(nodes, derived):
    names = ("chirality", "symmetry_breaking", "coherence_coupling")
    arrays = _aligned_field_arrays(nodes, *(derived[name] for name in names))
    return {"nodes": nodes, "num_nodes": len(nodes), **dict(zip(names, arrays))}


def compute_emergent_fields(G: Any) -> dict[str, Any]:
    """Read the three composite fields, aligned with returned ``nodes``.

    Capture each required base field once. These are algebraic snapshot
    coordinates; their names do not establish physical emergence.
    """
    from .unified import (
        _capture_structural_fields,
        _chirality_field,
        _coherence_coupling_field,
        _complex_geometric_field,
        _symmetry_breaking_field,
    )

    fields = _capture_structural_fields(G)
    psi = _complex_geometric_field(fields.k_phi, fields.j_phi)
    return _emergent_field_array_view(
        tuple(G),
        {
            "chirality": _chirality_field(
                fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
            ),
            "symmetry_breaking": _symmetry_breaking_field(
                fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
            ),
            "coherence_coupling": _coherence_coupling_field(fields.phi_s, psi),
        },
    )


def _tensor_field_array_view(nodes, derived):
    energy, charge, density = _aligned_field_arrays(
        nodes,
        derived["energy_density"],
        derived["historical_q_density"],
        derived["charge_density"],
    )
    return {
        "nodes": nodes,
        "energy_density": energy,
        "topological_charge": charge,
        "conservation_density": density,
        "conservation_quality": None,
        "conservation_sample_available": False,
        "conservation_scope": "single_snapshot_no_temporal_balance",
        "num_nodes": len(nodes),
    }


def compute_tensor_invariants(G: Any) -> dict[str, Any]:
    """Read quadratic/bilinear snapshot arrays aligned with ``nodes``.

    The legacy ``topological_charge`` is a continuous bilinear coordinate,
    not integer winding. ``conservation_quality`` is retained as ``None``:
    one snapshot cannot measure a temporal balance. Availability and scope
    are explicit; use ``conservation``'s paired snapshots for that assessment.
    """
    from .conservation import _charge_density_from_fields
    from .unified import (
        _capture_structural_fields,
        _energy_density_from_fields,
        _topological_charge,
    )

    fields = _capture_structural_fields(G)
    return _tensor_field_array_view(
        tuple(G),
        {
            "energy_density": _energy_density_from_fields(
                fields.phi_s, fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
            ),
            "historical_q_density": _topological_charge(
                fields.grad_phi, fields.k_phi, fields.j_phi, fields.j_dnfr
            ),
            "charge_density": _charge_density_from_fields(fields.phi_s, fields.k_phi),
        },
    )


def compute_unified_telemetry(G: Any) -> dict[str, Any]:
    """Compute the unified field telemetry suite.

    Provides comprehensive telemetry combining:
    - Canonical diagnostic tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
    - Extended canonical (J_φ, J_ΔNFR)
    - Unified complex field (Ψ = K_φ + i·J_φ)
    - Emergent fields (χ, S, C)
    - Tensor invariants (ε, Q, conservation)
    - Emergent pulse (conservative rhythm: ω_k = √λ_k, beats, vibration energy)

    The canonical/extended blocks are graph-state diagnostics. The ``pulse``
    block belongs to the auxiliary graph-wave model. Their joint presence in
    this dictionary is an API composition and does not identify an engine
    trajectory with the conservative model or make the tetrad a complete state
    observer.

    Args:
        G: TNFR network with the state attributes required by each diagnostic

    Returns:
        dict containing all unified field metrics for production telemetry

    Usage:
        telemetry = compute_unified_telemetry(G)
        correlation = telemetry["complex_field"]["correlation"]  # K_φ ↔ J_φ
        energy = np.mean(telemetry["tensor_invariants"]["energy_density"])

    References:
        - Structural-field scope in docs/STRUCTURAL_FIELDS_TETRAD.md
        - Coherence-length provenance in docs/XI_C_CANONICAL_PROMOTION.md
    """
    from .unified import (
        _complex_geometric_field,
        _StructuralFieldReadout,
        _unified_field_suite_from_fields,
    )

    # One declared snapshot supplies every algebraic view and scalar total.
    canonical_telemetry = compute_structural_telemetry(G)
    nodes = tuple(G)
    captured = _StructuralFieldReadout(
        phi_s=dict(canonical_telemetry["phi_s"]),
        grad_phi=dict(canonical_telemetry["grad_phi"]),
        k_phi=dict(canonical_telemetry["curv_phi"]),
        j_phi=dict(canonical_telemetry["j_phi"]),
        j_dnfr=dict(canonical_telemetry["j_dnfr"]),
    )
    derived = _unified_field_suite_from_fields(captured)
    extended_suite = {
        "phase_current": dict(captured.j_phi),
        "dnfr_flux": dict(captured.j_dnfr),
    }
    complex_field = _complex_field_array_view(
        nodes, _complex_geometric_field(captured.k_phi, captured.j_phi)
    )
    emergent_fields = _emergent_field_array_view(nodes, derived)
    tensor_invariants = _tensor_field_array_view(nodes, derived)
    conservation = dict(derived["conservation_metrics"])

    # Auxiliary symplectic substrate initialized from extracted graph fields.
    try:
        from .symplectic_substrate import (
            PhaseSpacePoint,
            background_potential,
            liouville_divergence,
            substrate_hamiltonian,
        )

        k_phi, j_phi, phi_s, j_dnfr, grad_phi = _aligned_field_arrays(
            nodes,
            captured.k_phi,
            captured.j_phi,
            captured.phi_s,
            captured.j_dnfr,
            captured.grad_phi,
        )
        _pt = PhaseSpacePoint(nodes, k_phi, j_phi, phi_s, j_dnfr, grad_phi)
        symplectic_substrate = {
            "phase_space_dimension": _pt.dimension,
            "hamiltonian": substrate_hamiltonian(_pt),
            "background_potential": background_potential(_pt),
            "liouville_divergence": liouville_divergence(_pt),
        }
    except Exception:
        symplectic_substrate = {}

    # Auxiliary graph-wave pulse -- the resonant
    # spectrum omega_k = sqrt(lambda_k), the dominant beat and the
    # self-similar signature), computed from the structural spectrum
    # (structural_diffusion.py). It is not an inferred engine trajectory.
    try:
        from .structural_diffusion import compute_emergent_pulse

        pulse = compute_emergent_pulse(G)
    except Exception:
        pulse = {}

    # Per-NFR resonance reads stored capacity and current phase alignment
    # (local synchrony and collective Kuramoto R). It does not measure an
    # oscillation period or derive a collective clock from synchronization.
    try:
        from .structural_diffusion import compute_nodal_pulse

        resonance = compute_nodal_pulse(G)
    except Exception:
        resonance = {}

    return {
        "canonical": canonical_telemetry,
        "extended_canonical": extended_suite,
        "complex_field": complex_field,
        "emergent_fields": emergent_fields,
        "tensor_invariants": tensor_invariants,
        "conservation": conservation,
        "symplectic_substrate": symplectic_substrate,
        "pulse": pulse,
        "resonance": resonance,
        "unified_field_version": "1.0.0",  # Track implementation version
    }


# ============================================================================
# SELF-OPTIMIZING MATHEMATICAL ANALYSIS (NEW - Nov 28, 2025)
# ============================================================================


def _field_magnitude_summary(telemetry: dict[str, Any]) -> dict[str, float | None]:
    """Reduce existing arrays; absent/empty samples have no measured mean.

    These mean absolute magnitudes feed configured advisory cuts only. They
    are descriptive statistics, not new state variables or dynamical rules.
    """
    from ..metrics.common import finite_mean_absolute

    channels = (
        ("psi_magnitude_mean", "complex_field", "psi_magnitude"),
        ("chirality_magnitude_mean", "emergent_fields", "chirality"),
        ("symmetry_breaking_magnitude_mean", "emergent_fields", "symmetry_breaking"),
        ("energy_density_mean", "tensor_invariants", "energy_density"),
    )
    summary = {}
    for name, block, field in channels:
        values = telemetry.get(block, {}).get(field, ())
        summary[name] = (
            finite_mean_absolute(values, name=field) if len(values) else None
        )
    return summary


def analyze_optimization_potential(G: Any) -> dict[str, Any]:
    """
    Analyze mathematical optimization potential using unified field analysis.

    Combines unified field telemetry with mathematical structure analysis
    to identify optimization opportunities automatically.

    Returns:
        dict containing:
        - field_analysis: Unified field characteristics
        - mathematical_insights: Structural properties for optimization
        - optimization_recommendations: Specific optimization strategies
        - predicted_improvements: Compatibility keys, ``None`` until measured
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        return {
            "error": "Self-optimizing engine not available",
            "field_analysis": {},
            "mathematical_insights": {},
            "optimization_recommendations": [],
            "predicted_improvements": {},
            "performance_evidence": "not_measured",
        }

    # Get unified field telemetry
    unified_telemetry = compute_unified_telemetry(G)
    magnitudes = _field_magnitude_summary(unified_telemetry)

    # Create self-optimizing engine
    engine = TNFRSelfOptimizingEngine(
        optimization_objective=OptimizationObjective.BALANCE_ALL
    )

    # Analyze mathematical landscape
    mathematical_insights = engine.analyze_mathematical_optimization_landscape(
        G, "field_computation"
    )

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
    chirality_magnitude = magnitudes["chirality_magnitude_mean"]

    if (
        chirality_magnitude is not None
        and chirality_magnitude > defaults.CHIRALITY_THRESHOLD
    ):
        field_optimization_hints.append("use_chirality_optimization")

    # Tensor invariant analysis
    avg_energy = magnitudes["energy_density_mean"]
    if avg_energy is not None:
        if avg_energy > defaults.HIGH_ENERGY_THRESHOLD:
            field_optimization_hints.append("use_high_energy_optimization")
        elif avg_energy < defaults.LOW_ENERGY_THRESHOLD:
            field_optimization_hints.append("use_low_energy_optimization")

    return {
        "field_analysis": unified_telemetry,
        "field_magnitude_summary": magnitudes,
        "mathematical_insights": mathematical_insights,
        "optimization_recommendations": field_optimization_hints,
        "predicted_improvements": {
            "field_correlation_speedup": None,
            "chirality_memory_reduction": None,
            "energy_computation_factor": None,
        },
        "performance_evidence": "not_measured",
    }


def recommend_field_optimization_strategy(
    G: Any, operation_type: str = "unified_telemetry"
) -> dict[str, Any]:
    """
    Recommend optimization strategy based on unified field analysis.

    Args:
        G: TNFR network graph
        operation_type: type of field operation to optimize

    Returns:
        Optimization strategy recommendations with mathematical justification
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        return {
            "error": "Self-optimizing engine not available",
            "recommendations": [],
            "strategy": "fallback_standard",
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
    magnitudes = analysis.get("field_magnitude_summary")
    if magnitudes is None:
        magnitudes = _field_magnitude_summary(field_analysis)

    psi_mean = magnitudes["psi_magnitude_mean"]
    if psi_mean is not None and psi_mean > defaults.COMPLEX_FIELD_THRESHOLD:
        field_specific_strategies.append("prioritize_complex_field_computation")

    symmetry_mean = magnitudes["symmetry_breaking_magnitude_mean"]
    if (
        symmetry_mean is not None
        and symmetry_mean > defaults.SYMMETRY_BREAKING_THRESHOLD
    ):
        field_specific_strategies.append("use_symmetry_breaking_acceleration")

    return {
        "unified_field_analysis": field_analysis,
        "field_magnitude_summary": magnitudes,
        "mathematical_recommendations": recommendations.recommended_strategies,
        "field_specific_strategies": field_specific_strategies,
        "predicted_speedups": recommendations.predicted_speedups,
        "optimization_insights": recommendations.mathematical_insights,
        "recommended_strategy": (
            field_specific_strategies[0]
            if field_specific_strategies
            else "standard_computation"
        ),
    }


def auto_optimize_field_computation(G: Any, **kwargs) -> dict[str, Any]:
    """
    Analyze field computation and execute the available advisory path.

    Recommendations are advisory. A strategy is not reported as applied unless a
    distinct measured execution path exists.

    Args:
        G: TNFR network graph
        **kwargs: Additional parameters for optimization

    Returns:
        Field telemetry, advisory details, and explicit measurement provenance
    """
    if not _SELF_OPTIMIZING_AVAILABLE:
        return {
            "result": compute_unified_telemetry(G),
            "optimization_applied": False,
            "strategy_used": "fallback_standard",
            "performance_improvement": None,
            "performance_evidence": "not_measured",
            "error": "Self-optimizing engine not available",
        }

    start_time = time.perf_counter()
    del kwargs

    try:
        recommendations = recommend_field_optimization_strategy(G, "unified_telemetry")
        result = recommendations.get("unified_field_analysis")
        if not result:
            result = compute_unified_telemetry(G)

        return {
            "result": result,
            "optimization_applied": False,
            "strategy_used": "advisory_only",
            "performance_improvement": None,
            "performance_evidence": "not_measured",
            "total_time": time.perf_counter() - start_time,
            "recommendations": recommendations,
            "optimization_details": {
                "message": "No alternate field-computation kernel is wired",
                "learning_updated": False,
            },
        }

    except Exception as e:
        return {
            "result": compute_unified_telemetry(G),
            "optimization_applied": False,
            "strategy_used": "fallback_error",
            "performance_improvement": None,
            "performance_evidence": "not_measured",
            "error": str(e),
            "total_time": time.perf_counter() - start_time,
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
