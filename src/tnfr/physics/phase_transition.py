r"""TNFR phase-transition diagnostics from structural-field symmetry.

The symmetry-breaking candidate

    𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²)

and chirality ``χ`` provide reproducible read-outs for controlled sweeps. The
operational phase labels compare the signed global means ``|⟨𝒮⟩|`` and
``|⟨χ⟩|`` with their finite-network spatial spreads. The separate magnitudes
``⟨|𝒮|⟩`` and ``⟨|χ|⟩`` describe local activity; they cannot establish global
symmetry breaking or homochirality because opposite signs may cancel.

This module measures susceptibility peaks, correlation length and an effective
time-series power-law exponent. It does not derive a universal second-order
transition, a universal exponent or divergence of correlation length. Those
claims require a declared control parameter, finite-size scaling, uncertainty
intervals and replication across graph families.

The implementation remains anchored to the nodal equation through the unified
fields. Its NON_LIFE/CRITICAL/LIFE names are operational classifications, not
biological or metaphysical conclusions.

See Also
--------
unified.py : 𝒮 and χ field computations (authoritative)
life.py    : Autopoietic coefficient A (static threshold)
cell.py    : Cellular criteria (static thresholds)
gauge.py   : U(1) gauge structure of Ψ = K_φ + i·J_φ
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..mathematics.unified_numerical import np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from .canonical import estimate_coherence_length

# Delegate to authoritative field computations (single source of truth)
from .unified import compute_chirality_field, compute_symmetry_breaking_field

# ============================================================================
# EMERGENT CLASSIFICATION — standardized spatial imbalance
# ============================================================================
# Audit 2026: classification uses |mean| / sqrt(Var/N).  Graph nodes are
# coupled and therefore are not independent samples in general.  The ratio is
# an operational standardized spatial imbalance, not a hypothesis-test
# z-score unless an external sampling model justifies that interpretation.
# The public ``symmetry_zscore`` name is retained for API compatibility.

#: Operational classification policy: a field is labelled broken when its
#: standardized spatial imbalance exceeds one.  This selected policy is not a
#: universal critical threshold.
Z_SIGNIFICANCE: float = 1.0


def symmetry_zscore(mean_abs: float, variance: float, n: int) -> float:
    r"""Return the standardized spatial imbalance ``|mean|/sqrt(Var/N)``.

    The denominator is the independent-sample standard error algebraically,
    but TNFR graph nodes are usually correlated.  Accordingly, this function
    supplies a deterministic classifier input rather than a p-value or a
    statistical significance claim.  It returns 0.0 for a uniform zero field
    and +∞ for a uniform non-zero field.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    mean_value = float(mean_abs)
    variance_value = float(variance)
    if not math.isfinite(mean_value) or mean_value < 0.0:
        raise ValueError("mean_abs must be finite and non-negative")
    if not math.isfinite(variance_value) or variance_value < 0.0:
        raise ValueError("variance must be finite and non-negative")
    if n == 0:
        return 0.0
    se = math.sqrt(variance_value / n)
    if se == 0.0:
        return 0.0 if mean_value == 0.0 else math.inf
    return mean_value / se


# ============================================================================
# DATA STRUCTURES
# ============================================================================


class Phase(Enum):
    """Legacy labels for an operational signed-imbalance classification."""

    NON_LIFE = "non_life"  # Both standardized imbalances below the policy cut
    CRITICAL = "critical"  # Order imbalance without chirality imbalance
    LIFE = "life"  # Both standardized imbalances above the policy cuts


@dataclass
class PhaseTransitionTelemetry:
    r"""Container for operational finite-state transition diagnostics.

    Captures selected structural read-outs for testing a candidate symmetry-
    breaking transition: order parameter, chirality, susceptibility, coherence
    length, and a finite time-series exponent fit.

    All quantities derive from the nodal equation via unified fields.

    Attributes
    ----------
    times : list[float]
        Strictly increasing structural-time coordinates.
    order_parameter : np.ndarray
        Network-averaged signed imbalance ⟨𝒮⟩(t).
    order_parameter_abs : np.ndarray
        |⟨𝒮⟩|(t) — magnitude of the signed global order parameter.
    chirality_mean : np.ndarray
        Network-averaged signed chirality ⟨χ⟩(t).
    chirality_abs_mean : np.ndarray
        ⟨|χ|⟩(t) — mean absolute chirality (non-zero even without
        preferred handedness if local chirality exists).
    susceptibility : np.ndarray
        χ_𝒮(t) = N · Var(𝒮) — finite-sample fluctuation susceptibility.
    coherence_length : np.ndarray
        ξ_C(t) — measured spatial correlation scale.
    phase_classification : list[Phase]
        Phase assignment per time step.
    order_zscore : np.ndarray
        Operational standardized spatial imbalance for 𝒮 at every step.
    chirality_zscore : np.ndarray
        Operational standardized spatial imbalance for χ at every step.
    node_count : np.ndarray
        Number of nodes at every step. A varying count makes raw
        susceptibility peaks unsuitable for finite-size inference.
    transition_time : float | None
        First structural time where the order z-score crosses one, with linear
        interpolation. The resulting phase may be CRITICAL or LIFE.
    critical_time : float | None
        Time of maximum sampled susceptibility.
    measured_exponent : float | None
        Effective time-series exponent from |⟨𝒮⟩| ~ |t − t_c|^{β}. It is
        protocol-dependent and is not a thermodynamic exponent unless time is
        explicitly related to a declared control parameter.
    exponent_fit_r_squared : float | None
        Coefficient of determination R² for the power-law fit.
    """

    times: list[float]
    order_parameter: np.ndarray
    order_parameter_abs: np.ndarray
    chirality_mean: np.ndarray
    chirality_abs_mean: np.ndarray
    susceptibility: np.ndarray
    coherence_length: np.ndarray
    phase_classification: list[Phase] = field(default_factory=list)
    transition_time: float | None = None
    critical_time: float | None = None
    measured_exponent: float | None = None
    exponent_fit_r_squared: float | None = None
    order_zscore: np.ndarray = field(default_factory=lambda: np.array([]))
    chirality_zscore: np.ndarray = field(default_factory=lambda: np.array([]))
    node_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


@dataclass
class PhaseSnapshot:
    """Candidate-transition read-outs for a single graph state.

    Lighter-weight alternative to :class:`PhaseTransitionTelemetry` for
    point-in-time analysis without a time series.
    """

    order_parameter: float  # ⟨𝒮⟩
    order_parameter_abs: float  # |⟨𝒮⟩|
    chirality_mean: float  # ⟨χ⟩
    chirality_abs_mean: float  # ⟨|χ|⟩
    susceptibility: float  # N · Var(𝒮)
    coherence_length: float  # ξ_C
    phase: Phase  # Classified phase
    has_homochirality: bool  # Operational chirality classification
    order_zscore: float = 0.0  # Standardized spatial imbalance of 𝒮
    chirality_zscore: float = 0.0  # Standardized spatial imbalance of χ
    node_count: int = 0


# ============================================================================
# CORE COMPUTATIONS
# ============================================================================


def compute_order_parameter(G: Any) -> dict[str, float]:
    r"""Compute the selected signed-imbalance order-parameter read-outs.

    Returns network statistics of 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²):

    - ⟨𝒮⟩ = (1/N) Σ_i 𝒮(i)  — mean (order parameter)
    - ⟨|𝒮|⟩ = (1/N) Σ_i |𝒮(i)|  — mean magnitude
    - Var(𝒮) = ⟨𝒮²⟩ − ⟨𝒮⟩²  — variance
    - χ_𝒮 = N · Var(𝒮)  — finite-sample susceptibility diagnostic

    Parameters
    ----------
    G : NetworkX graph
        Network with structural attributes (phase/theta, delta_nfr).

    Returns
    -------
    dict[str, float]
        Keys: 'mean', 'abs_mean', 'variance', 'susceptibility',
        'max', 'min', 'n_nodes'.
    """
    S_field = compute_symmetry_breaking_field(G)
    values = np.array(list(S_field.values()))
    N = len(values)
    if N == 0:
        return {
            "mean": 0.0,
            "abs_mean": 0.0,
            "variance": 0.0,
            "susceptibility": 0.0,
            "max": 0.0,
            "min": 0.0,
            "n_nodes": 0,
        }
    mean = float(np.mean(values))
    return {
        "mean": mean,
        "abs_mean": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "susceptibility": float(N * np.var(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "n_nodes": N,
    }


def compute_chirality_statistics(G: Any) -> dict[str, float]:
    r"""Compute finite-state chirality-imbalance statistics.

    Chirality χ = |∇φ|·K_φ − J_φ·J_ΔNFR.
    A non-zero sample mean records finite-state chirality imbalance; a symmetry-
    breaking or homochirality claim requires an independent sampling model.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    dict[str, float]
        Keys: 'mean', 'abs_mean', 'variance', 'max_abs', 'n_nodes'.
    """
    chi_field = compute_chirality_field(G)
    values = np.array(list(chi_field.values()))
    N = len(values)
    if N == 0:
        return {
            "mean": 0.0,
            "abs_mean": 0.0,
            "variance": 0.0,
            "max_abs": 0.0,
            "n_nodes": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "max_abs": float(np.max(np.abs(values))),
        "n_nodes": N,
    }


def classify_phase(
    order_z: float,
    chirality_z: float,
) -> Phase:
    r"""Classify the structural phase from emergent z-scores.

    The phase is decided by an operational standardized spatial imbalance.
    With ``order_z = |⟨𝒮⟩|/sqrt(Var(𝒮)/N)`` and the analogous chirality ratio
    (via :func:`symmetry_zscore`):

    - **NON_LIFE**: ``order_z ≤ 1`` — the signed mean does not exceed the
      selected standardized-imbalance cut.
    - **LIFE**: ``order_z > 1`` AND ``chirality_z > 1`` — both operational
      ratios lie above the selected cut.
    - **CRITICAL**: ``order_z > 1`` but ``chirality_z ≤ 1`` — order ratio
      above the cut without a chirality ratio above it (intermediate).

    Parameters
    ----------
    order_z : float
        Standardized spatial imbalance of |⟨𝒮⟩|.
    chirality_z : float
        Standardized spatial imbalance of |⟨χ⟩|.

    Returns
    -------
    Phase
        NON_LIFE, CRITICAL, or LIFE.
    """
    order_value = float(order_z)
    chirality_value = float(chirality_z)
    if math.isnan(order_value) or order_value < 0.0:
        raise ValueError("order_z must be non-negative and not NaN")
    if math.isnan(chirality_value) or chirality_value < 0.0:
        raise ValueError("chirality_z must be non-negative and not NaN")
    if order_value <= Z_SIGNIFICANCE:
        return Phase.NON_LIFE
    if chirality_value > Z_SIGNIFICANCE:
        return Phase.LIFE
    return Phase.CRITICAL


def capture_phase_snapshot(G: Any) -> PhaseSnapshot:
    """Capture instantaneous candidate-transition diagnostics.

    Parameters
    ----------
    G : NetworkX graph
        Network with structural attributes.

    Returns
    -------
    PhaseSnapshot
        Point-in-time operational structural read-outs.
    """
    op = compute_order_parameter(G)
    chi = compute_chirality_statistics(G)
    xi = estimate_coherence_length(G)
    if math.isnan(xi):
        xi = 0.0

    n_nodes = op["n_nodes"]
    order_z = symmetry_zscore(abs(op["mean"]), op["variance"], n_nodes)
    chirality_z = symmetry_zscore(abs(chi["mean"]), chi["variance"], n_nodes)
    phase = classify_phase(order_z, chirality_z)

    return PhaseSnapshot(
        order_parameter=op["mean"],
        order_parameter_abs=abs(op["mean"]),
        chirality_mean=chi["mean"],
        chirality_abs_mean=chi["abs_mean"],
        susceptibility=op["susceptibility"],
        coherence_length=xi,
        phase=phase,
        has_homochirality=chirality_z > Z_SIGNIFICANCE,
        order_zscore=order_z,
        chirality_zscore=chirality_z,
        node_count=int(n_nodes),
    )


# ============================================================================
# TIME-SERIES ANALYSIS — PHASE TRANSITION DETECTION
# ============================================================================


def detect_phase_transition(
    graph_sequence: Sequence[Any],
    times: Sequence[float],
) -> PhaseTransitionTelemetry:
    r"""Measure and classify candidate transition indicators in graph states.

    Computes the order parameter ⟨𝒮⟩, chirality ⟨χ⟩, susceptibility,
    and coherence length at each time step, then:

    1. Classifies each snapshot as NON_LIFE / CRITICAL / LIFE.
    2. Records the operational threshold-crossing time.
    3. Records the sampled susceptibility-peak time.
    4. Fits an effective exponent after the declared sampled peak time from
       |⟨𝒮⟩| ~ |t − t_c|^{p_fit}.

    Parameters
    ----------
    graph_sequence : Sequence[nx.Graph]
        Time-ordered TNFR network states.
    times : Sequence[float]
        Finite, strictly increasing structural times corresponding one-to-one
        with the graph states.

    Returns
    -------
    PhaseTransitionTelemetry
        Operational finite-state diagnostics including a measured exponent fit.
    """
    times = _validate_times(times)
    n = len(graph_sequence)
    if len(times) != n:
        raise ValueError(
            "graph_sequence and times must contain the same number of entries"
        )

    order_param = np.zeros(n)
    order_param_abs = np.zeros(n)
    chi_mean = np.zeros(n)
    chi_abs_mean = np.zeros(n)
    suscept = np.zeros(n)
    xi_c = np.zeros(n)
    order_z_series = np.zeros(n)
    chirality_z_series = np.zeros(n)
    node_counts = np.zeros(n, dtype=int)
    phases: list[Phase] = []

    for i, G in enumerate(graph_sequence):
        op = compute_order_parameter(G)
        chi = compute_chirality_statistics(G)
        xi = estimate_coherence_length(G)
        if math.isnan(xi):
            xi = 0.0

        order_param[i] = op["mean"]
        order_param_abs[i] = abs(op["mean"])
        chi_mean[i] = chi["mean"]
        chi_abs_mean[i] = chi["abs_mean"]
        suscept[i] = op["susceptibility"]
        xi_c[i] = xi

        n_nodes = op["n_nodes"]
        order_z = symmetry_zscore(abs(op["mean"]), op["variance"], n_nodes)
        chirality_z = symmetry_zscore(abs(chi["mean"]), chi["variance"], n_nodes)
        order_z_series[i] = order_z
        chirality_z_series[i] = chirality_z
        node_counts[i] = int(n_nodes)

        phase = classify_phase(order_z, chirality_z)
        phases.append(phase)

    # Interpolate the finite standardized-imbalance crossing. The cut is an
    # operational classifier policy, not a statistical significance level.
    transition_time = _find_crossing_time(times, order_z_series, Z_SIGNIFICANCE)

    # --- Candidate time: sampled peak susceptibility ---
    critical_time: float | None = None
    if n > 0 and np.max(suscept) > 0:
        peak_idx = int(np.argmax(suscept))
        critical_time = times[peak_idx]

    # --- Candidate effective-exponent fit ---
    measured_exp, r_squared = _fit_critical_exponent(
        times, order_param_abs, critical_time
    )

    return PhaseTransitionTelemetry(
        times=times,
        order_parameter=order_param,
        order_parameter_abs=order_param_abs,
        chirality_mean=chi_mean,
        chirality_abs_mean=chi_abs_mean,
        susceptibility=suscept,
        coherence_length=xi_c,
        phase_classification=phases,
        transition_time=transition_time,
        critical_time=critical_time,
        measured_exponent=measured_exp,
        exponent_fit_r_squared=r_squared,
        order_zscore=order_z_series,
        chirality_zscore=chirality_z_series,
        node_count=node_counts,
    )


# ============================================================================
# CRITICAL EXPONENT MEASUREMENT
# ============================================================================


def fit_critical_exponent(
    times: Sequence[float],
    order_parameter_abs: np.ndarray,
    critical_time: float | None = None,
) -> dict[str, float | None]:
    r"""Fit an effective time-series exponent from order-parameter scaling.

    For the candidate fit:

        |⟨𝒮⟩| ~ |t − t_c|^{p_fit}

    Fits p_fit via log-log linear regression using only samples after the
    declared candidate time with positive distance and order magnitude. At
    least three such samples are required; earlier samples are never
    substituted. This is not a
    universal critical exponent unless the supplied time coordinate is a
    declared control-parameter distance.

    Parameters
    ----------
    times : Sequence[float]
        Structural times.
    order_parameter_abs : np.ndarray
        |⟨𝒮⟩| time series.
    critical_time : float | None
        Declared candidate or sampled-peak time t_c. If None, no exponent is
        fitted.

    Returns
    -------
    dict[str, float | None]
        'exponent': fitted p_fit, 'r_squared': R². Values are None if the fit
        is impossible (insufficient data). There is no 'theoretical' value:
        the exponent is a measured observable, not a derived constant.
    """
    validated_times = _validate_times(times)
    order_abs = np.asarray(order_parameter_abs, dtype=float)
    if order_abs.ndim != 1:
        raise ValueError("order_parameter_abs must be a one-dimensional series")
    if len(validated_times) != len(order_abs):
        raise ValueError(
            "times and order_parameter_abs must contain the same number of entries"
        )
    if not np.all(np.isfinite(order_abs)):
        raise ValueError("order_parameter_abs must contain only finite values")
    if np.any(order_abs < 0.0):
        raise ValueError("order_parameter_abs cannot contain negative values")
    parsed_critical_time: float | None = None
    if critical_time is not None:
        try:
            parsed_critical_time = float(critical_time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("critical_time must be finite when provided") from exc
        if not math.isfinite(parsed_critical_time):
            raise ValueError("critical_time must be finite when provided")

    exponent, r_sq = _fit_critical_exponent(
        validated_times, order_abs, parsed_critical_time
    )
    return {
        "exponent": exponent,
        "r_squared": r_sq,
    }


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _validate_times(times: Sequence[float]) -> list[float]:
    """Return finite, strictly increasing structural-time coordinates."""
    validated = [float(value) for value in times]
    if not all(math.isfinite(value) for value in validated):
        raise ValueError("times must contain only finite values")
    if any(right <= left for left, right in zip(validated, validated[1:])):
        raise ValueError("times must be strictly increasing")
    return validated


def _find_crossing_time(
    times: list[float],
    values: np.ndarray,
    threshold: float,
) -> float | None:
    """Find the first time where *values* crosses *threshold* upward.

    Uses linear interpolation between adjacent time steps for precision.
    """
    n = len(values)
    if n < 2:
        return None

    for i in range(n - 1):
        if values[i] <= threshold < values[i + 1]:
            if not math.isfinite(float(values[i + 1])):
                return times[i + 1]
            # Linear interpolation
            dv = values[i + 1] - values[i]
            if abs(dv) < 1e-15:
                return times[i]
            alpha = (threshold - values[i]) / dv
            return times[i] + alpha * (times[i + 1] - times[i])

    # Already above threshold from start
    if values[0] > threshold:
        return times[0]

    return None


def _fit_critical_exponent(
    times: list[float],
    order_abs: np.ndarray,
    t_c: float | None,
) -> tuple[float | None, float | None]:
    r"""Fit |⟨𝒮⟩| ~ |t − t_c|^{p_fit} via log-log regression.

    Uses data points after the declared candidate time where both |t − t_c|
    and |⟨𝒮⟩| are positive. Returns (exponent, R²) or (None, None).
    """
    if t_c is None:
        return None, None

    t_arr = np.array(times, dtype=float)
    dt = np.abs(t_arr - t_c)

    # Select post-candidate points with non-trivial order parameter.
    mask = (dt > 1e-12) & (order_abs > 1e-15) & (t_arr >= t_c)
    if np.sum(mask) < 3:
        return None, None

    log_dt = np.log(dt[mask])
    log_S = np.log(order_abs[mask])

    # Linear regression: log|S| = p_fit · log|t − t_c| + const
    A = np.vstack([log_dt, np.ones_like(log_dt)]).T
    try:
        result = np.linalg.lstsq(A, log_S, rcond=None)
        coeffs = result[0]
        exponent = float(coeffs[0])

        # R² computation
        predicted = A @ coeffs
        ss_res = float(np.sum((log_S - predicted) ** 2))
        ss_tot = float(np.sum((log_S - np.mean(log_S)) ** 2))
        if ss_tot <= 1e-15:
            r_squared = 1.0 if ss_res <= 1e-15 else 0.0
        else:
            r_squared = 1.0 - ss_res / ss_tot

        return exponent, r_squared
    except (np.linalg.LinAlgError, ValueError):
        return None, None


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Enums & dataclasses
    "Phase",
    "PhaseTransitionTelemetry",
    "PhaseSnapshot",
    # Emergent classification
    "Z_SIGNIFICANCE",
    "symmetry_zscore",
    # Core computations
    "compute_order_parameter",
    "compute_chirality_statistics",
    "classify_phase",
    "capture_phase_snapshot",
    # Time-series detection
    "detect_phase_transition",
    # Critical exponent
    "fit_critical_exponent",
]
