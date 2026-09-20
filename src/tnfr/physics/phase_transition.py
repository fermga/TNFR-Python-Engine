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

The implementation reads the shared structural fields; it does not select an
operator or supply an evolution law. Its NON_LIFE/CRITICAL/LIFE names are
operational classifications, not biological or metaphysical conclusions.
In exact arithmetic, duplicating disconnected copies multiplies these
imbalances by sqrt(copy count), so a label can change without changing any
component's local dynamics. The labels do not certify autonomous formation.

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

from .._exact_time import finite_represented_real
from ..mathematics.unified_numerical import np
from ..metrics.common import (
    finite_mean,
    finite_mean_absolute,
    finite_pearson_correlation,
    finite_population_std,
)

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from .canonical import (
    CoherenceLengthEstimate,
    estimate_coherence_length_with_provenance,
)

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
    mean_value = finite_represented_real(mean_abs, "mean_abs")[0]
    variance_value = finite_represented_real(variance, "variance")[0]
    if not math.isfinite(mean_value) or mean_value < 0.0:
        raise ValueError("mean_abs must be finite and non-negative")
    if not math.isfinite(variance_value) or variance_value < 0.0:
        raise ValueError("variance must be finite and non-negative")
    if n == 0:
        return 0.0
    return _standardized_imbalance(mean_value, math.sqrt(variance_value), n)


def _standardized_imbalance(mean_abs: float, std: float, n: int) -> float:
    """Evaluate the selected ratio without dividing a tiny variance by N."""
    if n == 0 or mean_abs == 0.0:
        return 0.0
    if std == 0.0:
        return math.inf
    mean_mantissa, mean_exponent = math.frexp(mean_abs)
    std_mantissa, std_exponent = math.frexp(std)
    try:
        coefficient = mean_mantissa / std_mantissa * math.sqrt(n)
    except OverflowError as exc:
        raise ValueError("n exceeds the finite numerical count range") from exc
    try:
        return math.ldexp(coefficient, mean_exponent - std_exponent)
    except OverflowError:
        return math.inf


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

    These defined diagnostics reuse the shared structural fields. Their
    selection and finite-series estimators are not consequences of the
    nodal equation alone, nor do they supply an evolution law.

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
        ξ_C(t) — static fit or tagged spectral fallback; NaN when unavailable.
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
        interpolation, or the first observation if already above the cut.
        Initial occupation is not evidence of a newly observed transition.
        The resulting phase may be CRITICAL or LIFE.
    critical_time : float | None
        Time of maximum sampled susceptibility.
    measured_exponent : float | None
        Effective time-series exponent from |⟨𝒮⟩| ~ |t − t_c|^{β}. It is
        protocol-dependent and is not a thermodynamic exponent unless time is
        explicitly related to a declared control parameter.
    exponent_fit_r_squared : float | None
        Coefficient of determination R² for the power-law fit.
    coherence_length_provenance : list[CoherenceLengthEstimate]
        Per-state static-fit, spectral-fallback or unavailable evidence.
        Fit lengths and spectral scales have different declared units.
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
    coherence_length_provenance: list[CoherenceLengthEstimate] = field(
        default_factory=list
    )

    @property
    def coherence_length_available(self) -> np.ndarray:
        """Availability of either estimator, without identifying their units."""
        return np.isfinite(self.coherence_length) & (self.coherence_length > 0.0)


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
    coherence_length: float  # ξ_C, with NaN for an unavailable estimator
    phase: Phase  # Classified phase
    has_homochirality: bool  # Operational chirality classification
    order_zscore: float = 0.0  # Standardized spatial imbalance of 𝒮
    chirality_zscore: float = 0.0  # Standardized spatial imbalance of χ
    node_count: int = 0
    coherence_length_provenance: CoherenceLengthEstimate | None = None

    @property
    def coherence_length_available(self) -> bool:
        """Whether a finite positive fit or spectral estimate is available."""
        return math.isfinite(self.coherence_length) and self.coherence_length > 0.0


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
        'max', 'min', 'n_nodes', 'standardized_imbalance'. The imbalance is
        computed before squaring the field scale; a rounded zero variance
        must not make a nonconstant sample appear uniform. Variance and
        susceptibility may round below binary64 range; overflow is rejected.
        Susceptibility combines N before final rounding, so its represented
        value need not equal N times the separately rounded variance.
    """
    S_field = compute_symmetry_breaking_field(G)
    statistics, std = _field_statistics(S_field.values(), "order parameter")
    statistics["susceptibility"] = _squared_spread(
        std, statistics["n_nodes"], "susceptibility"
    )
    return statistics


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
        Keys: 'mean', 'abs_mean', 'variance', 'max_abs', 'n_nodes',
        'standardized_imbalance'. The selected imbalance retains variation
        even when the dimensional variance rounds below binary64 range.
    """
    chi_field = compute_chirality_field(G)
    statistics, _ = _field_statistics(chi_field.values(), "chirality")
    statistics["max_abs"] = max(abs(statistics.pop("min")), abs(statistics.pop("max")))
    return statistics


def _squared_spread(std: float, count: int, name: str) -> float:
    """Round count*std**2 once its scale has been combined, or reject overflow."""
    mantissa, exponent = math.frexp(std)
    try:
        return math.ldexp(count * mantissa * mantissa, 2 * exponent)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the finite numerical range") from exc


def _field_statistics(values, name: str) -> tuple[dict[str, float], float]:
    """Shared signed-field reduction; rescaling is numerical, not a new law."""
    samples = tuple(finite_represented_real(value, name)[0] for value in values)
    count = len(samples)
    magnitude = max(map(abs, samples), default=0.0)
    exponent = math.frexp(magnitude)[1]
    # Power-of-two scaling preserves adjacent large inputs and rescales wholly
    # subnormal fields before mean/variance materialization. Widely separated
    # scales still undergo ordinary binary64 rounding in the normalized chart.
    normalized = tuple(math.ldexp(value, -exponent) for value in samples)
    normalized_mean = finite_mean(normalized, name=name)
    normalized_std = finite_population_std(normalized, name=name)
    std = finite_population_std(samples, name=name)
    return {
        "mean": finite_mean(samples, name=name),
        "abs_mean": finite_mean_absolute(samples, name=name),
        "variance": _squared_spread(std, 1, "variance"),
        "max": max(samples, default=0.0),
        "min": min(samples, default=0.0),
        "n_nodes": count,
        "standardized_imbalance": _standardized_imbalance(
            abs(normalized_mean), normalized_std, count
        ),
    }, std


def classify_phase(
    order_z: float,
    chirality_z: float,
) -> Phase:
    r"""Classify a snapshot using the configured spatial-imbalance cut.

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
    # Positive infinity represents a constant nonzero field. Other inputs
    # follow strict raw-real admission rather than bool/text coercion.
    order_value = _classification_ratio(order_z, "order_z")
    chirality_value = _classification_ratio(chirality_z, "chirality_z")
    if math.isnan(order_value) or order_value < 0.0:
        raise ValueError("order_z must be non-negative and not NaN")
    if math.isnan(chirality_value) or chirality_value < 0.0:
        raise ValueError("chirality_z must be non-negative and not NaN")
    if order_value <= Z_SIGNIFICANCE:
        return Phase.NON_LIFE
    if chirality_value > Z_SIGNIFICANCE:
        return Phase.LIFE
    return Phase.CRITICAL


def _classification_ratio(value, name: str) -> float:
    if isinstance(value, (float, np.floating)) and value == math.inf:
        return math.inf
    return finite_represented_real(value, name)[0]


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
    xi = estimate_coherence_length_with_provenance(G)

    n_nodes = op["n_nodes"]
    order_z = op["standardized_imbalance"]
    chirality_z = chi["standardized_imbalance"]
    phase = classify_phase(order_z, chirality_z)

    return PhaseSnapshot(
        order_parameter=op["mean"],
        order_parameter_abs=abs(op["mean"]),
        chirality_mean=chi["mean"],
        chirality_abs_mean=chi["abs_mean"],
        susceptibility=op["susceptibility"],
        coherence_length=xi.value,
        phase=phase,
        has_homochirality=chirality_z > Z_SIGNIFICANCE,
        order_zscore=order_z,
        chirality_zscore=chirality_z,
        node_count=int(n_nodes),
        coherence_length_provenance=xi,
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
        Time-ordered supplied TNFR network states. This reader does not verify
        that an engine trajectory connects the supplied states.
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
    xi_provenance: list[CoherenceLengthEstimate] = []

    for i, G in enumerate(graph_sequence):
        snapshot = capture_phase_snapshot(G)
        order_param[i] = snapshot.order_parameter
        order_param_abs[i] = snapshot.order_parameter_abs
        chi_mean[i] = snapshot.chirality_mean
        chi_abs_mean[i] = snapshot.chirality_abs_mean
        suscept[i] = snapshot.susceptibility
        xi_c[i] = snapshot.coherence_length
        order_z_series[i] = snapshot.order_zscore
        chirality_z_series[i] = snapshot.chirality_zscore
        node_counts[i] = snapshot.node_count
        phases.append(snapshot.phase)
        assert snapshot.coherence_length_provenance is not None
        xi_provenance.append(snapshot.coherence_length_provenance)

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
        coherence_length_provenance=xi_provenance,
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
    raw_order = np.asarray(order_parameter_abs, dtype=object)
    if raw_order.ndim != 1:
        raise ValueError("order_parameter_abs must be a one-dimensional series")
    order_abs = np.array(
        [
            finite_represented_real(value, "order_parameter_abs")[0]
            for value in raw_order
        ]
    )
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
        parsed_critical_time = finite_represented_real(critical_time, "critical_time")[
            0
        ]

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
    validated = [finite_represented_real(value, "times")[0] for value in times]
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
    if n == 0:
        return None
    # Initial occupation precedes every later recrossing, even in a one-state
    # report. It is not evidence that a transition occurred before observation.
    if values[0] > threshold:
        return times[0]

    for i in range(n - 1):
        if values[i] <= threshold < values[i + 1]:
            if not math.isfinite(float(values[i + 1])):
                return times[i + 1]
            # Linear interpolation
            dv = values[i + 1] - values[i]
            alpha = (threshold - values[i]) / dv
            gap = times[i + 1] - times[i]
            if math.isfinite(gap):
                return times[i] + alpha * gap
            return (1.0 - alpha) * times[i] + alpha * times[i + 1]

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

    # Positivity is the admission contract; a fixed coordinate-size cutoff
    # would change availability merely by changing time or field units.
    pairs = [
        (time, value)
        for time, value in zip(times, order_abs)
        if time > t_c and value > 0.0
    ]
    if len(pairs) < 3:
        return None, None
    log_dt = []
    for time, _ in pairs:
        distance = time - t_c
        log_dt.append(
            math.log(distance)
            if math.isfinite(distance)
            else math.log(time / 2.0 - t_c / 2.0) + math.log(2.0)
        )
    log_order = [math.log(value) for _, value in pairs]
    time_std = finite_population_std(log_dt)
    if time_std == 0.0:
        # Distinct input times can lose separation after represented logs.
        return None, None
    order_std = finite_population_std(log_order)
    if order_std == 0.0:
        return 0.0, 1.0
    correlation = finite_pearson_correlation(log_dt, log_order)
    assert correlation is not None
    exponent = correlation * (order_std / time_std)
    return exponent, correlation * correlation


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
