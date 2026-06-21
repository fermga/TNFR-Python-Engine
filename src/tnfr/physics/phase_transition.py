r"""TNFR Phase Transition Module — Life/Non-Life as Universal Symmetry Breaking.

Derives the **life/non-life phase transition** from TNFR unified fields,
replacing static threshold criteria with a rigorous second-order phase
transition theory grounded in the nodal equation.

MAIN RESULT (Structural Phase Transition Theorem)
==================================================
The symmetry breaking field

    𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²)

serves as the **order parameter** of a second-order phase transition
between non-life (symmetric, ⟨𝒮⟩ = 0) and life (broken symmetry,
⟨𝒮⟩ ≠ 0).

Key predictions derived from the nodal equation (audit 2026: the
constant correspondence is an overlay, only π is a genuine scale):

1. **Symmetry classification**:
       𝒮 = 0  ⟹  symmetric phase (non-life)
       𝒮 ≠ 0  ⟹  broken symmetry (life)

2. **Chirality requirement**: Life requires non-zero chirality
   χ = |∇φ|·K_φ − J_φ·J_ΔNFR (biological homochirality).

3. **Critical exponent**: Near the transition the order parameter follows
   a power law |⟨𝒮⟩| ~ |p − p_c|^{β}. The exponent is an OBSERVABLE to be
   MEASURED (``fit_critical_exponent``); it is NOT a derived universal
   constant — a measurement across sweep protocols gives protocol-dependent
   values (audit 2026), so the reference scale γ/π ≈ 0.1837 used below is a
   calibrated TIER-2 parameter, not a prediction.

4. **Divergent correlation length**: ξ_C → ∞ at the critical point,
   as required for any continuous phase transition.

DERIVATION
==========
From the nodal equation ∂EPI/∂t = νf · ΔNFR(t):

**Step 1**: The symmetry breaking field 𝒮 measures the imbalance between
conjugate field pairs.  In the symmetric (non-life) phase, gradient and
curvature sectors are balanced: |∇φ|² ≈ K_φ² and J_φ² ≈ J_ΔNFR²,
giving ⟨𝒮⟩ = 0.

**Step 2**: Autopoietic dynamics (A > 1) create a persistent imbalance
where self-generation amplifies the gradient sector over the curvature
sector, breaking the conjugate symmetry: ⟨𝒮⟩ ≠ 0.

**Step 3**: The chirality field χ measures handedness.  Mirror symmetry
(χ = 0) prevails in non-life; autopoiesis selects a preferred chirality
(|χ| > 0), providing the homochirality signature of biological systems.

**Step 4**: Near the critical point p_c, the order parameter scales as
|⟨𝒮⟩| ~ |p − p_c|^{β}. The exponent β is MEASURED, not derived: a sweep
across protocols gives protocol-dependent values (audit 2026), so there is
no universal closed-form exponent. The constant γ/π ≈ 0.1837 is retained
only as a calibrated reference / noise-floor scale (TIER-2), not a
prediction of the nodal equation.

**Step 5**: The coherence length ξ_C diverges at p_c via standard
Landau theory, signalling the onset of long-range structural
correlations that enable collective autopoietic behavior.

Physics Foundation
------------------
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
- Unified fields: 𝒮 and χ from unified.py (single source of truth)
- Phase-gradient reference scale: γ/π ≈ 0.1837 (calibrated TIER-2; audit
  2026: NOT a derived universal exponent — the measured exponent is
  protocol-dependent)
- Coherence length: ξ_C from canonical.py

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

# Delegate to authoritative field computations (single source of truth)
from .unified import (
    compute_symmetry_breaking_field,
    compute_chirality_field,
)
from .canonical import (
    estimate_coherence_length,
)

# ============================================================================
# EMERGENT CLASSIFICATION — sampling-noise z-scores (NO magic constant)
# ============================================================================
# Audit 2026: the symmetric phase has ⟨𝒮⟩ = 0 EXACTLY (by symmetry), so any
# finite-N deviation is sampling noise with standard error SE = √(Var/N). The
# phase is decided by the z-score |⟨𝒮⟩|/SE — the statistical significance of
# the symmetry breaking, MEASURED from the system itself. The previous magic
# scales ((γ/π)², γ/(π+γ)) were verified INERT (they sat in a two-order-of-
# magnitude gap; sweeping them changed no classification) and were removed.
# The only cut is z = 1 (one sampling sigma) — the noise scale itself, not a
# free parameter.

#: Significance cut: a field is "broken" when its z-score exceeds one sampling
#: standard error (z > 1). This is the noise scale, not a tunable constant.
Z_SIGNIFICANCE: float = 1.0


def symmetry_zscore(mean_abs: float, variance: float, n: int) -> float:
    r"""Sampling-noise z-score |mean| / SE, with SE = √(Var/N).

    The emergent significance of a field's deviation from zero, measured
    against the system's own finite-N sampling noise — no magic constant.
    Returns 0.0 for a perfectly uniform zero field (Var=0, mean=0) and +∞
    for a uniform non-zero field (Var=0, mean>0, a fully broken state).
    """
    if n <= 0:
        return 0.0
    se = math.sqrt(max(variance, 0.0) / n)
    if se == 0.0:
        return 0.0 if mean_abs == 0.0 else math.inf
    return mean_abs / se

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Phase(Enum):
    """Structural phase classification from symmetry breaking analysis."""
    NON_LIFE = "non_life"       # Symmetric: ⟨𝒮⟩ ≈ 0, |⟨χ⟩| ≈ 0
    CRITICAL = "critical"       # Near transition: ⟨𝒮⟩ small, ξ_C large
    LIFE = "life"               # Broken symmetry: ⟨𝒮⟩ ≠ 0, |⟨χ⟩| > 0

@dataclass
class PhaseTransitionTelemetry:
    r"""Container for life/non-life phase transition telemetry.

    Captures the complete structural signature of the symmetry breaking
    transition, including order parameter, chirality, susceptibility,
    coherence length, and critical exponent measurements.

    All quantities derive from the nodal equation via unified fields.

    Attributes
    ----------
    times : list[float]
        Structural time stamps (Hz_str units).
    order_parameter : np.ndarray
        Network-averaged symmetry breaking ⟨𝒮⟩(t).  ⟨𝒮⟩ = 0 → non-life,
        ⟨𝒮⟩ ≠ 0 → life.
    order_parameter_abs : np.ndarray
        |⟨𝒮⟩|(t) — magnitude of order parameter for scaling analysis.
    chirality_mean : np.ndarray
        Network-averaged chirality ⟨χ⟩(t).  Non-zero → homochirality.
    chirality_abs_mean : np.ndarray
        ⟨|χ|⟩(t) — mean absolute chirality (non-zero even without
        preferred handedness if local chirality exists).
    susceptibility : np.ndarray
        χ_𝒮(t) = N · Var(𝒮) — fluctuation susceptibility.  Diverges
        at the critical point.
    coherence_length : np.ndarray
        ξ_C(t) — spatial correlation scale.  Diverges at criticality.
    phase_classification : list[Phase]
        Phase assignment per time step.
    transition_time : float | None
        First structural time where the system transitions from NON_LIFE
        to LIFE, with linear interpolation at the crossing.  None if no
        transition detected.
    critical_time : float | None
        Time of maximum susceptibility (closest to critical point).
    measured_exponent : float | None
        Fitted critical exponent from |⟨𝒮⟩| ~ |t − t_c|^{β} in the
        post-transition regime. This is the real observable (audit 2026:
        the exponent is protocol-dependent, not a universal constant — there
        is no derived 'theoretical' value to compare against).
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

@dataclass
class PhaseSnapshot:
    """Instantaneous phase transition diagnostics for a single graph state.

    Lighter-weight alternative to :class:`PhaseTransitionTelemetry` for
    point-in-time analysis without a time series.
    """
    order_parameter: float          # ⟨𝒮⟩
    order_parameter_abs: float      # |⟨𝒮⟩|
    chirality_mean: float           # ⟨χ⟩
    chirality_abs_mean: float       # ⟨|χ|⟩
    susceptibility: float           # N · Var(𝒮)
    coherence_length: float         # ξ_C
    phase: Phase                    # Classified phase
    has_homochirality: bool         # |⟨χ⟩| > threshold

# ============================================================================
# CORE COMPUTATIONS
# ============================================================================

def compute_order_parameter(G: Any) -> dict[str, float]:
    r"""Compute the phase transition order parameter from symmetry breaking.

    Returns network statistics of 𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²):

    - ⟨𝒮⟩ = (1/N) Σ_i 𝒮(i)  — mean (order parameter)
    - ⟨|𝒮|⟩ = (1/N) Σ_i |𝒮(i)|  — mean magnitude
    - Var(𝒮) = ⟨𝒮²⟩ − ⟨𝒮⟩²  — variance
    - χ_𝒮 = N · Var(𝒮)  — susceptibility (diverges at critical point)

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
            'mean': 0.0, 'abs_mean': 0.0, 'variance': 0.0,
            'susceptibility': 0.0, 'max': 0.0, 'min': 0.0, 'n_nodes': 0,
        }
    mean = float(np.mean(values))
    return {
        'mean': mean,
        'abs_mean': float(np.mean(np.abs(values))),
        'variance': float(np.var(values)),
        'susceptibility': float(N * np.var(values)),
        'max': float(np.max(values)),
        'min': float(np.min(values)),
        'n_nodes': N,
    }

def compute_chirality_statistics(G: Any) -> dict[str, float]:
    r"""Compute chirality field statistics for homochirality detection.

    Chirality χ = |∇φ|·K_φ − J_φ·J_ΔNFR.
    Non-zero ⟨χ⟩ indicates broken mirror symmetry (homochirality).

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
        return {'mean': 0.0, 'abs_mean': 0.0, 'variance': 0.0,
                'max_abs': 0.0, 'n_nodes': 0}
    return {
        'mean': float(np.mean(values)),
        'abs_mean': float(np.mean(np.abs(values))),
        'variance': float(np.var(values)),
        'max_abs': float(np.max(np.abs(values))),
        'n_nodes': N,
    }

def classify_phase(
    order_z: float,
    chirality_z: float,
) -> Phase:
    r"""Classify the structural phase from emergent z-scores.

    The phase is decided by the STATISTICAL SIGNIFICANCE of symmetry
    breaking, measured against the system's own sampling noise — no magic
    constant (audit 2026). With ``order_z = |⟨𝒮⟩|/SE`` and
    ``chirality_z = ⟨|χ|⟩/SE`` (SE = √(Var/N), via :func:`symmetry_zscore`):

    - **NON_LIFE**: ``order_z ≤ 1`` — ⟨𝒮⟩ is within one sampling sigma of
      zero (symmetric phase).
    - **LIFE**: ``order_z > 1`` AND ``chirality_z > 1`` — significant
      symmetry breaking with significant homochirality.
    - **CRITICAL**: ``order_z > 1`` but ``chirality_z ≤ 1`` — broken
      magnitude without a significant preferred handedness (intermediate).

    Parameters
    ----------
    order_z : float
        Sampling-noise z-score of |⟨𝒮⟩| (symmetry-breaking significance).
    chirality_z : float
        Sampling-noise z-score of ⟨|χ|⟩ (homochirality significance).

    Returns
    -------
    Phase
        NON_LIFE, CRITICAL, or LIFE.
    """
    if order_z <= Z_SIGNIFICANCE:
        return Phase.NON_LIFE
    if chirality_z > Z_SIGNIFICANCE:
        return Phase.LIFE
    return Phase.CRITICAL

def capture_phase_snapshot(G: Any) -> PhaseSnapshot:
    """Capture instantaneous phase transition diagnostics.

    Parameters
    ----------
    G : NetworkX graph
        Network with structural attributes.

    Returns
    -------
    PhaseSnapshot
        Point-in-time phase transition diagnostics.
    """
    op = compute_order_parameter(G)
    chi = compute_chirality_statistics(G)
    xi = estimate_coherence_length(G)
    if math.isnan(xi):
        xi = 0.0

    n_nodes = op['n_nodes']
    order_z = symmetry_zscore(op['abs_mean'], op['variance'], n_nodes)
    chirality_z = symmetry_zscore(chi['abs_mean'], chi['variance'], n_nodes)
    phase = classify_phase(order_z, chirality_z)

    return PhaseSnapshot(
        order_parameter=op['mean'],
        order_parameter_abs=op['abs_mean'],
        chirality_mean=chi['mean'],
        chirality_abs_mean=chi['abs_mean'],
        susceptibility=op['susceptibility'],
        coherence_length=xi,
        phase=phase,
        has_homochirality=chirality_z > Z_SIGNIFICANCE,
    )

# ============================================================================
# TIME-SERIES ANALYSIS — PHASE TRANSITION DETECTION
# ============================================================================

def detect_phase_transition(
    graph_sequence: Sequence[Any],
    times: Sequence[float],
) -> PhaseTransitionTelemetry:
    r"""Detect life/non-life phase transition from a time series of graph states.

    Computes the order parameter ⟨𝒮⟩, chirality ⟨χ⟩, susceptibility,
    and coherence length at each time step, then:

    1. Classifies each snapshot as NON_LIFE / CRITICAL / LIFE.
    2. Finds the transition time via interpolation on |⟨𝒮⟩|.
    3. Identifies the critical time (peak susceptibility).
    4. Fits the critical exponent from the post-transition power law
       |⟨𝒮⟩| ~ |t − t_c|^{γ_c}.

    Parameters
    ----------
    graph_sequence : Sequence[nx.Graph]
        Time-ordered TNFR network states.
    times : Sequence[float]
        Structural times corresponding to each graph state.

    Returns
    -------
    PhaseTransitionTelemetry
        Complete transition diagnostics including measured exponent.
    """
    times = list(times)
    n = len(graph_sequence)

    order_param = np.zeros(n)
    order_param_abs = np.zeros(n)
    chi_mean = np.zeros(n)
    chi_abs_mean = np.zeros(n)
    suscept = np.zeros(n)
    xi_c = np.zeros(n)
    order_z_series = np.zeros(n)
    phases: list[Phase] = []

    for i, G in enumerate(graph_sequence):
        op = compute_order_parameter(G)
        chi = compute_chirality_statistics(G)
        xi = estimate_coherence_length(G)
        if math.isnan(xi):
            xi = 0.0

        order_param[i] = op['mean']
        order_param_abs[i] = op['abs_mean']
        chi_mean[i] = chi['mean']
        chi_abs_mean[i] = chi['abs_mean']
        suscept[i] = op['susceptibility']
        xi_c[i] = xi

        n_nodes = op['n_nodes']
        order_z = symmetry_zscore(op['abs_mean'], op['variance'], n_nodes)
        chirality_z = symmetry_zscore(chi['abs_mean'], chi['variance'], n_nodes)
        order_z_series[i] = order_z

        phase = classify_phase(order_z, chirality_z)
        phases.append(phase)

    # --- Transition time: interpolate where the symmetry-breaking z-score ---
    # --- crosses the significance cut (z = 1), the emergent noise scale.   ---
    transition_time = _find_crossing_time(
        times, order_z_series, Z_SIGNIFICANCE
    )

    # --- Critical time: peak susceptibility ---
    critical_time: float | None = None
    if n > 0 and np.max(suscept) > 0:
        peak_idx = int(np.argmax(suscept))
        critical_time = times[peak_idx]

    # --- Critical exponent fit ---
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
    )

# ============================================================================
# CRITICAL EXPONENT MEASUREMENT
# ============================================================================

def fit_critical_exponent(
    times: Sequence[float],
    order_parameter_abs: np.ndarray,
    critical_time: float | None = None,
) -> dict[str, float | None]:
    r"""Fit the critical exponent from order parameter scaling.

    Near the transition:

        |⟨𝒮⟩| ~ |t − t_c|^{γ_c}

    Fits γ_c via log-log linear regression on the post-critical data.

    Parameters
    ----------
    times : Sequence[float]
        Structural times.
    order_parameter_abs : np.ndarray
        |⟨𝒮⟩| time series.
    critical_time : float | None
        Estimated critical time t_c.  If None, uses the midpoint where
        the order parameter first exceeds the noise floor.

    Returns
    -------
    dict[str, float | None]
        'exponent': fitted β, 'r_squared': R². Values are None if the fit
        is impossible (insufficient data). There is no 'theoretical' value:
        the exponent is a measured observable, not a derived constant.
    """
    exponent, r_sq = _fit_critical_exponent(
        list(times), order_parameter_abs, critical_time
    )
    return {
        'exponent': exponent,
        'r_squared': r_sq,
    }

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

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
    r"""Fit |⟨𝒮⟩| ~ |t − t_c|^{γ_c} via log-log regression.

    Uses data points in the post-critical regime where both |t − t_c|
    and |⟨𝒮⟩| are positive.  Returns (exponent, R²) or (None, None).
    """
    if t_c is None or len(times) < 4:
        return None, None

    t_arr = np.array(times, dtype=float)
    dt = np.abs(t_arr - t_c)

    # Select post-critical points with non-trivial order parameter
    mask = (dt > 1e-12) & (order_abs > 1e-15) & (t_arr >= t_c)
    if np.sum(mask) < 3:
        # Try both sides if not enough post-critical data
        mask = (dt > 1e-12) & (order_abs > 1e-15)
    if np.sum(mask) < 3:
        return None, None

    log_dt = np.log(dt[mask])
    log_S = np.log(order_abs[mask])

    # Linear regression: log|S| = γ_c · log|t − t_c| + const
    A = np.vstack([log_dt, np.ones_like(log_dt)]).T
    try:
        result = np.linalg.lstsq(A, log_S, rcond=None)
        coeffs = result[0]
        exponent = float(coeffs[0])

        # R² computation
        predicted = A @ coeffs
        ss_res = float(np.sum((log_S - predicted) ** 2))
        ss_tot = float(np.sum((log_S - np.mean(log_S)) ** 2))
        r_squared = 1.0 - ss_res / (ss_tot + 1e-15) if ss_tot > 1e-15 else 0.0

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
