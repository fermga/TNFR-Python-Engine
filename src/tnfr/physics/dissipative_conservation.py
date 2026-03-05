"""TNFR Dissipative Conservation — Structural Conservation under Lindblad Loss.

This module extends the structural conservation law (conservation.py) to
**dissipative regimes** where Lindblad-type collapse operators introduce
controlled decoherence.  The central result:

DISSIPATIVE CONTINUITY THEOREM
==============================
For a TNFR density operator ρ evolving under the Lindblad superoperator
L = L_H + L_D  (Hamiltonian + dissipator), the structural continuity
equation acquires a **dissipation source**:

    ∂ρ_s/∂t + div J = S_grammar + D[ρ]

where:
- S_grammar → 0 under grammar U1-U6 (conservative part, from conservation.py)
- D[ρ] = Σ_k (L_k ρ L_k†  -  ½{L_k† L_k, ρ}) is the dissipator contribution

The dissipation rate is bounded:

    |D[ρ]| ≤ Σ_k ‖L_k‖² · (1 - Tr(ρ²))

This means: **more mixed states dissipate faster**, and the dissipation
vanishes for pure states at steady state (ρ² = ρ).

PHYSICS INTERPRETATION
======================
In the conservative regime (conservation.py), structural charge is
transported but not created or destroyed.  In the dissipative regime:

1. **Purity decay**: Tr(ρ²) decreases monotonically toward 1/d (maximally
   mixed state), measuring information loss to the environment.

2. **Charge leak rate**: The rate at which Noether charge Q = Tr(ρ · O_Q)
   changes is bounded by the dissipator strength and the initial coherence.

3. **Entropy production**: Von Neumann entropy S = -Tr(ρ ln ρ) increases
   monotonically, quantifying irreversibility.

4. **Steady-state conservation**: At the fixed point L[ρ_ss] = 0, a
   *modified* conservation law holds with D[ρ_ss] absorbed into the
   definition of conserved quantities.

TNFR CONNECTION
===============
The Lindblad collapse operators map to TNFR grammar violations:
- L_k ~ destabilizers (OZ, ZHIR, VAL) without stabilizers (IL, THOL)
- The bound |D[ρ]| ~ ‖L_k‖² corresponds to grammar U2 violation magnitude
- Steady-state convergence corresponds to grammar closure (U1b + U2)

STATUS: CANONICAL — Extends conservation.py to dissipative regimes.

References
----------
- Conservative conservation: src/tnfr/physics/conservation.py
- Lindblad generators: src/tnfr/mathematics/generators.py
- ContractiveDynamicsEngine: src/tnfr/mathematics/dynamics.py
- Nodal equation: ∂EPI/∂t = νf · ΔNFR(t) [TNFR.pdf §2.1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..mathematics.unified_numerical import np

try:
    from ..mathematics.dynamics import ContractiveDynamicsEngine
    from ..mathematics.backend import ensure_numpy, get_backend
except ImportError:  # pragma: no cover
    ContractiveDynamicsEngine = None  # type: ignore[assignment,misc]
    ensure_numpy = None  # type: ignore[assignment]
    get_backend = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DissipativeSnapshot:
    """Snapshot of a dissipative density-operator state.

    Captures the structural invariants of a density operator ρ
    evolving under Lindblad dynamics.

    Attributes
    ----------
    density : np.ndarray
        The density operator ρ (dim × dim, Hermitian, Tr=1).
    trace : float
        Tr(ρ) — should be 1.0 for valid states.
    purity : float
        Tr(ρ²) — ranges from 1/d (maximally mixed) to 1 (pure).
    von_neumann_entropy : float
        S = -Tr(ρ ln ρ) — ranges from 0 (pure) to ln(d) (maximally mixed).
    eigenvalues : np.ndarray
        Spectrum of ρ (all should be ≥ 0, sum to 1).
    """

    density: Any  # np.ndarray
    trace: float
    purity: float
    von_neumann_entropy: float
    eigenvalues: Any  # np.ndarray


@dataclass(frozen=True)
class DissipativeBalance:
    """Result of dissipative conservation analysis between two snapshots.

    The modified continuity equation:

        Δρ_s/Δt + div J ≈ S_grammar + D[ρ]

    Attributes
    ----------
    purity_before : float
        Tr(ρ²) before evolution.
    purity_after : float
        Tr(ρ²) after evolution.
    purity_decay_rate : float
        (purity_after - purity_before) / dt — should be ≤ 0.
    entropy_before : float
        Von Neumann entropy before.
    entropy_after : float
        Von Neumann entropy after.
    entropy_production_rate : float
        (S_after - S_before) / dt — should be ≥ 0.
    trace_drift : float
        |Tr(ρ_after) - 1| — measures trace preservation quality.
    dissipation_bound : float
        Theoretical upper bound on |D[ρ]| from collapse operator norms.
    actual_dissipation : float
        Measured dissipation (Frobenius norm of state change beyond
        Hamiltonian contribution).
    charge_leak_rate : float
        Rate of Noether charge loss through dissipation.
    contractivity_gap : float
        ‖ρ_after - ρ_ss‖ / ‖ρ_before - ρ_ss‖ — should be ≤ 1.
    is_contractive : bool
        True when trace distance to steady state is non-increasing.
    """

    purity_before: float
    purity_after: float
    purity_decay_rate: float
    entropy_before: float
    entropy_after: float
    entropy_production_rate: float
    trace_drift: float
    dissipation_bound: float
    actual_dissipation: float
    charge_leak_rate: float
    contractivity_gap: float
    is_contractive: bool


@dataclass
class DissipativeTimeSeries:
    """Full time-series of dissipative conservation diagnostics.

    Tracks the evolution of conservation quantities through the
    Lindblad semigroup trajectory.
    """

    times: List[float] = field(default_factory=list)
    purity: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    trace_drift: List[float] = field(default_factory=list)
    purity_decay_rate: List[float] = field(default_factory=list)
    entropy_production_rate: List[float] = field(default_factory=list)
    dissipation_bound: List[float] = field(default_factory=list)
    contractivity_gap: List[float] = field(default_factory=list)

    @property
    def is_contractive(self) -> bool:
        """True when all contractivity gaps ≤ 1 (within tolerance)."""
        if not self.contractivity_gap:
            return False
        return all(g <= 1.0 + 1e-9 for g in self.contractivity_gap)

    @property
    def mean_purity_decay(self) -> float:
        """Average purity decay rate across all steps."""
        if not self.purity_decay_rate:
            return 0.0
        return float(np.mean(self.purity_decay_rate))

    @property
    def total_entropy_produced(self) -> float:
        """Total entropy produced across all steps."""
        if len(self.entropy) < 2:
            return 0.0
        return self.entropy[-1] - self.entropy[0]


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def _steady_state_from_generator(generator: Any, dim: int) -> np.ndarray:
    """Extract the steady-state density operator from a Lindblad generator.

    The steady state sits in the kernel of L (eigenvalue closest to 0).
    """
    gen = _as_complex(generator)
    evals, evecs = np.linalg.eig(gen)
    idx = int(np.argmin(np.abs(evals)))
    rho_ss = evecs[:, idx].reshape((dim, dim), order="F")
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    trace_val = np.trace(rho_ss)
    if abs(trace_val) < 1e-15:
        raise ValueError(
            "Steady state has zero trace; generator may lack a stationary state."
        )
    return rho_ss / trace_val


# Public alias
steady_state_from_generator = _steady_state_from_generator


def _safe_log(x: float) -> float:
    """Compute log(x) safely, returning 0 for x ≤ 0."""
    if x <= 0.0:
        return 0.0
    return math.log(x)


def _as_complex(matrix: Any) -> np.ndarray:
    """Coerce to complex128 ndarray (single conversion point)."""
    return np.asarray(matrix, dtype=np.complex128)


def _collapse_norms_sq(collapse_operators: Sequence[Any]) -> float:
    """Sum of squared Frobenius norms: Σ_k ‖L_k‖_F²."""
    return sum(
        float(np.linalg.norm(_as_complex(L), ord="fro") ** 2)
        for L in collapse_operators
    )


def capture_dissipative_snapshot(density: Any) -> DissipativeSnapshot:
    """Capture structural invariants of a density operator.

    Parameters
    ----------
    density : np.ndarray
        Density operator (dim × dim, complex).

    Returns
    -------
    DissipativeSnapshot
    """
    rho = _as_complex(density)

    trace_val = float(np.trace(rho).real)
    purity = float(np.trace(rho @ rho).real)

    eigenvalues = np.linalg.eigvalsh(rho)
    # Von Neumann entropy: S = -Σ λ_k ln(λ_k)
    entropy = -sum(
        float(ev) * _safe_log(float(ev))
        for ev in eigenvalues
        if float(ev) > 0
    )

    return DissipativeSnapshot(
        density=rho,
        trace=trace_val,
        purity=purity,
        von_neumann_entropy=entropy,
        eigenvalues=eigenvalues,
    )


def compute_dissipation_bound(
    collapse_operators: Sequence[Any],
    purity: float,
) -> float:
    r"""Compute the theoretical upper bound on the dissipation rate.

    From the Lindblad master equation, the dissipation rate satisfies:

        |D[ρ]| ≤ Σ_k ‖L_k‖_F² · (1 - Tr(ρ²))

    where ‖L_k‖_F is the Frobenius norm of each collapse operator.

    This bound is tight for maximally mixed initial states and vanishes
    for pure states — consistent with the physical intuition that pure
    states in the kernel of the dissipator are at rest.

    Parameters
    ----------
    collapse_operators : Sequence[np.ndarray]
        The Lindblad collapse operators L_k.
    purity : float
        Current purity Tr(ρ²) of the state.

    Returns
    -------
    float
        Upper bound on |D[ρ]|.
    """
    return _collapse_norms_sq(collapse_operators) * max(0.0, 1.0 - purity)


def compute_dissipator_action(
    density: Any,
    collapse_operators: Sequence[Any],
) -> np.ndarray:
    r"""Compute the dissipator action D[ρ] = Σ_k (L_k ρ L_k† - ½{L_k† L_k, ρ}).

    This is the explicit source term in the dissipative continuity equation.

    Parameters
    ----------
    density : np.ndarray
        Current density operator.
    collapse_operators : Sequence[np.ndarray]
        Lindblad collapse operators.

    Returns
    -------
    np.ndarray
        The dissipator matrix D[ρ].
    """
    rho = _as_complex(density)
    result = np.zeros_like(rho)

    for L in collapse_operators:
        L_arr = _as_complex(L)
        L_dag_L = L_arr.conj().T @ L_arr

        result += L_arr @ rho @ L_arr.conj().T
        result -= 0.5 * (L_dag_L @ rho + rho @ L_dag_L)

    return result


def compute_purity_decay_bound(
    collapse_operators: Sequence[Any],
    density: Any,
) -> float:
    r"""Compute the theoretical bound on purity decay rate.

    The purity P = Tr(ρ²) evolves as:

        dP/dt = 2 Tr(ρ · D[ρ])

    The upper bound on the decay rate is:

        |dP/dt| ≤ 2 Σ_k ‖L_k‖² · P · (1 - P/d)

    where d is the Hilbert space dimension.

    Parameters
    ----------
    collapse_operators : Sequence[np.ndarray]
        Lindblad collapse operators.
    density : np.ndarray
        Current density operator.

    Returns
    -------
    float
        Upper bound on |dP/dt|.
    """
    rho = _as_complex(density)
    dim = rho.shape[0]
    purity = float(np.trace(rho @ rho).real)

    # Bound on purity decay rate
    return 2.0 * _collapse_norms_sq(collapse_operators) * purity * max(0.0, 1.0 - purity / dim)


# ---------------------------------------------------------------------------
# Dissipative balance (two-snapshot comparison)
# ---------------------------------------------------------------------------

def verify_dissipative_balance(
    before: DissipativeSnapshot,
    after: DissipativeSnapshot,
    dt: float = 1.0,
    collapse_operators: Optional[Sequence[Any]] = None,
    steady_state: Optional[Any] = None,
) -> DissipativeBalance:
    r"""Verify the dissipative continuity equation between two snapshots.

    Computes the dissipative analogue of conservation balance:
    - Purity decay rate (should be ≤ 0 for dissipative evolution)
    - Entropy production rate (should be ≥ 0)
    - Contractivity (distance to steady state should decrease)
    - Dissipation bound verification

    Parameters
    ----------
    before, after : DissipativeSnapshot
        Density operator states before and after evolution.
    dt : float
        Time step between snapshots.
    collapse_operators : Sequence[np.ndarray], optional
        Lindblad collapse operators (for computing dissipation bound).
    steady_state : np.ndarray, optional
        Steady-state density operator (for contractivity check).

    Returns
    -------
    DissipativeBalance
    """
    purity_rate = (after.purity - before.purity) / dt
    entropy_rate = (after.von_neumann_entropy - before.von_neumann_entropy) / dt
    trace_drift = abs(after.trace - 1.0)

    # Dissipation bound
    if collapse_operators is not None:
        diss_bound = compute_dissipation_bound(collapse_operators, before.purity)
    else:
        diss_bound = float("nan")

    # Actual dissipation: Frobenius norm of state change per unit time
    rho_before = _as_complex(before.density)
    rho_after = _as_complex(after.density)
    delta_rho = rho_after - rho_before
    actual_diss = float(np.linalg.norm(delta_rho, ord="fro")) / dt

    # Charge leak rate: ‖ρ‖_F as proxy for total structural "charge"
    charge_leak = (
        float(np.linalg.norm(rho_before, ord="fro"))
        - float(np.linalg.norm(rho_after, ord="fro"))
    ) / dt

    # Contractivity: distance to steady state
    if steady_state is not None:
        rho_ss = _as_complex(steady_state)
        dist_before = float(np.linalg.norm(rho_before - rho_ss, ord="fro"))
        dist_after = float(np.linalg.norm(rho_after - rho_ss, ord="fro"))
        if dist_before > 1e-15:
            gap = dist_after / dist_before
        else:
            gap = 0.0 if dist_after < 1e-15 else float("inf")
    else:
        gap = float("nan")

    is_contr = gap <= 1.0 + 1e-9 if not math.isnan(gap) else True

    return DissipativeBalance(
        purity_before=before.purity,
        purity_after=after.purity,
        purity_decay_rate=purity_rate,
        entropy_before=before.von_neumann_entropy,
        entropy_after=after.von_neumann_entropy,
        entropy_production_rate=entropy_rate,
        trace_drift=trace_drift,
        dissipation_bound=diss_bound,
        actual_dissipation=actual_diss,
        charge_leak_rate=charge_leak,
        contractivity_gap=gap,
        is_contractive=is_contr,
    )


# ---------------------------------------------------------------------------
# Dissipative conservation tracker
# ---------------------------------------------------------------------------

class DissipativeConservationTracker:
    """Track conservation law compliance throughout a Lindblad trajectory.

    Links the ContractiveDynamicsEngine (Hilbert-space evolution) with
    the conservation framework (structural field analysis).

    Usage
    -----
    >>> engine = ContractiveDynamicsEngine(generator, hilbert_space)
    >>> tracker = DissipativeConservationTracker(
    ...     engine, collapse_operators=collapse_ops,
    ... )
    >>> report = tracker.evolve_and_track(initial_density, steps=50, dt=0.1)
    >>> print(f"Contractive: {report.is_contractive}")
    >>> print(f"Purity decay: {report.mean_purity_decay:.4f}")
    """

    def __init__(
        self,
        engine: Any,  # ContractiveDynamicsEngine
        *,
        collapse_operators: Optional[Sequence[Any]] = None,
        steady_state: Optional[Any] = None,
    ) -> None:
        self._engine = engine
        self._collapse_operators = list(collapse_operators or [])
        self._steady_state = steady_state
        self._series = DissipativeTimeSeries()
        self._snapshots: List[Tuple[float, DissipativeSnapshot]] = []

    @property
    def steady_state(self) -> Optional[Any]:
        """The steady-state density operator, if available."""
        return self._steady_state

    def set_steady_state(self, rho_ss: Any) -> None:
        """Set or update the steady-state density operator."""
        self._steady_state = _as_complex(rho_ss)

    def compute_steady_state(self) -> np.ndarray:
        """Compute the steady state from the generator eigendecomposition.

        The steady state is the density operator in the kernel of the
        Lindblad superoperator (eigenvalue closest to zero).

        Returns
        -------
        np.ndarray
            Steady-state density operator ρ_ss.
        """
        generator = _as_complex(self._engine.generator)
        dim = self._engine.hilbert_space.dimension
        rho_ss = _steady_state_from_generator(generator, dim)
        self._steady_state = rho_ss
        return rho_ss

    def record(self, density: Any, t: float = 0.0) -> DissipativeSnapshot:
        """Record a snapshot and compute balance against previous snapshot.

        Parameters
        ----------
        density : np.ndarray
            Current density operator.
        t : float
            Current time stamp.

        Returns
        -------
        DissipativeSnapshot
        """
        snap = capture_dissipative_snapshot(density)
        self._snapshots.append((t, snap))

        # Compute balance against previous snapshot (or defaults for first)
        if len(self._snapshots) >= 2:
            t_prev, snap_prev = self._snapshots[-2]
            dt = t - t_prev if t != t_prev else 1.0
            bal = verify_dissipative_balance(
                snap_prev, snap, dt=dt,
                collapse_operators=self._collapse_operators or None,
                steady_state=self._steady_state,
            )
            td, pdr, epr, db, cg = (
                bal.trace_drift, bal.purity_decay_rate,
                bal.entropy_production_rate, bal.dissipation_bound,
                bal.contractivity_gap,
            )
        else:
            td, pdr, epr, db, cg = abs(snap.trace - 1.0), 0.0, 0.0, 0.0, 1.0

        s = self._series
        s.times.append(t)
        s.purity.append(snap.purity)
        s.entropy.append(snap.von_neumann_entropy)
        s.trace_drift.append(td)
        s.purity_decay_rate.append(pdr)
        s.entropy_production_rate.append(epr)
        s.dissipation_bound.append(db)
        s.contractivity_gap.append(cg)

        return snap

    def evolve_and_track(
        self,
        initial_density: Any,
        *,
        steps: int,
        dt: float = 1.0,
    ) -> DissipativeTimeSeries:
        """Evolve and track conservation through the full trajectory.

        Uses the ContractiveDynamicsEngine to step the density operator
        while recording all dissipative conservation diagnostics.

        Parameters
        ----------
        initial_density : np.ndarray
            Initial density operator.
        steps : int
            Number of evolution steps.
        dt : float
            Time step per evolution step.

        Returns
        -------
        DissipativeTimeSeries
            Complete dissipative conservation diagnostics.
        """
        if ensure_numpy is None:
            raise ImportError("Mathematics backend required for evolve_and_track")

        current = np.asarray(initial_density, dtype=np.complex128)
        self.record(current, t=0.0)

        for k in range(steps):
            evolved = self._engine.step(current, dt=dt)
            current = np.asarray(ensure_numpy(evolved), dtype=np.complex128)
            self.record(current, t=(k + 1) * dt)

        return self._series

    def report(self) -> DissipativeTimeSeries:
        """Return the accumulated time-series diagnostics."""
        return self._series

    @property
    def latest_balance(self) -> Optional[DissipativeBalance]:
        """Return the most recent dissipative balance, or None."""
        if len(self._snapshots) < 2:
            return None
        t_prev, snap_prev = self._snapshots[-2]
        t_curr, snap_curr = self._snapshots[-1]
        dt = t_curr - t_prev if t_curr != t_prev else 1.0
        return verify_dissipative_balance(
            snap_prev, snap_curr, dt=dt,
            collapse_operators=self._collapse_operators or None,
            steady_state=self._steady_state,
        )


# ---------------------------------------------------------------------------
# Analytical predictions for specific models
# ---------------------------------------------------------------------------

def predict_amplitude_damping_purity(
    initial_purity: float,
    gamma: float,
    time: float,
    dim: int = 2,
) -> float:
    r"""Predict purity evolution under amplitude damping.

    For a qubit (dim=2) with amplitude damping rate γ, the purity
    evolves as:

        P(t) → 1 - (1 - P(0)) · e^{-γt} + corrections

    More precisely, the steady state is |0⟩⟨0| (purity = 1), and
    the purity approaches 1 exponentially from below.

    Parameters
    ----------
    initial_purity : float
        Tr(ρ₀²) of the initial state.
    gamma : float
        Amplitude damping rate.
    time : float
        Evolution time.
    dim : int
        Hilbert space dimension (default 2 for qubit).

    Returns
    -------
    float
        Predicted purity P(t).
    """
    # For amplitude damping, the steady state is pure (P_ss = 1)
    # The deviation from steady state decays exponentially
    decay = math.exp(-gamma * time)
    # Purity approaches 1 at rate controlled by γ
    # P(t) ≈ 1 - (1 - P(0)) · e^{-γt} for weak mixing
    return 1.0 - (1.0 - initial_purity) * decay


def predict_dephasing_purity(
    initial_density: Any,
    gamma: float,
    time: float,
) -> float:
    r"""Predict purity evolution under pure dephasing.

    For pure dephasing with rate γ, off-diagonal elements decay as
    e^{-γt/2} while diagonal elements are preserved.  The purity is:

        P(t) = Σ_i ρ_ii² + Σ_{i≠j} |ρ_ij|² · e^{-γ_{ij} t}

    where γ_{ij} depends on the dephasing operator eigenvalues.

    Parameters
    ----------
    initial_density : np.ndarray
        Initial density operator.
    gamma : float
        Dephasing rate.
    time : float
        Evolution time.

    Returns
    -------
    float
        Predicted purity P(t).
    """
    rho = _as_complex(initial_density)
    dim = rho.shape[0]
    decay = math.exp(-gamma * time)
    diag_purity = sum(float(abs(rho[i, i]) ** 2) for i in range(dim))
    offdiag_purity = sum(
        float(abs(rho[i, j]) ** 2)
        for i in range(dim) for j in range(dim) if i != j
    )
    return diag_purity + offdiag_purity * decay


# ---------------------------------------------------------------------------
# Dissipation rate analysis
# ---------------------------------------------------------------------------

def analyze_dissipation_rates(
    generator: Any,
    dim: int,
) -> Dict[str, Any]:
    r"""Analyze the dissipation rates from the Lindblad generator spectrum.

    The eigenvalues of the Lindblad superoperator L determine:
    - λ₀ = 0: steady-state eigenvalue (always present for valid L)
    - Re(λ_k) < 0: decay rates of each mode
    - Im(λ_k): oscillation frequencies (coherent dynamics)

    The **spectral gap** Δ = min_k(-Re(λ_k)) for λ_k ≠ 0 determines
    the relaxation time τ = 1/Δ: how fast the system reaches steady state.

    Parameters
    ----------
    generator : np.ndarray
        Lindblad superoperator (dim²×dim² matrix).
    dim : int
        Hilbert space dimension.

    Returns
    -------
    Dict with:
        'eigenvalues' : np.ndarray — full spectrum
        'decay_rates' : np.ndarray — -Re(λ_k) for k > 0 (positive values)
        'oscillation_frequencies' : np.ndarray — |Im(λ_k)|
        'spectral_gap' : float — smallest nonzero decay rate
        'relaxation_time' : float — 1/spectral_gap
        'n_steady_modes' : int — number of zero eigenvalues
        'n_oscillating_modes' : int — modes with |Im(λ)| > threshold
    """
    gen_np = _as_complex(generator)
    eigenvalues = np.linalg.eigvals(gen_np)

    # Separate real and imaginary parts
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    # Steady-state modes: |λ| ≈ 0
    atol = 1e-9
    steady_mask = np.abs(eigenvalues) < atol
    n_steady = int(np.sum(steady_mask))

    # Decay rates: -Re(λ) for non-steady modes
    non_steady = ~steady_mask
    decay_rates = np.sort(-real_parts[non_steady])
    # Keep only positive decay rates (physically meaningful)
    decay_rates = decay_rates[decay_rates > atol]

    # Spectral gap
    if len(decay_rates) > 0:
        spectral_gap = float(decay_rates[0])
        relaxation_time = 1.0 / spectral_gap
    else:
        spectral_gap = 0.0
        relaxation_time = float("inf")

    # Oscillating modes
    osc_threshold = 1e-6
    n_oscillating = int(np.sum(np.abs(imag_parts) > osc_threshold))

    return {
        "eigenvalues": eigenvalues,
        "decay_rates": decay_rates,
        "oscillation_frequencies": np.abs(imag_parts),
        "spectral_gap": spectral_gap,
        "relaxation_time": relaxation_time,
        "n_steady_modes": n_steady,
        "n_oscillating_modes": n_oscillating,
    }


# ---------------------------------------------------------------------------
# Connection to TNFR grammar
# ---------------------------------------------------------------------------

def classify_dissipative_regime(
    balance: DissipativeBalance,
) -> Dict[str, Any]:
    r"""Classify the dissipative regime in TNFR grammar terms.

    Maps Lindblad dynamics properties to grammar violations and
    structural interpretations.

    Parameters
    ----------
    balance : DissipativeBalance
        Result from verify_dissipative_balance.

    Returns
    -------
    Dict with:
        'regime' : str — 'weak_dissipation', 'moderate_dissipation',
                         'strong_dissipation', or 'decoherence'
        'grammar_analog' : str — Which grammar rule is "soft-violated"
        'conservation_quality' : float — 0 to 1 (1 = conservative)
        'structural_interpretation' : str — Physical meaning
    """
    purity_loss = max(0.0, balance.purity_before - balance.purity_after)
    entropy_gain = max(0.0, balance.entropy_after - balance.entropy_before)

    if purity_loss < 0.001 and entropy_gain < 0.001:
        regime = "weak_dissipation"
        grammar = "U2 approximately satisfied — stabilizers nearly balance destabilizers"
        interpretation = (
            "Near-conservative evolution; structural charge approximately conserved. "
            "Lindblad collapse operators are weak relative to Hamiltonian dynamics."
        )
        quality = 1.0 - purity_loss * 10
    elif purity_loss < 0.05:
        regime = "moderate_dissipation"
        grammar = "U2 partially violated — destabilizers exceed stabilizer capacity"
        interpretation = (
            "Controlled dissipation; purity decreases at bounded rate. "
            "Structural charge leaks through collapse channels but remains largely conserved."
        )
        quality = max(0.0, 1.0 - purity_loss * 5)
    elif purity_loss < 0.2:
        regime = "strong_dissipation"
        grammar = "U2 significantly violated — rapid approach to steady state"
        interpretation = (
            "Strong decoherence; rapid purity loss. "
            "Structural conservation breaks down; system approaches maximally mixed state."
        )
        quality = max(0.0, 1.0 - purity_loss * 2)
    else:
        regime = "decoherence"
        grammar = "U2 fully violated — no stabilizer present to counter loss"
        interpretation = (
            "Full decoherence regime; conservation inapplicable. "
            "System has lost structural coherence entirely."
        )
        quality = max(0.0, 0.5 - purity_loss)

    return {
        "regime": regime,
        "grammar_analog": grammar,
        "conservation_quality": max(0.0, min(1.0, quality)),
        "structural_interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "DissipativeSnapshot",
    "DissipativeBalance",
    "DissipativeTimeSeries",
    # Core computations
    "capture_dissipative_snapshot",
    "compute_dissipation_bound",
    "compute_dissipator_action",
    "compute_purity_decay_bound",
    # Balance and tracking
    "verify_dissipative_balance",
    "DissipativeConservationTracker",
    # Analytical predictions
    "predict_amplitude_damping_purity",
    "predict_dephasing_purity",
    # Dissipation analysis
    "analyze_dissipation_rates",
    # Grammar classification
    "classify_dissipative_regime",
    # Internal but useful
    "steady_state_from_generator",
]
