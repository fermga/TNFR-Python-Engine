"""Canonical operator-role predicates and physics-motivated grammar calibration.

This module collects declared initiation, closure and stabilization roles.
Consumers share these predicates rather than duplicating operator lists. Role
membership is a grammar policy, not a derivation of autonomous event selection.
The numerical recency/debt formulas calibrate grammar policies to a scalar
relaxation surrogate; they are not universal
trajectory bounds. See ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md``.

Core TNFR Equation
------------------
∂EPI/∂t = νf · ΔNFR(t)

Where:
- EPI: Primary Information Structure (coherent form)
- νf: Structural frequency (reorganization rate, Hz_str)
- ΔNFR: Structural reorganization pressure

Operator Activation Contracts
---------------------------
The unforced continuous nodal row is nonzero only when νf and ΔNFR are nonzero.
Initialization and latent form thresholds are additional operator contracts.
The nodal derivative is
well-defined at EPI=0 whenever νf and ΔNFR are finite; the generator requirement
is not a consequence of a singular derivative at zero.

Endpoint Scope
--------------
Operational closure is membership in the supported endpoint-role set.
It need not suppress the nodal rate or yield a stationary state. Even a rate
tending to zero does not alone prove finite accumulated change; stability and
convergence require separate trajectory assumptions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real

__all__ = [
    "derive_start_operators_from_physics",
    "derive_end_operators_from_physics",
    "derive_stabilizers_from_physics",
    "derive_destabilizers_from_physics",
    "derive_transformers_from_physics",
    "derive_bifurcation_triggers_from_physics",
    "derive_bifurcation_handlers_from_physics",
    "derive_bifurcation_window_from_physics",
    "derive_u2_debt_capacity_from_physics",
    "can_generate_epi_from_null",
    "can_activate_latent_epi",
    "can_stabilize_reorganization",
    "achieves_operational_closure",
    "increases_structural_pressure",
    "provides_negative_feedback",
    "executes_bifurcation",
    "triggers_bifurcation",
    "handles_bifurcation",
]


def _scalar_relaxation(nu_f: float, dt: float | None) -> Fraction:
    """Admit the shared nonnegative scalar surrogate, not a graph trajectory.

    The binary64 product is the configured relaxation coefficient. Subsequent
    rational arithmetic prevents cancellation, reciprocal overflow and integer
    boundary rounding from changing the specified scalar policy.
    """
    if dt is None:
        from ..constants.canonical import DT_CANONICAL

        dt = DT_CANONICAL
    values = []
    for name, value in (("nu_f", nu_f), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite positive real number")
        try:
            scalar = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite positive real number") from exc
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be a finite positive real number")
        values.append(scalar)
    relax = values[0] * values[1]
    if not math.isfinite(relax) or not 0.0 < relax <= 1.0:
        raise ValueError(
            "The represented nu_f*dt must lie in (0, 1] for the "
            "nonnegative scalar relaxation surrogate"
        )
    return Fraction.from_float(relax)


def derive_bifurcation_window_from_physics(
    nu_f: float = 1.0, dt: float | None = None
) -> int:
    r"""Return the U4b recency policy calibrated to a scalar decay surrogate.

    The calibration uses ``q = 1 - nu_f*dt*rho`` with fixed ``rho = 1`` and
    selects the first step with ``q**n < 1/(pi+1)``, capped at 64 operator
    positions. At ``nu_f=1, dt=0.5`` the result is the canonical
    **3-operation** window, shared by every
    destabilizer. The historical public function name is retained.

    On loopless graphs without isolates, ``trace(L_rw)/N = 1`` is the mean
    eigenvalue, not the rate of each pressure perturbation. An Euler diffusion
    mode has multiplier ``1-nu_f*dt*lambda_k``. On a 21-node path its Fiedler
    mode needs 231 steps to reach the target at the canonical frequency and
    step. Thus this policy is not a topology-independent modal relaxation bound.

    Finite positive inputs must give a represented ``0 < nu_f*dt <= 1``;
    zero capacity has no finite relaxation time, and negative multipliers
    belong to a different surrogate. The admitted ``q=0`` case needs one step.
    Rational comparisons use the represented product and represented band;
    the 64-position policy cap need not satisfy the decay inequality. None of
    these scalar calculations certifies graph Euler stability. See
    ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for assumptions and witnesses.

    Parameters
    ----------
    nu_f : float
        Finite positive structural frequency. Default ``1.0``.
    dt : float, optional
        Finite positive step. Defaults to ``DT_CANONICAL`` (0.5).

    Returns
    -------
    int
        Calibrated U4b recency-window length in operator positions.

    Raises
    ------
    ValueError
        If either input or its represented product is outside the surrogate's
        domain. Booleans, nonfinite values and underflowed products are rejected.
    """
    q = 1 - _scalar_relaxation(nu_f, dt)
    band = Fraction.from_float(1.0 / (math.pi + 1.0))
    n = 1
    remaining = q
    while remaining >= band and n < 64:
        remaining *= q
        n += 1
    return n


def derive_u2_debt_capacity_from_physics(
    nu_f: float = 1.0, dt: float | None = None
) -> int:
    r"""Return the U2 debt policy calibrated to a scalar forced recurrence.

    With fixed ``rho=1`` and ``0 <= q=1-nu_f*dt*rho < 1``, a unit-forced
    scalar recurrence has steady state ``sum(q**k)=1/(nu_f*dt*rho)``.
    Its floor calibrates the maximum operator-bookkeeping debt. At
    ``nu_f=1, dt=0.5`` this is the canonical **2**. The same surrogate sets
    the U4b recency window in :func:`derive_bifurcation_window_from_physics`.

    This geometric sum is not a maximum physical pressure, does not describe
    every graph mode, and does not prove convergence of the nodal integral
    under sustained forcing. Grammar debt counts declared operator obligations;
    a trajectory bound additionally needs feedback, gains, time steps, and a
    norm. Inputs share the finite positive domain of the U4b calibration and
    must give a represented product in ``(0,1]``. The exact reciprocal floor of
    that binary64 product avoids overflow at subnormal values and incorrect
    integer-boundary rounding. Invalid inputs do not become zero debt.
    See ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for the analytic distinction.

    Parameters
    ----------
    nu_f : float
        Finite positive structural frequency. Default ``1.0``.
    dt : float, optional
        Finite positive step. Defaults to ``DT_CANONICAL`` (0.5).

    Returns
    -------
    int
        Operator-bookkeeping capacity ``floor(1/(nu_f*dt*rho))`` within the
        admitted scalar domain; a grammar policy rather than a theorem.

    Raises
    ------
    ValueError
        If either input or its represented product is outside the surrogate's
        domain. Booleans, nonfinite values and underflowed products are rejected.
    """
    relax = _scalar_relaxation(nu_f, dt)
    return relax.denominator // relax.numerator


def can_generate_epi_from_null(operator: str) -> bool:
    """Check the registered standalone EPI-generator role.

    AL supplies its own form increment at ``EPI = 0``. This role does not
    exhaust maps that can move a zero local coordinate: Reception can import
    neighboring form. It does not claim that a generator creates capacity or
    pressure. AL requires basal νf and leaves νf and ΔNFR unchanged.

    Parameters
    ----------
    operator : str
        Canonical operator name (e.g., "emission", "reception")

    Returns
    -------
    bool
        True if the operator has the standalone EPI-generator role

    Notes
    -----
    Physical Rationale:

    **EMISSION (AL)**: ✓ Can generate from null
    - Writes a positive increment to the EPI channel
    - Does not directly write νf, phase, or ΔNFR
    - Active capacity/pressure and the nodal update remain separate conditions

    **RECEPTION (EN)**: ✗ Not assigned the generator role
    - Blends existing neighboring EPI; it has no independent source term
    - A zero local coordinate can nevertheless receive nonzero neighboring form
    - U1 membership does not classify every possible local state change
    """
    # Physical generators: create EPI via field emission
    return operator == "emission"


def can_activate_latent_epi(operator: str) -> bool:
    """Check the registered U1 activation/handoff role.

    The historical predicate name does not promise a positive capacity update
    or departure from zero nodal rate. REMESH's node glyph is advisory; its
    separately invoked network realization mixes signed present/delayed form.
    NAV's direct and public paths have different auxiliary effects and live
    admission. Neither membership requires a positive sign for scalar EPI.
    Actual form, capacity and temporal evidence belong to the executing path.
    """
    return operator in {"recursivity", "transition"}


def can_stabilize_reorganization(operator: str) -> bool:
    """Check whether an operator supplies the νf-side closure role.

    The historical predicate name is retained. This classifies the registered
    Silence role; it does not measure a trajectory or assert one-step arrival
    at equilibrium.

    Parameters
    ----------
    operator : str
        Canonical operator name

    Returns
    -------
    bool
        True if the operator is registered to suppress the nodal rate via νf

    Notes
    -----
    Physical Rationale:

    **SILENCE (SHA)**: ✓ supplies rate suppression
    - Scales νf by a configured factor below one
    - Repetition can drive νf and therefore νf·ΔNFR toward zero when pressure
      remains bounded; a single label does not prove stationarity
    - Preserves EPI intact (memory/latency)
    - Canonical structural silence

    **COHERENCE (IL)**: ✗ Not sufficient alone
    - Reduces |ΔNFR| (decreases gradient)
    - But doesn't guarantee ∂EPI/∂t → 0
    - EPI can still evolve slowly
    - Best as intermediate, not terminal
    """
    return operator == "silence"


def achieves_operational_closure(operator: str) -> bool:
    """Check whether an operator belongs to the operational-closure policy.

    The historical predicate name is retained. Membership marks an operator-
    history boundary or handoff; it does not prove a stable successor state.

    Parameters
    ----------
    operator : str
        Canonical operator name

    Returns
    -------
    bool
        True if the operator is registered as an operational closure

    Notes
    -----
    Physical Rationale:

    **TRANSITION (NAV)**: ✓ Achieves closure
    - Hands off to next phase/regime
    - Completes current structural cycle
    - Opens new cycle in target phase
    - Natural boundary operator

    **RECURSIVITY (REMESH)**: ✓ Achieves closure
    - Registered operational boundary after a fractal echo
    - Closure membership does not prove termination at an attractor
    - Identity preservation is checked from the realized transition

    **DISSONANCE (OZ)**: ? Questionable closure
    - Generates high ΔNFR (instability)
    - Can be terminal in specific contexts (postponed conflict)
    - But typically leads to further transformation
    - Controversial as general terminator
    """
    return operator in {"transition", "recursivity", "dissonance"}


def derive_start_operators_from_physics() -> frozenset[str]:
    """Collect registered U1 initiation roles from the declared contracts.

    A sequence can start with an operator if it satisfies at least one:
    1. Can generate EPI from null state (generative capacity)
    2. Can activate latent/dormant EPI (activation capacity)

    Returns
    -------
    frozenset[str]
        set of canonical operator names that can validly start sequences

    Examples
    --------
    >>> ops = derive_start_operators_from_physics()
    >>> "emission" in ops
    True
    >>> "recursivity" in ops
    True
    >>> "reception" in ops
    False

    Notes
    -----
    **Registered Start Operators:**

    1. **emission** - EPI generator
       - Creates EPI from null via field emission
       - Requires supplied basal capacity; does not write νf or ΔNFR
       - Acts on existing nodal support

    2. **recursivity** - EPI activator
       - Replicates existing/latent patterns
       - Echoes structure across scales
       - Physical: fractal activation

    3. **transition** - Registered activation/handoff boundary
       - State changes depend on direct/public path and admission
       - Membership alone does not guarantee a capacity increase

    **Other labels are outside the U1 initiation policy:**

    - **reception**: Blends existing EPI fields and has no generative term
    - **coherence**: Stabilizes existing form, cannot create from null
    - **dissonance**: Perturbs existing pressure
    - **coupling**: Links compatible existing nodes
    - **resonance**: Blends admitted signed form over compatible neighbors
    - **silence**: Suspends reorganization, needs active νf to suspend
    - **expansion/contraction**: Adjust existing capacity and pressure
    - **self_organization**: Creates sub-EPIs from existing structure
    - **mutation**: Transforms across thresholds, needs base structure

    See Also
    --------
    can_generate_epi_from_null : Check generative capacity
    can_activate_latent_epi : Check activation capacity
    """
    return frozenset(
        op
        for op in _all_canonical_operator_names()
        if can_generate_epi_from_null(op) or can_activate_latent_epi(op)
    )


def derive_end_operators_from_physics() -> frozenset[str]:
    """Derive registered end operators from TNFR closure predicates.

    A sequence can end with an operator if it satisfies at least one policy:
    1. Supplies νf-side rate suppression
    2. Belongs to the operational closure/handoff set

    Returns
    -------
    frozenset[str]
        set of canonical operator names that can validly end sequences

    Examples
    --------
    >>> ops = derive_end_operators_from_physics()
    >>> "silence" in ops
    True
    >>> "transition" in ops
    True
    >>> "emission" in ops
    False

    Notes
    -----
    **Derived End Operators:**

    1. **silence** - Rate-suppression closure
       - Scales νf downward; repeated application approaches latency
       - Preserves EPI intact
       - Physical: structural suspension

    2. **transition** - Closure
       - Hands off to next phase
       - Completes current cycle
       - Physical: regime boundary

    3. **recursivity** - Fractal closure
       - Registered boundary after delayed/multi-scale EPI mixing
       - Does not certify an attractor or asymptotic termination
       - Physical: operational fractality

    4. **dissonance** - Questionable closure
       - High ΔNFR state (tension)
       - Can represent postponed conflict
       - Physical: contained instability
       - Included for backward compatibility

    **Other labels are outside the U1 endpoint policy:**

    - **emission**: Sources an EPI jump; it does not imply a later positive rate
    - **reception**: Captures input (ongoing process)
    - **coherence**: Reduces ΔNFR but doesn't force ∂EPI/∂t = 0
    - **coupling**: Creates links (ongoing connection)
    - **resonance**: Amplifies coherence (active propagation)
    - **expansion**: Raises capacity; state dimension need not change
    - **contraction**: Concentrates trajectories (active compression)
    - **self_organization**: Creates cascades (ongoing emergence)
    - **mutation**: Crosses thresholds (active transformation)

    See Also
    --------
    can_stabilize_reorganization : Check stabilization capacity
    achieves_operational_closure : Check closure capacity
    """
    return frozenset(
        op
        for op in _all_canonical_operator_names()
        if can_stabilize_reorganization(op) or achieves_operational_closure(op)
    )


# ===========================================================================
# U2 / U4 classification — role policy across the four state channels
# ===========================================================================
#
# Operator contracts have exactly four primary channels: EPI, νf, phase, and
# ΔNFR (the centralized registry is ``operators.operator_contracts``). U2 roles
# are orthogonal metadata: OZ acts directly on pressure, ZHIR perturbs phase,
# and VAL raises capacity, yet all three incur U2 destabilizer debt. IL directly
# reduces pressure; THOL supplies stabilizing reorganization during a
# bifurcation. Debt coverage is a finite-word policy and is not a convergence
# theorem for ∫νf·ΔNFR dt.
# U4 records operator roles around bifurcation thresholds. The label-level
# predicates do not establish that ∂²EPI/∂t² crossed τ, that a handler absorbed
# a perturbation, or that all destabilizer channels share one scalar threshold.
# The derive_* helpers turn the predicates into the centralized grammar sets.


def increases_structural_pressure(operator: str) -> bool:
    """Return whether an operator incurs U2 destabilizer debt.

    The historical function name is retained for API compatibility. Only OZ
    directly raises the primary ``ΔNFR`` channel. ZHIR acts on phase and VAL on
    ``νf``; their declared perturbations also incur U2 debt. Thus this predicate
    classifies grammar roles, not a measured sign of instantaneous pressure or
    proof about the nodal integral. Exactly three operators return ``True``:

    **DISSONANCE (OZ)**: ✓ destabilizer
    - Contract: "must increase |ΔNFR|" — injects controlled instability directly
      into the structural-pressure channel.

    **EXPANSION (VAL)**: ✓ destabilizer
    - Raises the νf capacity channel, increasing the response to any nonzero
      pressure. This is a policy-classified capacity perturbation; a pressure
      increase is not asserted.

    **MUTATION (ZHIR)**: ✓ destabilizer
    - Transforms θ → θ'. The realized wrapped phase gradient can rise or fall,
      so the destabilizer label records the declared phase perturbation.

    **Why others are NOT destabilizers:**
    - TRANSITION (NAV): a *controlled* trajectory between attractors — it is a
      generator/closure, not an assigned U2 destabilizer.
    - RECEPTION (EN): integrates the neighbour EPI field while leaving stored
      pressure and change rate fixed at the immediate jump boundary — neutral,
      not positive feedback.
    - CONTRACTION (NUL): reduces νf while densifying ΔNFR by the reciprocal
      factor; the registry treats it as a simplifier rather than U2 debt.
    """
    return operator in {"dissonance", "expansion", "mutation"}


def provides_negative_feedback(operator: str) -> bool:
    """Return whether an operator supplies U2 stabilizer coverage.

    The historical name is retained for API compatibility. IL directly reduces
    structural pressure. THOL has a reorganizing, coherence-preserving contract
    during sub-EPI formation; it need not monotonically reduce instantaneous
    pressure. Neither role alone proves convergence. Two operators return
    ``True``:

    **COHERENCE (IL)**: ✓ stabilizer
    - Contract: reduces |ΔNFR| without reducing C(t), a direct pressure feedback.

    **SELF-ORGANIZATION (THOL)**: ✓ stabilizer
    - Autopoietic reorganization preserves global form while managing a
      bifurcation. Parent/child coherence must be measured separately.
    """
    return operator in {"coherence", "self_organization"}


def executes_bifurcation(operator: str) -> bool:
    """Return the declared U4b transformer role, not a dynamical bifurcation test.

    **MUTATION (ZHIR)**: ✓ transformer
    - Phase transition θ → θ' when ΔEPI/Δt > ξ — crosses a structural threshold,
      requiring a recent declared pressure/phase/capacity perturbation plus a
      stable base (prior IL). The label check does not measure either threshold.

    **SELF-ORGANIZATION (THOL)**: ✓ transformer
    - On an admitted invocation, public THOL proposes a child when the measured
      magnitude |∂²EPI/∂t²| exceeds configured τ and the hierarchy permits birth.
      This role neither causes the invocation nor detects loss of rank,
      uniqueness or stability of a complete nodal evolution law.
    """
    return operator in {"mutation", "self_organization"}


def triggers_bifurcation(operator: str) -> bool:
    """Return the declared U4a trigger role without asserting a bifurcation.

    **DISSONANCE (OZ)**: ✓ trigger — may change later structural acceleration.
    **MUTATION (ZHIR)**: ✓ trigger — applies an admitted phase transformation.
    A finite phase change alone establishes neither a spectral crossing nor a
    change of solution branches. Both entries classify grammar context; actual
    acceleration, admission and event occurrence require separate evidence.
    """
    return operator in {"dissonance", "mutation"}


def handles_bifurcation(operator: str) -> bool:
    """U4a Bifurcation-handler test: stabilizes a triggered bifurcation.

    **SELF-ORGANIZATION (THOL)**: ✓ handler — channels the reorganization into
    sub-EPIs (controlled cascade).
    **COHERENCE (IL)**: ✓ handler — damps the elevated |ΔNFR| back toward
    equilibrium.
    """
    return operator in {"self_organization", "coherence"}


def _all_canonical_operator_names() -> frozenset[str]:
    """The 13 canonical operator function names (single source)."""
    from .operator_names import CANONICAL_OPERATOR_NAMES

    return frozenset(CANONICAL_OPERATOR_NAMES)


def derive_stabilizers_from_physics() -> frozenset[str]:
    """Derive the U2 stabilizer-coverage set from declared role predicates."""
    return frozenset(
        op for op in _all_canonical_operator_names() if provides_negative_feedback(op)
    )


def derive_destabilizers_from_physics() -> frozenset[str]:
    """Derive the U2 debt set (pressure, phase, or capacity perturbation)."""
    return frozenset(
        op
        for op in _all_canonical_operator_names()
        if increases_structural_pressure(op)
    )


def derive_transformers_from_physics() -> frozenset[str]:
    """Collect the centralized U4b transformer roles from declared predicates."""
    return frozenset(
        op for op in _all_canonical_operator_names() if executes_bifurcation(op)
    )


def derive_bifurcation_triggers_from_physics() -> frozenset[str]:
    """Derive the U4a bifurcation-trigger set."""
    return frozenset(
        op for op in _all_canonical_operator_names() if triggers_bifurcation(op)
    )


def derive_bifurcation_handlers_from_physics() -> frozenset[str]:
    """Derive the U4a bifurcation-handler set."""
    return frozenset(
        op for op in _all_canonical_operator_names() if handles_bifurcation(op)
    )
