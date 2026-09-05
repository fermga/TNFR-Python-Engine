"""Canonical operator-role predicates and physics-motivated grammar calibration.

This module derives which operators can validly start or end sequences from
their declared nodal-channel contracts. Consumers share these predicates rather
than duplicating operator lists. The numerical recency/debt formulas calibrate
grammar policies to a scalar relaxation surrogate; they are not universal
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
Active reorganization requires nonzero νf and ΔNFR. Initialization and latent
form thresholds are additional operator contracts. The nodal derivative is
well-defined at EPI=0 whenever νf and ΔNFR are finite; the generator requirement
is not a consequence of a singular derivative at zero.

Node Termination Conditions
----------------------------
A sequence terminates coherently when:
1. ∂EPI/∂t → 0 (reorganization stabilizes)
2. EPI remains stable (coherence sustained)
3. No open transitions (operational closure)
"""

from __future__ import annotations

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


def derive_bifurcation_window_from_physics(
    nu_f: float = 1.0, dt: float | None = None
) -> int:
    r"""Return the U4b recency policy calibrated to a scalar decay surrogate.

    The calibration uses ``q = 1 - nu_f*dt*rho`` with fixed ``rho = 1`` and
    selects the first step with ``q**n < 1/(pi+1)``. At ``nu_f=1, dt=0.5``
    the result is the canonical **3-operation** window, shared by every
    destabilizer. The public function name and numerical behavior are retained.

    On loopless graphs without isolates, ``trace(L_rw)/N = 1`` is the mean
    eigenvalue, not the rate of each pressure perturbation. An Euler diffusion
    mode has multiplier ``1-nu_f*dt*lambda_k``. On a 21-node path its Fiedler
    mode needs 231 steps to reach the target at the canonical frequency and
    step. Thus this policy is not a topology-independent modal relaxation bound.

    The ``q <= 0`` one-step return and the 64-step cap are compatibility
    fallbacks. A negative modal multiplier can oscillate or grow; these
    branches do not certify Euler stability. See
    ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for assumptions and witnesses.

    Parameters
    ----------
    nu_f : float
        Structural frequency (reorganisation capacity).  Default ``1.0``.
    dt : float, optional
        Integration step.  Defaults to the canonical ``DT_CANONICAL`` (0.5).

    Returns
    -------
    int
        Calibrated U4b recency-window length in operator positions.
    """
    import math

    if dt is None:
        from ..constants.canonical import DT_CANONICAL

        dt = DT_CANONICAL
    rho = 1.0  # fixed mean-rate surrogate; not every graph mode's rate
    q = 1.0 - float(nu_f) * float(dt) * rho  # scalar calibration multiplier
    if q <= 0.0:
        return 1  # compatibility floor, not a modal-stability certificate
    band = 1.0 / (math.pi + 1.0)  # the coherence-band fraction (π only)
    n = 1
    while q ** n >= band and n < 64:
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
    norm. The nonpositive-rate return and all numerical behavior are retained.
    See ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for the analytic distinction.

    Parameters
    ----------
    nu_f : float
        Structural frequency (reorganisation capacity).  Default ``1.0``.
    dt : float, optional
        Integration step.  Defaults to the canonical ``DT_CANONICAL`` (0.5).

    Returns
    -------
    int
        Operator-bookkeeping capacity ``floor(1/(nu_f*dt*rho))`` for positive
        ``nu_f*dt``, otherwise zero; a grammar policy rather than a theorem.
    """
    import math

    if dt is None:
        from ..constants.canonical import DT_CANONICAL

        dt = DT_CANONICAL
    rho = 1.0  # fixed mean-rate surrogate; not every graph mode's rate
    relax = float(nu_f) * float(dt) * rho  # = 1 - q (the per-step relaxation)
    if relax <= 0.0:
        return 0
    return int(math.floor(1.0 / relax))


def can_generate_epi_from_null(operator: str) -> bool:
    """Check if operator can generate EPI from null/zero state.

    According to TNFR physics, an operator can generate EPI from nothing
    when it can:
    1. Create positive νf from νf=0 (initiate reorganization capacity)
    2. Generate positive ΔNFR from equilibrium (create structural pressure)

    Parameters
    ----------
    operator : str
        Canonical operator name (e.g., "emission", "reception")

    Returns
    -------
    bool
        True if operator can create EPI from null state

    Notes
    -----
    Physical Rationale:

    **EMISSION (AL)**: ✓ Can generate from null
    - Creates outward coherence pulse
    - Generates positive νf (activates reorganization)
    - Creates positive ΔNFR (initiates structural pressure)
    - From ∂EPI/∂t = νf · ΔNFR: can produce ∂EPI/∂t > 0 from zero

    **RECEPTION (EN)**: ✗ Cannot generate from null
    - Requires external coherence to capture
    - Needs existing EPI > 0 to anchor incoming energy
    - Cannot create structure from absolute void
    """
    # Physical generators: create EPI via field emission
    return operator == "emission"


def can_activate_latent_epi(operator: str) -> bool:
    """Check if operator can activate pre-existing latent EPI.

    Some operators can't create EPI from absolute zero but can activate
    structure that already exists in dormant/latent form (νf ≈ 0, but EPI > 0).

    Parameters
    ----------
    operator : str
        Canonical operator name

    Returns
    -------
    bool
        True if operator can activate latent structure

    Notes
    -----
    Physical Rationale:

    **RECURSIVITY (REMESH)**: ✓ Can activate latent
    - Echoes/replicates existing patterns
    - Requires source EPI > 0 to replicate
    - Increases νf of dormant structure
    - Fractality: can activate nested EPIs

    **TRANSITION (NAV)**: ✓ Can activate latent
    - Moves node from one phase to another
    - Can transition from dormant (νf ≈ 0) to active (νf > 0)
    - Requires EPI > 0 in target phase
    """
    return operator in {"recursivity", "transition"}


def can_stabilize_reorganization(operator: str) -> bool:
    """Check if operator can reduce ∂EPI/∂t → 0 (stabilize evolution).

    Terminal operators must reduce the rate of structural change to zero
    or near-zero, achieving stability.

    Parameters
    ----------
    operator : str
        Canonical operator name

    Returns
    -------
    bool
        True if operator achieves ∂EPI/∂t → 0

    Notes
    -----
    Physical Rationale:

    **SILENCE (SHA)**: ✓ Achieves ∂EPI/∂t → 0
    - Reduces νf → νf_min ≈ 0
    - From ∂EPI/∂t = νf · ΔNFR: forces ∂EPI/∂t → 0
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
    """Check if operator provides operational closure (completes cycle).

    Some operators naturally close structural sequences by establishing
    a complete operational cycle or handing off to a stable successor state.

    Parameters
    ----------
    operator : str
        Canonical operator name

    Returns
    -------
    bool
        True if operator achieves operational closure

    Notes
    -----
    Physical Rationale:

    **TRANSITION (NAV)**: ✓ Achieves closure
    - Hands off to next phase/regime
    - Completes current structural cycle
    - Opens new cycle in target phase
    - Natural boundary operator

    **RECURSIVITY (REMESH)**: ✓ Achieves closure
    - Fractal echo creates self-similar closure
    - Nested EPI structure naturally terminates
    - Operational fractality preserves identity

    **DISSONANCE (OZ)**: ? Questionable closure
    - Generates high ΔNFR (instability)
    - Can be terminal in specific contexts (postponed conflict)
    - But typically leads to further transformation
    - Controversial as general terminator
    """
    # Operators that naturally close cycles
    closures = {"transition", "recursivity"}

    # DISSONANCE is currently in VALID_END_OPERATORS but questionable
    # Include it for backward compatibility but flag for review
    # Physical justification: postponed conflict, contained tension
    questionable = {"dissonance"}

    return operator in closures or operator in questionable


def derive_start_operators_from_physics() -> frozenset[str]:
    """Derive valid start operators from TNFR physical principles.

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
    **Derived Start Operators:**

    1. **emission** - EPI generator
       - Creates EPI from null via field emission
       - Generates νf > 0 and ΔNFR > 0
       - Physical: outward coherence pulse

    2. **recursivity** - EPI activator
       - Replicates existing/latent patterns
       - Echoes structure across scales
       - Physical: fractal activation

    3. **transition** - Phase activator
       - Activates node from different phase
       - Moves from dormant to active
       - Physical: regime hand-off

    **Why Others Cannot Start:**

    - **reception**: Needs external source + existing EPI to anchor
    - **coherence**: Stabilizes existing form, cannot create from null
    - **dissonance**: Perturbs existing structure, needs EPI > 0
    - **coupling**: Links existing nodes, requires both nodes active
    - **resonance**: Amplifies existing coherence, needs EPI > 0
    - **silence**: Suspends reorganization, needs active νf to suspend
    - **expansion/contraction**: Transform existing structure dimensionally
    - **self_organization**: Creates sub-EPIs from existing structure
    - **mutation**: Transforms across thresholds, needs base structure

    See Also
    --------
    can_generate_epi_from_null : Check generative capacity
    can_activate_latent_epi : Check activation capacity
    """
    # Import here to avoid circular dependency
    from .operator_names import EMISSION, RECURSIVITY, TRANSITION

    generators = {EMISSION}  # Can create EPI from null
    activators = {RECURSIVITY, TRANSITION}  # Can activate latent EPI

    # A valid start operator must be either a generator or activator
    valid_starts = generators | activators

    return frozenset(valid_starts)


def derive_end_operators_from_physics() -> frozenset[str]:
    """Derive valid end operators from TNFR physical principles.

    A sequence can end with an operator if it satisfies at least one:
    1. Stabilizes reorganization (∂EPI/∂t → 0)
    2. Achieves operational closure (completes cycle)

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

    1. **silence** - Stabilizer
       - Forces ∂EPI/∂t → 0 via νf → 0
       - Preserves EPI intact
       - Physical: structural suspension

    2. **transition** - Closure
       - Hands off to next phase
       - Completes current cycle
       - Physical: regime boundary

    3. **recursivity** - Fractal closure
       - Self-similar pattern completion
       - Nested EPI termination
       - Physical: operational fractality

    4. **dissonance** - Questionable closure
       - High ΔNFR state (tension)
       - Can represent postponed conflict
       - Physical: contained instability
       - Included for backward compatibility

    **Why Others Cannot End:**

    - **emission**: Generates activation (∂EPI/∂t > 0), not closure
    - **reception**: Captures input (ongoing process)
    - **coherence**: Reduces ΔNFR but doesn't force ∂EPI/∂t = 0
    - **coupling**: Creates links (ongoing connection)
    - **resonance**: Amplifies coherence (active propagation)
    - **expansion**: Increases dimensionality (active growth)
    - **contraction**: Concentrates trajectories (active compression)
    - **self_organization**: Creates cascades (ongoing emergence)
    - **mutation**: Crosses thresholds (active transformation)

    See Also
    --------
    can_stabilize_reorganization : Check stabilization capacity
    achieves_operational_closure : Check closure capacity
    """
    # Import here to avoid circular dependency
    from .operator_names import DISSONANCE, RECURSIVITY, SILENCE, TRANSITION

    stabilizers = {SILENCE}  # Forces ∂EPI/∂t → 0
    closures = {TRANSITION, RECURSIVITY}  # Completes operational cycles

    # DISSONANCE is questionable but included for backward compatibility
    # Represents postponed conflict / contained tension patterns
    questionable = {DISSONANCE}

    valid_ends = stabilizers | closures | questionable

    return frozenset(valid_ends)


# ===========================================================================
# U2 / U4 classification — derived from the structural-pressure channel ΔNFR
# ===========================================================================
#
# U2 (convergence/boundedness) is a property of the integral ∫νf·ΔNFR dt.  An
# operator's U2 role is therefore determined by the SIGN of its effect on the
# structural pressure |ΔNFR|:
#   - DESTABILIZER: increases |ΔNFR| (positive feedback → integral may diverge)
#   - STABILIZER:   reduces  |ΔNFR| (negative feedback → integral converges)
# U4 (bifurcation) is governed by the second derivative ∂²EPI/∂t²: triggers
# raise it past τ, handlers absorb it, transformers execute the threshold
# crossing.  Each predicate below encodes the per-operator nodal-equation
# rationale (see the operator contracts in AGENTS.md §Operators); the
# derive_* helpers turn the predicates into the canonical operator sets.


def increases_structural_pressure(operator: str) -> bool:
    """U2 Destabilizer test: does the operator raise |ΔNFR| (positive feedback)?

    From ∂EPI/∂t = νf·ΔNFR, an operator destabilizes when it raises the
    structural pressure |ΔNFR|, pushing the integral ∫νf·ΔNFR dt toward
    divergence.  Exactly three canonical operators do this:

    **DISSONANCE (OZ)**: ✓ destabilizer
    - Contract: "must increase |ΔNFR|" — injects controlled instability directly
      into the structural-pressure channel.

    **EXPANSION (VAL)**: ✓ destabilizer
    - dim(EPI) increases — every new structural degree of freedom enters
      unaligned with the existing form, raising |ΔNFR|.

    **MUTATION (ZHIR)**: ✓ destabilizer
    - θ → θ' phase jump — desynchronizes the node from its neighbours, raising
      the phase gradient |∇φ| (the phase channel of ΔNFR).

    **Why others are NOT destabilizers:**
    - TRANSITION (NAV): a *controlled* trajectory between attractors — it is a
      generator/closure, not positive feedback; it does not drive unbounded
      pressure growth.
    - RECEPTION (EN): integrates incoming resonance (contract: must not reduce
      C(t)) — neutral, not positive feedback.
    - CONTRACTION (NUL): dim(EPI) decreases — removes degrees of freedom, the
      opposite of expansion.
    """
    return operator in {"dissonance", "expansion", "mutation"}


def provides_negative_feedback(operator: str) -> bool:
    """U2 Stabilizer test: does the operator reduce |ΔNFR| (negative feedback)?

    A stabilizer drives ∫νf·ΔNFR dt toward convergence by reducing the
    structural pressure.  Two canonical operators do this:

    **COHERENCE (IL)**: ✓ stabilizer
    - Contract: "reduces |ΔNFR|, increases C(t)" — direct negative feedback on
      the structural-pressure channel (measured: network |ΔNFR| 0.46 → 0.25).

    **SELF-ORGANIZATION (THOL)**: ✓ stabilizer
    - Autopoietic stabilization — bounds the aggregate child reorganization
      while preserving the global form (U5: C_parent ≥ α·Σ C_child).
    """
    return operator in {"coherence", "self_organization"}


def executes_bifurcation(operator: str) -> bool:
    """U4b Transformer test: does the operator execute a structural bifurcation
    that needs threshold context (a recent destabilizer)?

    **MUTATION (ZHIR)**: ✓ transformer
    - Phase transition θ → θ' when ΔEPI/Δt > ξ — crosses a structural threshold,
      requiring elevated |ΔNFR| (recent destabilizer) plus a stable base
      (prior IL).

    **SELF-ORGANIZATION (THOL)**: ✓ transformer
    - Spontaneous autopoietic reorganization — spawns sub-EPIs once the second
      derivative ∂²EPI/∂t² exceeds τ.
    """
    return operator in {"mutation", "self_organization"}


def triggers_bifurcation(operator: str) -> bool:
    """U4a Bifurcation-trigger test: ∂²EPI/∂t² > τ may follow the operator.

    **DISSONANCE (OZ)**: ✓ trigger — may push ∂²EPI/∂t² past τ.
    **MUTATION (ZHIR)**: ✓ trigger — a phase transformation is itself a
    bifurcation event.
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
    """Derive the U2 stabilizer set: operators that reduce |ΔNFR|."""
    return frozenset(
        op for op in _all_canonical_operator_names() if provides_negative_feedback(op)
    )


def derive_destabilizers_from_physics() -> frozenset[str]:
    """Derive the U2 destabilizer set: operators that increase |ΔNFR|."""
    return frozenset(
        op
        for op in _all_canonical_operator_names()
        if increases_structural_pressure(op)
    )


def derive_transformers_from_physics() -> frozenset[str]:
    """Derive the U4b transformer set: operators that execute bifurcations."""
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
