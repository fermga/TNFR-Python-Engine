"""Public operator names and shared grammatical role sets.

``physics_derivation`` owns the predicates used to materialize U1 start/end
roles and the configured U2/U4 scalar-surrogate calibration. The historical
function names are compatibility identifiers, not proofs that the nodal
product uniquely determines event roles, timing or selection.

Emission sources form on an existing node with basal capacity; it preserves
stored capacity and pressure. Transition and Recursivity have declared
initiation/closure roles whose actual state effects depend on the execution
path. Signed scalar form is allowed, and neither role guarantees awakening
zero capacity. Silence's rate consequence depends on its configured capacity
update and excludes additive forcing; a terminal word need not be stationary.

Use the operator-contract registry for effects, ``physics_derivation`` for role
predicates, and ``theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md`` for mathematical scope.
"""

from __future__ import annotations

from typing import Any

from .physics_derivation import (
    derive_bifurcation_window_from_physics,
    derive_destabilizers_from_physics,
    derive_end_operators_from_physics,
    derive_start_operators_from_physics,
    derive_transformers_from_physics,
    derive_u2_debt_capacity_from_physics,
)

# Canonical operator identifiers (English tokens)
EMISSION = "emission"
RECEPTION = "reception"
COHERENCE = "coherence"
DISSONANCE = "dissonance"
COUPLING = "coupling"
RESONANCE = "resonance"
SILENCE = "silence"
EXPANSION = "expansion"
CONTRACTION = "contraction"
SELF_ORGANIZATION = "self_organization"
MUTATION = "mutation"
TRANSITION = "transition"
RECURSIVITY = "recursivity"

# Canonical collections -------------------------------------------------------

CANONICAL_OPERATOR_NAMES = frozenset(
    {
        EMISSION,
        RECEPTION,
        COHERENCE,
        DISSONANCE,
        COUPLING,
        RESONANCE,
        SILENCE,
        EXPANSION,
        CONTRACTION,
        SELF_ORGANIZATION,
        MUTATION,
        TRANSITION,
        RECURSIVITY,
    }
)

ALL_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES
# Backward-compatible alias; values are lowercase public executable identifiers.
ENGLISH_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES

# Materialize shared declared role predicates; do not maintain another list.
VALID_START_OPERATORS = derive_start_operators_from_physics()
INTERMEDIATE_OPERATORS = frozenset({DISSONANCE, COUPLING, RESONANCE})
VALID_END_OPERATORS = derive_end_operators_from_physics()
SELF_ORGANIZATION_CLOSURES = frozenset({SILENCE, CONTRACTION})

# R4 Bifurcation control: operators that enable structural transformations
# CANONICAL destabilizer set = {OZ, ZHIR, VAL} (dissonance, mutation, expansion),
# matching tnfr.operators.grammar_types.DESTABILIZERS (single source of truth,
# derived in physics_derivation.increases_structural_pressure).  These three
# roles act on pressure (OZ), phase (ZHIR), and capacity (VAL), respectively.
# Membership records U2 debt, not a guaranteed increase in every pressure field.
DESTABILIZERS = derive_destabilizers_from_physics()
TRANSFORMERS = derive_transformers_from_physics()
# U4b recency policy calibrated to the scalar surrogate q=1-nu_f*dt.
# The first n with q**n < 1/(pi+1) is 3 at nu_f=1, dt=0.5. This counts
# operator positions; it does not bound every graph mode's relaxation time.
BIFURCATION_WINDOW = derive_bifurcation_window_from_physics()
# U2 bookkeeping capacity uses floor(1/(nu_f*dt)) from the same surrogate,
# giving 2 at the defaults. This selected debt limit is not a physical pressure
# bound or a necessary/sufficient condition for convergence of the nodal integral.
U2_DEBT_CAPACITY = derive_u2_debt_capacity_from_physics()

# Every destabilizer in DESTABILIZERS = {OZ, ZHIR, VAL} shares the SINGLE
# configured window BIFURCATION_WINDOW. The earlier graduated reach split is
# retired. On a loopless graph without isolates trace(L_rw)/N=1 is only a mean
# eigenvalue; actual relaxation depends on the relevant spectrum and model.
# DESTABILIZERS and BIFURCATION_WINDOW own membership and the recency policy.


def canonical_operator_name(name: str) -> str:
    """Return the canonical operator token for ``name``."""

    return name


def operator_display_name(name: str) -> str:
    """Return the display label for ``name`` (currently the canonical token)."""

    return canonical_operator_name(name)


__all__ = [
    "EMISSION",
    "RECEPTION",
    "COHERENCE",
    "DISSONANCE",
    "COUPLING",
    "RESONANCE",
    "SILENCE",
    "EXPANSION",
    "CONTRACTION",
    "SELF_ORGANIZATION",
    "MUTATION",
    "TRANSITION",
    "RECURSIVITY",
    "CANONICAL_OPERATOR_NAMES",
    "ENGLISH_OPERATOR_NAMES",
    "ALL_OPERATOR_NAMES",
    "VALID_START_OPERATORS",
    "INTERMEDIATE_OPERATORS",
    "VALID_END_OPERATORS",
    "SELF_ORGANIZATION_CLOSURES",
    "DESTABILIZERS",
    "TRANSFORMERS",
    "BIFURCATION_WINDOW",
    "U2_DEBT_CAPACITY",
    "canonical_operator_name",
    "operator_display_name",
    "validate_physics_derivation",
]


def validate_physics_derivation() -> dict[str, Any]:
    """Check that public role sets agree with their shared predicate owner.

    This is an implementation-consistency check, not an independent physical
    derivation of those roles from the nodal equation.

    Returns
    -------
    dict[str, Any]
        Validation report with keys:
        - "start_operators_valid": bool
        - "end_operators_valid": bool
        - "start_operators_expected": frozenset
        - "start_operators_actual": frozenset
        - "end_operators_expected": frozenset
        - "end_operators_actual": frozenset
        - "discrepancies": list of str

    Notes
    -----
    This function is primarily for testing and validation. It ensures that
    any manual updates to VALID_START_OPERATORS or VALID_END_OPERATORS remain
    consistent with the declared shared predicates.

    If discrepancies are found, the function logs warnings but does not raise
    exceptions, allowing for intentional overrides with clear audit trail.
    """
    from .physics_derivation import (
        derive_end_operators_from_physics,
        derive_start_operators_from_physics,
    )

    expected_starts = derive_start_operators_from_physics()
    expected_ends = derive_end_operators_from_physics()

    discrepancies = []

    start_valid = VALID_START_OPERATORS == expected_starts
    if not start_valid:
        missing = expected_starts - VALID_START_OPERATORS
        extra = VALID_START_OPERATORS - expected_starts
        if missing:
            discrepancies.append(
                f"VALID_START_OPERATORS missing physics-derived operators: {missing}"
            )
        if extra:
            discrepancies.append(
                f"VALID_START_OPERATORS contains non-physics operators: {extra}"
            )

    end_valid = VALID_END_OPERATORS == expected_ends
    if not end_valid:
        missing = expected_ends - VALID_END_OPERATORS
        extra = VALID_END_OPERATORS - expected_ends
        if missing:
            discrepancies.append(
                f"VALID_END_OPERATORS missing physics-derived operators: {missing}"
            )
        if extra:
            discrepancies.append(
                f"VALID_END_OPERATORS contains non-physics operators: {extra}"
            )

    return {
        "start_operators_valid": start_valid,
        "end_operators_valid": end_valid,
        "start_operators_expected": expected_starts,
        "start_operators_actual": VALID_START_OPERATORS,
        "end_operators_expected": expected_ends,
        "end_operators_actual": VALID_END_OPERATORS,
        "discrepancies": discrepancies,
    }


def __getattr__(name: str) -> Any:
    """Provide a consistent ``AttributeError`` when names are missing."""

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
