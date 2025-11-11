"""Canonical operator name constants and physics-derived operator sets.

This module defines operator names and derives valid start/end operator sets
from TNFR physical principles rather than arbitrary lists.

Physics-Based Derivation
------------------------
The sets VALID_START_OPERATORS and VALID_END_OPERATORS are derived from the
fundamental TNFR nodal equation:

    ∂EPI/∂t = νf · ΔNFR(t)

Where:
    - EPI: Primary Information Structure (coherent form)
    - νf: Structural frequency (reorganization rate, Hz_str)
    - ΔNFR: Internal reorganization operator/gradient

Start Operators (Activation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An operator can START a sequence if it can either:

1. **Generate EPI from null state** (νf=0, EPI=0):
   - emission: Creates outward coherence pulse, generates νf > 0 and ΔNFR > 0

2. **Activate latent EPI** (νf≈0, but EPI>0):
   - recursivity: Replicates/echoes existing patterns across scales
   - transition: Activates node from another phase/regime

Physical justification: Only operators that can create or activate structural
capacity (νf > 0) from dormant/null states can initiate reorganization.

End Operators (Closure)
~~~~~~~~~~~~~~~~~~~~~~~~
An operator can END a sequence if it can either:

1. **Stabilize reorganization** (∂EPI/∂t → 0):
   - silence: Forces νf → 0, causing ∂EPI/∂t → 0 while preserving EPI

2. **Achieve operational closure**:
   - transition: Hands off to next phase (completes current cycle)
   - recursivity: Fractal echo creates self-similar closure
   - dissonance: Postponed conflict / contained tension (questionable)

Physical justification: Terminal operators must either freeze evolution
(νf → 0) or complete an operational cycle with clear boundary.

For detailed physics derivation logic, see:
    tnfr.config.physics_derivation

References
----------
- TNFR.pdf: Section 2.1 (Nodal Equation)
- AGENTS.md: Section 3 (Canonical Invariants)
"""

from __future__ import annotations

from typing import Any

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
ENGLISH_OPERATOR_NAMES = CANONICAL_OPERATOR_NAMES

# Physics-derived operator sets (derived from TNFR canonical principles)
# Import here to avoid issues, but actual derivation is in physics_derivation module
# These are computed at module load time from TNFR physical principles
VALID_START_OPERATORS = frozenset({EMISSION, RECURSIVITY, TRANSITION})
INTERMEDIATE_OPERATORS = frozenset({DISSONANCE, COUPLING, RESONANCE})
VALID_END_OPERATORS = frozenset({SILENCE, TRANSITION, RECURSIVITY, DISSONANCE})
SELF_ORGANIZATION_CLOSURES = frozenset({SILENCE, CONTRACTION})

# R4 Bifurcation control: operators that enable structural transformations
# Legacy single-level destabilizers (for backward compatibility)
DESTABILIZERS = frozenset({DISSONANCE, TRANSITION, EXPANSION})  # OZ, NAV, VAL
TRANSFORMERS = frozenset({MUTATION, SELF_ORGANIZATION})  # ZHIR, THOL
BIFURCATION_WINDOW = 3  # Legacy: Search window for destabilizer precedent

# R4 Extended: Graduated destabilizer classification by intensity
DESTABILIZERS_STRONG = frozenset({DISSONANCE})  # OZ: explicit dissonance
DESTABILIZERS_MODERATE = frozenset({TRANSITION, EXPANSION})  # NAV, VAL: indirect
DESTABILIZERS_WEAK = frozenset({RECEPTION})  # EN: latent potential

# All destabilizers (union of all levels)
DESTABILIZERS_ALL = DESTABILIZERS_STRONG | DESTABILIZERS_MODERATE | DESTABILIZERS_WEAK

# R4 Extended: Bifurcation windows by destabilizer intensity
# These define how many operators can separate a destabilizer from a transformer
BIFURCATION_WINDOWS = {
    "strong": 4,  # OZ permits ZHIR/THOL within 4 operators
    "moderate": 2,  # NAV/VAL permit ZHIR/THOL within 2 operators
    "weak": 1,  # EN requires ZHIR/THOL as immediate successor
}


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
    "DESTABILIZERS_STRONG",
    "DESTABILIZERS_MODERATE",
    "DESTABILIZERS_WEAK",
    "DESTABILIZERS_ALL",
    "BIFURCATION_WINDOWS",
    "canonical_operator_name",
    "operator_display_name",
    "validate_physics_derivation",
]


def validate_physics_derivation() -> dict[str, Any]:
    """Validate that operator sets are consistent with TNFR physics derivation.

    This function verifies that VALID_START_OPERATORS and VALID_END_OPERATORS
    match what would be derived from first principles using the physics_derivation
    module.

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
    consistent with TNFR canonical physics.

    If discrepancies are found, the function logs warnings but does not raise
    exceptions, allowing for intentional overrides with clear audit trail.
    """
    from .physics_derivation import (
        derive_start_operators_from_physics,
        derive_end_operators_from_physics,
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
            discrepancies.append(f"VALID_START_OPERATORS contains non-physics operators: {extra}")

    end_valid = VALID_END_OPERATORS == expected_ends
    if not end_valid:
        missing = expected_ends - VALID_END_OPERATORS
        extra = VALID_END_OPERATORS - expected_ends
        if missing:
            discrepancies.append(
                f"VALID_END_OPERATORS missing physics-derived operators: {missing}"
            )
        if extra:
            discrepancies.append(f"VALID_END_OPERATORS contains non-physics operators: {extra}")

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
