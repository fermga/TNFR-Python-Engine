"""Physics-based derivation of canonical start/end operators from TNFR principles.

This module derives which operators can validly start or end sequences based on
the fundamental TNFR nodal equation and structural coherence principles, rather
than using arbitrary static lists.

Core TNFR Equation
------------------
∂EPI/∂t = νf · ΔNFR(t)

Where:
- EPI: Primary Information Structure (coherent form)
- νf: Structural frequency (reorganization rate, Hz_str)
- ΔNFR: Internal reorganization operator/gradient

Node Activation Conditions
---------------------------
A node activates (exists structurally) when:
1. νf > 0 (has reorganization capacity)
2. ΔNFR ≠ 0 (has structural pressure)
3. EPI ≥ ε (minimum coherence threshold)

Node Termination Conditions
----------------------------
A sequence terminates coherently when:
1. ∂EPI/∂t → 0 (reorganization stabilizes)
2. EPI remains stable (coherence sustained)
3. No open transitions (operational closure)
"""

from __future__ import annotations

from typing import FrozenSet

__all__ = [
    "derive_start_operators_from_physics",
    "derive_end_operators_from_physics",
    "can_generate_epi_from_null",
    "can_activate_latent_epi",
    "can_stabilize_reorganization",
    "achieves_operational_closure",
]


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


def derive_start_operators_from_physics() -> FrozenSet[str]:
    """Derive valid start operators from TNFR physical principles.

    A sequence can start with an operator if it satisfies at least one:
    1. Can generate EPI from null state (generative capacity)
    2. Can activate latent/dormant EPI (activation capacity)

    Returns
    -------
    frozenset[str]
        Set of canonical operator names that can validly start sequences

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
    from .operator_names import (
        EMISSION,
        RECURSIVITY,
        TRANSITION,
    )

    generators = {EMISSION}  # Can create EPI from null
    activators = {RECURSIVITY, TRANSITION}  # Can activate latent EPI

    # A valid start operator must be either a generator or activator
    valid_starts = generators | activators

    return frozenset(valid_starts)


def derive_end_operators_from_physics() -> FrozenSet[str]:
    """Derive valid end operators from TNFR physical principles.

    A sequence can end with an operator if it satisfies at least one:
    1. Stabilizes reorganization (∂EPI/∂t → 0)
    2. Achieves operational closure (completes cycle)

    Returns
    -------
    frozenset[str]
        Set of canonical operator names that can validly end sequences

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
    from .operator_names import (
        SILENCE,
        TRANSITION,
        RECURSIVITY,
        DISSONANCE,
    )

    stabilizers = {SILENCE}  # Forces ∂EPI/∂t → 0
    closures = {TRANSITION, RECURSIVITY}  # Completes operational cycles

    # DISSONANCE is questionable but included for backward compatibility
    # Represents postponed conflict / contained tension patterns
    questionable = {DISSONANCE}

    valid_ends = stabilizers | closures | questionable

    return frozenset(valid_ends)
