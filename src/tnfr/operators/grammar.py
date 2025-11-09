"""TNFR Canonical Grammar - Single Source of Truth.

This module implements the canonical TNFR grammar constraints that emerge
inevitably from TNFR physics.

All rules derive from the nodal equation ∂EPI/∂t = νf · ΔNFR(t), canonical
invariants, and formal contracts. No organizational conventions.

Canonical Constraints (U1-U4)
------------------------------
U1: STRUCTURAL INITIATION & CLOSURE
    U1a: Start with generators when needed
    U1b: End with closure operators
    Basis: ∂EPI/∂t undefined at EPI=0, sequences need coherent endpoints

U2: CONVERGENCE & BOUNDEDNESS
    If destabilizers, then include stabilizers
    Basis: ∫νf·ΔNFR dt must converge (integral convergence theorem)

U3: RESONANT COUPLING
    If coupling/resonance, then verify phase compatibility
    Basis: AGENTS.md Invariant #5 + resonance physics

U4: BIFURCATION DYNAMICS
    U4a: If bifurcation triggers, then include handlers
    U4b: If transformers, then recent destabilizer (+ prior IL for ZHIR)
    Basis: Contract OZ + bifurcation theory

For complete derivations and physics basis, see UNIFIED_GRAMMAR_RULES.md

References
----------
- UNIFIED_GRAMMAR_RULES.md: Complete physics derivations and mappings
- AGENTS.md: Canonical invariants and formal contracts
- TNFR.pdf: Nodal equation and bifurcation theory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..types import NodeId
    from .definitions import Operator

__all__ = [
    "GrammarValidator",
    "validate_grammar",
    # Operator sets
    "GENERATORS",
    "CLOSURES",
    "STABILIZERS",
    "DESTABILIZERS",
    "COUPLING_RESONANCE",
    "BIFURCATION_TRIGGERS",
    "BIFURCATION_HANDLERS",
    "TRANSFORMERS",
]


# ============================================================================
# Operator Sets (Derived from TNFR Physics)
# ============================================================================

# U1a: Generators - Create EPI from null/dormant states
GENERATORS = frozenset({"emission", "transition", "recursivity"})

# U1b: Closures - Leave system in coherent attractor states
CLOSURES = frozenset({"silence", "transition", "recursivity", "dissonance"})

# U2: Stabilizers - Provide negative feedback for convergence
STABILIZERS = frozenset({"coherence", "self_organization"})

# U2: Destabilizers - Increase |ΔNFR| (positive feedback)
DESTABILIZERS = frozenset({"dissonance", "mutation", "expansion"})

# U3: Coupling/Resonance - Require phase verification
COUPLING_RESONANCE = frozenset({"coupling", "resonance"})

# U4a: Bifurcation triggers - May initiate phase transitions
BIFURCATION_TRIGGERS = frozenset({"dissonance", "mutation"})

# U4a: Bifurcation handlers - Manage reorganization when ∂²EPI/∂t² > τ
BIFURCATION_HANDLERS = frozenset({"self_organization", "coherence"})

# U4b: Transformers - Execute structural bifurcations
TRANSFORMERS = frozenset({"mutation", "self_organization"})


class GrammarValidator:
    """Validates sequences using canonical TNFR grammar constraints.

    Implements U1-U4 rules that emerge inevitably from TNFR physics.
    This is the single source of truth for grammar validation.

    All rules derive from:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Canonical invariants (AGENTS.md §3)
    - Formal contracts (AGENTS.md §4)

    No organizational conventions are enforced.
    """

    @staticmethod
    def validate_initiation(
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, str]:
        """Validate U1a: Structural initiation.

        Physical basis: If EPI=0, then ∂EPI/∂t is undefined or zero.
        Cannot evolve structure that doesn't exist.

        Generators create structure from:
        - AL (Emission): vacuum via emission
        - NAV (Transition): latent EPI via regime shift
        - REMESH (Recursivity): dormant structure across scales

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if epi_initial > 0.0:
            # Already initialized, no generator required
            return True, "U1a: EPI>0, initiation not required"

        if not sequence:
            return False, "U1a violated: Empty sequence with EPI=0"

        first_op = getattr(sequence[0], "canonical_name", sequence[0].name.lower())

        if first_op not in GENERATORS:
            return (
                False,
                f"U1a violated: EPI=0 requires generator (got '{first_op}'). "
                f"Valid: {sorted(GENERATORS)}",
            )

        return True, f"U1a satisfied: starts with generator '{first_op}'"

    @staticmethod
    def validate_closure(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U1b: Structural closure.

        Physical basis: Sequences are bounded action potentials in structural
        space. Like physical waves, they must have termination that leaves
        system in coherent attractor states.

        Closures stabilize via:
        - SHA (Silence): Terminal closure - freezes evolution (νf → 0)
        - NAV (Transition): Handoff closure - transfers to next regime
        - REMESH (Recursivity): Recursive closure - distributes across scales
        - OZ (Dissonance): Intentional closure - preserves activation/tension

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        if not sequence:
            return False, "U1b violated: Empty sequence has no closure"

        last_op = getattr(sequence[-1], "canonical_name", sequence[-1].name.lower())

        if last_op not in CLOSURES:
            return (
                False,
                f"U1b violated: Sequence must end with closure (got '{last_op}'). "
                f"Valid: {sorted(CLOSURES)}",
            )

        return True, f"U1b satisfied: ends with closure '{last_op}'"

    @staticmethod
    def validate_convergence(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U2: Convergence and boundedness.

        Physical basis: Without stabilizers, ∫νf·ΔNFR dt → ∞ (diverges).
        Stabilizers provide negative feedback ensuring integral convergence.

        From integrated nodal equation:
            EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

        Without stabilizers:
            d(ΔNFR)/dt > 0 always → ΔNFR ~ e^(λt) → integral diverges

        With stabilizers (IL or THOL):
            d(ΔNFR)/dt can be < 0 → ΔNFR bounded → integral converges

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        # Check if sequence contains destabilizers
        destabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in DESTABILIZERS
        ]

        if not destabilizers_present:
            # No destabilizers = no divergence risk
            return True, "U2: not applicable (no destabilizers present)"

        # Check for stabilizers
        stabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in STABILIZERS
        ]

        if not stabilizers_present:
            return (
                False,
                f"U2 violated: destabilizers {destabilizers_present} present "
                f"without stabilizer. Integral ∫νf·ΔNFR dt may diverge. "
                f"Add: {sorted(STABILIZERS)}",
            )

        return (
            True,
            f"U2 satisfied: stabilizers {stabilizers_present} "
            f"bound destabilizers {destabilizers_present}",
        )

    @staticmethod
    def validate_resonant_coupling(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U3: Resonant coupling.

        Physical basis: AGENTS.md Invariant #5 states "no coupling is valid
        without explicit phase verification (synchrony)".

        Resonance physics requires phase compatibility:
            |φᵢ - φⱼ| ≤ Δφ_max

        Without phase verification:
            Nodes with incompatible phases (antiphase) could attempt coupling
            → Destructive interference → Violates resonance physics

        With phase verification:
            Only synchronous nodes couple → Constructive interference

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        U3 is a META-rule: it requires that when UM (Coupling) or RA (Resonance)
        operators are used, the implementation MUST verify phase compatibility.
        The actual phase check happens in operator preconditions.

        This grammar rule documents the requirement and ensures awareness
        that phase checks are MANDATORY (Invariant #5), not optional.
        """
        # Check if sequence contains coupling/resonance operators
        coupling_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in COUPLING_RESONANCE
        ]

        if not coupling_ops:
            # No coupling/resonance = U3 not applicable
            return True, "U3: not applicable (no coupling/resonance operators)"

        # U3 satisfied: Sequence contains coupling/resonance
        # Phase verification is MANDATORY per Invariant #5
        # Actual check happens in operator preconditions
        return (
            True,
            f"U3 awareness: operators {coupling_ops} require phase verification "
            f"(MANDATORY per Invariant #5). Enforced in preconditions.",
        )

    @staticmethod
    def validate_bifurcation_triggers(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U4a: Bifurcation triggers need handlers.

        Physical basis: AGENTS.md Contract OZ states dissonance may trigger
        bifurcation if ∂²EPI/∂t² > τ. When bifurcation is triggered, handlers
        are required to manage structural reorganization.

        Bifurcation physics:
            If ∂²EPI/∂t² > τ → multiple reorganization paths viable
            → System enters bifurcation regime
            → Requires handlers (THOL or IL) for stable transition

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)
        """
        # Check if sequence contains bifurcation triggers
        trigger_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in BIFURCATION_TRIGGERS
        ]

        if not trigger_ops:
            # No triggers = U4a not applicable
            return True, "U4a: not applicable (no bifurcation triggers)"

        # Check for handlers
        handler_ops = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in BIFURCATION_HANDLERS
        ]

        if not handler_ops:
            return (
                False,
                f"U4a violated: bifurcation triggers {trigger_ops} present "
                f"without handler. If ∂²EPI/∂t² > τ, bifurcation may occur unmanaged. "
                f"Add: {sorted(BIFURCATION_HANDLERS)}",
            )

        return (
            True,
            f"U4a satisfied: bifurcation triggers {trigger_ops} "
            f"have handlers {handler_ops}",
        )

    @staticmethod
    def validate_transformer_context(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U4b: Transformers need context.

        Physical basis: Bifurcations require threshold energy to cross
        critical points. Transformers (ZHIR, THOL) need recent destabilizers
        to provide sufficient |ΔNFR| for phase transitions.

        ZHIR (Mutation) requirements:
            1. Prior IL: Stable base prevents transformation from chaos
            2. Recent destabilizer: Threshold energy for bifurcation

        THOL (Self-organization) requirements:
            1. Recent destabilizer: Disorder to self-organize

        "Recent" = within ~3 operators (ΔNFR decays via structural relaxation)

        Parameters
        ----------
        sequence : List[Operator]
            Sequence of operators to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, message)

        Notes
        -----
        This implements "graduated destabilization" - transformers need
        sufficient ΔNFR context. The ~3 operator window captures when
        |ΔNFR| remains above bifurcation threshold.
        """
        # Check if sequence contains transformers
        transformer_ops = []
        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name in TRANSFORMERS:
                transformer_ops.append((i, op_name))

        if not transformer_ops:
            return True, "U4b: not applicable (no transformers)"

        # For each transformer, check context
        violations = []
        for idx, transformer_name in transformer_ops:
            # Check for recent destabilizer (within 3 operators before)
            window_start = max(0, idx - 3)
            recent_destabilizers = []
            prior_il = False

            for j in range(window_start, idx):
                op_name = getattr(
                    sequence[j], "canonical_name", sequence[j].name.lower()
                )
                if op_name in DESTABILIZERS:
                    recent_destabilizers.append((j, op_name))
                if op_name == "coherence":
                    prior_il = True

            # Check requirements
            if not recent_destabilizers:
                violations.append(
                    f"{transformer_name} at position {idx} lacks recent destabilizer "
                    f"(none in window [{window_start}:{idx}]). "
                    f"Need: {sorted(DESTABILIZERS)}"
                )

            # Additional requirement for ZHIR: prior IL
            if transformer_name == "mutation" and not prior_il:
                violations.append(
                    f"mutation at position {idx} lacks prior IL (coherence) "
                    f"for stable transformation base"
                )

        if violations:
            return (False, f"U4b violated: {'; '.join(violations)}")

        return (True, f"U4b satisfied: transformers have proper context")

    @classmethod
    def validate(
        cls,
        sequence: List[Operator],
        epi_initial: float = 0.0,
    ) -> tuple[bool, List[str]]:
        """Validate sequence using all unified canonical constraints.

        This validates pure TNFR physics:
        - U1: Structural initiation & closure
        - U2: Convergence & boundedness
        - U3: Resonant coupling
        - U4: Bifurcation dynamics

        Parameters
        ----------
        sequence : List[Operator]
            Sequence to validate
        epi_initial : float, optional
            Initial EPI value (default: 0.0)

        Returns
        -------
        tuple[bool, List[str]]
            (is_valid, messages)
            is_valid: True if all constraints satisfied
            messages: List of validation messages
        """
        messages = []
        all_valid = True

        # U1a: Initiation
        valid_init, msg_init = cls.validate_initiation(sequence, epi_initial)
        messages.append(f"U1a: {msg_init}")
        all_valid = all_valid and valid_init

        # U1b: Closure
        valid_closure, msg_closure = cls.validate_closure(sequence)
        messages.append(f"U1b: {msg_closure}")
        all_valid = all_valid and valid_closure

        # U2: Convergence
        valid_conv, msg_conv = cls.validate_convergence(sequence)
        messages.append(f"U2: {msg_conv}")
        all_valid = all_valid and valid_conv

        # U3: Resonant coupling
        valid_coupling, msg_coupling = cls.validate_resonant_coupling(sequence)
        messages.append(f"U3: {msg_coupling}")
        all_valid = all_valid and valid_coupling

        # U4a: Bifurcation triggers
        valid_triggers, msg_triggers = cls.validate_bifurcation_triggers(sequence)
        messages.append(f"U4a: {msg_triggers}")
        all_valid = all_valid and valid_triggers

        # U4b: Transformer context
        valid_context, msg_context = cls.validate_transformer_context(sequence)
        messages.append(f"U4b: {msg_context}")
        all_valid = all_valid and valid_context

        return all_valid, messages


def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """Validate sequence using canonical TNFR grammar constraints.

    Convenience function that returns only boolean result.
    For detailed messages, use GrammarValidator.validate().

    Parameters
    ----------
    sequence : List[Operator]
        Sequence of operators to validate
    epi_initial : float, optional
        Initial EPI value (default: 0.0)

    Returns
    -------
    bool
        True if sequence satisfies all canonical constraints

    Examples
    --------
    >>> from tnfr.operators.definitions import Emission, Coherence, Silence
    >>> ops = [Emission(), Coherence(), Silence()]
    >>> validate_grammar(ops, epi_initial=0.0)  # doctest: +SKIP
    True

    Notes
    -----
    This validator is 100% physics-based. All constraints emerge from:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - TNFR invariants (AGENTS.md §3)
    - Formal operator contracts (AGENTS.md §4)

    See UNIFIED_GRAMMAR_RULES.md for complete derivations.
    """
    is_valid, _ = GrammarValidator.validate(sequence, epi_initial)
    return is_valid
