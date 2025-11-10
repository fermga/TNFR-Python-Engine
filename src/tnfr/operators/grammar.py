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

from enum import Enum
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph, Glyph
    from .definitions import Operator
else:
    from ..types import Glyph


class StructuralPattern(Enum):
    """Classification of structural patterns in TNFR sequences.
    
    Used by canonical_patterns module for backward compatibility.
    Deprecated - use pattern_detection module for new code.
    """
    BIFURCATED = "bifurcated"
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    COMPLEX = "complex"
    COMPRESS = "compress"
    EXPLORE = "explore"
    RESONATE = "resonate"


# ============================================================================
# Glyph-Function Name Mappings
# ============================================================================

# Mapping from Glyph to canonical function name
GLYPH_TO_FUNCTION = {
    Glyph.AL: "emission",
    Glyph.EN: "reception",
    Glyph.IL: "coherence",
    Glyph.OZ: "dissonance",
    Glyph.UM: "coupling",
    Glyph.RA: "resonance",
    Glyph.SHA: "silence",
    Glyph.VAL: "expansion",
    Glyph.NUL: "contraction",
    Glyph.THOL: "self_organization",
    Glyph.ZHIR: "mutation",
    Glyph.NAV: "transition",
    Glyph.REMESH: "recursivity",
}

# Reverse mapping from function name to Glyph
FUNCTION_TO_GLYPH = {v: k for k, v in GLYPH_TO_FUNCTION.items()}


def glyph_function_name(
    val: Any,
    *,
    default: Any = None,
) -> Any:
    """Convert glyph to canonical function name.
    
    Parameters
    ----------
    val : Glyph | str | None
        Glyph enum, glyph string value ('IL', 'OZ'), or function name to convert
    default : str | None, optional
        Default value if conversion fails
        
    Returns
    -------
    str | None
        Canonical function name or default
        
    Notes
    -----
    Glyph enum inherits from str, so we must check for Enum type
    BEFORE checking isinstance(val, str), otherwise Glyph instances
    will be returned unchanged instead of being converted.
    
    The function handles three input types:
    1. Glyph enum (e.g., Glyph.IL) → function name (e.g., 'coherence')
    2. Glyph string value (e.g., 'IL') → function name (e.g., 'coherence')  
    3. Function name (e.g., 'coherence') → returned as-is
    """
    if val is None:
        return default
    # Check for Glyph/Enum BEFORE str (Glyph inherits from str)
    if isinstance(val, Enum):
        return GLYPH_TO_FUNCTION.get(val, default)
    if isinstance(val, str):
        # Check if it's a glyph string value ('IL', 'OZ', etc)
        # Build reverse lookup on first use
        if not hasattr(glyph_function_name, '_glyph_value_map'):
            glyph_function_name._glyph_value_map = {
                g.value: func for g, func in GLYPH_TO_FUNCTION.items()
            }
        # Try to convert glyph value to function name
        func_name = glyph_function_name._glyph_value_map.get(val)
        if func_name:
            return func_name
        # Otherwise assume it's already a function name
        return val
    return GLYPH_TO_FUNCTION.get(val, default)


def function_name_to_glyph(
    val: Any,
    *,
    default: Any = None,
) -> Any:
    """Convert function name to glyph.
    
    Parameters
    ----------
    val : str | Glyph | None
        Function name or glyph to convert
    default : Glyph | None, optional
        Default value if conversion fails
        
    Returns
    -------
    Glyph | None
        Glyph or default
    """
    if val is None:
        return default
    if isinstance(val, Glyph):
        return val
    return FUNCTION_TO_GLYPH.get(val, default)


__all__ = [
    "GrammarValidator",
    "GrammarContext",
    "validate_grammar",
    "StructuralPattern",
    # Error classes
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "GrammarConfigurationError",
    "record_grammar_violation",
    # Glyph mappings
    "GLYPH_TO_FUNCTION",
    "FUNCTION_TO_GLYPH",
    "glyph_function_name",
    "function_name_to_glyph",
    # Grammar application functions
    "apply_glyph_with_grammar",
    "on_applied_glyph",
    "enforce_canonical_grammar",  # Deprecated stub for compatibility
    # Sequence validation (deprecated stubs for compatibility)
    "validate_sequence",
    "parse_sequence",
    # Operator sets
    "GENERATORS",
    "CLOSURES",
    "STABILIZERS",
    "DESTABILIZERS",
    "COUPLING_RESONANCE",
    "BIFURCATION_TRIGGERS",
    "BIFURCATION_HANDLERS",
    "TRANSFORMERS",
    "RECURSIVE_GENERATORS",
    "SCALE_STABILIZERS",
]


# ============================================================================
# Operator Sets (Derived from TNFR Physics)
# ============================================================================

# U1a: Generators - Create EPI from null/dormant states
GENERATORS = frozenset({"emission", "transition", "recursivity"})

# U1b: Closures - Leave system in coherent attractor states
CLOSURES = frozenset({"silence", "transition", "recursivity", "dissonance"})

# U2: Stabilizers - Provide negative feedback for convergence
STABILIZERS = frozenset({"coherence", "self_organization", "reception"})

# U2: Destabilizers - Increase |ΔNFR| (positive feedback)
DESTABILIZERS = frozenset({"dissonance", "mutation", "expansion", "contraction"})

# U3: Coupling/Resonance - Require phase verification
COUPLING_RESONANCE = frozenset({"coupling", "resonance"})

# U4a: Bifurcation triggers - May initiate phase transitions
BIFURCATION_TRIGGERS = frozenset({"dissonance", "mutation"})

# U4a: Bifurcation handlers - Manage reorganization when ∂²EPI/∂t² > τ
BIFURCATION_HANDLERS = frozenset({"self_organization", "coherence"})

# U4b: Transformers - Execute structural bifurcations
TRANSFORMERS = frozenset({"mutation", "self_organization"})

# U5: Multi-Scale Coherence - Recursive generators and scale stabilizers
RECURSIVE_GENERATORS = frozenset({"recursivity"})
SCALE_STABILIZERS = frozenset({"coherence", "self_organization"})


# ============================================================================
# Grammar Errors
# ============================================================================


class StructuralGrammarError(RuntimeError):
    """Base class for structural grammar violations.
    
    Attributes
    ----------
    rule : str
        Grammar rule that was violated
    candidate : str
        Operator/glyph that caused violation
    message : str
        Error description
    window : int | None
        Grammar window if applicable
    threshold : float | None
        Threshold value if applicable
    order : Sequence[str] | None
        Operator sequence if applicable
    context : dict
        Additional context information
    """
    
    def __init__(
        self,
        *,
        rule: str,
        candidate: str,
        message: str,
        window: int | None = None,
        threshold: float | None = None,
        order: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.rule = rule
        self.candidate = candidate
        self.message = message
        self.window = window
        self.threshold = threshold
        self.order = order
        self.context = context or {}
        super().__init__(message)
    
    def attach_context(self, **context: Any) -> "StructuralGrammarError":
        """Attach additional context to error.
        
        Parameters
        ----------
        **context : Any
            Additional context key-value pairs
            
        Returns
        -------
        StructuralGrammarError
            Self for chaining
        """
        self.context.update(context)
        return self
    
    def to_payload(self) -> dict[str, Any]:
        """Convert error to dictionary payload.
        
        Returns
        -------
        dict
            Error information as dictionary
        """
        return {
            "rule": self.rule,
            "candidate": self.candidate,
            "message": self.message,
            "window": self.window,
            "threshold": self.threshold,
            "order": self.order,
            "context": self.context,
        }


class RepeatWindowError(StructuralGrammarError):
    """Error for repeated operator within window."""
    pass


class MutationPreconditionError(StructuralGrammarError):
    """Error for mutation without proper preconditions."""
    pass


class TholClosureError(StructuralGrammarError):
    """Error for THOL without proper closure."""
    pass


class TransitionCompatibilityError(StructuralGrammarError):
    """Error for incompatible transition."""
    pass


class SequenceSyntaxError(ValueError):
    """Error in sequence syntax.
    
    Attributes
    ----------
    index : int
        Position in sequence where error occurred
    token : object
        Token that caused the error
    message : str
        Error description
    """
    
    def __init__(self, index: int, token: Any, message: str):
        self.index = index
        self.token = token
        self.message = message
        super().__init__(f"At index {index}, token '{token}': {message}")


class GrammarConfigurationError(ValueError):
    """Error in grammar configuration.
    
    Attributes
    ----------
    section : str
        Configuration section with error
    messages : list[str]
        Error messages
    details : list[tuple[str, str]]
        Additional details
    """
    
    def __init__(
        self,
        section: str,
        messages: list[str],
        *,
        details: list[tuple[str, str]] | None = None,
    ):
        self.section = section
        self.messages = messages
        self.details = details or []
        msg = f"Configuration error in {section}: {'; '.join(messages)}"
        super().__init__(msg)


def record_grammar_violation(
    G: "TNFRGraph",
    node: "NodeId",
    error: StructuralGrammarError,
    *,
    stage: str,
) -> None:
    """Record grammar violation in node metadata.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing node
    node : NodeId
        Node where violation occurred
    error : StructuralGrammarError
        Grammar error to record
    stage : str
        Processing stage when error occurred
    """
    if "grammar_violations" not in G.nodes[node]:
        G.nodes[node]["grammar_violations"] = []
    G.nodes[node]["grammar_violations"].append({
        "stage": stage,
        "error": error.to_payload(),
    })


# ============================================================================
# Grammar Context
# ============================================================================


class GrammarContext:
    """Context object for grammar validation.
    
    Minimal implementation for import compatibility.
    
    Attributes
    ----------
    G : TNFRGraph
        Graph being validated
    cfg_soft : dict
        Soft configuration parameters
    cfg_canon : dict
        Canonical configuration parameters
    norms : dict
        Normalization parameters
    """
    
    def __init__(
        self,
        G: "TNFRGraph",
        cfg_soft: dict[str, Any] | None = None,
        cfg_canon: dict[str, Any] | None = None,
        norms: dict[str, Any] | None = None,
    ):
        self.G = G
        self.cfg_soft = cfg_soft or {}
        self.cfg_canon = cfg_canon or {}
        self.norms = norms or {}
    
    @classmethod
    def from_graph(cls, G: "TNFRGraph") -> "GrammarContext":
        """Create context from graph.
        
        Parameters
        ----------
        G : TNFRGraph
            Graph to create context from
            
        Returns
        -------
        GrammarContext
            New context instance
        """
        return cls(G)


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

    @staticmethod
    def validate_remesh_amplification(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U2-REMESH: Recursive amplification control.

        Physical basis: REMESH implements temporal coupling EPI(t) ↔ EPI(t-τ)
        which creates feedback that amplifies structural changes. When combined
        with destabilizers, this can cause unbounded growth.

        From integrated nodal equation:
            EPI(t_f) = EPI(t_0) + ∫_{t_0}^{t_f} νf·ΔNFR dτ

        REMESH temporal mixing:
            EPI_mixed = (1-α)·EPI_now + α·EPI_past

        Without stabilizers:
            REMESH + destabilizers → recursive amplification
            → ∫ νf·ΔNFR dt → ∞ (feedback loop)
            → System fragments

        With stabilizers:
            IL or THOL provides negative feedback
            → Bounded recursive evolution
            → ∫ νf·ΔNFR dt < ∞

        Specific combinations requiring stabilizers:
            - REMESH + VAL: Recursive expansion needs coherence stabilization
            - REMESH + OZ: Recursive bifurcation needs self-organization handlers
            - REMESH + ZHIR: Replicative mutation needs coherence consolidation

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
        This rule is DISTINCT from general U2 (convergence). While U2 checks
        for destabilizers needing stabilizers, U2-REMESH specifically addresses
        REMESH's amplification property: it multiplies the effect of destabilizers
        through recursive feedback across temporal/spatial scales.

        Physical derivation: See src/tnfr/operators/remesh.py module docstring,
        section "Grammar Implications from Physical Analysis" → U2: CONVERGENCE.
        """
        # Check if sequence contains REMESH
        has_remesh = any(
            getattr(op, "canonical_name", op.name.lower()) == "recursivity"
            for op in sequence
        )

        if not has_remesh:
            return True, "U2-REMESH: not applicable (no recursivity present)"

        # Check for destabilizers
        destabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in DESTABILIZERS
        ]

        if not destabilizers_present:
            return True, "U2-REMESH: satisfied (no destabilizers to amplify)"

        # Check for stabilizers
        stabilizers_present = [
            getattr(op, "canonical_name", op.name.lower())
            for op in sequence
            if getattr(op, "canonical_name", op.name.lower()) in STABILIZERS
        ]

        if not stabilizers_present:
            return (
                False,
                f"U2-REMESH violated: recursivity amplifies destabilizers "
                f"{destabilizers_present} via recursive feedback. "
                f"Integral ∫νf·ΔNFR dt may diverge (unbounded growth). "
                f"Required: {sorted(STABILIZERS)} to bound recursive amplification",
            )

        return (
            True,
            f"U2-REMESH satisfied: stabilizers {stabilizers_present} "
            f"bound recursive amplification of {destabilizers_present}",
        )

    @staticmethod
    def validate_multiscale_coherence(sequence: List[Operator]) -> tuple[bool, str]:
        """Validate U5: Multi-scale coherence preservation.

        Physical basis: Multi-scale hierarchical structures created by REMESH
        with depth>1 require coherence conservation across scales. This emerges
        inevitably from the nodal equation applied to hierarchical systems.

        From the nodal equation at each hierarchical level:
            ∂EPI_parent/∂t = νf_parent · ΔNFR_parent(t)
            ∂EPI_child_i/∂t = νf_child_i · ΔNFR_child_i(t)  for each child i

        For hierarchical systems with N children:
            EPI_parent = f(EPI_child_1, ..., EPI_child_N)  (structural coupling)

        Taking time derivative and applying chain rule:
            ∂EPI_parent/∂t = Σ (∂f/∂EPI_child_i) · ∂EPI_child_i/∂t
                           = Σ w_i · νf_child_i · ΔNFR_child_i(t)

        where w_i = ∂f/∂EPI_child_i are coupling weights.

        Equating with nodal equation for parent:
            νf_parent · ΔNFR_parent = Σ w_i · νf_child_i · ΔNFR_child_i

        For coherence C(t) = measure of structural stability:
            C_parent ~ 1/|ΔNFR_parent|  (lower pressure = higher coherence)
            C_child_i ~ 1/|ΔNFR_child_i|

        This gives the conservation inequality:
            C_parent ≥ α · Σ C_child_i

        Where α = (1/√N) · η_phase(N) · η_coupling(N) captures:
        - 1/√N: Scale factor from coupling weight distribution
        - η_phase: Phase synchronization efficiency (U3 requirement)
        - η_coupling: Structural coupling efficiency losses
        - Typical range: α ∈ [0.1, 0.4]

        Without stabilizers:
            Deep REMESH (depth>1) creates nested EPIs
            → ΔNFR_parent grows from uncoupled child fluctuations
            → C_parent decreases below α·ΣC_child
            → Violation of conservation → System fragments

        With stabilizers (IL or THOL):
            IL/THOL reduce |ΔNFR| at each level (direct from operator contracts)
            → Maintains C_parent ≥ α·ΣC_child at all hierarchical levels
            → Conservation preserved → Bounded multi-scale evolution

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
        U5 is INDEPENDENT of U2+U4b:
        - U2/U4b: TEMPORAL dimension (operator sequences in time)
        - U5: SPATIAL dimension (hierarchical nesting in structure)

        Decision test case that passes U2+U4b but fails U5:
            [AL, REMESH(depth=3), SHA]
            - U2: ✓ No destabilizers (trivially convergent)
            - U4b: ✓ REMESH not a transformer (U4b doesn't apply)
            - U5: ✗ Deep recursivity without stabilization → fragmentation

        Physical derivation: See UNIFIED_GRAMMAR_RULES.md § U5
        Canonicity: STRONG (derived from nodal equation + structural coupling)

        References
        ----------
        - TNFR.pdf § 2.1: Nodal equation ∂EPI/∂t = νf · ΔNFR(t)
        - Problem statement: "El pulso que nos atraviesa.pdf"
        - AGENTS.md: Invariant #7 (Operational Fractality)
        - Contract IL: Reduces |ΔNFR| at all scales
        - Contract THOL: Autopoietic closure across hierarchical levels
        """
        # Check for deep REMESH (depth > 1)
        # Note: Currently Recursivity doesn't expose depth parameter in operator
        # This is a forward-looking validation for when depth is added
        deep_remesh_indices = []
        
        for i, op in enumerate(sequence):
            op_name = getattr(op, "canonical_name", op.name.lower())
            if op_name == "recursivity":
                # Check if operator has depth attribute
                depth = getattr(op, "depth", 1)  # Default depth=1 if not present
                if depth > 1:
                    deep_remesh_indices.append((i, depth))

        if not deep_remesh_indices:
            # No deep REMESH present, U5 not applicable
            return True, "U5: not applicable (no deep recursivity depth>1 present)"

        # For each deep REMESH, check for stabilizers in window
        violations = []
        for idx, depth in deep_remesh_indices:
            # Check window of ±3 operators for scale stabilizers
            window_start = max(0, idx - 3)
            window_end = min(len(sequence), idx + 4)
            
            has_stabilizer = False
            stabilizers_in_window = []
            
            for j in range(window_start, window_end):
                op_name = getattr(
                    sequence[j], "canonical_name", sequence[j].name.lower()
                )
                if op_name in SCALE_STABILIZERS:
                    has_stabilizer = True
                    stabilizers_in_window.append((j, op_name))

            if not has_stabilizer:
                violations.append(
                    f"recursivity at position {idx} (depth={depth}) lacks scale "
                    f"stabilizer in window [{window_start}:{window_end}]. "
                    f"Deep hierarchical nesting requires {sorted(SCALE_STABILIZERS)} "
                    f"for multi-scale coherence preservation (C_parent ≥ α·ΣC_child)"
                )

        if violations:
            return (False, f"U5 violated: {'; '.join(violations)}")

        return (
            True,
            f"U5 satisfied: deep recursivity has scale stabilizers "
            f"for multi-scale coherence preservation",
        )

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
        - U5: Multi-scale coherence

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

        # U2-REMESH: Recursive amplification control
        valid_remesh, msg_remesh = cls.validate_remesh_amplification(sequence)
        messages.append(f"U2-REMESH: {msg_remesh}")
        all_valid = all_valid and valid_remesh

        # U5: Multi-scale coherence
        valid_multiscale, msg_multiscale = cls.validate_multiscale_coherence(sequence)
        messages.append(f"U5: {msg_multiscale}")
        all_valid = all_valid and valid_multiscale

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


# ============================================================================
# Grammar Application Functions (Minimal Stubs for Import Compatibility)
# ============================================================================


def apply_glyph_with_grammar(
    G: "TNFRGraph",
    nodes: Any,
    glyph: Any,
    window: Any = None,
) -> None:
    """Apply glyph to nodes with grammar validation.
    
    Applies the specified glyph to each node in the iterable using the canonical
    TNFR operator implementation.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing nodes
    nodes : Any
        Node, list of nodes, or node iterable to apply glyph to
    glyph : Any
        Glyph to apply
    window : Any, optional
        Grammar window constraint
        
    Notes
    -----
    This function delegates to apply_glyph for each node, which wraps
    the node in NodeNX and applies the glyph operation.
    """
    from . import apply_glyph
    
    # Handle single node or iterable of nodes
    # Check if it's a single hashable node or an iterable
    try:
        # Try to treat as single hashable node
        hash(nodes)
        # If hashable, it's a single node
        nodes_iter = [nodes]
    except (TypeError, AttributeError):
        # Not hashable, treat as iterable
        # Convert to list to allow multiple iterations if needed
        try:
            nodes_iter = list(nodes)
        except TypeError:
            # If not iterable, wrap in list
            nodes_iter = [nodes]
    
    for node in nodes_iter:
        apply_glyph(G, node, glyph, window=window)


def on_applied_glyph(G: "TNFRGraph", n: "NodeId", applied: Any) -> None:
    """Record glyph application in node history.
    
    Minimal stub for tracking operator sequences.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing node
    n : NodeId
        Node identifier
    applied : Any
        Applied glyph or operator name
    """
    # Minimal stub for telemetry
    if "glyph_history" not in G.nodes[n]:
        G.nodes[n]["glyph_history"] = []
    G.nodes[n]["glyph_history"].append(applied)





def enforce_canonical_grammar(
    G: "TNFRGraph",
    n: "NodeId",
    cand: Any,
    ctx: Any = None,
) -> Any:
    """Minimal stub for backward compatibility.
    
    This function is a no-op stub maintained for compatibility with existing
    code that expects this interface. It simply returns the candidate as-is.
    
    For actual grammar validation, use validate_grammar() from unified_grammar.
    
    Parameters
    ----------
    G : TNFRGraph
        Graph containing node
    n : NodeId
        Node identifier  
    cand : Any
        Candidate glyph/operator
    ctx : Any, optional
        Grammar context (ignored)
        
    Returns
    -------
    Any
        The candidate unchanged
    """
    return cand

# ============================================================================


def validate_sequence(
    names: Any = None,
    **kwargs: Any,
) -> Any:
    """DEPRECATED: Minimal stub for backward compatibility only.
    
    This function exists only for import compatibility with legacy code.
    It returns a mock success result.
    
    For actual grammar validation, use validate_grammar() from unified_grammar module.
    
    Parameters
    ----------
    names : Iterable[str] | object, optional
        Sequence of operator names (ignored)
    **kwargs : Any
        Additional validation options (ignored)
        
    Returns
    -------
    ValidationOutcome
        Mock validation result (always passes)
    """
    class ValidationStub:
        def __init__(self):
            self.passed = True
            self.message = "Validation stub - use validate_grammar() instead"
            self.metadata = {}
    return ValidationStub()


def parse_sequence(names: Any) -> Any:
    """DEPRECATED: Minimal stub for backward compatibility only.
    
    This function exists only for import compatibility with legacy code.
    
    For actual grammar operations, use the unified_grammar module.
    
    Parameters
    ----------
    names : Iterable[str]
        Sequence of operator names
        
    Returns
    -------
    SequenceValidationResult
        Mock parse result
    """
    class ParseStub:
        def __init__(self):
            self.tokens = list(names) if names else []
            self.canonical_tokens = self.tokens
            self.passed = True
            self.message = "Parse stub - use unified grammar instead"
            self.metadata = {}
            self.error = None
    return ParseStub()

# Grammar Validator Class
# ============================================================================
