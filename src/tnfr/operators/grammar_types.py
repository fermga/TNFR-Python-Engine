"""TNFR Grammar: Types and Exceptions

Enums, exception classes, and validation result types for TNFR grammar.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..constants.canonical import PHI  # Golden ratio for U6 escape threshold

if TYPE_CHECKING:
    from ..types import Glyph, NodeId, TNFRGraph
else:
    NodeId = Any
    TNFRGraph = Any
    from ..types import Glyph

from ..config.physics_derivation import (
    derive_bifurcation_handlers_from_physics as _derive_handlers,
)
from ..config.physics_derivation import (
    derive_bifurcation_triggers_from_physics as _derive_triggers,
)
from ..config.physics_derivation import (
    derive_destabilizers_from_physics as _derive_destabilizers,
)
from ..config.physics_derivation import (
    derive_end_operators_from_physics as _derive_ends,
)
from ..config.physics_derivation import (
    derive_stabilizers_from_physics as _derive_stabilizers,
)
from ..config.physics_derivation import (
    derive_start_operators_from_physics as _derive_starts,
)
from ..config.physics_derivation import (
    derive_transformers_from_physics as _derive_transformers,
)
from ..validation.base import ValidationOutcome

# ============================================================================
# Operator Sets (Derived from TNFR Physics)
# ============================================================================
#
# SINGLE SOURCE OF TRUTH.  These sets are DERIVED from the per-operator
# nodal-equation predicates in ``tnfr.config.physics_derivation`` — they are not
# hand-maintained frozensets.  Every grammar consumer (the U1-U6 validator,
# grammar_dynamics, grammar_patterns, the runtime preconditions) must import
# from here.  The derivation rationale (why each operator is a generator /
# closure / stabilizer / destabilizer / transformer) lives in the predicate
# docstrings of physics_derivation, grounded in ∂EPI/∂t = νf·ΔNFR.


# U1a: Generators - Create EPI from null/dormant states (AL, NAV, REMESH)
GENERATORS = _derive_starts()

# U1b: Closures - Leave system in coherent attractor states (SHA, NAV, REMESH, OZ)
CLOSURES = _derive_ends()

# U2: Stabilizers - Reduce |ΔNFR| (negative feedback → integral converges): IL, THOL
STABILIZERS = _derive_stabilizers()

# U2: Destabilizers - Increase |ΔNFR| (positive feedback): OZ, ZHIR, VAL
DESTABILIZERS = _derive_destabilizers()

# U3: Coupling/Resonance - Require phase verification
COUPLING_RESONANCE = frozenset({"coupling", "resonance"})

# U4a: Bifurcation triggers - May initiate phase transitions (OZ, ZHIR)
BIFURCATION_TRIGGERS = _derive_triggers()

# U4a: Bifurcation handlers - Manage reorganization when ∂²EPI/∂t² > τ (THOL, IL)
BIFURCATION_HANDLERS = _derive_handlers()

# U4b: Transformers - Execute structural bifurcations (ZHIR, THOL)
TRANSFORMERS = _derive_transformers()

# U5: Multi-Scale Coherence - Recursive generators and scale stabilizers
RECURSIVE_GENERATORS = frozenset({"recursivity"})
SCALE_STABILIZERS = STABILIZERS


class StructuralPattern(Enum):
    """Classification labels for TNFR operator sequences.

    Two distinct axes are mixed in this legacy enum. The CANONICAL grammar
    typology is the structural-shape axis — exactly five types from TNFR.pdf
    §2.3 "Tabla comparativa de estructuras glíficas" (LINEAR, BIFURCATED,
    FRACTAL, CYCLIC, HIERARCHICAL), mirrored by
    :class:`tnfr.operators.grammar_canon.StructuralType`. The remaining members
    are NON-canonical heuristic application labels (domain, learning process,
    operational meta) kept for backward compatibility; they do not emerge from
    the nodal dynamics and are not part of the canonical grammar typology.

    For canonical structural classification use ``grammar_canon.StructuralType``
    and ``CANONICAL_STRUCTURAL_TYPES`` below; ``grammar_canon`` also provides
    the mapping from any label here to its canonical structural type.
    """

    # --- CANONICAL structural typology (TNFR.pdf "Tabla comparativa") ---------
    BIFURCATED = "bifurcated"  # Bifurcada — OZ → [ZHIR | NUL] (union)
    LINEAR = "linear"  # Lineal — concatenation
    HIERARCHICAL = "hierarchical"  # Jerárquica — nested THOL[...] (Dyck)
    FRACTAL = "fractal"  # Fractal — self-similar repeat (star)
    CYCLIC = "cyclic"  # Cíclica — close-and-reopen feedback cycle

    # --- LEGACY heuristic labels (NON-canonical application metadata) ---------
    # Application domain axis (not a structural shape):
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    ORGANIZATIONAL = "organizational"
    CREATIVE = "creative"
    REGENERATIVE = "regenerative"
    # Operational meta axis (reduce to a canonical type via grammar_canon):
    COMPLEX = "complex"
    COMPRESS = "compress"
    EXPLORE = "explore"
    RESONATE = "resonate"
    BOOTSTRAP = "bootstrap"
    STABILIZE = "stabilize"
    # Learning-process axis (not a structural shape):
    BASIC_LEARNING = "basic_learning"
    DEEP_LEARNING = "deep_learning"
    EXPLORATORY_LEARNING = "exploratory_learning"
    CONSOLIDATION_CYCLE = "consolidation_cycle"
    ADAPTIVE_MUTATION = "adaptive_mutation"
    UNKNOWN = "unknown"


#: The five canonical structural types (TNFR.pdf "Tabla comparativa"), the only
#: members of :class:`StructuralPattern` that constitute the canonical grammar
#: typology. Mirrored by :class:`grammar_canon.StructuralType`.
CANONICAL_STRUCTURAL_TYPES = frozenset(
    {
        StructuralPattern.LINEAR,
        StructuralPattern.BIFURCATED,
        StructuralPattern.FRACTAL,
        StructuralPattern.CYCLIC,
        StructuralPattern.HIERARCHICAL,
    }
)

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
    # Prefer strict Glyph check BEFORE str (Glyph inherits from str)
    if isinstance(val, Glyph):
        return GLYPH_TO_FUNCTION.get(val, default)
    if isinstance(val, str):
        # Check if it's a glyph string value ('IL', 'OZ', etc)
        # Build reverse lookup on first use
        if not hasattr(glyph_function_name, "_glyph_value_map"):
            glyph_function_name._glyph_value_map = {
                g.value: func for g, func in GLYPH_TO_FUNCTION.items()
            }
        # Try to convert glyph value to function name
        func_name = glyph_function_name._glyph_value_map.get(val)
        if func_name:
            return func_name
        # Otherwise assume it's already a function name
        return val
    # Unknown type: cannot map safely
    return default


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
    # Validation result types
    "SequenceValidationResult",
    "StructuralPattern",
    # Error classes
    "StructuralGrammarError",
    "StructuralPotentialConfinementError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "SequenceSyntaxError",
    "GrammarConfigurationError",
    # Glyph mappings
    "GLYPH_TO_FUNCTION",
    "FUNCTION_TO_GLYPH",
    "glyph_function_name",
    "function_name_to_glyph",
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


class MutationPreconditionError(StructuralGrammarError):
    """Error for mutation without proper preconditions."""


class TholClosureError(StructuralGrammarError):
    """Error for THOL without proper closure."""


class TransitionCompatibilityError(StructuralGrammarError):
    """Error for incompatible transition."""


class StructuralPotentialConfinementError(StructuralGrammarError):
    """Error for structural potential drift exceeding escape threshold (U6).

    Raised when Δ Φ_s ≥ φ ≈ 1.618, indicating system escaping potential well
    and entering fragmentation regime.
    """

    def __init__(
        self,
        delta_phi_s: float,
        threshold: float = PHI,
        sequence: list[str] | None = None,
    ):
        msg = (
            f"U6 STRUCTURAL POTENTIAL CONFINEMENT violated: "
            f"Δ Φ_s = {delta_phi_s:.3f} ≥ {threshold:.3f} (escape threshold). "
            f"System entering fragmentation regime. "
            f"Valid sequences maintain Δ Φ_s ≈ 0.6 (30% of threshold)."
        )
        super().__init__(
            rule="U6_CONFINEMENT",
            candidate="sequence",
            message=msg,
            threshold=threshold,
            order=sequence,
            context={"delta_phi_s": delta_phi_s},
        )


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


class SequenceValidationResult(ValidationOutcome[tuple[str, ...]]):
    """Validation outcome for operator sequences with rich metadata.

    Attributes
    ----------
    tokens : tuple[str, ...]
        Original input tokens (non-canonical)
    canonical_tokens : tuple[str, ...]
        Canonicalized operator names
    message : str
        Human-readable validation message
    metadata : Mapping[str, object]
        Additional validation metadata (detected_pattern, flags, etc.)
    error : SequenceSyntaxError | None
        Syntax error details if validation failed
    """

    __slots__ = ("tokens", "canonical_tokens", "message", "metadata", "error")

    def __init__(
        self,
        *,
        tokens: Sequence[str],
        canonical_tokens: Sequence[str],
        passed: bool,
        message: str,
        metadata: Mapping[str, object] | None = None,
        summary: Mapping[str, object] | None = None,
        artifacts: Mapping[str, object] | None = None,
        error: SequenceSyntaxError | None = None,
    ) -> None:
        tokens_tuple = tuple(tokens)
        canonical_tuple = tuple(canonical_tokens)
        metadata_map = dict(metadata or {})

        summary_map = (
            dict(summary)
            if summary is not None
            else {
                "message": message,
                "tokens": canonical_tuple,
                "metadata": metadata_map,
            }
        )
        if error is not None and "error" not in summary_map:
            summary_map["error"] = {
                "index": error.index,
                "token": error.token,
                "message": error.message,
            }

        artifacts_map = (
            dict(artifacts)
            if artifacts is not None
            else {
                "canonical_tokens": canonical_tuple,
                "tokens": tokens_tuple,
            }
        )

        super().__init__(
            subject=canonical_tuple,
            passed=passed,
            summary=summary_map,
            artifacts=artifacts_map,
        )

        self.tokens = tokens_tuple
        self.canonical_tokens = canonical_tuple
        self.message = message
        self.metadata = metadata_map
        self.error = error


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
    G,  # TNFRGraph (runtime fallback)
    node,  # NodeId (runtime fallback)
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
    G.nodes[node]["grammar_violations"].append(
        {
            "stage": stage,
            "error": error.to_payload(),
        }
    )
