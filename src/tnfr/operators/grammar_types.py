"""TNFR Grammar: Types and Exceptions

Enums, exception classes, and validation result types for TNFR grammar.

Terminology (TNFR semantics):
- "node" == resonant locus (structural coherence site); kept for NetworkX compatibility
- Future semantic aliasing ("locus") must preserve public API stability
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph, Glyph
    from .definitions import Operator
else:
    NodeId = Any
    TNFRGraph = Any
    from ..types import Glyph

from ..validation.base import ValidationOutcome

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


class StructuralPattern(Enum):
    """Classification of structural patterns in TNFR sequences.

    Used by canonical_patterns module for backward compatibility.
    Deprecated - use pattern_detection module for new code.
    """

    BIFURCATED = "bifurcated"
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"
    ORGANIZATIONAL = "organizational"
    CREATIVE = "creative"
    REGENERATIVE = "regenerative"
    COMPLEX = "complex"
    COMPRESS = "compress"
    EXPLORE = "explore"
    RESONATE = "resonate"
    BOOTSTRAP = "bootstrap"
    STABILIZE = "stabilize"
    LINEAR = "linear"
    HIERARCHICAL = "hierarchical"
    FRACTAL = "fractal"
    CYCLIC = "cyclic"
    BASIC_LEARNING = "basic_learning"
    DEEP_LEARNING = "deep_learning"
    EXPLORATORY_LEARNING = "exploratory_learning"
    CONSOLIDATION_CYCLE = "consolidation_cycle"
    ADAPTIVE_MUTATION = "adaptive_mutation"
    UNKNOWN = "unknown"


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
    "GrammarValidator",
    "GrammarContext",
    "validate_grammar",
    # U6 telemetry helpers (non-blocking warnings)
    "warn_phase_gradient_telemetry",
    "warn_phase_curvature_telemetry",
    "warn_coherence_length_telemetry",
    "validate_structural_potential_confinement",
    "SequenceValidationResult",
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
    "validate_sequence_with_health",
    "SequenceValidationResultWithHealth",
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
    # Added compatibility exports appended later
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


class MutationPreconditionError(StructuralGrammarError):
    """Error for mutation without proper preconditions."""


class TholClosureError(StructuralGrammarError):
    """Error for THOL without proper closure."""


class TransitionCompatibilityError(StructuralGrammarError):
    """Error for incompatible transition."""


class StructuralPotentialConfinementError(StructuralGrammarError):
    """Error for structural potential drift exceeding escape threshold (U6).

    Raised when Δ Φ_s ≥ 2.0, indicating system escaping potential well
    and entering fragmentation regime.
    """

    def __init__(
        self, delta_phi_s: float, threshold: float = 2.0, sequence: list[str] | None = None
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


class SequenceValidationResult(ValidationOutcome[Tuple[str, ...]]):
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

        summary_map = dict(summary) if summary is not None else {
            "message": message,
            "tokens": canonical_tuple,
            "metadata": metadata_map,
        }
        if error is not None and "error" not in summary_map:
            summary_map["error"] = {
                "index": error.index,
                "token": error.token,
                "message": error.message,
            }

        artifacts_map = dict(artifacts) if artifacts is not None else {
            "canonical_tokens": canonical_tuple,
            "tokens": tokens_tuple,
        }

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


