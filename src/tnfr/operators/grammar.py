"""TNFR Canonical Grammar (single source of truth).

Shared grammar validation and public operator-language adapters.
The nodal equation motivates the rules; operator contracts and calibrated
policies supply premises not determined by that equation alone.

Terminology (TNFR semantics):
- "node" here means resonant locus (coherence site); kept for
    compatibility with graph libraries. Unrelated to Node.js runtime.
- Future aliasing ("locus") must preserve public API stability.

Word admission, live operator preconditions and trajectory guarantees are
different checks. U1-U5 encode the supported operator language; U6 reads
potential drift. Neither layer selects a unique autonomous evolution law.

Canonical Constraints (U1-U6)
------------------------------
U1: STRUCTURAL INITIATION & CLOSURE
    U1a: Start with generators when needed
    U1b: End with closure operators
    Basis: initialization/endpoint contracts; νf·ΔNFR is defined at EPI=0

U2: STABILIZATION COVERAGE & DEBT
    If destabilizers, then include stabilizers
    Basis: calibrated causal debt; convergence needs separate trajectory bounds

U3: RESONANT COUPLING
    If coupling/resonance, then verify phase compatibility
    Basis: AGENTS.md Invariant #2 + resonance physics

U4: BIFURCATION DYNAMICS
    U4a: If bifurcation triggers, then include handlers
    U4b: If transformers, then recent destabilizer (+ prior IL for ZHIR)
    Basis: trigger/handler and history contracts, not a bifurcation theorem

U5: MULTI-SCALE COHERENCE
    If deep REMESH (depth > 1), require scale stabilizers (IL/THOL).
    Basis: nesting contracts; quantitative coherence and reduced dynamics
    require an explicit hierarchy and closure hypotheses

U6: STRUCTURAL POTENTIAL DRIFT POLICY
    Observe mean_i |Δ Φ_s(i)| < π/2 between declared reference and observed fields
    Basis: selected telemetry policy for the emergent Φ_s field
    Scope: finite read-only alert; it is not a graph-independent bound or a
    certificate of future confinement/fragmentation

For requirements and mathematical limits, see UNIFIED_GRAMMAR_RULES.md and
DIAGNOSTIC_AND_GRAMMAR_SCOPE.md.

References
----------
- UNIFIED_GRAMMAR_RULES.md: Canonical policies, hypotheses and mappings
- AGENTS.md: Canonical invariants and formal contracts
- theory/FUNDAMENTAL_THEORY.md: Nodal identity and complete-law premises
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import Glyph, NodeId, TNFRGraph
    from .definitions import Operator  # noqa: F401
else:
    # Runtime fallbacks to avoid type expression errors in string annotations
    NodeId = Any  # type: ignore  # Runtime alias
    TNFRGraph = Any  # type: ignore  # Runtime alias
    from ..types import Glyph  # noqa: F401

from ..config.operator_names import (  # noqa: F401
    CANONICAL_OPERATOR_NAMES,
    INTERMEDIATE_OPERATORS,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
)
from ..validation.base import ValidationOutcome  # noqa: F401
from ..validation.compatibility import (  # noqa: F401
    CompatibilityLevel,
    get_compatibility_level,
)

# Operator registry & glyph mappings (backward compatibility)
from .definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Emission,
    Expansion,
    Mutation,
    Reception,
    Recursivity,
    Resonance,
    SelfOrganization,
    Silence,
    Transition,
)

# Application
from .grammar_application import (
    apply_glyph_with_grammar,
    enforce_canonical_grammar,
    on_applied_glyph,
)

# Context
from .grammar_context import GrammarContext

# Core validator
from .grammar_core import GrammarValidator

# Grammar-Aware Dynamics (proactive incremental U1-U6 enforcement)
from .grammar_dynamics import (
    CandidateResult,
    GrammarViolation,
    enforce_grammar_on_glyph,
    filter_candidates,
    suggest_alternative,
    validate_candidate,
    validate_sequence_incremental,
)

# Pattern recognition
from .grammar_patterns import (
    CANONICAL_IL_SEQUENCES,
    IL_ANTIPATTERNS,
    SequenceValidationResultWithHealth,
    optimize_il_sequence,
    parse_sequence,
    recognize_il_sequences,
    suggest_il_sequence,
    validate_sequence,
    validate_sequence_with_health,
)

# Telemetry
from .grammar_telemetry import (
    warn_coherence_length_telemetry,
    warn_phase_curvature_telemetry,
    warn_phase_gradient_telemetry,
)

# Re-export all grammar components (backward compatibility)
from .grammar_types import (  # Operator sets
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    CLOSURES,
    COUPLING_RESONANCE,
    DESTABILIZERS,
    FUNCTION_TO_GLYPH,
    GENERATORS,
    GLYPH_TO_FUNCTION,
    RECURSIVE_GENERATORS,
    SCALE_STABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
    GrammarConfigurationError,
    MutationPreconditionError,
    RepeatWindowError,
    SequenceSyntaxError,
    SequenceValidationResult,
    StructuralGrammarError,
    StructuralPattern,
    StructuralPotentialConfinementError,
    TholClosureError,
    TransitionCompatibilityError,
    function_name_to_glyph,
    glyph_function_name,
    record_grammar_violation,
)

# U6 validation
from .grammar_u6 import validate_structural_potential_confinement

# Main validation entry point
from .grammar_validate import validate_grammar
from .registry import OPERATORS

# Provide a name→class mapping including canonical aliases (auto-registered)
OPERATOR_NAME_TO_CLASS = {n: cls for n, cls in OPERATORS.items()}

# Backward compatibility: keep operator classes referenced so import tools
# and static analyzers treat them as intentionally re-exported.
_BACKWARD_COMPAT_OPERATORS = (
    Emission,
    Reception,
    Coherence,
    Dissonance,
    Coupling,
    Resonance,
    Silence,
    Expansion,
    Contraction,
    SelfOrganization,
    Mutation,
    Transition,
    Recursivity,
)


__all__ = [
    # Types
    "StructuralPattern",
    # Exceptions
    "StructuralGrammarError",
    "RepeatWindowError",
    "MutationPreconditionError",
    "TholClosureError",
    "TransitionCompatibilityError",
    "StructuralPotentialConfinementError",
    "SequenceSyntaxError",
    "GrammarConfigurationError",
    # Validation
    "SequenceValidationResult",
    "validate_grammar",
    "validate_sequence",
    "parse_sequence",
    "validate_sequence_with_health",
    # U6
    "validate_structural_potential_confinement",
    # Core
    "GrammarContext",
    "GrammarValidator",
    # Application
    "apply_glyph_with_grammar",
    "on_applied_glyph",
    "enforce_canonical_grammar",
    # Grammar-Aware Dynamics
    "GrammarViolation",
    "CandidateResult",
    "validate_candidate",
    "filter_candidates",
    "suggest_alternative",
    "enforce_grammar_on_glyph",
    "validate_sequence_incremental",
    # Helpers
    "glyph_function_name",
    "function_name_to_glyph",
    "record_grammar_violation",
    "SequenceValidationResultWithHealth",
    "recognize_il_sequences",
    "optimize_il_sequence",
    "suggest_il_sequence",
    "CANONICAL_IL_SEQUENCES",
    "IL_ANTIPATTERNS",
    # Telemetry
    "warn_phase_gradient_telemetry",
    "warn_phase_curvature_telemetry",
    "warn_coherence_length_telemetry",
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
    # Registry & glyph compatibility exports
    "GLYPH_TO_FUNCTION",
    "FUNCTION_TO_GLYPH",
    "OPERATORS",
    "OPERATOR_NAME_TO_CLASS",
]
