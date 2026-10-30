"""Public grammar types are reexported from their runtime owners."""

from __future__ import annotations

from ..config.operator_names import CANONICAL_OPERATOR_NAMES as CANONICAL_OPERATOR_NAMES
from ..config.operator_names import INTERMEDIATE_OPERATORS as INTERMEDIATE_OPERATORS
from ..config.operator_names import SELF_ORGANIZATION as SELF_ORGANIZATION
from ..config.operator_names import (
    SELF_ORGANIZATION_CLOSURES as SELF_ORGANIZATION_CLOSURES,
)
from ..config.operator_names import VALID_END_OPERATORS as VALID_END_OPERATORS
from ..config.operator_names import VALID_START_OPERATORS as VALID_START_OPERATORS
from ..types import Glyph as Glyph
from ..types import NodeId as NodeId
from ..types import TNFRGraph as TNFRGraph
from ..validation.base import ValidationOutcome as ValidationOutcome
from ..validation.compatibility import CompatibilityLevel as CompatibilityLevel
from ..validation.compatibility import (
    get_compatibility_level as get_compatibility_level,
)
from .definitions import Coherence as Coherence
from .definitions import Contraction as Contraction
from .definitions import Coupling as Coupling
from .definitions import Dissonance as Dissonance
from .definitions import Emission as Emission
from .definitions import Expansion as Expansion
from .definitions import Mutation as Mutation
from .definitions import Operator
from .definitions import Reception as Reception
from .definitions import Recursivity as Recursivity
from .definitions import Resonance as Resonance
from .definitions import SelfOrganization as SelfOrganization
from .definitions import Silence as Silence
from .definitions import Transition as Transition
from .grammar_application import apply_glyph_with_grammar as apply_glyph_with_grammar
from .grammar_application import enforce_canonical_grammar as enforce_canonical_grammar
from .grammar_application import on_applied_glyph as on_applied_glyph
from .grammar_context import GrammarContext as GrammarContext
from .grammar_core import GrammarValidator as GrammarValidator
from .grammar_dynamics import CandidateResult as CandidateResult
from .grammar_dynamics import GrammarViolation as GrammarViolation
from .grammar_dynamics import enforce_grammar_on_glyph as enforce_grammar_on_glyph
from .grammar_dynamics import filter_candidates as filter_candidates
from .grammar_dynamics import suggest_alternative as suggest_alternative
from .grammar_dynamics import validate_candidate as validate_candidate
from .grammar_dynamics import (
    validate_sequence_incremental as validate_sequence_incremental,
)
from .grammar_patterns import CANONICAL_IL_SEQUENCES as CANONICAL_IL_SEQUENCES
from .grammar_patterns import IL_ANTIPATTERNS as IL_ANTIPATTERNS
from .grammar_patterns import (
    SequenceValidationResultWithHealth as SequenceValidationResultWithHealth,
)
from .grammar_patterns import optimize_il_sequence as optimize_il_sequence
from .grammar_patterns import parse_sequence as parse_sequence
from .grammar_patterns import recognize_il_sequences as recognize_il_sequences
from .grammar_patterns import suggest_il_sequence as suggest_il_sequence
from .grammar_patterns import validate_sequence as validate_sequence
from .grammar_patterns import (
    validate_sequence_with_health as validate_sequence_with_health,
)
from .grammar_telemetry import (
    warn_coherence_length_telemetry as warn_coherence_length_telemetry,
)
from .grammar_telemetry import (
    warn_phase_curvature_telemetry as warn_phase_curvature_telemetry,
)
from .grammar_telemetry import (
    warn_phase_gradient_telemetry as warn_phase_gradient_telemetry,
)
from .grammar_types import BIFURCATION_HANDLERS as BIFURCATION_HANDLERS
from .grammar_types import BIFURCATION_TRIGGERS as BIFURCATION_TRIGGERS
from .grammar_types import CLOSURES as CLOSURES
from .grammar_types import COUPLING_RESONANCE as COUPLING_RESONANCE
from .grammar_types import DESTABILIZERS as DESTABILIZERS
from .grammar_types import FUNCTION_TO_GLYPH as FUNCTION_TO_GLYPH
from .grammar_types import GENERATORS as GENERATORS
from .grammar_types import GLYPH_TO_FUNCTION as GLYPH_TO_FUNCTION
from .grammar_types import RECURSIVE_GENERATORS as RECURSIVE_GENERATORS
from .grammar_types import SCALE_STABILIZERS as SCALE_STABILIZERS
from .grammar_types import STABILIZERS as STABILIZERS
from .grammar_types import TRANSFORMERS as TRANSFORMERS
from .grammar_types import GrammarConfigurationError as GrammarConfigurationError
from .grammar_types import MutationPreconditionError as MutationPreconditionError
from .grammar_types import RepeatWindowError as RepeatWindowError
from .grammar_types import SequenceSyntaxError as SequenceSyntaxError
from .grammar_types import SequenceValidationResult as SequenceValidationResult
from .grammar_types import StructuralGrammarError as StructuralGrammarError
from .grammar_types import StructuralPattern as StructuralPattern
from .grammar_types import (
    StructuralPotentialConfinementError as StructuralPotentialConfinementError,
)
from .grammar_types import TholClosureError as TholClosureError
from .grammar_types import TransitionCompatibilityError as TransitionCompatibilityError
from .grammar_types import function_name_to_glyph as function_name_to_glyph
from .grammar_types import glyph_function_name as glyph_function_name
from .grammar_types import record_grammar_violation as record_grammar_violation
from .grammar_u6 import (
    validate_structural_potential_confinement as validate_structural_potential_confinement,
)
from .grammar_validate import validate_grammar as validate_grammar
from .registry import OPERATORS as OPERATORS

OPERATOR_NAME_TO_CLASS: dict[str, type[Operator]]

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
