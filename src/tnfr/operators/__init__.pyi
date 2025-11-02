from typing import Any

from tnfr.validation import (
    GrammarContext as _GrammarContext,
    StructuralGrammarError as _StructuralGrammarError,
    RepeatWindowError as _RepeatWindowError,
    MutationPreconditionError as _MutationPreconditionError,
    TholClosureError as _TholClosureError,
    TransitionCompatibilityError as _TransitionCompatibilityError,
    SequenceValidationResult as _SequenceValidationResult,
    SequenceSyntaxError as _SequenceSyntaxError,
    apply_glyph_with_grammar as _apply_glyph_with_grammar,
    enforce_canonical_grammar as _enforce_canonical_grammar,
    on_applied_glyph as _on_applied_glyph,
    parse_sequence as _parse_sequence,
    validate_sequence as _validate_sequence,
)

Operator: Any
Emission: Any
Reception: Any
Coherence: Any
Dissonance: Any
Coupling: Any
Resonance: Any
Silence: Any
Expansion: Any
Contraction: Any
SelfOrganization: Any
Mutation: Any
Transition: Any
Recursivity: Any
GLYPH_OPERATIONS: Any
JitterCache: Any
JitterCacheManager: Any
OPERATORS: Any
GrammarContext = _GrammarContext
StructuralGrammarError = _StructuralGrammarError
RepeatWindowError = _RepeatWindowError
MutationPreconditionError = _MutationPreconditionError
TholClosureError = _TholClosureError
TransitionCompatibilityError = _TransitionCompatibilityError
SequenceValidationResult = _SequenceValidationResult
SequenceSyntaxError = _SequenceSyntaxError
_gram_state: Any
apply_glyph: Any
apply_glyph_obj: Any
apply_glyph_with_grammar = _apply_glyph_with_grammar
apply_network_remesh: Any
apply_remesh_if_globally_stable: Any
apply_topological_remesh: Any
discover_operators: Any
enforce_canonical_grammar = _enforce_canonical_grammar
get_glyph_factors: Any
get_jitter_manager: Any
get_neighbor_epi: Any
on_applied_glyph = _on_applied_glyph
parse_sequence = _parse_sequence
random_jitter: Any
reset_jitter_manager: Any
validate_sequence = _validate_sequence
