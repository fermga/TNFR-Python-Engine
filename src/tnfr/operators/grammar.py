"""TNFR Canonical Grammar - Single Source of Truth.

This module implements the canonical TNFR grammar constraints that emerge
inevitably from TNFR physics.

Terminology (TNFR semantics):
- "node" in this file means resonant locus (structural coherence site) and is kept
    for compatibility with underlying graph libraries (e.g., NetworkX). It is unrelated
    to the Node.js runtime.
- Future semantic aliasing ("locus") must preserve public API stability.

All rules derive from the nodal equation ∂EPI/∂t = νf · ΔNFR(t), canonical
invariants, and formal contracts. No organizational conventions.

Canonical Constraints (U1-U6)
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

U5: MULTI-SCALE COHERENCE
    If deep REMESH (recursivity with depth>1), require scale stabilizers (IL / THOL)
    Basis: Hierarchical nodal equation + coherence conservation (C_parent ≥ α·ΣC_child)

U6: STRUCTURAL POTENTIAL CONFINEMENT (Promoted 2025-11-11)
    Verify Δ Φ_s < 2.0 (escape threshold)
    Basis: Emergent Φ_s field from ΔNFR distribution + empirical validation
    Status: CANONICAL - 2,400+ experiments, corr(Δ Φ_s, ΔC) = -0.822, CV = 0.1%

For complete derivations and physics basis, see UNIFIED_GRAMMAR_RULES.md

References
----------
- UNIFIED_GRAMMAR_RULES.md: Complete physics derivations and mappings
- AGENTS.md: Canonical invariants and formal contracts
- TNFR.pdf: Nodal equation and bifurcation theory
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence, Tuple

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph, Glyph
    from .definitions import Operator
else:
    # Runtime fallbacks to avoid type expression errors in string annotations
    NodeId = Any  # type: ignore  # Runtime alias
    TNFRGraph = Any  # type: ignore  # Runtime alias
    from ..types import Glyph

from ..config.operator_names import (
    BIFURCATION_WINDOWS,
    CANONICAL_OPERATOR_NAMES,
    DESTABILIZERS_MODERATE,
    DESTABILIZERS_STRONG,
    DESTABILIZERS_WEAK,
    INTERMEDIATE_OPERATORS,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
)
from ..validation.base import ValidationOutcome
from ..validation.compatibility import (
    CompatibilityLevel,
    get_compatibility_level,
)



# Re-export all grammar components for backward compatibility

# Types and exceptions
from .grammar_types import (
    StructuralPattern,
    StructuralGrammarError,
    RepeatWindowError,
    MutationPreconditionError,
    TholClosureError,
    TransitionCompatibilityError,
    StructuralPotentialConfinementError,
    SequenceSyntaxError,
    SequenceValidationResult,
    GrammarConfigurationError,
    record_grammar_violation,
    glyph_function_name,
    function_name_to_glyph,
    # Operator sets
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
    COUPLING_RESONANCE,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
    RECURSIVE_GENERATORS,
    SCALE_STABILIZERS,
)

# Context
from .grammar_context import GrammarContext

# Core validator
from .grammar_core import GrammarValidator

# U6 validation
from .grammar_u6 import validate_structural_potential_confinement

# Main validation entry point
from .grammar_validate import validate_grammar

# Telemetry
from .grammar_telemetry import (
    warn_phase_gradient_telemetry,
    warn_phase_curvature_telemetry,
    warn_coherence_length_telemetry,
)

# Application
from .grammar_application import (
    apply_glyph_with_grammar,
    on_applied_glyph,
    enforce_canonical_grammar,
)

# Pattern recognition
from .grammar_patterns import (
    validate_sequence,
    parse_sequence,
    SequenceValidationResultWithHealth,
    validate_sequence_with_health,
    recognize_il_sequences,
    optimize_il_sequence,
    suggest_il_sequence,
    CANONICAL_IL_SEQUENCES,
    IL_ANTIPATTERNS,
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
]
