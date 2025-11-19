"""TNFR Canonical Grammar (single source of truth).

Implements grammar constraints emerging from TNFR physics.

Terminology (TNFR semantics):
- "node" here means resonant locus (coherence site); kept for
    compatibility with graph libraries. Unrelated to Node.js runtime.
- Future aliasing ("locus") must preserve public API stability.

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
    If deep REMESH (depth > 1), require scale stabilizers (IL/THOL).
    Basis: Hierarchical nodal equation + coherence conservation
    (C_parent ≥ α·ΣC_child).

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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph, Glyph
    from .definitions import Operator  # noqa: F401
else:
    # Runtime fallbacks to avoid type expression errors in string annotations
    NodeId = Any  # type: ignore  # Runtime alias
    TNFRGraph = Any  # type: ignore  # Runtime alias
    from ..types import Glyph  # noqa: F401

from ..config.operator_names import (  # noqa: F401
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
from ..validation.base import ValidationOutcome  # noqa: F401
from ..validation.compatibility import (  # noqa: F401
    CompatibilityLevel,
    get_compatibility_level,
)


# Re-export all grammar components (backward compatibility)
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

# Operator registry & glyph mappings (backward compatibility)
from .definitions import (
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
from .registry import discover_operators, OPERATORS
from .grammar_types import GLYPH_TO_FUNCTION, FUNCTION_TO_GLYPH

# Ensure registry populated for tests expecting direct name lookups
discover_operators()

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


def get_grammar_cache_stats() -> dict[str, dict[str, int]]:
    """Return cache statistics for grammar-level cached functions.

    Inspects imported callables for a ``cache_info`` attribute (from
    ``functools.lru_cache``). Returns mapping ``{function_name: cache_info}``
    where ``cache_info`` is converted to a plain dict.
    Physics-neutral: read-only telemetry; does not modify caches.
    """
    stats: dict[str, dict[str, int]] = {}
    import inspect
    for name, obj in list(globals().items()):
        if inspect.isfunction(obj) and hasattr(obj, "cache_info"):
            try:  # pragma: no cover - defensive
                info = obj.cache_info()
                maxsize = info.maxsize if info.maxsize is not None else -1
                stats[name] = {
                    "hits": info.hits,
                    "misses": info.misses,
                    "maxsize": maxsize,
                    "currsize": info.currsize,
                }
            except Exception:  # pragma: no cover
                pass
    return stats


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
    # Registry & glyph compatibility exports
    "GLYPH_TO_FUNCTION",
    "FUNCTION_TO_GLYPH",
    "OPERATORS",
    "OPERATOR_NAME_TO_CLASS",
    # Telemetry helpers
    "get_grammar_cache_stats",
]
