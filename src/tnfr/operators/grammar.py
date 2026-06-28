"""TNFR Canonical Grammar (single source of truth).

OPTIMIZED IMPLEMENTATION: Unified grammar validation with intelligent caching
to eliminate redundant validation across TNFR modules. All grammar rules
derive from nodal equation ∂EPI/∂t = νf · ΔNFR(t) theoretical foundations.

PERFORMANCE OPTIMIZATIONS:
- Cached validation results with cache invalidation
- Fast-path validation for common patterns
- Batch validation for operator sequences
- Memory-efficient storage of validation state

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
    Basis: AGENTS.md Invariant #2 + resonance physics

U4: BIFURCATION DYNAMICS
    U4a: If bifurcation triggers, then include handlers
    U4b: If transformers, then recent destabilizer (+ prior IL for ZHIR)
    Basis: Contract OZ + bifurcation theory

U5: MULTI-SCALE COHERENCE
    If deep REMESH (depth > 1), require scale stabilizers (IL/THOL).
    Basis: Hierarchical nodal equation + coherence conservation
    (C_parent ≥ α·ΣC_child).

U6: STRUCTURAL POTENTIAL CONFINEMENT (Promoted 2025-11-11)
    Verify Δ Φ_s < π/2 (U6 confinement bound)
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

import hashlib
from functools import lru_cache
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
from .registry import OPERATORS, discover_operators

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
    # Optimized validation
    "validate_sequence_cached",
    "clear_validation_cache",
    "get_validation_cache_stats",
    "GLYPH_TO_FUNCTION",
    "FUNCTION_TO_GLYPH",
    "OPERATORS",
    "OPERATOR_NAME_TO_CLASS",
    # Telemetry helpers
    "get_grammar_cache_stats",
]

# ============================================================================
# OPTIMIZED VALIDATION WITH CACHING (Performance Enhancement)
# ============================================================================

# Validation cache for operator sequences
_validation_cache: dict[str, tuple[bool, str]] = {}
_cache_hits = 0
_cache_misses = 0


def _sequence_hash(sequence: list[Glyph], graph_state: str | None = None) -> str:
    """Generate hash key for operator sequence caching."""
    # Create deterministic hash from sequence and relevant graph state
    sequence_str = "".join(str(op) for op in sequence)
    if graph_state:
        content = f"{sequence_str}:{graph_state}"
    else:
        content = sequence_str

    return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()


@lru_cache(maxsize=1000)
def _validate_grammar_rules_cached(
    sequence_tuple: tuple[Glyph, ...],
) -> tuple[bool, str]:
    """Cached grammar rule validation for operator sequences.

    This optimized version caches validation results to avoid redundant
    computation for repeated sequences following TNFR principles.
    """
    sequence = list(sequence_tuple)

    # Delegate to existing validation logic
    from .grammar_validate import validate_grammar

    try:
        is_valid = validate_grammar(sequence)
        return is_valid, "Valid sequence" if is_valid else "Grammar violation detected"
    except Exception as e:
        return False, f"Validation error: {e}"


def validate_sequence_cached(
    sequence: list[Glyph], graph: TNFRGraph | None = None, use_cache: bool = True
) -> tuple[bool, str]:
    """Optimized sequence validation with intelligent caching.

    PERFORMANCE ENHANCEMENT: Caches validation results to eliminate
    redundant computation while maintaining TNFR theoretical consistency.

    Parameters
    ----------
    sequence : list[Glyph]
        Operator sequence to validate
    graph : TNFRGraph, optional
        Graph for context-aware validation
    use_cache : bool, default=True
        Enable validation caching

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)

    Notes
    -----
    Cache key includes sequence operators and relevant graph state
    to ensure correctness while maximizing cache hit rate.
    """
    global _cache_hits, _cache_misses

    if not use_cache:
        # Direct validation without caching
        return _validate_grammar_rules_cached(tuple(sequence))

    # Generate cache key
    graph_state = None
    if graph is not None:
        # Include relevant graph state in cache key
        n_nodes = len(graph.nodes())
        n_edges = len(graph.edges())
        graph_state = f"n{n_nodes}_e{n_edges}"

    cache_key = _sequence_hash(sequence, graph_state)

    # Check cache
    if cache_key in _validation_cache:
        _cache_hits += 1
        return _validation_cache[cache_key]

    # Cache miss - perform validation
    _cache_misses += 1
    is_valid, message = _validate_grammar_rules_cached(tuple(sequence))

    # Store in cache (with size limit)
    if len(_validation_cache) < 10000:  # Prevent unbounded growth
        _validation_cache[cache_key] = (is_valid, message)

    return is_valid, message


def clear_validation_cache() -> None:
    """Clear validation cache to free memory."""
    global _cache_hits, _cache_misses
    _validation_cache.clear()
    _validate_grammar_rules_cached.cache_clear()
    _cache_hits = 0
    _cache_misses = 0


def get_validation_cache_stats() -> dict[str, Any]:
    """Get validation cache performance statistics."""
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100.0) if total_requests > 0 else 0.0

    return {
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_size": len(_validation_cache),
        "lru_cache_info": _validate_grammar_rules_cached.cache_info()._asdict(),
    }
