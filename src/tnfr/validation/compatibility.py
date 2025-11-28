"""Compatibility matrices and validation for operator sequences in TNFR.

Physics Basis:
- Adjacent operators must have resonant phase compatibility
- Incompatible sequences lead to destructive interference
- See UNIFIED_GRAMMAR_RULES.md for complete derivations
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..config.operator_names import (
    COHERENCE,
    CONTRACTION,
    COUPLING,
    DISSONANCE,
    EMISSION,
    EXPANSION,
    MUTATION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from ..types import Glyph
# NOTE: Cannot import grammar at module level (circular dependency)
# Import happens lazily in _translate_structural()

__all__ = [
    "CompatibilityLevel",
    "GRADUATED_COMPATIBILITY",
    "get_compatibility_level",
    "get_canon_compat",
    "get_canon_fallback",
]


class CompatibilityLevel(Enum):
    """Graduated compatibility levels for structural operator transitions.

    Reflects the theoretical richness of TNFR by distinguishing between
    optimal, acceptable, contextual, and incompatible transitions.

    Attributes
    ----------
    EXCELLENT : str
        Optimal transition that directly supports structural coherence.
        Example: EMISSION → COHERENCE (initiation → stabilization)
    GOOD : str
        Acceptable transition that maintains structural integrity.
        Example: EMISSION → RESONANCE (initiation → amplification)
    CAUTION : str
        Contextually dependent transition requiring careful validation.
        Example: EMISSION → DISSONANCE (initiation → tension)
        Generates warnings to alert users of potential incoherence.
    AVOID : str
        Incompatible transition that violates structural coherence.
        Example: SILENCE → DISSONANCE (pause → tension is contradictory)
        Raises SequenceSyntaxError when encountered.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    CAUTION = "caution"
    AVOID = "avoid"


# Graduated compatibility matrix expressing structural operator transition quality
# Maps each operator to its allowed next operators categorized by compatibility level
GRADUATED_COMPATIBILITY: dict[str, dict[str, list[str]]] = {
    # EMISSION (AL) - Initiates resonant patterns, seeds coherence outward
    EMISSION: {
        "excellent": [
            COHERENCE,
            TRANSITION,
            RECEPTION,
        ],  # Initiation → stabilization/handoff/anchoring
        "good": [RESONANCE, EXPANSION, COUPLING],  # Amplification, exploration, linking
        "caution": [DISSONANCE],  # Direct tension after initiation requires context
        "avoid": [
            SILENCE,
            EMISSION,
            MUTATION,
            CONTRACTION,
            SELF_ORGANIZATION,
            RECURSIVITY,
        ],
    },
    # RECEPTION (EN) - Anchors inbound energy into the EPI
    RECEPTION: {
        "excellent": [
            COHERENCE,
            COUPLING,
            SELF_ORGANIZATION,
        ],  # Anchoring → stabilization/linking/autonomous cascades
        "good": [RESONANCE],  # Amplification after receiving
        "caution": [],
        "avoid": [
            SILENCE,
            EMISSION,
            DISSONANCE,
            EXPANSION,
            CONTRACTION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
            RECEPTION,
        ],
    },
    # COHERENCE (IL) - Compresses ΔNFR drift to stabilize C(t)
    COHERENCE: {
        "excellent": [
            RESONANCE,
            EXPANSION,
            COUPLING,
        ],  # Stability → amplification/exploration/linking
        "good": [
            SILENCE,
            TRANSITION,
            CONTRACTION,
            SELF_ORGANIZATION,
            RECURSIVITY,
        ],  # Valid progressions
        "caution": [
            MUTATION,
            DISSONANCE,
        ],  # Post-stabilization tension/mutation needs context
        "avoid": [
            EMISSION,
            RECEPTION,
            COHERENCE,
        ],  # Cannot re-initiate, re-anchor, or re-stabilize
    },
    # DISSONANCE (OZ) - Injects controlled tension for probes
    DISSONANCE: {
        "excellent": [
            MUTATION,
            TRANSITION,
            SELF_ORGANIZATION,
        ],  # Tension → transformation/handoff/reorganization
        "good": [
            CONTRACTION,
            RESONANCE,
            RECURSIVITY,
            COHERENCE,
        ],  # Concentration, amplification, fractal echo, stabilization
        "caution": [DISSONANCE],  # Repeated dissonance needs careful management
        "avoid": [SILENCE, EMISSION, RECEPTION, COUPLING, EXPANSION],
    },
    # COUPLING (UM) - Synchronizes bidirectional coherence links
    COUPLING: {
        "excellent": [
            RESONANCE,
            COHERENCE,
            EXPANSION,
        ],  # Linking → amplification/stabilization/exploration
        "good": [TRANSITION, SILENCE],  # Handoff or pause after coupling
        "caution": [],
        "avoid": [
            EMISSION,
            RECEPTION,
            DISSONANCE,
            CONTRACTION,
            MUTATION,
            SELF_ORGANIZATION,
            RECURSIVITY,
            COUPLING,
        ],
    },
    # RESONANCE (RA) - Amplifies aligned structural frequency
    RESONANCE: {
        "excellent": [
            COHERENCE,
            EXPANSION,
            COUPLING,
        ],  # Amplification → stabilization/exploration/linking
        "good": [
            TRANSITION,
            SILENCE,
            EMISSION,
            RECURSIVITY,
        ],  # Handoff, pause, re-initiation, fractal echo
        "caution": [],
        "avoid": [
            RECEPTION,
            DISSONANCE,
            CONTRACTION,
            MUTATION,
            SELF_ORGANIZATION,
            RESONANCE,
        ],
    },
    # SILENCE (SHA) - Suspends reorganization while preserving form
    SILENCE: {
        "excellent": [
            EMISSION,
            RECEPTION,
        ],  # Resume from pause → initiation or anchoring
        "good": [],
        "caution": [],
        "avoid": [
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            EXPANSION,
            CONTRACTION,
            MUTATION,
            TRANSITION,
            SELF_ORGANIZATION,
            RECURSIVITY,
            SILENCE,
        ],
    },
    # EXPANSION (VAL) - Dilates structure to explore volume
    EXPANSION: {
        "excellent": [
            COUPLING,
            RESONANCE,
            COHERENCE,
        ],  # Exploration → linking/amplification/stabilization
        "good": [
            TRANSITION,
            CONTRACTION,
        ],  # Handoff or compression after expansion
        "caution": [],
        "avoid": [
            EMISSION,
            RECEPTION,
            DISSONANCE,
            SILENCE,
            MUTATION,
            SELF_ORGANIZATION,
            RECURSIVITY,
            EXPANSION,
        ],
    },
    # CONTRACTION (NUL) - Concentrates trajectories into core
    CONTRACTION: {
        "excellent": [
            EMISSION,
            COHERENCE,
        ],  # Concentration → re-initiation or stabilization
        "good": [SILENCE],  # Contraction can close with silence
        "caution": [],
        "avoid": [
            RECEPTION,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            EXPANSION,
            MUTATION,
            TRANSITION,
            SELF_ORGANIZATION,
            RECURSIVITY,
            CONTRACTION,
        ],
    },
    # SELF_ORGANIZATION (THOL) - Spawns autonomous cascades
    SELF_ORGANIZATION: {
        "excellent": [
            COHERENCE,
            COUPLING,
            RESONANCE,
        ],  # Autonomous cascades → stabilization/linking/amplification
        "good": [
            DISSONANCE,
            MUTATION,
            TRANSITION,
            SILENCE,
            CONTRACTION,
            EMISSION,
            SELF_ORGANIZATION,
        ],  # Nested/sequential self-organization
        "caution": [],
        "avoid": [RECEPTION, EXPANSION, RECURSIVITY],
    },
    # MUTATION (ZHIR) - Pivots node across structural thresholds
    MUTATION: {
        "excellent": [
            COHERENCE,
            TRANSITION,
            SILENCE,
        ],  # Transformation → stabilization/handoff/pause
        "good": [],
        "caution": [],
        "avoid": [
            EMISSION,
            RECEPTION,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            EXPANSION,
            CONTRACTION,
            MUTATION,
            SELF_ORGANIZATION,
            RECURSIVITY,
        ],
    },
    # TRANSITION (NAV) - Guides controlled regime hand-offs
    TRANSITION: {
        "excellent": [
            RESONANCE,
            COHERENCE,
            COUPLING,
            RECEPTION,
        ],  # Handoff → amplification/stabilization/linking/anchoring
        "good": [
            DISSONANCE,
            MUTATION,
            SILENCE,
            TRANSITION,
        ],  # Tension, transformation, pause, continued handoff
        "caution": [],
        "avoid": [
            EMISSION,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            RECURSIVITY,
        ],
    },
    # RECURSIVITY (REMESH) - Echoes patterns across nested EPIs
    RECURSIVITY: {
        "excellent": [
            COHERENCE,
            RESONANCE,
        ],  # Fractal echo → stabilization/amplification
        "good": [
            DISSONANCE,
            RECEPTION,
            COUPLING,
            TRANSITION,
        ],  # Tension, anchoring, linking, handoff after recursive pattern
        "caution": [],
        "avoid": [
            EMISSION,
            SILENCE,
            EXPANSION,
            CONTRACTION,
            MUTATION,
            SELF_ORGANIZATION,
            RECURSIVITY,
        ],
    },
}


def get_compatibility_level(prev: str, next_op: str) -> CompatibilityLevel:
    """Return the compatibility level between two structural operators.

    Parameters
    ----------
    prev : str
        Previous operator in canonical form (e.g., "emission", "coherence").
    next_op : str
        Next operator in canonical form (e.g., "dissonance", "resonance").

    Returns
    -------
    CompatibilityLevel
        The graduated compatibility level: EXCELLENT, GOOD, CAUTION, or AVOID.

    Examples
    --------
    >>> get_compatibility_level("emission", "coherence")
    CompatibilityLevel.EXCELLENT

    >>> get_compatibility_level("emission", "dissonance")
    CompatibilityLevel.CAUTION

    >>> get_compatibility_level("silence", "dissonance")
    CompatibilityLevel.AVOID

    Notes
    -----
    This function implements the graduated compatibility matrix following TNFR
    canonical theory. Transitions are categorized as:

    - EXCELLENT: Optimal structural progression
    - GOOD: Acceptable structural progression
    - CAUTION: Contextually dependent, requires validation
    - AVOID: Incompatible, violates structural coherence
    """
    if prev not in GRADUATED_COMPATIBILITY:
        # Unknown operator defaults to AVOID
        return CompatibilityLevel.AVOID

    levels = GRADUATED_COMPATIBILITY[prev]

    # Check each level in order of preference
    if next_op in levels.get("excellent", []):
        return CompatibilityLevel.EXCELLENT
    elif next_op in levels.get("good", []):
        return CompatibilityLevel.GOOD
    elif next_op in levels.get("caution", []):
        return CompatibilityLevel.CAUTION
    else:
        return CompatibilityLevel.AVOID


# Generate backward-compatible binary compatibility table from graduated matrix
# This combines excellent, good, and caution levels as "allowed" transitions
def _generate_binary_compat() -> dict[str, set[str]]:
    """Generate binary compatibility table from graduated matrix.

    Combines EXCELLENT, GOOD, and CAUTION levels into a single "allowed" set
    for backward compatibility with existing code that expects binary validation.
    """
    compat: dict[str, set[str]] = {}
    for operator, levels in GRADUATED_COMPATIBILITY.items():
        allowed = set()
        allowed.update(levels.get("excellent", []))
        allowed.update(levels.get("good", []))
        allowed.update(levels.get("caution", []))
        compat[operator] = allowed
    return compat


# Canonical compatibilities (allowed next operators) expressed via structural names
# Derived from GRADUATED_COMPATIBILITY for backward compatibility
_STRUCTURAL_COMPAT: dict[str, set[str]] = _generate_binary_compat()


def _name_to_glyph(name: str) -> Glyph:
    # Lazy import to avoid circular dependency
    from ..operators import grammar as _grammar

    glyph = _grammar.function_name_to_glyph(name)
    if glyph is None:
        raise KeyError(f"No glyph mapped to structural operator '{name}'")
    return glyph


def _translate_structural() -> (
    tuple[dict[Glyph, set[Glyph]], dict[Glyph, Glyph]]
):
    """Translate structural operator names to Glyph enums."""
    compat: dict[Glyph, set[Glyph]] = {}
    for src, targets in _STRUCTURAL_COMPAT.items():
        src_glyph = _name_to_glyph(src)
        compat[src_glyph] = {_name_to_glyph(target) for target in targets}
    fallback: dict[Glyph, Glyph] = {}
    for src, target in _STRUCTURAL_FALLBACK.items():
        fallback[_name_to_glyph(src)] = _name_to_glyph(target)
    return compat, fallback


# Canonical fallbacks when a transition is not allowed (structural names)
_STRUCTURAL_FALLBACK: dict[str, str] = {
    EMISSION: RECEPTION,
    RECEPTION: COHERENCE,
    COHERENCE: RESONANCE,
    TRANSITION: RESONANCE,
    CONTRACTION: EMISSION,
    DISSONANCE: MUTATION,
    RESONANCE: COHERENCE,
    SILENCE: EMISSION,
    SELF_ORGANIZATION: TRANSITION,
    COUPLING: RESONANCE,
    EXPANSION: RESONANCE,
    MUTATION: COHERENCE,
    RECURSIVITY: COHERENCE,  # Fractal echo → stabilization
}

# Lazy initialization to avoid circular import
_CANON_COMPAT: dict[Glyph, set[Glyph]] | None = None
_CANON_FALLBACK: dict[Glyph, Glyph] | None = None


def _ensure_translated() -> None:
    """Ensure glyph tables are translated (lazy initialization)."""
    global _CANON_COMPAT, _CANON_FALLBACK
    if _CANON_COMPAT is None or _CANON_FALLBACK is None:
        _CANON_COMPAT, _CANON_FALLBACK = _translate_structural()


def get_canon_compat() -> dict[Glyph, set[Glyph]]:
    """Get canonical compatibility table (glyph → allowed next glyphs)."""
    _ensure_translated()
    return _CANON_COMPAT  # type: ignore[return-value]


def get_canon_fallback() -> dict[Glyph, Glyph]:
    """Get canonical fallback table (glyph → fallback glyph)."""
    _ensure_translated()
    return _CANON_FALLBACK  # type: ignore[return-value]


# For backward compatibility, provide module-level names that trigger lazy init
def __getattr__(name: str) -> Any:
    if name == "CANON_COMPAT":
        return get_canon_compat()
    elif name == "CANON_FALLBACK":
        return get_canon_fallback()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Re-export structural tables for internal consumers that operate on functional
# identifiers without exposing them as part of the public API.
_STRUCTURAL_COMPAT_TABLE = _STRUCTURAL_COMPAT
_STRUCTURAL_FALLBACK_TABLE = _STRUCTURAL_FALLBACK
