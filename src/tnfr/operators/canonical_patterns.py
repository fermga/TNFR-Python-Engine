"""Canonical operator sequences and archetypal patterns from TNFR theory.

This module defines the 6 canonical archetypal sequences involving OZ (Dissonance)
as documented in "El pulso que nos atraviesa" (Table 2.5 - Glyphic structural typology).

These sequences represent validated structural patterns that can be reused across
different domains and applications while maintaining TNFR coherence and grammar.

References
----------
"El pulso que nos atraviesa", Table 2.5: Glyphic structural typology
Section 2.3.8: Complete examples
Section 2.3.5: Advanced glyphic writing (Glyphic macros)
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple

from ..types import Glyph
from .grammar import StructuralPattern

__all__ = [
    "CanonicalSequence",
    "CANONICAL_SEQUENCES",
    "BIFURCATED_BASE",
    "BIFURCATED_COLLAPSE",
    "THERAPEUTIC_PROTOCOL",
    "THEORY_SYSTEM",
    "FULL_DEPLOYMENT",
    "MOD_STABILIZER",
]


class CanonicalSequence(NamedTuple):
    """Canonical operator sequence with theoretical metadata.
    
    Represents a validated archetypal sequence from TNFR theory with
    structural pattern classification, use cases, and domain context.
    
    Attributes
    ----------
    name : str
        Unique identifier for the sequence (e.g., 'bifurcated_base')
    glyphs : List[Glyph]
        Ordered sequence of structural glyphs (AL, EN, IL, OZ, etc.)
    pattern_type : StructuralPattern
        Structural pattern classification from grammar
    description : str
        Detailed explanation of structural function
    use_cases : List[str]
        Concrete application scenarios
    domain : str
        Primary domain: 'general', 'biomedical', 'cognitive', 'social'
    references : str
        Theoretical grounding from TNFR documentation
    """
    
    name: str
    glyphs: List[Glyph]
    pattern_type: StructuralPattern
    description: str
    use_cases: List[str]
    domain: str
    references: str


# ============================================================================
# Bifurcated Patterns: OZ creates decision points
# ============================================================================

BIFURCATED_BASE = CanonicalSequence(
    name="bifurcated_base",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description=(
        "Structural dissonance that generates bifurcation threshold. "
        "The node can reorganize (ZHIR) or collapse to latency (NUL). "
        "This pattern represents the creative resolution of dissonance "
        "through transformative mutation. "
        "Includes EN → IL (reception→coherence) for grammar validation."
    ),
    use_cases=[
        "Therapeutic intervention for emotional or cognitive blockages",
        "Analysis of cultural crises or paradigms under tension",
        "Design of systems with adaptive response to perturbations",
        "Modeling of decision points in complex networks",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Table 2.5, Section 2.3.4 (Bifurcation)",
)

BIFURCATED_COLLAPSE = CanonicalSequence(
    name="bifurcated_collapse",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.NUL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description=(
        "Alternative bifurcation path: dissonance leads to controlled collapse (NUL) "
        "instead of mutation. Useful for structural reset when transformation "
        "is not viable. The node returns to latency preserving potentiality. "
        "Includes EN → IL (reception→coherence) for grammar validation."
    ),
    use_cases=[
        "Cognitive reset after informational overload",
        "Strategic organizational disinvestment",
        "Return to potentiality after failed exploration",
        "Structural simplification facing unsustainable complexity",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Section 2.3.3 (Bifurcation and mutation)",
)


# ============================================================================
# Therapeutic Protocol: Reorganization Ritual
# ============================================================================

THERAPEUTIC_PROTOCOL = CanonicalSequence(
    name="therapeutic_protocol",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.RA, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description=(
        "Ritual or therapeutic protocol: symbolic emission (AL), stabilizing "
        "reception (EN), initial coherence (IL), creative dissonance as "
        "confrontation (OZ), subject mutation (ZHIR), stabilization of the "
        "new form (IL), resonant propagation (RA), post-resonance stabilization (IL), "
        "entry into latency (SHA). Personal or collective transformation cycle with "
        "creative resolution and coherent frequency transitions."
    ),
    use_cases=[
        "Personal transformation or initiation ceremonies",
        "Deep therapeutic restructuring sessions",
        "Symbolic accompaniment of vital change processes",
        "Collective or community healing rituals",
    ],
    domain="biomedical",
    references="El pulso que nos atraviesa, Ejemplo 3 (Sección 2.3.8)",
)


# ============================================================================
# Theory System: Epistemological Construction
# ============================================================================

THEORY_SYSTEM = CanonicalSequence(
    name="theory_system",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.THOL, Glyph.SHA],
    pattern_type=StructuralPattern.EDUCATIONAL,
    description=(
        "Emerging system of ideas or theory: initial emission (AL), information "
        "reception (EN), stabilization (IL), conceptual dissonance or paradox (OZ), "
        "mutation toward new paradigm (ZHIR), stabilization in coherent understanding (IL), "
        "self-organization into theoretical system (THOL), integration into embodied "
        "knowledge (SHA). Epistemological construction trajectory."
    ),
    use_cases=[
        "Design of epistemological frameworks or scientific paradigms",
        "Construction of coherent theories in social sciences",
        "Modeling of conceptual evolution in academic communities",
        "Development of philosophical systems or worldviews",
    ],
    domain="cognitive",
    references="El pulso que nos atraviesa, Example 2 (Section 2.3.8)",
)


# ============================================================================
# Full Deployment: Complete Deployment
# ============================================================================

FULL_DEPLOYMENT = CanonicalSequence(
    name="full_deployment",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.RA, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.COMPLEX,
    description=(
        "Complete nodal reorganization trajectory: initiating emission (AL), "
        "stabilizing reception (EN), initial coherence (IL), exploratory "
        "dissonance (OZ), transformative mutation (ZHIR), coherent stabilization (IL), "
        "resonant propagation (RA), post-resonance consolidation (IL), closure in latency (SHA). "
        "Exhaustive structural reorganization sequence with coherent frequency transitions."
    ),
    use_cases=[
        "Complete organizational transformation processes",
        "Radical innovation cycles with multiple phases",
        "Deep and transformative learning trajectories",
        "Systemic reorganization of communities or ecosystems",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Table 2.5 (Complete deployment)",
)


# ============================================================================
# MOD_STABILIZER: Reusable Glyphic Macro
# ============================================================================

MOD_STABILIZER = CanonicalSequence(
    name="mod_stabilizer",
    glyphs=[Glyph.REMESH, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.REMESH],
    pattern_type=StructuralPattern.EXPLORE,
    description=(
        "MOD_STABILIZER: glyphic macro for controlled transformation. "
        "Activates recursivity (REMESH), receives current state (EN), stabilizes (IL), "
        "introduces controlled dissonance (OZ), mutates structure (ZHIR), stabilizes "
        "new form (IL), closes with recursivity (REMESH). Reusable as "
        "modular subunit within more complex sequences. Represents the "
        "minimal pattern of exploration-transformation-consolidation with complete "
        "grammar validation (EN → IL) and recursive closure."
    ),
    use_cases=[
        "Safe transformation module for composition",
        "Reusable component in complex sequences",
        "Encapsulated creative resolution pattern",
        "Building block for T'HOL (self-organization)",
    ],
    domain="general",
    references="El pulso que nos atraviesa, Section 2.3.5 (Glyphic macros)",
)


# ============================================================================
# Registry of All Canonical Sequences
# ============================================================================

CANONICAL_SEQUENCES: Dict[str, CanonicalSequence] = {
    seq.name: seq
    for seq in [
        BIFURCATED_BASE,
        BIFURCATED_COLLAPSE,
        THERAPEUTIC_PROTOCOL,
        THEORY_SYSTEM,
        FULL_DEPLOYMENT,
        MOD_STABILIZER,
    ]
}
"""Registry of all canonical operator sequences.

Maps sequence names to their full CanonicalSequence definitions. This registry
provides programmatic access to validated archetypal patterns from TNFR theory.

Examples
--------
>>> from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES
>>> seq = CANONICAL_SEQUENCES["bifurcated_base"]
>>> print(f"{seq.name}: {' → '.join(g.value for g in seq.glyphs)}")
bifurcated_base: AL → IL → OZ → ZHIR → SHA
"""
