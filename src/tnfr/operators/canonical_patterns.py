"""Canonical operator sequences and archetypal patterns from TNFR theory.

.. deprecated:: 0.2.0
    This module is deprecated and will be removed in version 1.0.0.
    Use :mod:`tnfr.operators.pattern_detection` instead, which provides unified
    pattern detection with explicit U1-U4 grammar rule mappings.

    For canonical sequences, this module remains the authoritative source.
    For pattern detection, use the new unified module.

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

import warnings
from typing import Dict, List, NamedTuple

# Issue deprecation warning on import
warnings.warn(
    "canonical_patterns module is deprecated. "
    "For pattern detection, use tnfr.operators.pattern_detection instead. "
    "Canonical sequences remain available here for backward compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

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
    # New variants unlocked by high → zero frequency transition
    "CONTAINED_CRISIS",
    "RESONANCE_PEAK_HOLD",
    "MINIMAL_COMPRESSION",
    "PHASE_LOCK",
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
    glyphs=[
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.RA,
        Glyph.IL,
        Glyph.SHA,
    ],
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
    glyphs=[
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.THOL,
        Glyph.SHA,
    ],
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
    glyphs=[
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.RA,
        Glyph.IL,
        Glyph.SHA,
    ],
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
# New Canonical Variants: Direct high → zero Termination
# Unlocked by frequency transition update allowing high → zero for SHA
# ============================================================================

CONTAINED_CRISIS = CanonicalSequence(
    name="contained_crisis",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.SHA],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description=(
        "Direct crisis containment: dissonance immediately preserved without processing. "
        "OZ creates ΔNFR ↑ (tension, exploration), SHA immediately contains via νf → 0, "
        "preserving tension in suspended state for later resolution. This is the canonical "
        "'contained dissonance' pattern: trauma held safely, conflict postponed, tension "
        "preserved. Essential for trauma first response, emergency pause, and protective freeze "
        "when immediate processing would overwhelm the system."
    ),
    use_cases=[
        "Trauma containment (immediate safety response without processing)",
        "Crisis management (pause before system overwhelm)",
        "Strategic hold (preserve tension for optimal timing)",
        "Protective freeze (contain instability temporarily)",
        "Emergency stop (halt exploration when risk detected)",
    ],
    domain="therapeutic",
    references=(
        "SHA Grammar Validation Issue: OZ → SHA as valid therapeutic pattern. "
        "Frequency transition update: high → zero for containment/termination."
    ),
)

RESONANCE_PEAK_HOLD = CanonicalSequence(
    name="resonance_peak_hold",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.RA, Glyph.SHA],
    pattern_type=StructuralPattern.RESONATE,
    description=(
        "Peak state preservation: amplified resonance frozen at maximum coherence. "
        "RA amplifies network coherence (high C(t), increased νf), SHA suspends dynamics "
        "(νf → 0) while preserving peak state, creating 'resonance memory'. Used for "
        "flow state capture, optimal pattern preservation, and network snapshots. The "
        "amplified coherence is held in latent form for later reactivation without decay."
    ),
    use_cases=[
        "Flow state preservation (hold peak performance)",
        "Peak coherence memory (capture optimal synchronization)",
        "Network snapshot (freeze synchronized state)",
        "Resonance consolidation (lock amplified pattern)",
        "Optimal moment capture (preserve heightened state)",
    ],
    domain="cognitive",
    references=(
        "Frequency transition update: high → zero enables direct RA → SHA termination. "
        "Memory consolidation pattern for peak states."
    ),
)

MINIMAL_COMPRESSION = CanonicalSequence(
    name="minimal_compression",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.NUL, Glyph.SHA],
    pattern_type=StructuralPattern.COMPRESS,
    description=(
        "Compressed latency: structure reduced to essential form and preserved. "
        "NUL concentrates EPI to core (contraction, high νf), SHA immediately freezes "
        "minimal form (νf → 0), creating efficient storage. Distills pattern to fundamentals "
        "then preserves in latent state. Used for information compression, core essence "
        "extraction, and efficient memory storage of essential structure only."
    ),
    use_cases=[
        "Information compression (minimal viable form)",
        "Core essence extraction (distill to fundamentals)",
        "Efficient storage (compact representation)",
        "Essential pattern hold (preserve only critical structure)",
        "Minimal memory (reduce before suspend)",
    ],
    domain="cognitive",
    references=(
        "Frequency transition update: high → zero enables direct NUL → SHA termination. "
        "Compression-then-freeze pattern for efficient storage."
    ),
)

PHASE_LOCK = CanonicalSequence(
    name="phase_lock",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description=(
        "Phase transition hold: mutation immediately locked without stabilization. "
        "OZ creates bifurcation threshold (ΔNFR ↑), ZHIR pivots phase θ (high νf), "
        "SHA immediately locks new phase (νf → 0) before potential regression. Preserves "
        "transformed state in latent form, preventing return to previous configuration. "
        "Used for identity consolidation, paradigm shift memory, and transformation lock. "
        "Simpler than full ZHIR → IL → SHA when immediate preservation desired."
    ),
    use_cases=[
        "Identity shift hold (lock new configuration)",
        "Phase memory (preserve transformed state)",
        "Mutation consolidation (hold before integration)",
        "Paradigm capture (freeze new perspective)",
        "Transformation lock (prevent regression)",
    ],
    domain="cognitive",
    references=(
        "Frequency transition update: high → zero enables direct ZHIR → SHA termination. "
        "Immediate phase lock pattern without intermediate stabilization."
    ),
)


# ============================================================================
# MOD_STABILIZER: Reusable Glyphic Macro
# ============================================================================

MOD_STABILIZER = CanonicalSequence(
    name="mod_stabilizer",
    glyphs=[
        Glyph.REMESH,
        Glyph.EN,
        Glyph.IL,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.REMESH,
    ],
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
        # New variants unlocked by high → zero frequency transition
        CONTAINED_CRISIS,
        RESONANCE_PEAK_HOLD,
        MINIMAL_COMPRESSION,
        PHASE_LOCK,
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
