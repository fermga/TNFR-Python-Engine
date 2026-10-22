"""Named operator recipes used by the fluent SDK.

This registry owns the recipes; pattern detectors classify supplied words.
Names and domain tags are compatibility labels, not evidence of therapeutic,
cognitive or physical effects. Grammar admission and live preconditions are
separate from the outcome of an executed recipe. No recipe guarantees a
bifurcation, synchronization, stability or creation of a physical entity.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Glyph
from .grammar import StructuralPattern


@dataclass(frozen=True)
class CanonicalSequence:
    name: str
    glyphs: list[Glyph]
    pattern_type: StructuralPattern
    description: str
    use_cases: list[str]
    domain: str
    references: list[str]


# Ordered recipes; consumers read this registry rather than copying the words.
BIFURCATED_BASE = CanonicalSequence(
    name="bifurcated_base",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description="Dissonance and Mutation recipe with Coherence and Silence",
    use_cases=["operator-history studies", "mutation admission"],
    domain="general",
    references=["theory/UNIFIED_GRAMMAR_RULES.md"],
)

BIFURCATED_COLLAPSE = CanonicalSequence(
    name="bifurcated_collapse",
    glyphs=[Glyph.AL, Glyph.OZ, Glyph.NUL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description="Dissonance followed by Contraction, Coherence and Silence",
    use_cases=["contraction comparison", "operator-history studies"],
    domain="general",
    references=["theory/UNIFIED_GRAMMAR_RULES.md"],
)

THERAPEUTIC_PROTOCOL = CanonicalSequence(
    name="therapeutic_protocol",
    glyphs=[
        Glyph.AL,
        Glyph.EN,
        Glyph.IL,
        Glyph.OZ,
        Glyph.THOL,
        Glyph.IL,
        Glyph.SHA,
    ],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description="Emission, Reception and Self-organization recipe with "
    "Coherence and Silence; the name does not establish a therapeutic effect.",
    use_cases=["reception and self-organization studies"],
    domain="biomedical",
    references=["theory/STRUCTURAL_OPERATORS.md"],
)

THEORY_SYSTEM = CanonicalSequence(
    name="theory_system",
    glyphs=[Glyph.AL, Glyph.NAV, Glyph.UM, Glyph.RA, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.HIERARCHICAL,
    description="Transition, Coupling and Resonance recipe with Coherence and Silence",
    use_cases=["coupling and resonance studies"],
    domain="cognitive",
    references=["theory/UNIFIED_GRAMMAR_RULES.md"],
)

FULL_DEPLOYMENT = CanonicalSequence(
    name="full_deployment",
    glyphs=[
        Glyph.AL,
        Glyph.UM,
        Glyph.RA,
        Glyph.IL,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.SHA,
    ],
    pattern_type=StructuralPattern.COMPLEX,
    description="Coupling, Resonance and Mutation recipe with the U4b "
    "Coherence and recent-destabilizer context.",
    use_cases=["ordered operator composition"],
    domain="social",
    references=["ARCHITECTURE.md", "theory/STRUCTURAL_OPERATORS.md"],
)

MOD_STABILIZER = CanonicalSequence(
    name="mod_stabilizer",
    glyphs=[Glyph.AL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Emission, Coherence and Silence recipe (AL→IL→SHA)",
    use_cases=["module", "macro"],
    domain="general",
    references=["theory/UNIFIED_GRAMMAR_RULES.md"],
)

CONTAINED_CRISIS = CanonicalSequence(
    name="contained_crisis",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.SHA],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description="Reception and Dissonance recipe ending in Silence",
    use_cases=["pressure and capacity response studies"],
    domain="therapeutic",
    references=["theory/STRUCTURAL_OPERATORS.md"],
)

MINIMAL_COMPRESSION = CanonicalSequence(
    name="minimal_compression",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.NUL, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Minimal compression followed by coherence and silence",
    use_cases=["contraction and capacity response studies"],
    domain="general",
    references=["theory/STRUCTURAL_OPERATORS.md"],
)

PHASE_LOCK = CanonicalSequence(
    name="phase_lock",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Dissonance and Mutation recipe; phase locking is not guaranteed",
    use_cases=["mutation response studies"],
    domain="general",
    references=["theory/STRUCTURAL_OPERATORS.md"],
)

RESONANCE_PEAK_HOLD = CanonicalSequence(
    name="resonance_peak_hold",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.RA, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Resonance followed by Silence; this recipe does not detect a peak",
    use_cases=["resonance and capacity response studies"],
    domain="cognitive",
    references=["theory/STRUCTURAL_OPERATORS.md"],
)

# Public registry
CANONICAL_SEQUENCES: dict[str, CanonicalSequence] = {
    s.name: s
    for s in (
        BIFURCATED_BASE,
        BIFURCATED_COLLAPSE,
        THERAPEUTIC_PROTOCOL,
        THEORY_SYSTEM,
        FULL_DEPLOYMENT,
        MOD_STABILIZER,
        CONTAINED_CRISIS,
        MINIMAL_COMPRESSION,
        PHASE_LOCK,
        RESONANCE_PEAK_HOLD,
    )
}
