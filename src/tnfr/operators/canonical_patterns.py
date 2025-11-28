"""Compatibility shim for canonical operator sequences.

This module provides a minimal registry of canonical TNFR sequences and
metadata expected by tests and SDK helpers. The long-term single source of
truth for pattern logic is pattern_detection.py and patterns.py; this file
exists to preserve API stability for legacy imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..types import Glyph
from .grammar import StructuralPattern


@dataclass(frozen=True)
class CanonicalSequence:
    name: str
    glyphs: List[Glyph]
    pattern_type: StructuralPattern
    description: str
    use_cases: List[str]
    domain: str
    references: List[str]


# Define six archetypal sequences (all include OZ by test contract)
BIFURCATED_BASE = CanonicalSequence(
    name="bifurcated_base",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description="Bifurcation via mutation path with stabilization",
    use_cases=["exploration", "phase transition"],
    domain="general",
    references=["TNFR.pdf §U4", "AGENTS.md U4a"],
)

BIFURCATED_COLLAPSE = CanonicalSequence(
    name="bifurcated_collapse",
    glyphs=[Glyph.AL, Glyph.OZ, Glyph.NUL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.BIFURCATED,
    description="Bifurcation via contraction leading to collapse containment",
    use_cases=["collapse handling", "stress testing"],
    domain="general",
    references=["TNFR.pdf §U4", "UNIFIED_GRAMMAR_RULES.md"],
)

THERAPEUTIC_PROTOCOL = CanonicalSequence(
    name="therapeutic_protocol",
    glyphs=[
        Glyph.EN,
        Glyph.AL,
        Glyph.IL,
        Glyph.OZ,
        Glyph.THOL,
        Glyph.IL,
        Glyph.SHA,
    ],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description="Reception-led therapeutic cycle with self-organization",
    use_cases=["biomedical", "healing"],
    domain="biomedical",
    references=["The Pulse That Traverses Us", "EMISSION_METRICS_GUIDE.md"],
)

THEORY_SYSTEM = CanonicalSequence(
    name="theory_system",
    glyphs=[Glyph.AL, Glyph.NAV, Glyph.UM, Glyph.RA, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.HIERARCHICAL,
    description="Cognitive theory consolidation with coupling and resonance",
    use_cases=["cognitive", "learning"],
    domain="cognitive",
    references=["UNIFIED_GRAMMAR_RULES.md §U3", "docs/"],
)

FULL_DEPLOYMENT = CanonicalSequence(
    name="full_deployment",
    glyphs=[
        Glyph.AL,
        Glyph.UM,
        Glyph.RA,
        Glyph.OZ,
        Glyph.ZHIR,
        Glyph.IL,
        Glyph.SHA,
    ],
    pattern_type=StructuralPattern.COMPLEX,
    description="Deployment pipeline with exploration and stabilization",
    use_cases=["rollout", "integration"],
    domain="social",
    references=["ARCHITECTURE.md", "CROSS_REFERENCE_MATRIX.md"],
)

MOD_STABILIZER = CanonicalSequence(
    name="mod_stabilizer",
    glyphs=[Glyph.AL, Glyph.IL, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Minimal stabilization macro (AL→IL→SHA)",
    use_cases=["module", "macro"],
    domain="general",
    references=["UNIFIED_GRAMMAR_RULES.md §U2"],
)

# Additional patterns expected by tests
CONTAINED_CRISIS = CanonicalSequence(
    name="contained_crisis",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.SHA],
    pattern_type=StructuralPattern.THERAPEUTIC,
    description="Crisis containment through therapeutic intervention",
    use_cases=["crisis management", "containment", "therapeutic intervention"],
    domain="therapeutic",
    references=["TNFR.pdf §U4", "high frequency to zero transition"],
)

MINIMAL_COMPRESSION = CanonicalSequence(
    name="minimal_compression",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.NUL, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Minimal compression followed by coherence and silence",
    use_cases=["compression", "optimization", "space efficiency"],
    domain="general",
    references=["TNFR.pdf", "high frequency to zero transition"],
)

PHASE_LOCK = CanonicalSequence(
    name="phase_lock",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.OZ, Glyph.ZHIR, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Phase locking through mutation and stabilization",
    use_cases=["synchronization", "phase alignment", "network coherence"],
    domain="general",
    references=["TNFR.pdf", "high frequency to zero transition"],
)

RESONANCE_PEAK_HOLD = CanonicalSequence(
    name="resonance_peak_hold",
    glyphs=[Glyph.AL, Glyph.EN, Glyph.IL, Glyph.RA, Glyph.SHA],
    pattern_type=StructuralPattern.STABILIZE,
    description="Resonance peak detection with hold pattern",
    use_cases=["peak detection", "resonance holding", "cognitive processing"],
    domain="cognitive",
    references=["TNFR.pdf", "high frequency to zero transition"],
)


# Public registry
CANONICAL_SEQUENCES: Dict[str, CanonicalSequence] = {
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
