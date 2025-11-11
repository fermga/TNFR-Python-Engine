"""Unified TNFR Pattern Detection - Single Source of Truth.

Detects structural patterns in operator sequences based on unified grammar (U1-U5).
Consolidates functionality from canonical_patterns.py and patterns.py into a single
coherent module that explicitly maps patterns to grammar rules.

Pattern Categories
------------------
1. **Initiation Patterns**: Based on U1a (GENERATORS)
2. **Closure Patterns**: Based on U1b (CLOSURES)
3. **Convergence Patterns**: Based on U2 (STABILIZERS/DESTABILIZERS)
4. **Resonance Patterns**: Based on U3 (COUPLING_RESONANCE)
5. **Bifurcation Patterns**: Based on U4 (TRANSFORMERS)
6. **Composite Patterns**: Combinations of above
7. **Domain Patterns**: Application-specific patterns

All patterns align with unified grammar constraints from UNIFIED_GRAMMAR_RULES.md.

References
----------
- UNIFIED_GRAMMAR_RULES.md: Physics basis for patterns
- grammar.py: Operator sets (GENERATORS, CLOSURES, etc.)
- canonical_patterns.py: Archetypal sequences with metadata
- patterns.py: Advanced pattern detection algorithms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from .grammar import StructuralPattern

from ..config.operator_names import (
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)
from .grammar import (
    BIFURCATION_HANDLERS,
    BIFURCATION_TRIGGERS,
    DESTABILIZERS,
    STABILIZERS,
    TRANSFORMERS,
)

__all__ = [
    "PatternMatch",
    "UnifiedPatternDetector",
    "detect_pattern",
    "analyze_sequence",
]


@dataclass
class PatternMatch:
    """Detected pattern in operator sequence.

    Attributes
    ----------
    pattern_name : str
        Name of detected pattern (e.g., 'bootstrap', 'therapeutic')
    start_idx : int
        Starting index in sequence (0-based)
    end_idx : int
        Ending index in sequence (inclusive)
    confidence : float
        Match confidence score (0.0-1.0)
    grammar_rule : str
        Which U1-U5 rule this pattern relates to
    description : str
        Human-readable description of pattern
    structural_pattern : Optional[StructuralPattern]
        Corresponding StructuralPattern enum if applicable
    """

    pattern_name: str
    start_idx: int
    end_idx: int
    confidence: float
    grammar_rule: str
    description: str
    structural_pattern: Optional[StructuralPattern] = None


class UnifiedPatternDetector:
    """Unified pattern detector aligned with grammar rules U1-U5.

    This class consolidates pattern detection from both canonical_patterns.py
    (archetypal sequences) and patterns.py (advanced detection algorithms).

    Key Features
    ------------
    - Explicit mapping of patterns to U1-U5 grammar rules (temporal + multi-scale)
    - Coherence-weighted scoring for pattern prioritization
    - Detection of both canonical sequences and meta-patterns
    - Grammar validation integrated with pattern recognition

    Examples
    --------
    >>> from tnfr.operators.pattern_detection import UnifiedPatternDetector
    >>> detector = UnifiedPatternDetector()
    >>> sequence = ["emission", "coupling", "coherence"]
    >>> pattern = detector.detect_pattern(sequence)
    >>> print(pattern)  # doctest: +SKIP
    StructuralPattern.BOOTSTRAP
    """

    def __init__(self) -> None:
        """Initialize unified pattern detector."""
        # Import pattern detection logic from patterns.py
        from .patterns import AdvancedPatternDetector

        self._advanced_detector = AdvancedPatternDetector()

        # Pattern to grammar rule mappings
        self._pattern_grammar_map = {
            # U1a: Initiation patterns (use GENERATORS)
            "cold_start": "U1a",
            "phase_transition_start": "U1a",
            "fractal_awakening": "U1a",
            # U1b: Closure patterns (use CLOSURES)
            "terminal_silence": "U1b",
            "regime_handoff": "U1b",
            "fractal_distribution": "U1b",
            "intentional_tension": "U1b",
            # U2: Convergence patterns (STABILIZERS/DESTABILIZERS)
            "stabilization_cycle": "U2",
            "bounded_evolution": "U2",
            "stabilize": "U2",
            # U3: Resonance patterns (COUPLING_RESONANCE)
            "coupling_chain": "U3",
            "resonance_cascade": "U3",
            "phase_locked_network": "U3",
            "resonate": "U3",
            # U4: Bifurcation patterns (TRANSFORMERS)
            "graduated_destabilization": "U4b",
            "managed_bifurcation": "U4a",
            "stable_transformation": "U4b",
            "spontaneous_organization": "U4b",
            "bifurcated": "U4",
            "explore": "U4b",
            # Composite patterns
            "bootstrap": "U1a+U2",
            "regenerative_cycle": "U1a+U1b",
            "exploration": "U2+U4b",
            # Domain patterns (combinations)
            "therapeutic": "U1+U2+U4",
            "educational": "U1+U2+U4",
            "organizational": "U1+U2+U3+U4",
            "creative": "U1+U2+U4",
            "regenerative": "U1+U2+U3",
        }

    def detect_pattern(self, sequence: Sequence[str]) -> StructuralPattern:
        """Detect the best matching pattern in sequence.

        Uses coherence-weighted scoring from AdvancedPatternDetector.

        Parameters
        ----------
        sequence : Sequence[str]
            Canonical operator names to analyze

        Returns
        -------
        StructuralPattern
            The best matching structural pattern

        Examples
        --------
        >>> detector = UnifiedPatternDetector()
        >>> seq = ["emission", "coupling", "coherence"]
        >>> pattern = detector.detect_pattern(seq)  # doctest: +SKIP
        """
        return self._advanced_detector.detect_pattern(sequence)

    def detect_initiation_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect U1a-based initiation patterns.

        Patterns that use GENERATORS (emission, transition, recursivity) to
        create structure from null/dormant states.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of detected initiation patterns
        """
        patterns = []

        # Cold start: Begins with AL (Emission) from EPI=0
        if sequence and sequence[0] == EMISSION:
            patterns.append(
                PatternMatch(
                    pattern_name="cold_start",
                    start_idx=0,
                    end_idx=0,
                    confidence=1.0,
                    grammar_rule="U1a",
                    description="Emission from vacuum (EPI=0 → active structure)",
                )
            )

        # Phase transition start: Begins with NAV (Transition)
        if sequence and sequence[0] == TRANSITION:
            patterns.append(
                PatternMatch(
                    pattern_name="phase_transition_start",
                    start_idx=0,
                    end_idx=0,
                    confidence=1.0,
                    grammar_rule="U1a",
                    description="Transition activates latent EPI",
                )
            )

        # Fractal awakening: Begins with REMESH (Recursivity)
        if sequence and sequence[0] == RECURSIVITY:
            patterns.append(
                PatternMatch(
                    pattern_name="fractal_awakening",
                    start_idx=0,
                    end_idx=0,
                    confidence=1.0,
                    grammar_rule="U1a",
                    description="Recursivity echoes dormant structure",
                )
            )

        return patterns

    def detect_closure_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect U1b-based closure patterns.

        Patterns that use CLOSURES (silence, transition, recursivity, dissonance)
        to leave system in coherent attractor states.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of detected closure patterns
        """
        patterns = []

        if not sequence:
            return patterns

        last_op = sequence[-1]
        last_idx = len(sequence) - 1

        # Terminal silence: Ends with SHA (Silence)
        if last_op == SILENCE:
            patterns.append(
                PatternMatch(
                    pattern_name="terminal_silence",
                    start_idx=last_idx,
                    end_idx=last_idx,
                    confidence=1.0,
                    grammar_rule="U1b",
                    description="Silence freezes evolution (νf → 0)",
                )
            )

        # Regime handoff: Ends with NAV (Transition)
        if last_op == TRANSITION:
            patterns.append(
                PatternMatch(
                    pattern_name="regime_handoff",
                    start_idx=last_idx,
                    end_idx=last_idx,
                    confidence=1.0,
                    grammar_rule="U1b",
                    description="Transition transfers to next regime",
                )
            )

        # Fractal distribution: Ends with REMESH (Recursivity)
        if last_op == RECURSIVITY:
            patterns.append(
                PatternMatch(
                    pattern_name="fractal_distribution",
                    start_idx=last_idx,
                    end_idx=last_idx,
                    confidence=1.0,
                    grammar_rule="U1b",
                    description="Recursivity distributes across scales",
                )
            )

        # Intentional tension: Ends with OZ (Dissonance)
        if last_op == DISSONANCE:
            patterns.append(
                PatternMatch(
                    pattern_name="intentional_tension",
                    start_idx=last_idx,
                    end_idx=last_idx,
                    confidence=1.0,
                    grammar_rule="U1b",
                    description="Dissonance preserves activation/tension",
                )
            )

        return patterns

    def detect_convergence_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect U2-based convergence patterns.

        Patterns involving STABILIZERS and DESTABILIZERS to ensure bounded evolution.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of detected convergence patterns
        """
        patterns = []

        # Check for destabilizers without stabilizers (runaway risk)
        has_destabilizers = any(op in DESTABILIZERS for op in sequence)
        has_stabilizers = any(op in STABILIZERS for op in sequence)

        if has_destabilizers and not has_stabilizers:
            patterns.append(
                PatternMatch(
                    pattern_name="runaway_risk",
                    start_idx=0,
                    end_idx=len(sequence) - 1,
                    confidence=1.0,
                    grammar_rule="U2",
                    description="Destabilizers present without stabilizers (divergence risk)",
                )
            )

        # Stabilization cycle: destabilizers followed by stabilizers
        for i in range(len(sequence) - 1):
            if sequence[i] in DESTABILIZERS and sequence[i + 1] in STABILIZERS:
                patterns.append(
                    PatternMatch(
                        pattern_name="stabilization_cycle",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=1.0,
                        grammar_rule="U2",
                        description="Destabilizer → Stabilizer (bounded evolution)",
                    )
                )

        # Bounded evolution: alternating destabilizers and stabilizers
        if len(sequence) >= 4:
            alternating = True
            for i in range(0, len(sequence) - 1, 2):
                if i + 1 < len(sequence):
                    if not (sequence[i] in DESTABILIZERS and sequence[i + 1] in STABILIZERS):
                        alternating = False
                        break
            if alternating and has_destabilizers and has_stabilizers:
                patterns.append(
                    PatternMatch(
                        pattern_name="bounded_evolution",
                        start_idx=0,
                        end_idx=len(sequence) - 1,
                        confidence=0.8,
                        grammar_rule="U2",
                        description="Oscillation between destabilizers and stabilizers",
                    )
                )

        return patterns

    def detect_resonance_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect U3-based resonance patterns.

        Patterns involving COUPLING_RESONANCE operators that require phase verification.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of detected resonance patterns
        """
        patterns = []

        # Coupling chain: multiple UM (Coupling) in sequence
        coupling_count = sum(1 for op in sequence if op == COUPLING)
        if coupling_count >= 2:
            patterns.append(
                PatternMatch(
                    pattern_name="coupling_chain",
                    start_idx=0,
                    end_idx=len(sequence) - 1,
                    confidence=0.7,
                    grammar_rule="U3",
                    description=f"Multiple coupling operations ({coupling_count}) - requires phase verification",
                )
            )

        # Resonance cascade: multiple RA (Resonance) in sequence
        resonance_count = sum(1 for op in sequence if op == RESONANCE)
        if resonance_count >= 2:
            patterns.append(
                PatternMatch(
                    pattern_name="resonance_cascade",
                    start_idx=0,
                    end_idx=len(sequence) - 1,
                    confidence=0.7,
                    grammar_rule="U3",
                    description=f"Resonance propagation ({resonance_count} ops)",
                )
            )

        # Phase-locked network: alternating UM and RA
        for i in range(len(sequence) - 1):
            if (sequence[i] == COUPLING and sequence[i + 1] == RESONANCE) or (
                sequence[i] == RESONANCE and sequence[i + 1] == COUPLING
            ):
                patterns.append(
                    PatternMatch(
                        pattern_name="phase_locked_network",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=0.9,
                        grammar_rule="U3",
                        description="Coupling ↔ Resonance (synchronized network)",
                    )
                )

        return patterns

    def detect_bifurcation_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect U4-based bifurcation patterns.

        Patterns involving TRANSFORMERS and BIFURCATION_TRIGGERS.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of detected bifurcation patterns
        """
        patterns = []

        # Graduated destabilization: destabilizer → transformer (U4b)
        for i in range(len(sequence) - 1):
            if sequence[i] in DESTABILIZERS and sequence[i + 1] in TRANSFORMERS:
                patterns.append(
                    PatternMatch(
                        pattern_name="graduated_destabilization",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=1.0,
                        grammar_rule="U4b",
                        description=f"Destabilizer → Transformer ({sequence[i]} → {sequence[i+1]})",
                    )
                )

        # Managed bifurcation: trigger → handler (U4a)
        for i in range(len(sequence) - 1):
            if sequence[i] in BIFURCATION_TRIGGERS and sequence[i + 1] in BIFURCATION_HANDLERS:
                patterns.append(
                    PatternMatch(
                        pattern_name="managed_bifurcation",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=1.0,
                        grammar_rule="U4a",
                        description=f"Bifurcation trigger → handler ({sequence[i]} → {sequence[i+1]})",
                    )
                )

        # Stable transformation: IL → ZHIR sequence
        for i in range(len(sequence) - 1):
            if sequence[i] == COHERENCE and sequence[i + 1] == MUTATION:
                patterns.append(
                    PatternMatch(
                        pattern_name="stable_transformation",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=0.9,
                        grammar_rule="U4b",
                        description="Coherence → Mutation (stable base for transformation)",
                    )
                )

        # Spontaneous organization: disorder → THOL
        for i in range(len(sequence) - 1):
            if sequence[i] in DESTABILIZERS and sequence[i + 1] == SELF_ORGANIZATION:
                patterns.append(
                    PatternMatch(
                        pattern_name="spontaneous_organization",
                        start_idx=i,
                        end_idx=i + 1,
                        confidence=0.9,
                        grammar_rule="U4b",
                        description="Disorder → Self-organization",
                    )
                )

        return patterns

    def detect_all_patterns(self, sequence: Sequence[str]) -> List[PatternMatch]:
        """Detect all patterns in sequence.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        List[PatternMatch]
            List of all detected patterns
        """
        patterns = []
        patterns.extend(self.detect_initiation_patterns(sequence))
        patterns.extend(self.detect_closure_patterns(sequence))
        patterns.extend(self.detect_convergence_patterns(sequence))
        patterns.extend(self.detect_resonance_patterns(sequence))
        patterns.extend(self.detect_bifurcation_patterns(sequence))
        return patterns

    def get_grammar_rule_for_pattern(self, pattern_name: str) -> Optional[str]:
        """Get the grammar rule associated with a pattern.

        Parameters
        ----------
        pattern_name : str
            Name of the pattern

        Returns
        -------
        Optional[str]
            Grammar rule string (e.g., "U1a", "U2", "U1+U2+U4") or None
        """
        return self._pattern_grammar_map.get(pattern_name.lower())

    def analyze_sequence_composition(self, sequence: Sequence[str]) -> Mapping[str, Any]:
        """Perform comprehensive analysis of sequence structure.

        Delegates to AdvancedPatternDetector for detailed analysis.

        Parameters
        ----------
        sequence : Sequence[str]
            Operator sequence to analyze

        Returns
        -------
        Mapping[str, Any]
            Detailed analysis including patterns, scores, health metrics
        """
        return self._advanced_detector.analyze_sequence_composition(sequence)


# Convenience functions for backward compatibility


def detect_pattern(sequence: Sequence[str]) -> StructuralPattern:
    """Detect structural pattern in operator sequence.

    Convenience function that creates detector and returns pattern.

    Parameters
    ----------
    sequence : Sequence[str]
        Canonical operator names

    Returns
    -------
    StructuralPattern
        Detected structural pattern

    Examples
    --------
    >>> from tnfr.operators.pattern_detection import detect_pattern
    >>> pattern = detect_pattern(["emission", "coupling", "coherence"])  # doctest: +SKIP
    """
    detector = UnifiedPatternDetector()
    return detector.detect_pattern(sequence)


def analyze_sequence(sequence: Sequence[str]) -> Mapping[str, Any]:
    """Analyze operator sequence comprehensively.

    Convenience function for sequence analysis.

    Parameters
    ----------
    sequence : Sequence[str]
        Canonical operator names

    Returns
    -------
    Mapping[str, Any]
        Detailed analysis including patterns, scores, components

    Examples
    --------
    >>> from tnfr.operators.pattern_detection import analyze_sequence
    >>> analysis = analyze_sequence(["emission", "coupling", "coherence"])  # doctest: +SKIP
    """
    detector = UnifiedPatternDetector()
    return detector.analyze_sequence_composition(sequence)
