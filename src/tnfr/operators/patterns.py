"""Advanced pattern detection for structural operator sequences.

This module extends basic pattern recognition with domain-specific patterns
identified in TNFR theoretical applications. It provides precise detection
for therapeutic, educational, organizational, creative, and regenerative
sequences, as well as meta-patterns for common compositional components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .grammar import StructuralPattern

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

__all__ = ["AdvancedPatternDetector"]


class AdvancedPatternDetector:
    """Detects advanced structural patterns in TNFR operator sequences.
    
    Analyzes sequences to identify domain-specific patterns (therapeutic,
    educational, organizational, creative, regenerative) and meta-patterns
    (bootstrap, explore, stabilize). Falls back to basic pattern detection
    when advanced patterns are not present.
    """

    def __init__(self) -> None:
        """Initialize the advanced pattern detector."""
        # Pattern signatures for domain-specific sequences
        self._therapeutic_signature = [RECEPTION, EMISSION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE]
        self._educational_signature = [RECEPTION, EMISSION, COHERENCE, EXPANSION, DISSONANCE, MUTATION]
        self._organizational_signature = [TRANSITION, EMISSION, RECEPTION, COUPLING, RESONANCE, DISSONANCE, SELF_ORGANIZATION]
        self._creative_signature = [SILENCE, EMISSION, EXPANSION, DISSONANCE, MUTATION, SELF_ORGANIZATION]
        self._regenerative_signature = [COHERENCE, RESONANCE, EXPANSION, SILENCE, TRANSITION, EMISSION, RECEPTION, COUPLING, COHERENCE]
        
        # Meta-pattern signatures
        self._bootstrap_signature = [EMISSION, COUPLING, COHERENCE]
        self._explore_signature = [DISSONANCE, MUTATION, COHERENCE]
        self._stabilize_endings = {SILENCE, RESONANCE}

    def detect_pattern(self, sequence: Sequence[str]) -> StructuralPattern:
        """Detect the primary structural pattern in a sequence.
        
        Parameters
        ----------
        sequence : Sequence[str]
            Canonical operator names to analyze.
            
        Returns
        -------
        StructuralPattern
            The detected pattern type.
            
        Notes
        -----
        Detection proceeds in priority order:
        1. Domain-specific patterns (exact or partial matches)
        2. Basic patterns with high specificity (HIERARCHICAL, BIFURCATED)
        3. Meta-patterns (compositional components)
        4. Other basic patterns (CYCLIC, FRACTAL, LINEAR)
        5. COMPLEX for long sequences with multiple patterns
        6. UNKNOWN for unclassified sequences
        """
        from .grammar import StructuralPattern
        
        # Domain-specific patterns (highest priority)
        if self._is_therapeutic_pattern(sequence):
            return StructuralPattern.THERAPEUTIC
        if self._is_educational_pattern(sequence):
            return StructuralPattern.EDUCATIONAL
        if self._is_organizational_pattern(sequence):
            return StructuralPattern.ORGANIZATIONAL
        if self._is_creative_pattern(sequence):
            return StructuralPattern.CREATIVE
        if self._is_regenerative_pattern(sequence):
            return StructuralPattern.REGENERATIVE
        
        # High-specificity basic patterns (before meta-patterns for backward compatibility)
        # HIERARCHICAL: contains THOL
        if SELF_ORGANIZATION in sequence:
            return StructuralPattern.HIERARCHICAL
        
        # BIFURCATED: OZ followed by ZHIR or NUL
        for i in range(len(sequence) - 1):
            if sequence[i] == DISSONANCE and sequence[i + 1] in {MUTATION, CONTRACTION}:
                return StructuralPattern.BIFURCATED
        
        # Meta-patterns (medium priority)
        if self._is_bootstrap_pattern(sequence):
            return StructuralPattern.BOOTSTRAP
        if self._is_explore_pattern(sequence):
            return StructuralPattern.EXPLORE
        if self._is_stabilize_pattern(sequence):
            return StructuralPattern.STABILIZE
        
        # Complex sequences with multiple sub-patterns
        if len(sequence) > 8 and self._has_multiple_subpatterns(sequence):
            return StructuralPattern.COMPLEX
        
        # Remaining basic patterns (CYCLIC, FRACTAL, LINEAR)
        return self._detect_remaining_basic_patterns(sequence)

    def analyze_sequence_composition(
        self, sequence: Sequence[str]
    ) -> Mapping[str, Any]:
        """Perform comprehensive analysis of sequence structure.
        
        Parameters
        ----------
        sequence : Sequence[str]
            Canonical operator names to analyze.
            
        Returns
        -------
        Mapping[str, Any]
            Analysis containing:
            - primary_pattern: Main detected pattern
            - components: List of identified sub-patterns
            - complexity_score: Measure of sequence complexity (0.0-1.0)
            - domain_suitability: Scores for different application domains
            - structural_health: Coherence and stability metrics
        """
        primary = self.detect_pattern(sequence)
        components = self._identify_components(sequence)
        
        return {
            "primary_pattern": primary.value,
            "components": components,
            "complexity_score": self._calculate_complexity(sequence),
            "domain_suitability": self._assess_domain_fit(sequence),
            "structural_health": self._calculate_health_metrics(sequence),
        }

    # Domain-specific pattern detectors ----------------------------------------

    def _is_therapeutic_pattern(self, seq: Sequence[str]) -> bool:
        """Detect therapeutic pattern: listen→activate→crisis→reorganize→integrate.
        
        Signature: EN→AL→IL→OZ→THOL→IL
        Recognizes healing/therapeutic processes with controlled crisis.
        """
        return self._contains_subsequence(seq, self._therapeutic_signature)

    def _is_educational_pattern(self, seq: Sequence[str]) -> bool:
        """Detect educational pattern: receive→activate→expand→transform→consolidate.
        
        Signature: EN→AL→IL→VAL→OZ→ZHIR
        Recognizes transformative learning processes.
        """
        return self._contains_subsequence(seq, self._educational_signature)

    def _is_organizational_pattern(self, seq: Sequence[str]) -> bool:
        """Detect organizational pattern: navigate→activate→couple→resonate→evolve.
        
        Signature: NAV→AL→EN→UM→RA→OZ→THOL
        Recognizes institutional evolution and organizational change.
        """
        return self._contains_subsequence(seq, self._organizational_signature)

    def _is_creative_pattern(self, seq: Sequence[str]) -> bool:
        """Detect creative pattern: pause→activate→expand→transform→self-organize.
        
        Signature: SHA→AL→VAL→OZ→ZHIR→THOL
        Recognizes artistic and creative processes.
        """
        return self._contains_subsequence(seq, self._creative_signature)

    def _is_regenerative_pattern(self, seq: Sequence[str]) -> bool:
        """Detect regenerative pattern: cyclic self-sustaining structure.
        
        Signature: IL→RA→VAL→SHA→NAV→AL→EN→UM→IL
        Recognizes self-sustaining, regenerative cycles.
        """
        return self._contains_subsequence(seq, self._regenerative_signature)

    # Meta-pattern detectors ---------------------------------------------------

    def _is_bootstrap_pattern(self, seq: Sequence[str]) -> bool:
        """Detect bootstrap pattern: rapid initialization.
        
        Signature: AL→UM→IL
        Exact match for quick node activation.
        """
        return (
            len(seq) <= 5
            and self._contains_subsequence(seq, self._bootstrap_signature)
        )

    def _is_explore_pattern(self, seq: Sequence[str]) -> bool:
        """Detect explore pattern: controlled exploration.
        
        Signature: OZ→ZHIR→IL
        Recognizes controlled mutation and exploration sequences.
        """
        return self._contains_subsequence(seq, self._explore_signature)

    def _is_stabilize_pattern(self, seq: Sequence[str]) -> bool:
        """Detect stabilize pattern: consolidation ending.
        
        Signature: *→IL→{SHA|RA}
        Recognizes sequences ending with stabilization.
        """
        if len(seq) < 2:
            return False
        return (
            seq[-2] == COHERENCE
            and seq[-1] in self._stabilize_endings
        )

    # Basic pattern detection (fallback) ---------------------------------------

    def _detect_basic_pattern(self, seq: Sequence[str]) -> StructuralPattern:
        """Detect basic structural patterns using existing logic.
        
        This replicates the original _detect_pattern logic from _SequenceAutomaton
        to maintain backward compatibility.
        """
        from .grammar import StructuralPattern
        
        # Handle empty sequences
        if not seq:
            return StructuralPattern.UNKNOWN
        
        # Hierarchical: contains THOL
        if SELF_ORGANIZATION in seq:
            return StructuralPattern.HIERARCHICAL
        
        # Bifurcated: OZ followed by ZHIR or NUL
        for i in range(len(seq) - 1):
            if seq[i] == DISSONANCE and seq[i + 1] in {MUTATION, CONTRACTION}:
                return StructuralPattern.BIFURCATED
        
        # Cyclic: multiple NAV transitions
        if seq.count(TRANSITION) >= 2:
            return StructuralPattern.CYCLIC
        
        # Fractal: NAV with coupling or recursivity
        if TRANSITION in seq and (COUPLING in seq or RECURSIVITY in seq):
            return StructuralPattern.FRACTAL
        
        # Linear: simple progression without complexity
        if len(seq) <= 5 and DISSONANCE not in seq and MUTATION not in seq:
            return StructuralPattern.LINEAR
        
        return StructuralPattern.UNKNOWN

    def _detect_remaining_basic_patterns(self, seq: Sequence[str]) -> StructuralPattern:
        """Detect remaining basic patterns (CYCLIC, FRACTAL, LINEAR, UNKNOWN).
        
        Used after domain-specific, HIERARCHICAL, BIFURCATED, and meta-patterns
        have been checked. This ensures proper priority ordering while maintaining
        backward compatibility.
        """
        from .grammar import StructuralPattern
        
        # Handle empty sequences
        if not seq:
            return StructuralPattern.UNKNOWN
        
        # Cyclic: multiple NAV transitions
        if seq.count(TRANSITION) >= 2:
            return StructuralPattern.CYCLIC
        
        # Fractal: NAV with coupling or recursivity
        if TRANSITION in seq and (COUPLING in seq or RECURSIVITY in seq):
            return StructuralPattern.FRACTAL
        
        # Linear: simple progression without complexity
        if len(seq) <= 5 and DISSONANCE not in seq and MUTATION not in seq:
            return StructuralPattern.LINEAR
        
        return StructuralPattern.UNKNOWN

    # Helper methods -----------------------------------------------------------

    def _contains_subsequence(
        self, sequence: Sequence[str], pattern: Sequence[str]
    ) -> bool:
        """Check if pattern exists as a subsequence within sequence.
        
        Parameters
        ----------
        sequence : Sequence[str]
            The full operator sequence to search within.
        pattern : Sequence[str]
            The pattern subsequence to find.
            
        Returns
        -------
        bool
            True if pattern is found as a contiguous subsequence.
        """
        if len(pattern) > len(sequence):
            return False
        
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if all(
                sequence[i + j] == pattern[j]
                for j in range(pattern_len)
            ):
                return True
        return False

    def _identify_components(self, seq: Sequence[str]) -> list[str]:
        """Identify all pattern components present in the sequence.
        
        Returns a list of pattern types that appear as sub-patterns,
        useful for understanding sequence composition.
        """
        components = []
        
        # Check for meta-patterns as components
        if self._contains_subsequence(seq, self._bootstrap_signature):
            components.append("bootstrap")
        if self._contains_subsequence(seq, self._explore_signature):
            components.append("explore")
        if len(seq) >= 2 and seq[-2] == COHERENCE and seq[-1] in self._stabilize_endings:
            components.append("stabilize")
        
        # Check for domain pattern elements
        if DISSONANCE in seq:
            components.append("crisis")
        if SELF_ORGANIZATION in seq:
            components.append("reorganization")
        if MUTATION in seq:
            components.append("transformation")
        if seq.count(TRANSITION) >= 2:
            components.append("cyclic_navigation")
        
        return components

    def _calculate_complexity(self, seq: Sequence[str]) -> float:
        """Calculate complexity score based on sequence characteristics.
        
        Returns
        -------
        float
            Complexity score from 0.0 (simple) to 1.0 (highly complex).
        """
        if not seq:
            return 0.0
        
        # Factors contributing to complexity
        length_factor = min(len(seq) / 15.0, 1.0)  # Normalize to 15 operators
        
        # Count unique operators (diversity)
        unique_count = len(set(seq))
        diversity_factor = unique_count / len(seq)
        
        # Count high-complexity operators
        complex_ops = {DISSONANCE, MUTATION, SELF_ORGANIZATION, TRANSITION}
        complex_count = sum(1 for op in seq if op in complex_ops)
        complexity_factor = min(complex_count / 5.0, 1.0)  # Normalize to 5
        
        # Weighted average
        return (
            0.3 * length_factor +
            0.3 * diversity_factor +
            0.4 * complexity_factor
        )

    def _assess_domain_fit(self, seq: Sequence[str]) -> Mapping[str, float]:
        """Assess suitability for different application domains.
        
        Returns
        -------
        Mapping[str, float]
            Domain names mapped to suitability scores (0.0-1.0).
        """
        scores: dict[str, float] = {
            "therapeutic": 0.0,
            "educational": 0.0,
            "organizational": 0.0,
            "creative": 0.0,
            "regenerative": 0.0,
        }
        
        # Score based on partial pattern matches
        if self._contains_subsequence(seq, [RECEPTION, EMISSION, COHERENCE]):
            scores["therapeutic"] += 0.4
        if DISSONANCE in seq and SELF_ORGANIZATION in seq:
            scores["therapeutic"] += 0.3
        
        if self._contains_subsequence(seq, [RECEPTION, EMISSION, EXPANSION]):
            scores["educational"] += 0.4
        if MUTATION in seq:
            scores["educational"] += 0.3
        
        if TRANSITION in seq and COUPLING in seq:
            scores["organizational"] += 0.4
        if RESONANCE in seq:
            scores["organizational"] += 0.2
        
        if SILENCE in seq and EXPANSION in seq:
            scores["creative"] += 0.4
        if MUTATION in seq and SELF_ORGANIZATION in seq:
            scores["creative"] += 0.3
        
        if seq.count(TRANSITION) >= 2:
            scores["regenerative"] += 0.5
        if COHERENCE in seq and RESONANCE in seq:
            scores["regenerative"] += 0.3
        
        return scores

    def _calculate_health_metrics(self, seq: Sequence[str]) -> Mapping[str, Any]:
        """Calculate structural health indicators.
        
        Returns
        -------
        Mapping[str, Any]
            Health metrics including stabilizers, destabilizers, and balance.
        """
        stabilizers = sum(
            1 for op in seq
            if op in {COHERENCE, SILENCE, RESONANCE}
        )
        destabilizers = sum(
            1 for op in seq
            if op in {DISSONANCE, MUTATION, EXPANSION}
        )
        
        total = len(seq)
        if total == 0:
            balance = 0.0
        else:
            balance = (stabilizers - destabilizers) / total
        
        return {
            "stabilizer_count": stabilizers,
            "destabilizer_count": destabilizers,
            "balance": balance,
            "has_closure": seq[-1] in {SILENCE, TRANSITION, RECURSIVITY} if seq else False,
        }

    def _has_multiple_subpatterns(self, seq: Sequence[str]) -> bool:
        """Check if sequence contains multiple distinct sub-patterns.
        
        Returns True if sequence shows evidence of combining multiple
        structural patterns.
        """
        pattern_count = 0
        
        # Check for various sub-patterns
        if SELF_ORGANIZATION in seq:
            pattern_count += 1
        if seq.count(TRANSITION) >= 2:
            pattern_count += 1
        if DISSONANCE in seq and MUTATION in seq:
            pattern_count += 1
        if RECURSIVITY in seq:
            pattern_count += 1
        
        return pattern_count >= 2
