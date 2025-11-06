"""Advanced pattern detection for structural operator sequences.

This module provides unified pattern detection without hierarchies. All patterns
are scored based on how well they match the sequence, and the best match is
returned. This approach is simpler, more extensible, and aligns with TNFR's
principle of avoiding artificial hierarchies.
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
    """Unified pattern detector using match scoring instead of hierarchies.
    
    Each pattern is evaluated independently, and the pattern with the highest
    match score is returned. This treats all patterns equally and makes the
    system easily extensible with new patterns.
    """

    def __init__(self) -> None:
        """Initialize the unified pattern detector with all pattern signatures."""
        # Pattern signatures: each pattern maps to a list of operator subsequences
        # that characterize it, along with required/optional operators
        self._patterns = {
            # Fundamental patterns
            "LINEAR": {
                "subsequences": [],
                "max_length": 5,
                "excludes": {DISSONANCE, MUTATION, SELF_ORGANIZATION},
            },
            "HIERARCHICAL": {
                "requires": {SELF_ORGANIZATION},
            },
            "FRACTAL": {
                "requires": {TRANSITION},
                "requires_any": {COUPLING, RECURSIVITY},
            },
            "CYCLIC": {
                "min_count": {TRANSITION: 2},
            },
            "BIFURCATED": {
                "adjacent_pairs": [(DISSONANCE, MUTATION), (DISSONANCE, CONTRACTION)],
            },
            
            # Domain-specific patterns
            "THERAPEUTIC": {
                "subsequences": [
                    [RECEPTION, EMISSION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE]
                ],
            },
            "EDUCATIONAL": {
                "subsequences": [
                    [RECEPTION, EMISSION, COHERENCE, EXPANSION, DISSONANCE, MUTATION]
                ],
            },
            "ORGANIZATIONAL": {
                "subsequences": [
                    [TRANSITION, EMISSION, RECEPTION, COUPLING, RESONANCE, DISSONANCE, SELF_ORGANIZATION]
                ],
            },
            "CREATIVE": {
                "subsequences": [
                    [SILENCE, EMISSION, EXPANSION, DISSONANCE, MUTATION, SELF_ORGANIZATION]
                ],
            },
            "REGENERATIVE": {
                "subsequences": [
                    [COHERENCE, RESONANCE, EXPANSION, SILENCE, TRANSITION, EMISSION, RECEPTION, COUPLING, COHERENCE]
                ],
            },
            
            # Compositional patterns
            "BOOTSTRAP": {
                "subsequences": [[EMISSION, COUPLING, COHERENCE]],
                "max_length": 5,
            },
            "EXPLORE": {
                "subsequences": [[DISSONANCE, MUTATION, COHERENCE]],
            },
            "STABILIZE": {
                "ending_pairs": [(COHERENCE, SILENCE), (COHERENCE, RESONANCE)],
            },
            "RESONATE": {
                "subsequences": [[RESONANCE, COUPLING, RESONANCE]],
            },
            "COMPRESS": {
                "subsequences": [[CONTRACTION, COHERENCE, SILENCE]],
            },
        }

    def detect_pattern(self, sequence: Sequence[str]) -> StructuralPattern:
        """Detect the best matching pattern using unified scoring.
        
        Parameters
        ----------
        sequence : Sequence[str]
            Canonical operator names to analyze.
            
        Returns
        -------
        StructuralPattern
            The pattern with the highest match score.
        """
        from .grammar import StructuralPattern
        
        if not sequence:
            return StructuralPattern.MINIMAL
        
        if len(sequence) == 1:
            return StructuralPattern.MINIMAL
        
        # Score all patterns
        scores = {}
        for pattern_name, criteria in self._patterns.items():
            score = self._score_pattern(sequence, criteria)
            if score > 0:
                scores[pattern_name] = score
        
        # Handle COMPLEX: long sequences with multiple good matches
        if len(sequence) > 8 and len(scores) >= 3:
            return StructuralPattern.COMPLEX
        
        # Return best match or UNKNOWN
        if not scores:
            return StructuralPattern.UNKNOWN
        
        best_pattern = max(scores, key=scores.get)
        return StructuralPattern[best_pattern]

    def _score_pattern(
        self, sequence: Sequence[str], criteria: dict[str, Any]
    ) -> float:
        """Score how well a sequence matches pattern criteria.
        
        Returns
        -------
        float
            Score from 0.0 (no match) to 1.0+ (perfect match).
        """
        score = 0.0
        
        # Check subsequences (highest weight)
        if "subsequences" in criteria:
            for subseq in criteria["subsequences"]:
                if self._contains_subsequence(sequence, subseq):
                    # Score based on coverage: how much of sequence is in pattern
                    coverage = len(subseq) / len(sequence)
                    score += 0.8 * coverage
        
        # Check required operators
        if "requires" in criteria:
            if all(op in sequence for op in criteria["requires"]):
                score += 0.3
            else:
                return 0.0  # Hard requirement
        
        # Check any-of requirements
        if "requires_any" in criteria:
            if any(op in sequence for op in criteria["requires_any"]):
                score += 0.2
            else:
                return 0.0  # Hard requirement
        
        # Check minimum counts
        if "min_count" in criteria:
            for op, min_val in criteria["min_count"].items():
                if sequence.count(op) >= min_val:
                    score += 0.4
                else:
                    return 0.0  # Hard requirement
        
        # Check adjacent pairs
        if "adjacent_pairs" in criteria:
            for op1, op2 in criteria["adjacent_pairs"]:
                if self._has_adjacent_pair(sequence, op1, op2):
                    score += 0.5
                    break
            else:
                return 0.0  # None found
        
        # Check ending pairs
        if "ending_pairs" in criteria and len(sequence) >= 2:
            for op1, op2 in criteria["ending_pairs"]:
                if sequence[-2] == op1 and sequence[-1] == op2:
                    score += 0.4
                    break
            else:
                return 0.0  # None found
        
        # Check excludes
        if "excludes" in criteria:
            if any(op in sequence for op in criteria["excludes"]):
                return 0.0  # Exclusion violated
        
        # Check max length
        if "max_length" in criteria:
            if len(sequence) > criteria["max_length"]:
                return 0.0  # Too long
        
        return score

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
            - primary_pattern: Best matching pattern
            - pattern_scores: All patterns with their scores
            - components: List of identified sub-patterns
            - complexity_score: Measure of sequence complexity (0.0-1.0)
            - domain_suitability: Scores for different application domains
            - structural_health: Coherence and stability metrics
        """
        primary = self.detect_pattern(sequence)
        
        # Get scores for all patterns
        pattern_scores = {}
        for pattern_name, criteria in self._patterns.items():
            score = self._score_pattern(sequence, criteria)
            if score > 0:
                pattern_scores[pattern_name] = score
        
        components = self._identify_components(sequence)
        
        return {
            "primary_pattern": primary.value,
            "pattern_scores": pattern_scores,
            "components": components,
            "complexity_score": self._calculate_complexity(sequence),
            "domain_suitability": self._assess_domain_fit(sequence),
            "structural_health": self._calculate_health_metrics(sequence),
        }

    # Helper methods (keep existing implementations) --------------------------

    def _contains_subsequence(
        self, sequence: Sequence[str], pattern: Sequence[str]
    ) -> bool:
        """Check if pattern exists as a subsequence within sequence."""
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

    def _has_adjacent_pair(
        self, sequence: Sequence[str], op1: str, op2: str
    ) -> bool:
        """Check if op1 is immediately followed by op2."""
        for i in range(len(sequence) - 1):
            if sequence[i] == op1 and sequence[i + 1] == op2:
                return True
        return False

    def _identify_components(self, seq: Sequence[str]) -> list[str]:
        """Identify all pattern components present in the sequence."""
        components = []
        
        # Check each pattern for partial matches
        for pattern_name, criteria in self._patterns.items():
            score = self._score_pattern(seq, criteria)
            if 0 < score < 0.8:  # Partial match
                components.append(pattern_name.lower())
        
        # Add structural indicators
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
        """Calculate complexity score based on sequence characteristics."""
        if not seq:
            return 0.0
        
        # Factors contributing to complexity
        length_factor = min(len(seq) / 15.0, 1.0)
        
        unique_count = len(set(seq))
        diversity_factor = unique_count / len(seq)
        
        complex_ops = {DISSONANCE, MUTATION, SELF_ORGANIZATION, TRANSITION}
        complex_count = sum(1 for op in seq if op in complex_ops)
        complexity_factor = min(complex_count / 5.0, 1.0)
        
        return (
            0.3 * length_factor +
            0.3 * diversity_factor +
            0.4 * complexity_factor
        )

    def _assess_domain_fit(self, seq: Sequence[str]) -> Mapping[str, float]:
        """Assess suitability for different application domains."""
        scores: dict[str, float] = {}
        
        # Score based on domain-specific pattern matches
        domain_patterns = {
            "therapeutic": "THERAPEUTIC",
            "educational": "EDUCATIONAL",
            "organizational": "ORGANIZATIONAL",
            "creative": "CREATIVE",
            "regenerative": "REGENERATIVE",
        }
        
        for domain, pattern_name in domain_patterns.items():
            if pattern_name in self._patterns:
                score = self._score_pattern(seq, self._patterns[pattern_name])
                scores[domain] = min(score, 1.0)
        
        return scores

    def _calculate_health_metrics(self, seq: Sequence[str]) -> Mapping[str, Any]:
        """Calculate structural health indicators."""
        stabilizers = sum(
            1 for op in seq
            if op in {COHERENCE, SILENCE, RESONANCE}
        )
        destabilizers = sum(
            1 for op in seq
            if op in {DISSONANCE, MUTATION, EXPANSION}
        )
        
        total = len(seq)
        balance = (stabilizers - destabilizers) / total if total > 0 else 0.0
        
        return {
            "stabilizer_count": stabilizers,
            "destabilizer_count": destabilizers,
            "balance": balance,
            "has_closure": seq[-1] in {SILENCE, TRANSITION, RECURSIVITY} if seq else False,
        }
