"""Advanced pattern detection for structural operator sequences.

This module provides unified pattern detection using coherence-weighted scoring.
All patterns are evaluated independently, but their match scores are weighted by
their structural coherence level. This respects TNFR's principle that emergent
patterns (with self-organization, phase transitions) have fundamentally higher
structural complexity than simple compositional patterns.

Coherence weights reflect observable structural depth, not arbitrary rankings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Tuple

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
    """Pattern detector using coherence-weighted scoring.
    
    Each pattern is evaluated independently and scored based on match quality.
    The final score is weighted by the pattern's structural coherence level,
    which reflects observable complexity (emergence, self-organization, phase
    transitions) rather than arbitrary ranking.
    
    This allows: weak match with emergent pattern > strong match with simple pattern,
    while still enabling perfect simple matches to win over weak emergent matches.
    """

    # Coherence weights based on structural complexity (not arbitrary hierarchy)
    # These reflect measurable properties: emergence, nested structure, phase transitions
    COHERENCE_WEIGHTS = {
        # Level 3: Emergent self-organizing patterns (highest structural depth)
        "THERAPEUTIC": 3.0,      # Full healing cycle with controlled crisis resolution
        "EDUCATIONAL": 3.0,       # Complete transformative learning with phase shift
        "ORGANIZATIONAL": 3.0,    # Institutional evolution with emergent reorganization
        "CREATIVE": 3.0,          # Artistic emergence through self-organization
        "REGENERATIVE": 3.0,      # Self-sustaining cycle with autonomous renewal
        
        # Level 2: Structural transformations (medium complexity)
        "HIERARCHICAL": 2.0,      # Self-organization creates nested structure
        "BIFURCATED": 2.0,        # Phase transition (OZ→ZHIR) branches possibility space
        "FRACTAL": 2.0,           # Recursive structure across scales
        "CYCLIC": 2.0,            # Regenerative loops with multiple state transitions
        "DEEP_LEARNING": 2.0,     # Deep adaptive learning with self-organization
        "EXPLORATORY_LEARNING": 2.0,  # Learning through exploration and resonance
        "ADAPTIVE_MUTATION": 2.0,  # Transformative learning with mutation
        
        # Level 1: Compositional patterns (building blocks)
        "BOOTSTRAP": 1.0,         # Initialization sequence
        "EXPLORE": 1.0,           # Controlled exploration
        "STABILIZE": 1.0,         # Consolidation ending
        "RESONATE": 1.0,          # Amplification through coupling
        "COMPRESS": 1.0,          # Simplification sequence
        "BASIC_LEARNING": 1.0,    # Simple learning sequence
        "CONSOLIDATION_CYCLE": 1.0,  # Memory consolidation
        
        # Level 0: Simple patterns
        "LINEAR": 0.5,            # Basic progression without transformation
        "MINIMAL": 0.5,           # Single or very few operators
        
        # Special cases
        "COMPLEX": 1.5,           # Multiple patterns combined
        "UNKNOWN": 0.1,           # Fallback for unclassified sequences
    }

    def __init__(self) -> None:
        """Initialize the coherence-weighted pattern detector with caching."""
        # Pattern signatures: each pattern maps to a list of operator subsequences
        # that characterize it, along with required/optional operators
        self._patterns = {
            # Fundamental patterns
            "LINEAR": {
                "max_length": 5,
                "excludes": {DISSONANCE, MUTATION, SELF_ORGANIZATION},
                "min_score": 0.1,  # Base score for matching criteria
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
            
            # Adaptive learning patterns (AL + T'HOL canonical sequences)
            "BASIC_LEARNING": {
                "subsequences": [[EMISSION, RECEPTION, COHERENCE, SILENCE]],
                "max_length": 6,
            },
            "DEEP_LEARNING": {
                "subsequences": [[EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, COHERENCE, SILENCE]],
                "requires": {EMISSION, SELF_ORGANIZATION},
            },
            "EXPLORATORY_LEARNING": {
                "subsequences": [[EMISSION, RECEPTION, COHERENCE, DISSONANCE, SELF_ORGANIZATION, RESONANCE, COHERENCE, SILENCE]],
                "requires": {EMISSION, DISSONANCE, SELF_ORGANIZATION, RESONANCE},
            },
            "CONSOLIDATION_CYCLE": {
                "subsequences": [[EMISSION, RECEPTION, COHERENCE, RECURSIVITY]],
                "max_length": 6,
            },
            "ADAPTIVE_MUTATION": {
                "subsequences": [[EMISSION, RECEPTION, COHERENCE, DISSONANCE, MUTATION, TRANSITION]],
                "requires": {EMISSION, DISSONANCE, MUTATION},
            },
            
            "MINIMAL": {
                "max_length": 1,
                "min_score": 0.1,
            },
        }
        
        # Cache for pattern detection results (maxsize=256 to cache common sequences)
        self._detect_cache = lru_cache(maxsize=256)(self._detect_pattern_cached)

    def detect_pattern(self, sequence: Sequence[str]) -> StructuralPattern:
        """Detect the best matching pattern using coherence-weighted scoring.
        
        Parameters
        ----------
        sequence : Sequence[str]
            Canonical operator names to analyze.
            
        Returns
        -------
        StructuralPattern
            The pattern with the highest coherence-weighted score.
            
        Notes
        -----
        All patterns are evaluated and scored independently. The final score for
        each pattern is: match_score * coherence_weight.
        
        This means:
        - A weak match (0.3) with an emergent pattern (weight 3.0) scores 0.9
        - A strong match (0.4) with a compositional pattern (weight 1.0) scores 0.4
        - The emergent pattern wins, respecting structural depth
        
        However:
        - A perfect match (0.8) with LINEAR (weight 0.5) scores 0.4
        - A weak match (0.3) with THERAPEUTIC (weight 3.0) scores 0.9
        - THERAPEUTIC wins, but if LINEAR had 1.0 match, it would score 0.5 and still lose
        - This is correct: even a perfect simple pattern has less structural depth
        """
        # Convert to tuple for caching (sequences must be hashable)
        sequence_tuple = tuple(sequence) if not isinstance(sequence, tuple) else sequence
        return self._detect_cache(sequence_tuple)
    
    def _detect_pattern_cached(self, sequence_tuple: Tuple[str, ...]) -> StructuralPattern:
        """Cached implementation of pattern detection with grammar validation.
        
        Parameters
        ----------
        sequence_tuple : Tuple[str, ...]
            Immutable sequence of canonical operator names.
            
        Returns
        -------
        StructuralPattern
            The pattern with the highest coherence-weighted score.
            
        Notes
        -----
        **Grammar Validation**: Sequences are validated against TNFR canonical
        grammar before pattern matching. Invalid sequences return UNKNOWN pattern.
        This ensures that only grammatically coherent sequences are recognized
        as canonical patterns.
        
        **Note**: Grammar validation is performed without triggering pattern detection
        to avoid recursion.
        """
        from .grammar import StructuralPattern
        
        sequence = list(sequence_tuple)
        
        if not sequence:
            return StructuralPattern.UNKNOWN
        
        # **NEW**: Validate sequence grammar before pattern detection
        # Use validate_sequence which now won't call back to pattern detection
        try:
            from ..validation import validate_sequence
            validation_result = validate_sequence(sequence)
            if not validation_result.passed:
                # Sequence violates TNFR grammar - cannot be a canonical pattern
                return StructuralPattern.UNKNOWN
        except RecursionError:
            # If recursion occurs, skip grammar validation to break the cycle
            # This can happen if validate_sequence internally calls pattern detection
            pass
        
        # Score all patterns with coherence weighting
        weighted_scores = {}
        raw_scores = {}
        for pattern_name, criteria in self._patterns.items():
            raw_score = self._score_pattern(sequence, criteria)
            if raw_score > 0:
                weight = self.COHERENCE_WEIGHTS.get(pattern_name, 1.0)
                weighted_scores[pattern_name] = raw_score * weight
                raw_scores[pattern_name] = raw_score
        
        # Handle COMPLEX: long sequences with multiple good raw matches
        # But NOT if we have a clear domain pattern match
        if len(sequence) > 8 and len(raw_scores) >= 3:
            # Check if we have a strong domain pattern match (even if weighted)
            domain_patterns = ["THERAPEUTIC", "EDUCATIONAL", "ORGANIZATIONAL", "CREATIVE", "REGENERATIVE"]
            domain_weighted_scores = {
                p: weighted_scores.get(p, 0)
                for p in domain_patterns
                if p in weighted_scores
            }
            if domain_weighted_scores:
                max_domain_score = max(domain_weighted_scores.values())
                # If a domain pattern has the highest weighted score, don't mark as COMPLEX
                if max_domain_score == max(weighted_scores.values()):
                    pass  # Let it fall through to return the best pattern
                else:
                    # Check if we have diverse high-coherence patterns
                    high_coherence_patterns = [
                        p for p, s in raw_scores.items()
                        if s > 0.3 and self.COHERENCE_WEIGHTS.get(p, 1.0) >= 2.0
                    ]
                    if len(high_coherence_patterns) >= 2:
                        return StructuralPattern.COMPLEX
            else:
                # No domain patterns, check for general complexity
                high_coherence_patterns = [
                    p for p, s in raw_scores.items()
                    if s > 0.3 and self.COHERENCE_WEIGHTS.get(p, 1.0) >= 2.0
                ]
                if len(high_coherence_patterns) >= 2:
                    return StructuralPattern.COMPLEX
        
        # Return best weighted match or UNKNOWN
        if not weighted_scores:
            return StructuralPattern.UNKNOWN
        
        best_pattern = max(weighted_scores, key=weighted_scores.get)
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
        
        # Add base score if pattern has min_score (e.g., for LINEAR)
        if "min_score" in criteria:
            score += criteria["min_score"]
        
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
            - primary_pattern: Best matching pattern (coherence-weighted)
            - pattern_scores: Raw match scores for all patterns
            - weighted_scores: Coherence-weighted scores for all patterns
            - coherence_weights: The weight applied to each pattern
            - components: List of identified sub-patterns
            - complexity_score: Measure of sequence complexity (0.0-1.0)
            - domain_suitability: Scores for different application domains
            - structural_health: Coherence and stability metrics
        """
        primary = self.detect_pattern(sequence)
        
        # Get raw and weighted scores for all patterns
        pattern_scores = {}
        weighted_scores = {}
        for pattern_name, criteria in self._patterns.items():
            raw_score = self._score_pattern(sequence, criteria)
            if raw_score > 0:
                pattern_scores[pattern_name] = raw_score
                weight = self.COHERENCE_WEIGHTS.get(pattern_name, 1.0)
                weighted_scores[pattern_name] = raw_score * weight
        
        components = self._identify_components(sequence)
        
        return {
            "primary_pattern": primary.value,
            "pattern_scores": pattern_scores,
            "weighted_scores": weighted_scores,
            "coherence_weights": dict(self.COHERENCE_WEIGHTS),
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
        """Identify all pattern components present in the sequence.
        
        Components are identified based on partial matches OR presence of
        characteristic subsequences, even if the full pattern doesn't match.
        """
        components = []
        
        # Check each pattern for partial matches (score > 0 but not dominant)
        for pattern_name, criteria in self._patterns.items():
            score = self._score_pattern(seq, criteria)
            if 0 < score < 0.8:  # Partial match
                components.append(pattern_name.lower())
        
        # Also check for characteristic subsequences even if pattern doesn't fully match
        # (e.g., BOOTSTRAP subsequence in a long sequence)
        bootstrap_seq = [EMISSION, COUPLING, COHERENCE]
        if self._contains_subsequence(seq, bootstrap_seq) and "bootstrap" not in components:
            components.append("bootstrap")
        
        explore_seq = [DISSONANCE, MUTATION, COHERENCE]
        if self._contains_subsequence(seq, explore_seq) and "explore" not in components:
            components.append("explore")
        
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
            "basic_learning": "BASIC_LEARNING",
            "deep_learning": "DEEP_LEARNING",
            "exploratory_learning": "EXPLORATORY_LEARNING",
            "consolidation_cycle": "CONSOLIDATION_CYCLE",
            "adaptive_mutation": "ADAPTIVE_MUTATION",
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
