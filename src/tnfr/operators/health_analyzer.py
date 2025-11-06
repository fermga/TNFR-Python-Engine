"""Structural health metrics analyzer for TNFR operator sequences.

Provides quantitative assessment of sequence structural quality through
canonical TNFR metrics: coherence, balance, sustainability, and efficiency.
"""

from __future__ import annotations

from typing import List

from ..compat.dataclass import dataclass
from ..config.operator_names import (
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EXPANSION,
    MUTATION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    CONTRACTION,
    DESTABILIZERS,
    TRANSFORMERS,
)

__all__ = [
    "SequenceHealthMetrics",
    "SequenceHealthAnalyzer",
]


# Operator categories for health analysis
_STABILIZERS = frozenset({COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE})
_DESTABILIZERS = DESTABILIZERS  # OZ, NAV, VAL
_TRANSFORMERS = TRANSFORMERS  # ZHIR, THOL
_REGENERATORS = frozenset({TRANSITION, RECURSIVITY})  # NAV, REMESH


@dataclass
class SequenceHealthMetrics:
    """Structural health metrics for a TNFR operator sequence.
    
    All metrics range from 0.0 (poor) to 1.0 (excellent), measuring different
    aspects of sequence structural quality according to TNFR principles.
    
    Attributes
    ----------
    coherence_index : float
        Global sequential flow quality (0.0-1.0). Measures how well operators
        transition and whether the sequence forms a recognizable pattern.
    balance_score : float
        Equilibrium between stabilizers and destabilizers (0.0-1.0). Ideal
        sequences have balanced structural forces.
    sustainability_index : float
        Capacity for long-term maintenance (0.0-1.0). Considers final stabilization,
        resolved dissonance, and regenerative elements.
    complexity_efficiency : float
        Value-to-complexity ratio (0.0-1.0). Penalizes unnecessarily long sequences
        that don't provide proportional structural value.
    frequency_harmony : float
        Structural frequency transition smoothness (0.0-1.0). High when transitions
        respect νf harmonics.
    pattern_completeness : float
        How complete the detected pattern is (0.0-1.0). Full cycles score higher.
    transition_smoothness : float
        Quality of operator transitions (0.0-1.0). Measures valid transitions vs
        total transitions.
    overall_health : float
        Composite health index (0.0-1.0). Weighted average of primary metrics.
    sequence_length : int
        Number of operators in the sequence.
    dominant_pattern : str
        Detected structural pattern type (e.g., "activation", "therapeutic", "unknown").
    recommendations : List[str]
        Specific suggestions for improving sequence health.
    """
    
    coherence_index: float
    balance_score: float
    sustainability_index: float
    complexity_efficiency: float
    frequency_harmony: float
    pattern_completeness: float
    transition_smoothness: float
    overall_health: float
    sequence_length: int
    dominant_pattern: str
    recommendations: List[str]


class SequenceHealthAnalyzer:
    """Analyzer for structural health of TNFR operator sequences.
    
    Evaluates sequences along multiple dimensions to provide quantitative
    assessment of structural quality, coherence, and sustainability.
    
    Examples
    --------
    >>> from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
    >>> analyzer = SequenceHealthAnalyzer()
    >>> sequence = ["emission", "reception", "coherence", "silence"]
    >>> health = analyzer.analyze_health(sequence)
    >>> health.overall_health
    0.82
    >>> health.recommendations
    []
    """
    
    def __init__(self) -> None:
        """Initialize the health analyzer."""
        self._recommendations: List[str] = []
    
    def analyze_health(self, sequence: List[str]) -> SequenceHealthMetrics:
        """Perform complete structural health analysis of a sequence.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence to analyze (canonical names like "emission", "coherence").
        
        Returns
        -------
        SequenceHealthMetrics
            Comprehensive health metrics for the sequence.
        
        Examples
        --------
        >>> analyzer = SequenceHealthAnalyzer()
        >>> health = analyzer.analyze_health(["emission", "reception", "coherence", "silence"])
        >>> health.coherence_index > 0.7
        True
        """
        self._recommendations = []
        
        coherence = self._calculate_coherence(sequence)
        balance = self._calculate_balance(sequence)
        sustainability = self._calculate_sustainability(sequence)
        efficiency = self._calculate_efficiency(sequence)
        frequency = self._calculate_frequency_harmony(sequence)
        completeness = self._calculate_completeness(sequence)
        smoothness = self._calculate_smoothness(sequence)
        
        # Calculate overall health as weighted average
        # Primary metrics weighted more heavily
        overall = (
            coherence * 0.20 +
            balance * 0.20 +
            sustainability * 0.20 +
            efficiency * 0.15 +
            frequency * 0.10 +
            completeness * 0.10 +
            smoothness * 0.05
        )
        
        pattern = self._detect_pattern(sequence)
        
        return SequenceHealthMetrics(
            coherence_index=coherence,
            balance_score=balance,
            sustainability_index=sustainability,
            complexity_efficiency=efficiency,
            frequency_harmony=frequency,
            pattern_completeness=completeness,
            transition_smoothness=smoothness,
            overall_health=overall,
            sequence_length=len(sequence),
            dominant_pattern=pattern,
            recommendations=self._recommendations.copy(),
        )
    
    def _calculate_coherence(self, sequence: List[str]) -> float:
        """Calculate coherence index: how well the sequence flows.
        
        Factors:
        - Valid transitions between operators
        - Recognizable pattern structure
        - Structural closure (proper ending)
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Coherence score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        # Transition quality: ratio of valid transitions
        transition_quality = self._assess_all_transitions(sequence)
        
        # Pattern clarity: does it form a recognizable structure?
        pattern_clarity = self._assess_pattern_clarity(sequence)
        
        # Structural closure: does it end properly?
        structural_closure = self._assess_closure(sequence)
        
        return (transition_quality + pattern_clarity + structural_closure) / 3.0
    
    def _calculate_balance(self, sequence: List[str]) -> float:
        """Calculate balance score: equilibrium between stabilizers and destabilizers.
        
        Ideal sequences have roughly equal stabilization and transformation forces.
        Severe imbalance reduces structural health.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Balance score (0.0-1.0)
        """
        if not sequence:
            return 0.5  # Neutral for empty
        
        stabilizers = sum(1 for op in sequence if op in _STABILIZERS)
        destabilizers = sum(1 for op in sequence if op in _DESTABILIZERS)
        
        # If neither present, neutral balance
        if stabilizers == 0 and destabilizers == 0:
            return 0.5
        
        # Calculate ratio: closer to 1.0 means better balance
        max_count = max(stabilizers, destabilizers)
        min_count = min(stabilizers, destabilizers)
        
        if max_count == 0:
            return 0.5
        
        ratio = min_count / max_count
        
        # Penalize severe imbalance (difference > half the sequence length)
        imbalance = abs(stabilizers - destabilizers)
        if imbalance > len(sequence) // 2:
            ratio *= 0.7  # Apply penalty
            self._recommendations.append(
                "Severe imbalance detected: add stabilizers or reduce destabilizers"
            )
        
        return ratio
    
    def _calculate_sustainability(self, sequence: List[str]) -> float:
        """Calculate sustainability index: capacity to maintain without collapse.
        
        Factors:
        - Final operator is a stabilizer
        - Dissonance is resolved (not left unbalanced)
        - Contains regenerative elements
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Sustainability score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        sustainability = 0.0
        
        # Factor 1: Ends with stabilizer (0.4 points)
        has_final_stabilizer = sequence[-1] in _STABILIZERS
        if has_final_stabilizer:
            sustainability += 0.4
        else:
            sustainability += 0.1  # Some credit for other endings
            self._recommendations.append(
                "Consider ending with a stabilizer (coherence, silence, resonance, or self_organization)"
            )
        
        # Factor 2: Resolved dissonance (0.3 points)
        unresolved_dissonance = self._count_unresolved_dissonance(sequence)
        if unresolved_dissonance == 0:
            sustainability += 0.3
        else:
            penalty = min(0.3, unresolved_dissonance * 0.1)
            sustainability += max(0, 0.3 - penalty)
            if unresolved_dissonance > 1:
                self._recommendations.append(
                    "Multiple unresolved dissonances detected: add stabilizers after destabilizing operators"
                )
        
        # Factor 3: Regenerative elements (0.3 points)
        has_regenerative = any(op in _REGENERATORS for op in sequence)
        if has_regenerative:
            sustainability += 0.3
        else:
            sustainability += 0.1  # Some credit even without
        
        return min(1.0, sustainability)
    
    def _calculate_efficiency(self, sequence: List[str]) -> float:
        """Calculate complexity efficiency: value achieved relative to length.
        
        Penalizes unnecessarily long sequences that don't provide proportional value.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Efficiency score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        # Assess structural value: diversity and balance of operator types
        pattern_value = self._assess_pattern_value(sequence)
        
        # Length penalty: sequences longer than 10 operators get penalized
        # Optimal range is 3-8 operators
        length = len(sequence)
        if length < 3:
            length_factor = 0.7  # Too short, limited value
        elif length <= 8:
            length_factor = 1.0  # Optimal range
        else:
            # Gradual penalty for length > 8
            excess = length - 8
            length_factor = max(0.5, 1.0 - (excess * 0.05))
        
        if length > 12:
            self._recommendations.append(
                f"Sequence is long ({length} operators): consider breaking into sub-sequences"
            )
        
        return pattern_value * length_factor
    
    def _calculate_frequency_harmony(self, sequence: List[str]) -> float:
        """Calculate frequency harmony: smoothness of νf transitions.
        
        Not yet fully implemented; returns high score as placeholder.
        Requires frequency transition matrix from grammar module.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Harmony score (0.0-1.0)
        """
        # Placeholder: assume good harmony unless we detect obvious issues
        # In future, integrate with STRUCTURAL_FREQUENCIES and FREQUENCY_TRANSITIONS
        return 0.85
    
    def _calculate_completeness(self, sequence: List[str]) -> float:
        """Calculate pattern completeness: how complete the pattern is.
        
        Complete patterns (with activation, transformation, stabilization) score higher.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Completeness score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        # Check for key phases
        has_activation = any(op in {"emission", "reception"} for op in sequence)
        has_transformation = any(op in _DESTABILIZERS | _TRANSFORMERS for op in sequence)
        has_stabilization = any(op in _STABILIZERS for op in sequence)
        has_completion = any(op in {"silence", "transition"} for op in sequence)
        
        phase_count = sum([has_activation, has_transformation, has_stabilization, has_completion])
        
        # All 4 phases = 1.0, 3 phases = 0.75, 2 phases = 0.5, 1 phase = 0.25
        return phase_count / 4.0
    
    def _calculate_smoothness(self, sequence: List[str]) -> float:
        """Calculate transition smoothness: quality of operator transitions.
        
        Measures ratio of valid/smooth transitions vs total transitions.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Smoothness score (0.0-1.0)
        """
        return self._assess_all_transitions(sequence)
    
    def _assess_all_transitions(self, sequence: List[str]) -> float:
        """Assess quality of all transitions in the sequence.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Transition quality ratio (0.0-1.0)
        """
        if len(sequence) < 2:
            return 1.0  # No transitions to assess
        
        # Simple heuristic: most transitions are valid in TNFR
        # Problematic: destabilizer → destabilizer without stabilization
        total_transitions = len(sequence) - 1
        problematic = 0
        
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_op = sequence[i + 1]
            
            # Check for problematic patterns
            # Multiple destabilizers in a row without stabilization
            if current in _DESTABILIZERS and next_op in _DESTABILIZERS:
                problematic += 0.5  # Partial penalty
        
        return max(0.0, 1.0 - (problematic / total_transitions))
    
    def _assess_pattern_clarity(self, sequence: List[str]) -> float:
        """Assess how clearly the sequence forms a recognizable pattern.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Pattern clarity score (0.0-1.0)
        """
        if len(sequence) < 3:
            return 0.5  # Too short for clear pattern
        
        # Check for canonical patterns
        pattern = self._detect_pattern(sequence)
        
        if pattern in {"activation", "therapeutic", "regenerative", "transformative"}:
            return 0.9  # Clear, recognized pattern
        elif pattern in {"stabilization", "exploratory"}:
            return 0.7  # Recognizable but simpler
        else:
            return 0.5  # No clear pattern
    
    def _assess_closure(self, sequence: List[str]) -> float:
        """Assess structural closure quality.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Closure quality score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        # Valid endings per grammar
        valid_endings = {SILENCE, TRANSITION, RECURSIVITY, DISSONANCE}
        
        if sequence[-1] in valid_endings:
            # Stabilizer endings are best
            if sequence[-1] in _STABILIZERS:
                return 1.0
            # Other valid endings are good
            return 0.8
        
        # Invalid ending
        return 0.3
    
    def _count_unresolved_dissonance(self, sequence: List[str]) -> int:
        """Count destabilizers not followed by stabilizers within reasonable window.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        int
            Count of unresolved dissonant operators
        """
        unresolved = 0
        window = 3  # Look ahead up to 3 operators
        
        for i, op in enumerate(sequence):
            if op in _DESTABILIZERS:
                # Check if a stabilizer appears in the next 'window' operators
                lookahead = sequence[i + 1:i + 1 + window]
                if not any(stabilizer in _STABILIZERS for stabilizer in lookahead):
                    unresolved += 1
        
        return unresolved
    
    def _assess_pattern_value(self, sequence: List[str]) -> float:
        """Assess the structural value of the pattern.
        
        Value is higher when:
        - Multiple operator types present (diversity)
        - Key structural phases are included
        - Balance between forces
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        float
            Pattern value score (0.0-1.0)
        """
        if not sequence:
            return 0.0
        
        # Diversity: unique operators used
        unique_count = len(set(sequence))
        diversity_score = min(1.0, unique_count / 6.0)  # 6+ operators is excellent diversity
        
        # Coverage: how many operator categories are represented
        categories_present = 0
        if any(op in {"emission", "reception"} for op in sequence):
            categories_present += 1  # Activation
        if any(op in _STABILIZERS for op in sequence):
            categories_present += 1  # Stabilization
        if any(op in _DESTABILIZERS for op in sequence):
            categories_present += 1  # Destabilization
        if any(op in _TRANSFORMERS for op in sequence):
            categories_present += 1  # Transformation
        
        coverage_score = categories_present / 4.0
        
        # Combine factors
        return (diversity_score * 0.5) + (coverage_score * 0.5)
    
    def _detect_pattern(self, sequence: List[str]) -> str:
        """Detect the dominant structural pattern type.
        
        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        
        Returns
        -------
        str
            Pattern name (e.g., "activation", "therapeutic", "unknown")
        """
        if not sequence:
            return "empty"
        
        # Check for common patterns
        starts_with_emission = sequence[0] == "emission"
        has_reception = "reception" in sequence
        has_coherence = COHERENCE in sequence
        has_dissonance = DISSONANCE in sequence
        has_self_org = SELF_ORGANIZATION in sequence
        has_regenerator = any(op in _REGENERATORS for op in sequence)
        
        # Pattern detection logic
        if starts_with_emission and has_reception and has_coherence:
            if has_dissonance and has_self_org:
                return "therapeutic"
            elif has_regenerator:
                return "regenerative"
            else:
                return "activation"
        
        if has_dissonance and has_self_org:
            return "transformative"
        
        if sum(1 for op in sequence if op in _STABILIZERS) > len(sequence) // 2:
            return "stabilization"
        
        if sum(1 for op in sequence if op in _DESTABILIZERS) > len(sequence) // 2:
            return "exploratory"
        
        return "unknown"
