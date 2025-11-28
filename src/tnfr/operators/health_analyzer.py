"""Structural health metrics analyzer for TNFR operator sequences.

Provides quantitative assessment of sequence structural quality through
canonical TNFR metrics: coherence, balance, sustainability, and efficiency.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import TNFRGraph

from ..compat.dataclass import dataclass
from ..config.operator_names import (
    COHERENCE,
    DISSONANCE,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    DESTABILIZERS,
    TRANSFORMERS,
)

__all__ = [
    "SequenceHealthMetrics",
    "SequenceHealthAnalyzer",
]


# Operator categories for health analysis
_STABILIZERS = frozenset({COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE})
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

    Uses caching to optimize repeated analysis of identical sequences,
    which is common in pattern exploration and batch validation workflows.

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
        """Initialize the health analyzer with caching support."""
        self._recommendations: List[str] = []
        # Cache for single-pass analysis results keyed by sequence tuple
        # Using maxsize=128 to avoid unbounded growth while caching common sequences
        self._analysis_cache = lru_cache(maxsize=128)(self._compute_single_pass)

    def _compute_single_pass(
        self, sequence_tuple: Tuple[str, ...]
    ) -> Tuple[int, int, int, int, int, List[Tuple[str, str]]]:
        """Compute sequence statistics in a single pass for efficiency.

        This method scans the sequence once and extracts all the information
        needed for the various health metrics, avoiding redundant iterations.

        Parameters
        ----------
        sequence_tuple : Tuple[str, ...]
            Immutable sequence of operators (tuple for hashability in cache).

        Returns
        -------
        Tuple containing:
            - stabilizer_count: int
            - destabilizer_count: int
            - transformer_count: int
            - regenerator_count: int
            - unique_ops: int
            - problematic_transitions: List[(op1, op2)] pairs

        Notes
        -----
        This function is cached using lru_cache to optimize repeated analysis
        of identical sequences, which is common in batch validation and
        pattern exploration workflows.
        """
        sequence = list(sequence_tuple)

        # Initialize counters
        stabilizer_count = 0
        destabilizer_count = 0
        transformer_count = 0
        regenerator_count = 0
        unique_ops_set = set()
        problematic_transitions = []

        # Single pass through sequence
        for i, op in enumerate(sequence):
            unique_ops_set.add(op)

            # Count operator categories
            if op in _STABILIZERS:
                stabilizer_count += 1
            if op in DESTABILIZERS:
                destabilizer_count += 1
            if op in TRANSFORMERS:
                transformer_count += 1
            if op in _REGENERATORS:
                regenerator_count += 1

            # Check transitions
            if i < len(sequence) - 1:
                next_op = sequence[i + 1]
                # Destabilizer → destabilizer is problematic
                if op in DESTABILIZERS and next_op in DESTABILIZERS:
                    problematic_transitions.append((op, next_op))

        return (
            stabilizer_count,
            destabilizer_count,
            transformer_count,
            regenerator_count,
            len(unique_ops_set),
            problematic_transitions,
        )

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

        # Use single-pass analysis for efficiency (cached)
        sequence_tuple = tuple(sequence)
        analysis = self._analysis_cache(sequence_tuple)

        # Extract results from single-pass analysis
        (
            stabilizer_count,
            destabilizer_count,
            transformer_count,
            regenerator_count,
            unique_count,
            problematic_transitions,
        ) = analysis

        coherence = self._calculate_coherence(sequence, problematic_transitions)
        balance = self._calculate_balance(sequence, stabilizer_count, destabilizer_count)
        sustainability = self._calculate_sustainability(
            sequence, stabilizer_count, destabilizer_count, regenerator_count
        )
        efficiency = self._calculate_efficiency(sequence, unique_count)
        frequency = self._calculate_frequency_harmony(sequence)
        completeness = self._calculate_completeness(
            sequence, stabilizer_count, destabilizer_count, transformer_count
        )
        smoothness = self._calculate_smoothness(sequence, problematic_transitions)

        # Calculate overall health as weighted average
        # Primary metrics weighted more heavily
        overall = (
            coherence * 0.20
            + balance * 0.20
            + sustainability * 0.20
            + efficiency * 0.15
            + frequency * 0.10
            + completeness * 0.10
            + smoothness * 0.05
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

    def _calculate_coherence(
        self, sequence: List[str], problematic_transitions: List[Tuple[str, str]]
    ) -> float:
        """Calculate coherence index: how well the sequence flows.

        Factors:
        - Valid transitions between operators
        - Recognizable pattern structure
        - Structural closure (proper ending)

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        problematic_transitions : List[Tuple[str, str]]
            Pre-computed list of problematic transition pairs

        Returns
        -------
        float
            Coherence score (0.0-1.0)
        """
        if not sequence:
            return 0.0

        # Transition quality: use pre-computed problematic transitions
        if len(sequence) < 2:
            transition_quality = 1.0
        else:
            total_transitions = len(sequence) - 1
            # Each problematic transition gets 0.5 penalty
            penalty = len(problematic_transitions) * 0.5
            transition_quality = max(0.0, 1.0 - (penalty / total_transitions))

        # Pattern clarity: does it form a recognizable structure?
        pattern_clarity = self._assess_pattern_clarity(sequence)

        # Structural closure: does it end properly?
        structural_closure = self._assess_closure(sequence)

        return (transition_quality + pattern_clarity + structural_closure) / 3.0

    def _calculate_balance(
        self, sequence: List[str], stabilizer_count: int, destabilizer_count: int
    ) -> float:
        """Calculate balance score: equilibrium between stabilizers and destabilizers.

        Ideal sequences have roughly equal stabilization and transformation forces.
        Severe imbalance reduces structural health.

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        stabilizer_count : int
            Pre-computed count of stabilizing operators
        destabilizer_count : int
            Pre-computed count of destabilizing operators

        Returns
        -------
        float
            Balance score (0.0-1.0)
        """
        if not sequence:
            return 0.5  # Neutral for empty

        # If neither present, neutral balance
        if stabilizer_count == 0 and destabilizer_count == 0:
            return 0.5

        # Calculate ratio: closer to 1.0 means better balance
        max_count = max(stabilizer_count, destabilizer_count)
        min_count = min(stabilizer_count, destabilizer_count)

        if max_count == 0:
            return 0.5

        ratio = min_count / max_count

        # Penalize severe imbalance (difference > half the sequence length)
        imbalance = abs(stabilizer_count - destabilizer_count)
        if imbalance > len(sequence) // 2:
            ratio *= 0.7  # Apply penalty
            self._recommendations.append(
                "Severe imbalance detected: add stabilizers or reduce destabilizers"
            )

        return ratio

    def _calculate_sustainability(
        self,
        sequence: List[str],
        stabilizer_count: int,
        destabilizer_count: int,
        regenerator_count: int,
    ) -> float:
        """Calculate sustainability index: capacity to maintain without collapse.

        Factors:
        - Final operator is a stabilizer
        - Dissonance is resolved (not left unbalanced)
        - Contains regenerative elements

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        stabilizer_count : int
            Pre-computed count of stabilizing operators
        destabilizer_count : int
            Pre-computed count of destabilizing operators
        regenerator_count : int
            Pre-computed count of regenerative operators

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
        # Use pre-computed regenerator count
        if regenerator_count > 0:
            sustainability += 0.3
        else:
            sustainability += 0.1  # Some credit even without

        return min(1.0, sustainability)

    def _calculate_efficiency(self, sequence: List[str], unique_count: int) -> float:
        """Calculate complexity efficiency: value achieved relative to length.

        Penalizes unnecessarily long sequences that don't provide proportional value.

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        unique_count : int
            Pre-computed count of unique operators in sequence

        Returns
        -------
        float
            Efficiency score (0.0-1.0)
        """
        if not sequence:
            return 0.0

        # Note: We call _assess_pattern_value for category coverage
        # This is minimal overhead as it's a single pass checking set memberships
        pattern_value = self._assess_pattern_value_optimized(sequence, unique_count)

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

        Note: Full implementation requires integration with STRUCTURAL_FREQUENCIES
        and FREQUENCY_TRANSITIONS from the grammar module. Currently returns
        a conservative estimate based on transition patterns.

        Parameters
        ----------
        sequence : List[str]
            Operator sequence

        Returns
        -------
        float
            Harmony score (0.0-1.0)
        """
        # Conservative estimate: assume good harmony unless obvious issues detected
        # Future enhancement: integrate with grammar.STRUCTURAL_FREQUENCIES
        return 0.85

    def _calculate_completeness(
        self,
        sequence: List[str],
        stabilizer_count: int,
        destabilizer_count: int,
        transformer_count: int,
    ) -> float:
        """Calculate pattern completeness: how complete the pattern is.

        Complete patterns (with activation, transformation, stabilization) score higher.

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        stabilizer_count : int
            Pre-computed count of stabilizing operators
        destabilizer_count : int
            Pre-computed count of destabilizing operators
        transformer_count : int
            Pre-computed count of transforming operators

        Returns
        -------
        float
            Completeness score (0.0-1.0)
        """
        if not sequence:
            return 0.0

        # Check for key phases using pre-computed counts and minimal checks
        has_activation = any(op in {"emission", "reception"} for op in sequence)
        has_transformation = destabilizer_count > 0 or transformer_count > 0
        has_stabilization = stabilizer_count > 0
        has_completion = any(op in {"silence", "transition"} for op in sequence)

        phase_count = sum([has_activation, has_transformation, has_stabilization, has_completion])

        # All 4 phases = 1.0, 3 phases = 0.75, 2 phases = 0.5, 1 phase = 0.25
        return phase_count / 4.0

    def _calculate_smoothness(
        self, sequence: List[str], problematic_transitions: List[Tuple[str, str]]
    ) -> float:
        """Calculate transition smoothness: quality of operator transitions.

        Measures ratio of valid/smooth transitions vs total transitions.

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        problematic_transitions : List[Tuple[str, str]]
            Pre-computed list of problematic transition pairs

        Returns
        -------
        float
            Smoothness score (0.0-1.0)
        """
        if len(sequence) < 2:
            return 1.0  # No transitions to assess

        total_transitions = len(sequence) - 1
        # Each problematic transition gets 0.5 penalty (same as in _calculate_coherence)
        penalty = len(problematic_transitions) * 0.5
        return max(0.0, 1.0 - (penalty / total_transitions))

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
            if op in DESTABILIZERS:
                # Check if a stabilizer appears in the next 'window' operators
                lookahead = sequence[i + 1 : i + 1 + window]
                if not any(stabilizer in _STABILIZERS for stabilizer in lookahead):
                    unresolved += 1

        return unresolved

    def _assess_pattern_value_optimized(self, sequence: List[str], unique_count: int) -> float:
        """Assess the structural value of the pattern using pre-computed unique count.

        Value is higher when:
        - Multiple operator types present (diversity)
        - Key structural phases are included
        - Balance between forces

        Parameters
        ----------
        sequence : List[str]
            Operator sequence
        unique_count : int
            Pre-computed count of unique operators in sequence

        Returns
        -------
        float
            Pattern value score (0.0-1.0)
        """
        if not sequence:
            return 0.0

        # Diversity: use pre-computed unique count
        diversity_score = min(1.0, unique_count / 6.0)  # 6+ operators is excellent diversity

        # Coverage: how many operator categories are represented
        # This is still a minimal single-pass check
        categories_present = 0
        if any(op in {"emission", "reception"} for op in sequence):
            categories_present += 1  # Activation
        if any(op in _STABILIZERS for op in sequence):
            categories_present += 1  # Stabilization
        if any(op in DESTABILIZERS for op in sequence):
            categories_present += 1  # Destabilization
        if any(op in TRANSFORMERS for op in sequence):
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

        if sum(1 for op in sequence if op in DESTABILIZERS) > len(sequence) // 2:
            return "exploratory"

        return "unknown"

    def analyze_thol_coherence(self, G: TNFRGraph) -> dict[str, Any] | None:
        """Analyze collective coherence of THOL bifurcations across the network.

        Examines all nodes that have undergone THOL bifurcation and provides
        statistics on their collective coherence metrics.

        Parameters
        ----------
        G : TNFRGraph
            Graph containing nodes with potential THOL bifurcations

        Returns
        -------
        dict or None
            Dictionary containing coherence statistics:
            - mean_coherence: Average coherence across all THOL nodes
            - min_coherence: Lowest coherence value observed
            - max_coherence: Highest coherence value observed
            - nodes_below_threshold: Count of nodes with coherence < 0.3
            - total_thol_nodes: Total nodes with sub-EPIs
            Returns None if no THOL bifurcations exist in the network.

        Notes
        -----
        TNFR Principle: Collective coherence measures the structural alignment
        of emergent sub-EPIs. Low coherence may indicate chaotic fragmentation
        rather than controlled emergence.

        This metric is particularly useful for:
        - Detecting pathological bifurcation patterns
        - Monitoring network-wide self-organization quality
        - Identifying nodes requiring stabilization

        Examples
        --------
        >>> analyzer = SequenceHealthAnalyzer()
        >>> # After running THOL operations on graph G
        >>> coherence_stats = analyzer.analyze_thol_coherence(G)
        >>> if coherence_stats:
        ...     print(f"Mean coherence: {coherence_stats['mean_coherence']:.3f}")
        ...     print(f"Nodes below threshold: {coherence_stats['nodes_below_threshold']}")
        """
        # Find all nodes with sub-EPIs (THOL bifurcation occurred)
        thol_nodes = []
        for node in G.nodes():
            if G.nodes[node].get("sub_epis"):
                thol_nodes.append(node)

        if not thol_nodes:
            return None

        # Collect coherence values
        coherences = []
        for node in thol_nodes:
            coh = G.nodes[node].get("_thol_collective_coherence")
            if coh is not None:
                coherences.append(coh)

        if not coherences:
            return None

        # Compute statistics
        mean_coherence = sum(coherences) / len(coherences)
        min_coherence = min(coherences)
        max_coherence = max(coherences)

        # Get threshold from graph config
        threshold = float(G.graph.get("THOL_MIN_COLLECTIVE_COHERENCE", 0.3))
        nodes_below_threshold = sum(1 for c in coherences if c < threshold)

        return {
            "mean_coherence": mean_coherence,
            "min_coherence": min_coherence,
            "max_coherence": max_coherence,
            "nodes_below_threshold": nodes_below_threshold,
            "total_thol_nodes": len(thol_nodes),
            "threshold": threshold,
        }
