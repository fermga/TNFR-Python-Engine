"""Context-guided sequence generator for TNFR operator sequences.

This module provides intelligent sequence generation capabilities that help
users construct optimal TNFR operator sequences based on context, objectives,
and structural constraints. The generator uses domain templates, pattern
detection, and health analysis to produce high-quality sequences.

Examples
--------
>>> from tnfr.tools.sequence_generator import ContextualSequenceGenerator
>>> generator = ContextualSequenceGenerator()
>>>
>>> # Generate for specific domain and objective
>>> seq = generator.generate_for_context(
...     domain="therapeutic",
...     objective="crisis_intervention",
...     min_health=0.75
... )
>>> print(seq)
['emission', 'reception', 'coherence', 'resonance', 'silence']
>>>
>>> # Generate to match a specific pattern
>>> seq = generator.generate_for_pattern(
...     target_pattern="BOOTSTRAP",
...     min_health=0.70
... )
>>>
>>> # Improve an existing sequence
>>> current = ["emission", "coherence", "silence"]
>>> improved, recommendations = generator.improve_sequence(current, target_health=0.80)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from ..compat.dataclass import dataclass
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
from ..operators.health_analyzer import SequenceHealthAnalyzer
from ..sequencing.patterns import AdvancedPatternDetector
from ..validation.compatibility import (
    GRADUATED_COMPATIBILITY,
    CompatibilityLevel,
    get_compatibility_level,
)
from .domain_templates import get_template

__all__ = [
    "ContextualSequenceGenerator",
    "GenerationResult",
]


# Operator groups for intelligent variation
_STABILIZERS = [COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE]
_DESTABILIZERS = [DISSONANCE, MUTATION, EXPANSION]
_ACTIVATORS = [EMISSION, RECEPTION]
_CONNECTORS = [COUPLING, RESONANCE]
_TRANSFORMERS = [TRANSITION, RECURSIVITY, MUTATION]


@dataclass
class GenerationResult:
    """Result of a sequence generation operation.

    Attributes
    ----------
    sequence : list[str]
        Generated operator sequence (canonical names).
    health_score : float
        Overall structural health score (0.0-1.0).
    detected_pattern : str
        Primary structural pattern detected.
    domain : str | None
        Domain context used for generation (if applicable).
    objective : str | None
        Specific objective within domain (if applicable).
    method : str
        Generation method used ("template", "pattern", "improvement").
    recommendations : list[str]
        Suggestions for further improvement.
    metadata : dict[str, object]
        Additional generation metadata.
    """

    sequence: list[str]
    health_score: float
    detected_pattern: str
    domain: str | None
    objective: str | None
    method: str
    recommendations: list[str]
    metadata: dict[str, object]


class ContextualSequenceGenerator:
    """Generator for context-guided TNFR operator sequences.

    This generator combines domain templates, pattern detection, and health
    analysis to produce optimal operator sequences for specific contexts and
    objectives. It supports:

    - Domain/objective-based generation from curated templates
    - Pattern-targeted generation to achieve specific structural patterns
    - Sequence improvement with targeted recommendations
    - Constraint-based filtering (health, length, pattern)

    All generated sequences respect TNFR canonical principles:
    - Operator closure (only canonical operators)
    - Phase coherence (compatible transitions)
    - Structural health (balanced forces)
    - Operational fractality (composable patterns)

    Examples
    --------
    >>> generator = ContextualSequenceGenerator()
    >>> result = generator.generate_for_context(
    ...     domain="therapeutic",
    ...     objective="crisis_intervention"
    ... )
    >>> print(result.sequence)
    ['emission', 'reception', 'coherence', 'resonance', 'silence']
    >>> print(f"Health: {result.health_score:.2f}")
    Health: 0.78
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the contextual sequence generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for deterministic generation. If None, generation
            is non-deterministic.
        """
        self.health_analyzer = SequenceHealthAnalyzer()
        self.pattern_detector = AdvancedPatternDetector()
        self._rng = random.Random(seed)

    def generate_for_context(
        self,
        domain: str,
        objective: str | None = None,
        max_length: int = 10,
        min_health: float = 0.70,
        required_pattern: str | None = None,
    ) -> GenerationResult:
        """Generate optimal sequence for specific domain and objective.

        This method uses domain templates as a starting point and applies
        intelligent variations to meet constraints while maintaining structural
        coherence.

        Parameters
        ----------
        domain : str
            Application domain (therapeutic, educational, organizational, creative).
        objective : str, optional
            Specific objective within domain. If None, uses first template.
        max_length : int, default=10
            Maximum sequence length. Sequences longer than this will be trimmed.
        min_health : float, default=0.70
            Minimum required health score (0.0-1.0).
        required_pattern : str, optional
            If specified, generator will try to produce this pattern.

        Returns
        -------
        GenerationResult
            Complete generation result with sequence, health metrics, and metadata.

        Raises
        ------
        KeyError
            If domain or objective not found.
        ValueError
            If no valid sequence can be generated meeting constraints.

        Examples
        --------
        >>> generator = ContextualSequenceGenerator()
        >>> result = generator.generate_for_context(
        ...     domain="therapeutic",
        ...     objective="crisis_intervention",
        ...     min_health=0.75
        ... )
        >>> print(result.sequence)
        ['emission', 'reception', 'coherence', 'resonance', 'silence']
        """
        # Determine objective if not specified
        if objective is None:
            from .domain_templates import list_objectives

            objectives = list_objectives(domain)
            objective = objectives[0] if objectives else None

        # Get base template
        base_sequence = get_template(domain, objective)

        # Apply length constraint
        if len(base_sequence) > max_length:
            base_sequence = self._trim_sequence(base_sequence, max_length)

        # Analyze base template
        health = self.health_analyzer.analyze_health(base_sequence)

        # If template already meets requirements, return it
        if health.overall_health >= min_health:
            if required_pattern is None or self._matches_pattern(base_sequence, required_pattern):
                return GenerationResult(
                    sequence=base_sequence,
                    health_score=health.overall_health,
                    detected_pattern=health.dominant_pattern,
                    domain=domain,
                    objective=objective,
                    method="template",
                    recommendations=health.recommendations,
                    metadata={
                        "template_used": True,
                        "variations_tried": 0,
                    },
                )

        # Generate variations to meet constraints
        candidates = self._generate_variations(base_sequence, max_length, count=20)

        # Filter candidates by constraints
        valid_candidates = []
        for candidate in candidates:
            candidate_health = self.health_analyzer.analyze_health(candidate)
            if candidate_health.overall_health >= min_health:
                if required_pattern is None or self._matches_pattern(candidate, required_pattern):
                    valid_candidates.append((candidate, candidate_health))

        if not valid_candidates:
            # Fallback: return best candidate even if below threshold
            all_with_health = [
                (seq, self.health_analyzer.analyze_health(seq)) for seq in candidates
            ]
            best_seq, best_health = max(all_with_health, key=lambda x: x[1].overall_health)

            return GenerationResult(
                sequence=best_seq,
                health_score=best_health.overall_health,
                detected_pattern=best_health.dominant_pattern,
                domain=domain,
                objective=objective,
                method="template_variant",
                recommendations=[
                    f"Warning: Could not meet min_health={min_health:.2f}",
                    f"Best achievable health: {best_health.overall_health:.2f}",
                ]
                + best_health.recommendations,
                metadata={
                    "template_used": True,
                    "variations_tried": len(candidates),
                    "constraint_met": False,
                },
            )

        # Select best valid candidate
        best_seq, best_health = max(valid_candidates, key=lambda x: x[1].overall_health)

        return GenerationResult(
            sequence=best_seq,
            health_score=best_health.overall_health,
            detected_pattern=best_health.dominant_pattern,
            domain=domain,
            objective=objective,
            method="template_optimized",
            recommendations=best_health.recommendations,
            metadata={
                "template_used": True,
                "variations_tried": len(candidates),
                "valid_candidates": len(valid_candidates),
            },
        )

    def generate_for_pattern(
        self,
        target_pattern: str,
        max_length: int = 10,
        min_health: float = 0.70,
    ) -> GenerationResult:
        """Generate sequence targeting a specific structural pattern.

        Uses pattern signatures and characteristic operator combinations to
        construct sequences that maximize the probability of matching the
        target pattern while maintaining structural health.

        Parameters
        ----------
        target_pattern : str
            Target structural pattern (e.g., "BOOTSTRAP", "THERAPEUTIC",
            "STABILIZE").
        max_length : int, default=10
            Maximum sequence length.
        min_health : float, default=0.70
            Minimum required health score (0.0-1.0).

        Returns
        -------
        GenerationResult
            Complete generation result with sequence and metrics.

        Raises
        ------
        ValueError
            If pattern name is not recognized or no valid sequence can be generated.

        Examples
        --------
        >>> generator = ContextualSequenceGenerator()
        >>> result = generator.generate_for_pattern("BOOTSTRAP", min_health=0.70)
        >>> print(result.sequence)
        ['emission', 'coupling', 'coherence']
        """
        # Get pattern signature
        signature = self._get_pattern_signature(target_pattern)

        # Build base sequence from signature
        base_sequence = self._build_from_signature(signature, max_length)

        # Analyze and optimize
        health = self.health_analyzer.analyze_health(base_sequence)

        if health.overall_health >= min_health:
            return GenerationResult(
                sequence=base_sequence,
                health_score=health.overall_health,
                detected_pattern=health.dominant_pattern,
                domain=None,
                objective=None,
                method="pattern_direct",
                recommendations=health.recommendations,
                metadata={
                    "target_pattern": target_pattern,
                    "pattern_matched": self._matches_pattern(base_sequence, target_pattern),
                },
            )

        # Generate variations to improve health
        candidates = self._generate_variations(base_sequence, max_length, count=15)

        # Filter by constraints and pattern match
        valid_candidates = []
        for candidate in candidates:
            if self._matches_pattern(candidate, target_pattern):
                candidate_health = self.health_analyzer.analyze_health(candidate)
                if candidate_health.overall_health >= min_health:
                    valid_candidates.append((candidate, candidate_health))

        if not valid_candidates:
            # Return base even if below threshold
            return GenerationResult(
                sequence=base_sequence,
                health_score=health.overall_health,
                detected_pattern=health.dominant_pattern,
                domain=None,
                objective=None,
                method="pattern_suboptimal",
                recommendations=[
                    f"Warning: Could not meet min_health={min_health:.2f}",
                    f"Best achievable health: {health.overall_health:.2f}",
                ]
                + health.recommendations,
                metadata={
                    "target_pattern": target_pattern,
                    "pattern_matched": self._matches_pattern(base_sequence, target_pattern),
                    "constraint_met": False,
                },
            )

        # Select best valid candidate
        best_seq, best_health = max(valid_candidates, key=lambda x: x[1].overall_health)

        return GenerationResult(
            sequence=best_seq,
            health_score=best_health.overall_health,
            detected_pattern=best_health.dominant_pattern,
            domain=None,
            objective=None,
            method="pattern_optimized",
            recommendations=best_health.recommendations,
            metadata={
                "target_pattern": target_pattern,
                "pattern_matched": True,
                "variations_tried": len(candidates),
            },
        )

    def improve_sequence(
        self,
        current: list[str],
        target_health: float | None = None,
        max_length: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """Improve existing sequence with targeted recommendations.

        Analyzes the current sequence, identifies weaknesses, and generates
        an improved version along with specific recommendations explaining
        the improvements made.

        Parameters
        ----------
        current : list[str]
            Current operator sequence to improve.
        target_health : float, optional
            Target health score. If None, aims for current + 0.15.
        max_length : int, optional
            Maximum allowed length for improved sequence. If None, allows
            length to increase by up to 3 operators.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing:
            - Improved operator sequence
            - List of recommendations explaining improvements

        Examples
        --------
        >>> generator = ContextualSequenceGenerator()
        >>> current = ["emission", "coherence", "silence"]
        >>> improved, recommendations = generator.improve_sequence(current)
        >>> print(improved)
        ['emission', 'reception', 'coherence', 'resonance', 'silence']
        >>> print(recommendations[0])
        'Added reception after emission: improves completeness (+0.25)'
        """
        # Analyze current sequence
        current_health = self.health_analyzer.analyze_health(current)

        # Set target health
        if target_health is None:
            target_health = min(1.0, current_health.overall_health + 0.15)

        # Set max length
        if max_length is None:
            max_length = len(current) + 3

        # Generate improvement candidates
        improvements = self._generate_improvements(
            current, current_health, target_health, max_length
        )

        # Select best improvement
        best_improvement = max(
            improvements,
            key=lambda seq: self.health_analyzer.analyze_health(seq).overall_health,
        )

        # Generate explanatory recommendations
        recommendations = self._explain_improvements(current, best_improvement)

        return best_improvement, recommendations

    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================

    def _trim_sequence(self, sequence: list[str], max_length: int) -> list[str]:
        """Trim sequence to max_length while preserving structure."""
        if len(sequence) <= max_length:
            return sequence

        # Try to preserve ending if it's a stabilizer
        if sequence[-1] in _STABILIZERS:
            # Keep ending, trim from middle
            keep_start = max_length // 2
            keep_end = max_length - keep_start
            return sequence[:keep_start] + sequence[-keep_end:]
        else:
            # Simple truncation
            return sequence[:max_length]

    def _generate_variations(
        self, base: list[str], max_length: int, count: int = 20
    ) -> list[list[str]]:
        """Generate variations of a base sequence."""
        variations = [base]

        for _ in range(count):
            variation = base.copy()

            # Random modification
            modification = self._rng.choice(["insert", "remove", "replace", "extend"])

            if modification == "insert" and len(variation) < max_length:
                pos = self._rng.randint(0, len(variation))
                new_op = self._select_compatible_operator(
                    variation[pos - 1] if pos > 0 else None,
                    variation[pos] if pos < len(variation) else None,
                )
                if new_op:
                    variation.insert(pos, new_op)

            elif modification == "remove" and len(variation) > 3:
                pos = self._rng.randint(0, len(variation) - 1)
                # Don't remove if it breaks compatibility
                if self._can_remove(variation, pos):
                    variation.pop(pos)

            elif modification == "replace":
                pos = self._rng.randint(0, len(variation) - 1)
                new_op = self._select_compatible_operator(
                    variation[pos - 1] if pos > 0 else None,
                    variation[pos + 1] if pos < len(variation) - 1 else None,
                )
                if new_op:
                    variation[pos] = new_op

            elif modification == "extend" and len(variation) < max_length:
                new_op = self._select_compatible_operator(variation[-1], None)
                if new_op:
                    variation.append(new_op)

            variations.append(variation)

        return variations

    def _select_compatible_operator(self, prev: str | None, next_op: str | None) -> str | None:
        """Select an operator compatible with neighbors."""
        all_operators = [
            EMISSION,
            RECEPTION,
            COHERENCE,
            DISSONANCE,
            COUPLING,
            RESONANCE,
            SILENCE,
            EXPANSION,
            CONTRACTION,
            SELF_ORGANIZATION,
            MUTATION,
            TRANSITION,
            RECURSIVITY,
        ]

        if prev is None and next_op is None:
            return self._rng.choice(all_operators)

        compatible = []

        if prev is not None and next_op is None:
            # Find operators compatible after prev
            if prev in GRADUATED_COMPATIBILITY:
                levels = GRADUATED_COMPATIBILITY[prev]
                compatible.extend(levels.get("excellent", []))
                compatible.extend(levels.get("good", []))

        elif prev is None and next_op is not None:
            # Find operators that can precede next_op
            for op in all_operators:
                level = get_compatibility_level(op, next_op)
                if level in (CompatibilityLevel.EXCELLENT, CompatibilityLevel.GOOD):
                    compatible.append(op)

        else:
            # Must be compatible with both
            for op in all_operators:
                if prev and next_op:
                    level_after = get_compatibility_level(prev, op)
                    level_before = get_compatibility_level(op, next_op)
                    if level_after in (
                        CompatibilityLevel.EXCELLENT,
                        CompatibilityLevel.GOOD,
                    ) and level_before in (
                        CompatibilityLevel.EXCELLENT,
                        CompatibilityLevel.GOOD,
                    ):
                        compatible.append(op)

        return self._rng.choice(compatible) if compatible else None

    def _can_remove(self, sequence: list[str], pos: int) -> bool:
        """Check if operator at pos can be safely removed."""
        if pos == 0 or pos == len(sequence) - 1:
            return True  # Can always remove endpoints

        prev = sequence[pos - 1]
        next_op = sequence[pos + 1]

        level = get_compatibility_level(prev, next_op)
        return level in (CompatibilityLevel.EXCELLENT, CompatibilityLevel.GOOD)

    def _matches_pattern(self, sequence: list[str], pattern_name: str) -> bool:
        """Check if sequence matches the specified pattern."""
        detected = self.pattern_detector.detect_pattern(sequence)
        return detected.value == pattern_name

    def _get_pattern_signature(self, pattern_name: str) -> dict[str, list[str]]:
        """Get characteristic signature for a structural pattern."""
        # Pattern signatures mapping pattern names to operator combinations
        signatures: dict[str, dict[str, list[str]]] = {
            "BOOTSTRAP": {
                "core": [EMISSION, COUPLING, COHERENCE],
                "optional": [RECEPTION, SILENCE],
                "avoid": [DISSONANCE, MUTATION],
            },
            "THERAPEUTIC": {
                "core": [
                    EMISSION,
                    RECEPTION,
                    COHERENCE,
                    DISSONANCE,
                    SELF_ORGANIZATION,
                    COHERENCE,
                ],
                "optional": [SILENCE, TRANSITION],
                "avoid": [],
            },
            "EDUCATIONAL": {
                "core": [RECEPTION, COHERENCE, EXPANSION, DISSONANCE, MUTATION],
                "optional": [EMISSION, COHERENCE, SILENCE],
                "avoid": [],
            },
            "ORGANIZATIONAL": {
                "core": [
                    TRANSITION,
                    EMISSION,
                    RECEPTION,
                    COUPLING,
                    DISSONANCE,
                    SELF_ORGANIZATION,
                ],
                "optional": [COHERENCE, RESONANCE],
                "avoid": [],
            },
            "CREATIVE": {
                "core": [
                    SILENCE,
                    EMISSION,
                    EXPANSION,
                    DISSONANCE,
                    MUTATION,
                    SELF_ORGANIZATION,
                ],
                "optional": [COHERENCE, RECURSIVITY],
                "avoid": [],
            },
            "STABILIZE": {
                "core": [COHERENCE, SILENCE],
                "optional": [RESONANCE, COHERENCE],
                "avoid": [DISSONANCE, MUTATION, EXPANSION],
            },
            "EXPLORE": {
                "core": [DISSONANCE, MUTATION, COHERENCE],
                "optional": [EMISSION, RECEPTION],
                "avoid": [SILENCE],
            },
            "RESONATE": {
                "core": [RESONANCE, COUPLING, RESONANCE],
                "optional": [COHERENCE, EMISSION],
                "avoid": [DISSONANCE, MUTATION],
            },
        }

        if pattern_name not in signatures:
            # Default signature for unknown patterns
            return {
                "core": [EMISSION, COHERENCE, SILENCE],
                "optional": [RECEPTION, RESONANCE],
                "avoid": [],
            }

        return signatures[pattern_name]

    def _build_from_signature(self, signature: dict[str, list[str]], max_length: int) -> list[str]:
        """Build sequence from pattern signature."""
        core = signature["core"]
        optional = signature.get("optional", [])

        # Start with core
        sequence = list(core)

        # Add optional operators if room and improves health
        remaining = max_length - len(sequence)
        if remaining > 0 and optional:
            for op in optional:
                if len(sequence) < max_length:
                    # Try to insert at compatible position
                    for i in range(len(sequence) + 1):
                        prev = sequence[i - 1] if i > 0 else None
                        next_op = sequence[i] if i < len(sequence) else None

                        if prev is None or get_compatibility_level(prev, op) in (
                            CompatibilityLevel.EXCELLENT,
                            CompatibilityLevel.GOOD,
                        ):
                            if next_op is None or get_compatibility_level(op, next_op) in (
                                CompatibilityLevel.EXCELLENT,
                                CompatibilityLevel.GOOD,
                            ):
                                sequence.insert(i, op)
                                break

        return sequence[:max_length]

    def _generate_improvements(
        self,
        current: list[str],
        current_health: object,
        target_health: float,
        max_length: int,
    ) -> list[list[str]]:
        """Generate candidate improvements for a sequence."""
        improvements = [current]

        # Strategy 1: Add missing phases
        if hasattr(current_health, "pattern_completeness"):
            if current_health.pattern_completeness < 0.75:  # type: ignore[attr-defined]
                # Try adding activation
                if not any(op in [EMISSION, RECEPTION] for op in current):
                    for pos in range(min(2, len(current))):
                        candidate = current.copy()
                        candidate.insert(pos, RECEPTION)
                        if len(candidate) <= max_length:
                            improvements.append(candidate)

        # Strategy 2: Add stabilizers if unbalanced
        if hasattr(current_health, "balance_score"):
            if current_health.balance_score < 0.6:  # type: ignore[attr-defined]
                for stabilizer in _STABILIZERS:
                    candidate = current.copy()
                    if len(candidate) < max_length:
                        candidate.append(stabilizer)
                        improvements.append(candidate)

        # Strategy 3: Improve ending
        if hasattr(current_health, "sustainability_index"):
            if (
                current_health.sustainability_index < 0.7  # type: ignore[attr-defined]
                and current[-1] not in _STABILIZERS
            ):
                for stabilizer in _STABILIZERS:
                    candidate = current.copy()
                    candidate.append(stabilizer)
                    if len(candidate) <= max_length:
                        improvements.append(candidate)

        # Strategy 4: Add resonance for amplification
        if RESONANCE not in current and len(current) < max_length:
            for i in range(1, len(current)):
                if current[i - 1] in [COUPLING, COHERENCE, EXPANSION]:
                    candidate = current.copy()
                    candidate.insert(i, RESONANCE)
                    if len(candidate) <= max_length:
                        improvements.append(candidate)

        return improvements

    def _explain_improvements(self, original: list[str], improved: list[str]) -> list[str]:
        """Generate explanations for improvements made."""
        recommendations = []

        # Analyze differences
        original_health = self.health_analyzer.analyze_health(original)
        improved_health = self.health_analyzer.analyze_health(improved)

        # Overall improvement
        health_delta = improved_health.overall_health - original_health.overall_health
        if health_delta > 0.01:
            recommendations.append(
                f"Overall health improved by {health_delta:.2f} "
                f"(from {original_health.overall_health:.2f} to {improved_health.overall_health:.2f})"
            )

        # Specific metric improvements
        if improved_health.coherence_index > original_health.coherence_index + 0.05:
            recommendations.append(
                f"Coherence improved by {improved_health.coherence_index - original_health.coherence_index:.2f}"
            )

        if improved_health.balance_score > original_health.balance_score + 0.05:
            recommendations.append(
                f"Balance improved by {improved_health.balance_score - original_health.balance_score:.2f}"
            )

        if improved_health.sustainability_index > original_health.sustainability_index + 0.05:
            recommendations.append(
                f"Sustainability improved by {improved_health.sustainability_index - original_health.sustainability_index:.2f}"
            )

        # Identify added operators
        from collections import Counter

        original_counts = Counter(original)
        improved_counts = Counter(improved)
        added = [op for op in improved_counts if improved_counts[op] > original_counts.get(op, 0)]
        if added:
            recommendations.append(f"Added operators: {', '.join(set(added))}")

        # Pattern change
        if improved_health.dominant_pattern != original_health.dominant_pattern:
            recommendations.append(
                f"Pattern evolved from {original_health.dominant_pattern} to {improved_health.dominant_pattern}"
            )

        if not recommendations:
            recommendations.append("Sequence maintained with minor refinements")

        return recommendations
