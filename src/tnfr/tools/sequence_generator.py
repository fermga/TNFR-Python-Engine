"""Context-guided sequence generator for TNFR operator sequences.

Domain templates and pattern heuristics propose candidates. The shared word
validator admits every offered result before heuristic health ranks it. Names,
scores and accepted words establish no application effect, live phase gate or
future stability. Length and requested pattern constraints are never waived.

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
>>> print(seq.metadata["grammar_passed"])
True
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
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

from .._exact_time import bounded_represented_real
from ..config.operator_names import (
    CANONICAL_OPERATOR_NAMES,
    COHERENCE,
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
from ..operators.grammar_context import _normalize_sequence_context
from ..operators.grammar_patterns import validate_sequence
from ..operators.grammar_types import StructuralPattern
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

# Operator groups for intelligent variation.
# NOTE: these are GENERATION heuristics (which operators to try when proposing
# candidate sequences), NOT the canonical grammar-classification sets used by the
# validator.  The canonical U1-U6 sets live in tnfr.operators.grammar_types
# (single source of truth); generated candidates are validated against those.
# In particular _TRANSFORMERS here = regime-shifting generation group (NAV, REMESH,
# ZHIR), distinct from the canonical U4b TRANSFORMERS = {ZHIR, THOL}.
_STABILIZERS = [COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE]
_DESTABILIZERS = [DISSONANCE, MUTATION, EXPANSION]  # == canonical {OZ, ZHIR, VAL}
_TRANSFORMERS = [TRANSITION, RECURSIVITY, MUTATION]  # generation heuristic, not U4b

# ---------------------------------------------------------------------------
# Health improvement thresholds
# ---------------------------------------------------------------------------
_MIN_PATTERN_COMPLETENESS = 0.75
_MIN_BALANCE_SCORE = 0.6
_HEALTH_IMPROVEMENT_DELTA = 0.01


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

    Offered sequences pass the shared word contracts under the recorded
    context. Live phase/history admission, actual collective composition and
    outcome measurements remain separate obligations.

    Examples
    --------
    >>> generator = ContextualSequenceGenerator()
    >>> result = generator.generate_for_context(
    ...     domain="therapeutic",
    ...     objective="crisis_intervention"
    ... )
    >>> print(result.metadata["grammar_passed"])
    True
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
        *,
        grammar_context: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Rank admitted complete words proposed for a domain label.

        Templates are unvalidated proposals, including fragments. Every offered
        result passes the shared word validator under ``grammar_context``;
        omitted context uses core admission and no inherited form. Length and
        a requested pattern are hard constraints. The health threshold is a
        ranking preference: an admitted fallback reports ``constraint_met=False``.
        Registered recipe fallbacks disclose their origin and imply no domain
        effect. Missing admissible candidates raise ``ValueError``.
        """
        context, max_length, min_health = self._admit_request(
            grammar_context, max_length, min_health
        )
        pattern = self._normalize_pattern(required_pattern)
        if objective is None:
            from .domain_templates import list_objectives

            objective = list_objectives(domain)[0]
        base = self._trim_sequence(get_template(domain, objective), max_length)
        candidates = [(base, "template")]
        direct = self._admitted_candidates(candidates, context, max_length, pattern)
        if direct and direct[0][1].overall_health >= min_health:
            return self._result(
                direct[0],
                context,
                min_health,
                domain,
                objective,
                {"variations_tried": 0, "required_pattern": pattern},
            )
        variations = self._generate_variations(base, max_length, count=20)
        candidates.extend((word, "template_variant") for word in variations)
        candidates.extend(self._registered_candidates())
        admitted = self._admitted_candidates(candidates, context, max_length, pattern)
        selected = self._select_admitted(admitted)
        return self._result(
            selected,
            context,
            min_health,
            domain,
            objective,
            {"variations_tried": len(variations), "required_pattern": pattern},
        )

    def generate_for_pattern(
        self,
        target_pattern: str,
        max_length: int = 10,
        min_health: float = 0.70,
        *,
        grammar_context: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Offer a grammar-admitted word matching a recognized pattern label.

        Pattern detection remains a heuristic classification, not proof of a
        collective structure. Unknown labels and unavailable length/pattern
        combinations raise; no fallback can silently drop either constraint.
        """
        context, max_length, min_health = self._admit_request(
            grammar_context, max_length, min_health
        )
        pattern = self._normalize_pattern(target_pattern)
        if pattern is None:
            raise ValueError("target_pattern is required")
        signature = self._get_pattern_signature(pattern)
        base = self._build_from_signature(signature, max_length)
        core = signature["core"]
        proposals = [(base, "pattern_direct"), (core, "pattern_signature")]
        # Optional signature tokens propose alternatives rather than silently
        # displacing the requested pattern in a single greedy construction.
        proposals.extend(
            (core[:index] + [token] + core[index:], "pattern_signature")
            for token in signature.get("optional", ())
            for index in range(len(core) + 1)
        )
        direct = self._admitted_candidates(proposals, context, max_length, pattern)
        if direct and direct[0][1].overall_health >= min_health:
            return self._result(
                direct[0],
                context,
                min_health,
                None,
                None,
                {"target_pattern": pattern, "pattern_matched": True},
            )
        variations = self._generate_variations(base, max_length, count=15)
        candidates = proposals + [(word, "pattern_variant") for word in variations]
        candidates.extend(self._registered_candidates())
        selected = self._select_admitted(
            self._admitted_candidates(candidates, context, max_length, pattern)
        )
        return self._result(
            selected,
            context,
            min_health,
            None,
            None,
            {
                "target_pattern": pattern,
                "pattern_matched": True,
                "variations_tried": len(variations),
            },
        )

    def improve_sequence(
        self,
        current: list[str],
        target_health: float | None = None,
        max_length: int | None = None,
        *,
        grammar_context: Mapping[str, Any] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Rank local proposals without returning an inadmissible word.

        The return shape is retained. Recommendations record the actual word
        context and heuristic threshold status. No operator is executed, and a
        higher health score establishes no state, stability or physical result.
        """
        if not isinstance(current, (list, tuple)) or not current:
            raise ValueError("current must be a nonempty operator sequence")
        if any(not isinstance(token, str) for token in current):
            raise TypeError("current operator tokens must be strings")
        current = list(current)
        if max_length is None:
            max_length = len(current) + 3
        requested_health = target_health
        context, max_length, target_health = self._admit_request(
            grammar_context, max_length, 0.0 if target_health is None else target_health
        )
        current_health = self.health_analyzer.analyze_health(current)
        if requested_health is None:
            target_health = min(1.0, current_health.overall_health + 0.15)
        proposals = self._generate_improvements(
            current, current_health, target_health, max_length
        )
        best, health, _, _ = self._select_admitted(
            self._admitted_candidates(
                [(word, "improvement") for word in proposals],
                context,
                max_length,
                None,
            )
        )
        recommendations = self._explain_improvements(current, best)
        recommendations.append(
            f"Word admission context: {context!r}; live gates remain required"
        )
        if health.overall_health < target_health:
            recommendations.append(
                "Requested heuristic health threshold was not reached"
            )
        return list(best), recommendations

    @staticmethod
    def _admit_request(context, max_length, min_health):
        context = _normalize_sequence_context(context)
        if context.get("diagnostic", False):
            raise ValueError(
                "diagnostic waivers cannot admit a generated complete word"
            )
        context["initial_epi_nonzero"] = context.get("initial_epi_nonzero", False)
        context["diagnostic"] = False
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, Integral)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")
        health, _ = bounded_represented_real(
            min_health, "heuristic health threshold", minimum=0, maximum=1
        )
        return context, int(max_length), health

    @staticmethod
    def _normalize_pattern(pattern):
        if pattern is None:
            return None
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a recognized string label")
        try:
            return StructuralPattern(pattern.strip().lower()).name
        except ValueError as exc:
            raise ValueError(f"Unknown structural pattern: {pattern!r}") from exc

    @staticmethod
    def _registered_candidates():
        # Reuse the executable SDK catalog; do not copy a second fallback table.
        from ..sdk.study import list_sequences

        return [
            (item["operators"], f"registered_recipe:{item['name']}")
            for item in list_sequences()
        ]

    def _admitted_candidates(self, candidates, context, max_length, pattern):
        admitted, seen = [], set()
        for proposed, origin in candidates:
            word = tuple(proposed)
            if word in seen or not 0 < len(word) <= max_length:
                continue
            seen.add(word)
            outcome = validate_sequence(list(word), context=context)
            if not outcome.passed or not outcome.metadata.get(
                "canonical_word_checked", False
            ):
                continue
            if pattern is not None and not self._matches_pattern(list(word), pattern):
                continue
            admitted.append(
                (
                    list(word),
                    self.health_analyzer.analyze_health(list(word)),
                    origin,
                    outcome,
                )
            )
        return admitted

    @staticmethod
    def _select_admitted(admitted):
        if not admitted:
            raise ValueError(
                "No complete word satisfies grammar, length and pattern constraints"
            )
        return max(admitted, key=lambda item: item[1].overall_health)

    @staticmethod
    def _result(selected, context, minimum, domain, objective, metadata):
        word, health, origin, outcome = selected
        recommendations = list(health.recommendations)
        met = health.overall_health >= minimum
        if not met:
            recommendations.append(
                "Requested heuristic health threshold was not reached"
            )
        if origin.startswith("registered_recipe:"):
            recommendations.append(
                f"Registered recipe fallback selected: {origin.split(':', 1)[1]}"
            )
        return GenerationResult(
            sequence=list(word),
            health_score=health.overall_health,
            detected_pattern=health.dominant_pattern,
            domain=domain,
            objective=objective,
            method=origin,
            recommendations=recommendations,
            metadata={
                **metadata,
                "constraint_met": met,
                "candidate_origin": origin,
                "grammar_context": deepcopy(context),
                "compatibility_profile": outcome.metadata["compatibility_profile"],
                "canonical_word_checked": True,
                "grammar_passed": outcome.passed,
                "grammar_message": outcome.message,
                "scope": "word_admission_only; no_live_state_or_domain_effect_certification",
            },
        )

    # =========================================================================
    # Candidate proposal heuristics; shared admission owns every offered result.
    # =========================================================================

    def _trim_sequence(self, sequence: list[str], max_length: int) -> list[str]:
        """Propose a length-limited fragment; later word admission is mandatory."""
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

    def _select_compatible_operator(
        self, prev: str | None, next_op: str | None
    ) -> str | None:
        """Select an operator compatible with neighbors."""
        all_operators = sorted(CANONICAL_OPERATOR_NAMES)

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
        return detected.value == pattern_name.lower()

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

    def _build_from_signature(
        self, signature: dict[str, list[str]], max_length: int
    ) -> list[str]:
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
                            if next_op is None or get_compatibility_level(
                                op, next_op
                            ) in (
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
            if current_health.pattern_completeness < _MIN_PATTERN_COMPLETENESS:  # type: ignore[attr-defined]
                # Try adding activation
                if not any(op in [EMISSION, RECEPTION] for op in current):
                    for pos in range(min(2, len(current))):
                        candidate = current.copy()
                        candidate.insert(pos, RECEPTION)
                        if len(candidate) <= max_length:
                            improvements.append(candidate)

        # Strategy 2: Add stabilizers if unbalanced
        if hasattr(current_health, "balance_score"):
            if current_health.balance_score < _MIN_BALANCE_SCORE:  # type: ignore[attr-defined]
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

    def _explain_improvements(
        self, original: list[str], improved: list[str]
    ) -> list[str]:
        """Generate explanations for improvements made."""
        recommendations = []

        # Analyze differences
        original_health = self.health_analyzer.analyze_health(original)
        improved_health = self.health_analyzer.analyze_health(improved)

        # Overall improvement
        health_delta = improved_health.overall_health - original_health.overall_health
        if health_delta > _HEALTH_IMPROVEMENT_DELTA:
            recommendations.append(
                f"Overall health improved by {health_delta:.2f} "
                f"(from {original_health.overall_health:.2f} to {improved_health.overall_health:.2f})"
            )

        # Specific metric improvements
        if (
            improved_health.flow_quality_score
            > original_health.flow_quality_score + 0.05
        ):
            flow_delta = (
                improved_health.flow_quality_score - original_health.flow_quality_score
            )
            recommendations.append(f"Flow quality improved by {flow_delta:.2f}")

        if improved_health.balance_score > original_health.balance_score + 0.05:
            recommendations.append(
                f"Balance improved by {improved_health.balance_score - original_health.balance_score:.2f}"
            )

        if (
            improved_health.sustainability_index
            > original_health.sustainability_index + 0.05
        ):
            recommendations.append(
                f"Sustainability improved by {improved_health.sustainability_index - original_health.sustainability_index:.2f}"
            )

        # Identify added operators
        from collections import Counter

        original_counts = Counter(original)
        improved_counts = Counter(improved)
        added = [
            op
            for op in improved_counts
            if improved_counts[op] > original_counts.get(op, 0)
        ]
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
