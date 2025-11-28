"""Advanced structural pattern detection heuristics.

This module provides a lightweight, heuristic implementation of the
``AdvancedPatternDetector`` that the test-suite and higher level APIs expect.
The detector recognises key structural motifs referenced across the project
(therapeutic, educational, bootstrap, etc.) and supplies compact sequence
analytics used by documentation tooling and SDK helpers.

The intent is not to be an exhaustive physics model – the canonical grammar
remains the single source of truth – but to offer a reproducible mapping from
operator sequences to well-known structural archetypes. All heuristics remain
traceable to TNFR grammar principles:

* Domain patterns blend U1–U4 rule signatures (e.g. therapeutic sequences
    combine reception, self-organisation and closure stabilisers).
* Meta patterns such as ``bootstrap`` and ``explore`` capture short pulses that
    tooling surfaces during guidance flows.
* Composition analysis reports stabiliser/destabiliser balance and highlights
    sub-pattern components so downstream code can provide actionable feedback.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

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
from .grammar import StructuralPattern

__all__ = ["AdvancedPatternDetector"]


_CANONICAL_ORDER = (
    EMISSION,
    RECEPTION,
    COHERENCE,
    RESONANCE,
    SILENCE,
    DISSONANCE,
    SELF_ORGANIZATION,
    MUTATION,
    TRANSITION,
    COUPLING,
    RECURSIVITY,
    EXPANSION,
    CONTRACTION,
)

_STABILIZERS = {COHERENCE, SELF_ORGANIZATION}
_DESTABILIZERS = {DISSONANCE, MUTATION, EXPANSION}
_INTERMEDIATE = {COUPLING, RESONANCE, DISSONANCE}

_COHERENCE_WEIGHTS = {
    StructuralPattern.THERAPEUTIC: 1.35,
    StructuralPattern.EDUCATIONAL: 1.25,
    StructuralPattern.ORGANIZATIONAL: 1.2,
    StructuralPattern.CREATIVE: 1.2,
    StructuralPattern.REGENERATIVE: 1.3,
    StructuralPattern.BOOTSTRAP: 1.05,
    StructuralPattern.EXPLORE: 1.1,
    StructuralPattern.STABILIZE: 1.15,
    StructuralPattern.BIFURCATED: 1.05,
    StructuralPattern.FRACTAL: 1.1,
    StructuralPattern.HIERARCHICAL: 1.05,
    StructuralPattern.CYCLIC: 1.0,
    StructuralPattern.COMPLEX: 1.15,
    StructuralPattern.COMPRESS: 0.95,
    StructuralPattern.RESONATE: 1.05,
    StructuralPattern.LINEAR: 0.9,
    StructuralPattern.BASIC_LEARNING: 1.0,
    StructuralPattern.DEEP_LEARNING: 1.1,
    StructuralPattern.EXPLORATORY_LEARNING: 1.1,
    StructuralPattern.CONSOLIDATION_CYCLE: 0.95,
    StructuralPattern.ADAPTIVE_MUTATION: 1.0,
    StructuralPattern.UNKNOWN: 0.5,
}


def _canonicalise(sequence: Sequence[str]) -> List[str]:
    """Return canonical lower-case operator tokens."""

    return [str(token).lower() for token in sequence]


class AdvancedPatternDetector:
    """Heuristic detector for high-level structural patterns.

    The detector prefers domain/metabolic patterns over baseline structural
    classifications so that the rich diagnostic stories remain available while
    still falling back to generic labels (``LINEAR``, ``FRACTAL`` …) when the
    sequence does not trigger a specialised signature.
    """

    def __init__(self) -> None:  # pragma: no cover - trivial initialiser
        self._cache: Dict[tuple[str, ...], StructuralPattern] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_pattern(self, sequence: Sequence[str]) -> StructuralPattern:
        canonical = tuple(_canonicalise(sequence))
        if not canonical:
            return StructuralPattern.UNKNOWN

        cached = self._cache.get(canonical)
        if cached is not None:
            return cached

        # TNFR Physics Priority: Domain patterns have priority over structural
        # patterns to capture rich diagnostic information (therapeutic, etc.)
        pattern = (
            self._detect_domain_pattern(canonical)
            or self._detect_learning_pattern(canonical)
            or self._detect_meta_pattern(canonical)
            or self._detect_structural_pattern(canonical)
            or StructuralPattern.UNKNOWN
        )
        self._cache[canonical] = pattern
        return pattern

    def analyze_sequence_composition(
        self, sequence: Sequence[str]
    ) -> Mapping[str, object]:
        canonical = _canonicalise(sequence)
        pattern = self.detect_pattern(canonical)
        components = self._identify_components(canonical)
        complexity_score = self._complexity_score(canonical)
        suitability = self._domain_suitability(canonical)
        health = self._structural_health(canonical)
        pattern_scores = self._pattern_scores(canonical, pattern)
        coherence_weights = self._coherence_weights()
        weighted_scores = {
            name: round(
                pattern_scores[name] * coherence_weights.get(name, 1.0),
                4,
            )
            for name in pattern_scores
        }

        return {
            "sequence": tuple(canonical),
            "primary_pattern": pattern.value,
            "pattern_scores": pattern_scores,
            "weighted_scores": weighted_scores,
            "coherence_weights": coherence_weights,
            "components": components,
            "complexity_score": complexity_score,
            "domain_suitability": suitability,
            "structural_health": health,
        }

    # ------------------------------------------------------------------
    # Domain-specific detection (highest priority)
    # ------------------------------------------------------------------

    def _detect_domain_pattern(
        self, seq: Sequence[str]
    ) -> StructuralPattern | None:
        # Defer to STABILIZE only for short (<7) or emission-led closures so
        # longer reception-led therapeutic sequences still classify correctly.
        if self._is_stabilize(seq) and (
            len(seq) <= 6 or (seq and seq[0] == EMISSION)
        ):
            return None
        if self._is_therapeutic(seq):
            return StructuralPattern.THERAPEUTIC
        # Prefer CREATIVE over EDUCATIONAL when both could match
        if self._is_creative(seq):
            return StructuralPattern.CREATIVE
        if self._is_educational(seq):
            return StructuralPattern.EDUCATIONAL
        if self._is_organizational(seq):
            return StructuralPattern.ORGANIZATIONAL
        if self._is_regenerative(seq):
            return StructuralPattern.REGENERATIVE
        return None

    def _detect_learning_pattern(
        self, seq: Sequence[str]
    ) -> StructuralPattern | None:
        # Do not classify as BASIC/DEEP/EXPLORATORY learning when explicit
        # stabilization closure (IL→{SHA|RA}) is present; prefer STABILIZE.
        if self._is_basic_learning(seq):
            return StructuralPattern.BASIC_LEARNING
        if self._is_deep_learning(seq):
            return StructuralPattern.DEEP_LEARNING
        if self._is_exploratory_learning(seq):
            return StructuralPattern.EXPLORATORY_LEARNING
        if self._is_consolidation_cycle(seq):
            return StructuralPattern.CONSOLIDATION_CYCLE
        if self._is_adaptive_mutation(seq):
            return StructuralPattern.ADAPTIVE_MUTATION
        return None

    def _detect_meta_pattern(
        self, seq: Sequence[str]
    ) -> StructuralPattern | None:
        if self._is_bootstrap(seq):
            return StructuralPattern.BOOTSTRAP
        if self._is_stabilize(seq):
            return StructuralPattern.STABILIZE
        # If a strong structural signature like FRACTAL is present (e.g.,
        # RECURSIVITY) in a longer sequence, prefer structural detection over
        # generic EXPLORE labeling.
        if self._is_explore(seq) and not self._is_fractal(seq):
            return StructuralPattern.EXPLORE
        return None

    def _detect_structural_pattern(
        self, seq: Sequence[str]
    ) -> StructuralPattern | None:
        if self._is_bifurcated(seq):
            return StructuralPattern.BIFURCATED
        if self._is_fractal(seq):
            return StructuralPattern.FRACTAL
        if self._is_hierarchical(seq):
            return StructuralPattern.HIERARCHICAL
        if self._is_cyclic(seq):
            return StructuralPattern.CYCLIC
        if self._is_complex(seq):
            return StructuralPattern.COMPLEX
        if self._is_compress(seq):
            return StructuralPattern.COMPRESS
        if self._is_resonate(seq):
            return StructuralPattern.RESONATE
        if self._is_linear(seq):
            return StructuralPattern.LINEAR
        return None

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count(sequence: Sequence[str], members: Iterable[str]) -> int:
        member_set = set(members)
        return sum(1 for token in sequence if token in member_set)

    @staticmethod
    def _contains(sequence: Sequence[str], *tokens: str) -> bool:
        view = set(sequence)
        return all(token in view for token in tokens)

    @staticmethod
    def _pairwise(sequence: Sequence[str]) -> Iterable[tuple[str, str]]:
        for i in range(len(sequence) - 1):
            yield sequence[i], sequence[i + 1]

    # Domain pattern heuristics

    def _is_therapeutic(self, seq: Sequence[str]) -> bool:
        # Therapeutic: EN + AL + IL + THOL, with OZ pulse and stable ending
        has_core = self._contains(
            seq, RECEPTION, EMISSION, COHERENCE, SELF_ORGANIZATION
        )
        has_dissonance = DISSONANCE in seq
        has_closure = bool(seq) and seq[-1] in {SILENCE, COHERENCE, TRANSITION}
        return has_core and has_dissonance and has_closure

    def _is_educational(self, seq: Sequence[str]) -> bool:
        if not self._contains(seq, EXPANSION, DISSONANCE, MUTATION):
            return False
        # Require ordered progression: VAL → OZ → ZHIR
        try:
            i_val = seq.index(EXPANSION)
            i_oz = seq.index(DISSONANCE)
            i_zhir = seq.index(MUTATION)
        except ValueError:
            return False
        return i_val < i_oz < i_zhir

    def _is_organizational(self, seq: Sequence[str]) -> bool:
        return self._contains(
            seq,
            TRANSITION,
            COUPLING,
            RESONANCE,
            SELF_ORGANIZATION,
            RECURSIVITY,
        )

    def _is_creative(self, seq: Sequence[str]) -> bool:
        return (
            SILENCE in seq
            and self._contains(
                seq,
                EXPANSION,
                MUTATION,
                SELF_ORGANIZATION,
                RESONANCE,
            )
        )

    def _is_regenerative(self, seq: Sequence[str]) -> bool:
        required = {
            COHERENCE,
            RESONANCE,
            EXPANSION,
            SILENCE,
            TRANSITION,
            EMISSION,
            RECEPTION,
            COUPLING,
        }
        return required.issubset(seq)

    # Learning pattern heuristics

    def _is_basic_learning(self, seq: Sequence[str]) -> bool:
        if tuple(seq) != (EMISSION, RECEPTION, COHERENCE, SILENCE):
            return False
        # If also matching stabilization closure semantics, prefer STABILIZE.
        if self._is_stabilize(seq):
            return False
        return True

    def _is_deep_learning(self, seq: Sequence[str]) -> bool:
        # Deep learning: comprehensive learning with substantive sequence
        has_core = self._contains(
            seq,
            EMISSION,
            RECEPTION,
            DISSONANCE,
            SELF_ORGANIZATION,
            COHERENCE,
        )
        # Require longer sequence to distinguish from basic hierarchical
        return has_core and len(seq) >= 9

    def _is_exploratory_learning(self, seq: Sequence[str]) -> bool:
        return self._contains(
            seq,
            DISSONANCE,
            SELF_ORGANIZATION,
            RESONANCE,
            COHERENCE,
        )

    def _is_consolidation_cycle(self, seq: Sequence[str]) -> bool:
        return tuple(seq[-2:]) == (COHERENCE, RECURSIVITY)

    def _is_adaptive_mutation(self, seq: Sequence[str]) -> bool:
        has_core = self._contains(seq, DISSONANCE, MUTATION)
        requires_handler = SELF_ORGANIZATION in seq
        return has_core and requires_handler and seq[-1] == TRANSITION

    # Meta-pattern heuristics

    def _is_bootstrap(self, seq: Sequence[str]) -> bool:
        return (
            tuple(seq[:3]) == (EMISSION, COUPLING, COHERENCE)
            and len(seq) <= 5
        )

    def _is_explore(self, seq: Sequence[str]) -> bool:
        # Explore: at least two destabilizers without THOL dominance
        has_self_org = SELF_ORGANIZATION in seq
        destabilizer_count = self._count(
            seq, {DISSONANCE, EXPANSION, MUTATION}
        )
        if has_self_org:
            return False
        if self._is_simple_bifurcation(seq):
            return False
        return destabilizer_count >= 2

    def _is_simple_bifurcation(self, seq: Sequence[str]) -> bool:
        """Check if sequence is a simple bifurcation pattern."""
        # Simple bifurcation: OZ with {ZHIR|NUL} and limited complexity
        has_trigger = DISSONANCE in seq and (
            MUTATION in seq or CONTRACTION in seq
        )
        return has_trigger and EXPANSION not in seq and len(seq) <= 7

    def _is_stabilize(self, seq: Sequence[str]) -> bool:
        # Stabilize: IL then closure (IL→SHA|RA), short seq (<=6), no OZ+ZHIR
        if len(seq) >= 2 and tuple(seq[-2:]) in {
            (COHERENCE, SILENCE),
            (COHERENCE, RESONANCE),
        }:
            if DISSONANCE in seq and MUTATION in seq:
                return False
            # Allow slightly longer sequences (<=7) to count as stabilize
            # when they present a single destabilizer but end coherently.
            return len(seq) <= 7
        return False

    # Structural heuristics

    def _is_bifurcated(self, seq: Sequence[str]) -> bool:
        # Bifurcation: OZ→{ZHIR|NUL} but not hierarchical (THOL primary)
        has_bifurcation = DISSONANCE in seq and (
            MUTATION in seq or CONTRACTION in seq
        )
        if not has_bifurcation:
            return False
        if SELF_ORGANIZATION in seq:
            return False
        # Require adjacency for simple bifurcated classification
        for a, b in self._pairwise(seq):
            if a == DISSONANCE and b in {MUTATION, CONTRACTION}:
                return True
        return False

    def _is_fractal(self, seq: Sequence[str]) -> bool:
        return RECURSIVITY in seq or (TRANSITION in seq and COUPLING in seq)

    def _is_hierarchical(self, seq: Sequence[str]) -> bool:
        return SELF_ORGANIZATION in seq

    def _is_cyclic(self, seq: Sequence[str]) -> bool:
        silence_cycle = (
            SILENCE in seq
            and EMISSION in seq
            and seq.index(SILENCE) < len(seq) - 1
        )
        nav_cycle = seq.count(TRANSITION) >= 2
        return silence_cycle or nav_cycle

    def _is_complex(self, seq: Sequence[str]) -> bool:
        unique = len(set(seq))
        return len(seq) >= 6 and unique >= 5

    def _is_compress(self, seq: Sequence[str]) -> bool:
        return CONTRACTION in seq

    def _is_resonate(self, seq: Sequence[str]) -> bool:
        return RESONANCE in seq and seq.count(RESONANCE) >= 2

    def _is_linear(self, seq: Sequence[str]) -> bool:
        allowed = {
            EMISSION,
            RECEPTION,
            COHERENCE,
            RESONANCE,
            SILENCE,
            TRANSITION,
        }
        return all(token in allowed for token in seq)

    # Composition helpers

    def _identify_components(self, seq: Sequence[str]) -> set[str]:
        components: set[str] = set()
    # Identify bootstrap as a component when prefix matches
        if len(seq) >= 3 and tuple(seq[:3]) == (EMISSION, COUPLING, COHERENCE):
            components.add("bootstrap")
        if self._is_explore(seq):
            components.add("explore")
        else:
            # Also identify contiguous OZ→ZHIR→IL as an explore component
            for a, b in self._pairwise(seq):
                pass
            for idx in range(len(seq) - 2):
                if (
                    seq[idx] == DISSONANCE
                    and seq[idx + 1] == MUTATION
                    and seq[idx + 2] == COHERENCE
                ):
                    components.add("explore")
                    break
        # Recognise a stabilization component whenever IL is immediately
        # followed by {SHA|RA} anywhere in the sequence, regardless of
        # overall length or presence of destabilizers.
        for a, b in self._pairwise(seq):
            if a == COHERENCE and b in {SILENCE, RESONANCE}:
                components.add("stabilize")
                break
        if self._contains(seq, RECURSIVITY):
            components.add("fractal")
        if self._contains(seq, SELF_ORGANIZATION):
            components.add("hierarchical")
        if self._contains(seq, RESONANCE):
            components.add("resonance")
        return components

    def _complexity_score(self, seq: Sequence[str]) -> float:
        if not seq:
            return 0.0
        unique = len(set(seq))
        transitions = sum(
            1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]
        )
        stabilisers = self._count(seq, _STABILIZERS)
        destabilisers = self._count(seq, _DESTABILIZERS)
        raw_score = len(seq) + 0.8 * unique + 0.5 * transitions
        raw_score += 0.3 * destabilisers + 0.2 * stabilisers
        return min(1.0, raw_score / 12.0)

    def _domain_suitability(self, seq: Sequence[str]) -> Dict[str, float]:
        scores = {
            "therapeutic": 0.0,
            "educational": 0.0,
            "organizational": 0.0,
            "creative": 0.0,
            "regenerative": 0.0,
        }
        if self._is_therapeutic(seq):
            scores["therapeutic"] = 0.8
        if self._is_educational(seq):
            scores["educational"] = 0.75
        if self._contains(seq, EXPANSION) and SELF_ORGANIZATION in seq:
            scores["creative"] = max(scores["creative"], 0.6)
        if self._is_organizational(seq):
            scores["organizational"] = 0.7
        if self._is_regenerative(seq):
            scores["regenerative"] = 0.9
        if DISSONANCE in seq and COHERENCE in seq:
            scores["therapeutic"] = max(scores["therapeutic"], 0.55)
        if MUTATION in seq:
            scores["educational"] = max(scores["educational"], 0.45)
        if RECURSIVITY in seq:
            scores["organizational"] = max(scores["organizational"], 0.4)
        return scores

    def _structural_health(self, seq: Sequence[str]) -> Dict[str, object]:
        counter = Counter(seq)
        stabilisers = self._count(seq, _STABILIZERS)
        destabilisers = self._count(seq, _DESTABILIZERS)
        balance = stabilisers - destabilisers
        has_closure = bool(seq) and seq[-1] in {
            SILENCE,
            TRANSITION,
            RECURSIVITY,
            DISSONANCE,
        }
        return {
            "stabilizer_count": stabilisers,
            "destabilizer_count": destabilisers,
            "balance": balance,
            "has_closure": has_closure,
            "frequency": {
                token: counter[token]
                for token in _CANONICAL_ORDER
                if counter[token] > 0
            },
        }

    def _pattern_scores(
        self,
        seq: Sequence[str],
        primary: StructuralPattern,
    ) -> Dict[str, float]:
        def assign(pattern: StructuralPattern, value: float) -> None:
            if value <= 0.0:
                return
            key = pattern.value
            current = scores.get(key, 0.0)
            scores[key] = round(max(current, value), 4)

        scores: Dict[str, float] = {}

        # Domain patterns carry highest confidence when matched
        if self._is_therapeutic(seq):
            assign(StructuralPattern.THERAPEUTIC, 0.9)
        if self._is_educational(seq):
            assign(StructuralPattern.EDUCATIONAL, 0.85)
        if self._is_organizational(seq):
            assign(StructuralPattern.ORGANIZATIONAL, 0.8)
        if self._is_creative(seq):
            assign(StructuralPattern.CREATIVE, 0.8)
        if self._is_regenerative(seq):
            assign(StructuralPattern.REGENERATIVE, 0.9)

        # Learning strata
        if self._is_basic_learning(seq):
            assign(StructuralPattern.BASIC_LEARNING, 0.7)
        if self._is_deep_learning(seq):
            assign(StructuralPattern.DEEP_LEARNING, 0.8)
        if self._is_exploratory_learning(seq):
            assign(StructuralPattern.EXPLORATORY_LEARNING, 0.75)
        if self._is_consolidation_cycle(seq):
            assign(StructuralPattern.CONSOLIDATION_CYCLE, 0.6)
        if self._is_adaptive_mutation(seq):
            assign(StructuralPattern.ADAPTIVE_MUTATION, 0.7)

        # Meta and structural patterns
        if self._is_bootstrap(seq):
            assign(StructuralPattern.BOOTSTRAP, 0.7)
        if self._is_explore(seq):
            assign(StructuralPattern.EXPLORE, 0.65)
        if self._is_stabilize(seq):
            assign(StructuralPattern.STABILIZE, 0.7)
        if self._is_bifurcated(seq):
            assign(StructuralPattern.BIFURCATED, 0.6)
        if self._is_fractal(seq):
            assign(StructuralPattern.FRACTAL, 0.65)
        if self._is_hierarchical(seq):
            assign(StructuralPattern.HIERARCHICAL, 0.6)
        if self._is_cyclic(seq):
            assign(StructuralPattern.CYCLIC, 0.55)
        if self._is_complex(seq):
            assign(StructuralPattern.COMPLEX, 0.6)
        if self._is_compress(seq):
            assign(StructuralPattern.COMPRESS, 0.5)
        if self._is_resonate(seq):
            assign(StructuralPattern.RESONATE, 0.55)
        if self._is_linear(seq):
            assign(StructuralPattern.LINEAR, 0.5)

        # Ensure the detected primary pattern is represented
        if primary is not StructuralPattern.UNKNOWN:
            baseline = 0.75 if primary in {
                StructuralPattern.THERAPEUTIC,
                StructuralPattern.REGENERATIVE,
                StructuralPattern.EDUCATIONAL,
            } else 0.6
            assign(primary, baseline)
        elif not scores:
            scores[StructuralPattern.UNKNOWN.value] = 0.2

        return scores

    def _coherence_weights(self) -> Dict[str, float]:
        return {
            pattern.value: _COHERENCE_WEIGHTS.get(pattern, 1.0)
            for pattern in StructuralPattern
        }
