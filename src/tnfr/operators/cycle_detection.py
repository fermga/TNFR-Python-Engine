"""Cycle detection and validation for regenerative TNFR sequences.

This module implements R5_REGENERATIVE_CYCLES validation, ensuring that
regenerative cycles are structurally valid and self-sustaining according
to TNFR canonical principles.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    pass

from ..compat.dataclass import dataclass
from ..config.operator_names import (
    COHERENCE,
    COUPLING,
    EMISSION,
    RECEPTION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
)

# Define stabilizers using canonical operator names (not glyph codes)
# These operators provide structural stability in TNFR sequences.
# Note: This uses canonical names (coherence, resonance, etc.) rather than
# glyph codes (IL, RA, etc.) from config.constants.STABILIZERS to match
# the sequence validation format throughout the system.
_STABILIZERS_SET = frozenset([COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE, COUPLING])

__all__ = [
    "REGENERATORS",
    "MIN_CYCLE_LENGTH",
    "MAX_CYCLE_LENGTH",
    "CycleType",
    "CycleAnalysis",
    "CycleDetector",
]

# Regenerators: operators that enable structural renewal and regeneration
REGENERATORS = [TRANSITION, RECURSIVITY, SILENCE]  # NAV, REMESH, SHA

# Cycle length constraints
MIN_CYCLE_LENGTH = 5  # Minimum operators for meaningful cyclic behavior
MAX_CYCLE_LENGTH = 13  # Maximum = all canonical operators once


class CycleType(Enum):
    """Types of regenerative cycles based on dominant regenerator."""

    LINEAR = "linear"  # Traditional non-cyclic sequence
    REGENERATIVE = "regenerative"  # Cycle with regenerators
    RECURSIVE = "recursive"  # REMESH-driven (fractal regeneration)
    MEDITATIVE = "meditative"  # SHA-driven (paused renewal)
    TRANSFORMATIVE = "transformative"  # NAV-driven (phase transition)


@dataclass(slots=True)
class CycleAnalysis:
    """Results of regenerative cycle analysis."""

    is_valid_regenerative: bool
    reason: str = ""
    cycle_type: CycleType = CycleType.LINEAR
    health_score: float = 0.0
    regenerator_position: int = -1
    stabilizer_count_before: int = 0
    stabilizer_count_after: int = 0
    balance_score: float = 0.0
    diversity_score: float = 0.0
    coherence_score: float = 0.0


class CycleDetector:
    """Detects and validates regenerative cycles in TNFR sequences.

    Implements R5_REGENERATIVE_CYCLES validation rules:
    - Cycles must have minimum length (MIN_CYCLE_LENGTH)
    - Must include stabilizers before AND after regenerator
    - Must achieve minimum structural health score (>0.6)
    - Validates balance, diversity, and coherence

    Note: Uses _STABILIZERS_SET with canonical operator names to match
    the sequence validation format. Reuses pattern detector methods for
    balance, diversity, and health calculations to avoid code duplication.
    """

    # Minimum health score for valid regenerative cycle
    MIN_HEALTH_SCORE = 0.6

    def analyze_potential_cycle(
        self, sequence: Sequence[str], regenerator_index: int
    ) -> CycleAnalysis:
        """Analyze if a regenerator creates a valid regenerative cycle.

        Parameters
        ----------
        sequence : Sequence[str]
            Complete operator sequence (canonical names).
        regenerator_index : int
            Position of the regenerator operator.

        Returns
        -------
        CycleAnalysis
            Detailed analysis of cycle validity and characteristics.
        """
        # 1. Check minimum length
        if len(sequence) < MIN_CYCLE_LENGTH:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="too_short",
                cycle_type=CycleType.LINEAR,
            )

        # 2. Check maximum length
        if len(sequence) > MAX_CYCLE_LENGTH:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="too_long",
                cycle_type=CycleType.LINEAR,
            )

        # 3. Verify regenerator is valid
        if regenerator_index < 0 or regenerator_index >= len(sequence):
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="invalid_regenerator_position",
                cycle_type=CycleType.LINEAR,
            )

        regenerator = sequence[regenerator_index]
        if regenerator not in REGENERATORS:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="not_a_regenerator",
                cycle_type=CycleType.LINEAR,
            )

        # 4. Check stabilizers before and after regenerator
        before_segment = sequence[:regenerator_index]
        after_segment = sequence[regenerator_index + 1 :]

        stabilizers_before = self._count_stabilizers(before_segment)
        stabilizers_after = self._count_stabilizers(after_segment)

        if stabilizers_before == 0 or stabilizers_after == 0:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="no_stabilization",
                cycle_type=CycleType.LINEAR,
                stabilizer_count_before=stabilizers_before,
                stabilizer_count_after=stabilizers_after,
            )

        # 5. Calculate structural health
        balance = self._calculate_balance(sequence)
        diversity = self._calculate_diversity(sequence)
        coherence = self._calculate_sequence_coherence(sequence)

        health_score = (balance + diversity + coherence) / 3.0

        # 6. Determine cycle type
        cycle_type = self._determine_cycle_type(regenerator)

        # 7. Validate health threshold
        is_valid = health_score >= self.MIN_HEALTH_SCORE

        return CycleAnalysis(
            is_valid_regenerative=is_valid,
            reason="valid" if is_valid else "low_health_score",
            cycle_type=cycle_type,
            health_score=health_score,
            regenerator_position=regenerator_index,
            stabilizer_count_before=stabilizers_before,
            stabilizer_count_after=stabilizers_after,
            balance_score=balance,
            diversity_score=diversity,
            coherence_score=coherence,
        )

    def analyze_full_cycle(self, sequence: Sequence[str]) -> CycleAnalysis:
        """Analyze complete sequence for regenerative cycle properties.

        Searches for regenerators in the sequence and validates the
        strongest regenerative cycle found.

        Parameters
        ----------
        sequence : Sequence[str]
            Complete operator sequence (canonical names).

        Returns
        -------
        CycleAnalysis
            Analysis of the best regenerative cycle found, or
            indication that sequence is not regenerative.
        """
        if len(sequence) < MIN_CYCLE_LENGTH:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="too_short",
                cycle_type=CycleType.LINEAR,
            )

        # Find all regenerators in sequence
        regenerator_positions = [i for i, op in enumerate(sequence) if op in REGENERATORS]

        if not regenerator_positions:
            return CycleAnalysis(
                is_valid_regenerative=False,
                reason="no_regenerator",
                cycle_type=CycleType.LINEAR,
            )

        # Analyze each regenerator position and keep best result
        best_analysis = None
        best_health = -1.0

        for pos in regenerator_positions:
            analysis = self.analyze_potential_cycle(sequence, pos)
            if analysis.health_score > best_health:
                best_health = analysis.health_score
                best_analysis = analysis

        return best_analysis or CycleAnalysis(
            is_valid_regenerative=False,
            reason="no_valid_cycle",
            cycle_type=CycleType.LINEAR,
        )

    def _count_stabilizers(self, segment: Sequence[str]) -> int:
        """Count stabilizing operators in a sequence segment.

        Uses canonical operator names: coherence, self_organization, silence, resonance, coupling.
        """
        return sum(1 for op in segment if op in _STABILIZERS_SET)

    def _calculate_balance(self, sequence: Sequence[str]) -> float:
        """Calculate structural balance score (0.0-1.0).

        Reuses the existing pattern detector's health calculation approach
        but adapted for cycle-specific validation. Balance measures equilibrium
        between stabilizing operators.
        """
        from .patterns import AdvancedPatternDetector

        if not sequence:
            return 0.0

        # Use existing pattern detector for consistent health metrics
        detector = AdvancedPatternDetector()
        health_metrics = detector._structural_health(sequence)

        # Adapt balance calculation for cycle validation
        # Cycles need good balance (not too much stabilization, not too chaotic)
        balance_raw = health_metrics.get("balance", 0.0)

        # Normalize: optimal balance is around 0.2-0.4 (slightly more stabilizers)
        # Convert to 0-1 score where 0.3 is optimal
        if -0.1 <= balance_raw <= 0.5:
            # Good range
            score = 1.0 - abs(balance_raw - 0.3) * 1.5
        else:
            # Outside good range
            score = max(0.0, 0.5 - abs(balance_raw - 0.3) * 0.5)

        return max(0.0, min(1.0, score))

    def _calculate_diversity(self, sequence: Sequence[str]) -> float:
        """Calculate operator diversity score (0.0-1.0).

        Reuses complexity calculation from AdvancedPatternDetector which
        includes diversity as a component.
        """
        if not sequence:
            return 0.0

        # For cycles, we primarily care about diversity component
        unique_count = len(set(sequence))
        total_count = len(sequence)
        diversity_ratio = unique_count / total_count

        # Bonus for using many unique operators (> 5)
        if unique_count >= 5:
            bonus = min(0.2, (unique_count - 5) * 0.05)
            diversity_ratio = min(1.0, diversity_ratio + bonus)

        return diversity_ratio

    def _calculate_sequence_coherence(self, sequence: Sequence[str]) -> float:
        """Calculate structural coherence score (0.0-1.0).

        Combines insights from pattern detector with cycle-specific
        coherence requirements (good start/end, essential elements).
        """
        from .patterns import AdvancedPatternDetector

        if not sequence:
            return 0.0

        detector = AdvancedPatternDetector()
        health_metrics = detector._structural_health(sequence)

        score = 0.0

        # 1. Good start (emission or reception)
        if sequence[0] in {EMISSION, RECEPTION, COHERENCE}:
            score += 0.25

        # 2. Good ending (check has_closure from health metrics)
        if health_metrics.get("has_closure", False):
            score += 0.25

        # 3. Contains coupling (network integration)
        if COUPLING in sequence:
            score += 0.15

        # 4. Contains resonance (amplification)
        if RESONANCE in sequence:
            score += 0.15

        # 5. Has emission or reception (information flow)
        if EMISSION in sequence or RECEPTION in sequence:
            score += 0.10

        # 6. Bonus for cyclic closure (starts and ends with stabilizers)
        if len(sequence) >= 2:
            if sequence[0] in _STABILIZERS_SET and sequence[-1] in _STABILIZERS_SET:
                score += 0.10

        return min(1.0, score)

    def _determine_cycle_type(self, regenerator: str) -> CycleType:
        """Determine cycle type based on dominant regenerator."""
        if regenerator == TRANSITION:
            return CycleType.TRANSFORMATIVE  # NAV
        elif regenerator == RECURSIVITY:
            return CycleType.RECURSIVE  # REMESH
        elif regenerator == SILENCE:
            return CycleType.MEDITATIVE  # SHA
        else:
            return CycleType.REGENERATIVE  # Generic
