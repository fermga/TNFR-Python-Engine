"""Compatibility report adapter for the shared operator-word validator.

This class retains its public report shape without a competing semantic rule
table. Word admission does not certify live state, stability or emergence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .invariants import InvariantSeverity, InvariantViolation

__all__ = ["SequenceSemanticValidator"]


class SequenceSemanticValidator:
    """Report shared word-policy rejection in the legacy violation format."""

    def validate_semantic_sequence(
        self, sequence: list[str], *, context: Mapping[str, Any] | None = None
    ) -> list[InvariantViolation]:
        """Delegate admission; never infer a dynamical property from labels."""
        from ..operators.grammar_patterns import validate_sequence

        outcome = validate_sequence(sequence, context=context)
        if outcome.passed:
            return []
        return [
            InvariantViolation(
                invariant_id=0,
                severity=InvariantSeverity.ERROR,
                description="Operator word admission: "
                + outcome.summary.get("message", "word rejected"),
                suggestion="Review the declared word and its admitted context.",
            )
        ]
