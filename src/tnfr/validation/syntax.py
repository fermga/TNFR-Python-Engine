"""Compatibility wrapper around :mod:`tnfr.operators.grammar` validators."""

from __future__ import annotations

from collections.abc import Iterable

from . import ValidationOutcome
from ..operators.grammar import SequenceValidationResult, validate_sequence as _validate

__all__ = ("validate_sequence",)


def _to_validation_outcome(result: SequenceValidationResult) -> ValidationOutcome[tuple[str, ...]]:
    summary = dict(result.summary)
    artifacts = {
        "canonical_tokens": result.canonical_tokens,
        "metadata": result.metadata,
    }
    return ValidationOutcome(
        subject=result.tokens,
        passed=result.passed,
        summary=summary,
        artifacts=artifacts,
    )


def validate_sequence(
    names: Iterable[str] | object = None, **kwargs: object
) -> ValidationOutcome[tuple[str, ...]]:
    """Validate minimal TNFR syntax rules returning a legacy :class:`ValidationOutcome`."""

    if names is None and "names" not in kwargs:
        raise TypeError("validate_sequence() missing required argument: 'names'")

    if names is None and "names" in kwargs:
        names = kwargs.pop("names")

    result = _validate(names, **kwargs)
    return _to_validation_outcome(result)
