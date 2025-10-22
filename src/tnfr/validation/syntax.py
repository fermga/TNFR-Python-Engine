"""Syntax validation for TNFR operator sequences."""

from __future__ import annotations

from collections.abc import Iterable

from ..operators.registry import OPERATORS
from ..config.operator_names import (
    COHERENCE,
    INTERMEDIATE_OPERATORS,
    RECEPTION,
    SELF_ORGANIZATION,
    SELF_ORGANIZATION_CLOSURES,
    VALID_END_OPERATORS,
    VALID_START_OPERATORS,
    canonical_operator_name,
    operator_display_name,
)

__all__ = ("validate_sequence",)


_MISSING = object()


_CANONICAL_START = tuple(sorted(VALID_START_OPERATORS))
_CANONICAL_INTERMEDIATE = tuple(sorted(INTERMEDIATE_OPERATORS))
_CANONICAL_END = tuple(sorted(VALID_END_OPERATORS))


def _format_token_group(tokens: tuple[str, ...]) -> str:
    return ", ".join(operator_display_name(token) for token in sorted(tokens))


def _validate_start(token: str) -> tuple[bool, str]:
    """Ensure the sequence begins with a valid structural operator."""

    if not isinstance(token, str):
        return False, "tokens must be str"
    if token not in VALID_START_OPERATORS:
        valid_tokens = _format_token_group(_CANONICAL_START)
        return False, f"must start with {valid_tokens}"
    return True, ""


def _validate_intermediate(
    found_reception: bool, found_coherence: bool, seen_intermediate: bool
) -> tuple[bool, str]:
    """Check that the central TNFR segment is present."""

    if not (found_reception and found_coherence):
        return False, f"missing {RECEPTION}→{COHERENCE} segment"
    if not seen_intermediate:
        intermediate_tokens = _format_token_group(_CANONICAL_INTERMEDIATE)
        return False, f"missing {intermediate_tokens} segment"
    return True, ""


def _validate_end(last_token: str, open_thol: bool) -> tuple[bool, str]:
    """Validate closing operator and any pending THOL blocks."""

    if last_token not in VALID_END_OPERATORS:
        cierre_tokens = _format_token_group(_CANONICAL_END)
        return False, f"sequence must end with {cierre_tokens}"
    if open_thol:
        return False, "THOL block without closure"
    return True, ""


def _validate_known_tokens(
    token_to_canonical: dict[str, str]
) -> tuple[bool, str]:
    """Ensure all tokens map to canonical operators."""

    unknown_tokens = {
        alias
        for alias, canonical in token_to_canonical.items()
        if canonical not in OPERATORS
    }
    if unknown_tokens:
        ordered = ", ".join(sorted(unknown_tokens))
        return False, f"unknown tokens: {ordered}"
    return True, ""


def _validate_token_sequence(names: list[str]) -> tuple[bool, str]:
    """Validate token format and logical coherence in one pass."""

    if not names:
        return False, "empty sequence"

    ok, msg = _validate_start(names[0])
    if not ok:
        return False, msg

    token_to_canonical: dict[str, str] = {}
    found_reception = False
    found_coherence = False
    seen_intermediate = False
    open_thol = False

    for name in names:
        if not isinstance(name, str):
            return False, "tokens must be str"
        canonical = canonical_operator_name(name)
        token_to_canonical[name] = canonical

        if canonical == RECEPTION and not found_reception:
            found_reception = True
        elif found_reception and canonical == COHERENCE and not found_coherence:
            found_coherence = True
        elif (
            found_coherence
            and not seen_intermediate
            and canonical in INTERMEDIATE_OPERATORS
        ):
            seen_intermediate = True

        if canonical == SELF_ORGANIZATION:
            open_thol = True
        elif open_thol and canonical in SELF_ORGANIZATION_CLOSURES:
            open_thol = False

    ok, msg = _validate_known_tokens(token_to_canonical)
    if not ok:
        return False, msg
    ok, msg = _validate_intermediate(found_reception, found_coherence, seen_intermediate)
    if not ok:
        return False, msg
    ok, msg = _validate_end(names[-1], open_thol)
    if not ok:
        return False, msg
    return True, "ok"


def validate_sequence(
    names: Iterable[str] | object = _MISSING, **kwargs: object
) -> tuple[bool, str]:
    """Validate minimal TNFR syntax rules."""

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"validate_sequence() got unexpected keyword argument(s): {unexpected}"
        )

    if names is _MISSING:
        raise TypeError("validate_sequence() missing required argument: 'names'")

    sequence = list(names)
    return _validate_token_sequence(sequence)
