"""Unified validation interface consolidating grammar, graph and spectral checks.

This package re-exports the canonical grammar helpers implemented in
``tnfr.operators.grammar`` and the runtime validators so downstream code can rely
on a single import path for structural validation primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, TypeVar, runtime_checkable

SubjectT = TypeVar("SubjectT")


@dataclass(slots=True)
class ValidationOutcome(Generic[SubjectT]):
    """Result emitted by all canonical TNFR validators."""

    subject: SubjectT
    """The validated subject in canonical form."""

    passed: bool
    """Whether the validation succeeded without invariant violations."""

    summary: Mapping[str, Any]
    """Structured diagnostics describing the performed checks."""

    artifacts: Mapping[str, Any] | None = None
    """Optional artefacts (e.g. clamped nodes, normalised vectors)."""


@runtime_checkable
class Validator(Protocol[SubjectT]):
    """Contract implemented by runtime and spectral validators."""

    def validate(self, subject: SubjectT, /, **kwargs: Any) -> ValidationOutcome[SubjectT]:
        """Validate ``subject`` returning a :class:`ValidationOutcome`."""

    def report(self, outcome: "ValidationOutcome[SubjectT]") -> str:
        """Produce a concise textual explanation for ``outcome``."""


from .compatibility import CANON_COMPAT, CANON_FALLBACK
from ..operators import grammar as _grammar
from ..types import Glyph
from .graph import GRAPH_VALIDATORS, run_validators
from .window import validate_window
from .runtime import GraphCanonicalValidator, apply_canonical_clamps, validate_canon
from .rules import coerce_glyph, get_norm, glyph_fallback, normalized_dnfr
from .soft_filters import (
    acceleration_norm,
    check_repeats,
    maybe_force,
    soft_grammar_filters,
)
_GRAMMAR_EXPORTS = tuple(getattr(_grammar, "__all__", ()))

globals().update({name: getattr(_grammar, name) for name in _GRAMMAR_EXPORTS})

_RUNTIME_EXPORTS = (
    "ValidationOutcome",
    "Validator",
    "GraphCanonicalValidator",
    "apply_canonical_clamps",
    "validate_canon",
    "GRAPH_VALIDATORS",
    "run_validators",
    "CANON_COMPAT",
    "CANON_FALLBACK",
    "validate_window",
    "coerce_glyph",
    "get_norm",
    "glyph_fallback",
    "normalized_dnfr",
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
    "NFRValidator",
)

__all__ = _GRAMMAR_EXPORTS + _RUNTIME_EXPORTS

_ENFORCE_CANONICAL_GRAMMAR = _grammar.enforce_canonical_grammar


def enforce_canonical_grammar(
    G: Any,
    n: Any,
    cand: Any,
    ctx: Any | None = None,
) -> Any:
    """Proxy to :func:`tnfr.operators.grammar.enforce_canonical_grammar` preserving Glyph outputs."""

    result = _ENFORCE_CANONICAL_GRAMMAR(G, n, cand, ctx)
    if isinstance(cand, Glyph) and not isinstance(result, Glyph):
        translated = _grammar.function_name_to_glyph(result)
        if translated is None and isinstance(result, str):
            try:
                translated = Glyph(result)
            except (TypeError, ValueError):
                translated = None
        if translated is not None:
            return translated
    return result


def __getattr__(name: str) -> Any:
    if name == "NFRValidator":
        from .spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)

