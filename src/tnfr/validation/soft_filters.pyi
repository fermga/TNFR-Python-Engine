from __future__ import annotations

from typing import Any, Callable, Mapping

from ..types import Glyph
from .grammar import GrammarContext

__all__ = (
    "acceleration_norm",
    "check_repeats",
    "maybe_force",
    "soft_grammar_filters",
)

def acceleration_norm(ctx: GrammarContext, nd: Mapping[str, Any]) -> float: ...
def check_repeats(ctx: GrammarContext, n: Any, cand: Glyph | str) -> Glyph | str: ...
def maybe_force(
    ctx: GrammarContext,
    n: Any,
    cand: Glyph | str,
    original: Glyph | str,
    accessor: Callable[[GrammarContext, Mapping[str, Any]], float],
    key: str,
) -> Glyph | str: ...
def soft_grammar_filters(
    ctx: GrammarContext,
    n: Any,
    cand: Glyph | str,
    *,
    original: Glyph | str | None = ...,
    template: Glyph | str | None = ...,
) -> Glyph | str: ...
