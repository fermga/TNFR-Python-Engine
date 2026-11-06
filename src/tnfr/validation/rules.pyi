from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from ..operators.grammar_context import GrammarContext
from ..types import Glyph

__all__ = (
    "coerce_glyph",
    "glyph_fallback",
    "get_norm",
    "normalized_dnfr",
    "_norm_attr",
    "_si",
)

_T = TypeVar("_T")

def coerce_glyph(val: _T) -> Glyph | _T: ...
def glyph_fallback(cand_key: str, fallbacks: Mapping[str, Any]) -> Glyph | str: ...
def _admitted_glyph(value: Any, label: str) -> Glyph: ...
def get_norm(ctx: GrammarContext, key: str) -> float: ...
def _norm_attr(
    ctx: GrammarContext,
    nd: Mapping[str, Any],
    attr_alias: str | tuple[str, ...],
    norm_key: str,
) -> float: ...
def _si(nd: Mapping[str, Any]) -> float: ...
def normalized_dnfr(ctx: GrammarContext, nd: Mapping[str, Any]) -> float: ...
