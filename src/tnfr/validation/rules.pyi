from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from typing import Any, Mapping, TypeVar

from ..types import Glyph, NodeId
from .grammar import GrammarContext

__all__ = (
    "coerce_glyph",
    "glyph_fallback",
    "get_norm",
    "normalized_dnfr",
    "_norm_attr",
    "_si",
    "_check_oz_to_zhir",
    "_check_thol_closure",
    "_check_compatibility",
)

_T = TypeVar("_T")

def coerce_glyph(val: _T) -> Glyph | _T: ...
def glyph_fallback(cand_key: str, fallbacks: Mapping[str, Any]) -> Glyph | str: ...
def get_norm(ctx: GrammarContext, key: str) -> float: ...
def _norm_attr(
    ctx: GrammarContext, nd: Mapping[str, Any], attr_alias: str, norm_key: str
) -> float: ...
def _si(nd: Mapping[str, Any]) -> float: ...
def normalized_dnfr(ctx: GrammarContext, nd: Mapping[str, Any]) -> float: ...
def _check_oz_to_zhir(ctx: GrammarContext, n: NodeId, cand: Glyph | str) -> Glyph | str: ...
def _check_thol_closure(
    ctx: GrammarContext,
    n: NodeId,
    cand: Glyph | str,
    st: dict[str, Any],
) -> Glyph | str: ...
def _check_compatibility(ctx: GrammarContext, n: NodeId, cand: Glyph | str) -> Glyph | str: ...
