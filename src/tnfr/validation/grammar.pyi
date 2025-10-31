from collections.abc import Iterable
from typing import Any

from ..node import NodeProtocol
from ..types import Glyph, NodeId, TNFRGraph

__all__ = (
    "GrammarContext",
    "_gram_state",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "apply_glyph_with_grammar",
)


class GrammarContext:
    G: TNFRGraph
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]

    @classmethod
    def from_graph(cls, G: TNFRGraph) -> "GrammarContext": ...


def _gram_state(nd: dict[str, Any]) -> dict[str, Any]: ...


def enforce_canonical_grammar(
    G: TNFRGraph,
    n: NodeId,
    cand: Glyph | str,
    ctx: GrammarContext | None = ...,
) -> Glyph | str: ...


def on_applied_glyph(G: TNFRGraph, n: NodeId, applied: Glyph | str) -> None: ...


def apply_glyph_with_grammar(
    G: TNFRGraph,
    nodes: Iterable[NodeId | NodeProtocol] | None,
    glyph: Glyph | str,
    window: int | None = ...,
) -> None: ...
