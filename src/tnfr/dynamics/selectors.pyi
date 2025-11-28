from __future__ import annotations

import abc
from ..selector import _selector_parallel_jobs as _selector_parallel_jobs
from ..types import (
    GlyphCode as GlyphCode,
    GlyphSelector,
    HistoryState,
    NodeId,
    TNFRGraph,
)
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GlyphCode",
    "AbstractSelector",
    "DefaultGlyphSelector",
    "ParametricGlyphSelector",
    "default_glyph_selector",
    "parametric_glyph_selector",
    "_SelectorPreselection",
    "_configure_selector_weights",
    "_apply_selector",
    "_apply_glyphs",
    "_selector_parallel_jobs",
    "_prepare_selector_preselection",
    "_resolve_preselected_glyph",
    "_choose_glyph",
]

class AbstractSelector(ABC, metaclass=abc.ABCMeta):
    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...
    @abstractmethod
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...
    def __call__(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...

def _configure_selector_weights(G: TNFRGraph) -> Mapping[str, float]: ...
@dataclass
class _SelectorPreselection:
    kind: str
    metrics: Mapping[Any, tuple[float, float, float]]
    base_choices: Mapping[Any, GlyphCode]
    thresholds: Mapping[str, float] | None = ...
    margin: float | None = ...

class DefaultGlyphSelector(AbstractSelector):
    def __init__(self) -> None: ...
    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...

class ParametricGlyphSelector(AbstractSelector):
    def __init__(self) -> None: ...
    def prepare(self, graph: TNFRGraph, nodes: Sequence[NodeId]) -> None: ...
    def select(self, graph: TNFRGraph, node: NodeId) -> GlyphCode: ...

default_glyph_selector: Incomplete
parametric_glyph_selector: Incomplete

def _choose_glyph(
    G: TNFRGraph,
    n: NodeId,
    selector: GlyphSelector,
    use_canon: bool,
    h_al: MutableMapping[Any, int],
    h_en: MutableMapping[Any, int],
    al_max: int,
    en_max: int,
) -> GlyphCode: ...
def _prepare_selector_preselection(
    G: TNFRGraph, selector: GlyphSelector, nodes: Sequence[NodeId]
) -> _SelectorPreselection | None: ...
def _resolve_preselected_glyph(
    G: TNFRGraph,
    n: NodeId,
    selector: GlyphSelector,
    preselection: _SelectorPreselection | None,
) -> GlyphCode: ...
def _apply_glyphs(G: TNFRGraph, selector: GlyphSelector, hist: HistoryState) -> None: ...
def _apply_selector(G: TNFRGraph) -> GlyphSelector: ...
