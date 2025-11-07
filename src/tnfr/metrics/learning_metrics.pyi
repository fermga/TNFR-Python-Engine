"""Type stubs for learning metrics module."""

from typing import Any

from ..types import TNFRGraph

def compute_learning_plasticity(
    G: TNFRGraph,
    node: Any,
    window: int = ...,
) -> float: ...

def compute_consolidation_index(
    G: TNFRGraph,
    node: Any,
    window: int = ...,
) -> float: ...

def compute_learning_efficiency(
    G: TNFRGraph,
    node: Any,
) -> float: ...
