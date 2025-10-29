from collections.abc import Mapping
from typing import Callable, Sequence

from ..types import TNFRGraph
from ..validation.graph import AliasSequence, GRAPH_VALIDATORS, NodeData, ValidatorFunc

__all__: tuple[str, ...]

def validate_window(window: int, *, positive: bool = ...) -> int: ...
def run_validators(graph: TNFRGraph) -> None: ...
