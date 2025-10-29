from collections.abc import Mapping, Sequence
from typing import Callable, Tuple

from ..types import EPIValue, NodeId, StructuralFrequency, TNFRGraph

ValidatorFunc = Callable[[TNFRGraph], None]
NodeData = Mapping[str, object]
AliasSequence = Sequence[str]

GRAPH_VALIDATORS: Tuple[ValidatorFunc, ...]

def validate_window(window: int, *, positive: bool = ...) -> int: ...

def run_validators(graph: TNFRGraph) -> None: ...
