from collections.abc import Mapping
from typing import Callable, Sequence

from ..types import TNFRGraph
from .._compat import TypeAlias

ValidatorFunc: TypeAlias = Callable[[TNFRGraph], None]
NodeData: TypeAlias = Mapping[str, object]
AliasSequence: TypeAlias = Sequence[str]

__all__: tuple[str, ...]

def __getattr__(name: str) -> object: ...

def validate_window(window: int, *, positive: bool = ...) -> int: ...

def run_validators(graph: TNFRGraph) -> None: ...

GRAPH_VALIDATORS: tuple[ValidatorFunc, ...]
