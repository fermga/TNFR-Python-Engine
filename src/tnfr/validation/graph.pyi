from collections.abc import Sequence
from typing import Tuple

from ..types import (
    EPIValue,
    NodeAttrMap,
    NodeId,
    StructuralFrequency,
    TNFRGraph,
    ValidatorFunc,
)

NodeData = NodeAttrMap
AliasSequence = Sequence[str]

GRAPH_VALIDATORS: Tuple[ValidatorFunc, ...]

def run_validators(graph: TNFRGraph) -> None: ...
