from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from ..types import NodeId, TNFRGraph
from . import ValidationOutcome, Validator

class GraphCanonicalValidator(Validator[TNFRGraph]):
    def __init__(
        self,
        *,
        recompute_frequency_maxima: bool = ...,
        enforce_graph_validators: bool = ...,
    ) -> None: ...
    def validate(self, subject: TNFRGraph, /, **kwargs: Any) -> ValidationOutcome[TNFRGraph]: ...
    def report(self, outcome: ValidationOutcome[TNFRGraph]) -> str: ...

def apply_canonical_clamps(
    nd: MutableMapping[str, Any],
    G: TNFRGraph | None = ...,
    node: NodeId | None = ...,
) -> None: ...
def validate_canon(G: TNFRGraph) -> ValidationOutcome[TNFRGraph]: ...

__all__: tuple[str, ...]
