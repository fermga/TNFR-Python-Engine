"""Type stubs for adaptive learning module."""

from typing import Any

from ..types import TNFRGraph

class AdaptiveLearningSystem:
    """System for adaptive learning using TNFR operators."""

    G: TNFRGraph
    node: Any
    learning_rate: float
    consolidation_threshold: float

    def __init__(
        self,
        graph: TNFRGraph,
        node: Any,
        learning_rate: float = ...,
        consolidation_threshold: float = ...,
    ) -> None: ...
    def learn_from_input(
        self,
        stimulus: float,
        consolidate: bool = ...,
    ) -> None: ...
    def _is_dissonant(self, stimulus: float) -> bool: ...
    def consolidate_memory(self) -> None: ...
    def adaptive_cycle(self, num_iterations: int = ...) -> None: ...
    def _should_stabilize(self) -> bool: ...
    def deep_learning_cycle(self) -> None: ...
    def exploratory_learning_cycle(self) -> None: ...
    def adaptive_mutation_cycle(self) -> None: ...
