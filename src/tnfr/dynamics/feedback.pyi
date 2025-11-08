"""Type stubs for tnfr.dynamics.feedback module."""

from typing import Any
from ..types import TNFRGraph, NodeId

class StructuralFeedbackLoop:
    G: TNFRGraph
    node: NodeId
    target_coherence: float
    tau_adaptive: float
    learning_rate: float

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        target_coherence: float = ...,
        tau_adaptive: float = ...,
        learning_rate: float = ...,
    ) -> None: ...
    def regulate(self) -> str: ...
    def _compute_local_coherence(self) -> float: ...
    def adapt_thresholds(self, performance_metric: float) -> None: ...
    def homeostatic_cycle(self, num_steps: int = ...) -> None: ...
