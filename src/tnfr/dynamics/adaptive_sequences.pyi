"""Type stubs for tnfr.dynamics.adaptive_sequences module."""

from typing import Any, Dict, List

from ..types import NodeId, TNFRGraph

class AdaptiveSequenceSelector:
    G: TNFRGraph
    node: NodeId
    sequences: Dict[str, List[str]]
    performance_scores: Dict[str, List[float]]
    performance: Dict[str, List[float]]
    seed: int

    def __init__(
        self, graph: TNFRGraph, node: NodeId, seed: int | None = ...
    ) -> None: ...
    def select_sequence(self, context: Dict[str, Any]) -> List[str]: ...
    def record_performance(
        self,
        sequence_name: str,
        coherence_gain: float | None = ...,
        *,
        performance_score: float | None = ...,
    ) -> None: ...
