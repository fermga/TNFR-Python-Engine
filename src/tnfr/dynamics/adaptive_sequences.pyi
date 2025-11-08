"""Type stubs for tnfr.dynamics.adaptive_sequences module."""

from typing import Any, Dict, List
from ..types import TNFRGraph, NodeId

class AdaptiveSequenceSelector:
    G: TNFRGraph
    node: NodeId
    sequences: Dict[str, List[str]]
    performance: Dict[str, List[float]]

    def __init__(self, graph: TNFRGraph, node: NodeId) -> None: ...
    def select_sequence(self, context: Dict[str, Any]) -> List[str]: ...
    def record_performance(self, sequence_name: str, coherence_gain: float) -> None: ...
