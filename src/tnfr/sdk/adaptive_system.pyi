"""Type stubs for tnfr.sdk.adaptive_system module."""

from ..types import TNFRGraph, NodeId
from ..dynamics.feedback import StructuralFeedbackLoop
from ..dynamics.adaptive_sequences import AdaptiveSequenceSelector
from ..dynamics.homeostasis import StructuralHomeostasis
from ..dynamics.learning import AdaptiveLearningSystem
from ..dynamics.metabolism import StructuralMetabolism

class TNFRAdaptiveSystem:
    G: TNFRGraph
    node: NodeId
    feedback: StructuralFeedbackLoop
    sequence_selector: AdaptiveSequenceSelector
    homeostasis: StructuralHomeostasis
    learning: AdaptiveLearningSystem
    metabolism: StructuralMetabolism

    def __init__(self, graph: TNFRGraph, node: NodeId) -> None: ...
    def autonomous_evolution(self, num_cycles: int = ...) -> None: ...
    def _measure_stress(self) -> float: ...
