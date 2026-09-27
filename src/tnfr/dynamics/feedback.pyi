"""Type stubs for tnfr.dynamics.feedback module."""

from ..types import NodeId, TNFRGraph

class StructuralFeedbackLoop:
    G: TNFRGraph
    node: NodeId
    target_coherence: float
    tau_adaptive: float
    learning_rate: float
    COHERENCE_TOL_LOW: float
    COHERENCE_TOL_HIGH: float
    DNFR_THRESHOLD: float
    EPI_THRESHOLD: float
    backend: None
    _use_optimizations: bool

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        target_coherence: float = ...,
        tau_adaptive: float = ...,
        learning_rate: float = ...,
        coherence_tolerance_low: float = ...,
        coherence_tolerance_high: float = ...,
        dnfr_threshold: float = ...,
        epi_threshold: float = ...,
    ) -> None: ...
    def regulate(self) -> str: ...
    def _compute_local_coherence(self) -> float: ...
    def adapt_thresholds(self, performance_metric: float) -> None: ...
    def homeostatic_cycle(self, num_steps: int = ...) -> None: ...
