"""Type stubs for tnfr.dynamics.homeostasis module."""

from typing import Tuple

from ..types import NodeId, TNFRGraph

class StructuralHomeostasis:
    G: TNFRGraph
    node: NodeId
    epi_range: Tuple[float, float]
    vf_range: Tuple[float, float]
    dnfr_range: Tuple[float, float]

    def __init__(
        self,
        graph: TNFRGraph,
        node: NodeId,
        epi_range: tuple[float, float] = ...,
        vf_range: tuple[float, float] = ...,
        dnfr_range: tuple[float, float] = ...,
    ) -> None: ...
    def maintain_equilibrium(self) -> None: ...
