"""Type stubs for TNFR SDK fluent API."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx

class NetworkConfig:
    random_seed: Optional[int]
    validate_invariants: bool
    auto_stabilization: bool
    default_vf_range: tuple[float, float]
    default_epi_range: tuple[float, float]

    def __init__(
        self,
        random_seed: Optional[int] = None,
        validate_invariants: bool = True,
        auto_stabilization: bool = True,
        default_vf_range: tuple[float, float] = (0.1, 1.0),
        default_epi_range: tuple[float, float] = (0.1, 0.9),
    ) -> None: ...

class NetworkResults:
    coherence: float
    sense_indices: Dict[str, float]
    delta_nfr: Dict[str, float]
    graph: Any
    avg_vf: Optional[float]
    avg_phase: Optional[float]

    def __init__(
        self,
        coherence: float,
        sense_indices: Dict[str, float],
        delta_nfr: Dict[str, float],
        graph: Any,
        avg_vf: Optional[float] = None,
        avg_phase: Optional[float] = None,
    ) -> None: ...
    def summary(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...

class TNFRNetwork:
    name: str

    def __init__(
        self,
        name: str = "tnfr_network",
        config: Optional[NetworkConfig] = None,
    ) -> None: ...
    def add_nodes(
        self,
        count: int,
        vf_range: Optional[tuple[float, float]] = None,
        epi_range: Optional[tuple[float, float]] = None,
        phase_range: tuple[float, float] = (0.0, 6.283185307179586),
        random_seed: Optional[int] = None,
    ) -> TNFRNetwork: ...
    def connect_nodes(
        self,
        connection_probability: float = 0.3,
        connection_pattern: str = "random",
    ) -> TNFRNetwork: ...
    def apply_sequence(
        self,
        sequence: Union[str, List[str]],
        repeat: int = 1,
    ) -> TNFRNetwork: ...
    def measure(self) -> NetworkResults: ...
    def visualize(self, **kwargs: Any) -> TNFRNetwork: ...
    def save(self, filepath: Union[str, Path]) -> TNFRNetwork: ...
    @property
    def graph(self) -> nx.Graph: ...
