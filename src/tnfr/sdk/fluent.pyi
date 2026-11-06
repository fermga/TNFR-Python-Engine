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
    unified_fields: Optional[Dict[str, Any]]
    mutation_workflows: Optional[List[Dict[str, Any]]]
    unified_fields_error: Optional[Dict[str, str]]

    def __init__(
        self,
        coherence: float,
        sense_indices: Dict[str, float],
        delta_nfr: Dict[str, float],
        graph: Any,
        avg_vf: Optional[float] = None,
        avg_phase: Optional[float] = None,
        unified_fields: Optional[Dict[str, Any]] = None,
        mutation_workflows: Optional[List[Dict[str, Any]]] = None,
        unified_fields_error: Optional[Dict[str, str]] = None,
    ) -> None: ...
    @property
    def unified_fields_available(self) -> bool: ...
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
        context: Optional[Dict[str, Any]] = None,
    ) -> TNFRNetwork: ...
    def apply_evidence_gated_mutation(
        self,
        *,
        repeat: int = 1,
        mutation_sequence: str = "creative_mutation",
        abstention_sequence: str = "exploration",
        context: Optional[Dict[str, Any]] = None,
    ) -> TNFRNetwork: ...
    def apply_adaptive_sequence(
        self,
        base_sequence: str = "basic_activation",
        target_coherence: float = ...,
    ) -> TNFRNetwork: ...
    def apply_multiple_sequences(
        self, sequences: list[tuple[str, dict[str, Any] | None, int]]
    ) -> TNFRNetwork: ...
    def apply_sequence_chain(self, chain_pattern: str = "standard") -> TNFRNetwork: ...
    def apply_canonical_sequence(
        self,
        sequence_name: str,
        node: int | None = None,
        collect_metrics: bool = True,
    ) -> TNFRNetwork: ...
    def list_canonical_sequences(
        self, domain: str | None = None, with_oz: bool = False
    ) -> dict[str, Any]: ...
    def measure(self) -> NetworkResults: ...
    def visualize(self, **kwargs: Any) -> TNFRNetwork: ...
    def save(self, filepath: Union[str, Path]) -> TNFRNetwork: ...
    def export_to_dict(self) -> dict: ...
    def clone(self) -> TNFRNetwork: ...
    def reset(self) -> TNFRNetwork: ...
    def get_node_count(self) -> int: ...
    def get_edge_count(self) -> int: ...
    def get_average_degree(self) -> float: ...
    def get_density(self) -> float: ...
    def analyze_optimization_potential(self) -> dict[str, Any]: ...
    def auto_optimize(
        self, operation_type: str = "network_simulation"
    ) -> TNFRNetwork: ...
    def learn_from_performance(
        self, performance_data: dict[str, Any] | None = None
    ) -> TNFRNetwork: ...
    def get_optimization_recommendations(
        self, operation_type: str = "measurement"
    ) -> dict[str, Any]: ...
    @property
    def graph(self) -> nx.Graph: ...
