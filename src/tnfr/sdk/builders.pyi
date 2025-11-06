"""Type stubs for TNFR SDK builders."""

from typing import Dict, List, Optional
from .fluent import NetworkResults

class TNFRExperimentBuilder:
    @staticmethod
    def small_world_study(
        nodes: int = 50,
        rewiring_prob: float = 0.1,
        steps: int = 10,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def synchronization_study(
        nodes: int = 30,
        coupling_strength: float = 0.5,
        steps: int = 20,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def creativity_emergence(
        nodes: int = 20,
        mutation_intensity: float = 0.3,
        steps: int = 15,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def compare_topologies(
        node_count: int = 40,
        steps: int = 10,
        topologies: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, NetworkResults]: ...
    @staticmethod
    def phase_transition_study(
        nodes: int = 50,
        initial_coupling: float = 0.1,
        final_coupling: float = 0.9,
        steps_per_level: int = 5,
        coupling_levels: int = 5,
        random_seed: Optional[int] = None,
    ) -> Dict[float, NetworkResults]: ...
    @staticmethod
    def resilience_study(
        nodes: int = 40,
        initial_steps: int = 10,
        perturbation_steps: int = 5,
        recovery_steps: int = 10,
        random_seed: Optional[int] = None,
    ) -> Dict[str, NetworkResults]: ...
