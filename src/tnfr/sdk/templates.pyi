"""Type stubs for TNFR SDK templates."""

from typing import Optional
from .fluent import NetworkResults

class TNFRTemplates:
    @staticmethod
    def social_network_simulation(
        people: int = 50,
        connections_per_person: int = 5,
        simulation_steps: int = 20,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def neural_network_model(
        neurons: int = 100,
        connectivity: float = 0.15,
        activation_cycles: int = 30,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def ecosystem_dynamics(
        species: int = 25,
        interaction_strength: float = 0.25,
        evolution_steps: int = 50,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def creative_process_model(
        ideas: int = 15,
        inspiration_level: float = 0.4,
        development_cycles: int = 12,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
    @staticmethod
    def organizational_network(
        agents: int = 40,
        hierarchy_depth: int = 3,
        coordination_steps: int = 25,
        random_seed: Optional[int] = None,
    ) -> NetworkResults: ...
