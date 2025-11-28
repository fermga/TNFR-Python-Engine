"""Pre-configured templates for common TNFR use cases.

This module provides ready-to-use templates for domain-specific TNFR
applications. Each template encodes structural patterns and operator
sequences appropriate for modeling different types of complex systems
while maintaining TNFR theoretical fidelity.

Examples
--------
Model social network dynamics:

>>> from tnfr.sdk import TNFRTemplates
>>> results = TNFRTemplates.social_network_simulation(
...     people=50, connections_per_person=6, simulation_steps=25
... )
>>> print(results.summary())

Model neural network with TNFR principles:

>>> results = TNFRTemplates.neural_network_model(
...     neurons=100, connectivity=0.15, activation_cycles=30
... )
"""

from __future__ import annotations

from typing import Optional

from .fluent import TNFRNetwork, NetworkResults

__all__ = ["TNFRTemplates"]


class TNFRTemplates:
    """Pre-configured templates for common domain-specific use cases.

    This class provides static methods that encode expert knowledge about
    how to apply TNFR to different domains. Each template configures
    appropriate structural frequencies, topologies, and operator sequences
    for its target domain.

    Methods are named after the domain they model and return
    :class:`NetworkResults` instances ready for analysis.
    """

    @staticmethod
    def social_network_simulation(
        people: int = 50,
        connections_per_person: int = 5,
        simulation_steps: int = 20,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Simulate social network dynamics using TNFR.

        Models human social networks where nodes represent individuals with
        moderate structural frequencies (representing human timescales) and
        small-world connectivity (reflecting real social structures).

        The simulation applies activation, synchronization, and consolidation
        phases that mirror social dynamics: initial interaction, alignment
        of behaviors/beliefs, and stabilization of relationships.

        Parameters
        ----------
        people : int, default=50
            Number of individuals in the social network.
        connections_per_person : int, default=5
            Average number of social connections per person.
        simulation_steps : int, default=20
            Number of simulation steps to run.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results containing coherence metrics and sense indices.

        Examples
        --------
        >>> results = TNFRTemplates.social_network_simulation(people=100)
        >>> print(f"Social coherence: {results.coherence:.3f}")
        """
        connection_prob = connections_per_person / people

        network = TNFRNetwork("social_network")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Human timescale frequencies: moderate reorganization rates
        network.add_nodes(people, vf_range=(0.3, 0.7))

        # Small-world topology reflects real social structures
        network.connect_nodes(connection_prob, "small_world")

        # Simulate social dynamics in phases
        steps_per_phase = simulation_steps // 3

        # Phase 1: Initial activation (meeting, interacting)
        network.apply_sequence("basic_activation", repeat=steps_per_phase)

        # Phase 2: Network synchronization (alignment, influence)
        network.apply_sequence("network_sync", repeat=steps_per_phase)

        # Phase 3: Consolidation (stabilization of relationships)
        network.apply_sequence("consolidation", repeat=simulation_steps - 2 * steps_per_phase)

        return network.measure()

    @staticmethod
    def neural_network_model(
        neurons: int = 100,
        connectivity: float = 0.15,
        activation_cycles: int = 30,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Model neural network using TNFR structural principles.

        Represents neurons as TNFR nodes with moderate to high structural
        frequencies (within TNFR bounds) and sparse random connectivity
        (typical of cortical networks). Applies rapid activation cycles
        to model neural firing patterns.

        Parameters
        ----------
        neurons : int, default=100
            Number of neurons in the network.
        connectivity : float, default=0.15
            Connection probability between neurons (sparse connectivity).
        activation_cycles : int, default=30
            Number of activation cycles to simulate.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results with neural coherence and sense indices.

        Examples
        --------
        >>> results = TNFRTemplates.neural_network_model(neurons=200)
        >>> avg_si = sum(results.sense_indices.values()) / len(results.sense_indices)
        >>> print(f"Average neural sense: {avg_si:.3f}")
        """
        network = TNFRNetwork("neural_model")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Neural frequencies: high end of valid range (0.5-1.0 Hz_str)
        network.add_nodes(neurons, vf_range=(0.5, 1.0))

        # Sparse random connectivity typical of cortical networks
        network.connect_nodes(connectivity, "random")

        # Rapid activation cycles modeling neural firing
        network.apply_sequence("basic_activation", repeat=activation_cycles)

        return network.measure()

    @staticmethod
    def ecosystem_dynamics(
        species: int = 25,
        interaction_strength: float = 0.25,
        evolution_steps: int = 50,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Model ecosystem dynamics with TNFR structural evolution.

        Represents species as nodes with diverse structural frequencies
        (within TNFR bounds) and medium connectivity (species interactions).
        Alternates between mutation (innovation), synchronization (adaptation),
        and consolidation (stable ecosystems).

        Parameters
        ----------
        species : int, default=25
            Number of species in the ecosystem.
        interaction_strength : float, default=0.25
            Probability of ecological interactions between species.
        evolution_steps : int, default=50
            Number of evolutionary steps to simulate.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results showing ecosystem coherence and species sense indices.

        Examples
        --------
        >>> results = TNFRTemplates.ecosystem_dynamics(species=30)
        >>> print(f"Ecosystem stability: {results.coherence:.3f}")
        """
        network = TNFRNetwork("ecosystem")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Biological timescales: diversity within bounds (0.2-0.9 Hz_str)
        network.add_nodes(species, vf_range=(0.2, 0.9))

        # Random interaction network
        network.connect_nodes(interaction_strength, "random")

        # Simulate evolution in cycles
        num_cycles = evolution_steps // 10
        for cycle in range(num_cycles):
            phase = cycle % 3

            if phase == 0:
                # Innovation: mutations and new forms
                network.apply_sequence("creative_mutation", repeat=3)
            elif phase == 1:
                # Adaptation: species synchronize to environment
                network.apply_sequence("network_sync", repeat=5)
            else:
                # Stabilization: ecosystem consolidates
                network.apply_sequence("consolidation", repeat=2)

        return network.measure()

    @staticmethod
    def creative_process_model(
        ideas: int = 15,
        inspiration_level: float = 0.4,
        development_cycles: int = 12,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Model creative processes using TNFR structural evolution.

        Represents ideas as nodes with diverse structural frequencies
        (creative exploration within TNFR bounds) starting with sparse
        connectivity (disconnected ideas). Applies exploration, mutation,
        and synthesis sequences to model creative ideation and development.

        Parameters
        ----------
        ideas : int, default=15
            Number of initial ideas/concepts.
        inspiration_level : float, default=0.4
            Level of cross-pollination between ideas (rewiring probability).
        development_cycles : int, default=12
            Number of creative development cycles.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results showing creative coherence and idea sense indices.

        Examples
        --------
        >>> results = TNFRTemplates.creative_process_model(ideas=20)
        >>> print(f"Creative coherence: {results.coherence:.3f}")
        """
        network = TNFRNetwork("creative_process")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Diverse frequencies for creative exploration (0.3-0.9 Hz_str)
        network.add_nodes(ideas, vf_range=(0.3, 0.9))

        # Sparse initial connectivity: ideas start disconnected
        network.connect_nodes(0.1, "random")

        # Creative process in phases
        cycles_per_phase = development_cycles // 3

        # Phase 1: Exploration (divergent thinking)
        network.apply_sequence("exploration", repeat=cycles_per_phase)

        # Phase 2: Development (mutation and elaboration)
        network.apply_sequence("creative_mutation", repeat=cycles_per_phase)

        # Phase 3: Integration (convergent synthesis)
        network.apply_sequence("network_sync", repeat=development_cycles - 2 * cycles_per_phase)

        return network.measure()

    @staticmethod
    def organizational_network(
        agents: int = 40,
        hierarchy_depth: int = 3,
        coordination_steps: int = 25,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Model organizational networks with hierarchical structure.

        Creates a hierarchical network structure representing organizational
        levels with moderate structural frequencies (organizational timescales).
        Models coordination and information flow through the hierarchy.

        Parameters
        ----------
        agents : int, default=40
            Number of agents/roles in the organization.
        hierarchy_depth : int, default=3
            Number of hierarchical levels.
        coordination_steps : int, default=25
            Number of coordination cycles to simulate.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results showing organizational coherence.

        Examples
        --------
        >>> results = TNFRTemplates.organizational_network(agents=50)
        >>> print(f"Organizational coherence: {results.coherence:.3f}")
        """
        network = TNFRNetwork("organizational_network")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Organizational timescales: moderate frequencies
        network.add_nodes(agents, vf_range=(0.2, 0.8))

        # Small-world topology approximates organizational structure
        # (local teams + cross-functional connections)
        network.connect_nodes(0.15, "small_world")

        # Simulate organizational dynamics
        steps_per_phase = coordination_steps // 2

        # Phase 1: Information propagation and alignment
        network.apply_sequence("network_sync", repeat=steps_per_phase)

        # Phase 2: Stabilization of coordinated action
        network.apply_sequence("consolidation", repeat=coordination_steps - steps_per_phase)

        return network.measure()
