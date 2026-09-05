"""Pre-configured templates for common TNFR use cases.

This module provides operational topology and operator-sequence examples
under domain-oriented names. All structural evolution uses canonical words.

These are operational graph examples, not validated domain models. Random
initial phases can make a requested word inadmissible at the live U3 gate;
the call raises and returns no result. Callers needing to inspect or recover
partially evolved state should construct a TNFRNetwork explicitly and apply
appropriate canonical preparation. These templates do not bypass phase checks.

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

from ._defaults import (
    SDK_CONNECTIVITY_DEFAULT,
    SDK_INSPIRATION_LEVEL,
    SDK_INTERACTION_STRENGTH,
    SDK_REWIRING_PROB_DEFAULT,
    SDK_VF_RANGE_LOW_MAX,
    SDK_VF_RANGE_LOW_MIN,
    SDK_VF_RANGE_MODERATE_MAX,
    SDK_VF_RANGE_MODERATE_MIN,
)
from ._topology import (
    contact_degree, hierarchical_edges, nonnegative_integer, positive_integer,
    probability, rewired_contact_edges,
)
from .fluent import NetworkConfig, NetworkResults, TNFRNetwork

__all__ = ["TNFRTemplates"]


class TNFRTemplates:
    """Operational graph examples with explicit topology and cycle controls.

    These methods choose initial structural frequencies, graph support and
    canonical operator words. Their domain names are illustrative and do not
    certify empirical models of social, neural or organizational systems.

    Methods are named after the domain they model and return
    :class:`NetworkResults` instances ready for analysis.
    """

    @staticmethod
    def social_network_simulation(
        people: int = 50,
        connections_per_person: int = 5,
        simulation_steps: int = 20,
        random_seed: int | None = None,
    ) -> NetworkResults:
        """Evolve a contact graph with a specified initial mean degree.

        A ring-based scaffold is rewired at SDK_REWIRING_PROB_DEFAULT while
        preserving edge count. Individual degrees can vary after rewiring.
        Activation, synchronization and consolidation use canonical words.

        Parameters
        ----------
        people : int, default=50
            Number of individuals in the social network.
        connections_per_person : int, default=5
            Exact initial mean degree, an integer in [0, people - 1].
            The product people * connections_per_person must be even.
            Rewiring probability is independent of this contact count.
        simulation_steps : int, default=20
            Non-negative number of canonical word applications. One third
            each is assigned to activation and synchronization; consolidation
            receives the remainder. A live grammar rejection stops execution.
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
        people = positive_integer(people, "people")
        connections_per_person = contact_degree(people, connections_per_person)
        simulation_steps = nonnegative_integer(simulation_steps, "simulation_steps")

        network = TNFRNetwork("social_network", config=NetworkConfig(random_seed=random_seed))

        # Operational frequency range; no conversion to human timescales.
        network.add_nodes(
            people, vf_range=(SDK_VF_RANGE_LOW_MIN, SDK_VF_RANGE_MODERATE_MAX)
        )

        graph = network.graph
        graph.add_edges_from(rewired_contact_edges(
            list(graph), connections_per_person, SDK_REWIRING_PROB_DEFAULT, network._rng,
        ))
        graph.graph["template_topology"] = {
            "kind": "rewired_contact_ring", "mean_degree": connections_per_person,
            "rewiring_probability": SDK_REWIRING_PROB_DEFAULT,
        }

        # Simulate social dynamics in phases
        steps_per_phase = simulation_steps // 3

        # Phase 1: Initial activation (meeting, interacting)
        network.apply_sequence("basic_activation", repeat=steps_per_phase)

        # Phase 2: Network synchronization (alignment, influence)
        network.apply_sequence("network_sync", repeat=steps_per_phase)

        # Phase 3: Consolidation (stabilization of relationships)
        network.apply_sequence(
            "consolidation", repeat=simulation_steps - 2 * steps_per_phase
        )

        return network.measure()

    @staticmethod
    def neural_network_model(
        neurons: int = 100,
        connectivity: float = SDK_CONNECTIVITY_DEFAULT,
        activation_cycles: int = 30,
        random_seed: int | None = None,
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
        connectivity : float, default=SDK_CONNECTIVITY_DEFAULT
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
        network = TNFRNetwork("neural_model", config=NetworkConfig(random_seed=random_seed))

        # Neural frequencies: high end of valid range (0.5-1.0 Hz_str)
        network.add_nodes(
            neurons, vf_range=(SDK_VF_RANGE_MODERATE_MIN, SDK_VF_RANGE_MODERATE_MAX)
        )  # Neural frequencies canonical

        # Sparse random connectivity typical of cortical networks
        network.connect_nodes(connectivity, "random")

        # Rapid activation cycles modeling neural firing
        network.apply_sequence("basic_activation", repeat=activation_cycles)

        return network.measure()

    @staticmethod
    def ecosystem_dynamics(
        species: int = 25,
        interaction_strength: float = SDK_INTERACTION_STRENGTH,
        evolution_steps: int = 50,
        random_seed: int | None = None,
    ) -> NetworkResults:
        """Cycle canonical transformation, synchronization and consolidation words.

        Species labels describe an operational random support graph, not an
        empirically calibrated ecosystem model. Every requested step is one
        complete word: creative_mutation, network_sync, then consolidation.

        Parameters
        ----------
        species : int, default=25
            Number of species in the ecosystem.
        interaction_strength : float, default=0.25
            Probability of ecological interactions between species.
        evolution_steps : int, default=50
            Non-negative number of canonical word applications, with no
            discarded remainder. A live grammar rejection stops execution.
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
        species = positive_integer(species, "species")
        evolution_steps = nonnegative_integer(evolution_steps, "evolution_steps")
        interaction_strength = probability(interaction_strength)
        network = TNFRNetwork("ecosystem", config=NetworkConfig(random_seed=random_seed))

        # Shared operational frequency interval, measured in Hz_str.
        network.add_nodes(
            species, vf_range=(SDK_VF_RANGE_LOW_MIN, SDK_VF_RANGE_MODERATE_MAX)
        )

        # Random interaction network
        network.connect_nodes(interaction_strength, "random")

        words = ("creative_mutation", "network_sync", "consolidation")
        for step in range(evolution_steps):
            network.apply_sequence(words[step % len(words)])

        return network.measure()

    @staticmethod
    def creative_process_model(
        ideas: int = 15,
        inspiration_level: float = SDK_INSPIRATION_LEVEL,
        development_cycles: int = 12,
        random_seed: int | None = None,
    ) -> NetworkResults:
        """Evolve an operational rewired idea graph through canonical words.

        The small-world scaffold starts from a ring lattice, and the inspiration
        parameter controls rewiring. This topology choice is not a physical
        measure of creativity. Exploration, mutation and synthesis remain the
        existing named operator words.

        Parameters
        ----------
        ideas : int, default=15
            Number of initial ideas/concepts.
        inspiration_level : float, default=0.4
            Rewiring probability in [0, 1], independent of the ring's edge
            count. Zero retains the lattice; one attempts every rewire.
            Tiny or complete scaffolds may have no alternative edges.
        development_cycles : int, default=12
            Non-negative number of canonical word applications. Exploration
            and mutation receive one third each; synchronization receives
            the remainder. A live grammar rejection stops execution.
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
        ideas = positive_integer(ideas, "ideas")
        inspiration_level = probability(inspiration_level)
        development_cycles = nonnegative_integer(development_cycles, "development_cycles")
        network = TNFRNetwork("creative_process", config=NetworkConfig(random_seed=random_seed))

        # Shared operational frequency interval, measured in Hz_str.
        network.add_nodes(
            ideas, vf_range=(SDK_VF_RANGE_LOW_MIN, SDK_VF_RANGE_MODERATE_MAX)
        )

        network.connect_nodes(inspiration_level, "small_world")
        network.graph.graph["template_topology"] = {
            "kind": "small_world", "rewiring_probability": inspiration_level,
        }

        # Creative process in phases
        cycles_per_phase = development_cycles // 3

        # Phase 1: Exploration (divergent thinking)
        network.apply_sequence("exploration", repeat=cycles_per_phase)

        # Phase 2: Development (mutation and elaboration)
        network.apply_sequence("creative_mutation", repeat=cycles_per_phase)

        # Phase 3: Integration (convergent synthesis)
        network.apply_sequence(
            "network_sync", repeat=development_cycles - 2 * cycles_per_phase
        )

        return network.measure()

    @staticmethod
    def organizational_network(
        agents: int = 40,
        hierarchy_depth: int = 3,
        coordination_steps: int = 25,
        random_seed: int | None = None,
    ) -> NetworkResults:
        """Evolve an undirected graph with explicit organizational layers.

        Depth one is a peer ring. Deeper graphs have one root, evenly populated
        subsequent layers, within-layer rings and one preceding-layer parent
        per child. Node hierarchy_level metadata records these graph layers;
        this scaffold does not construct nested EPIs or certify U5 coherence.

        Parameters
        ----------
        agents : int, default=40
            Number of agents/roles in the organization.
        hierarchy_depth : int, default=3
            Number of nonempty graph layers, an integer in [1, agents].
            A depth equal to agents produces a chain from the first node.
        coordination_steps : int, default=25
            Non-negative number of canonical word applications. Half use
            network_sync; consolidation receives the remainder. A live
            grammar rejection stops execution.
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
        agents = positive_integer(agents, "agents")
        hierarchy_depth = positive_integer(hierarchy_depth, "hierarchy_depth")
        if hierarchy_depth > agents:
            raise ValueError("hierarchy_depth cannot exceed agents")
        coordination_steps = nonnegative_integer(coordination_steps, "coordination_steps")
        network = TNFRNetwork("organizational_network", config=NetworkConfig(random_seed=random_seed))

        # Shared operational frequency interval, measured in Hz_str.
        network.add_nodes(
            agents, vf_range=(SDK_VF_RANGE_LOW_MIN, SDK_VF_RANGE_LOW_MAX)
        )

        graph = network.graph
        edges, levels = hierarchical_edges(list(graph), hierarchy_depth)
        graph.add_edges_from(edges)
        for node, level in levels.items():
            graph.nodes[node]["hierarchy_level"] = level
        graph.graph["template_topology"] = {
            "kind": "layered_organization", "hierarchy_depth": hierarchy_depth,
        }

        # Simulate organizational dynamics
        steps_per_phase = coordination_steps // 2

        # Phase 1: Information propagation and alignment
        network.apply_sequence("network_sync", repeat=steps_per_phase)

        # Phase 2: Stabilization of coordinated action
        network.apply_sequence(
            "consolidation", repeat=coordination_steps - steps_per_phase
        )

        return network.measure()
