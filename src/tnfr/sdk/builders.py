"""Builder patterns for common TNFR experiment workflows.

This module provides builder pattern implementations for standard TNFR
experiments. Builders offer more control than templates while still
simplifying common research patterns.

Examples
--------
Run a small-world network study:

>>> from tnfr.sdk import TNFRExperimentBuilder
>>> results = TNFRExperimentBuilder.small_world_study(
...     nodes=50, rewiring_prob=0.1, steps=10
... )

Compare different network topologies:

>>> comparison = TNFRExperimentBuilder.compare_topologies(
...     node_count=40, steps=10
... )
>>> for topology, results in comparison.items():
...     print(f"{topology}: coherence={results.coherence:.3f}")
"""

from __future__ import annotations

from typing import Dict, Optional

from .fluent import TNFRNetwork, NetworkResults

__all__ = ["TNFRExperimentBuilder"]


class TNFRExperimentBuilder:
    """Builder pattern for standard TNFR experiments.

    This class provides static methods that implement common experimental
    patterns in TNFR research. Each method configures and runs a complete
    experiment, returning structured results for analysis.

    Builders are more flexible than templates, allowing researchers to
    control specific parameters while handling boilerplate setup.
    """

    @staticmethod
    def small_world_study(
        nodes: int = 50,
        rewiring_prob: float = 0.1,
        steps: int = 10,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Study small-world network properties with TNFR dynamics.

        Creates a Watts-Strogatz small-world network and evolves it through
        TNFR operator sequences to study how small-world topology affects
        coherence and synchronization.

        Parameters
        ----------
        nodes : int, default=50
            Number of nodes in the network.
        rewiring_prob : float, default=0.1
            Rewiring probability for small-world construction.
        steps : int, default=10
            Number of activation steps to apply.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Complete results including coherence and sense indices.

        Examples
        --------
        >>> results = TNFRExperimentBuilder.small_world_study(
        ...     nodes=100, rewiring_prob=0.2
        ... )
        >>> print(f"Network coherence: {results.coherence:.3f}")
        """
        network = TNFRNetwork("small_world_study")
        if random_seed is not None:
            network._config.random_seed = random_seed

        return (
            network.add_nodes(nodes)
            .connect_nodes(rewiring_prob, "small_world")
            .apply_sequence("basic_activation", repeat=steps)
            .measure()
        )

    @staticmethod
    def synchronization_study(
        nodes: int = 30,
        coupling_strength: float = 0.5,
        steps: int = 20,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Study synchronization in densely coupled TNFR networks.

        Creates a network with similar structural frequencies (within TNFR
        bounds) and dense coupling, then applies synchronization sequences
        to study phase locking and coherence emergence.

        Parameters
        ----------
        nodes : int, default=30
            Number of nodes in the network.
        coupling_strength : float, default=0.5
            Connection probability (controls coupling density).
        steps : int, default=20
            Number of synchronization steps.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results showing synchronization metrics.

        Examples
        --------
        >>> results = TNFRExperimentBuilder.synchronization_study(
        ...     nodes=50, coupling_strength=0.7
        ... )
        >>> avg_si = sum(results.sense_indices.values()) / len(results.sense_indices)
        >>> print(f"Synchronization (avg Si): {avg_si:.3f}")
        """
        network = TNFRNetwork("sync_study")
        if random_seed is not None:
            network._config.random_seed = random_seed

        # Similar frequencies promote synchronization (within bounds: 0.6-0.9)
        network.add_nodes(nodes, vf_range=(0.6, 0.9))
        network.connect_nodes(coupling_strength, "random")

        # Multi-phase synchronization protocol
        for step in range(steps):
            if step < 5:
                # Initial activation
                network.apply_sequence("basic_activation")
            elif step < 15:
                # Synchronization phase
                network.apply_sequence("network_sync")
            else:
                # Consolidation
                network.apply_sequence("consolidation")

        return network.measure()

    @staticmethod
    def creativity_emergence(
        nodes: int = 20,
        mutation_intensity: float = 0.3,
        steps: int = 15,
        random_seed: Optional[int] = None,
    ) -> NetworkResults:
        """Study creative emergence through controlled mutation.

        Models creative processes by starting with diverse frequencies
        and applying mutation operators to study how new coherent forms
        emerge from structural reorganization.

        Parameters
        ----------
        nodes : int, default=20
            Number of nodes (ideas/concepts).
        mutation_intensity : float, default=0.3
            Not currently used, reserved for future mutation parameters.
        steps : int, default=15
            Number of creative mutation cycles.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NetworkResults
            Results showing creative coherence emergence.

        Examples
        --------
        >>> results = TNFRExperimentBuilder.creativity_emergence(nodes=25)
        >>> print(f"Creative coherence: {results.coherence:.3f}")
        """
        network = TNFRNetwork("creativity_study")
        if random_seed is not None:
            network._config.random_seed = random_seed

        return (
            network.add_nodes(nodes, vf_range=(0.2, 0.8))  # High diversity
            .connect_nodes(0.2, "ring")  # Conservative connectivity
            .apply_sequence("creative_mutation", repeat=steps)
            .measure()
        )

    @staticmethod
    def compare_topologies(
        node_count: int = 40,
        steps: int = 10,
        topologies: Optional[list[str]] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, NetworkResults]:
        """Compare TNFR dynamics across different network topologies.

        Creates multiple networks with identical node properties but
        different topological structures, then compares their evolution
        under the same operator sequences.

        Parameters
        ----------
        node_count : int, default=40
            Number of nodes in each network.
        steps : int, default=10
            Number of activation steps to apply.
        topologies : list[str], optional
            List of topologies to compare. If None, uses
            ["random", "ring", "small_world"].
        random_seed : int, optional
            Random seed for reproducibility across all networks.

        Returns
        -------
        Dict[str, NetworkResults]
            Dictionary mapping topology names to their results.

        Examples
        --------
        >>> comparison = TNFRExperimentBuilder.compare_topologies(
        ...     node_count=50, steps=15
        ... )
        >>> for topo, res in comparison.items():
        ...     print(f"{topo}: C(t)={res.coherence:.3f}")
        """
        if topologies is None:
            topologies = ["random", "ring", "small_world"]

        results = {}

        for topology in topologies:
            network = TNFRNetwork(f"topology_study_{topology}")
            if random_seed is not None:
                network._config.random_seed = random_seed

            network.add_nodes(node_count)
            network.connect_nodes(0.3, topology)
            network.apply_sequence("basic_activation", repeat=steps)

            results[topology] = network.measure()

        return results

    @staticmethod
    def phase_transition_study(
        nodes: int = 50,
        initial_coupling: float = 0.1,
        final_coupling: float = 0.9,
        steps_per_level: int = 5,
        coupling_levels: int = 5,
        random_seed: Optional[int] = None,
    ) -> Dict[float, NetworkResults]:
        """Study phase transitions by varying coupling strength.

        Investigates how network coherence changes as coupling strength
        increases, potentially revealing critical phase transitions in
        TNFR network dynamics.

        Parameters
        ----------
        nodes : int, default=50
            Number of nodes in the network.
        initial_coupling : float, default=0.1
            Starting coupling strength.
        final_coupling : float, default=0.9
            Final coupling strength.
        steps_per_level : int, default=5
            Number of evolution steps at each coupling level.
        coupling_levels : int, default=5
            Number of coupling levels to test.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[float, NetworkResults]
            Mapping from coupling strength to network results.

        Examples
        --------
        >>> transition = TNFRExperimentBuilder.phase_transition_study(nodes=60)
        >>> for coupling, res in sorted(transition.items()):
        ...     print(f"Coupling {coupling:.2f}: C(t)={res.coherence:.3f}")
        """
        import numpy as np

        coupling_values = np.linspace(initial_coupling, final_coupling, coupling_levels)
        results = {}

        for coupling in coupling_values:
            network = TNFRNetwork(f"phase_study_{coupling:.2f}")
            if random_seed is not None:
                network._config.random_seed = random_seed

            network.add_nodes(nodes)
            network.connect_nodes(float(coupling), "random")
            network.apply_sequence("network_sync", repeat=steps_per_level)

            results[float(coupling)] = network.measure()

        return results

    @staticmethod
    def resilience_study(
        nodes: int = 40,
        initial_steps: int = 10,
        perturbation_steps: int = 5,
        recovery_steps: int = 10,
        random_seed: Optional[int] = None,
    ) -> Dict[str, NetworkResults]:
        """Study network resilience to perturbations.

        Establishes a stable network, applies dissonance perturbations,
        then measures recovery through stabilization sequences. Reveals
        network resilience properties.

        Parameters
        ----------
        nodes : int, default=40
            Number of nodes in the network.
        initial_steps : int, default=10
            Steps to establish initial stable state.
        perturbation_steps : int, default=5
            Steps of dissonance perturbation.
        recovery_steps : int, default=10
            Steps to observe recovery.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, NetworkResults]
            Results at 'initial', 'perturbed', and 'recovered' states.

        Examples
        --------
        >>> resilience = TNFRExperimentBuilder.resilience_study(nodes=50)
        >>> initial_c = resilience['initial'].coherence
        >>> recovered_c = resilience['recovered'].coherence
        >>> print(f"Recovery: {recovered_c / initial_c:.1%}")
        """
        network = TNFRNetwork("resilience_study")
        if random_seed is not None:
            network._config.random_seed = random_seed

        results = {}

        # Phase 1: Establish stable network
        network.add_nodes(nodes)
        network.connect_nodes(0.3, "small_world")
        network.apply_sequence("stabilization", repeat=initial_steps)
        results["initial"] = network.measure()

        # Phase 2: Apply perturbation
        network.apply_sequence("creative_mutation", repeat=perturbation_steps)
        results["perturbed"] = network.measure()

        # Phase 3: Recovery
        network.apply_sequence("stabilization", repeat=recovery_steps)
        results["recovered"] = network.measure()

        return results
