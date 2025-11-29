"""Fluent API for simplified TNFR network creation and simulation.

This module implements the core :class:`TNFRNetwork` class that provides
a user-friendly, chainable interface for working with TNFR networks. The
API hides low-level complexity while maintaining full theoretical fidelity
to TNFR invariants and structural operators.

Examples
--------
Create and simulate a simple TNFR network:

>>> network = TNFRNetwork("my_experiment")
>>> network.add_nodes(10).connect_nodes(0.3, "random")
>>> network.apply_sequence("basic_activation", repeat=5)
>>> results = network.measure()
>>> print(results.summary())

Chain operations for rapid prototyping:

>>> results = (TNFRNetwork("quick_test")
...            .add_nodes(20)
...            .connect_nodes(0.4, "ring")
...            .apply_sequence("network_sync", repeat=10)
...            .measure())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

import networkx as nx

from ..structural import create_nfr, run_sequence
from ..metrics.coherence import compute_coherence
from ..metrics.sense_index import compute_Si
from ..constants.aliases import ALIAS_DNFR
from ..validation import validate_sequence

# Auto-optimization imports (NEW - Nov 28, 2025)
try:
    from ..physics.fields import (
        auto_optimize_field_computation,
        analyze_optimization_potential,
        recommend_field_optimization_strategy
    )
    from ..dynamics.self_optimizing_engine import TNFRSelfOptimizingEngine, OptimizationObjective
    _AUTO_OPTIMIZATION_AVAILABLE = True
except ImportError:
    auto_optimize_field_computation = None
    analyze_optimization_potential = None
    recommend_field_optimization_strategy = None
    TNFRSelfOptimizingEngine = None
    OptimizationObjective = None
    _AUTO_OPTIMIZATION_AVAILABLE = False

__all__ = ["TNFRNetwork", "NetworkConfig", "NetworkResults"]


# Predefined operator sequences for common patterns (optimized with Grammar 2.0)
# All sequences must respect TNFR grammar rules:
# - Start with emission or recursivity
# - Include reception→coherence segment
# - Include coupling/dissonance/resonance segment
# - End with recursivity, silence, or transition
#
# Sequences optimized for structural health ≥ 0.7 using Grammar 2.0:
# - Balanced stabilizers/destabilizers
# - Harmonic frequency transitions
# - Proper closure operators
# - Pattern completeness
NAMED_SEQUENCES = {
    # Basic activation pattern - optimized with expansion for balance
    # Health: 0.79 (good) - Pattern: activation
    # Includes controlled expansion for structural balance
    "basic_activation": [
        "emission",  # AL: Initiate coherent structure
        "reception",  # EN: Stabilize incoming energy
        "coherence",  # IL: Primary stabilization (required)
        "expansion",  # VAL: Controlled growth (balance +0.33)
        "resonance",  # RA: Amplify coherent structure
        "silence",  # SHA: Sustainable pause state
    ],
    # Stabilization with expansion - optimized for regenerative cycles
    # Health: 0.76 (good) - Pattern: regenerative
    # Enables recursive consolidation with controlled expansion
    "stabilization": [
        "emission",  # AL: Initiate structure
        "reception",  # EN: Gather information
        "coherence",  # IL: Stabilize
        "expansion",  # VAL: Controlled growth (balance +0.50)
        "resonance",  # RA: Amplify through network
        "recursivity",  # REMESH: Enable fractal recursion
    ],
    # Creative mutation - corrected for OZ→IL physics
    # Pattern: stabilized transformation with controlled mutation
    "creative_mutation": [
        "emission",     # AL: Initiate exploration
        "coherence",    # IL: Establish stable base
        "dissonance",   # OZ: Introduce creative tension
        "mutation",     # ZHIR: Phase transformation (after OZ)
        "coherence",    # IL: Stabilize new state
        "resonance",    # RA: Amplify new patterns
        "silence",      # SHA: Integration pause
    ],
    # Network synchronization - optimized with transition for regenerative capability
    # Health: 0.77 (good) - Pattern: regenerative
    # Enables phase synchronization across multi-node networks with dissonance for balance
    "network_sync": [
        "emission",  # AL: Initiate network activity
        "reception",  # EN: Gather network state
        "coherence",  # IL: Stabilize local structure
        "coupling",  # UM: Establish phase synchronization
        "resonance",  # RA: Propagate through network
        "transition",  # NAV: Enable regenerative cycles (changed from silence)
    ],
    # Exploration - corrected for physics compliance
    # Pattern: stable exploration with controlled discovery
    "exploration": [
        "emission",     # AL: Begin exploration
        "reception",    # EN: Sense environment
        "coherence",    # IL: Stabilize base
        "dissonance",   # OZ: Introduce exploration tension
        "resonance",    # RA: Amplify discoveries
        "transition",   # NAV: Navigate to new state
    ],
    # Consolidation - optimized with expansion for structural balance
    # Health: 0.80 (good) - Pattern: stabilization
    # Recursive consolidation with controlled expansion
    "consolidation": [
        "recursivity",  # REMESH: Start from fractal structure
        "reception",  # EN: Gather current state
        "coherence",  # IL: Consolidate structure
        "expansion",  # VAL: Controlled growth (balance +0.25)
        "resonance",  # RA: Amplify consolidated state
        "coherence",  # IL: Re-stabilize after expansion
        "silence",  # SHA: Sustained stable state
    ],
    
    # === NUEVAS SECUENCIAS AVANZADAS ===
    
    # Healing pattern - for network recovery and coherence restoration
    "healing": [
        "emission",     # AL: Restart from clean state
        "reception",    # EN: Gather current network state
        "coherence",    # IL: Primary stabilization
        "coupling",     # UM: Restore connections
        "resonance",    # RA: Harmonize network
        "coherence",    # IL: Final stabilization
        "silence",      # SHA: Sustained healing state
    ],
    
    # Amplification - for boosting network activity and propagation
    "amplification": [
        "emission",     # AL: Initiate high-energy state
        "coupling",     # UM: Establish strong links
        "resonance",    # RA: Primary amplification
        "coherence",    # IL: Stabilize amplified state
        "expansion",    # VAL: Controlled growth
        "resonance",    # RA: Secondary amplification
        "transition",   # NAV: Navigate to sustained state
    ],
    
    # Deep transformation - for major structural changes
    "deep_transformation": [
        "emission",        # AL: Fresh start
        "coherence",       # IL: Stable foundation
        "expansion",       # VAL: Increase complexity
        "coherence",       # IL: Stabilize expansion
        "dissonance",      # OZ: Transformation trigger
        "mutation",        # ZHIR: Major phase change
        "coherence",       # IL: Integrate transformation
        "resonance",       # RA: Propagate new patterns
        "silence",         # SHA: Consolidate changes
    ],
    
    # Network harmonization - for multi-node synchronization
    "harmonization": [
        "recursivity",     # REMESH: Multi-scale start
        "reception",       # EN: Sense all nodes
        "coherence",       # IL: Local stabilization
        "coupling",        # UM: Phase synchronization
        "resonance",       # RA: Network-wide harmonics
        "coherence",       # IL: Lock in harmony
        "transition",      # NAV: Maintain sync state
    ],
    
    # Progressive learning - for adaptive behavior development
    "progressive_learning": [
        "emission",        # AL: New learning episode
        "reception",       # EN: Gather information
        "coherence",       # IL: Initial understanding
        "expansion",       # VAL: Broaden perspective
        "coupling",        # UM: Connect concepts
        "resonance",       # RA: Reinforce learning
        "recursivity",     # REMESH: Multi-level integration
    ],
    
    # Recovery pattern - for system restoration after disruption
    "recovery": [
        "emission",        # AL: Restart core functions (U1a generator)
        "reception",       # EN: Assess current state
        "coherence",       # IL: Restore stability
        "coupling",        # UM: Rebuild connections
        "resonance",       # RA: Stabilize recovery
        "transition",      # NAV: Return to normal operation
    ],
    
    # Innovation catalyst - for breakthrough discovery patterns
    "innovation": [
        "transition",      # NAV: Shift to exploration mode
        "reception",       # EN: Gather diverse inputs
        "coherence",       # IL: Stable processing base
        "dissonance",      # OZ: Creative tension
        "mutation",        # ZHIR: Breakthrough transformation
        "coherence",       # IL: Stabilize new insights (required after ZHIR)
        "resonance",       # RA: Amplify stabilized innovation
        "recursivity",     # REMESH: Scale innovation
    ],
    
    # Optimization cycle - for continuous improvement
    "optimization": [
        "recursivity",     # REMESH: Analyze current patterns
        "reception",       # EN: Gather performance data
        "coherence",       # IL: Stabilize baseline
        "expansion",       # VAL: Explore improvements
        "contraction",     # NUL: Focus on best options
        "coherence",       # IL: Optimize selection
        "transition",      # NAV: Implement improvements
    ],
    
    # Fractal growth - for self-similar expansion across scales
    "fractal_growth": [
        "emission",        # AL: Seed pattern
        "coherence",       # IL: Stabilize core
        "expansion",       # VAL: First scale growth
        "coherence",       # IL: Stabilize expansion (required after VAL)
        "recursivity",     # REMESH: Self-similar replication
        "coupling",        # UM: Connect scales
        "resonance",       # RA: Harmonize across levels
        "silence",         # SHA: Sustained fractal state
    ],
}


@dataclass
class NetworkConfig:
    """Configuration for TNFR network creation.

    Parameters
    ----------
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility.
    validate_invariants : bool, default=True
        Whether to validate TNFR invariants after operator application.
    auto_stabilization : bool, default=True
        Whether to automatically apply stabilization after mutations.
    default_vf_range : tuple[float, float], default=(0.1, 1.0)
        Default range for structural frequency (νf) generation in Hz_str.
    default_epi_range : tuple[float, float], default=(0.1, 0.9)
        Default range for EPI (Primary Information Structure) generation.
    """

    random_seed: Optional[int] = None
    validate_invariants: bool = True
    auto_stabilization: bool = True
    default_vf_range: tuple[float, float] = (0.1, 1.0)
    default_epi_range: tuple[float, float] = (0.1, 0.9)


@dataclass
class NetworkResults:
    """Results from TNFR network simulation.

    Attributes
    ----------
    coherence : float
        Global coherence C(t) of the network.
    sense_indices : Dict[str, float]
        Sense index Si for each node, measuring stable reorganization capacity.
    delta_nfr : Dict[str, float]
        Internal reorganization gradient ΔNFR for each node.
    graph : TNFRGraph
        The underlying NetworkX graph with full TNFR state.
    avg_vf : float, optional
        Average structural frequency across all nodes.
    avg_phase : float, optional
        Average phase angle across all nodes.
    """

    coherence: float
    sense_indices: Dict[str, float]
    delta_nfr: Dict[str, float]
    graph: Any  # TNFRGraph
    avg_vf: Optional[float] = None
    avg_phase: Optional[float] = None
    unified_fields: Optional[Dict[str, Any]] = None  # Nov 28, 2025 - unified field telemetry

    def summary(self) -> str:
        """Generate human-readable summary of network results.

        Returns
        -------
        str
            Formatted summary string with key metrics.
        """
        si_values = list(self.sense_indices.values())
        dnfr_values = list(self.delta_nfr.values())

        avg_si = sum(si_values) / len(si_values) if si_values else 0.0
        avg_dnfr = sum(dnfr_values) / len(dnfr_values) if dnfr_values else 0.0

        # Unified fields summary (if available)
        unified_summary = ""
        if self.unified_fields:
            try:
                cf = self.unified_fields.get("complex_field", {})
                correlation = cf.get("correlation", 0.0)
                ti = self.unified_fields.get("tensor_invariants", {})
                conservation = ti.get("conservation_quality", 0.0)
                unified_summary = f"""
  • K_φ ↔ J_φ Correlation: {correlation:.3f}
  • Conservation Quality: {conservation:.3f}"""
            except Exception:
                pass

        return f"""
TNFR Network Results:
  • Coherence C(t): {self.coherence:.3f}
  • Nodes: {len(self.sense_indices)}
  • Avg Sense Index Si: {avg_si:.3f}
  • Avg ΔNFR: {avg_dnfr:.3f}
  • Avg νf: {self.avg_vf:.3f} Hz_str{' (computed)' if self.avg_vf else ''}
  • Avg Phase: {self.avg_phase:.3f} rad{' (computed)' if self.avg_phase else ''}{unified_summary}
""".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to serializable dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all metrics and summary statistics.
        """
        si_values = list(self.sense_indices.values())
        dnfr_values = list(self.delta_nfr.values())

        return {
            "coherence": self.coherence,
            "sense_indices": self.sense_indices,
            "delta_nfr": self.delta_nfr,
            "summary_stats": {
                "node_count": len(self.sense_indices),
                "avg_si": sum(si_values) / len(si_values) if si_values else 0.0,
                "avg_delta_nfr": (sum(dnfr_values) / len(dnfr_values) if dnfr_values else 0.0),
                "avg_vf": self.avg_vf,
                "avg_phase": self.avg_phase,
            },
        }


class TNFRNetwork:
    """Fluent API for creating and simulating TNFR networks.

    This class provides a simplified, chainable interface for working with
    TNFR networks. All methods return ``self`` to enable method chaining.
    The API automatically handles node creation, operator validation, and
    metric computation while preserving TNFR structural invariants.

    Parameters
    ----------
    name : str, default="tnfr_network"
        Name identifier for the network.
    config : NetworkConfig, optional
        Configuration settings. If None, uses default configuration.

    Examples
    --------
    Create a network with fluent interface:

    >>> network = TNFRNetwork("experiment_1")
    >>> network.add_nodes(15).connect_nodes(0.25, "random")
    >>> network.apply_sequence("basic_activation", repeat=3)
    >>> results = network.measure()

    One-line network creation and simulation:

    >>> results = (TNFRNetwork("test")
    ...            .add_nodes(10)
    ...            .connect_nodes(0.3)
    ...            .apply_sequence("network_sync", repeat=5)
    ...            .measure())
    """

    def __init__(self, name: str = "tnfr_network", config: Optional[NetworkConfig] = None):
        """Initialize TNFR network with given name and configuration."""
        self.name = name
        self._config = config if config is not None else NetworkConfig()
        self._graph: Optional[nx.Graph] = None
        self._results: Optional[NetworkResults] = None
        self._node_counter = 0

        # Initialize RNG if seed provided
        if _HAS_NUMPY and self._config.random_seed is not None:
            self._rng = np.random.RandomState(self._config.random_seed)
        else:
            import random

            if self._config.random_seed is not None:
                random.seed(self._config.random_seed)
            self._rng = random  # type: ignore[assignment]

    def add_nodes(
        self,
        count: int,
        vf_range: Optional[tuple[float, float]] = None,
        epi_range: Optional[tuple[float, float]] = None,
        phase_range: tuple[float, float] = (0.0, 6.283185307179586),  # (0, 2π)
        random_seed: Optional[int] = None,
    ) -> TNFRNetwork:
        """Add nodes with random TNFR properties within valid ranges.

        Creates ``count`` new nodes with structural properties (EPI, νf, phase)
        randomly sampled within specified ranges. All values respect TNFR
        invariants and canonical units (νf in Hz_str, phase in radians).

        Parameters
        ----------
        count : int
            Number of nodes to create.
        vf_range : tuple[float, float], optional
            Range for structural frequency νf in Hz_str. If None, uses
            config default. Values should be positive.
        epi_range : tuple[float, float], optional
            Range for Primary Information Structure (EPI). If None, uses
            config default. Typical range is (0.1, 0.9) to avoid extremes.
        phase_range : tuple[float, float], default=(0, 2π)
            Range for phase in radians. Default covers full circle.
        random_seed : int, optional
            Override seed for this operation only. If None, uses network seed.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Examples
        --------
        Add nodes with default properties:

        >>> network = TNFRNetwork().add_nodes(5)

        Add nodes with custom frequency range:

        >>> network = TNFRNetwork().add_nodes(10, vf_range=(0.5, 2.0))
        """
        if self._graph is None:
            self._graph = nx.Graph()

        if vf_range is None:
            vf_range = self._config.default_vf_range
        if epi_range is None:
            epi_range = self._config.default_epi_range

        # Setup RNG for this operation
        if _HAS_NUMPY:
            rng = np.random.RandomState(random_seed) if random_seed is not None else self._rng

            for _ in range(count):
                node_id = f"node_{self._node_counter}"
                self._node_counter += 1

                # Generate valid TNFR properties
                vf = rng.uniform(*vf_range)
                phase = rng.uniform(*phase_range)
                epi = rng.uniform(*epi_range)

                # Create NFR node with structural properties
                self._graph, _ = create_nfr(
                    node_id,
                    graph=self._graph,
                    vf=vf,
                    theta=phase,
                    epi=epi,
                )
        else:
            # Fallback to standard random
            import random

            if random_seed is not None:
                random.seed(random_seed)

            for _ in range(count):
                node_id = f"node_{self._node_counter}"
                self._node_counter += 1

                vf = random.uniform(*vf_range)
                phase = random.uniform(*phase_range)
                epi = random.uniform(*epi_range)

                self._graph, _ = create_nfr(
                    node_id,
                    graph=self._graph,
                    vf=vf,
                    theta=phase,
                    epi=epi,
                )

        return self

    def connect_nodes(
        self,
        connection_probability: float = 0.3,
        connection_pattern: str = "random",
    ) -> TNFRNetwork:
        """Connect nodes according to specified topology pattern.

        Establishes coupling between nodes using common network patterns.
        Connections enable resonance and phase synchronization between nodes.

        Parameters
        ----------
        connection_probability : float, default=0.3
            For random pattern: probability of edge between any two nodes.
            For small_world: rewiring probability (Watts-Strogatz model).
        connection_pattern : str, default="random"
            Topology pattern to use. Options:
            - "random": Erdős-Rényi random graph
            - "ring": Each node connects to next in circle
            - "small_world": Watts-Strogatz small-world network

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If graph has no nodes or invalid pattern specified.

        Examples
        --------
        Create random network:

        >>> network = TNFRNetwork().add_nodes(10).connect_nodes(0.3, "random")

        Create ring lattice:

        >>> network = TNFRNetwork().add_nodes(15).connect_nodes(pattern="ring")

        Create small-world network:

        >>> network = TNFRNetwork().add_nodes(20).connect_nodes(0.1, "small_world")
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No nodes in graph. Call add_nodes() first.")

        nodes = list(self._graph.nodes())

        if connection_pattern == "random":
            # Erdős-Rényi random graph
            if _HAS_NUMPY:
                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1 :]:
                        if self._rng.random() < connection_probability:
                            self._graph.add_edge(node1, node2)
            else:
                import random

                for i, node1 in enumerate(nodes):
                    for node2 in nodes[i + 1 :]:
                        if random.random() < connection_probability:
                            self._graph.add_edge(node1, node2)

        elif connection_pattern == "ring":
            # Ring lattice
            for i in range(len(nodes)):
                next_node = nodes[(i + 1) % len(nodes)]
                self._graph.add_edge(nodes[i], next_node)

        elif connection_pattern == "small_world":
            # Watts-Strogatz small-world network
            # Start with ring lattice, then rewire
            k = max(4, int(len(nodes) * 0.1))  # ~10% degree, minimum 4

            # Create initial ring with k nearest neighbors
            for i in range(len(nodes)):
                for j in range(1, k // 2 + 1):
                    target = (i + j) % len(nodes)
                    if nodes[i] != nodes[target]:  # Avoid self-loops
                        self._graph.add_edge(nodes[i], nodes[target])

            # Rewire edges with given probability
            edges = list(self._graph.edges())
            if _HAS_NUMPY:
                for u, v in edges:
                    if self._rng.random() < connection_probability:
                        # Remove edge and create new random edge
                        self._graph.remove_edge(u, v)
                        # Find node not already connected
                        candidates = [n for n in nodes if n != u and not self._graph.has_edge(u, n)]
                        if candidates:
                            idx = int(self._rng.randint(0, len(candidates)))
                            if idx >= len(candidates):
                                idx = len(candidates) - 1
                            w = candidates[idx]
                            self._graph.add_edge(u, w)
            else:
                import random

                for u, v in edges:
                    if random.random() < connection_probability:
                        self._graph.remove_edge(u, v)
                        candidates = [n for n in nodes if n != u and not self._graph.has_edge(u, n)]
                        if candidates:
                            w = random.choice(candidates)
                            self._graph.add_edge(u, w)

        else:
            available = ", ".join(["random", "ring", "small_world"])
            raise ValueError(
                f"Unknown connection pattern '{connection_pattern}'. " f"Available: {available}"
            )

        return self

    def apply_sequence(
        self,
        sequence: Union[str, List[str]],
        repeat: int = 1,
    ) -> TNFRNetwork:
        """Apply structural operator sequence to evolve the network.

        Executes a validated sequence of TNFR operators that reorganize
        network structure according to the nodal equation ∂EPI/∂t = νf·ΔNFR(t).
        Sequences can be predefined names or custom operator lists. The
        sequence is applied to all nodes in the network.

        Parameters
        ----------
        sequence : str or List[str]
            Either a predefined sequence name or list of operator names.
            Predefined sequences:
            - "basic_activation": [emission, reception, coherence, resonance, silence]
            - "stabilization": [emission, reception, coherence, resonance, recursivity]
            - "creative_mutation": [emission, dissonance, reception, coherence, mutation, resonance, silence]
            - "network_sync": [emission, reception, coherence, coupling, resonance, silence]
            - "exploration": [emission, dissonance, reception, coherence, resonance, transition]
            - "consolidation": [recursivity, reception, coherence, resonance, silence]
        repeat : int, default=1
            Number of times to apply the sequence.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If graph has no nodes or sequence name is invalid.

        Examples
        --------
        Apply predefined sequence:

        >>> network = (TNFRNetwork()
        ...            .add_nodes(10)
        ...            .connect_nodes(0.3)
        ...            .apply_sequence("basic_activation", repeat=5))

        Apply custom operator sequence:

        >>> network.apply_sequence(["emission", "reception", "coherence", "resonance", "silence"])
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No nodes in graph. Call add_nodes() first.")

        # Expand named sequences
        if isinstance(sequence, str):
            if sequence not in NAMED_SEQUENCES:
                available = ", ".join(sorted(NAMED_SEQUENCES.keys()))
                raise ValueError(f"Unknown sequence '{sequence}'. Available: {available}")
            operator_list = NAMED_SEQUENCES[sequence]
        else:
            operator_list = sequence

        # Validate sequence if configured
        if self._config.validate_invariants:
            validate_sequence(operator_list)

        # Convert operator names to operator instances
        from ..operators.registry import get_operator_class

        operator_instances = [get_operator_class(name)() for name in operator_list]

        # Apply sequence repeatedly to all nodes
        for _ in range(repeat):
            for node in list(self._graph.nodes()):
                run_sequence(self._graph, node, operator_instances)

        return self
    
    def apply_adaptive_sequence(self, base_sequence: str = "basic_activation", target_coherence: float = 0.7) -> "TNFRNetwork":
        """Apply adaptive sequence selection based on network state.
        
        Automatically selects and applies the most appropriate sequence based on
        current network coherence and structure.
        
        Parameters
        ----------
        base_sequence : str, default="basic_activation"
            Fallback sequence if adaptive selection fails.
        target_coherence : float, default=0.7
            Desired coherence level for adaptive optimization.
            
        Returns
        -------
        TNFRNetwork
            Self for method chaining.
        """
        if not self._graph.nodes:
            return self.apply_sequence(base_sequence)
            
        # Measure current state
        current_coherence = compute_coherence(self._graph)
        
        # Adaptive sequence selection logic
        if current_coherence < 0.3:
            # Low coherence - use healing
            selected_sequence = "healing"
        elif current_coherence < 0.5:
            # Medium coherence - use stabilization
            selected_sequence = "stabilization"
        elif current_coherence < target_coherence:
            # Good coherence but below target - use harmonization
            selected_sequence = "harmonization"
        elif current_coherence >= 0.8:
            # High coherence - ready for innovation or optimization
            selected_sequence = "optimization" if len(self._graph.nodes) > 5 else "innovation"
        else:
            # Default case
            selected_sequence = base_sequence
            
        return self.apply_sequence(selected_sequence)
    
    def apply_sequence_chain(self, chain_pattern: str = "standard") -> "TNFRNetwork":
        """Apply predefined chains of sequences for complex behaviors.
        
        Parameters
        ----------
        chain_pattern : str, default="standard"
            Type of sequence chain: "standard", "learning", "innovation", "recovery"
            
        Returns
        -------
        TNFRNetwork
            Self for method chaining.
        """
        chains = {
            "standard": [
                ("basic_activation", None, 1),
                ("harmonization", None, 1),
                ("optimization", None, 1)
            ],
            "learning": [
                ("basic_activation", None, 1),
                ("progressive_learning", None, 2),
                ("consolidation", None, 1)
            ],
            "innovation": [
                ("stabilization", None, 1),
                ("innovation", None, 1),
                ("amplification", None, 1),
                ("consolidation", None, 1)
            ],
            "recovery": [
                ("recovery", None, 1),
                ("healing", None, 2),
                ("harmonization", None, 1)
            ],
            "transformation": [
                ("stabilization", None, 1),
                ("deep_transformation", None, 1),
                ("healing", None, 1),
                ("consolidation", None, 1)
            ]
        }
        
        if chain_pattern not in chains:
            available = ", ".join(chains.keys())
            raise ValueError(f"Unknown chain pattern '{chain_pattern}'. Available: {available}")
            
        return self.apply_multiple_sequences(chains[chain_pattern])

    def measure(self) -> NetworkResults:
        """Calculate TNFR metrics and return structured results.

        Computes coherence C(t), sense indices Si, and ΔNFR values for
        all nodes, plus aggregate statistics. Results are cached internally
        and returned as a :class:`NetworkResults` instance.

        Returns
        -------
        NetworkResults
            Structured container with all computed metrics.

        Raises
        ------
        ValueError
            If no network has been created.

        Examples
        --------
        Measure and display results:

        >>> results = network.measure()
        >>> print(results.summary())

        Access specific metrics:

        >>> coherence = results.coherence
        >>> si_values = results.sense_indices
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")

        # Compute coherence C(t)
        coherence = compute_coherence(self._graph)

        # Compute sense indices Si for all nodes
        si_dict = compute_Si(self._graph, inplace=False)

        # Extract ΔNFR values
        delta_nfr_dict = {}
        for node_id in self._graph.nodes():
            delta_nfr_dict[node_id] = self._graph.nodes[node_id].get(ALIAS_DNFR, 0.0)

        # Compute aggregate statistics
        vf_sum = 0.0
        phase_sum = 0.0
        node_count = self._graph.number_of_nodes()

        for node_id in self._graph.nodes():
            node_data = self._graph.nodes[node_id]
            vf_sum += node_data.get("nu_f", 0.0)
            phase_sum += node_data.get("phase", 0.0)

        avg_vf = vf_sum / node_count if node_count > 0 else 0.0
        avg_phase = phase_sum / node_count if node_count > 0 else 0.0

        # Unified field telemetry (Nov 28, 2025 - comprehensive audit integration)
        unified_telemetry = {}
        try:
            from ..physics.fields import compute_unified_telemetry
            unified_telemetry = compute_unified_telemetry(self._graph)
        except (ImportError, Exception):
            # Graceful degradation if unified fields not available
            pass

        # Create and cache results
        self._results = NetworkResults(
            coherence=coherence,
            sense_indices=si_dict,
            delta_nfr=delta_nfr_dict,
            graph=self._graph,
            avg_vf=avg_vf,
            avg_phase=avg_phase,
            unified_fields=unified_telemetry,  # New field for unified telemetry
        )

        return self._results

    def apply_canonical_sequence(
        self,
        sequence_name: str,
        node: Optional[int] = None,
        collect_metrics: bool = True,
    ) -> TNFRNetwork:
        """Apply a canonical predefined operator sequence from TNFR theory.

        Executes one of the 6 archetypal sequences involving OZ (Dissonance)
        from "El pulso que nos atraviesa" (Table 2.5). These sequences represent
        validated structural patterns with documented use cases and domain contexts.

        Parameters
        ----------
        sequence_name : str
            Name of canonical sequence. Available sequences:
            - 'bifurcated_base': OZ → ZHIR (mutation path)
            - 'bifurcated_collapse': OZ → NUL (collapse path)
            - 'therapeutic_protocol': Complete healing cycle
            - 'theory_system': Epistemological construction
            - 'full_deployment': Complete reorganization trajectory
            - 'mod_stabilizer': OZ → ZHIR → IL (reusable macro)
        node : int, optional
            Target node ID. If None, applies to the most recently added node.
        collect_metrics : bool, default=True
            Whether to collect detailed operator metrics during execution.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If sequence_name is not recognized or network has no nodes.

        Examples
        --------
        Apply therapeutic protocol:

        >>> net = TNFRNetwork("therapy_session")
        >>> net.add_nodes(1).apply_canonical_sequence("therapeutic_protocol")
        >>> results = net.measure()
        >>> print(f"Coherence: {results.coherence:.3f}")

        Apply MOD_STABILIZER as reusable transformation module:

        >>> net = TNFRNetwork("modular")
        >>> net.add_nodes(1)
        >>> net.apply_canonical_sequence("mod_stabilizer").measure()

        See Also
        --------
        list_canonical_sequences : List available sequences with filters
        apply_sequence : Apply predefined or custom operator sequences

        Notes
        -----
        Canonical sequences are archetypal patterns from TNFR theory documented
        in "El pulso que nos atraviesa", Tabla 2.5. Each sequence has been
        validated for structural coherence and grammar compliance.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No nodes in graph. Call add_nodes() first.")

        # Import canonical sequences registry
        from ..operators.canonical_patterns import CANONICAL_SEQUENCES

        if sequence_name not in CANONICAL_SEQUENCES:
            available = ", ".join(sorted(CANONICAL_SEQUENCES.keys()))
            raise ValueError(
                f"Unknown canonical sequence '{sequence_name}'. " f"Available: {available}"
            )

        sequence = CANONICAL_SEQUENCES[sequence_name]

        # Determine target node
        if node is None:
            # Use last added node
            nodes_list = list(self._graph.nodes())
            target_node = nodes_list[-1] if nodes_list else 0
        else:
            target_node = node
            if target_node not in self._graph.nodes():
                raise ValueError(f"Node {target_node} not found in network")

        # Configure metrics collection
        self._graph.graph["COLLECT_OPERATOR_METRICS"] = collect_metrics

        # Map glyphs to operator instances
        from ..operators.definitions import (
            Emission,
            Reception,
            Coherence,
            Dissonance,
            Coupling,
            Resonance,
            Silence,
            Expansion,
            Contraction,
            SelfOrganization,
            Mutation,
            Transition,
            Recursivity,
        )
        from ..types import Glyph

        glyph_to_operator = {
            Glyph.AL: Emission(),
            Glyph.EN: Reception(),
            Glyph.IL: Coherence(),
            Glyph.OZ: Dissonance(),
            Glyph.UM: Coupling(),
            Glyph.RA: Resonance(),
            Glyph.SHA: Silence(),
            Glyph.VAL: Expansion(),
            Glyph.NUL: Contraction(),
            Glyph.THOL: SelfOrganization(),
            Glyph.ZHIR: Mutation(),
            Glyph.NAV: Transition(),
            Glyph.REMESH: Recursivity(),
        }

        operators = [glyph_to_operator[g] for g in sequence.glyphs]
        run_sequence(self._graph, target_node, operators)

        return self

    def list_canonical_sequences(
        self,
        domain: Optional[str] = None,
        with_oz: bool = False,
    ) -> Dict[str, Any]:
        """List available canonical sequences with optional filters.

        Returns a dictionary of canonical operator sequences from TNFR theory.
        Sequences can be filtered by domain or by presence of OZ (Dissonance).

        Parameters
        ----------
        domain : str, optional
            Filter by domain. Options:
            - 'general': Cross-domain patterns
            - 'biomedical': Therapeutic and healing sequences
            - 'cognitive': Epistemological and learning patterns
            - 'social': Organizational and collective sequences
        with_oz : bool, default=False
            If True, only return sequences containing OZ (Dissonance) operator.

        Returns
        -------
        dict
            Dictionary mapping sequence names to CanonicalSequence objects.
            Each entry contains: name, glyphs, pattern_type, description,
            use_cases, domain, and references.

        Examples
        --------
        List all canonical sequences:

        >>> net = TNFRNetwork("explorer")
        >>> sequences = net.list_canonical_sequences()
        >>> for name in sequences:
        ...     print(name)
        bifurcated_base
        bifurcated_collapse
        therapeutic_protocol
        theory_system
        full_deployment
        mod_stabilizer

        List only sequences with OZ:

        >>> oz_sequences = net.list_canonical_sequences(with_oz=True)
        >>> print(f"Found {len(oz_sequences)} sequences with OZ")
        Found 6 sequences with OZ

        List biomedical domain sequences:

        >>> bio_sequences = net.list_canonical_sequences(domain="biomedical")
        >>> for name, seq in bio_sequences.items():
        ...     print(f"{name}: {seq.description[:50]}...")

        See Also
        --------
        apply_canonical_sequence : Apply a canonical sequence to the network
        """
        from ..operators.canonical_patterns import CANONICAL_SEQUENCES
        from ..types import Glyph

        sequences = CANONICAL_SEQUENCES.copy()

        # Filter by domain if specified
        if domain is not None:
            sequences = {name: seq for name, seq in sequences.items() if seq.domain == domain}

        # Filter by OZ presence if requested
        if with_oz:
            sequences = {name: seq for name, seq in sequences.items() if Glyph.OZ in seq.glyphs}

        return sequences

    def visualize(self, **kwargs: Any) -> TNFRNetwork:
        """Visualize the network with TNFR metrics.

        Creates a visual representation of the network showing node states
        and connections. Requires matplotlib to be installed.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to visualization function.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        ValueError
            If no network has been created.

        Notes
        -----
        This is a placeholder for future visualization functionality.
        Current implementation will raise NotImplementedError.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")

        # Compute metrics if not done yet
        if self._results is None:
            self.measure()

        # Visualization will be implemented in future PR
        raise NotImplementedError(
            "Visualization functionality will be added in a future update. "
            "Use NetworkX's drawing functions directly on network._graph for now."
        )

    def save(self, filepath: Union[str, Path]) -> TNFRNetwork:
        """Save network state and results to file.

        Serializes the network graph and computed metrics to a file for
        later analysis or reproduction.

        Parameters
        ----------
        filepath : str or Path
            Path where network data should be saved.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.

        Raises
        ------
        ValueError
            If no network has been created.

        Notes
        -----
        This is a placeholder for future I/O functionality.
        Current implementation will raise NotImplementedError.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")

        # Compute metrics if not done yet
        if self._results is None:
            self.measure()

        # I/O functionality will be implemented in future PR
        raise NotImplementedError(
            "Save functionality will be added in a future update. "
            "Use networkx.write_gpickle or similar for now."
        )

    @property
    def graph(self) -> nx.Graph:
        """Access the underlying NetworkX graph.

        Returns
        -------
        nx.Graph
            The NetworkX graph with TNFR node attributes.

        Raises
        ------
        ValueError
            If no network has been created yet.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")
        return self._graph

    def get_node_count(self) -> int:
        """Get the number of nodes in the network.

        Returns
        -------
        int
            Number of nodes.

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")
        return self._graph.number_of_nodes()

    def get_edge_count(self) -> int:
        """Get the number of edges in the network.

        Returns
        -------
        int
            Number of edges.

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")
        return self._graph.number_of_edges()

    def get_average_degree(self) -> float:
        """Get the average degree of nodes in the network.

        Returns
        -------
        float
            Average node degree.

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")
        if self._graph.number_of_nodes() == 0:
            return 0.0
        return 2.0 * self._graph.number_of_edges() / self._graph.number_of_nodes()

    def get_density(self) -> float:
        """Get the density of the network.

        Network density is the ratio of actual edges to possible edges.

        Returns
        -------
        float
            Network density between 0 and 1.

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")
        n = self._graph.number_of_nodes()
        if n <= 1:
            return 0.0
        m = self._graph.number_of_edges()
        max_edges = n * (n - 1) / 2
        return m / max_edges if max_edges > 0 else 0.0

    def clone(self) -> TNFRNetwork:
        """Create a copy of the network structure.

        Returns
        -------
        TNFRNetwork
            A new network with copied structure. Note that this copies
            the graph structure but not all internal state (like locks).

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")

        import networkx as nx

        new_network = TNFRNetwork(f"{self.name}_copy", config=self._config)
        # Use NetworkX's copy method which handles TNFR graphs properly
        new_network._graph = nx.Graph(self._graph)
        new_network._node_counter = self._node_counter
        return new_network

    def reset(self) -> TNFRNetwork:
        """Reset the network to empty state.

        Returns
        -------
        TNFRNetwork
            Self for method chaining.
        """
        self._graph = None
        self._results = None
        self._node_counter = 0
        return self

    # -------------------------------------------------------------------------
    # Auto-optimization methods (NEW - Nov 28, 2025)
    # -------------------------------------------------------------------------

    def analyze_optimization_potential(self) -> Dict[str, Any]:
        """
        Analyze mathematical optimization potential for the current network.
        
        Uses unified field analysis and mathematical structure inspection
        to identify optimization opportunities automatically.
        
        Returns
        -------
        Dict[str, Any]
            Analysis results including:
            - field_analysis: Unified field characteristics
            - optimization_recommendations: Specific strategies  
            - predicted_improvements: Expected performance gains
            
        Raises
        ------
        ValueError
            If no network has been created.
            
        Examples
        --------
        Analyze optimization opportunities:
        
        >>> analysis = network.analyze_optimization_potential()
        >>> print(f"Recommendations: {analysis['optimization_recommendations']}")
        >>> print(f"Predicted speedup: {analysis['predicted_improvements']}")
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")
            
        if not _AUTO_OPTIMIZATION_AVAILABLE:
            return {
                "optimization_available": False,
                "error": "Auto-optimization engine not available",
                "field_analysis": {},
                "optimization_recommendations": [],
                "predicted_improvements": {}
            }
        
        try:
            analysis = analyze_optimization_potential(self._graph)
            return {
                "optimization_available": True,
                **analysis
            }
        except Exception as e:
            return {
                "optimization_available": False,
                "error": str(e),
                "field_analysis": {},
                "optimization_recommendations": [],
                "predicted_improvements": {}
            }

    def auto_optimize(self, operation_type: str = "network_simulation") -> TNFRNetwork:
        """
        Automatically apply optimization strategies based on mathematical analysis.
        
        This method uses the self-optimizing engine to analyze the current
        network state and automatically apply the best optimization strategy.
        
        Parameters
        ----------
        operation_type : str, default="network_simulation"
            Type of operation to optimize for.
            
        Returns
        -------
        TNFRNetwork
            Self for method chaining.
            
        Raises
        ------
        ValueError
            If no network has been created.
            
        Examples
        --------
        Apply automatic optimization:
        
        >>> network.auto_optimize("network_simulation")
        >>> results = network.measure()  # Optimized computation
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")
            
        if not _AUTO_OPTIMIZATION_AVAILABLE:
            print("Warning: Auto-optimization engine not available, skipping.")
            return self
            
        try:
            optimization_result = auto_optimize_field_computation(self._graph, operation_type=operation_type)
            
            # Store optimization information for future reference
            if not hasattr(self, '_optimization_history'):
                self._optimization_history = []
            
            self._optimization_history.append({
                "timestamp": __import__("time").time(),
                "operation_type": operation_type,
                "result": optimization_result
            })
            
            if optimization_result.get("optimization_applied", False):
                print(f"✅ Optimization applied: {optimization_result.get('strategy_used', 'unknown')}")
                improvement = optimization_result.get("performance_improvement", 1.0)
                if improvement > 1.1:
                    print(f"   Expected speedup: {improvement:.2f}x")
            else:
                print("ℹ️ No optimization applied (using standard computation)")
                
        except Exception as e:
            print(f"Warning: Auto-optimization failed: {e}")
            
        return self

    def learn_from_performance(
        self, 
        performance_data: Optional[Dict[str, Any]] = None
    ) -> TNFRNetwork:
        """
        Learn optimization strategies from performance data.
        
        Records performance experience and updates optimization policies
        for future automatic optimization decisions.
        
        Parameters
        ----------
        performance_data : Dict[str, Any], optional
            Performance metrics to learn from. If None, uses internal timing.
            
        Returns
        -------
        TNFRNetwork
            Self for method chaining.
            
        Examples
        --------
        Record performance learning:
        
        >>> network.learn_from_performance({"execution_time": 1.23, "accuracy": 0.99})
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")
            
        if not _AUTO_OPTIMIZATION_AVAILABLE:
            return self
            
        try:
            # Create self-optimizing engine
            engine = TNFRSelfOptimizingEngine(
                optimization_objective=OptimizationObjective.BALANCE_ALL
            )
            
            # Prepare experience data
            experience_data = {
                "graph_properties": {
                    "nodes": self._graph.number_of_nodes(),
                    "edges": self._graph.number_of_edges(),
                    "density": (2 * self._graph.number_of_edges() / 
                               max(1, self._graph.number_of_nodes() * (self._graph.number_of_nodes() - 1)))
                },
                "operation_type": "sdk_operation",
                "performance_metrics": performance_data or {},
                "timestamp": __import__("time").time(),
                "success": True
            }
            
            # This would normally create and record an OptimizationExperience
            # For now, just store the learning data
            if not hasattr(self, '_learning_history'):
                self._learning_history = []
            
            self._learning_history.append(experience_data)
            
        except Exception as e:
            print(f"Warning: Performance learning failed: {e}")
            
        return self

    def get_optimization_recommendations(self, operation_type: str = "measurement") -> Dict[str, Any]:
        """
        Get optimization strategy recommendations for specific operation.
        
        Parameters
        ----------
        operation_type : str, default="measurement"
            Type of operation to get recommendations for.
            
        Returns
        -------
        Dict[str, Any]
            Optimization recommendations and strategy analysis.
            
        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None or self._graph.number_of_nodes() == 0:
            raise ValueError("No network created. Use add_nodes() first.")
            
        if not _AUTO_OPTIMIZATION_AVAILABLE:
            return {
                "recommendations_available": False,
                "error": "Auto-optimization engine not available"
            }
        
        try:
            recommendations = recommend_field_optimization_strategy(self._graph, operation_type)
            return {
                "recommendations_available": True,
                **recommendations
            }
        except Exception as e:
            return {
                "recommendations_available": False,
                "error": str(e)
            }

    def export_to_dict(self) -> dict:
        """Export network structure to dictionary format.

        Returns
        -------
        dict
            Dictionary with network metadata and structure.

        Raises
        ------
        ValueError
            If no network has been created.
        """
        if self._graph is None:
            raise ValueError("No network created. Use add_nodes() first.")

        # Measure if not done yet
        if self._results is None:
            self.measure()

        return {
            "name": self.name,
            "metadata": {
                "nodes": self.get_node_count(),
                "edges": self.get_edge_count(),
                "density": self.get_density(),
                "average_degree": self.get_average_degree(),
            },
            "metrics": self._results.to_dict() if self._results else None,
            "config": {
                "random_seed": self._config.random_seed,
                "validate_invariants": self._config.validate_invariants,
                "vf_range": self._config.default_vf_range,
                "epi_range": self._config.default_epi_range,
            },
        }
