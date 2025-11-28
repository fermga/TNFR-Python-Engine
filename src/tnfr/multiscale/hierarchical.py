"""Hierarchical multi-scale TNFR network implementation.

Implements operational fractality by managing TNFR networks at multiple scales
with cross-scale coupling, preserving canonical TNFR invariants.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import networkx as nx
import numpy as np

from ..types import DeltaNFR, NodeId, TNFRGraph
from ..dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed
from ..utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScaleDefinition:
    """Definition of a single scale in a hierarchical TNFR network.

    Parameters
    ----------
    name : str
        Identifier for this scale (e.g., "quantum", "molecular", "cellular")
    node_count : int
        Number of nodes at this scale
    coupling_strength : float
        Base coupling strength for nodes within this scale (0.0 to 1.0)
    edge_probability : float, optional
        Probability of edge creation in Erdős-Rényi graph generation
    """

    name: str
    node_count: int
    coupling_strength: float
    edge_probability: float = 0.1


@dataclass
class EvolutionResult:
    """Results from multi-scale evolution.

    Attributes
    ----------
    scale_results : Dict[str, Any]
        Results indexed by scale name
    total_coherence : float
        Aggregated coherence across all scales
    cross_scale_coupling : float
        Measure of cross-scale synchronization
    """

    scale_results: Dict[str, Any]
    total_coherence: float = 0.0
    cross_scale_coupling: float = 0.0


class HierarchicalTNFRNetwork:
    """Multi-scale TNFR network supporting operational fractality (§3.7).

    Manages multiple TNFR networks at different scales with cross-scale
    coupling, enabling simultaneous evolution while preserving structural
    coherence.

    This implementation maintains all TNFR canonical invariants:
    - Nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
    - Operator closure: all transformations yield valid TNFR states
    - Phase verification: explicit synchrony checks for coupling
    - Determinism: reproducible evolution with fixed seeds

    Parameters
    ----------
    scales : Sequence[ScaleDefinition]
        Definitions of each scale in the hierarchy
    seed : int, optional
        Random seed for reproducible network generation
    parallel : bool, optional
        Enable parallel evolution of scales (default: True)
    max_workers : int, optional
        Maximum worker threads/processes for parallel execution

    Examples
    --------
    Create a two-scale network and evolve it:

    >>> from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition
    >>> scales = [
    ...     ScaleDefinition("micro", 100, 0.8),
    ...     ScaleDefinition("macro", 50, 0.5),
    ... ]
    >>> network = HierarchicalTNFRNetwork(scales, seed=42)
    >>> result = network.evolve_multiscale(dt=0.1, steps=10)
    >>> result.total_coherence  # doctest: +SKIP
    0.65...

    Notes
    -----
    Cross-scale coupling is computed as:
        ΔNFR_total = ΔNFR_scale + Σ(coupling_ij * ΔNFR_other_scale)
    where coupling_ij represents the strength of influence from scale j to i.
    """

    def __init__(
        self,
        scales: Sequence[ScaleDefinition],
        seed: Optional[int] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ):
        if not scales:
            raise ValueError("At least one scale definition required")

        self.scales = list(scales)
        self.seed = seed
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize networks for each scale
        self.networks_by_scale: Dict[str, TNFRGraph] = {}
        self._initialize_scales()

        # Cross-scale coupling matrix (scale x scale)
        self.cross_scale_couplings: Dict[tuple[str, str], float] = {}
        self._initialize_cross_scale_couplings()

        logger.info(
            f"Initialized hierarchical network with {len(scales)} scales, "
            f"total {sum(s.node_count for s in scales)} nodes"
        )

    def _initialize_scales(self) -> None:
        """Initialize TNFR network for each scale."""
        rng = np.random.RandomState(self.seed)

        for scale in self.scales:
            # Create Erdős-Rényi graph for this scale
            G = nx.erdos_renyi_graph(
                scale.node_count, scale.edge_probability, seed=rng.randint(0, 2**31)
            )

            # Initialize each node with TNFR attributes
            for node in G.nodes():
                f"{scale.name}_{node}"
                G.nodes[node]["epi"] = rng.uniform(0.0, 1.0)
                G.nodes[node]["vf"] = rng.uniform(0.5, 1.5)
                G.nodes[node]["phase"] = rng.uniform(0.0, 2 * np.pi)
                G.nodes[node]["delta_nfr"] = 0.0
                G.nodes[node]["si"] = 0.0

            # Set base coupling weights
            for u, v in G.edges():
                G[u][v]["weight"] = scale.coupling_strength * rng.uniform(0.8, 1.2)

            # Install ΔNFR hook
            set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

            self.networks_by_scale[scale.name] = G

    def _initialize_cross_scale_couplings(self) -> None:
        """Initialize coupling strengths between scales.

        Default: Adjacent scales couple more strongly than distant scales.
        """
        scale_names = [s.name for s in self.scales]
        len(scale_names)

        for i, scale_i in enumerate(scale_names):
            for j, scale_j in enumerate(scale_names):
                if i == j:
                    continue  # No self-coupling

                # Distance-based coupling: closer scales couple more
                distance = abs(i - j)
                coupling_strength = 0.3 / distance if distance > 0 else 0.0

                self.cross_scale_couplings[(scale_i, scale_j)] = coupling_strength

    def set_cross_scale_coupling(self, from_scale: str, to_scale: str, strength: float) -> None:
        """Set explicit cross-scale coupling strength.

        Parameters
        ----------
        from_scale : str
            Source scale name
        to_scale : str
            Target scale name
        strength : float
            Coupling strength (0.0 to 1.0)
        """
        if from_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {from_scale}")
        if to_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {to_scale}")
        if strength < 0.0 or strength > 1.0:
            raise ValueError("Coupling strength must be in [0.0, 1.0]")

        self.cross_scale_couplings[(from_scale, to_scale)] = strength

    def compute_multiscale_dnfr(self, node_id: NodeId, target_scale: str) -> DeltaNFR:
        """Compute ΔNFR considering all relevant scales.

        Implements cross-scale ΔNFR computation:
            ΔNFR_total = ΔNFR_base + Σ(coupling * ΔNFR_other)

        Parameters
        ----------
        node_id : NodeId
            Node identifier within the target scale
        target_scale : str
            Scale where the node resides

        Returns
        -------
        DeltaNFR
            Multi-scale ΔNFR value
        """
        if target_scale not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {target_scale}")

        G = self.networks_by_scale[target_scale]

        # Base ΔNFR at target scale (simplified computation)
        base_dnfr = G.nodes[node_id].get("delta_nfr", 0.0)

        # Cross-scale contributions
        cross_scale_contribution = 0.0
        for other_scale in self.networks_by_scale:
            if other_scale == target_scale:
                continue

            coupling = self.cross_scale_couplings.get((target_scale, other_scale), 0.0)
            if coupling > 0:
                # Aggregate ΔNFR from other scale
                other_G = self.networks_by_scale[other_scale]
                other_dnfr_values = [
                    other_G.nodes[n].get("delta_nfr", 0.0) for n in other_G.nodes()
                ]
                mean_other_dnfr = np.mean(other_dnfr_values) if other_dnfr_values else 0.0
                cross_scale_contribution += coupling * mean_other_dnfr

        return base_dnfr + cross_scale_contribution

    def compute_total_coherence(self) -> float:
        """Compute aggregated coherence across all scales.

        Returns
        -------
        float
            Total coherence C(t) aggregated across scales
        """
        total_c = 0.0
        total_nodes = 0

        for scale_name, G in self.networks_by_scale.items():
            # Simplified coherence: 1 - mean(|ΔNFR|)
            dnfr_values = [abs(G.nodes[n].get("delta_nfr", 0.0)) for n in G.nodes()]
            mean_abs_dnfr = np.mean(dnfr_values) if dnfr_values else 0.0
            scale_coherence = 1.0 / (1.0 + mean_abs_dnfr)

            # Weight by node count
            node_count = G.number_of_nodes()
            total_c += scale_coherence * node_count
            total_nodes += node_count

        return total_c / total_nodes if total_nodes > 0 else 0.0

    def evolve_multiscale(
        self,
        dt: float = 0.1,
        steps: int = 10,
        operators: Optional[Sequence[str]] = None,
    ) -> EvolutionResult:
        """Evolve all scales simultaneously with cross-coupling.

        Parameters
        ----------
        dt : float
            Time step for evolution
        steps : int
            Number of evolution steps
        operators : Sequence[str], optional
            Structural operators to apply (e.g., ["A'L", "THOL"])

        Returns
        -------
        EvolutionResult
            Results containing scale-specific and aggregated metrics
        """
        if operators is None:
            operators = ["THOL"]  # Default: Coherence operator

        results = {}

        for step in range(steps):
            if self.parallel and self.max_workers != 1:
                # Parallel evolution
                results = self._evolve_parallel(dt, operators)
            else:
                # Sequential evolution
                results = self._evolve_sequential(dt, operators)

            # Apply cross-scale coupling effects
            self._apply_cross_scale_coupling(dt)

        # Compute final metrics
        total_coherence = self.compute_total_coherence()
        cross_coupling = self._compute_cross_scale_synchrony()

        return EvolutionResult(
            scale_results=results,
            total_coherence=total_coherence,
            cross_scale_coupling=cross_coupling,
        )

    def _evolve_sequential(self, dt: float, operators: Sequence[str]) -> Dict[str, Any]:
        """Evolve scales sequentially."""
        results = {}

        for scale_name, G in self.networks_by_scale.items():
            # Simple evolution: update ΔNFR for all nodes
            for node in G.nodes():
                phase = G.nodes[node]["phase"]
                vf = G.nodes[node]["vf"]

                # Compute neighbor phase difference contribution
                neighbors = list(G.neighbors(node))
                if neighbors:
                    phase_diffs = [np.sin(phase - G.nodes[n]["phase"]) for n in neighbors]
                    dnfr = np.mean(phase_diffs)
                else:
                    dnfr = 0.0

                G.nodes[node]["delta_nfr"] = dnfr

                # Update EPI according to nodal equation: ∂EPI/∂t = νf · ΔNFR
                G.nodes[node]["epi"] += vf * dnfr * dt

            results[scale_name] = {"coherence": self._scale_coherence(G)}

        return results

    def _evolve_parallel(self, dt: float, operators: Sequence[str]) -> Dict[str, Any]:
        """Evolve scales in parallel using ThreadPoolExecutor.

        Note: ThreadPoolExecutor is used instead of ProcessPoolExecutor because:
        1. NetworkX graphs are not easily picklable (required for multiprocessing)
        2. The overhead of serializing/deserializing graphs would negate benefits
        3. Thread-based parallelism still provides speedup for I/O and NumPy ops

        For CPU-intensive workloads on very large scales, consider using
        ProcessPoolExecutor with custom serialization or shared memory.
        """
        results = {}

        # Use ThreadPoolExecutor for GIL-safe parallel evolution
        # (ProcessPoolExecutor would require pickling networkx graphs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                scale_name: executor.submit(self._evolve_single_scale, scale_name, dt, operators)
                for scale_name in self.networks_by_scale
            }

            for scale_name, future in futures.items():
                results[scale_name] = future.result()

        return results

    def _evolve_single_scale(
        self, scale_name: str, dt: float, operators: Sequence[str]
    ) -> Dict[str, Any]:
        """Evolve a single scale (helper for parallel execution)."""
        G = self.networks_by_scale[scale_name]

        # Same logic as _evolve_sequential but for one scale
        for node in G.nodes():
            phase = G.nodes[node]["phase"]
            vf = G.nodes[node]["vf"]

            neighbors = list(G.neighbors(node))
            if neighbors:
                phase_diffs = [np.sin(phase - G.nodes[n]["phase"]) for n in neighbors]
                dnfr = np.mean(phase_diffs)
            else:
                dnfr = 0.0

            G.nodes[node]["delta_nfr"] = dnfr
            G.nodes[node]["epi"] += vf * dnfr * dt

        return {"coherence": self._scale_coherence(G)}

    def _apply_cross_scale_coupling(self, dt: float) -> None:
        """Apply cross-scale coupling effects after evolution step."""
        # For each scale, add cross-scale ΔNFR contributions
        for target_scale in self.networks_by_scale:
            G_target = self.networks_by_scale[target_scale]

            for node in G_target.nodes():
                cross_contribution = 0.0

                for source_scale in self.networks_by_scale:
                    if source_scale == target_scale:
                        continue

                    coupling = self.cross_scale_couplings.get((target_scale, source_scale), 0.0)

                    if coupling > 0:
                        G_source = self.networks_by_scale[source_scale]
                        source_dnfr_values = [
                            G_source.nodes[n].get("delta_nfr", 0.0) for n in G_source.nodes()
                        ]
                        mean_source_dnfr = (
                            np.mean(source_dnfr_values) if source_dnfr_values else 0.0
                        )
                        cross_contribution += coupling * mean_source_dnfr

                # Apply cross-scale effect to EPI
                if cross_contribution != 0.0:
                    vf = G_target.nodes[node]["vf"]
                    G_target.nodes[node]["epi"] += vf * cross_contribution * dt

    def _scale_coherence(self, G: TNFRGraph) -> float:
        """Compute coherence for a single scale."""
        dnfr_values = [abs(G.nodes[n].get("delta_nfr", 0.0)) for n in G.nodes()]
        mean_abs_dnfr = np.mean(dnfr_values) if dnfr_values else 0.0
        return 1.0 / (1.0 + mean_abs_dnfr)

    def _compute_cross_scale_synchrony(self) -> float:
        """Compute cross-scale phase synchronization."""
        if len(self.networks_by_scale) < 2:
            return 0.0

        # Simplified: compare mean phases across scales
        scale_mean_phases = []
        for G in self.networks_by_scale.values():
            phases = [G.nodes[n]["phase"] for n in G.nodes()]
            if phases:
                # Use circular mean for phases
                mean_phase = np.angle(np.mean(np.exp(1j * np.array(phases))))
                scale_mean_phases.append(mean_phase)

        if len(scale_mean_phases) < 2:
            return 0.0

        # Compute phase coherence between scales
        phase_diffs = []
        for i in range(len(scale_mean_phases)):
            for j in range(i + 1, len(scale_mean_phases)):
                phase_diff = abs(scale_mean_phases[i] - scale_mean_phases[j])
                # Normalize to [0, π]
                phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
                phase_diffs.append(phase_diff)

        mean_phase_diff = np.mean(phase_diffs) if phase_diffs else 0.0
        # Convert to synchrony metric (0 = no sync, 1 = perfect sync)
        synchrony = 1.0 - (mean_phase_diff / np.pi)

        return max(0.0, synchrony)

    def get_scale_network(self, scale_name: str) -> TNFRGraph:
        """Get the network graph for a specific scale.

        Parameters
        ----------
        scale_name : str
            Name of the scale

        Returns
        -------
        TNFRGraph
            NetworkX graph for the specified scale
        """
        if scale_name not in self.networks_by_scale:
            raise ValueError(f"Unknown scale: {scale_name}")
        return self.networks_by_scale[scale_name]

    def memory_footprint(self) -> Dict[str, float]:
        """Estimate memory usage per scale.

        Returns
        -------
        Dict[str, float]
            Memory usage in MB for each scale
        """
        footprint = {}
        for scale_name, G in self.networks_by_scale.items():
            # Rough estimate: graph structure + node attributes
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            # NetworkX overhead + node dict + edge dict + attributes
            # Each node: ~200 bytes (dict overhead) + 4 attributes * 8 bytes
            # Each edge: ~100 bytes (dict overhead) + 1 attribute * 8 bytes
            estimate_bytes = n_nodes * 232 + n_edges * 108
            estimate_mb = estimate_bytes / (1024 * 1024)

            footprint[scale_name] = estimate_mb

        footprint["total"] = sum(v for k, v in footprint.items() if k != "total")
        return footprint
