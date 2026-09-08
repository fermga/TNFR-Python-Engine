"""Snapshot centralization diagnostics for TNFR graph states.

The analyzers rank possible coordination nodes using graph spectrum, signed
scalar EPI, structural frequency, and wrapped phase separation.  The nodal
equation supplies the channel meanings, but it does not imply that EPI flows
towards a graph center, that a high-frequency node is a coordinator, or that a
recommended topology improves runtime, stability, or fault tolerance.

This module therefore returns heuristic scores and a topology recommendation.
It neither mutates topology nor benchmarks the recommendation.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..types import require_finite_real_scalar_epi
from ..utils import angle_diff

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

try:
    from ..mathematics.spectral import get_laplacian_spectrum

    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    EMERGENT_CENTRALITY_THRESHOLD_CANONICAL,
    EMERGENT_COORDINATION_BOOST_CANONICAL,
    EMERGENT_COORDINATION_THRESHOLD_CANONICAL,
    EMERGENT_COUPLING_STRENGTH_CANONICAL,
    EMERGENT_FREQ_BALANCE_CANONICAL,
    EMERGENT_STABILITY_THRESHOLD_CANONICAL,
    NODAL_OPT_COUPLING_CANONICAL,
)

try:
    from .spectral_structural_fusion import TNFRSpectralStructuralFusionEngine

    HAS_SPECTRAL_STRUCTURAL_FUSION = True
except ImportError:
    HAS_SPECTRAL_STRUCTURAL_FUSION = False


def _node_scalar_epi(G: Any, node: Any) -> float:
    """Read one finite signed scalar EPI through canonical alias precedence."""

    raw = get_attr(
        G.nodes[node],
        ALIAS_EPI,
        0.0,
        strict=True,
        conv=lambda value: value,
    )
    return require_finite_real_scalar_epi(raw, f"node {node!r} EPI")


def _finite_nonnegative_mean(values: list[float], label: str) -> float:
    """Average finite nonnegative values without overflowing their sum."""

    if not values:
        return 0.0
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise TNFRValueError(f"{label} must contain finite nonnegative values")
    scale = max(values)
    if scale == 0.0:
        return 0.0
    result = scale * math.fsum(value / scale for value in values) / len(values)
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} mean exceeds the finite scalar range")
    return result


def _finite_product(*values: float, label: str) -> float:
    """Multiply finite factors and reject an unrepresentable result."""

    result = math.prod(values)
    if not math.isfinite(result):
        raise TNFRValueError(f"{label} exceeds the finite scalar range")
    return result


def _node_real_channel(
    G: Any,
    node: Any,
    aliases: tuple[str, ...],
    default: float,
    label: str,
    *,
    nonnegative: bool = False,
) -> float:
    """Read one finite real scalar channel through strict alias precedence."""

    raw = get_attr(
        G.nodes[node], aliases, default, strict=True, conv=lambda value: value
    )
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise TNFRValueError(f"node {node!r} {label} must be a finite real scalar")
    try:
        value = float(raw)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"node {node!r} {label} must be a finite real scalar"
        ) from exc
    if not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise TNFRValueError(
            f"node {node!r} {label} must be a {qualifier} real scalar"
        )
    return value


class CentralizationStrategy(Enum):
    """Strategies for emergent centralization."""

    SPECTRAL_DOMINANCE = "spectral_dominance"  # Based on eigenmode centrality
    INFORMATION_FLOW = "information_flow"  # Based on EPI concentration
    FREQUENCY_HIERARCHY = "frequency_hierarchy"  # Based on νf values
    LOAD_BALANCING = "load_balancing"  # Based on computational load
    PHASE_COORDINATION = "phase_coordination"  # Based on phase synchronization
    ADAPTIVE_TOPOLOGY = "adaptive_topology"  # Based on dynamic restructuring


@dataclass
class CentralizationNode:
    """A node identified as a coordination center."""

    node_id: Any
    centrality_score: float
    coordination_capacity: float
    current_load: float | None
    specialization: str  # type of coordination this node excels at
    connected_cluster: list[Any]  # Nodes coordinated by this center
    mathematical_signature: dict[str, Any]


@dataclass
class CentralizationPattern:
    """Discovered centralization pattern in the network.

    ``efficiency_gain`` is a historical compatibility name for a bounded
    coordination-coverage score. It is not a measured runtime gain.
    """

    strategy: CentralizationStrategy
    coordination_nodes: list[CentralizationNode]
    efficiency_gain: float
    stability_measure: float
    adaptation_rate: float
    mathematical_basis: dict[str, Any]
    load_distribution: dict[Any, float]


@dataclass
class CentralizationResult:
    """Result of a read-only centralization recommendation.

    No candidate topology is executed or benchmarked. Compatibility fields for
    measured improvements therefore remain empty/``None``; ``diagnostic_scores``
    contains the quantities that were actually computed.
    """

    discovered_patterns: list[CentralizationPattern]
    optimal_strategy: CentralizationStrategy
    recommended_topology: dict[str, Any]
    performance_improvements: dict[str, float | None]
    coordination_efficiency: float | None
    fault_tolerance: float | None
    execution_time: float
    diagnostic_scores: dict[str, float | None] = field(default_factory=dict)


class TNFREmergentCentralizationEngine:
    """
    Rank candidate coordination patterns in one graph snapshot.

    The engine produces diagnostics and recommendations only; it does not apply
    topology changes or establish performance improvements.
    """

    def __init__(self, enable_adaptive_topology: bool = True):
        self.enable_adaptive_topology = enable_adaptive_topology

        # Centralization state
        self.discovered_patterns = []
        self.current_coordination_nodes = {}
        self.performance_history = []

        # Operational thresholds (engine tuning, not TNFR physics)
        self.centrality_threshold = (
            EMERGENT_CENTRALITY_THRESHOLD_CANONICAL  # = 0.74 (operational)
        )
        self.coordination_threshold = (
            EMERGENT_COORDINATION_THRESHOLD_CANONICAL  # ≈ 0.5550
        )
        self.stability_threshold = (
            EMERGENT_STABILITY_THRESHOLD_CANONICAL  # ≈ 0.5903
        )

        # Performance tracking
        self.centralization_attempts = 0
        self.successful_centralizations = 0

        # Thread safety
        self._lock = threading.Lock()
        self.fusion_engine = (
            TNFRSpectralStructuralFusionEngine()
            if HAS_SPECTRAL_STRUCTURAL_FUSION
            else None
        )

    def _prefetch_spectral_state(self, G: Any) -> None:
        """Ensure spectral + structural caches are warmed before analysis."""
        if self.fusion_engine is None or G is None:
            return

        self.fusion_engine.prewarm_state(G)

    def _coordinate_cache_with_pattern(
        self, G: Any, pattern: CentralizationPattern
    ) -> None:
        """Delegate cache coordination to the fusion engine using pattern nodes."""
        if self.fusion_engine is None:
            return

        self.fusion_engine.coordinate_cache_with_central_nodes(
            G,
            pattern.coordination_nodes,
            strategy=pattern.strategy.value,
        )

    def analyze_spectral_centralization(self, G: Any) -> list[CentralizationNode]:
        """
        Discover centralization based on spectral properties.

        Uses the magnitude of the Fiedler-vector component as a candidate
        coordination score.
        """
        coordination_nodes = []

        if not HAS_NETWORKX or not HAS_SPECTRAL or G is None:
            return coordination_nodes

        self._prefetch_spectral_state(G)

        # Get spectral decomposition
        eigenvalues, eigenvectors = get_laplacian_spectrum(G)

        # Read the Fiedler vector from the ordered Laplacian basis.
        if len(eigenvectors) > 0:
            # Use the Fiedler vector (second smallest eigenvalue) for coordination
            if len(eigenvalues) > 1:
                fiedler_vector = eigenvectors[:, 1]  # Second smallest eigenvalue

                # Find nodes with high coordination potential
                nodes = list(G.nodes())
                for i, node in enumerate(nodes):
                    centrality = abs(fiedler_vector[i])

                    if centrality > self.centrality_threshold:
                        # Calculate coordination capacity based on network position
                        degree = G.degree(node)
                        epi_value = _node_scalar_epi(G, node)
                        vf_value = _node_real_channel(
                            G, node, ALIAS_VF, 1.0, "structural frequency", nonnegative=True
                        )

                        # Mathematical signature for this coordination node
                        signature = {
                            "spectral_centrality": centrality,
                            "fiedler_component": fiedler_vector[i],
                            "degree": degree,
                            "epi": epi_value,
                            "vf": vf_value,
                            "eigenvalue_proximity": (
                                min(abs(eigenvalues - vf_value))
                                if len(eigenvalues) > 0
                                else 0
                            ),
                        }

                        # Find connected cluster
                        neighbors = list(G.neighbors(node))
                        cluster = [node] + neighbors[
                            : int(degree * EMERGENT_COUPLING_STRENGTH_CANONICAL)
                        ]  # Deterministic prefix under graph neighbor iteration order

                        coord_node = CentralizationNode(
                            node_id=node,
                            centrality_score=centrality,
                            coordination_capacity=_finite_product(
                                float(degree),
                                float(centrality),
                                vf_value,
                                label="spectral coordination capacity",
                            ),
                            current_load=None,  # No load measurement is available in this snapshot
                            specialization="spectral_coordination",
                            connected_cluster=cluster,
                            mathematical_signature=signature,
                        )
                        coordination_nodes.append(coord_node)

        return coordination_nodes

    def analyze_information_flow_centralization(
        self, G: Any
    ) -> list[CentralizationNode]:
        """
        Discover centralization based on information (EPI) flow patterns.

        Ranks nodes by EPI magnitude concentration and local signed contrast.
        """
        coordination_nodes = []

        if not HAS_NETWORKX or G is None:
            return coordination_nodes

        # Analyze the signed scalar EPI chart and its magnitude concentration.
        epi_values = {node: _node_scalar_epi(G, node) for node in G.nodes()}
        epi_magnitudes = {node: abs(value) for node, value in epi_values.items()}
        magnitude_scale = max(epi_magnitudes.values(), default=0.0)

        if magnitude_scale > 0.0:
            scaled_total = math.fsum(
                magnitude / magnitude_scale for magnitude in epi_magnitudes.values()
            )
            for node in G.nodes():
                signed_epi = epi_values[node]
                epi_magnitude = epi_magnitudes[node]
                epi_fraction = (epi_magnitude / magnitude_scale) / scaled_total

                # High EPI concentration indicates coordination potential
                if (
                    epi_fraction > NODAL_OPT_COUPLING_CANONICAL
                ):  # ≈ 0.099 - Significant EPI concentration
                    # Analyze information flow capacity
                    neighbors = list(G.neighbors(node))
                    neighbor_epi = [epi_values[n] for n in neighbors]

                    # Mean signed-chart contrast with graph neighbors.
                    contrasts = [
                        abs(signed_epi - neighbor_value)
                        for neighbor_value in neighbor_epi
                    ]
                    info_gradient = _finite_nonnegative_mean(
                        contrasts, "neighbor EPI contrasts"
                    )

                    # Coordination capacity based on information processing
                    vf_value = _node_real_channel(
                        G, node, ALIAS_VF, 1.0, "structural frequency", nonnegative=True
                    )
                    coordination_capacity = _finite_product(
                        epi_fraction,
                        info_gradient,
                        vf_value,
                        label="information coordination capacity",
                    )

                    if coordination_capacity > self.coordination_threshold:
                        signature = {
                            "epi_concentration": epi_fraction,
                            "information_gradient": info_gradient,
                            "epi_magnitude": epi_magnitude,
                            "signed_epi": signed_epi,
                            "neighbor_count": len(neighbors),
                            "vf": vf_value,
                            "processing_capacity": coordination_capacity,
                        }

                        # Connected cluster based on information similarity
                        similar_nodes = [
                            n
                            for n in neighbors
                            if abs(epi_values[n] - signed_epi)
                            < info_gradient * EMERGENT_FREQ_BALANCE_CANONICAL
                        ]
                        cluster = [node] + similar_nodes

                        coord_node = CentralizationNode(
                            node_id=node,
                            centrality_score=epi_fraction,
                            coordination_capacity=coordination_capacity,
                            current_load=None,
                            specialization="information_coordination",
                            connected_cluster=cluster,
                            mathematical_signature=signature,
                        )
                        coordination_nodes.append(coord_node)

        return coordination_nodes

    def analyze_frequency_hierarchy_centralization(
        self, G: Any
    ) -> list[CentralizationNode]:
        """
        Discover centralization based on frequency (νf) hierarchy.

        Ranks nodes by frequency relative to this snapshot and their neighbors.
        """
        coordination_nodes = []

        if not HAS_NETWORKX or G is None:
            return coordination_nodes

        # Analyze νf distribution
        vf_values = {
            node: _node_real_channel(
                G, node, ALIAS_VF, 1.0, "structural frequency", nonnegative=True
            )
            for node in G.nodes()
        }
        max_vf = max(vf_values.values()) if vf_values else 1.0

        # High-frequency nodes become natural coordinators
        for node in G.nodes():
            vf = vf_values[node]
            relative_frequency = vf / max_vf if max_vf > 0 else 0

            if (
                relative_frequency > EMERGENT_STABILITY_THRESHOLD_CANONICAL
            ):  # ≈ 0.590 - Top frequency nodes
                # Calculate coordination capacity based on frequency advantage
                neighbors = list(G.neighbors(node))
                neighbor_vf = [vf_values.get(n, 1.0) for n in neighbors]

                # Frequency dominance over neighbors
                frequency_advantage = _finite_nonnegative_mean(
                    [max(0.0, vf - nvf) for nvf in neighbor_vf],
                    "neighbor frequency advantages",
                )

                degree = G.degree(node)
                coordination_capacity = _finite_product(
                    relative_frequency,
                    frequency_advantage,
                    float(degree),
                    label="frequency coordination capacity",
                )

                if coordination_capacity > self.coordination_threshold:
                    signature = {
                        "relative_frequency": relative_frequency,
                        "absolute_frequency": vf,
                        "frequency_advantage": frequency_advantage,
                        "degree": degree,
                        "neighbor_frequencies": neighbor_vf,
                        "synchronization_potential": (
                            min(neighbor_vf) / vf if neighbor_vf and vf > 0 else 0
                        ),
                    }

                    # Cluster includes nodes that can synchronize with this frequency
                    sync_threshold = (
                        vf * EMERGENT_COUPLING_STRENGTH_CANONICAL
                    )  # Within 30% of coordinator frequency
                    sync_neighbors = [
                        n for n in neighbors if vf_values.get(n, 1.0) >= sync_threshold
                    ]
                    cluster = [node] + sync_neighbors

                    coord_node = CentralizationNode(
                        node_id=node,
                        centrality_score=relative_frequency,
                        coordination_capacity=coordination_capacity,
                        current_load=None,
                        specialization="frequency_coordination",
                        connected_cluster=cluster,
                        mathematical_signature=signature,
                    )
                    coordination_nodes.append(coord_node)

        return coordination_nodes

    def analyze_phase_coordination_centralization(
        self, G: Any
    ) -> list[CentralizationNode]:
        """
        Discover centralization based on phase synchronization potential.

        Rank nodes by a reciprocal wrapped-distance affinity and capacity.

        The affinity is an operational coordination score. It is distinct from
        both structural coherence C(t) and the Kuramoto order parameter.
        """
        coordination_nodes = []

        if not HAS_NETWORKX or G is None:
            return coordination_nodes

        # Analyze phase distribution
        phase_values = {
            node: _node_real_channel(G, node, ALIAS_THETA, 0.0, "phase")
            for node in G.nodes()
        }

        for node in G.nodes():
            phase = phase_values[node]
            neighbors = list(G.neighbors(node))

            if len(neighbors) > 2:  # Need sufficient connections for coordination
                neighbor_phases = [phase_values.get(n, 0.0) for n in neighbors]

                # Bounded affinity derived only from shortest-arc separation.
                # It is a coordination heuristic, not structural coherence C(t)
                # and not the Kuramoto phase-order parameter.
                phase_differences = [
                    abs(angle_diff(phase, nphase)) for nphase in neighbor_phases
                ]
                avg_phase_diff = float(np.mean(phase_differences))
                phase_distance_affinity = 1.0 / (1.0 + avg_phase_diff)

                # Phase coordination capacity
                vf = _node_real_channel(
                    G, node, ALIAS_VF, 1.0, "structural frequency", nonnegative=True
                )
                coordination_capacity = _finite_product(
                    phase_distance_affinity,
                    float(len(neighbors)),
                    vf,
                    label="phase coordination capacity",
                )

                if (
                    phase_distance_affinity
                    > EMERGENT_CENTRALITY_THRESHOLD_CANONICAL
                    and coordination_capacity > self.coordination_threshold
                ):  # ≈ 0.737
                    signature = {
                        "phase_distance_affinity": phase_distance_affinity,
                        # Compatibility alias; explicitly not structural C(t).
                        "phase_coherence": phase_distance_affinity,
                        "phase_metric_kind": "reciprocal_mean_wrapped_distance",
                        "average_phase_difference": avg_phase_diff,
                        "neighbor_count": len(neighbors),
                        "vf": vf,
                        "phase": phase,
                        "synchronization_strength": coordination_capacity,
                    }

                    # Cluster includes phase-synchronized neighbors
                    sync_threshold = np.pi / 4  # Within 45 degrees
                    sync_neighbors = [
                        n
                        for n, nphase in zip(neighbors, neighbor_phases)
                        if abs(angle_diff(phase, nphase)) < sync_threshold
                    ]
                    cluster = [node] + sync_neighbors

                    coord_node = CentralizationNode(
                        node_id=node,
                        centrality_score=phase_distance_affinity,
                        coordination_capacity=coordination_capacity,
                        current_load=None,
                        specialization="phase_coordination",
                        connected_cluster=cluster,
                        mathematical_signature=signature,
                    )
                    coordination_nodes.append(coord_node)

        return coordination_nodes

    def discover_centralization_patterns(self, G: Any) -> list[CentralizationPattern]:
        """
        Discover all centralization patterns in the network.
        """
        patterns = []

        # Analyze each centralization strategy
        strategies = [
            (
                CentralizationStrategy.SPECTRAL_DOMINANCE,
                self.analyze_spectral_centralization,
            ),
            (
                CentralizationStrategy.INFORMATION_FLOW,
                self.analyze_information_flow_centralization,
            ),
            (
                CentralizationStrategy.FREQUENCY_HIERARCHY,
                self.analyze_frequency_hierarchy_centralization,
            ),
            (
                CentralizationStrategy.PHASE_COORDINATION,
                self.analyze_phase_coordination_centralization,
            ),
        ]

        for strategy, analyzer in strategies:
            coordination_nodes = analyzer(G)

            if coordination_nodes:
                # Calculate pattern metrics
                capacities = [
                    node.coordination_capacity for node in coordination_nodes
                ]
                capacity_scale = max(capacities, default=0.0)
                scaled_capacity_total = (
                    math.fsum(value / capacity_scale for value in capacities)
                    if capacity_scale > 0.0
                    else 0.0
                )
                total_capacity_raw = capacity_scale * scaled_capacity_total
                total_capacity = (
                    total_capacity_raw if math.isfinite(total_capacity_raw) else None
                )
                avg_centrality = np.mean(
                    [node.centrality_score for node in coordination_nodes]
                )

                # Bounded coordination-node coverage score (historical field name).
                efficiency_gain = min(
                    len(coordination_nodes)
                    / len(G.nodes())
                    * EMERGENT_COORDINATION_BOOST_CANONICAL,
                    1.0,
                )

                # Mean detector centrality score (historical field name).
                stability_measure = avg_centrality

                # Mathematical basis
                mathematical_basis = {
                    "coordination_node_count": len(coordination_nodes),
                    "total_coordination_capacity": total_capacity,
                    "average_centrality": avg_centrality,
                    "coverage_fraction": len(
                        set().union(
                            *[node.connected_cluster for node in coordination_nodes]
                        )
                    )
                    / len(G.nodes()),
                }

                # Load distribution across coordination nodes
                if scaled_capacity_total > 0.0:
                    load_distribution = {
                        node.node_id: (
                            node.coordination_capacity / capacity_scale
                        )
                        / scaled_capacity_total
                        for node in coordination_nodes
                    }
                else:
                    load_distribution = {}

                pattern = CentralizationPattern(
                    strategy=strategy,
                    coordination_nodes=coordination_nodes,
                    efficiency_gain=efficiency_gain,
                    stability_measure=stability_measure,
                    adaptation_rate=NODAL_OPT_COUPLING_CANONICAL,  # Default adaptation rate
                    mathematical_basis=mathematical_basis,
                    load_distribution=load_distribution,
                )
                patterns.append(pattern)

        return patterns

    def optimize_centralization(
        self, G: Any, objective: str = "efficiency"
    ) -> CentralizationResult:
        """
        Return the highest-scoring recommendation for the requested objective.

        The compatibility method name predates its read-only behavior.
        """
        start_time = time.perf_counter()
        with self._lock:
            self.centralization_attempts += 1

        # Discover all centralization patterns
        patterns = self.discover_centralization_patterns(G)

        if not patterns:
            return CentralizationResult(
                discovered_patterns=[],
                optimal_strategy=CentralizationStrategy.SPECTRAL_DOMINANCE,
                recommended_topology={},
                performance_improvements={},
                coordination_efficiency=None,
                fault_tolerance=None,
                execution_time=time.perf_counter() - start_time,
                diagnostic_scores={},
            )

        # Select optimal strategy based on objective
        if objective == "efficiency":
            best_pattern = max(patterns, key=lambda p: p.efficiency_gain)
        elif objective == "stability":
            best_pattern = max(patterns, key=lambda p: p.stability_measure)
        else:  # balanced
            best_pattern = max(
                patterns, key=lambda p: p.efficiency_gain * p.stability_measure
            )

        # Generate recommendations
        recommended_topology = {
            "coordination_nodes": [
                node.node_id for node in best_pattern.coordination_nodes
            ],
            "coordination_strategy": best_pattern.strategy.value,
            "load_distribution": best_pattern.load_distribution,
            "cluster_assignments": {
                node.node_id: node.connected_cluster
                for node in best_pattern.coordination_nodes
            },
        }

        # Snapshot scores. No before/after execution is performed here, so the
        # compatibility improvement fields cannot carry measured claims.
        load_values = list(best_pattern.load_distribution.values())
        load_uniformity_score = (
            float(1.0 - np.var(load_values)) if load_values else None
        )
        coordination_redundancy_fraction = len(
            best_pattern.coordination_nodes
        ) / max(1, len(G.nodes()))
        diagnostic_scores = {
            "coordination_coverage_score": float(best_pattern.efficiency_gain),
            "centrality_stability_score": float(best_pattern.stability_measure),
            "load_distribution_uniformity_score": load_uniformity_score,
            "coordination_redundancy_fraction": coordination_redundancy_fraction,
        }
        performance_improvements: dict[str, float | None] = {}

        execution_time = time.perf_counter() - start_time

        # Coordinate cache hierarchy with selected pattern
        self._coordinate_cache_with_pattern(G, best_pattern)

        # Update internal state
        with self._lock:
            self.discovered_patterns = patterns
            self.current_coordination_nodes = {
                node.node_id: node for node in best_pattern.coordination_nodes
            }
            # Selection is recorded, but no execution-level success is inferred
            # from the heuristic coverage score.

        return CentralizationResult(
            discovered_patterns=patterns,
            optimal_strategy=best_pattern.strategy,
            recommended_topology=recommended_topology,
            performance_improvements=performance_improvements,
            coordination_efficiency=None,
            fault_tolerance=None,
            execution_time=execution_time,
            diagnostic_scores=diagnostic_scores,
        )

    def get_centralization_statistics(self) -> dict[str, Any]:
        """Get statistics about centralization analysis."""
        return {
            "centralization_attempts": self.centralization_attempts,
            "successful_centralizations": self.successful_centralizations,
            "success_rate": self.successful_centralizations
            / max(1, self.centralization_attempts),
            "current_coordination_nodes": len(self.current_coordination_nodes),
            "discovered_patterns": len(self.discovered_patterns),
            "adaptive_topology_requested": self.enable_adaptive_topology,
            "adaptive_topology_applied": False,
            "thresholds": {
                "centrality": self.centrality_threshold,
                "coordination": self.coordination_threshold,
                "stability": self.stability_threshold,
            },
            "available_modules": {
                "networkx": HAS_NETWORKX,
                "spectral": HAS_SPECTRAL,
                "spectral_structural_fusion": HAS_SPECTRAL_STRUCTURAL_FUSION,
            },
        }


# Factory functions
def create_emergent_centralization_engine(
    **kwargs: Any,
) -> TNFREmergentCentralizationEngine:
    """Create emergent centralization engine."""
    return TNFREmergentCentralizationEngine(**kwargs)


def optimize_network_centralization(
    G: Any, objective: str = "efficiency", **kwargs: Any
) -> CentralizationResult:
    """Convenience function for network centralization optimization."""
    engine = create_emergent_centralization_engine(**kwargs)
    return engine.optimize_centralization(G, objective)


def discover_coordination_nodes(G: Any) -> list[CentralizationNode]:
    """Convenience function to discover coordination nodes."""
    engine = create_emergent_centralization_engine()
    patterns = engine.discover_centralization_patterns(G)

    all_coordination_nodes = []
    for pattern in patterns:
        all_coordination_nodes.extend(pattern.coordination_nodes)

    return all_coordination_nodes
