"""
TNFR Emergent Centralization Engine

This module implements intelligent centralization patterns that emerge naturally
from the mathematical structure of the nodal equation ∂EPI/∂t = νf · ΔNFR(t).

Mathematical Foundation:
The nodal equation reveals natural centralization principles:

1. **Information Concentration**: EPI naturally flows to network centers
2. **Frequency Synchronization**: High-νf nodes become natural coordinators  
3. **ΔNFR Equilibration**: Computation load balances across optimal topologies
4. **Spectral Coordination**: Eigenmode structure defines natural hierarchies
5. **Phase-Locked Networks**: Synchronous regions form computational clusters
6. **Adaptive Topologies**: Network structure evolves to optimize computation

Emergent Centralization Features:
- Automatic discovery of computational coordination points
- Dynamic load redistribution based on mathematical properties
- Self-organizing computational hierarchies
- Natural fault tolerance through mathematical redundancy
- Adaptive resource allocation using spectral structure
- Emergent consensus mechanisms via phase locking

Status: CANONICAL EMERGENT CENTRALIZATION ENGINE
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time
import threading

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import TNFR components
HAS_TNFR_ENGINES = True  # Assume available

HAS_PHYSICS_FIELDS = True  # Assume available

try:
    from ..mathematics.spectral import get_laplacian_spectrum
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# PHASE 6 EXTENDED: Canonical constants for emergent centralization
from ..constants.canonical import (
    EMERGENT_COUPLING_STRENGTH_CANONICAL,  # φ/(π+γ) ≈ 0.7320 (0.7 → canonical)
    EMERGENT_FREQ_BALANCE_CANONICAL,       # e/(π+e) ≈ 0.4638 (0.5 → canonical) 
    NODAL_OPT_COUPLING_CANONICAL,          # γ/(π+e) ≈ 0.0985 (0.1 → canonical)
    EMERGENT_EFFICIENCY_GAIN_CANONICAL,    # γ/π ≈ 0.1837 (0.2 → canonical)
    EMERGENT_COORDINATION_BOOST_CANONICAL, # 2·φ/π ≈ 1.0309 (2.0 → canonical)
)

try:
    from .spectral_structural_fusion import TNFRSpectralStructuralFusionEngine
    HAS_SPECTRAL_STRUCTURAL_FUSION = True
except ImportError:
    HAS_SPECTRAL_STRUCTURAL_FUSION = False


class CentralizationStrategy(Enum):
    """Strategies for emergent centralization."""
    SPECTRAL_DOMINANCE = "spectral_dominance"      # Based on eigenmode centrality
    INFORMATION_FLOW = "information_flow"          # Based on EPI concentration
    FREQUENCY_HIERARCHY = "frequency_hierarchy"    # Based on νf values
    LOAD_BALANCING = "load_balancing"             # Based on computational load
    PHASE_COORDINATION = "phase_coordination"      # Based on phase synchronization
    ADAPTIVE_TOPOLOGY = "adaptive_topology"        # Based on dynamic restructuring


@dataclass
class CentralizationNode:
    """A node identified as a coordination center."""
    node_id: Any
    centrality_score: float
    coordination_capacity: float
    current_load: float
    specialization: str  # Type of coordination this node excels at
    connected_cluster: List[Any]  # Nodes coordinated by this center
    mathematical_signature: Dict[str, Any]


@dataclass
class CentralizationPattern:
    """Discovered centralization pattern in the network."""
    strategy: CentralizationStrategy
    coordination_nodes: List[CentralizationNode]
    efficiency_gain: float
    stability_measure: float
    adaptation_rate: float
    mathematical_basis: Dict[str, Any]
    load_distribution: Dict[Any, float]


@dataclass
class CentralizationResult:
    """Result of centralization analysis and optimization."""
    discovered_patterns: List[CentralizationPattern]
    optimal_strategy: CentralizationStrategy
    recommended_topology: Dict[str, Any]
    performance_improvements: Dict[str, float]
    coordination_efficiency: float
    fault_tolerance: float
    execution_time: float


class TNFREmergentCentralizationEngine:
    """
    Engine for discovering and implementing emergent centralization patterns.
    
    This engine analyzes the mathematical structure of TNFR networks to discover
    natural coordination and centralization opportunities.
    """
    
    def __init__(self, enable_adaptive_topology: bool = True):
        self.enable_adaptive_topology = enable_adaptive_topology
        
        # Centralization state
        self.discovered_patterns = []
        self.current_coordination_nodes = {}
        self.performance_history = []
        
        # Mathematical thresholds (canonical derivations from φ, γ, π, e)
        self.centrality_threshold = 0.7370610757229365  # φ/(φ+γ) ≈ 0.737
        self.coordination_threshold = 0.5550110513847934  # 1/(φ + γ/π) ≈ 0.555
        self.stability_threshold = 0.5903096618115984  # (φ+γ)/(π+γ) ≈ 0.590
        
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

    def _coordinate_cache_with_pattern(self, G: Any, pattern: CentralizationPattern) -> None:
        """Delegate cache coordination to the fusion engine using pattern nodes."""
        if self.fusion_engine is None:
            return

        self.fusion_engine.coordinate_cache_with_central_nodes(
            G,
            pattern.coordination_nodes,
            strategy=pattern.strategy.value,
        )
        
    def analyze_spectral_centralization(
        self,
        G: Any
    ) -> List[CentralizationNode]:
        """
        Discover centralization based on spectral properties.
        
        Uses eigenvector centrality and spectral structure to identify
        natural coordination points.
        """
        coordination_nodes = []
        
        if not HAS_NETWORKX or not HAS_SPECTRAL or G is None:
            return coordination_nodes
            
        self._prefetch_spectral_state(G)

        # Get spectral decomposition
        eigenvalues, eigenvectors = get_laplacian_spectrum(G)
        
        # Calculate eigenvector centrality from dominant eigenvector
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
                        epi_value = G.nodes[node].get('EPI', 0.0)
                        vf_value = G.nodes[node].get('vf', 1.0)
                        
                        # Mathematical signature for this coordination node
                        signature = {
                            "spectral_centrality": centrality,
                            "fiedler_component": fiedler_vector[i],
                            "degree": degree,
                            "epi": epi_value,
                            "vf": vf_value,
                            "eigenvalue_proximity": min(abs(eigenvalues - vf_value)) if len(eigenvalues) > 0 else 0
                        }
                        
                        # Find connected cluster
                        neighbors = list(G.neighbors(node))
                        cluster = [node] + neighbors[:int(degree * EMERGENT_COUPLING_STRENGTH_CANONICAL)]  # Include most connected neighbors
                        
                        coord_node = CentralizationNode(
                            node_id=node,
                            centrality_score=centrality,
                            coordination_capacity=degree * centrality * vf_value,
                            current_load=0.0,  # Will be updated during operation
                            specialization="spectral_coordination",
                            connected_cluster=cluster,
                            mathematical_signature=signature
                        )
                        coordination_nodes.append(coord_node)
                        
        return coordination_nodes
        
    def analyze_information_flow_centralization(
        self,
        G: Any
    ) -> List[CentralizationNode]:
        """
        Discover centralization based on information (EPI) flow patterns.
        
        Identifies nodes that naturally accumulate or distribute information.
        """
        coordination_nodes = []
        
        if not HAS_NETWORKX or G is None:
            return coordination_nodes
            
        # Analyze EPI distribution and flow
        epi_values = {node: G.nodes[node].get('EPI', 0.0) for node in G.nodes()}
        total_epi = sum(abs(epi) for epi in epi_values.values())
        
        if total_epi > 0:
            for node in G.nodes():
                epi = abs(epi_values[node])
                epi_fraction = epi / total_epi
                
                # High EPI concentration indicates coordination potential
                if epi_fraction > 0.09850273565687083:  # γ/(π+e) ≈ 0.099 - Significant EPI concentration
                    # Analyze information flow capacity
                    neighbors = list(G.neighbors(node))
                    neighbor_epi = [abs(epi_values.get(n, 0.0)) for n in neighbors]
                    
                    # Information gradient (how much EPI difference with neighbors)
                    info_gradient = sum(abs(epi - nepi) for nepi in neighbor_epi) / max(1, len(neighbor_epi))
                    
                    # Coordination capacity based on information processing
                    vf_value = G.nodes[node].get('vf', 1.0)
                    coordination_capacity = epi_fraction * info_gradient * vf_value
                    
                    if coordination_capacity > self.coordination_threshold:
                        signature = {
                            "epi_concentration": epi_fraction,
                            "information_gradient": info_gradient,
                            "total_information": epi,
                            "neighbor_count": len(neighbors),
                            "vf": vf_value,
                            "processing_capacity": coordination_capacity
                        }
                        
                        # Connected cluster based on information similarity
                        similar_nodes = [
                            n
                            for n in neighbors
                            if abs(epi_values.get(n, 0.0) - epi) < info_gradient * EMERGENT_FREQ_BALANCE_CANONICAL
                        ]
                        cluster = [node] + similar_nodes
                        
                        coord_node = CentralizationNode(
                            node_id=node,
                            centrality_score=epi_fraction,
                            coordination_capacity=coordination_capacity,
                            current_load=0.0,
                            specialization="information_coordination",
                            connected_cluster=cluster,
                            mathematical_signature=signature
                        )
                        coordination_nodes.append(coord_node)
                        
        return coordination_nodes
        
    def analyze_frequency_hierarchy_centralization(
        self,
        G: Any
    ) -> List[CentralizationNode]:
        """
        Discover centralization based on frequency (νf) hierarchy.
        
        High-frequency nodes naturally become coordinators.
        """
        coordination_nodes = []
        
        if not HAS_NETWORKX or G is None:
            return coordination_nodes
            
        # Analyze νf distribution
        vf_values = {node: G.nodes[node].get('vf', 1.0) for node in G.nodes()}
        max_vf = max(vf_values.values()) if vf_values else 1.0
        
        # High-frequency nodes become natural coordinators
        for node in G.nodes():
            vf = vf_values[node]
            relative_frequency = vf / max_vf if max_vf > 0 else 0
            
            if relative_frequency > 0.5903096618115984:  # (φ+γ)/(π+γ) ≈ 0.590 - Top frequency nodes
                # Calculate coordination capacity based on frequency advantage
                neighbors = list(G.neighbors(node))
                neighbor_vf = [vf_values.get(n, 1.0) for n in neighbors]
                
                # Frequency dominance over neighbors
                frequency_advantage = sum(max(0, vf - nvf) for nvf in neighbor_vf) / max(1, len(neighbor_vf))
                
                degree = G.degree(node)
                coordination_capacity = relative_frequency * frequency_advantage * degree
                
                if coordination_capacity > self.coordination_threshold:
                    signature = {
                        "relative_frequency": relative_frequency,
                        "absolute_frequency": vf,
                        "frequency_advantage": frequency_advantage,
                        "degree": degree,
                        "neighbor_frequencies": neighbor_vf,
                        "synchronization_potential": min(neighbor_vf) / vf if neighbor_vf and vf > 0 else 0
                    }
                    
                    # Cluster includes nodes that can synchronize with this frequency
                    sync_threshold = vf * EMERGENT_COUPLING_STRENGTH_CANONICAL  # Within 30% of coordinator frequency
                    sync_neighbors = [
                        n for n in neighbors if vf_values.get(n, 1.0) >= sync_threshold
                    ]
                    cluster = [node] + sync_neighbors
                    
                    coord_node = CentralizationNode(
                        node_id=node,
                        centrality_score=relative_frequency,
                        coordination_capacity=coordination_capacity,
                        current_load=0.0,
                        specialization="frequency_coordination",
                        connected_cluster=cluster,
                        mathematical_signature=signature
                    )
                    coordination_nodes.append(coord_node)
                    
        return coordination_nodes
        
    def analyze_phase_coordination_centralization(
        self,
        G: Any
    ) -> List[CentralizationNode]:
        """
        Discover centralization based on phase synchronization potential.
        
        Nodes that can coordinate phase across the network become centers.
        """
        coordination_nodes = []
        
        if not HAS_NETWORKX or G is None:
            return coordination_nodes
            
        # Analyze phase distribution
        phase_values = {node: G.nodes[node].get('phase', 0.0) for node in G.nodes()}
        
        for node in G.nodes():
            phase = phase_values[node]
            neighbors = list(G.neighbors(node))
            
            if len(neighbors) > 2:  # Need sufficient connections for coordination
                neighbor_phases = [phase_values.get(n, 0.0) for n in neighbors]
                
                # Calculate phase coherence with neighbors
                phase_differences = [abs(phase - nphase) for nphase in neighbor_phases]
                avg_phase_diff = np.mean(phase_differences)
                phase_coherence = 1.0 / (1.0 + avg_phase_diff)  # Higher coherence = lower differences
                
                # Phase coordination capacity
                vf = G.nodes[node].get('vf', 1.0)
                coordination_capacity = phase_coherence * len(neighbors) * vf
                
                if phase_coherence > 0.7370610757229365 and coordination_capacity > self.coordination_threshold:  # φ/(φ+γ) ≈ 0.737
                    signature = {
                        "phase_coherence": phase_coherence,
                        "average_phase_difference": avg_phase_diff,
                        "neighbor_count": len(neighbors),
                        "vf": vf,
                        "phase": phase,
                        "synchronization_strength": coordination_capacity
                    }
                    
                    # Cluster includes phase-synchronized neighbors
                    sync_threshold = np.pi / 4  # Within 45 degrees
                    sync_neighbors = [
                        n
                        for n, nphase in zip(neighbors, neighbor_phases)
                        if abs(phase - nphase) < sync_threshold
                    ]
                    cluster = [node] + sync_neighbors
                    
                    coord_node = CentralizationNode(
                        node_id=node,
                        centrality_score=phase_coherence,
                        coordination_capacity=coordination_capacity,
                        current_load=0.0,
                        specialization="phase_coordination",
                        connected_cluster=cluster,
                        mathematical_signature=signature
                    )
                    coordination_nodes.append(coord_node)
                    
        return coordination_nodes
        
    def discover_centralization_patterns(
        self,
        G: Any
    ) -> List[CentralizationPattern]:
        """
        Discover all centralization patterns in the network.
        """
        patterns = []
        
        # Analyze each centralization strategy
        strategies = [
            (CentralizationStrategy.SPECTRAL_DOMINANCE, self.analyze_spectral_centralization),
            (CentralizationStrategy.INFORMATION_FLOW, self.analyze_information_flow_centralization),
            (CentralizationStrategy.FREQUENCY_HIERARCHY, self.analyze_frequency_hierarchy_centralization),
            (CentralizationStrategy.PHASE_COORDINATION, self.analyze_phase_coordination_centralization)
        ]
        
        for strategy, analyzer in strategies:
            coordination_nodes = analyzer(G)
            
            if coordination_nodes:
                # Calculate pattern metrics
                total_capacity = sum(node.coordination_capacity for node in coordination_nodes)
                avg_centrality = np.mean([node.centrality_score for node in coordination_nodes])
                
                # Efficiency gain estimate (more coordination nodes = better load distribution)
                efficiency_gain = min(len(coordination_nodes) / len(G.nodes()) * EMERGENT_COORDINATION_BOOST_CANONICAL, 1.0)
                
                # Stability measure (higher centrality = more stable)
                stability_measure = avg_centrality
                
                # Mathematical basis
                mathematical_basis = {
                    "coordination_node_count": len(coordination_nodes),
                    "total_coordination_capacity": total_capacity,
                    "average_centrality": avg_centrality,
                    "coverage_fraction": len(set().union(*[node.connected_cluster for node in coordination_nodes])) / len(G.nodes())
                }
                
                # Load distribution across coordination nodes
                if total_capacity > 0:
                    load_distribution = {
                        node.node_id: node.coordination_capacity / total_capacity
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
                    load_distribution=load_distribution
                )
                patterns.append(pattern)
                
        return patterns
        
    def optimize_centralization(
        self,
        G: Any,
        objective: str = "efficiency"
    ) -> CentralizationResult:
        """
        Optimize network centralization for the given objective.
        """
        start_time = time.perf_counter()
        
        # Discover all centralization patterns
        patterns = self.discover_centralization_patterns(G)
        
        if not patterns:
            return CentralizationResult(
                discovered_patterns=[],
                optimal_strategy=CentralizationStrategy.SPECTRAL_DOMINANCE,
                recommended_topology={},
                performance_improvements={},
                coordination_efficiency=0.0,
                fault_tolerance=0.0,
                execution_time=time.perf_counter() - start_time
            )
            
        # Select optimal strategy based on objective
        if objective == "efficiency":
            best_pattern = max(patterns, key=lambda p: p.efficiency_gain)
        elif objective == "stability":
            best_pattern = max(patterns, key=lambda p: p.stability_measure)
        else:  # balanced
            best_pattern = max(patterns, key=lambda p: p.efficiency_gain * p.stability_measure)
            
        # Generate recommendations
        recommended_topology = {
            "coordination_nodes": [node.node_id for node in best_pattern.coordination_nodes],
            "coordination_strategy": best_pattern.strategy.value,
            "load_distribution": best_pattern.load_distribution,
            "cluster_assignments": {
                node.node_id: node.connected_cluster
                for node in best_pattern.coordination_nodes
            }
        }
        
        # Calculate performance improvements
        performance_improvements = {
            "coordination_efficiency": best_pattern.efficiency_gain,
            "stability_improvement": best_pattern.stability_measure,
            "load_balance_improvement": float(
                1.0 - np.var(list(best_pattern.load_distribution.values()))
            ),
        }
        
        # Calculate fault tolerance (redundancy in coordination)
        fault_tolerance = len(best_pattern.coordination_nodes) / max(1, len(G.nodes()))
        
        execution_time = time.perf_counter() - start_time

        # Coordinate cache hierarchy with selected pattern
        self._coordinate_cache_with_pattern(G, best_pattern)
        
        # Update internal state
        with self._lock:
            self.discovered_patterns = patterns
            self.current_coordination_nodes = {
                node.node_id: node for node in best_pattern.coordination_nodes
            }
            self.centralization_attempts += 1
            if best_pattern.efficiency_gain > EMERGENT_EFFICIENCY_GAIN_CANONICAL:
                self.successful_centralizations += 1
                
        return CentralizationResult(
            discovered_patterns=patterns,
            optimal_strategy=best_pattern.strategy,
            recommended_topology=recommended_topology,
            performance_improvements=performance_improvements,
            coordination_efficiency=best_pattern.efficiency_gain,
            fault_tolerance=fault_tolerance,
            execution_time=execution_time
        )
        
    def get_centralization_statistics(self) -> Dict[str, Any]:
        """Get statistics about centralization analysis."""
        return {
            "centralization_attempts": self.centralization_attempts,
            "successful_centralizations": self.successful_centralizations,
            "success_rate": self.successful_centralizations / max(1, self.centralization_attempts),
            "current_coordination_nodes": len(self.current_coordination_nodes),
            "discovered_patterns": len(self.discovered_patterns),
            "adaptive_topology_enabled": self.enable_adaptive_topology,
            "thresholds": {
                "centrality": self.centrality_threshold,
                "coordination": self.coordination_threshold,
                "stability": self.stability_threshold
            },
            "available_modules": {
                "networkx": HAS_NETWORKX,
                "spectral": HAS_SPECTRAL,
                "physics_fields": HAS_PHYSICS_FIELDS,
                "tnfr_engines": HAS_TNFR_ENGINES
            }
        }


# Factory functions
def create_emergent_centralization_engine(**kwargs: Any) -> TNFREmergentCentralizationEngine:
    """Create emergent centralization engine."""
    return TNFREmergentCentralizationEngine(**kwargs)


def optimize_network_centralization(
    G: Any,
    objective: str = "efficiency",
    **kwargs: Any
) -> CentralizationResult:
    """Convenience function for network centralization optimization."""
    engine = create_emergent_centralization_engine(**kwargs)
    return engine.optimize_centralization(G, objective)


def discover_coordination_nodes(G: Any) -> List[CentralizationNode]:
    """Convenience function to discover coordination nodes."""
    engine = create_emergent_centralization_engine()
    patterns = engine.discover_centralization_patterns(G)
    
    all_coordination_nodes = []
    for pattern in patterns:
        all_coordination_nodes.extend(pattern.coordination_nodes)
        
    return all_coordination_nodes
