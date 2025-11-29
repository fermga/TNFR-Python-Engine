"""
TNFR Self-Optimizing Mathematical Engine

This module implements self-optimization that emerges naturally from analyzing
the mathematical structure of the nodal equation ∂EPI/∂t = νf · ΔNFR(t).

Mathematical Foundation:
The nodal equation reveals natural optimization landscapes:

1. **Gradient Flows**: ΔNFR naturally defines optimization directions
2. **Energy Functionals**: EPI configurations have natural energy measures
3. **Constraint Manifolds**: Grammar rules create constraint manifolds
4. **Variational Principles**: Operator sequences minimize action functionals
5. **Learning Dynamics**: Repeated patterns improve through experience
6. **Adaptive Algorithms**: The system learns optimal strategies automatically

Self-Optimization Mechanisms:
- Automatic cache strategy learning based on mathematical importance
- Dynamic operator sequence optimization using variational principles
- Adaptive precision management based on mathematical requirements
- Self-tuning computational backend selection
- Emergent load balancing through mathematical analysis
- Natural parallelization discovery via spectral decomposition

Status: CANONICAL SELF-OPTIMIZING ENGINE
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque
import threading

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

HAS_SCIPY = True  # Assume available for mathematical analysis

# Import canonical constants for Phase 6 magic number elimination
from ...constants.canonical import (
    SELF_OPT_CACHE_SIZE_CANONICAL,
    SELF_OPT_COMPRESSION_HIGH_CANONICAL,
    SELF_OPT_HORIZON_HIGH_CANONICAL,
    SELF_OPT_CHIRALITY_THRESHOLD_CANONICAL,
    SELF_OPT_SYMMETRY_THRESHOLD_CANONICAL,
    SELF_OPT_COUPLING_LOW_CANONICAL,
    SELF_OPT_CHARGE_THRESHOLD_CANONICAL,
    SELF_OPT_ENERGY_HIGH_CANONICAL,
    SELF_OPT_EPI_VARIANCE_LOW_CANONICAL,
    SELF_OPT_VF_RANGE_LOW_CANONICAL,
    SELF_OPT_DNFR_HIGH_CANONICAL,
    SELF_OPT_DENSITY_SPARSE_CANONICAL,
    SELF_OPT_DENSITY_DENSE_CANONICAL,
    SELF_OPT_IMPROVEMENT_SIGNIFICANT_CANONICAL,
    SELF_OPT_CACHE_LOW_FRACTION_CANONICAL,
    SELF_OPT_SPEEDUP_HIGH_CANONICAL,
    SELF_OPT_CACHE_EXPANSION_CANONICAL,
    SELF_OPT_CACHE_HIGH_FRACTION_CANONICAL,
    SELF_OPT_SPEEDUP_LOW_CANONICAL,
    SELF_OPT_CACHE_CONTRACTION_CANONICAL,
    # PHASE 6 EXTENDED: Additional constants for remaining magic numbers
    PI,                               # π ≈ 3.1416 (3.0 → canonical)
    NODAL_OPT_COUPLING_CANONICAL,     # γ/(π+e) ≈ 0.0985 (0.1 → canonical)
)


def _extract_scalar_epi(val: Any) -> float:
    """Extract scalar magnitude from potentially complex/dict EPI value."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, complex):
        return float(np.abs(val))
    if isinstance(val, dict):
        if 'continuous' in val:
            c = val['continuous']
            if isinstance(c, (tuple, list)) and len(c) > 0:
                v = c[0]
                return float(np.abs(v)) if isinstance(v, complex) else float(v)
    return 0.0


# Import Unified Fields (New Nov 2025)
try:
    from ..physics.fields import (
        compute_unified_telemetry,
        compute_complex_geometric_field,
        compute_tensor_invariants
    )
    HAS_UNIFIED_FIELDS = True
except ImportError:
    HAS_UNIFIED_FIELDS = False

# Import TNFR engines
try:
    from .unified_backend import TNFRUnifiedBackend
    from .optimization_orchestrator import TNFROptimizationOrchestrator, OptimizationStrategy
    from .emergent_mathematical_patterns import TNFREmergentPatternEngine, EmergentPatternType
    HAS_ENGINES = True
except ImportError:
    HAS_ENGINES = False

HAS_MATH_BACKENDS = True  # Assume available


class OptimizationObjective(Enum):
    """Self-optimization objectives."""
    MINIMIZE_COMPUTATION_TIME = "minimize_time"
    MAXIMIZE_CACHE_EFFICIENCY = "maximize_cache"
    MINIMIZE_MEMORY_USAGE = "minimize_memory"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_ALL = "balance_all"


class LearningStrategy(Enum):
    """Learning strategies for optimization."""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    BAYESIAN_OPTIMIZATION = "bayesian"
    MATHEMATICAL_ANALYSIS = "mathematical"
    HYBRID = "hybrid"


@dataclass
class OptimizationExperience:
    """Experience record for learning."""
    graph_properties: Dict[str, Any]
    operation_type: str
    strategy_used: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: float
    success: bool
    mathematical_signature: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationPolicy:
    """Learned optimization policy."""
    policy_name: str
    objective: OptimizationObjective
    conditions: Dict[str, Any]  # When to apply this policy
    actions: Dict[str, Any]     # What to do
    confidence: float
    success_rate: float
    average_improvement: float
    applications_count: int = 0


@dataclass
class SelfOptimizationResult:
    """Result of self-optimization analysis."""
    learned_policies: List[OptimizationPolicy]
    optimization_improvements: Dict[str, float]
    recommended_strategies: List[str]
    mathematical_insights: Dict[str, Any]
    predicted_speedups: Dict[str, float]
    adaptive_configurations: Dict[str, Any]
    execution_time: float


class TNFRSelfOptimizingEngine:
    """
    Self-optimizing engine that learns optimal strategies from mathematical structure.
    
    This engine discovers optimization patterns by analyzing the mathematical
    properties of the nodal equation and learning from experience.
    """
    
    def __init__(
        self,
        learning_strategy: LearningStrategy = LearningStrategy.MATHEMATICAL_ANALYSIS,
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL,
        max_experience_history: int = 1000
    ):
        self.learning_strategy = learning_strategy
        self.optimization_objective = optimization_objective
        self.max_experience_history = max_experience_history
        
        # Learning state
        self.experience_history: deque = deque(maxlen=max_experience_history)
        self.learned_policies: List[OptimizationPolicy] = []
        self.mathematical_insights: Dict[str, Any] = {}
        
        # Performance tracking
        self.optimization_attempts = 0
        self.successful_optimizations = 0
        self.cumulative_improvements = defaultdict(float)
        
        # Adaptive configuration
        self.adaptive_config = {
            "cache_size_mb": 256.0,
            "precision": "float64",
            "backend_preference": "numpy",
            "parallel_threshold": 50,
            "spectral_threshold": 20
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize engines if available
        if HAS_ENGINES:
            self.unified_backend = TNFRUnifiedBackend()
            self.orchestrator = TNFROptimizationOrchestrator()
            self.pattern_engine = TNFREmergentPatternEngine()
        else:
            self.unified_backend = None
            self.orchestrator = None
            self.pattern_engine = None
            
    def analyze_mathematical_optimization_landscape(
        self,
        G: Any,
        operation_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze mathematical structure to discover optimization opportunities.
        
        Uses the mathematical properties of the nodal equation to identify
        natural optimization strategies.
        """
        insights = {}
        
        if not HAS_NETWORKX or G is None:
            return insights
            
        # Basic graph properties
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        # Unified Field Analysis (New Nov 2025)
        unified_insights = {}
        if HAS_UNIFIED_FIELDS:
            try:
                # Compute full unified telemetry
                telemetry = compute_unified_telemetry(G)
                
                # Extract Complex Field
                complex_data = telemetry.get("complex_field", {})
                psi_mag_array = complex_data.get("magnitude", np.array([]))
                psi_scalar = np.mean(psi_mag_array) if len(psi_mag_array) > 0 else 0.0
                
                # Extract Emergent Fields
                emergent_data = telemetry.get("emergent_fields", {})
                chi_array = emergent_data.get("chirality", np.array([]))
                sb_array = emergent_data.get("symmetry_breaking", np.array([]))
                cc_array = emergent_data.get("coherence_coupling", np.array([]))
                
                chi_scalar = np.mean(np.abs(chi_array)) if len(chi_array) > 0 else 0.0
                sb_scalar = np.mean(sb_array) if len(sb_array) > 0 else 0.0
                cc_scalar = np.mean(cc_array) if len(cc_array) > 0 else 0.0
                
                # Extract Tensor Invariants
                tensor_data = telemetry.get("tensor_invariants", {})
                ed_array = tensor_data.get("energy_density", np.array([]))
                tc_array = tensor_data.get("topological_charge", np.array([]))
                
                ed_scalar = np.mean(ed_array) if len(ed_array) > 0 else 0.0
                tc_scalar = np.mean(np.abs(tc_array)) if len(tc_array) > 0 else 0.0
                
                unified_insights = {
                    "psi_magnitude": float(psi_scalar),
                    "chirality": float(chi_scalar),
                    "symmetry_breaking": float(sb_scalar),
                    "coherence_coupling": float(cc_scalar),
                    "energy_density": float(ed_scalar),
                    "topological_charge": float(tc_scalar)
                }
                
                insights["unified_field_analysis"] = unified_insights
            except Exception as e:
                # Fallback if computation fails
                insights["unified_field_error"] = str(e)

        # Mathematical structure analysis
        insights["graph_structure"] = {
            "nodes": num_nodes,
            "edges": num_edges,
            "density": density,
            "is_connected": nx.is_connected(G) if HAS_NETWORKX else False,
            "avg_degree": 2 * num_edges / num_nodes if num_nodes > 0 else 0
        }
        
        # Spectral properties for optimization
        if self.pattern_engine:
            pattern_result = self.pattern_engine.discover_all_patterns(G)
            
            # Extract optimization hints from discovered patterns
            optimization_hints = []
            for pattern in pattern_result.discovered_patterns:
                if pattern.compression_ratio > SELF_OPT_COMPRESSION_HIGH_CANONICAL:
                    optimization_hints.append(f"use_compression_{pattern.pattern_type.value}")
                if pattern.prediction_horizon > PI:
                    optimization_hints.append(f"use_prediction_{pattern.pattern_type.value}")
                if pattern.pattern_type == EmergentPatternType.EIGENMODE_RESONANCE:
                    optimization_hints.append("use_spectral_methods")
                if pattern.pattern_type == EmergentPatternType.FRACTAL_SCALING:
                    optimization_hints.append("use_hierarchical_methods")
                    
            insights["pattern_optimization_hints"] = optimization_hints
            insights["mathematical_patterns"] = len(pattern_result.discovered_patterns)
            insights["compression_potential"] = pattern_result.compression_potential
            
        # Nodal equation analysis
        epi_values = [_extract_scalar_epi(G.nodes[node].get('EPI', 0.0)) for node in G.nodes()]
        vf_values = [G.nodes[node].get('vf', 1.0) for node in G.nodes()]
        dnfr_values = [G.nodes[node].get('DNFR', 0.0) for node in G.nodes()]
        
        # Mathematical properties for optimization
        epi_variance = np.var(epi_values)
        vf_range = np.max(vf_values) - np.min(vf_values) if vf_values else 0
        dnfr_magnitude = np.mean(np.abs(dnfr_values)) if dnfr_values else 0
        
        # Optimization recommendations based on mathematical properties
        recommendations = []
        
        # Unified Field Recommendations (New Nov 2025)
        if "unified_field_analysis" in insights:
            ufa = insights["unified_field_analysis"]
            
            # Chirality-based optimization
            if abs(ufa.get("chirality", 0)) > SELF_OPT_CHIRALITY_THRESHOLD_CANONICAL:
                recommendations.append("use_chiral_optimization")
                
            # Symmetry breaking handling
            if ufa.get("symmetry_breaking", 0) > SELF_OPT_SYMMETRY_THRESHOLD_CANONICAL:
                recommendations.append("use_phase_transition_handling")
                
            # Coherence coupling optimization
            if ufa.get("coherence_coupling", 0) < SELF_OPT_COUPLING_LOW_CANONICAL:
                recommendations.append("enhance_coherence_coupling")
                
            # Topological charge handling
            if abs(ufa.get("topological_charge", 0)) > NODAL_OPT_COUPLING_CANONICAL:
                recommendations.append("topological_defect_correction")
                
            # Energy density optimization
            if ufa.get("energy_density", 0) > SELF_OPT_ENERGY_HIGH_CANONICAL:
                recommendations.append("high_energy_stabilization")

        if epi_variance < 0.01:
            recommendations.append("low_variance_epi_optimization")
        if vf_range < NODAL_OPT_COUPLING_CANONICAL:
            recommendations.append("uniform_vf_optimization")
        if dnfr_magnitude > 1.0:
            recommendations.append("high_dnfr_stabilization")
        if num_nodes > 100:
            recommendations.append("large_graph_optimization")
        if density > SELF_OPT_DENSITY_DENSE_CANONICAL:
            recommendations.append("dense_graph_optimization")
        elif density < SELF_OPT_DENSITY_SPARSE_CANONICAL:
            recommendations.append("sparse_graph_optimization")
            
        insights["nodal_equation_analysis"] = {
            "epi_variance": epi_variance,
            "vf_range": vf_range,
            "dnfr_magnitude": dnfr_magnitude,
            "optimization_recommendations": recommendations
        }
        
        return insights
        
    def learn_from_experience(
        self,
        experience: OptimizationExperience
    ) -> None:
        """
        Learn optimization strategies from performance experience.
        
        Uses mathematical analysis to extract general principles.
        """
        with self._lock:
            self.experience_history.append(experience)
            self.optimization_attempts += 1
            if experience.success:
                self.successful_optimizations += 1
                
        # Analyze experience for patterns
        if len(self.experience_history) >= 10:  # Minimum data for learning
            self._extract_optimization_policies()
            self._update_adaptive_configuration()
            
    def _extract_optimization_policies(self) -> None:
        """Extract general optimization policies from experience."""
        # Group experiences by similar conditions
        condition_groups = defaultdict(list)
        
        for exp in self.experience_history:
            if exp.success:
                # Create condition signature
                graph_size = exp.graph_properties.get("nodes", 0)
                density = exp.graph_properties.get("density", 0.0)
                operation = exp.operation_type
                
                # Discretize conditions for pattern recognition
                size_bucket = "small" if graph_size < 20 else "medium" if graph_size < 100 else "large"
                density_bucket = "sparse" if density < SELF_OPT_DENSITY_SPARSE_CANONICAL else "medium" if density < SELF_OPT_DENSITY_DENSE_CANONICAL else "dense"
                
                condition_key = (size_bucket, density_bucket, operation)
                condition_groups[condition_key].append(exp)
                
        # Extract policies from groups with sufficient data
        new_policies = []
        for condition_key, experiences in condition_groups.items():
            if len(experiences) >= 3:  # Minimum for reliable pattern
                size_bucket, density_bucket, operation = condition_key
                
                # Find best strategy for this condition
                strategy_performance = defaultdict(list)
                for exp in experiences:
                    strategy = exp.strategy_used
                    improvement = exp.performance_metrics.get("speedup_factor", 1.0)
                    strategy_performance[strategy].append(improvement)
                    
                # Select best strategy
                best_strategy = None
                best_avg_improvement = 0
                for strategy, improvements in strategy_performance.items():
                    avg_improvement = np.mean(improvements)
                    if avg_improvement > best_avg_improvement:
                        best_avg_improvement = avg_improvement
                        best_strategy = strategy
                        
                if best_strategy and best_avg_improvement > SELF_OPT_IMPROVEMENT_SIGNIFICANT_CANONICAL:  # Significant improvement
                    policy = OptimizationPolicy(
                        policy_name=f"{size_bucket}_{density_bucket}_{operation}_policy",
                        objective=self.optimization_objective,
                        conditions={
                            "graph_size_bucket": size_bucket,
                            "density_bucket": density_bucket,
                            "operation_type": operation
                        },
                        actions={
                            "recommended_strategy": best_strategy,
                            "expected_improvement": best_avg_improvement
                        },
                        confidence=min(len(experiences) / 10.0, 1.0),
                        success_rate=len(experiences) / len([e for e in self.experience_history 
                                                           if self._matches_conditions(e, condition_key)]),
                        average_improvement=best_avg_improvement
                    )
                    new_policies.append(policy)
                    
        # Update learned policies
        self.learned_policies.extend(new_policies)
        
        # Remove outdated policies (keep only best 20)
        if len(self.learned_policies) > 20:
            self.learned_policies.sort(key=lambda p: p.confidence * p.average_improvement, reverse=True)
            self.learned_policies = self.learned_policies[:20]
            
    def _matches_conditions(self, experience: OptimizationExperience, condition_key: Tuple) -> bool:
        """Check if experience matches condition key."""
        size_bucket, density_bucket, operation = condition_key
        
        graph_size = experience.graph_properties.get("nodes", 0)
        density = experience.graph_properties.get("density", 0.0)
        
        exp_size_bucket = "small" if graph_size < 20 else "medium" if graph_size < 100 else "large"
        exp_density_bucket = "sparse" if density < SELF_OPT_DENSITY_SPARSE_CANONICAL else "medium" if density < SELF_OPT_DENSITY_DENSE_CANONICAL else "dense"
        
        return (exp_size_bucket == size_bucket and 
                exp_density_bucket == density_bucket and
                experience.operation_type == operation)
        
    def _update_adaptive_configuration(self) -> None:
        """Update adaptive configuration based on learned patterns."""
        if not self.experience_history:
            return
            
        # Analyze recent performance
        recent_experiences = list(self.experience_history)[-50:]  # Last 50 experiences
        successful_experiences = [e for e in recent_experiences if e.success]
        
        if not successful_experiences:
            return
            
        # Update cache size based on memory vs performance tradeoff
        memory_usage = [e.performance_metrics.get("memory_used_mb", 0) 
                       for e in successful_experiences]
        speedups = [e.performance_metrics.get("speedup_factor", 1.0) 
                   for e in successful_experiences]
        
        if len(memory_usage) > 0 and len(speedups) > 0:
            # Simple heuristic: increase cache if low memory usage but good speedup
            avg_memory = np.mean(memory_usage)
            avg_speedup = np.mean(speedups)
            
            if avg_memory < self.adaptive_config["cache_size_mb"] * SELF_OPT_CACHE_LOW_FRACTION_CANONICAL and avg_speedup > SELF_OPT_SPEEDUP_HIGH_CANONICAL:
                self.adaptive_config["cache_size_mb"] *= SELF_OPT_CACHE_EXPANSION_CANONICAL
            elif avg_memory > self.adaptive_config["cache_size_mb"] * SELF_OPT_CACHE_HIGH_FRACTION_CANONICAL and avg_speedup < SELF_OPT_SPEEDUP_LOW_CANONICAL:
                self.adaptive_config["cache_size_mb"] *= SELF_OPT_CACHE_CONTRACTION_CANONICAL
                
        # Update backend preference
        backend_performance = defaultdict(list)
        for exp in successful_experiences:
            backend = exp.parameters.get("backend", "numpy")
            speedup = exp.performance_metrics.get("speedup_factor", 1.0)
            backend_performance[backend].append(speedup)
            
        if backend_performance:
            best_backend = max(backend_performance.keys(), 
                             key=lambda b: np.mean(backend_performance[b]))
            self.adaptive_config["backend_preference"] = best_backend
            
    def recommend_optimization_strategy(
        self,
        G: Any,
        operation_type: str = "general",
        current_performance: Optional[Dict[str, float]] = None
    ) -> SelfOptimizationResult:
        """
        Recommend optimization strategy based on mathematical analysis and learning.
        """
        start_time = time.perf_counter()
        
        # Analyze mathematical structure
        mathematical_insights = self.analyze_mathematical_optimization_landscape(G, operation_type)
        
        # Find matching learned policies
        matching_policies = []
        if HAS_NETWORKX and G:
            num_nodes = len(G.nodes())
            num_edges = len(G.edges())
            density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            
            size_bucket = "small" if num_nodes < 20 else "medium" if num_nodes < 100 else "large"
            density_bucket = "sparse" if density < SELF_OPT_DENSITY_SPARSE_CANONICAL else "medium" if density < SELF_OPT_DENSITY_DENSE_CANONICAL else "dense"
            
            for policy in self.learned_policies:
                conditions = policy.conditions
                if (conditions.get("graph_size_bucket") == size_bucket and
                    conditions.get("density_bucket") == density_bucket and
                    conditions.get("operation_type") == operation_type):
                    matching_policies.append(policy)
                    
        # Generate recommendations
        recommended_strategies = []
        predicted_speedups = {}
        
        # From learned policies
        for policy in matching_policies:
            strategy = policy.actions.get("recommended_strategy")
            if strategy:
                recommended_strategies.append(strategy)
                predicted_speedups[strategy] = policy.average_improvement
                
        # From mathematical analysis
        math_recommendations = mathematical_insights.get("nodal_equation_analysis", {}).get("optimization_recommendations", [])
        recommended_strategies.extend(math_recommendations)
        
        # From pattern analysis
        pattern_hints = mathematical_insights.get("pattern_optimization_hints", [])
        recommended_strategies.extend(pattern_hints)
        
        # Remove duplicates while preserving order
        recommended_strategies = list(dict.fromkeys(recommended_strategies))
        
        # Calculate optimization improvements
        optimization_improvements = {}
        if current_performance:
            for strategy, speedup in predicted_speedups.items():
                optimization_improvements[strategy] = (speedup - 1.0) / speedup
                
        execution_time = time.perf_counter() - start_time
        
        return SelfOptimizationResult(
            learned_policies=matching_policies,
            optimization_improvements=optimization_improvements,
            recommended_strategies=recommended_strategies,
            mathematical_insights=mathematical_insights,
            predicted_speedups=predicted_speedups,
            adaptive_configurations=dict(self.adaptive_config),
            execution_time=execution_time
        )
        
    def optimize_automatically(
        self,
        G: Any,
        operation_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Automatically apply best optimization strategy.
        
        Uses learned policies and mathematical analysis to select and apply
        the optimal strategy.
        """
        # Get recommendations
        recommendations = self.recommend_optimization_strategy(G, operation_type)
        
        # Apply best strategy
        if recommendations.recommended_strategies and self.orchestrator:
            best_strategy = recommendations.recommended_strategies[0]
            
            # Map strategy name to OptimizationStrategy enum
            strategy_mapping = {
                "spectral_methods": OptimizationStrategy.SPECTRAL_FFT,
                "vectorized": OptimizationStrategy.NODAL_VECTORIZED,
                "cache": OptimizationStrategy.ADELIC_CACHE,
                "structural": OptimizationStrategy.STRUCTURAL_MEMO,
                "hybrid": OptimizationStrategy.HYBRID
            }
            
            # Find matching strategy
            optimization_strategy = OptimizationStrategy.AUTO
            for name_part, strategy in strategy_mapping.items():
                if name_part in best_strategy.lower():
                    optimization_strategy = strategy
                    break
                    
            # Execute optimization
            try:
                profile = self.orchestrator.analyze_optimization_profile(G, operation_type)
                result = self.orchestrator.execute_optimization(
                    G, operation_type, optimization_strategy, **kwargs
                )
                
                # Record experience
                experience = OptimizationExperience(
                    graph_properties={
                        "nodes": len(G.nodes()) if HAS_NETWORKX and G else 0,
                        "edges": len(G.edges()) if HAS_NETWORKX and G else 0,
                        "density": profile.edge_density
                    },
                    operation_type=operation_type,
                    strategy_used=optimization_strategy.value,
                    parameters=kwargs,
                    performance_metrics={
                        "speedup_factor": result.speedup_factor,
                        "execution_time": result.execution_time,
                        "memory_used_mb": result.memory_used_mb,
                        "cache_hits": result.cache_hits
                    },
                    timestamp=time.time(),
                    success=result.accuracy_preserved,
                    mathematical_signature=recommendations.mathematical_insights
                )
                
                self.learn_from_experience(experience)
                
                return {
                    "optimization_result": result,
                    "strategy_used": optimization_strategy.value,
                    "recommendations": recommendations,
                    "learning_updated": True
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "recommendations": recommendations,
                    "learning_updated": False
                }
        else:
            return {
                "message": "No optimization applied",
                "recommendations": recommendations,
                "learning_updated": False
            }
            
    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export learned optimization knowledge."""
        return {
            "learned_policies": [
                {
                    "name": p.policy_name,
                    "objective": p.objective.value,
                    "conditions": p.conditions,
                    "actions": p.actions,
                    "confidence": p.confidence,
                    "success_rate": p.success_rate,
                    "average_improvement": p.average_improvement,
                    "applications": p.applications_count
                }
                for p in self.learned_policies
            ],
            "adaptive_configuration": dict(self.adaptive_config),
            "performance_statistics": {
                "total_attempts": self.optimization_attempts,
                "successful_optimizations": self.successful_optimizations,
                "success_rate": self.successful_optimizations / max(1, self.optimization_attempts),
                "experience_count": len(self.experience_history)
            },
            "mathematical_insights": dict(self.mathematical_insights)
        }
        
    def import_learned_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Import previously learned optimization knowledge."""
        with self._lock:
            # Import policies
            if "learned_policies" in knowledge:
                imported_policies = []
                for p_data in knowledge["learned_policies"]:
                    policy = OptimizationPolicy(
                        policy_name=p_data["name"],
                        objective=OptimizationObjective(p_data["objective"]),
                        conditions=p_data["conditions"],
                        actions=p_data["actions"],
                        confidence=p_data["confidence"],
                        success_rate=p_data["success_rate"],
                        average_improvement=p_data["average_improvement"],
                        applications_count=p_data.get("applications", 0)
                    )
                    imported_policies.append(policy)
                self.learned_policies.extend(imported_policies)
                
            # Import adaptive configuration
            if "adaptive_configuration" in knowledge:
                self.adaptive_config.update(knowledge["adaptive_configuration"])
                
            # Import insights
            if "mathematical_insights" in knowledge:
                self.mathematical_insights.update(knowledge["mathematical_insights"])


# Factory functions
def create_self_optimizing_engine(**kwargs) -> TNFRSelfOptimizingEngine:
    """Create self-optimizing engine."""
    return TNFRSelfOptimizingEngine(**kwargs)


def auto_optimize_tnfr_computation(
    G: Any, 
    operation_type: str = "general", 
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for automatic optimization."""
    engine = create_self_optimizing_engine()
    return engine.optimize_automatically(G, operation_type, **kwargs)