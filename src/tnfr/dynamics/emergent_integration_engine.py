"""
TNFR Emergent Integration Engine

This engine discovers candidate integration opportunities that emerge from
the mathematical structure of the nodal equation. It executes only adapters
that are actually wired, marks unavailable candidates explicitly, and keeps
performance claims neutral until measurements are supplied.

Mathematical Foundation:
The nodal equation ∂EPI/∂t = νf · ΔNFR(t) creates natural mathematical
structures that can be unified across computational domains:

1. **Spectral Unification**: Eigendecompositions appear in FFT arithmetic,
   structural fields (Φ_s, |∇φ|, K_φ, ξ_C), and centralization analysis.
   These can share computational artifacts.

2. **Cache Coherence**: Mathematical dependencies create natural cache
   invalidation patterns. Structural fields depend on eigendecompositions,
   coordination depends on centrality metrics.

3. **Adaptive Coordination**: Phase coordination using Kuramoto order
   parameter can inform cache placement and prefetch strategies.

4. **Vectorization Opportunities**: Nodal optimizer's vectorized operations
   can be extended to structural field batch computations.

5. **Temporal Prediction**: Multi-scale temporal caching can predict
   structural field evolution based on nodal equation integration.

6. **Mathematical Consistency**: All optimizations must preserve TNFR
   invariants and maintain grammar compliance.

Status: evidence-aware compatibility and integration coordinator
"""

import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

from ..mathematics.unified_numerical import np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    INTEGRATION_CENTRALITY_THRESHOLD_CANONICAL,
    INTEGRATION_CONFIDENCE_HIGH_CANONICAL,
    INTEGRATION_CONFIDENCE_LOW_CANONICAL,
    INTEGRATION_CONFIDENCE_MEDIUM_CANONICAL,
    INTEGRATION_CONFIDENCE_MINIMAL_CANONICAL,
    INTEGRATION_CONFIDENCE_SYNC_CANONICAL,
    INTEGRATION_CONFIDENCE_THRESHOLD_CANONICAL,
    INTEGRATION_SYNC_THRESHOLD_CANONICAL,
)
from ..utils.cache import _compute_dependency_hash

# Import all TNFR engines for integration analysis
try:
    from .emergent_centralization import TNFREmergentCentralizationEngine
    from .fft_cache_coordinator import get_fft_cache_coordinator
    from .nodal_optimizer import create_nodal_optimizer
    from .optimization_orchestrator import TNFROptimizationOrchestrator
    from .self_optimizing_engine import TNFRSelfOptimizingMathematicalEngine
    from .spectral_structural_fusion import TNFRSpectralStructuralFusionEngine
    from .structural_cache import get_structural_cache
    from .unified_mathematical_cache_orchestrator import (
        TNFRUnifiedMathematicalCacheOrchestrator,
    )

    HAS_ALL_ENGINES = True
except ImportError:
    HAS_ALL_ENGINES = False


class IntegrationOpportunity(Enum):
    """Types of integration opportunities that can emerge."""

    SPECTRAL_SHARING = "spectral_sharing"  # Share eigendecompositions
    CACHE_COORDINATION = "cache_coordination"  # Coordinate cache strategies
    VECTORIZATION_FUSION = "vectorization_fusion"  # Batch similar computations
    TEMPORAL_PREDICTION = "temporal_prediction"  # Predict future computations
    PHASE_INFORMED_CACHING = "phase_informed_caching"  # Use phase dynamics for cache
    MATHEMATICAL_CONSISTENCY = (
        "mathematical_consistency"  # Ensure mathematical invariants
    )


@dataclass
class IntegrationPattern:
    """Discovered integration pattern with mathematical foundation."""

    pattern_id: str
    opportunity_type: IntegrationOpportunity
    mathematical_basis: str  # Mathematical justification
    involved_engines: set[str]
    integration_strategy: dict[str, Any]
    expected_benefit: dict[str, float | None]  # None unless evidence is supplied
    mathematical_requirements: list[str]  # Invariants that must be preserved
    confidence_score: float
    validation_results: dict[str, Any] | None = None
    benefit_evidence: str = "not_measured"
    implementation_status: str = "candidate"
    confidence_basis: str = "heuristic_capability_match"

    def __post_init__(self) -> None:
        """Keep forecast-shaped compatibility fields evidence-neutral."""
        if self.benefit_evidence != "measured":
            self.benefit_evidence = "not_measured"
            self.expected_benefit = {key: None for key in self.expected_benefit}
            return
        self.expected_benefit = {
            key: (
                float(value)
                if isinstance(key, str)
                and isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                else None
            )
            for key, value in self.expected_benefit.items()
        }


@dataclass
class IntegrationResult:
    """Result of applying an integration pattern."""

    pattern_applied: str
    success: bool
    performance_improvement: dict[str, float]
    mathematical_consistency_maintained: bool | None
    resource_savings: dict[str, float]
    side_effects: list[str]
    timestamp: float
    duration_seconds: float = 0.0
    application_status: str = "unknown"
    performance_evidence: str = "not_measured"
    validation_evidence: str = "not_assessed"
    details: dict[str, Any] = field(default_factory=dict)


class TNFREmergentIntegrationEngine:
    """
    Discover candidate integrations and execute the adapters that exist.

    Mathematical requirements, implementation readiness, state validation and
    performance evidence are reported separately. Candidate discovery alone
    does not establish an optimization or a speedup.
    """

    def __init__(self):
        # Engine instances
        if HAS_ALL_ENGINES:
            self.cache_orchestrator = TNFRUnifiedMathematicalCacheOrchestrator()
            self.optimization_orchestrator = TNFROptimizationOrchestrator()
            try:
                self.self_optimizer = TNFRSelfOptimizingMathematicalEngine()
            except Exception:
                self.self_optimizer = None
            self.spectral_fusion = TNFRSpectralStructuralFusionEngine()
            self.centralization = TNFREmergentCentralizationEngine()
            self.nodal_optimizer = create_nodal_optimizer()
            self.structural_cache = get_structural_cache()
            self.fft_cache = get_fft_cache_coordinator()
        else:
            # Create placeholders
            self.cache_orchestrator = None
            self.optimization_orchestrator = None
            self.self_optimizer = None
            self.spectral_fusion = None
            self.centralization = None
            self.nodal_optimizer = None
            self.structural_cache = None
            self.fft_cache = None

        # Integration state
        self.discovered_patterns: dict[str, IntegrationPattern] = {}
        self.applied_integrations: list[IntegrationResult] = []
        self.integration_opportunities: list[IntegrationPattern] = []

        # Mathematical consistency tracking
        self.mathematical_invariants = [
            "eigendecomposition_consistency",
            "phase_synchronization_preservation",
            "structural_field_accuracy",
            "nodal_equation_compliance",
            "cache_coherence_maintained",
        ]

        # Thread safety
        self._lock = threading.RLock()

    def discover_integration_opportunities(self, G: Any) -> list[IntegrationPattern]:
        """
        Discover integration opportunities by analyzing mathematical structure.

        This method analyzes the relationships between all TNFR engines to
        identify natural unification points based on mathematical foundations.
        """
        opportunities = []

        with self._lock:
            # 1. Spectral sharing analysis
            spectral_pattern = self._analyze_spectral_sharing_opportunities(G)
            if spectral_pattern:
                opportunities.append(spectral_pattern)

            # 2. Cache coordination analysis
            cache_pattern = self._analyze_cache_coordination_opportunities(G)
            if cache_pattern:
                opportunities.append(cache_pattern)

            # 3. Vectorization fusion analysis
            vectorization_pattern = self._analyze_vectorization_fusion_opportunities(G)
            if vectorization_pattern:
                opportunities.append(vectorization_pattern)

            # 4. Temporal prediction analysis
            temporal_pattern = self._analyze_temporal_prediction_opportunities(G)
            if temporal_pattern:
                opportunities.append(temporal_pattern)

            # 5. Phase-informed caching analysis
            phase_pattern = self._analyze_phase_informed_caching_opportunities(G)
            if phase_pattern:
                opportunities.append(phase_pattern)

            self.integration_opportunities = opportunities

        return opportunities

    def _analyze_spectral_sharing_opportunities(
        self, G: Any
    ) -> IntegrationPattern | None:
        """Analyze opportunities for sharing spectral decompositions."""
        if not HAS_ALL_ENGINES or not HAS_NETWORKX or G is None:
            return None

        # Check if multiple engines would benefit from same eigendecomposition
        engines_using_spectral = []

        if self.spectral_fusion:
            engines_using_spectral.append("spectral_structural_fusion")

        if self.fft_cache:
            engines_using_spectral.append("fft_cache_coordinator")

        engines_using_spectral.append("structural_fields")

        if len(engines_using_spectral) >= 2:
            pattern_id = f"spectral_sharing_{int(time.time())}"

            return IntegrationPattern(
                pattern_id=pattern_id,
                opportunity_type=IntegrationOpportunity.SPECTRAL_SHARING,
                mathematical_basis=(
                    "Graph Laplacian eigendecomposition shared across structural "
                    "fields, FFT arithmetic, and centralization analysis"
                ),
                involved_engines=set(engines_using_spectral),
                integration_strategy={
                    "method": "shared_eigendecomposition",
                    "cache_key": "laplacian_eigensystem",
                    "coordination_engine": "spectral_structural_fusion",
                },
                expected_benefit={
                    "computation_time_reduction": None,
                    "memory_savings": None,
                    "cache_efficiency": None,
                },
                mathematical_requirements=[
                    "eigendecomposition_consistency",
                    "spectral_accuracy_preservation",
                ],
                confidence_score=INTEGRATION_CONFIDENCE_HIGH_CANONICAL,
                implementation_status="available",
            )

        return None

    def _analyze_cache_coordination_opportunities(
        self, G: Any
    ) -> IntegrationPattern | None:
        """Analyze opportunities for coordinating cache strategies."""
        if not self.cache_orchestrator or G is None:
            return None

        # Check if centralization patterns can inform cache placement
        if self.centralization:
            pattern_id = f"cache_coordination_{int(time.time())}"

            return IntegrationPattern(
                pattern_id=pattern_id,
                opportunity_type=IntegrationOpportunity.CACHE_COORDINATION,
                mathematical_basis=(
                    "Network centrality metrics can select shared-cache anchors "
                    "whose performance benefit remains to be measured"
                ),
                involved_engines={
                    "cache_orchestrator",
                    "emergent_centralization",
                    "structural_cache",
                },
                integration_strategy={
                    "method": "centrality_guided_placement",
                    "centrality_threshold": INTEGRATION_CENTRALITY_THRESHOLD_CANONICAL,
                    "coordination_frequency": "adaptive",
                },
                expected_benefit={
                    "cache_hit_rate_improvement": None,
                    "memory_usage_reduction": None,
                    "access_time_improvement": None,
                },
                mathematical_requirements=[
                    "centrality_consistency",
                    "cache_coherence_maintained",
                ],
                confidence_score=INTEGRATION_CONFIDENCE_MEDIUM_CANONICAL,
                implementation_status=(
                    "available" if self.spectral_fusion is not None else "unavailable"
                ),
            )

        return None

    def _analyze_vectorization_fusion_opportunities(
        self, G: Any
    ) -> IntegrationPattern | None:
        """Analyze opportunities for fusing vectorized computations."""
        if not self.nodal_optimizer or G is None:
            return None

        # Check if structural field computations can be batched with nodal operations
        if len(G.nodes()) > 10:
            pattern_id = f"vectorization_fusion_{int(time.time())}"

            return IntegrationPattern(
                pattern_id=pattern_id,
                opportunity_type=IntegrationOpportunity.VECTORIZATION_FUSION,
                mathematical_basis=(
                    "Nodal equation vectorization could be extended to structural "
                    "field batches once a shared kernel is implemented"
                ),
                involved_engines={
                    "nodal_optimizer",
                    "structural_fields",
                    "spectral_fusion",
                },
                integration_strategy={
                    "method": "batch_field_computation",
                    "batch_size": min(len(G.nodes()), 64),
                    "vectorization_threshold": 8,
                },
                expected_benefit={
                    "computation_speedup": None,
                    "memory_efficiency": None,
                    "cpu_utilization": None,
                },
                mathematical_requirements=[
                    "nodal_equation_compliance",
                    "vectorization_accuracy",
                ],
                confidence_score=INTEGRATION_CONFIDENCE_LOW_CANONICAL,
                implementation_status="unavailable",
            )

        return None

    def _analyze_temporal_prediction_opportunities(
        self, G: Any
    ) -> IntegrationPattern | None:
        """Analyze opportunities for temporal prediction caching."""
        if not self.nodal_optimizer or G is None:
            return None

        # Check if temporal caching can predict structural field evolution
        pattern_id = f"temporal_prediction_{int(time.time())}"

        return IntegrationPattern(
            pattern_id=pattern_id,
            opportunity_type=IntegrationOpportunity.TEMPORAL_PREDICTION,
            mathematical_basis=(
                "A future-state cache would require a validated predictor tied "
                "to the ∂EPI/∂t dynamics"
            ),
            involved_engines={
                "nodal_optimizer",
                "structural_cache",
                "cache_orchestrator",
            },
            integration_strategy={
                "method": "predictive_evolution_caching",
                "prediction_horizon": 5,  # time steps
                "confidence_threshold": INTEGRATION_CONFIDENCE_THRESHOLD_CANONICAL,
            },
            expected_benefit={
                "cache_precomputation_success": None,
                "computation_avoidance": None,
                "response_time_improvement": None,
            },
            mathematical_requirements=["temporal_consistency", "evolution_accuracy"],
            confidence_score=INTEGRATION_CONFIDENCE_MINIMAL_CANONICAL,
            implementation_status="unavailable",
        )

    def _analyze_phase_informed_caching_opportunities(
        self, G: Any
    ) -> IntegrationPattern | None:
        """Analyze opportunities for using phase dynamics to inform caching."""
        if G is None or not HAS_ALL_ENGINES:
            return None

        try:
            # Check if phase synchronization patterns can guide cache strategies
            if len(G.nodes()) > 5:
                pattern_id = f"phase_informed_caching_{int(time.time())}"

                return IntegrationPattern(
                    pattern_id=pattern_id,
                    opportunity_type=IntegrationOpportunity.PHASE_INFORMED_CACHING,
                    mathematical_basis=(
                        "Phase synchronization telemetry is a candidate input for "
                        "a cache prefetch policy that has not yet been implemented"
                    ),
                    involved_engines={
                        "coordination",
                        "cache_orchestrator",
                        "structural_cache",
                    },
                    integration_strategy={
                        "method": "phase_guided_prefetch",
                        "synchronization_threshold": INTEGRATION_SYNC_THRESHOLD_CANONICAL,
                        "prefetch_distance": 2,
                    },
                    expected_benefit={
                        "prefetch_accuracy": None,
                        "cache_efficiency": None,
                        "synchronization_prediction": None,
                    },
                    mathematical_requirements=[
                        "phase_synchronization_preservation",
                        "kuramoto_consistency",
                    ],
                    confidence_score=INTEGRATION_CONFIDENCE_SYNC_CANONICAL,
                    implementation_status="unavailable",
                )
        except Exception:
            pass

        return None

    def apply_integration_pattern(
        self, pattern: IntegrationPattern, G: Any, validate_mathematics: bool = True
    ) -> IntegrationResult:
        """Apply one executable integration and report its evidence boundary."""
        start_time = time.perf_counter()

        with self._lock:
            reference_signature = self._structural_state_signature(G)

            try:
                if pattern.opportunity_type == IntegrationOpportunity.SPECTRAL_SHARING:
                    success, details = self._apply_spectral_sharing(pattern, G)
                elif (
                    pattern.opportunity_type
                    == IntegrationOpportunity.CACHE_COORDINATION
                ):
                    success, details = self._apply_cache_coordination(pattern, G)
                elif (
                    pattern.opportunity_type
                    == IntegrationOpportunity.VECTORIZATION_FUSION
                ):
                    success, details = self._apply_vectorization_fusion(pattern, G)
                elif (
                    pattern.opportunity_type
                    == IntegrationOpportunity.TEMPORAL_PREDICTION
                ):
                    success, details = self._apply_temporal_prediction(pattern, G)
                elif (
                    pattern.opportunity_type
                    == IntegrationOpportunity.PHASE_INFORMED_CACHING
                ):
                    success, details = self._apply_phase_informed_caching(pattern, G)
                else:
                    success, details = False, {
                        "application_status": "unavailable",
                        "reason": "integration_type_has_no_executable_adapter",
                        "performance_evidence": "not_measured",
                    }

            except Exception as e:
                success, details = False, {
                    "application_status": "failed",
                    "reason": "integration_execution_failed",
                    "error": str(e),
                    "performance_evidence": "not_measured",
                }

            performance_evidence = details.get(
                "performance_evidence", "not_measured"
            )
            performance_improvement = self._measured_metric_mapping(
                details.get("performance_improvement"), performance_evidence
            )
            resource_savings = self._measured_metric_mapping(
                details.get("resource_savings"), performance_evidence
            )
            performance_evidence = (
                "measured"
                if performance_improvement or resource_savings
                else "not_measured"
            )
            details = {
                **details,
                "performance_evidence": performance_evidence,
                "performance_improvement": performance_improvement,
                "resource_savings": resource_savings,
            }

            mathematical_consistency: bool | None = None
            validation_evidence = "not_requested"
            if validate_mathematics and success:
                mathematical_consistency = self._validate_mathematical_consistency(
                    G, pattern, reference_signature=reference_signature
                )
                if mathematical_consistency is None:
                    validation_evidence = "unavailable"
                elif mathematical_consistency:
                    validation_evidence = "structural_state_signature_match"
                else:
                    validation_evidence = "structural_state_signature_mismatch"
                    success = False
                    details = {
                        **details,
                        "application_status": "validation_failed",
                        "reason": "integration_modified_canonical_graph_state",
                    }

            pattern.validation_results = {
                "status": validation_evidence,
                "structural_state_preserved": mathematical_consistency,
            }
            application_status = details.get(
                "application_status", "applied" if success else "failed"
            )

            result = IntegrationResult(
                pattern_applied=pattern.pattern_id,
                success=success,
                performance_improvement=performance_improvement,
                mathematical_consistency_maintained=mathematical_consistency,
                resource_savings=resource_savings,
                side_effects=details.get("side_effects", []),
                timestamp=time.time(),
                duration_seconds=time.perf_counter() - start_time,
                application_status=application_status,
                performance_evidence=performance_evidence,
                validation_evidence=validation_evidence,
                details=dict(details),
            )

            self.applied_integrations.append(result)
            if success:
                self.discovered_patterns[pattern.pattern_id] = pattern

        return result

    def _apply_spectral_sharing(
        self, pattern: IntegrationPattern, G: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Materialize an authenticated shared spectral/structural cache entry."""
        del pattern
        if self.spectral_fusion is None or G is None:
            return False, {
                "application_status": "unavailable",
                "reason": "spectral_fusion_engine_not_available",
                "performance_evidence": "not_measured",
            }
        shared_fields = self.spectral_fusion.compute_structural_fields(
            G, force_recompute=False
        )
        signature = getattr(shared_fields, "spectral_basis_signature", None)
        if not isinstance(signature, str) or not signature:
            return False, {
                "application_status": "abstained",
                "reason": "authenticated_spectral_basis_not_materialized",
                "performance_evidence": "not_measured",
            }
        return True, {
            "application_status": "applied",
            "shared_eigendecomposition": True,
            "spectral_basis_signature": signature,
            "performance_evidence": "not_measured",
            "resource_savings": {},
        }

    def _apply_cache_coordination(
        self, pattern: IntegrationPattern, G: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Register discovered coordination nodes in the shared cache."""
        del pattern
        if self.centralization is None or self.spectral_fusion is None or G is None:
            return False, {
                "application_status": "unavailable",
                "reason": "cache_coordination_components_not_available",
                "performance_evidence": "not_measured",
            }
        patterns_discovered = self.centralization.discover_centralization_patterns(G)
        if not patterns_discovered:
            return False, {
                "application_status": "abstained",
                "reason": "no_coordination_pattern_discovered",
                "performance_evidence": "not_measured",
            }
        best_pattern = max(patterns_discovered, key=lambda item: item.efficiency_gain)
        coordination_nodes = list(best_pattern.coordination_nodes)
        if not coordination_nodes:
            return False, {
                "application_status": "abstained",
                "reason": "selected_pattern_has_no_coordination_nodes",
                "performance_evidence": "not_measured",
            }
        self.spectral_fusion.coordinate_cache_with_central_nodes(
            G, coordination_nodes, strategy="mathematical"
        )
        return True, {
            "application_status": "applied",
            "coordination_patterns": len(patterns_discovered),
            "coordination_nodes_registered": len(coordination_nodes),
            "performance_evidence": "not_measured",
            "resource_savings": {},
        }

    def _apply_vectorization_fusion(
        self, pattern: IntegrationPattern, G: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Abstain until a fused field/nodal kernel exists."""
        del pattern, G
        return False, {
            "application_status": "unavailable",
            "reason": "vectorization_fusion_kernel_not_implemented",
            "performance_evidence": "not_measured",
        }

    def _apply_temporal_prediction(
        self, pattern: IntegrationPattern, G: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Abstain until a validated future-state predictor exists."""
        del pattern, G
        return False, {
            "application_status": "unavailable",
            "reason": "temporal_prediction_model_not_implemented",
            "performance_evidence": "not_measured",
        }

    def _apply_phase_informed_caching(
        self, pattern: IntegrationPattern, G: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Abstain until phase telemetry is connected to a cache prefetcher."""
        del pattern, G
        return False, {
            "application_status": "unavailable",
            "reason": "phase_informed_prefetcher_not_implemented",
            "performance_evidence": "not_measured",
        }

    @staticmethod
    def _measured_metric_mapping(
        metrics: Any, evidence: str
    ) -> dict[str, float]:
        """Return only finite numeric metrics backed by measured evidence."""
        if evidence != "measured" or not isinstance(metrics, dict):
            return {}
        measured: dict[str, float] = {}
        for name, value in metrics.items():
            if (
                isinstance(name, str)
                and isinstance(value, Real)
                and not isinstance(value, bool)
            ):
                numeric = float(value)
                if math.isfinite(numeric):
                    measured[name] = numeric
        return measured

    @staticmethod
    def _structural_state_signature(G: Any) -> str | None:
        """Hash the topology and canonical nodal channels when available."""
        if G is None or not callable(getattr(G, "nodes", None)):
            return None
        try:
            signature = _compute_dependency_hash(
                G,
                {
                    "graph_topology",
                    "node_epi",
                    "node_vf",
                    "node_phase",
                    "node_dnfr",
                    "node_depi",
                },
            )
        except Exception:
            return None
        return signature or None

    def _measure_baseline_performance(
        self, G: Any, pattern: IntegrationPattern
    ) -> dict[str, float]:
        """Return measured baselines; this compatibility engine records none."""
        del G, pattern
        return {}

    def _validate_mathematical_consistency(
        self,
        G: Any,
        pattern: IntegrationPattern,
        *,
        reference_signature: str | None = None,
    ) -> bool | None:
        """Compare canonical graph state before and after an integration."""
        del pattern
        if reference_signature is None:
            return None
        current_signature = self._structural_state_signature(G)
        if current_signature is None:
            return None
        return current_signature == reference_signature

    def get_integration_statistics(self) -> dict[str, Any]:
        """Get comprehensive integration statistics."""
        with self._lock:
            successful_integrations = [
                r for r in self.applied_integrations if r.success
            ]
            measured_performance: dict[str, list[float]] = {}
            for result in successful_integrations:
                for name, value in result.performance_improvement.items():
                    measured_performance.setdefault(name, []).append(value)
            mean_performance_by_metric = {
                name: float(np.mean(values))
                for name, values in measured_performance.items()
            }
            homogeneous_average = (
                next(iter(mean_performance_by_metric.values()))
                if len(mean_performance_by_metric) == 1
                else None
            )
            validated_consistency = [
                result.mathematical_consistency_maintained
                for result in self.applied_integrations
                if result.mathematical_consistency_maintained is not None
            ]

            return {
                "total_opportunities_discovered": len(self.integration_opportunities),
                "total_patterns_discovered": len(self.discovered_patterns),
                "total_integrations_attempted": len(self.applied_integrations),
                "successful_integrations": len(successful_integrations),
                "success_rate": len(successful_integrations)
                / max(len(self.applied_integrations), 1),
                "average_performance_improvement": homogeneous_average,
                "mean_performance_improvement_by_metric": mean_performance_by_metric,
                "performance_evidence": (
                    "measured" if mean_performance_by_metric else "not_measured"
                ),
                "mathematical_consistency_rate": (
                    float(np.mean(validated_consistency))
                    if validated_consistency
                    else None
                ),
                "validation_evidence": (
                    "assessed" if validated_consistency else "not_assessed"
                ),
                "integration_types_used": list(
                    set(
                        [
                            pattern.opportunity_type.value
                            for pattern in self.discovered_patterns.values()
                        ]
                    )
                ),
                "engines_available": {
                    "cache_orchestrator": self.cache_orchestrator is not None,
                    "optimization_orchestrator": self.optimization_orchestrator
                    is not None,
                    "self_optimizer": self.self_optimizer is not None,
                    "spectral_fusion": self.spectral_fusion is not None,
                    "centralization": self.centralization is not None,
                    "nodal_optimizer": self.nodal_optimizer is not None,
                    "structural_cache": self.structural_cache is not None,
                    "fft_cache": self.fft_cache is not None,
                },
            }


# Global integration engine instance
_global_integration_engine = None


def get_emergent_integration_engine() -> TNFREmergentIntegrationEngine:
    """Get or create the global emergent integration engine."""
    global _global_integration_engine
    if _global_integration_engine is None:
        _global_integration_engine = TNFREmergentIntegrationEngine()
    return _global_integration_engine


def discover_and_apply_integrations(
    G: Any, auto_apply: bool = True, validate_mathematics: bool = True
) -> dict[str, Any]:
    """
    Convenience function to discover and optionally apply integration opportunities.

    Returns comprehensive statistics about discovered opportunities and
    integration results.
    """
    engine = get_emergent_integration_engine()

    # Discover opportunities
    opportunities = engine.discover_integration_opportunities(G)

    results = {
        "opportunities_discovered": len(opportunities),
        "opportunity_details": [
            {
                "type": opp.opportunity_type.value,
                "confidence": opp.confidence_score,
                "expected_benefits": opp.expected_benefit,
                "engines_involved": sorted(opp.involved_engines),
                "implementation_status": opp.implementation_status,
                "benefit_evidence": opp.benefit_evidence,
                "confidence_basis": opp.confidence_basis,
            }
            for opp in opportunities
        ],
        "integration_results": [],
    }

    # Auto-apply high-confidence opportunities
    if auto_apply:
        for opportunity in opportunities:
            if (
                opportunity.confidence_score
                > INTEGRATION_CENTRALITY_THRESHOLD_CANONICAL
            ):  # High confidence threshold
                integration_result = engine.apply_integration_pattern(
                    opportunity, G, validate_mathematics
                )
                results["integration_results"].append(
                    {
                        "pattern_id": integration_result.pattern_applied,
                        "success": integration_result.success,
                        "performance_improvement": integration_result.performance_improvement,
                        "mathematical_consistency": (
                            integration_result.mathematical_consistency_maintained
                        ),
                        "application_status": integration_result.application_status,
                        "performance_evidence": integration_result.performance_evidence,
                        "validation_evidence": integration_result.validation_evidence,
                    }
                )

    # Add engine statistics
    results["engine_statistics"] = engine.get_integration_statistics()

    return results
