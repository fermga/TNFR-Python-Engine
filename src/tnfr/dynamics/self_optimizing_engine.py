"""Evidence-scoped optimization recommendations for TNFR computations.

The nodal equation defines the physical channels inspected by this module. It
does not select a faster backend or prove that a computational strategy is
optimal. Recommendations combine snapshot diagnostics with experience records;
policies and speedup predictions are learned only from finite measurements
reported in an optimizer's authoritative evidence payload.

Candidate-energy and balance values remain diagnostics. They do not validate
grammar or prescribe canonical operators.
"""

import hashlib
import json
import math
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    NODAL_OPT_COUPLING_CANONICAL,
)
from ..constants.canonical import PI  # π ≈ 3.1416 (structural scale)
from ..constants.operational import (
    SELF_OPT_CACHE_CONTRACTION_CANONICAL,
    SELF_OPT_CACHE_EXPANSION_CANONICAL,
    SELF_OPT_CACHE_HIGH_FRACTION_CANONICAL,
    SELF_OPT_CACHE_LOW_FRACTION_CANONICAL,
    SELF_OPT_CHIRALITY_THRESHOLD_CANONICAL,
    SELF_OPT_COMPRESSION_HIGH_CANONICAL,
    SELF_OPT_COUPLING_LOW_CANONICAL,
    SELF_OPT_DENSITY_DENSE_CANONICAL,
    SELF_OPT_DENSITY_SPARSE_CANONICAL,
    SELF_OPT_ENERGY_HIGH_CANONICAL,
    SELF_OPT_IMPROVEMENT_SIGNIFICANT_CANONICAL,
    SELF_OPT_SPEEDUP_HIGH_CANONICAL,
    SELF_OPT_SPEEDUP_LOW_CANONICAL,
    SELF_OPT_SYMMETRY_THRESHOLD_CANONICAL,
)
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..types import require_finite_real_scalar_epi
from ..operators.grammar import glyph_function_name, validate_sequence

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


def _finite_measurement(
    value: Any, *, positive: bool = False, nonnegative: bool = False
) -> float | None:
    """Return a finite measured scalar, or ``None`` when evidence is absent."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    try:
        measured = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(measured):
        return None
    if positive and measured <= 0.0:
        return None
    if nonnegative and measured < 0.0:
        return None
    return measured


def _required_graph_scalar(
    G: Any,
    node: Any,
    aliases: Sequence[str],
    label: str,
    *,
    default: float = 0.0,
    nonnegative: bool = False,
) -> float:
    """Read one finite graph scalar through strict alias precedence."""

    raw = get_attr(
        G.nodes[node], aliases, default, strict=True, conv=lambda value: value
    )
    value = _finite_measurement(raw, nonnegative=nonnegative)
    if value is None:
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise TNFRValueError(f"node {node!r} {label} must be a {qualifier} real scalar")
    return value


def _node_scalar_epi(G: Any, node: Any) -> float:
    """Read one finite signed scalar EPI through canonical alias precedence."""

    raw = get_attr(
        G.nodes[node], ALIAS_EPI, 0.0, strict=True, conv=lambda value: value
    )
    return require_finite_real_scalar_epi(raw, f"node {node!r} EPI")


def _result_evidence(result: Any) -> dict[str, float | int]:
    """Extract only authoritative finite measurements from an optimizer result."""

    evidence: dict[str, float | int] = {}
    execution_time = _finite_measurement(
        getattr(result, "execution_time", None), nonnegative=True
    )
    if execution_time is not None:
        evidence["execution_time"] = execution_time

    details = getattr(result, "details", {})
    if not isinstance(details, Mapping):
        return evidence
    performance = details.get("performance_measurements")
    if isinstance(performance, Mapping):
        speedup = _finite_measurement(performance.get("speedup_factor"), positive=True)
        memory = _finite_measurement(
            performance.get("memory_used_mb"), nonnegative=True
        )
        if speedup is not None:
            evidence["speedup_factor"] = speedup
        if memory is not None:
            evidence["memory_used_mb"] = memory

    cache = details.get("cache_measurements")
    if isinstance(cache, Mapping):
        hits = cache.get("hits")
        if (
            isinstance(hits, int)
            and not isinstance(hits, bool)
            and hits >= 0
        ):
            evidence["cache_hits"] = hits
    return evidence


# Import Unified Fields (New Nov 2025)
try:
    from ..physics.fields import compute_unified_telemetry

    HAS_UNIFIED_FIELDS = True
except ImportError:
    HAS_UNIFIED_FIELDS = False

# Import Structural Integrity Monitor (operator contracts + finite alerts)
try:
    from ..physics.integrity import StructuralIntegrityMonitor

    HAS_INTEGRITY_MONITOR = True
except ImportError:
    HAS_INTEGRITY_MONITOR = False

# Import finite structural-balance diagnostics for optimization telemetry (P5)
try:
    from ..physics.conservation import (
        capture_conservation_snapshot,
        compute_lyapunov_derivative,
        detect_grammar_violations_from_conservation,
        verify_conservation_balance,
    )

    HAS_CONSERVATION = True
except ImportError:
    HAS_CONSERVATION = False

try:
    from ..metrics.common import compute_coherence
    from ..metrics.sense_index import compute_Si

    HAS_METRIC_OPERATORS = True
except ImportError:  # pragma: no cover - optional dependency in trimmed builds
    HAS_METRIC_OPERATORS = False
    compute_coherence = None  # type: ignore
    compute_Si = None  # type: ignore

# Import TNFR engines
try:
    from ..engines.pattern_discovery.mathematical_patterns import (
        EmergentPatternType,
        TNFREmergentPatternEngine,
    )
    from .optimization_orchestrator import (
        OptimizationStrategy,
        TNFROptimizationOrchestrator,
    )
    from .unified_backend import TNFRUnifiedBackend

    HAS_ENGINES = True
except ImportError:
    HAS_ENGINES = False

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_OUTPUT_DIR = Path("results") / "self_optimization"

# --- Self-optimization safety thresholds ---
_MIN_BALANCE_QUALITY_ALERT = 0.7
_MAX_STRUCTURAL_CHARGE_DRIFT_ALERT = 0.1
_MAX_MONITOR_ALERT_RATE = 0.2
_MIN_EPI_VARIANCE = 0.01  # below → variance too low, optimize


def _sanitize_label(value: Any | None, default: str) -> str:
    """Convert arbitrary identifiers to filesystem-safe labels."""
    if value is None:
        text = default
    else:
        text = str(value).strip()
    if not text:
        text = default
    sanitized = _SAFE_LABEL_RE.sub("_", text)
    return sanitized[:64] or default


def _finite_population_variance(values: Sequence[float]) -> float | None:
    """Return population variance, or ``None`` when it exceeds binary64."""

    if not values:
        return 0.0
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = [value / scale for value in values]
    mean = fmean(normalized)
    normalized_variance = fmean((value - mean) ** 2 for value in normalized)
    if normalized_variance == 0.0:
        return 0.0
    if scale > math.sqrt(float(np.finfo(float).max) / normalized_variance):
        return None
    result = scale * scale * normalized_variance
    return result if math.isfinite(result) else None


def _mean_numeric(values: Sequence[Any]) -> float | None:
    """Compute a mean only when every supplied sample is finite and numeric."""

    data: list[float] = []
    for value in values:
        measured = _finite_measurement(value)
        if measured is None:
            return None
        data.append(measured)
    if not data:
        return None
    return fmean(data)


def _graph_structure(graph: Any) -> dict[str, Any]:
    """Graph diagnostics shared by recommendations and persisted snapshots."""
    count = graph.number_of_nodes()
    edges = graph.number_of_edges()
    return {
        "nodes": count, "edges": edges, "density": nx.density(graph),
        "is_connected": (
            (nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph))
            if count else False
        ),
        "avg_degree": 2 * edges / count if count else 0.0,
    }


def _sense_index_mean(payload: Any) -> float | None:
    """Reduce compute_Si outputs (dict, array) to a scalar average."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        return _mean_numeric(list(payload.values()))
    try:
        array = np.asarray(payload, dtype=float)
    except Exception:
        return None
    if array.size == 0:
        return None
    return _mean_numeric(list(array.reshape(-1)))


def _json_safe(value: Any) -> Any:
    """Convert complex objects (NumPy, mappings) to JSON-safe structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _json_safe(v) for k, v in vars(value).items()}
    return repr(value)


def _canonicalize_sequence_tokens(sequence: Sequence[Any]) -> list[str]:
    """Normalize operator or glyph tokens into canonical operator names."""
    tokens: list[str] = []
    for entry in sequence:
        if entry is None:
            continue
        candidate = getattr(entry, "name", entry)
        text = str(candidate).strip()
        if not text:
            continue
        canonical = glyph_function_name(text, default=None)
        normalized = canonical or text
        tokens.append(normalized.lower())
    return tokens


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

    graph_properties: dict[str, Any]
    operation_type: str
    strategy_used: str
    parameters: dict[str, Any]
    performance_metrics: dict[str, Any]
    timestamp: float
    success: bool
    mathematical_signature: dict[str, Any] | None = None


@dataclass
class OptimizationPolicy:
    """Learned optimization policy."""

    policy_name: str
    objective: OptimizationObjective
    conditions: dict[str, Any]  # When to apply this policy
    actions: dict[str, Any]  # What to do
    confidence: float
    success_rate: float
    average_improvement: float
    applications_count: int = 0


@dataclass
class SelfOptimizationResult:
    """Result of self-optimization analysis.

    ``conservation_feedback`` is retained for compatibility.  Its accurately
    scoped view is :attr:`balance_feedback`; neither field validates grammar.
    """

    learned_policies: list[OptimizationPolicy]
    optimization_improvements: dict[str, float]
    recommended_strategies: list[str]
    mathematical_insights: dict[str, Any]
    predicted_speedups: dict[str, float]
    adaptive_configurations: dict[str, Any]
    execution_time: float
    conservation_feedback: dict[str, float] | None = None

    @property
    def balance_feedback(self) -> dict[str, float] | None:
        """Monitor feedback, with sample count, or ``None`` if unattached."""
        return self.conservation_feedback

    @property
    def balance_alert_reviews(self) -> tuple[str, ...]:
        """Review prompts kept outside executable optimization strategies."""
        reviews = self.mathematical_insights.get("balance_alert_reviews", ())
        return tuple(str(review) for review in reviews)


class TNFRSelfOptimizingEngine:
    """
    Recommend computational strategies from graph diagnostics and measurements.

    A learned policy summarizes available finite performance evidence. It is
    conditional empirical guidance, not a proof of global optimality.
    """

    def __init__(
        self,
        learning_strategy: LearningStrategy = LearningStrategy.MATHEMATICAL_ANALYSIS,
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCE_ALL,
        max_experience_history: int = 1000,
    ):
        self.learning_strategy = learning_strategy
        self.optimization_objective = optimization_objective
        self.max_experience_history = max_experience_history

        # Learning state
        self.experience_history: deque = deque(maxlen=max_experience_history)
        self.learned_policies: list[OptimizationPolicy] = []
        self.mathematical_insights: dict[str, Any] = {}

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
            "spectral_threshold": 20,
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
        self, G: Any, operation_type: str = "general"
    ) -> dict[str, Any]:
        """
        Analyze mathematical structure to discover optimization opportunities.

        Uses the mathematical properties of the nodal equation to identify
        natural optimization strategies.
        """
        insights = {}

        if not HAS_NETWORKX or G is None:
            return insights

        # Basic graph properties
        graph_structure = _graph_structure(G)
        num_nodes = graph_structure["nodes"]
        density = graph_structure["density"]

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
                    "topological_charge": float(tc_scalar),
                }

                insights["unified_field_analysis"] = unified_insights
            except Exception as e:
                # Fallback if computation fails
                insights["unified_field_error"] = str(e)

        # Finite balance / candidate-energy feedback. These diagnostics do not
        # validate grammar and are kept separate from operator selection.
        if HAS_INTEGRITY_MONITOR:
            monitor = StructuralIntegrityMonitor.get(G) if G is not None else None
            if monitor is not None:
                fv = monitor.feedback_vector()
                insights["balance_feedback"] = fv
                # Backward-compatible outer key.
                insights["conservation_feedback"] = fv

        # Mathematical structure analysis
        insights["graph_structure"] = graph_structure

        # Spectral properties for optimization
        if self.pattern_engine:
            pattern_result = self.pattern_engine.discover_all_patterns(G)

            # Extract optimization hints from discovered patterns
            optimization_hints = []
            for pattern in pattern_result.discovered_patterns:
                if (
                    pattern.compression_ratio is not None
                    and pattern.compression_ratio > SELF_OPT_COMPRESSION_HIGH_CANONICAL
                ):
                    optimization_hints.append(
                        f"use_compression_{pattern.pattern_type.value}"
                    )
                if (
                    pattern.prediction_horizon is not None
                    and pattern.prediction_horizon > PI
                ):
                    optimization_hints.append(
                        f"use_prediction_{pattern.pattern_type.value}"
                    )
                if pattern.pattern_type == EmergentPatternType.EIGENMODE_RESONANCE:
                    optimization_hints.append("use_spectral_methods")
                if pattern.pattern_type == EmergentPatternType.FRACTAL_SCALING:
                    optimization_hints.append("use_hierarchical_methods")

            insights["pattern_optimization_hints"] = optimization_hints
            insights["mathematical_patterns"] = len(pattern_result.discovered_patterns)
            insights["compression_potential"] = pattern_result.compression_potential

        # Nodal equation analysis
        epi_values = [_node_scalar_epi(G, node) for node in G.nodes()]
        vf_values = [
            _required_graph_scalar(
                G,
                node,
                ALIAS_VF,
                "structural frequency",
                default=1.0,
                nonnegative=True,
            )
            for node in G.nodes()
        ]
        dnfr_values = [
            _required_graph_scalar(G, node, ALIAS_DNFR, "DeltaNFR")
            for node in G.nodes()
        ]

        # Mathematical properties for optimization
        epi_variance = _finite_population_variance(epi_values)
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

        # Diagnostic review prompts. A residual, structural-charge drift, or
        # candidate-energy increase does not prescribe an operator, identify
        # a U-rule failure, or enter the executable optimization strategy list.
        balance_alert_reviews: list[str] = []
        cf = insights.get("balance_feedback", insights.get("conservation_feedback"))
        if cf is not None:
            has_balance_samples = cf.get("balance_sample_count", 1.0) > 0.0
            balance_quality = cf.get(
                "balance_quality", cf.get("conservation_quality", 1.0)
            )
            candidate_derivative = cf.get(
                "candidate_energy_derivative", cf.get("energy_derivative", 0.0)
            )
            charge_drift = cf.get(
                "mean_structural_charge_drift",
                cf.get("structural_charge_drift", cf.get("charge_drift", 0.0)),
            )
            alert_rate = cf.get(
                "monitor_alert_rate", cf.get("violation_rate", 0.0)
            )
            if has_balance_samples and balance_quality < _MIN_BALANCE_QUALITY_ALERT:
                balance_alert_reviews.append("balance_quality_low_review")
            if has_balance_samples and candidate_derivative > 0.0:
                balance_alert_reviews.append("candidate_energy_increase_review")
            if (
                has_balance_samples
                and charge_drift > _MAX_STRUCTURAL_CHARGE_DRIFT_ALERT
            ):
                balance_alert_reviews.append("structural_charge_drift_review")
            if alert_rate > _MAX_MONITOR_ALERT_RATE:
                balance_alert_reviews.append("monitor_alert_rate_review")
        insights["balance_alert_reviews"] = balance_alert_reviews

        if epi_variance is not None and epi_variance < _MIN_EPI_VARIANCE:
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
            "optimization_recommendations": recommendations,
        }

        return insights

    def learn_from_experience(self, experience: OptimizationExperience) -> None:
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
                size_bucket = (
                    "small"
                    if graph_size < 20
                    else "medium" if graph_size < 100 else "large"
                )
                density_bucket = (
                    "sparse"
                    if density < SELF_OPT_DENSITY_SPARSE_CANONICAL
                    else (
                        "medium"
                        if density < SELF_OPT_DENSITY_DENSE_CANONICAL
                        else "dense"
                    )
                )

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
                    improvement = _finite_measurement(
                        exp.performance_metrics.get("speedup_factor"), positive=True
                    )
                    if improvement is not None:
                        strategy_performance[strategy].append(improvement)

                # Select best strategy
                best_strategy = None
                best_avg_improvement = 0.0
                best_sample_count = 0
                for strategy, improvements in strategy_performance.items():
                    if len(improvements) < 3:
                        continue
                    avg_improvement = fmean(improvements)
                    if avg_improvement > best_avg_improvement:
                        best_avg_improvement = avg_improvement
                        best_sample_count = len(improvements)
                        best_strategy = strategy

                if (
                    best_strategy
                    and best_avg_improvement
                    > SELF_OPT_IMPROVEMENT_SIGNIFICANT_CANONICAL
                ):  # Significant improvement
                    policy = OptimizationPolicy(
                        policy_name=f"{size_bucket}_{density_bucket}_{operation}_policy",
                        objective=self.optimization_objective,
                        conditions={
                            "graph_size_bucket": size_bucket,
                            "density_bucket": density_bucket,
                            "operation_type": operation,
                        },
                        actions={
                            "recommended_strategy": best_strategy,
                            "expected_improvement": best_avg_improvement,
                            "measurement_sample_count": best_sample_count,
                            "evidence_basis": "measured_speedup_factor",
                        },
                        confidence=min(best_sample_count / 10.0, 1.0),
                        success_rate=len(experiences)
                        / max(
                            1,
                            len(
                                [
                                    e
                                    for e in self.experience_history
                                    if self._matches_conditions(e, condition_key)
                                ]
                            ),
                        ),
                        average_improvement=best_avg_improvement,
                        applications_count=best_sample_count,
                    )
                    new_policies.append(policy)

        # Update learned policies
        self.learned_policies.extend(new_policies)

        # Remove outdated policies (keep only best 20)
        if len(self.learned_policies) > 20:
            self.learned_policies.sort(
                key=lambda p: p.confidence * p.average_improvement, reverse=True
            )
            self.learned_policies = self.learned_policies[:20]

    def _matches_conditions(
        self, experience: OptimizationExperience, condition_key: tuple
    ) -> bool:
        """Check if experience matches condition key."""
        size_bucket, density_bucket, operation = condition_key

        graph_size = experience.graph_properties.get("nodes", 0)
        density = experience.graph_properties.get("density", 0.0)

        exp_size_bucket = (
            "small" if graph_size < 20 else "medium" if graph_size < 100 else "large"
        )
        exp_density_bucket = (
            "sparse"
            if density < SELF_OPT_DENSITY_SPARSE_CANONICAL
            else "medium" if density < SELF_OPT_DENSITY_DENSE_CANONICAL else "dense"
        )

        return (
            exp_size_bucket == size_bucket
            and exp_density_bucket == density_bucket
            and experience.operation_type == operation
        )

    def _update_adaptive_configuration(self) -> None:
        """Update adaptive configuration based on learned patterns."""
        if not self.experience_history:
            return

        # Analyze recent performance
        recent_experiences = list(self.experience_history)[-50:]  # Last 50 experiences
        successful_experiences = [e for e in recent_experiences if e.success]

        if not successful_experiences:
            return

        # Update cache size only from runs that measured both quantities.
        memory_speedup_pairs = []
        for experience in successful_experiences:
            memory = _finite_measurement(
                experience.performance_metrics.get("memory_used_mb"),
                nonnegative=True,
            )
            speedup = _finite_measurement(
                experience.performance_metrics.get("speedup_factor"),
                positive=True,
            )
            if memory is not None and speedup is not None:
                memory_speedup_pairs.append((memory, speedup))

        if memory_speedup_pairs:
            avg_memory = fmean(pair[0] for pair in memory_speedup_pairs)
            avg_speedup = fmean(pair[1] for pair in memory_speedup_pairs)

            if (
                avg_memory
                < self.adaptive_config["cache_size_mb"]
                * SELF_OPT_CACHE_LOW_FRACTION_CANONICAL
                and avg_speedup > SELF_OPT_SPEEDUP_HIGH_CANONICAL
            ):
                self.adaptive_config[
                    "cache_size_mb"
                ] *= SELF_OPT_CACHE_EXPANSION_CANONICAL
            elif (
                avg_memory
                > self.adaptive_config["cache_size_mb"]
                * SELF_OPT_CACHE_HIGH_FRACTION_CANONICAL
                and avg_speedup < SELF_OPT_SPEEDUP_LOW_CANONICAL
            ):
                self.adaptive_config[
                    "cache_size_mb"
                ] *= SELF_OPT_CACHE_CONTRACTION_CANONICAL

        # Update backend preference
        backend_performance = defaultdict(list)
        for exp in successful_experiences:
            backend = exp.parameters.get("backend", "numpy")
            speedup = _finite_measurement(
                exp.performance_metrics.get("speedup_factor"), positive=True
            )
            if speedup is not None:
                backend_performance[backend].append(speedup)

        if backend_performance:
            best_backend = max(
                backend_performance.keys(),
                key=lambda b: np.mean(backend_performance[b]),
            )
            self.adaptive_config["backend_preference"] = best_backend

        # Track the finite structural-charge diagnostic across experiences.
        structural_charge_drifts = []
        for experience in recent_experiences:
            raw_drift = experience.performance_metrics.get(
                "structural_charge_drift",
                experience.performance_metrics.get("conservation_charge_drift"),
            )
            drift = _finite_measurement(raw_drift)
            if drift is not None:
                structural_charge_drifts.append(drift)
        if structural_charge_drifts:
            mean_drift = fmean(structural_charge_drifts)
            self.adaptive_config["mean_structural_charge_drift"] = mean_drift
            # Backward-compatible configuration alias.
            self.adaptive_config["mean_conservation_drift"] = mean_drift

    def recommend_optimization_strategy(
        self,
        G: Any,
        operation_type: str = "general",
        current_performance: dict[str, float] | None = None,
    ) -> SelfOptimizationResult:
        """
        Recommend optimization strategy based on mathematical analysis and learning.
        """
        start_time = time.perf_counter()

        # Analyze mathematical structure
        mathematical_insights = self.analyze_mathematical_optimization_landscape(
            G, operation_type
        )

        # Find matching learned policies
        matching_policies = []
        if HAS_NETWORKX and G:
            num_nodes = len(G.nodes())
            density = nx.density(G)

            size_bucket = (
                "small" if num_nodes < 20 else "medium" if num_nodes < 100 else "large"
            )
            density_bucket = (
                "sparse"
                if density < SELF_OPT_DENSITY_SPARSE_CANONICAL
                else "medium" if density < SELF_OPT_DENSITY_DENSE_CANONICAL else "dense"
            )

            for policy in self.learned_policies:
                conditions = policy.conditions
                if (
                    conditions.get("graph_size_bucket") == size_bucket
                    and conditions.get("density_bucket") == density_bucket
                    and conditions.get("operation_type") == operation_type
                ):
                    matching_policies.append(policy)

        # Generate recommendations
        recommended_strategies = []
        predicted_speedups = {}

        # From learned policies
        for policy in matching_policies:
            strategy = policy.actions.get("recommended_strategy")
            measured_prediction = _finite_measurement(
                policy.average_improvement, positive=True
            )
            if strategy and measured_prediction is not None:
                recommended_strategies.append(strategy)
                predicted_speedups[strategy] = measured_prediction

        # From mathematical analysis
        math_recommendations = mathematical_insights.get(
            "nodal_equation_analysis", {}
        ).get("optimization_recommendations", [])
        recommended_strategies.extend(math_recommendations)

        # From pattern analysis
        pattern_hints = mathematical_insights.get("pattern_optimization_hints", [])
        recommended_strategies.extend(pattern_hints)

        # Remove duplicates while preserving order
        recommended_strategies = list(dict.fromkeys(recommended_strategies))

        # Balance and candidate-energy alerts are surfaced for review above.
        # They do not reorder computational strategies because they do not
        # identify a cause or prescribe a canonical structural operator.
        cf = mathematical_insights.get(
            "balance_feedback", mathematical_insights.get("conservation_feedback")
        )

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
            execution_time=execution_time,
            conservation_feedback=cf,
        )

    def optimize_automatically(
        self, G: Any, operation_type: str = "general", **kwargs
    ) -> dict[str, Any]:
        """
        Automatically apply best optimization strategy.

        Uses learned policies and mathematical analysis to select and apply
        the optimal strategy.
        """
        exec_kwargs = dict(kwargs)
        dry_run = bool(exec_kwargs.pop("dry_run", False))
        capture_snapshots = bool(exec_kwargs.pop("capture_snapshots", dry_run))
        seed_value = exec_kwargs.pop("seed", exec_kwargs.pop("random_seed", None))
        node_label = (
            exec_kwargs.pop("node", None)
            or exec_kwargs.pop("node_id", None)
            or exec_kwargs.pop("target_node", None)
            or exec_kwargs.pop("focus_node", None)
        )
        partition_label = exec_kwargs.pop("partition_id", None)
        report_label = partition_label if partition_label is not None else node_label
        output_dir = exec_kwargs.pop("output_dir", _DEFAULT_OUTPUT_DIR)
        operator_sequence = exec_kwargs.pop("operator_sequence", None)
        glyph_sequence = exec_kwargs.pop("glyph_sequence", None)
        sequence_context = exec_kwargs.pop("sequence_context", None)

        # Get recommendations
        recommendations = self.recommend_optimization_strategy(G, operation_type)
        baseline_snapshot = (
            self._capture_structural_snapshot(G) if capture_snapshots else None
        )
        validation_report = self._prepare_sequence_validation(
            operator_sequence,
            glyph_sequence,
            sequence_context,
            G=G,
            node=node_label,
        )

        if dry_run:
            payload = self._build_dry_run_payload(
                recommendations,
                baseline_snapshot,
                validation_report,
                operation_type,
                seed_value,
                report_label,
            )
            if partition_label is not None:
                payload["metadata"]["partition_id"] = str(partition_label)
                payload["metadata"]["node"] = _sanitize_label(node_label, "global")
            safe_seed = _sanitize_label(seed_value, "unseeded")
            safe_node = _sanitize_label(report_label, "global")
            if partition_label is not None:
                label_hash = hashlib.sha256(str(partition_label).encode("utf-8")).hexdigest()[:12]
                safe_node = f"{safe_node}_{label_hash}"
            snapshot_path, signature = self._persist_dry_run_payload(
                payload,
                output_dir,
                safe_seed,
                safe_node,
            )
            snapshots = (
                {
                    "before": baseline_snapshot,
                    "after": baseline_snapshot,
                }
                if baseline_snapshot
                else None
            )
            return {
                "dry_run": True,
                "snapshot_path": str(snapshot_path),
                "signature": signature,
                "recommendations": recommendations,
                "learning_updated": False,
                "telemetry_snapshots": snapshots,
                "validation": validation_report,
            }

        # Apply best strategy
        if recommendations.recommended_strategies and self.orchestrator:
            best_strategy = recommendations.recommended_strategies[0]

            # Map strategy name to OptimizationStrategy enum
            strategy_mapping = {
                "spectral_methods": OptimizationStrategy.SPECTRAL_FFT,
                "vectorized": OptimizationStrategy.NODAL_VECTORIZED,
                "cache": OptimizationStrategy.ADELIC_CACHE,
                "structural": OptimizationStrategy.STRUCTURAL_MEMO,
                "hybrid": OptimizationStrategy.HYBRID,
            }

            # Find matching strategy
            optimization_strategy = OptimizationStrategy.AUTO
            for name_part, strategy in strategy_mapping.items():
                if name_part in best_strategy.lower():
                    optimization_strategy = strategy
                    break

            # Capture a baseline for finite structural-balance diagnostics.
            conservation_before = None
            if HAS_CONSERVATION and G is not None:
                try:
                    conservation_before = capture_conservation_snapshot(G)
                except Exception:
                    conservation_before = None

            # Execute optimization
            try:
                profile = self.orchestrator.analyze_optimization_profile(
                    G, operation_type
                )
                result = self.orchestrator.execute_optimization(
                    G, operation_type, optimization_strategy, **exec_kwargs
                )

                # Finite balance / candidate-energy post-check. These fields
                # are diagnostic and do not validate grammar or decide whether
                # the computational optimization succeeded.
                conservation_result = None
                if (
                    conservation_before is not None
                    and HAS_CONSERVATION
                    and G is not None
                ):
                    try:
                        conservation_after = capture_conservation_snapshot(G)
                        balance = verify_conservation_balance(
                            conservation_before, conservation_after
                        )
                        lyapunov = compute_lyapunov_derivative(
                            conservation_before, conservation_after
                        )
                        alerts = detect_grammar_violations_from_conservation(balance)
                        conservation_result = {
                            "structural_charge_drift": balance.charge_drift,
                            "balance_rms_residual": balance.rms_residual,
                            "balance_quality": balance.conservation_quality,
                            "candidate_energy_nonincreasing": (
                                lyapunov.energy_derivative <= 0.0
                            ),
                            "candidate_energy_within_numerical_tolerance": (
                                lyapunov.is_stable
                            ),
                            "candidate_energy_derivative": (
                                lyapunov.energy_derivative
                            ),
                            "balance_alerts_detected": alerts["alerts_detected"],
                            "balance_alert_types": alerts.get("alert_types", []),
                            "grammar_validated": False,
                            # Backward-compatible historical keys. The legacy
                            # grammar-shaped fields stay false/empty.
                            "charge_drift": balance.charge_drift,
                            "rms_residual": balance.rms_residual,
                            "lyapunov_stable": lyapunov.is_stable,
                            "energy_derivative": lyapunov.energy_derivative,
                            "violations_detected": False,
                            "violation_types": [],
                        }
                    except Exception:
                        pass

                # Record only authoritative measurements. Legacy result
                # sentinels (speedup=1, memory=0, cache=0) are not evidence.
                perf_metrics: dict[str, Any] = _result_evidence(result)
                if conservation_result is not None:
                    perf_metrics["structural_charge_drift"] = conservation_result[
                        "structural_charge_drift"
                    ]
                    perf_metrics["candidate_energy_derivative"] = (
                        conservation_result["candidate_energy_derivative"]
                    )
                    perf_metrics["balance_rms_residual"] = conservation_result[
                        "balance_rms_residual"
                    ]
                    # Backward-compatible metric aliases.
                    perf_metrics["conservation_charge_drift"] = conservation_result[
                        "charge_drift"
                    ]
                    perf_metrics["conservation_energy_derivative"] = (
                        conservation_result["energy_derivative"]
                    )
                    perf_metrics["conservation_rms_residual"] = conservation_result[
                        "rms_residual"
                    ]

                experience = OptimizationExperience(
                    graph_properties={
                        "nodes": len(G.nodes()) if HAS_NETWORKX and G else 0,
                        "edges": len(G.edges()) if HAS_NETWORKX and G else 0,
                        "density": profile.edge_density,
                    },
                    operation_type=operation_type,
                    strategy_used=optimization_strategy.value,
                    parameters=exec_kwargs,
                    performance_metrics=perf_metrics,
                    timestamp=time.time(),
                    success=result.accuracy_preserved,
                    mathematical_signature=recommendations.mathematical_insights,
                )

                self.learn_from_experience(experience)
                after_snapshot = (
                    self._capture_structural_snapshot(G) if capture_snapshots else None
                )
                return {
                    "optimization_result": result,
                    "strategy_used": optimization_strategy.value,
                    "recommendations": recommendations,
                    "learning_updated": True,
                    "balance_diagnostics": conservation_result,
                    # Backward-compatible outer key.
                    "conservation": conservation_result,
                    "telemetry_snapshots": (
                        {
                            "before": baseline_snapshot,
                            "after": after_snapshot,
                        }
                        if capture_snapshots
                        else None
                    ),
                    "validation": validation_report,
                }

            except Exception as e:
                return {
                    "error": str(e),
                    "recommendations": recommendations,
                    "learning_updated": False,
                    "telemetry_snapshots": (
                        {"before": baseline_snapshot} if baseline_snapshot else None
                    ),
                    "validation": validation_report,
                }
        else:
            return {
                "message": "No optimization applied",
                "recommendations": recommendations,
                "learning_updated": False,
                "telemetry_snapshots": (
                    {"before": baseline_snapshot} if baseline_snapshot else None
                ),
                "validation": validation_report,
            }

    def export_learned_knowledge(self) -> dict[str, Any]:
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
                    "applications": p.applications_count,
                }
                for p in self.learned_policies
            ],
            "adaptive_configuration": dict(self.adaptive_config),
            "performance_statistics": {
                "total_attempts": self.optimization_attempts,
                "successful_optimizations": self.successful_optimizations,
                "success_rate": self.successful_optimizations
                / max(1, self.optimization_attempts),
                "experience_count": len(self.experience_history),
            },
            "mathematical_insights": dict(self.mathematical_insights),
        }

    def import_learned_knowledge(self, knowledge: dict[str, Any]) -> None:
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
                        applications_count=p_data.get("applications", 0),
                    )
                    imported_policies.append(policy)
                self.learned_policies.extend(imported_policies)

            # Import adaptive configuration
            if "adaptive_configuration" in knowledge:
                self.adaptive_config.update(knowledge["adaptive_configuration"])

            # Import insights
            if "mathematical_insights" in knowledge:
                self.mathematical_insights.update(knowledge["mathematical_insights"])

    def _capture_structural_snapshot(self, G: Any) -> dict[str, Any] | None:
        """Capture graph metrics and unified telemetry for reporting."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if G is None or not HAS_NETWORKX:
            snapshot = {"timestamp": timestamp, "graph": None, "telemetry": None}
        else:
            graph_metrics = _graph_structure(G)
            snapshot = {
                "timestamp": timestamp,
                "graph": graph_metrics,
                "telemetry": None,
            }

        if HAS_UNIFIED_FIELDS and G is not None:
            try:
                telemetry = compute_unified_telemetry(G)
                snapshot["telemetry"] = _json_safe(telemetry)
            except (
                Exception
            ) as exc:  # pragma: no cover - telemetry errors are informational
                snapshot["telemetry_error"] = str(exc)

        if HAS_METRIC_OPERATORS and G is not None and HAS_NETWORKX:
            coherence_value: float | None = None
            sense_value: float | None = None
            try:
                coherence_value = (
                    float(compute_coherence(G)) if compute_coherence else None
                )
            except Exception:
                coherence_value = None
            try:
                if compute_Si:
                    si_payload = compute_Si(G, inplace=False)
                    sense_value = _sense_index_mean(si_payload)
            except Exception:
                sense_value = None
            if coherence_value is not None:
                snapshot["coherence"] = coherence_value
            if sense_value is not None:
                snapshot["sense_index"] = sense_value
        return snapshot

    def _prepare_sequence_validation(
        self,
        operator_sequence: Any,
        glyph_sequence: Any,
        context: dict[str, Any] | None,
        *,
        G: Any = None,
        node: Any = None,
    ) -> dict[str, Any] | None:
        """Validate operator/glyph sequences when provided.

        Performs two layers of validation:
        1. **Batch** — full-sequence grammar check via ``validate_sequence()``.
        2. **Incremental** — per-step check against the node's live history
           via ``validate_sequence_incremental()`` (when *G* and *node* are
           available).  This catches violations that only become visible in
           context (e.g. destabilizer debt from prior operations).
        """
        sequence = (
            operator_sequence if operator_sequence is not None else glyph_sequence
        )
        if sequence is None:
            return None
        if isinstance(sequence, str):
            sequence_iterable = [sequence]
        else:
            sequence_iterable = list(sequence)
        tokens = _canonicalize_sequence_tokens(sequence_iterable)
        if not tokens:
            return None
        if context:
            outcome = validate_sequence(tokens, context=context)
        else:
            outcome = validate_sequence(tokens)
        summary = _json_safe(getattr(outcome, "summary", {}))
        if not outcome.passed:
            message = (
                summary.get("message") or outcome.message or "grammar validation failed"
            )
            raise TNFRValueError(
                f"Operator sequence failed grammar validation: {message}",
                context={
                    "tokens": tokens,
                    "summary": summary,
                    "outcome_message": outcome.message,
                },
                suggestion="Ensure sequence follows U1-U6 grammar rules.",
            )

        # Incremental per-node validation (proactive, GAP #4)
        incremental_report: list[dict[str, Any]] | None = None
        if G is not None and node is not None and node in G.nodes:
            try:
                from ..operators.grammar_dynamics import validate_sequence_incremental

                step_results = validate_sequence_incremental(G, node, tokens)
                step_violations = [
                    {
                        "step": i,
                        "token": sr.candidate,
                        "allowed": sr.allowed,
                        "violations": [
                            {
                                "rule": v.rule,
                                "message": v.message,
                                "severity": v.severity,
                            }
                            for v in sr.violations
                        ],
                        "suggested": sr.suggested_alternative,
                    }
                    for i, sr in enumerate(step_results)
                    if sr.violations
                ]
                if step_violations:
                    incremental_report = step_violations
            except Exception:
                pass  # incremental validation is advisory, never blocks

        report: dict[str, Any] = {
            "passed": True,
            "message": outcome.message,
            "summary": summary,
            "tokens": _json_safe(list(getattr(outcome, "tokens", tokens))),
            "canonical_tokens": _json_safe(
                list(getattr(outcome, "canonical_tokens", tokens))
            ),
        }
        if incremental_report:
            report["incremental_violations"] = incremental_report
        return report

    def _serialize_recommendations(
        self, recommendations: SelfOptimizationResult
    ) -> dict[str, Any]:
        """Convert recommendation dataclass into JSON-serializable payload."""
        return {
            "recommended_strategies": list(recommendations.recommended_strategies),
            "predicted_speedups": _json_safe(recommendations.predicted_speedups),
            "optimization_improvements": _json_safe(
                recommendations.optimization_improvements
            ),
            "mathematical_insights": _json_safe(recommendations.mathematical_insights),
            "adaptive_configurations": _json_safe(
                recommendations.adaptive_configurations
            ),
            "execution_time": recommendations.execution_time,
            "learned_policies": [
                {
                    "name": policy.policy_name,
                    "objective": policy.objective.value,
                    "conditions": _json_safe(policy.conditions),
                    "actions": _json_safe(policy.actions),
                    "confidence": policy.confidence,
                    "success_rate": policy.success_rate,
                    "average_improvement": policy.average_improvement,
                    "applications": policy.applications_count,
                }
                for policy in recommendations.learned_policies
            ],
        }

    def _build_dry_run_payload(
        self,
        recommendations: SelfOptimizationResult,
        baseline_snapshot: dict[str, Any] | None,
        validation_report: dict[str, Any] | None,
        operation_type: str,
        seed_value: Any,
        node_label: Any,
    ) -> dict[str, Any]:
        """Assemble payload for dry-run persistence."""
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        seed_label = _sanitize_label(seed_value, "unseeded")
        node_name = _sanitize_label(node_label, "global")
        telemetry_snapshots = None
        if baseline_snapshot is not None:
            telemetry_snapshots = {
                "before": baseline_snapshot,
                "after": baseline_snapshot,
            }
        payload = {
            "metadata": {
                "timestamp": timestamp,
                "operation_type": operation_type,
                "seed": seed_label,
                "node": node_name,
                "dry_run": True,
                "objective": self.optimization_objective.value,
                "learning_strategy": self.learning_strategy.value,
            },
            "telemetry": telemetry_snapshots,
            "recommendations": self._serialize_recommendations(recommendations),
            "validation": validation_report or {"status": "not_provided"},
            "learning_state": {
                "experience_count": len(self.experience_history),
                "policy_count": len(self.learned_policies),
                "successful_optimizations": self.successful_optimizations,
            },
        }
        return _json_safe(payload)

    def _persist_dry_run_payload(
        self,
        payload: dict[str, Any],
        output_dir: Any,
        safe_seed: str,
        safe_node: str,
    ) -> tuple[Path, str]:
        """Write payload to disk and emit SHA-256 signature."""
        base_dir = Path(output_dir)
        target_dir = base_dir / safe_seed
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{safe_node}.json"
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        file_path.write_text(serialized, encoding="utf-8")
        signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        signature_path = file_path.with_suffix(file_path.suffix + ".sha256")
        signature_path.write_text(f"{signature}  {file_path.name}\n", encoding="utf-8")
        return file_path, signature


# Factory functions
def create_self_optimizing_engine(**kwargs) -> TNFRSelfOptimizingEngine:
    """Create self-optimizing engine."""
    return TNFRSelfOptimizingEngine(**kwargs)


def auto_optimize_tnfr_computation(
    G: Any, operation_type: str = "general", **kwargs
) -> dict[str, Any]:
    """Convenience function for automatic optimization."""
    engine = create_self_optimizing_engine()
    return engine.optimize_automatically(G, operation_type, **kwargs)
