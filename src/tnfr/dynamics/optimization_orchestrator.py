"""Evidence-aware coordination of TNFR computational strategies.

The orchestrator dispatches nodal proposals, the scoped FFT EPI-diffusion
realization, adelic trace interpolation, and structural-field memoization. It
records elapsed wall time and cache counters when an engine exposes them.
Speedup and process-memory claims require an explicit measured baseline; absent
that evidence, their authoritative values are ``None`` in
``details["performance_measurements"]``. Legacy scalar result fields retain
neutral compatibility values and are never learned as performance evidence.

Strategy selection is a deterministic operational policy informed by compatible
operations and any previously measured speedups. It is not a theorem that the
selected strategy is optimal for a particular graph or machine.
"""

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_VF
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..operators.network_stage import GraphTransactionSnapshot
from ..operators.nodal_equation import DEFAULT_NODAL_EQUATION_TOLERANCE
from ..physics.structural_diffusion import structural_diffusion_operator
from ..types import real_scalar_epi

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import all optimization engines
try:
    from .adelic import AdelicDynamics
    from .fft_engine import create_fft_engine
    from .nodal_optimizer import create_nodal_optimizer
    from .structural_cache import get_structural_cache

    HAS_OPTIMIZATION_ENGINES = True
except ImportError:
    HAS_OPTIMIZATION_ENGINES = False

# Operational engine-tuning knobs (not TNFR physics) → tnfr.constants.operational
from ..constants.operational import (
    OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,
    OPT_ORCH_BEST_THRESHOLD_CANONICAL,
    OPT_ORCH_DENSE_BOOST_CANONICAL,
    OPT_ORCH_DENSITY_THRESHOLD_CANONICAL,
    OPT_ORCH_FFT_BOOST_CANONICAL,
    OPT_ORCH_SMALL_PENALTY_CANONICAL,
    OPT_ORCH_VECTORIZED_BOOST_CANONICAL,
)


class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios."""

    AUTO = "auto"  # Automatic selection
    SPECTRAL_FFT = "spectral_fft"  # Public compatibility name for spectral methods
    NODAL_VECTORIZED = "nodal_vec"  # Vectorized nodal equation
    ADELIC_CACHE = "adelic_cache"  # Cached trace computations
    STRUCTURAL_MEMO = "struct_memo"  # Structural field memoization
    HYBRID = "hybrid"  # Combination approach


FFT_EPI_DIFFUSION_OPERATION = "epi_diffusion"
FFT_EPI_DIFFUSION_PRESSURE_MODEL = "epi_diffusion"


def validate_fft_epi_diffusion_dispatch(
    operation: str, pressure_model: Any = None
) -> str:
    """Validate the graph-spectral engine's single physical realization.

    The FFT dynamics engine evolves only the canonical EPI diffusion channel,
    ``DeltaNFR_epi = -L_rw EPI``. Dispatchers must establish that contract
    before handing the live graph to the mutating engine.
    """
    if operation != FFT_EPI_DIFFUSION_OPERATION:
        raise TNFRValueError(
            "SPECTRAL_FFT supports only operation='epi_diffusion'"
        )
    resolved_model = (
        FFT_EPI_DIFFUSION_PRESSURE_MODEL
        if pressure_model is None
        else pressure_model
    )
    if resolved_model != FFT_EPI_DIFFUSION_PRESSURE_MODEL:
        raise TNFRValueError(
            "SPECTRAL_FFT requires pressure_model='epi_diffusion'"
        )
    return FFT_EPI_DIFFUSION_PRESSURE_MODEL


@dataclass
class OptimizationProfile:
    """Profile for optimization decision-making."""

    graph_size: int = 0
    edge_density: float = 0.0
    operation_type: str = "general"
    expected_iterations: int = 1
    memory_budget_mb: float = 256.0
    prefer_accuracy: bool = True
    enable_caching: bool = True
    available_strategies: list[OptimizationStrategy] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Strategy result with compatibility fields and explicit evidence.

    execution_time is measured by the public dispatcher. The scalar
    speedup_factor=1.0 and memory_used_mb=0.0 values are neutral legacy
    sentinels whenever the corresponding authoritative measurement in
    details["performance_measurements"] is None. Zero cache counters are likewise
    neutral when details["cache_measurements"] reports None. The legacy
    accuracy_preserved flag mirrors the declared verification result; its basis
    and baseline scope live in details["accuracy_verification"].
    """

    strategy_used: OptimizationStrategy
    execution_time: float
    speedup_factor: float
    cache_hits: int
    cache_misses: int
    memory_used_mb: float
    accuracy_preserved: bool
    details: dict[str, Any] = field(default_factory=dict)


def _finite_real(value: Any, label: str, *, nonnegative: bool = False) -> float:
    """Return a finite non-Boolean real value."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TNFRValueError(f"{label} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise TNFRValueError(f"{label} must be a {qualifier} real scalar")
    return result


def _positive_integer(value: Any, label: str) -> int:
    """Return a strictly positive non-Boolean integer."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TNFRValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise TNFRValueError(f"{label} must be a positive integer")
    return result


def _counter_value(mapping: Mapping[str, Any], key: str) -> int | None:
    """Read a nonnegative integer counter without inventing absent data."""

    value = mapping.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        return None
    result = int(value)
    return result if result >= 0 else None


def _counter_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], key: str
) -> int | None:
    """Return a monotone counter delta when both snapshots expose the counter."""

    start = _counter_value(before, key)
    end = _counter_value(after, key)
    if start is None or end is None or end < start:
        return None
    return end - start


def _performance_measurements(
    *,
    speedup_factor: float | None = None,
    memory_used_mb: float | None = None,
    **measurements: Any,
) -> dict[str, Any]:
    """Build the authoritative performance-evidence payload."""

    return {
        "speedup_factor": speedup_factor,
        "memory_used_mb": memory_used_mb,
        **measurements,
    }


def _unavailable_result(
    strategy: OptimizationStrategy, message: str, *, basis: str = "engine_availability"
) -> OptimizationResult:
    """Return a structured failed result without performance claims."""

    return OptimizationResult(
        strategy_used=strategy,
        execution_time=0.0,
        speedup_factor=1.0,
        cache_hits=0,
        cache_misses=0,
        memory_used_mb=0.0,
        accuracy_preserved=False,
        details={
            "error": message,
            "performance_measurements": _performance_measurements(),
            "cache_measurements": {"hits": None, "misses": None},
            "accuracy_verification": {
                "basis": basis,
                "passed": False,
                "baseline_comparison_performed": False,
            },
        },
    )


def _read_scalar_epi(graph: Any, node: Any) -> float:
    """Read the signed scalar EPI chart used by the nodal proposal."""

    raw = get_attr(
        graph.nodes[node],
        ALIAS_EPI,
        0.0,
        strict=True,
        conv=lambda value: value,
    )
    scalar = real_scalar_epi(raw)
    if scalar is None:
        raise TNFRValueError(f"node {node!r} EPI has no real scalar embedding")
    return _finite_real(scalar, f"node {node!r} EPI")


def _structural_entries_equal(left: Any, right: Any) -> bool:
    """Compare the complete public structural-cache entry."""

    scalar_fields = (
        "phi_s",
        "grad_phi",
        "k_phi",
        "xi_c",
        "coherence",
        "phase_sync",
        "timestamp",
        "topology_hash",
        "spectral_basis_signature",
        "coordination_nodes",
    )
    try:
        if any(getattr(left, name) != getattr(right, name) for name in scalar_fields):
            return False
        for name in ("eigenvalues", "eigenvectors"):
            lhs = getattr(left, name)
            rhs = getattr(right, name)
            if lhs is None or rhs is None:
                if lhs is not rhs:
                    return False
            elif not np.array_equal(np.asarray(lhs), np.asarray(rhs), equal_nan=True):
                return False
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _measured_speedup(result: OptimizationResult) -> float | None:
    """Read a positive measured speedup from one result's evidence payload."""

    measurements = result.details.get("performance_measurements")
    if not isinstance(measurements, Mapping):
        return None
    value = measurements.get("speedup_factor")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    measured = float(value)
    return measured if math.isfinite(measured) and measured > 0.0 else None


class TNFROptimizationOrchestrator:
    """Coordinate scoped optimization services and retain measured evidence."""

    def __init__(self, default_memory_budget: float = 512.0):
        self.default_memory_budget = default_memory_budget

        # Initialize optimization engines
        if HAS_OPTIMIZATION_ENGINES:
            self.nodal_optimizer = create_nodal_optimizer()
            self.structural_cache = get_structural_cache()
            self.fft_engine = create_fft_engine()
            self.adelic_engine = AdelicDynamics()
        else:
            self.nodal_optimizer = None
            self.structural_cache = None
            self.fft_engine = None
            self.adelic_engine = None

        # Performance tracking
        self.optimization_history: list[OptimizationResult] = []
        self.strategy_performance: dict[OptimizationStrategy, list[float]] = {}

        # Retained as a neutral compatibility attribute. This coordinator
        # delegates caching to its component engines.
        self.global_cache = None

    def analyze_optimization_profile(
        self, G: Any, operation_type: str = "general"
    ) -> OptimizationProfile:
        """
        Build the graph/operation profile consumed by the selection policy.
        """
        if not HAS_NETWORKX or G is None:
            return OptimizationProfile()

        # Basic graph metrics
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        edge_density = (
            (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
        )

        # Determine available strategies based on graph properties
        available_strategies = []

        # The graph-spectral dynamics engine realizes only EPI diffusion.
        if (
            operation_type == FFT_EPI_DIFFUSION_OPERATION
            and num_nodes > 20
            and edge_density > OPT_ORCH_DENSITY_THRESHOLD_CANONICAL
        ):  # = 0.1 (operational)
            available_strategies.append(OptimizationStrategy.SPECTRAL_FFT)

        # Nodal proposal service is eligible above the configured size band.
        if num_nodes > 10:
            available_strategies.append(OptimizationStrategy.NODAL_VECTORIZED)

        # Adelic trace interpolation is scoped to its declared operation labels.
        if operation_type in ["temporal", "arithmetic", "trace"]:
            available_strategies.append(OptimizationStrategy.ADELIC_CACHE)

        # Structural memoization remains an available read-only service.
        available_strategies.append(OptimizationStrategy.STRUCTURAL_MEMO)

        # Auto and hybrid always available
        available_strategies.extend(
            [OptimizationStrategy.AUTO, OptimizationStrategy.HYBRID]
        )

        return OptimizationProfile(
            graph_size=num_nodes,
            edge_density=edge_density,
            operation_type=operation_type,
            available_strategies=available_strategies,
            memory_budget_mb=self.default_memory_budget,
        )

    def select_optimal_strategy(
        self,
        profile: OptimizationProfile,
        force_strategy: OptimizationStrategy | None = None,
    ) -> OptimizationStrategy:
        """Apply the declared selection policy and measured timing history.

        The compatibility name is retained; without a representative baseline
        this method selects a candidate and does not certify global optimality.
        """
        if (
            force_strategy is not None
            and force_strategy in profile.available_strategies
        ):
            return force_strategy

        # Performance-based selection using historical data
        best_strategy = OptimizationStrategy.AUTO
        best_score = 0.0

        for strategy in profile.available_strategies:
            if strategy == OptimizationStrategy.AUTO:
                continue

            # Calculate strategy score based on:
            # 1. Historical performance
            # 2. Graph characteristics
            # 3. Resource constraints

            score = 1.0  # Base score

            # Only explicitly measured baseline ratios enter this history.
            if strategy in self.strategy_performance:
                avg_speedup = float(np.mean(self.strategy_performance[strategy]))
                score *= avg_speedup

            # Graph size preferences
            if strategy == OptimizationStrategy.SPECTRAL_FFT:
                if profile.graph_size > 50:
                    # Operational large-graph policy multiplier.
                    score *= OPT_ORCH_FFT_BOOST_CANONICAL
                elif profile.graph_size < 20:
                    # Operational small-graph policy multiplier.
                    score *= OPT_ORCH_SMALL_PENALTY_CANONICAL

            elif strategy == OptimizationStrategy.NODAL_VECTORIZED:
                if 10 <= profile.graph_size <= 100:
                    # Operational medium-size policy multiplier.
                    score *= OPT_ORCH_VECTORIZED_BOOST_CANONICAL

            elif strategy == OptimizationStrategy.ADELIC_CACHE:
                if profile.operation_type in ["temporal", "arithmetic"]:
                    # Operational adelic-label policy multiplier.
                    score *= OPT_ORCH_ARITHMETIC_BOOST_CANONICAL

            # Density preferences
            if strategy == OptimizationStrategy.SPECTRAL_FFT:
                if (
                    profile.edge_density > OPT_ORCH_DENSE_BOOST_CANONICAL
                ):  # ≈ 0.3710 → canonical
                    # Operational dense-graph policy multiplier.
                    score *= OPT_ORCH_BEST_THRESHOLD_CANONICAL

            # Update best strategy
            if score > best_score:
                best_score = score
                best_strategy = strategy

        # Fallback to hybrid for complex cases
        if (
            best_strategy == OptimizationStrategy.AUTO
            or best_score < OPT_ORCH_BEST_THRESHOLD_CANONICAL
        ):  # ≈ 0.7006 → canonical
            if OptimizationStrategy.HYBRID in profile.available_strategies:
                return OptimizationStrategy.HYBRID
            else:
                return OptimizationStrategy.NODAL_VECTORIZED  # Compatibility fallback

        return best_strategy

    def execute_optimization(
        self, G: Any, operation: str, strategy: OptimizationStrategy, **kwargs
    ) -> OptimizationResult:
        """Execute one strategy and attach measured timing and verification."""

        start_time = time.perf_counter()
        selected_strategy = strategy
        if strategy == OptimizationStrategy.AUTO:
            profile = self.analyze_optimization_profile(G, operation)
            selected_strategy = self.select_optimal_strategy(profile)
            if selected_strategy == OptimizationStrategy.AUTO:
                selected_strategy = OptimizationStrategy.HYBRID

        try:
            if selected_strategy == OptimizationStrategy.SPECTRAL_FFT:
                result = self._execute_fft_optimization(G, operation, **kwargs)
            elif selected_strategy == OptimizationStrategy.NODAL_VECTORIZED:
                result = self._execute_nodal_optimization(G, operation, **kwargs)
            elif selected_strategy == OptimizationStrategy.ADELIC_CACHE:
                result = self._execute_adelic_optimization(G, operation, **kwargs)
            elif selected_strategy == OptimizationStrategy.STRUCTURAL_MEMO:
                result = self._execute_structural_optimization(G, operation, **kwargs)
            elif selected_strategy == OptimizationStrategy.HYBRID:
                result = self._execute_hybrid_optimization(G, operation, **kwargs)
            else:
                raise TNFRValueError(
                    f"Unsupported optimization strategy: {selected_strategy!r}"
                )
        except Exception as exc:
            result = _unavailable_result(
                selected_strategy,
                str(exc),
                basis="execution_exception",
            )

        elapsed = time.perf_counter() - start_time
        result.execution_time = elapsed
        result.strategy_used = selected_strategy

        measurements = result.details.get("performance_measurements")
        if not isinstance(measurements, dict):
            measurements = _performance_measurements()
            result.details["performance_measurements"] = measurements
        measurements.setdefault("speedup_factor", None)
        measurements.setdefault("memory_used_mb", None)

        verification = result.details.get("accuracy_verification")
        if not isinstance(verification, dict):
            verification = {
                "basis": "strategy_verification_missing",
                "passed": False,
                "baseline_comparison_performed": False,
            }
            result.details["accuracy_verification"] = verification
        verified = verification.get("passed")
        result.accuracy_preserved = (
            verified is True and "error" not in result.details
        )

        self._update_performance_history(result)
        return result

    def _execute_fft_optimization(
        self, G: Any, operation: str, **kwargs
    ) -> OptimizationResult:
        """Execute the graph-spectral EPI-diffusion realization atomically."""

        pressure_model = validate_fft_epi_diffusion_dispatch(
            operation, kwargs.get("pressure_model")
        )
        residual_tolerance = _finite_real(
            kwargs.get(
                "nodal_residual_tolerance",
                G.graph.get(
                    "NODAL_EQUATION_TOLERANCE",
                    DEFAULT_NODAL_EQUATION_TOLERANCE,
                ),
            ),
            "nodal_residual_tolerance",
            nonnegative=True,
        )
        if not self.fft_engine:
            return _unavailable_result(
                OptimizationStrategy.SPECTRAL_FFT,
                "FFT engine not available",
            )

        num_steps = kwargs.get("num_steps", 10)
        dt = kwargs.get("dt", 0.01)
        transaction = GraphTransactionSnapshot(G)
        committed = False
        failure: BaseException | None = None
        try:
            fft_results = self.fft_engine.run_fft_simulation(G, num_steps, dt)
            if not isinstance(fft_results, Mapping):
                raise TNFRValueError("FFT engine must return a result mapping")

            details = dict(fft_results)
            raw_residual = fft_results.get("max_nodal_residual")
            try:
                max_nodal_residual = float(raw_residual)
            except (OverflowError, TypeError, ValueError):
                max_nodal_residual = float("nan")
            equation_verified = bool(
                fft_results.get("status") == "success"
                and np.isfinite(max_nodal_residual)
                and max_nodal_residual <= residual_tolerance
            )
            raw_cache_hits = _counter_value(fft_results, "cache_hits")
            cache_hits = raw_cache_hits if raw_cache_hits is not None else 0
            details.update(
                {
                    "operation": FFT_EPI_DIFFUSION_OPERATION,
                    "pressure_model": pressure_model,
                    "transaction_committed": equation_verified,
                    "performance_measurements": _performance_measurements(
                        throughput_steps_per_second=fft_results.get(
                            "steps_per_second"
                        )
                    ),
                    "cache_measurements": {
                        "hits": raw_cache_hits,
                        "misses": None,
                        "scope": "simulation_delta",
                    },
                    "accuracy_verification": {
                        "basis": "nodal_equation_residual",
                        "max_nodal_residual": raw_residual,
                        "tolerance": residual_tolerance,
                        "passed": equation_verified,
                        "baseline_comparison_performed": False,
                    },
                }
            )
            result = OptimizationResult(
                strategy_used=OptimizationStrategy.SPECTRAL_FFT,
                execution_time=_finite_real(
                    fft_results.get("simulation_time", 0.0),
                    "FFT simulation_time",
                    nonnegative=True,
                ),
                speedup_factor=1.0,
                cache_hits=cache_hits,
                cache_misses=0,
                memory_used_mb=0.0,
                accuracy_preserved=equation_verified,
                details=details,
            )
            committed = equation_verified
            return result
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if not committed:
                if failure is None:
                    transaction.restore(G)
                else:
                    transaction.restore_after_failure(G, failure)

    def _execute_nodal_optimization(
        self, G: Any, operation: str, **kwargs
    ) -> OptimizationResult:
        """Compute and verify one detached simultaneous nodal proposal."""

        if not self.nodal_optimizer:
            return _unavailable_result(
                OptimizationStrategy.NODAL_VECTORIZED,
                "Nodal optimizer not available",
            )

        dt = _finite_real(kwargs.get("dt", 0.01), "dt")
        if dt <= 0.0:
            raise TNFRValueError("dt must be positive")
        residual_tolerance = _finite_real(
            kwargs.get(
                "nodal_residual_tolerance",
                G.graph.get(
                    "NODAL_EQUATION_TOLERANCE",
                    DEFAULT_NODAL_EQUATION_TOLERANCE,
                ),
            ),
            "nodal_residual_tolerance",
            nonnegative=True,
        )

        stats_before = self.nodal_optimizer.get_optimization_stats()
        local_start = time.perf_counter()
        proposals = self.nodal_optimizer.compute_vectorized_nodal_evolution(G, dt)
        local_elapsed = time.perf_counter() - local_start
        stats_after = self.nodal_optimizer.get_optimization_stats()
        if not isinstance(proposals, Mapping):
            raise TNFRValueError("Nodal optimizer must return a proposal mapping")
        proposal_metadata = getattr(proposals, "metadata", {})
        if not isinstance(proposal_metadata, Mapping):
            proposal_metadata = {}
        proposal_metadata = dict(proposal_metadata)

        nodes, laplacian = structural_diffusion_operator(G)
        node_order = tuple(nodes)
        complete = (
            set(proposals) == set(node_order)
            and len(proposals) == len(node_order)
        )
        epi = np.asarray(
            [_read_scalar_epi(G, node) for node in node_order],
            dtype=float,
        )
        frequencies = np.asarray(
            [
                _finite_real(
                    get_attr(
                        G.nodes[node],
                        ALIAS_VF,
                        1.0,
                        strict=True,
                        conv=lambda value: value,
                    ),
                    f"node {node!r} structural frequency",
                    nonnegative=True,
                )
                for node in node_order
            ],
            dtype=float,
        )
        pressure = -(np.asarray(laplacian, dtype=float) @ epi)
        residuals: list[float] = []
        finite_phase = True
        if complete:
            for index, node in enumerate(node_order):
                proposal = proposals[node]
                try:
                    if len(proposal) != 2:
                        raise TypeError
                    proposed_epi = _finite_real(
                        proposal[0], f"node {node!r} proposed EPI"
                    )
                    _finite_real(proposal[1], f"node {node!r} proposed phase")
                except (IndexError, KeyError, TypeError) as exc:
                    raise TNFRValueError(
                        "Each nodal proposal must be a finite (EPI, phase) pair"
                    ) from exc
                realised_rate = (proposed_epi - epi[index]) / dt
                residuals.append(
                    abs(realised_rate - frequencies[index] * pressure[index])
                )
        else:
            finite_phase = False

        max_residual = float(max(residuals, default=0.0))
        equation_verified = bool(
            complete
            and finite_phase
            and math.isfinite(max_residual)
            and max_residual <= residual_tolerance
        )
        hit_delta = _counter_delta(stats_before, stats_after, "cache_hits")
        miss_delta = _counter_delta(stats_before, stats_after, "cache_misses")
        return OptimizationResult(
            strategy_used=OptimizationStrategy.NODAL_VECTORIZED,
            execution_time=local_elapsed,
            speedup_factor=1.0,
            cache_hits=hit_delta or 0,
            cache_misses=miss_delta or 0,
            memory_used_mb=0.0,
            accuracy_preserved=equation_verified,
            details={
                "operation": operation,
                "node_updates": len(proposals),
                "state_committed": False,
                "proposal_metadata": proposal_metadata,
                "stability_not_certified": bool(
                    proposal_metadata.get("stability_not_certified", False)
                ),
                "stats_before": stats_before,
                "stats_after": stats_after,
                "performance_measurements": _performance_measurements(
                    proposal_nodes_per_second=(
                        len(proposals) / local_elapsed if local_elapsed > 0.0 else None
                    )
                ),
                "cache_measurements": {
                    "hits": hit_delta,
                    "misses": miss_delta,
                    "scope": "operation_delta",
                },
                "accuracy_verification": {
                    "basis": "epi_diffusion_nodal_residual_and_finite_phase",
                    "complete_node_coverage": complete,
                    "max_nodal_residual": max_residual,
                    "tolerance": residual_tolerance,
                    "passed": equation_verified,
                    "baseline_comparison_performed": False,
                },
            },
        )

    def _execute_adelic_optimization(
        self, G: Any, operation: str, **kwargs
    ) -> OptimizationResult:
        """Precompute an adelic trace grid and verify interpolation residuals."""

        del G
        if not self.adelic_engine:
            return _unavailable_result(
                OptimizationStrategy.ADELIC_CACHE,
                "Adelic engine not available",
            )

        t_start = _finite_real(kwargs.get("t_start", 10.0), "t_start")
        t_end = _finite_real(kwargs.get("t_end", 20.0), "t_end")
        if t_end <= t_start:
            raise TNFRValueError("t_end must be greater than t_start")
        resolution = _positive_integer(
            kwargs.get("landscape_resolution", 1000),
            "landscape_resolution",
        )
        if resolution < 2:
            raise TNFRValueError("landscape_resolution must be at least 2")
        sample_count = _positive_integer(
            kwargs.get("verification_samples", 100),
            "verification_samples",
        )
        interpolation_tolerance = _finite_real(
            kwargs.get("adelic_interpolation_tolerance", 1e-3),
            "adelic_interpolation_tolerance",
            nonnegative=True,
        )

        local_start = time.perf_counter()
        self.adelic_engine.precompute_trace_landscape(
            t_start, t_end, resolution=resolution
        )
        test_times = np.linspace(t_start, t_end, sample_count)
        interpolated = np.asarray(
            [
                self.adelic_engine.compute_geometric_trace(float(t))
                for t in test_times
            ],
            dtype=float,
        )

        primes = np.asarray(self.adelic_engine.primes, dtype=float)
        frequencies = np.asarray(self.adelic_engine.nu_f, dtype=float)
        if (
            primes.ndim != 1
            or frequencies.shape != primes.shape
            or np.any(primes <= 0.0)
            or not np.all(np.isfinite(primes))
            or not np.all(np.isfinite(frequencies))
        ):
            raise TNFRValueError("Adelic prime and frequency arrays are inconsistent")
        weights = frequencies / np.sqrt(primes)
        direct = np.abs(
            np.sum(
                np.exp(1j * np.outer(test_times, frequencies)) * weights,
                axis=1,
            )
        )
        residual = np.abs(interpolated - direct)
        max_residual = float(np.max(residual)) if residual.size else 0.0
        verified = bool(
            np.all(np.isfinite(interpolated))
            and np.all(np.isfinite(direct))
            and math.isfinite(max_residual)
            and max_residual <= interpolation_tolerance
        )
        local_elapsed = time.perf_counter() - local_start

        trace_cache = getattr(self.adelic_engine, "_trace_cache", None)
        managed_cache_bytes: int | None = None
        if isinstance(trace_cache, tuple) and len(trace_cache) == 2:
            managed_cache_bytes = sum(
                int(np.asarray(component).nbytes) for component in trace_cache
            )
        managed_memory_mb = (
            managed_cache_bytes / (1024.0 * 1024.0)
            if managed_cache_bytes is not None
            else None
        )

        return OptimizationResult(
            strategy_used=OptimizationStrategy.ADELIC_CACHE,
            execution_time=local_elapsed,
            speedup_factor=1.0,
            cache_hits=0,
            cache_misses=0,
            memory_used_mb=managed_memory_mb or 0.0,
            accuracy_preserved=verified,
            details={
                "operation": operation,
                "trace_evaluations": sample_count,
                "landscape_resolution": resolution,
                "performance_measurements": _performance_measurements(
                    memory_used_mb=managed_memory_mb,
                    trace_evaluations_per_second=(
                        sample_count / local_elapsed if local_elapsed > 0.0 else None
                    ),
                    memory_scope=(
                        "managed_trace_grid_only"
                        if managed_cache_bytes is not None
                        else None
                    ),
                ),
                "cache_measurements": {
                    "hits": None,
                    "misses": None,
                    "landscape_interpolation_evaluations": sample_count,
                },
                "accuracy_verification": {
                    "basis": "interpolated_trace_against_direct_formula",
                    "max_absolute_residual": max_residual,
                    "tolerance": interpolation_tolerance,
                    "verification_samples": sample_count,
                    "passed": verified,
                    "baseline_comparison_performed": True,
                },
            },
        )

    def _execute_structural_optimization(
        self, G: Any, operation: str, **kwargs
    ) -> OptimizationResult:
        """Verify one structural-field cache miss followed by an exact hit."""

        del kwargs
        if not self.structural_cache:
            return _unavailable_result(
                OptimizationStrategy.STRUCTURAL_MEMO,
                "Structural cache not available",
            )

        stats_before = self.structural_cache.get_cache_stats()
        local_start = time.perf_counter()
        uncached = self.structural_cache.get_structural_fields(
            G, force_recompute=True
        )
        cached = self.structural_cache.get_structural_fields(G)
        local_elapsed = time.perf_counter() - local_start
        stats_after = self.structural_cache.get_cache_stats()

        hit_delta = _counter_delta(stats_before, stats_after, "hits")
        miss_delta = _counter_delta(stats_before, stats_after, "misses")
        equivalent = _structural_entries_equal(uncached, cached)
        verified = bool(equivalent and hit_delta is not None and hit_delta >= 1)

        return OptimizationResult(
            strategy_used=OptimizationStrategy.STRUCTURAL_MEMO,
            execution_time=local_elapsed,
            speedup_factor=1.0,
            cache_hits=hit_delta or 0,
            cache_misses=miss_delta or 0,
            memory_used_mb=0.0,
            accuracy_preserved=verified,
            details={
                "operation": operation,
                "cache_totals": dict(stats_after),
                "performance_measurements": _performance_measurements(),
                "cache_measurements": {
                    "hits": hit_delta,
                    "misses": miss_delta,
                    "scope": "operation_delta",
                },
                "accuracy_verification": {
                    "basis": "uncached_and_cached_structural_entry_equivalence",
                    "cache_hit_observed": hit_delta is not None and hit_delta >= 1,
                    "entries_equal": equivalent,
                    "passed": verified,
                    "baseline_comparison_performed": True,
                },
            },
        )

    def _execute_hybrid_optimization(
        self, G: Any, operation: str, **kwargs
    ) -> OptimizationResult:
        """Run compatible components and require every verification to pass."""

        components: list[OptimizationResult] = []
        if self.structural_cache:
            components.append(
                self._execute_structural_optimization(G, operation, **kwargs)
            )

        structural_failed = bool(
            components and not components[-1].accuracy_preserved
        )
        if (
            not structural_failed
            and operation == FFT_EPI_DIFFUSION_OPERATION
            and self.fft_engine
            and len(G.nodes()) > 20
        ):
            components.append(self._execute_fft_optimization(G, operation, **kwargs))

        if not components:
            components.append(self._execute_nodal_optimization(G, operation, **kwargs))

        passed = all(component.accuracy_preserved for component in components)
        primary = components[-1]
        combined_strategies = [
            component.strategy_used.value for component in components
        ]
        cache_hits = sum(component.cache_hits for component in components)
        cache_misses = sum(component.cache_misses for component in components)
        component_memories = []
        for component in components:
            measurements = component.details.get("performance_measurements")
            memory = (
                measurements.get("memory_used_mb")
                if isinstance(measurements, Mapping)
                else None
            )
            component_memories.append(memory)
        measured_memory = (
            float(sum(component_memories))
            if component_memories
            and all(
                isinstance(value, Real)
                and not isinstance(value, (bool, np.bool_))
                and math.isfinite(float(value))
                for value in component_memories
            )
            else None
        )

        details = dict(primary.details)
        details.update(
            {
                "combined_strategies": combined_strategies,
                "component_verifications": [
                    {
                        "strategy": component.strategy_used.value,
                        "verification": component.details.get(
                            "accuracy_verification"
                        ),
                        "error": component.details.get("error"),
                    }
                    for component in components
                ],
                "performance_measurements": _performance_measurements(
                    memory_used_mb=measured_memory
                ),
                "cache_measurements": {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "scope": "sum_of_component_operation_deltas",
                },
                "accuracy_verification": {
                    "basis": "all_hybrid_component_verifications",
                    "passed": passed,
                    "baseline_comparison_performed": any(
                        bool(
                            component.details.get(
                                "accuracy_verification", {}
                            ).get("baseline_comparison_performed")
                        )
                        for component in components
                    ),
                },
            }
        )
        if not passed and "error" not in details:
            details["error"] = "One or more hybrid component verifications failed"

        return OptimizationResult(
            strategy_used=OptimizationStrategy.HYBRID,
            execution_time=sum(component.execution_time for component in components),
            speedup_factor=1.0,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            memory_used_mb=measured_memory or 0.0,
            accuracy_preserved=passed,
            details=details,
        )

    def _update_performance_history(self, result: OptimizationResult) -> None:
        """Retain results and learn only verified, measured speedups."""

        self.optimization_history.append(result)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

        measured_speedup = _measured_speedup(result)
        if not result.accuracy_preserved or measured_speedup is None:
            return

        strategy = result.strategy_used
        self.strategy_performance.setdefault(strategy, []).append(measured_speedup)
        if len(self.strategy_performance[strategy]) > 20:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][
                -20:
            ]

    def optimize_graph_operation(
        self,
        G: Any,
        operation: str = "general",
        strategy: OptimizationStrategy | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Profile, select, and execute one graph-computation strategy.
        """
        if strategy is None:
            strategy = OptimizationStrategy.AUTO

        profile = self.analyze_optimization_profile(G, operation)
        # An explicit FFT request is a physical-model request, so preserve it and
        # let the prewrite dispatcher guard accept or reject it deterministically.
        selected_strategy = (
            strategy
            if strategy == OptimizationStrategy.SPECTRAL_FFT
            else self.select_optimal_strategy(profile, strategy)
        )

        return self.execute_optimization(G, operation, selected_strategy, **kwargs)

    def get_orchestrator_stats(self) -> dict[str, Any]:
        """Return operation counts and explicitly measured speedup summaries."""

        total_operations = len(self.optimization_history)
        if total_operations == 0:
            return {"status": "no_operations"}

        strategy_stats: dict[str, dict[str, Any]] = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_stats[strategy.value] = {
                    "avg_speedup": float(np.mean(performances)),
                    "max_speedup": float(np.max(performances)),
                    "measured_operations": len(performances),
                }

        recent_speedups = [
            measured
            for result in self.optimization_history[-10:]
            if (measured := _measured_speedup(result)) is not None
        ]
        return {
            "total_operations": total_operations,
            "verified_operations": sum(
                result.accuracy_preserved for result in self.optimization_history
            ),
            "strategy_performance": strategy_stats,
            "recent_avg_speedup": (
                float(np.mean(recent_speedups)) if recent_speedups else None
            ),
            "recent_measured_speedup_samples": len(recent_speedups),
            "engines_available": {
                "nodal_optimizer": self.nodal_optimizer is not None,
                "structural_cache": self.structural_cache is not None,
                "fft_engine": self.fft_engine is not None,
                "adelic_engine": self.adelic_engine is not None,
            },
            "cache_available": self.structural_cache is not None,
        }


# Global orchestrator instance
_global_orchestrator = None


def get_orchestrator() -> TNFROptimizationOrchestrator:
    """Get or create the global optimization orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = TNFROptimizationOrchestrator()
    return _global_orchestrator


def optimize_tnfr_operation(
    G: Any, operation: str = "general", **kwargs
) -> OptimizationResult:
    """Convenience function for TNFR optimization."""
    orchestrator = get_orchestrator()
    return orchestrator.optimize_graph_operation(G, operation, **kwargs)
