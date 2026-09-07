"""Explicit computation facade for canonical TNFR kernels.

Each route keeps its physical model and executor provenance visible. The facade
coordinates detached nodal proposals, spectral diagnostics, the structural
tetrad, declared temporal integration, and atomic operator words. It does not
claim that selecting a numerical library changes a route which never uses that
library, and it rejects unimplemented cross-scale coupling.
"""

import math
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Integral
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..alias import get_attr
from ..constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..errors import TNFRValueError
from ..types import real_scalar_epi
from ..mathematics.unified_numerical import np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import the canonical dependency-signature reader.
try:
    from ..utils.cache import _compute_dependency_hash

    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

# Import optimization engines
try:
    from .fft_engine import FFTDynamicsEngine
    from .nodal_optimizer import NodalEquationOptimizer

    HAS_OPTIMIZATION_ENGINES = True
except ImportError:
    HAS_OPTIMIZATION_ENGINES = False

# Import spectral analysis
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft

    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import physics fields
try:
    from ..physics.fields import (
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        estimate_coherence_length,
    )

    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False


class ComputationType(Enum):
    """Types of TNFR computations."""

    NODAL_EVOLUTION = "nodal_evolution"  # ∂EPI/∂t integration
    SPECTRAL_ANALYSIS = "spectral_analysis"  # GFT/IGFT operations
    FIELD_COMPUTATION = "field_computation"  # Φ_s, |∇φ|, K_φ, ξ_C
    TEMPORAL_INTEGRATION = "temporal_integration"  # Multi-step evolution
    OPERATOR_APPLICATION = "operator_application"  # Structural operators
    # Compatibility value. Execution is rejected until a canonical coupling
    # map through THOL/REMESH is specified.
    CROSS_SCALE_COUPLING = "cross_scale_coupling"


@dataclass
class UnifiedComputationRequest:
    """Request for unified computation."""

    computation_type: ComputationType
    graph: Any
    parameters: dict[str, Any] = field(default_factory=dict)
    preferred_backend: str | None = None
    enable_cache: bool = True
    return_trajectory: bool = False
    # Compatibility request metadata; current explicit routes do not alter
    # physics or executor based on this value.
    optimization_level: int = 2


@dataclass
class UnifiedComputationResult:
    """Result of unified computation."""

    computation_type: ComputationType
    results: dict[str, Any]
    backend_used: str
    execution_time: float
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_strategy: str = "none"
    memory_used_mb: float | None = None
    accuracy_metrics: dict[str, float] = field(default_factory=dict)


class TNFRUnifiedBackend:
    """
    Coordinate explicitly implemented TNFR computation routes.

    ``preferred_backend`` is retained as a compatibility hint. Current routes
    call named TNFR kernels directly and report those executors instead of a
    numerical backend which they did not invoke.
    """

    def __init__(self, default_backend: str = "numpy", cache_size_mb: float = 256.0):
        if not isinstance(default_backend, str):
            raise TypeError("default_backend must be a string compatibility hint")
        self.default_backend = default_backend
        if isinstance(cache_size_mb, (bool, np.bool_)):
            raise ValueError("cache_size_mb must be a positive finite scalar")
        try:
            self.cache_size_mb = float(cache_size_mb)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "cache_size_mb must be a positive finite scalar"
            ) from exc
        if not math.isfinite(self.cache_size_mb) or self.cache_size_mb <= 0.0:
            raise ValueError("cache_size_mb must be a positive finite scalar")
        self._cache_budget_bytes = int(self.cache_size_mb * 1024 * 1024)

        # Only engines actually used by the explicit routes are instantiated.
        self._nodal_optimizer = (
            NodalEquationOptimizer() if HAS_OPTIMIZATION_ENGINES else None
        )
        self._fft_engine = FFTDynamicsEngine() if HAS_OPTIMIZATION_ENGINES else None

        # Cross-computation cache (shared between all engines)
        self._spectral_cache: dict[str, Any] = {}
        self._field_cache: dict[str, Any] = {}
        self._cache_order: OrderedDict[tuple[str, str], int] = OrderedDict()
        self._cache_size_bytes = 0
        self._cache_hits_total = 0
        self._cache_misses_total = 0

        # Performance tracking
        self._computation_history = []

    def select_optimal_backend(self, request: UnifiedComputationRequest) -> str:
        """Return the explicit TNFR runtime facade used by every current route.

        Mathematical backend objects are inventoried for diagnostics, but none
        of the methods in this class dispatches through them. Reporting JAX or
        Torch here would therefore be false provenance.
        """

        if request.preferred_backend is not None and not isinstance(
            request.preferred_backend, str
        ):
            raise TNFRValueError("preferred_backend must be a string or None")
        return "tnfr-runtime"

    def execute_computation(
        self, request: UnifiedComputationRequest
    ) -> UnifiedComputationResult:
        """Execute one declared computation without semantic fallback.

        Invalid state, model, or operator requests propagate their original
        error. A failed optimized path must never be relabeled as a successful
        computation under a different pressure model.
        """

        if not isinstance(request, UnifiedComputationRequest):
            raise TypeError("request must be a UnifiedComputationRequest")
        if request.graph is None:
            raise TNFRValueError("unified computation requires a graph")

        if (
            isinstance(request.optimization_level, bool)
            or not isinstance(request.optimization_level, Integral)
            or not 0 <= request.optimization_level <= 2
        ):
            raise TNFRValueError("optimization_level must be an integer in [0, 2]")

        start_time = time.perf_counter()
        cache_hits_before = self._cache_hits_total
        cache_misses_before = self._cache_misses_total
        backend_name = self.select_optimal_backend(request)
        if request.computation_type == ComputationType.NODAL_EVOLUTION:
            results = self._execute_nodal_evolution(request, backend_name)
        elif request.computation_type == ComputationType.SPECTRAL_ANALYSIS:
            results = self._execute_spectral_analysis(request, backend_name)
        elif request.computation_type == ComputationType.FIELD_COMPUTATION:
            results = self._execute_field_computation(request, backend_name)
        elif request.computation_type == ComputationType.TEMPORAL_INTEGRATION:
            results = self._execute_temporal_integration(request, backend_name)
        elif request.computation_type == ComputationType.OPERATOR_APPLICATION:
            results = self._execute_operator_application(request, backend_name)
        elif request.computation_type == ComputationType.CROSS_SCALE_COUPLING:
            raise TNFRValueError(
                "Cross-scale coupling has no implemented canonical state map; "
                "use an atomic THOL/REMESH operator word once its contract applies"
            )
        else:
            raise TNFRValueError(
                f"Unknown computation type: {request.computation_type}",
                context={
                    "computation_type": request.computation_type,
                    "available": [kind.name for kind in ComputationType],
                },
                suggestion="Use a valid ComputationType enum value.",
            )

        result = UnifiedComputationResult(
            computation_type=request.computation_type,
            results=results,
            backend_used=str(results.get("backend", backend_name)),
            execution_time=time.perf_counter() - start_time,
            cache_hits=self._cache_hits_total - cache_hits_before,
            cache_misses=self._cache_misses_total - cache_misses_before,
            optimization_strategy="none",
        )
        self._computation_history.append(result)
        return result

    def _execute_nodal_evolution(
        self, request: UnifiedComputationRequest, backend: str
    ) -> dict[str, Any]:
        """Return one detached nodal proposal under an explicit pressure model.

        ``stored_delta_nfr`` integrates the live canonical pressure attribute.
        ``epi_diffusion`` is the narrower graph-diffusion realization
        ``DeltaNFR_epi = -L_rw EPI`` and may use the nodal optimizer. The two
        models are never substituted for one another based on an optimization
        level.
        """
        graph = request.graph
        raw_dt = request.parameters.get("dt", 0.01)
        if isinstance(raw_dt, (bool, np.bool_)):
            raise TNFRValueError("dt must be a positive finite real scalar")
        try:
            dt = float(raw_dt)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TNFRValueError("dt must be a positive finite real scalar") from exc
        if not np.isfinite(dt) or dt <= 0.0:
            raise TNFRValueError("dt must be a positive finite real scalar")

        pressure_model = request.parameters.get(
            "pressure_model", "stored_delta_nfr"
        )
        if pressure_model == "epi_diffusion":
            if self._nodal_optimizer is None:
                raise TNFRValueError("EPI diffusion optimizer is unavailable")
            states = self._nodal_optimizer.compute_vectorized_nodal_evolution(
                graph, dt
            )
        elif pressure_model == "stored_delta_nfr":
            states = {}
            for node in graph.nodes():
                node_data = graph.nodes[node]
                epi = real_scalar_epi(
                    get_attr(
                        node_data,
                        ALIAS_EPI,
                        0.0,
                        strict=True,
                        conv=lambda value: value,
                    )
                )
                if epi is None:
                    raise TNFRValueError(
                        f"node {node!r} EPI has no scalar nodal embedding"
                    )
                nu_f = float(get_attr(node_data, ALIAS_VF, 1.0, strict=True))
                dnfr = float(get_attr(node_data, ALIAS_DNFR, 0.0, strict=True))
                phase = float(get_attr(node_data, ALIAS_THETA, 0.0, strict=True))
                values = np.asarray((epi, nu_f, dnfr, phase), dtype=float)
                if not np.all(np.isfinite(values)) or nu_f < 0.0:
                    raise TNFRValueError(
                        f"node {node!r} nodal state must be finite with nu_f >= 0"
                    )
                states[node] = (float(epi + dt * nu_f * dnfr), phase)
        else:
            raise TNFRValueError(
                "pressure_model must be 'stored_delta_nfr' or 'epi_diffusion'"
            )

        return {
            "nodal_states": states,
            "backend": (
                "tnfr-nodal-optimizer"
                if pressure_model == "epi_diffusion"
                else "tnfr-nodal-proposal"
            ),
            "pressure_model": pressure_model,
            "detached": True,
        }

    def _cache_signature(self, graph: Any, dependencies: set[str]) -> str:
        """Return an exact in-process signature with explicit node order."""

        order = tuple(
            (type(node).__module__, type(node).__qualname__, repr(node))
            for node in graph.nodes()
        )
        if _CACHE_AVAILABLE:
            dependency_hash = _compute_dependency_hash(graph, dependencies)
        else:
            dependency_hash = repr(
                (
                    tuple(graph.nodes(data=True)),
                    tuple(graph.edges(data=True)),
                )
            )
        return repr((dependency_hash, order))

    @staticmethod
    def _readonly_array(values: Any) -> np.ndarray:
        result = np.array(values, copy=True)
        result.setflags(write=False)
        return result

    @staticmethod
    def _estimate_cache_bytes(value: Any, seen: set[int] | None = None) -> int:
        """Estimate owned cache memory, counting nested arrays by byte size."""

        if seen is None:
            seen = set()
        object_id = id(value)
        if object_id in seen:
            return 0
        seen.add(object_id)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        size = sys.getsizeof(value)
        if isinstance(value, Mapping):
            return size + sum(
                TNFRUnifiedBackend._estimate_cache_bytes(key, seen)
                + TNFRUnifiedBackend._estimate_cache_bytes(item, seen)
                for key, item in value.items()
            )
        if isinstance(value, (tuple, list, set, frozenset)):
            return size + sum(
                TNFRUnifiedBackend._estimate_cache_bytes(item, seen) for item in value
            )
        return size

    def _cache_get(self, kind: str, cache: dict[str, Any], key: str) -> Any | None:
        value = cache.get(key)
        if value is None:
            self._cache_misses_total += 1
            return None
        self._cache_hits_total += 1
        token = (kind, key)
        if token in self._cache_order:
            self._cache_order.move_to_end(token)
        return value

    def _cache_store(
        self, kind: str, cache: dict[str, Any], key: str, value: Any
    ) -> None:
        """Store under one shared approximate memory budget with LRU eviction."""

        token = (kind, key)
        previous_size = self._cache_order.pop(token, 0)
        self._cache_size_bytes -= previous_size
        cache[key] = value
        size = self._estimate_cache_bytes(value)
        self._cache_order[token] = size
        self._cache_size_bytes += size
        caches = {
            "spectral": self._spectral_cache,
            "field": self._field_cache,
        }
        while self._cache_size_bytes > self._cache_budget_bytes and self._cache_order:
            (old_kind, old_key), old_size = self._cache_order.popitem(last=False)
            caches[old_kind].pop(old_key, None)
            self._cache_size_bytes -= old_size

    def _execute_spectral_analysis(
        self, request: UnifiedComputationRequest, backend: str
    ) -> dict[str, Any]:
        """Compute a graph spectrum and GFT under an exact topology key."""

        if not HAS_SPECTRAL:
            raise TNFRValueError("spectral analysis is unavailable")
        graph = request.graph
        cache_key = self._cache_signature(
            graph, {"graph_topology", "precision_mode"}
        )
        cached = (
            self._cache_get("spectral", self._spectral_cache, cache_key)
            if request.enable_cache
            else None
        )
        if cached is None:
            eigenvalues, eigenvectors = get_laplacian_spectrum(graph)
            cached = (
                self._readonly_array(eigenvalues),
                self._readonly_array(eigenvectors),
            )
            if request.enable_cache:
                self._cache_store("spectral", self._spectral_cache, cache_key, cached)
        eigenvalues, eigenvectors = cached

        signal_values: list[float] = []
        for node in graph.nodes():
            raw = get_attr(
                graph.nodes[node],
                ALIAS_EPI,
                0.0,
                strict=True,
                conv=lambda value: value,
            )
            scalar = real_scalar_epi(raw)
            if scalar is None or not np.isfinite(scalar):
                raise TNFRValueError(
                    f"node {node!r} EPI has no finite scalar spectral embedding"
                )
            signal_values.append(float(scalar))
        signal = np.asarray(signal_values, dtype=float)
        spectral_coefficients = np.asarray(gft(signal, eigenvectors))

        result: dict[str, Any] = {
            "eigenvalues": np.array(eigenvalues, copy=True),
            "eigenvectors": np.array(eigenvectors, copy=True),
            "signal": np.array(signal, copy=True),
            "spectral_coefficients": np.array(
                spectral_coefficients, copy=True
            ),
            "backend": "tnfr-spectral-api",
            "detached": True,
        }
        if "filter_cutoff" in request.parameters:
            raw_cutoff = request.parameters["filter_cutoff"]
            if isinstance(raw_cutoff, (bool, np.bool_)):
                raise TNFRValueError("filter_cutoff must be a finite real scalar")
            try:
                cutoff = float(raw_cutoff)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TNFRValueError(
                    "filter_cutoff must be a finite real scalar"
                ) from exc
            if not np.isfinite(cutoff):
                raise TNFRValueError("filter_cutoff must be a finite real scalar")
            filtered_coefficients = np.array(
                spectral_coefficients, copy=True
            )
            filtered_coefficients[np.real(eigenvalues) > cutoff] = 0
            result["filtered_signal"] = np.asarray(
                igft(filtered_coefficients, eigenvectors)
            )
        return result

    def _execute_field_computation(
        self, request: UnifiedComputationRequest, backend: str
    ) -> dict[str, Any]:
        """Compute read-only fields under an exact structural-state key."""

        if not HAS_PHYSICS:
            raise TNFRValueError("structural field computation is unavailable")
        graph = request.graph
        cache_key = self._cache_signature(
            graph,
            {
                "graph_topology",
                "node_epi",
                "node_vf",
                "node_phase",
                "node_dnfr",
                "node_depi",
                "precision_mode",
            },
        )
        cached_fields = (
            self._cache_get("field", self._field_cache, cache_key)
            if request.enable_cache
            else None
        )
        if cached_fields is not None:
            return deepcopy(cached_fields)

        results = {
            "phi_s": compute_structural_potential(graph),
            "phase_gradient": compute_phase_gradient(graph),
            "phase_curvature": compute_phase_curvature(graph),
            "coherence_length": estimate_coherence_length(graph),
            "backend": "tnfr-field-readers",
            "detached": True,
        }
        if request.enable_cache:
            self._cache_store(
                "field", self._field_cache, cache_key, deepcopy(results)
            )
        return deepcopy(results)

    def _execute_temporal_integration(
        self, request: UnifiedComputationRequest, backend: str
    ) -> dict[str, Any]:
        """Commit a multi-step trajectory under one explicit pressure model."""
        graph = request.graph
        num_steps = request.parameters.get("num_steps", 10)
        dt = request.parameters.get("dt", 0.01)
        pressure_model = request.parameters.get(
            "pressure_model", "stored_delta_nfr"
        )

        # The graph-spectral engine implements only the EPI diffusion channel.
        # Graph size and optimization level must never change pressure semantics.
        if pressure_model == "epi_diffusion":
            if self._fft_engine is None:
                raise TNFRValueError("EPI diffusion engine is unavailable")
            result = self._fft_engine.run_fft_simulation(
                graph, num_steps, dt, request.return_trajectory
            )
            result["pressure_model"] = pressure_model
            result["backend"] = "tnfr-fft-diffusion"
            return result
        if pressure_model != "stored_delta_nfr":
            raise TNFRValueError(
                "pressure_model must be 'stored_delta_nfr' or 'epi_diffusion'"
            )

        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TNFRValueError("num_steps must be a nonnegative integer")
        if num_steps < 0:
            raise TNFRValueError("num_steps must be a nonnegative integer")
        if isinstance(dt, (bool, np.bool_)):
            raise TNFRValueError("dt must be a positive finite real scalar")
        try:
            dt_value = float(dt)
        except (OverflowError, TypeError, ValueError) as exc:
            raise TNFRValueError("dt must be a positive finite real scalar") from exc
        if not np.isfinite(dt_value) or (num_steps and dt_value <= 0.0):
            raise TNFRValueError("dt must be a positive finite real scalar")

        from .integrators import update_epi_via_nodal_equation

        trajectory = []
        for step in range(num_steps):
            update_epi_via_nodal_equation(
                graph,
                dt=dt_value,
                method=request.parameters.get("method", "euler"),
            )
            if request.return_trajectory:
                states = {
                    node: (
                        float(get_attr(graph.nodes[node], ALIAS_EPI, 0.0)),
                        float(get_attr(graph.nodes[node], ALIAS_THETA, 0.0)),
                    )
                    for node in graph.nodes
                }
                trajectory.append(
                    {
                        "step": step,
                        "time": float(graph.graph.get("_t", 0.0)),
                        "nodal_states": states,
                    }
                )

        return {
            "final_time": float(graph.graph.get("_t", 0.0)),
            "trajectory": trajectory if request.return_trajectory else None,
            "backend": "tnfr-nodal-integrator",
            "pressure_model": pressure_model,
        }

    def _execute_operator_application(
        self, request: UnifiedComputationRequest, backend: str
    ) -> dict[str, Any]:
        """Apply one complete grammar-valid word as an atomic graph transaction."""

        graph = request.graph
        raw_sequence = request.parameters.get("operators")
        if (
            isinstance(raw_sequence, (str, bytes))
            or not isinstance(raw_sequence, Sequence)
            or not raw_sequence
        ):
            raise TNFRValueError("operators must be a non-empty sequence")
        if "node" not in request.parameters:
            raise TNFRValueError("operator application requires a target node")
        node = request.parameters["node"]
        if node not in graph:
            raise TNFRValueError(f"target node {node!r} is not present in the graph")

        from ..operators.definitions_base import Operator
        from ..operators.grammar_execution import ValidatedSequence
        from ..operators.grammar_types import glyph_function_name
        from ..operators.network_stage import GraphTransactionSnapshot
        from ..operators.registry import get_operator_class

        operators: list[Operator] = []
        for index, candidate in enumerate(raw_sequence):
            if isinstance(candidate, Operator):
                operators.append(candidate)
                continue
            try:
                canonical_name = glyph_function_name(candidate)
                operators.append(get_operator_class(canonical_name)())
            except (KeyError, TypeError, ValueError) as exc:
                raise TNFRValueError(
                    f"operators[{index}] is not a canonical TNFR operator"
                ) from exc

        raw_kwargs = request.parameters.get("operator_kwargs")
        if raw_kwargs is None:
            per_operator_kwargs: list[dict[str, Any]] = [
                {} for _ in operators
            ]
        elif (
            isinstance(raw_kwargs, (str, bytes))
            or not isinstance(raw_kwargs, Sequence)
            or len(raw_kwargs) != len(operators)
        ):
            raise TNFRValueError(
                "operator_kwargs must align one-to-one with operators"
            )
        else:
            per_operator_kwargs = []
            for index, item in enumerate(raw_kwargs):
                if not isinstance(item, Mapping):
                    raise TNFRValueError(
                        f"operator_kwargs[{index}] must be a mapping"
                    )
                if "sequence_context" in item:
                    raise TNFRValueError(
                        "sequence_context is owned by the unified backend"
                    )
                per_operator_kwargs.append(dict(item))

        initial_epi = real_scalar_epi(
            get_attr(
                graph.nodes[node],
                ALIAS_EPI,
                0.0,
                strict=True,
                conv=lambda value: value,
            )
        )
        if initial_epi is None:
            raise TNFRValueError(
                f"node {node!r} EPI has no scalar nodal embedding"
            )
        word = ValidatedSequence(
            operators, context={"initial_epi_nonzero": initial_epi != 0.0}
        )
        transaction = GraphTransactionSnapshot(graph)
        try:
            for index, (operator, kwargs) in enumerate(
                zip(operators, per_operator_kwargs)
            ):
                operator(
                    graph,
                    node,
                    sequence_context=word.step(index),
                    **kwargs,
                )
        except BaseException:
            monitor = graph.graph.get("integrity_monitor")
            discard_pending = getattr(
                monitor, "discard_pending_operator", None
            )
            if callable(discard_pending):
                try:
                    discard_pending()
                except Exception:
                    pass
            transaction.restore(graph)
            raise

        return {
            "applied_operators": [operator.name for operator in operators],
            "node": node,
            "backend": "tnfr-operator-runtime",
            "mutated": True,
            "atomic": True,
        }

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics across all computations."""
        if not self._computation_history:
            return {"total_computations": 0}

        total_time = sum(r.execution_time for r in self._computation_history)
        avg_time = total_time / len(self._computation_history)

        # Backend usage statistics
        backend_usage = {}
        for result in self._computation_history:
            backend = result.backend_used
            backend_usage[backend] = backend_usage.get(backend, 0) + 1

        # Computation type statistics
        type_usage = {}
        for result in self._computation_history:
            comp_type = result.computation_type.value
            type_usage[comp_type] = type_usage.get(comp_type, 0) + 1

        return {
            "total_computations": len(self._computation_history),
            "total_time": total_time,
            "average_time": avg_time,
            "backend_usage": backend_usage,
            "computation_type_usage": type_usage,
            "cache_availability": _CACHE_AVAILABLE,
            "optimization_engines_available": HAS_OPTIMIZATION_ENGINES,
            "mathematical_backend_dispatch": False,
            "cache_hits": self._cache_hits_total,
            "cache_misses": self._cache_misses_total,
            "cache_size_bytes": self._cache_size_bytes,
            "cache_budget_bytes": self._cache_budget_bytes,
        }

    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self._spectral_cache.clear()
        self._field_cache.clear()
        self._cache_order.clear()
        self._cache_size_bytes = 0
        self._cache_hits_total = 0
        self._cache_misses_total = 0

        if self._nodal_optimizer:
            self._nodal_optimizer.clear_optimization_cache()


# Factory functions
def create_unified_backend(**kwargs) -> TNFRUnifiedBackend:
    """Create unified computational backend."""
    return TNFRUnifiedBackend(**kwargs)


def execute_unified_computation(
    computation_type: ComputationType,
    graph: Any,
    backend: TNFRUnifiedBackend | None = None,
    **kwargs,
) -> UnifiedComputationResult:
    """Convenience function for unified computation."""
    if backend is None:
        backend = create_unified_backend()

    request = UnifiedComputationRequest(
        computation_type=computation_type, graph=graph, parameters=kwargs
    )

    return backend.execute_computation(request)
