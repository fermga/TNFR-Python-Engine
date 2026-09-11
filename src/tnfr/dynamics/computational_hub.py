"""Explicit dispatch hub for TNFR computation adapters.

The hub routes a declared operation only to an adapter that implements that
operation. It records observed execution time and propagates adapter failures;
it does not infer mathematical equivalence, speedup, or accuracy from the nodal
equation alone. Cache services remain auxiliary stores rather than executors.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue
from typing import Any, Callable

from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from .optimization_orchestrator import (
    FFT_EPI_DIFFUSION_OPERATION,
    TNFROptimizationOrchestrator,
    validate_fft_epi_diffusion_dispatch,
)

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import engines independently so one optional subsystem cannot disable the rest.
try:
    from .adelic import AdelicDynamics

    HAS_ADELIC_ENGINE = True
except ImportError:
    HAS_ADELIC_ENGINE = False
    AdelicDynamics = None  # type: ignore[assignment]

try:
    from .advanced_fft_arithmetic import TNFRAdvancedFFTEngine

    HAS_ADVANCED_FFT_ENGINE = True
except ImportError:
    HAS_ADVANCED_FFT_ENGINE = False
    TNFRAdvancedFFTEngine = None  # type: ignore[assignment]

try:
    from .fft_engine import FFTDynamicsEngine

    HAS_FFT_ENGINE = True
except ImportError:
    HAS_FFT_ENGINE = False
    FFTDynamicsEngine = None  # type: ignore[assignment]

try:
    from .multi_modal_cache import TNFRUnifiedMultiModalCache

    HAS_MULTI_MODAL_CACHE = True
except ImportError:
    HAS_MULTI_MODAL_CACHE = False
    TNFRUnifiedMultiModalCache = None  # type: ignore[assignment]

try:
    from .nodal_optimizer import NodalEquationOptimizer

    HAS_NODAL_OPTIMIZER = True
except ImportError:
    HAS_NODAL_OPTIMIZER = False
    NodalEquationOptimizer = None  # type: ignore[assignment]

try:
    from .structural_cache import StructuralCoherenceCache

    HAS_STRUCTURAL_CACHE = True
except ImportError:
    HAS_STRUCTURAL_CACHE = False
    StructuralCoherenceCache = None  # type: ignore[assignment]

try:
    from .unified_backend import (
        ComputationType,
        TNFRUnifiedBackend,
        UnifiedComputationRequest,
    )

    HAS_UNIFIED_BACKEND = True
except ImportError:
    HAS_UNIFIED_BACKEND = False
    ComputationType = None  # type: ignore[assignment,misc]
    TNFRUnifiedBackend = None  # type: ignore[assignment]
    UnifiedComputationRequest = None  # type: ignore[assignment]

HAS_OPTIMIZATION_ORCHESTRATOR = True
HAS_ALL_ENGINES = all(
    (
        HAS_ADELIC_ENGINE,
        HAS_ADVANCED_FFT_ENGINE,
        HAS_FFT_ENGINE,
        HAS_MULTI_MODAL_CACHE,
        HAS_NODAL_OPTIMIZER,
        HAS_STRUCTURAL_CACHE,
        HAS_UNIFIED_BACKEND,
        HAS_OPTIMIZATION_ORCHESTRATOR,
    )
)

# Import mathematical backends
try:
    from ..mathematics.backend import available_backends, get_backend

    HAS_MATH_BACKENDS = True
except ImportError:
    HAS_MATH_BACKENDS = False
    get_backend = None
    available_backends = None


class ComputationPriority(Enum):
    """Priority levels for computation requests."""

    CRITICAL = 1  # Real-time operator applications
    HIGH = 2  # Interactive computations
    NORMAL = 3  # Standard analysis
    LOW = 4  # Background optimization
    BATCH = 5  # Large batch processing


class EngineType(Enum):
    """Available computational engines."""

    AUTO = "auto"
    UNIFIED_BACKEND = "unified_backend"
    OPTIMIZATION_ORCHESTRATOR = "orchestrator"
    ADVANCED_FFT = "advanced_fft"
    NODAL_OPTIMIZER = "nodal_optimizer"
    FFT_ENGINE = "fft_engine"
    STRUCTURAL_CACHE = "structural_cache"
    ADELIC_DYNAMICS = "adelic_dynamics"
    MULTI_MODAL_CACHE = "multi_modal_cache"


ENGINE_IMPORT_AVAILABILITY = {
    EngineType.UNIFIED_BACKEND: HAS_UNIFIED_BACKEND,
    EngineType.OPTIMIZATION_ORCHESTRATOR: HAS_OPTIMIZATION_ORCHESTRATOR,
    EngineType.ADVANCED_FFT: HAS_ADVANCED_FFT_ENGINE,
    EngineType.NODAL_OPTIMIZER: HAS_NODAL_OPTIMIZER,
    EngineType.FFT_ENGINE: HAS_FFT_ENGINE,
    EngineType.STRUCTURAL_CACHE: HAS_STRUCTURAL_CACHE,
    EngineType.ADELIC_DYNAMICS: HAS_ADELIC_ENGINE,
    EngineType.MULTI_MODAL_CACHE: HAS_MULTI_MODAL_CACHE,
}

UNIFIED_OPERATION_TO_COMPUTATION_TYPE = (
    {
        "general_computation": ComputationType.NODAL_EVOLUTION,
        "nodal_evolution": ComputationType.NODAL_EVOLUTION,
        "spectral_analysis": ComputationType.SPECTRAL_ANALYSIS,
        "field_computation": ComputationType.FIELD_COMPUTATION,
        "structural_fields": ComputationType.FIELD_COMPUTATION,
        "temporal_integration": ComputationType.TEMPORAL_INTEGRATION,
        "temporal_evolution": ComputationType.TEMPORAL_INTEGRATION,
        "multi_step": ComputationType.TEMPORAL_INTEGRATION,
        FFT_EPI_DIFFUSION_OPERATION: ComputationType.TEMPORAL_INTEGRATION,
        "operator_application": ComputationType.OPERATOR_APPLICATION,
        "operator_sequence": ComputationType.OPERATOR_APPLICATION,
        "cross_scale_coupling": ComputationType.CROSS_SCALE_COUPLING,
    }
    if HAS_UNIFIED_BACKEND
    else {}
)


@dataclass
class ComputationRequest:
    """Unified computation request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    engine_type: EngineType = EngineType.UNIFIED_BACKEND
    operation: str = "general_computation"
    graph: Any | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: ComputationPriority = ComputationPriority.NORMAL
    callback: Callable | None = None
    dependencies: set[str] = field(default_factory=set)
    timeout_seconds: float = 300.0
    enable_cache: bool = True
    require_accuracy: bool = True


@dataclass
class ComputationResult:
    """Unified computation result."""

    request_id: str
    engine_used: EngineType
    operation: str
    success: bool
    result_data: Any = None
    execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_used_mb: float = 0.0
    backend_used: str = "numpy"
    accuracy_metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemResources:
    """System resource status."""

    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    cpu_count: int = 1
    gpu_available: bool = False
    active_computations: int = 0
    cache_utilization: float = 0.0
    load_average: float = 0.0


class TNFRComputationalHub:
    """
    Dispatch declared computations across independently available adapters.

    Structural and multimodal caches are imported capabilities, not execution
    engines, so their compatibility enum values are never registered here.
    """

    def __init__(
        self,
        max_workers: int = 4,
        memory_budget_mb: float = 1024.0,
        enable_gpu: bool = True,
        cache_size_mb: float = 512.0,
    ):
        self.max_workers = max_workers
        self.memory_budget_mb = memory_budget_mb
        self.enable_gpu = enable_gpu
        self.cache_size_mb = cache_size_mb

        # Initialize each available engine independently.
        self._engines: dict[EngineType, Any] = {}
        self._engine_initialization_errors: dict[EngineType, str] = {}
        if HAS_UNIFIED_BACKEND:
            self._register_engine(EngineType.UNIFIED_BACKEND, TNFRUnifiedBackend)
        self._register_engine(
            EngineType.OPTIMIZATION_ORCHESTRATOR,
            TNFROptimizationOrchestrator,
        )
        if HAS_ADVANCED_FFT_ENGINE:
            self._register_engine(EngineType.ADVANCED_FFT, TNFRAdvancedFFTEngine)
        if HAS_NODAL_OPTIMIZER:
            self._register_engine(EngineType.NODAL_OPTIMIZER, NodalEquationOptimizer)
        if HAS_FFT_ENGINE:
            self._register_engine(EngineType.FFT_ENGINE, FFTDynamicsEngine)
        if HAS_ADELIC_ENGINE:
            self._register_engine(EngineType.ADELIC_DYNAMICS, AdelicDynamics)

        # Computation coordination
        self._request_queue = PriorityQueue()
        self._result_cache = {}
        self._active_requests = {}

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_thread = None
        self._shutdown = False

        # Resource management
        self._resource_monitor = SystemResources()
        self._load_balancer = {}

        # Performance tracking
        self._performance_history = []
        self._engine_performance = {engine: [] for engine in EngineType}

        # Cross-engine shared memory
        self._shared_memory_pool = {}
        self._memory_locks = {}

        # Start background processing
        self._start_queue_processor()

    def _register_engine(
        self, engine_type: EngineType, factory: Callable[[], Any]
    ) -> None:
        """Register one engine without suppressing independent engines."""
        try:
            self._engines[engine_type] = factory()
        except Exception as exc:
            self._engine_initialization_errors[engine_type] = str(exc)

    def _start_queue_processor(self) -> None:
        """Start background queue processing thread."""

        def process_queue():
            while not self._shutdown:
                try:
                    if not self._request_queue.empty():
                        priority, timestamp, request = self._request_queue.get(
                            timeout=1.0
                        )
                        self._process_request_async(request)
                except Exception:
                    continue  # Keep processing

        self._queue_thread = threading.Thread(target=process_queue, daemon=True)
        self._queue_thread.start()

    def submit_computation(self, request: ComputationRequest) -> str:
        """
        Submit computation request to hub.

        Returns request ID for tracking.
        """
        # Validate request
        if not self._validate_request(request):
            raise TNFRValueError(
                f"Invalid computation request: {request}",
                context={"request": str(request)},
                suggestion="Ensure request has valid operation and parameters.",
            )

        # Add to queue with priority
        timestamp = time.time()
        self._request_queue.put((request.priority.value, timestamp, request))
        self._active_requests[request.request_id] = request

        return request.request_id

    def get_result(
        self, request_id: str, timeout: float | None = None
    ) -> ComputationResult | None:
        """Get computation result by request ID."""
        start_time = time.time()

        while True:
            # Check if result is ready
            if request_id in self._result_cache:
                result = self._result_cache.pop(request_id)
                self._active_requests.pop(request_id, None)
                return result

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None

            # Brief sleep to avoid busy waiting
            time.sleep(0.01)

    def execute_computation_sync(
        self, request: ComputationRequest
    ) -> ComputationResult:
        """Execute computation synchronously."""
        request_id = self.submit_computation(request)
        result = self.get_result(request_id, timeout=request.timeout_seconds)

        if result is None:
            return ComputationResult(
                request_id=request_id,
                engine_used=request.engine_type,
                operation=request.operation,
                success=False,
                error_message="Computation timed out",
            )

        return result

    def _process_request_async(self, request: ComputationRequest) -> None:
        """Process computation request asynchronously."""

        def process():
            try:
                result = self._execute_computation(request)
                self._result_cache[request.request_id] = result

                # Call callback if provided
                if request.callback:
                    request.callback(result)

            except Exception as e:
                error_result = ComputationResult(
                    request_id=request.request_id,
                    engine_used=request.engine_type,
                    operation=request.operation,
                    success=False,
                    error_message=str(e),
                )
                self._result_cache[request.request_id] = error_result

        # Submit to thread pool
        self._executor.submit(process)

    def _execute_computation(self, request: ComputationRequest) -> ComputationResult:
        """Execute single computation request."""
        start_time = time.time()

        if request.operation == "system_status":
            return ComputationResult(
                request_id=request.request_id,
                engine_used=request.engine_type,
                operation=request.operation,
                success=True,
                result_data=self.get_system_status(),
                execution_time=time.perf_counter() - start_time,
            )

        # Select optimal engine for this computation
        selected_engine = self._select_optimal_engine(request)

        # Route to appropriate engine
        try:
            if selected_engine == EngineType.UNIFIED_BACKEND:
                result_data = self._execute_unified_backend(request)

            elif selected_engine == EngineType.OPTIMIZATION_ORCHESTRATOR:
                result_data = self._execute_optimization_orchestrator(request)

            elif selected_engine == EngineType.ADVANCED_FFT:
                result_data = self._execute_advanced_fft(request)

            elif selected_engine == EngineType.NODAL_OPTIMIZER:
                result_data = self._execute_nodal_optimizer(request)

            elif selected_engine == EngineType.FFT_ENGINE:
                result_data = self._execute_fft_engine(request)

            elif selected_engine == EngineType.ADELIC_DYNAMICS:
                result_data = self._execute_adelic_dynamics(request)

            else:
                raise TNFRValueError(
                    f"Unknown engine type: {selected_engine}",
                    context={
                        "engine": selected_engine,
                        "available": [e.name for e in EngineType],
                    },
                    suggestion="Use a valid EngineType enum value.",
                )

        except Exception as e:
            return ComputationResult(
                request_id=request.request_id,
                engine_used=selected_engine,
                operation=request.operation,
                success=False,
                error_message=str(e),
            )

        execution_time = time.perf_counter() - start_time

        # Update performance tracking
        self._engine_performance[selected_engine].append(execution_time)

        return ComputationResult(
            request_id=request.request_id,
            engine_used=selected_engine,
            operation=request.operation,
            success=True,
            result_data=result_data,
            execution_time=execution_time,
        )

    def _select_optimal_engine(self, request: ComputationRequest) -> EngineType:
        """
        Select optimal engine based on request characteristics and system state.

        This selection emerges from mathematical analysis of the computation type.
        """
        # Preserve the historical explicit-engine behavior. Automatic routing is
        # opt-in through EngineType.AUTO so the public default remains stable.
        if request.engine_type != EngineType.AUTO:
            if request.engine_type in self._engines:
                return request.engine_type
            raise TNFRValueError(
                f"Requested engine is unavailable: {request.engine_type.value}"
            )

        # Intelligent selection based on operation and graph properties
        if request.graph is not None and HAS_NETWORKX:
            num_nodes = len(request.graph.nodes())

            # Large graphs benefit from specialized FFT engines
            if num_nodes > 100:
                if (
                    request.operation == "harmonic_analysis"
                    and EngineType.ADVANCED_FFT in self._engines
                ):
                    return EngineType.ADVANCED_FFT
                if (
                    request.operation == FFT_EPI_DIFFUSION_OPERATION
                    and EngineType.FFT_ENGINE in self._engines
                ):
                    return EngineType.FFT_ENGINE

            # Medium graphs good for nodal optimization
            elif 20 <= num_nodes <= 100:
                if (
                    request.operation == FFT_EPI_DIFFUSION_OPERATION
                    and EngineType.FFT_ENGINE in self._engines
                ):
                    return EngineType.FFT_ENGINE
                if (
                    request.operation == "nodal_evolution"
                    and EngineType.NODAL_OPTIMIZER in self._engines
                    and request.parameters.get("pressure_model") == "epi_diffusion"
                ):
                    return EngineType.NODAL_OPTIMIZER

        # The unified backend has an explicit operation table and is the safe
        # fallback for every operation it can represent.
        if (
            request.operation in UNIFIED_OPERATION_TO_COMPUTATION_TYPE
            and EngineType.UNIFIED_BACKEND in self._engines
        ):
            return EngineType.UNIFIED_BACKEND

        if request.operation in {"temporal", "arithmetic", "trace", "general"}:
            if EngineType.OPTIMIZATION_ORCHESTRATOR in self._engines:
                return EngineType.OPTIMIZATION_ORCHESTRATOR

        raise TNFRValueError(
            f"No available engine supports operation={request.operation!r}"
        )

    def _execute_unified_backend(self, request: ComputationRequest) -> Any:
        """Execute one operation through its declared unified computation type."""
        try:
            computation_type = UNIFIED_OPERATION_TO_COMPUTATION_TYPE[
                request.operation
            ]
        except KeyError as exc:
            supported = ", ".join(sorted(UNIFIED_OPERATION_TO_COMPUTATION_TYPE))
            raise TNFRValueError(
                f"Unified backend does not support operation={request.operation!r}; "
                f"supported operations: {supported}"
            ) from exc

        parameters = dict(request.parameters)
        if request.operation == FFT_EPI_DIFFUSION_OPERATION:
            parameters["pressure_model"] = validate_fft_epi_diffusion_dispatch(
                request.operation, parameters.get("pressure_model")
            )
        return_trajectory = parameters.get("return_trajectory", False)
        if not isinstance(return_trajectory, bool):
            raise TNFRValueError("return_trajectory must be boolean")

        engine = self._engines[EngineType.UNIFIED_BACKEND]

        unified_request = UnifiedComputationRequest(
            computation_type=computation_type,
            graph=request.graph,
            parameters=parameters,
            enable_cache=request.enable_cache,
            return_trajectory=return_trajectory,
        )

        result = engine.execute_computation(unified_request)
        return result.results

    def _execute_optimization_orchestrator(self, request: ComputationRequest) -> Any:
        """Execute using optimization orchestrator."""
        engine = self._engines[EngineType.OPTIMIZATION_ORCHESTRATOR]

        # Analyze and execute with optimal strategy
        profile = engine.analyze_optimization_profile(request.graph, request.operation)
        strategy = engine.select_optimal_strategy(profile)

        result = engine.execute_optimization(
            request.graph, request.operation, strategy, **request.parameters
        )
        verification = result.details.get("accuracy_verification")
        verification_failed = (
            isinstance(verification, dict) and verification.get("passed") is False
        )
        if (
            not result.accuracy_preserved
            or "error" in result.details
            or verification_failed
        ):
            reason = result.details.get("error", "accuracy verification failed")
            raise TNFRValueError(
                f"Optimization orchestrator failed: {reason}",
                context={
                    "strategy": result.strategy_used.value,
                    "accuracy_preserved": result.accuracy_preserved,
                    "details": result.details,
                },
            )
        measurements = result.details.get("performance_measurements")
        if not isinstance(measurements, dict):
            measurements = {
                "speedup_factor": result.speedup_factor,
                "memory_used_mb": result.memory_used_mb,
            }

        return {
            "strategy_used": result.strategy_used.value,
            "execution_time": result.execution_time,
            "speedup_factor": measurements.get("speedup_factor"),
            "memory_used_mb": measurements.get("memory_used_mb"),
            "accuracy_verification": result.details.get(
                "accuracy_verification",
                {
                    "basis": "legacy_strategy_contract",
                    "passed": result.accuracy_preserved,
                },
            ),
            "cache_performance": {
                "hits": result.cache_hits,
                "misses": result.cache_misses,
            },
            "details": result.details,
        }

    def _execute_advanced_fft(self, request: ComputationRequest) -> Any:
        """Execute using advanced FFT engine."""
        engine = self._engines[EngineType.ADVANCED_FFT]

        parameters = dict(request.parameters)
        operation = parameters.pop("spectral_operation", "harmonic_analysis")

        if operation == "harmonic_analysis":
            result = engine.harmonic_analysis(request.graph, **parameters)
        elif operation == "spectral_filtering":
            result = engine.spectral_filtering(request.graph, **parameters)
        elif operation == "coherence_analysis":
            graph2 = parameters.pop("graph2", None)
            if graph2 is None:
                raise TNFRValueError(
                    "Coherence analysis requires second graph",
                    context={"parameters": request.parameters.keys()},
                    suggestion="Provide 'graph2' in request parameters for coherence analysis.",
                )
            result = engine.cross_spectral_coherence(
                request.graph, graph2, **parameters
            )
        elif operation == "spectral_convolution":
            result = engine.spectral_convolution(request.graph, **parameters)
        else:
            raise TNFRValueError(
                f"Unsupported advanced FFT operation: {operation!r}",
                context={
                    "supported": [
                        "harmonic_analysis",
                        "spectral_filtering",
                        "coherence_analysis",
                        "spectral_convolution",
                    ]
                },
            )

        return result.output_data

    def _execute_nodal_optimizer(self, request: ComputationRequest) -> Any:
        """Return a detached EPI-diffusion proposal using the nodal optimizer."""
        if request.operation == FFT_EPI_DIFFUSION_OPERATION:
            pressure_model = validate_fft_epi_diffusion_dispatch(
                request.operation, request.parameters.get("pressure_model")
            )
        elif (
            request.operation == "nodal_evolution"
            and request.parameters.get("pressure_model") == "epi_diffusion"
        ):
            pressure_model = "epi_diffusion"
        else:
            raise TNFRValueError(
                "Nodal optimizer requires operation='epi_diffusion' or "
                "operation='nodal_evolution' with pressure_model='epi_diffusion'"
            )
        engine = self._engines[EngineType.NODAL_OPTIMIZER]

        dt = request.parameters.get("dt", 0.01)
        result = engine.compute_vectorized_nodal_evolution(request.graph, dt)

        return {
            "nodal_evolution": result,
            "pressure_model": pressure_model,
            "detached": True,
            "optimization_stats": engine.get_optimization_stats(),
        }

    def _execute_fft_engine(self, request: ComputationRequest) -> Any:
        """Execute the FFT engine under its explicit EPI-pressure contract."""
        pressure_model = validate_fft_epi_diffusion_dispatch(
            request.operation, request.parameters.get("pressure_model")
        )
        engine = self._engines[EngineType.FFT_ENGINE]

        num_steps = request.parameters.get("num_steps", 10)
        dt = request.parameters.get("dt", 0.01)

        result = engine.run_fft_simulation(request.graph, num_steps, dt)
        if isinstance(result, dict):
            return {
                **result,
                "operation": FFT_EPI_DIFFUSION_OPERATION,
                "pressure_model": pressure_model,
            }
        return {
            "result": result,
            "operation": FFT_EPI_DIFFUSION_OPERATION,
            "pressure_model": pressure_model,
        }

    def _execute_adelic_dynamics(self, request: ComputationRequest) -> Any:
        """Dispatch one explicit exploratory adelic operation."""

        engine = self._engines[EngineType.ADELIC_DYNAMICS]
        methods = {
            "geometric_trace": engine.compute_geometric_trace,
            "nodal_gradient": engine.compute_nodal_gradient,
            "resonance_search": engine.run_resonance_search,
            "adelic_structural_fields": engine.compute_structural_fields,
            "adelic_step": engine.step,
        }
        try:
            method = methods[request.operation]
        except KeyError as exc:
            raise TNFRValueError(
                f"Unsupported adelic operation: {request.operation!r}",
                context={"supported": sorted(methods)},
            ) from exc
        return method(**dict(request.parameters))

    def _validate_request(self, request: ComputationRequest) -> bool:
        """Validate computation request."""
        if not request.request_id:
            return False
        if (
            request.engine_type != EngineType.AUTO
            and request.engine_type not in self._engines
        ):
            return False
        if request.graph is None and request.operation != "system_status":
            return False
        return True

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "active_requests": len(self._active_requests),
            "queued_requests": self._request_queue.qsize(),
            "available_engines": list(self._engines.keys()),
            "resource_status": {
                "memory_budget_mb": self.memory_budget_mb,
                "cache_size_mb": self.cache_size_mb,
                "max_workers": self.max_workers,
            },
            "performance_summary": {
                "total_computations": len(self._performance_history),
                "engine_performance": {
                    engine.value: {
                        "count": len(times),
                        "avg_time": np.mean(times) if times else 0.0,
                        "total_time": np.sum(times) if times else 0.0,
                    }
                    for engine, times in self._engine_performance.items()
                },
            },
            "engines_available": HAS_ALL_ENGINES,
            "engine_import_availability": {
                engine.value: available
                for engine, available in ENGINE_IMPORT_AVAILABILITY.items()
            },
            "engine_initialization_errors": {
                engine.value: message
                for engine, message in self._engine_initialization_errors.items()
            },
            "math_backends_available": HAS_MATH_BACKENDS,
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the computational hub."""
        self._shutdown = True

        if self._queue_thread and self._queue_thread.is_alive():
            self._queue_thread.join(timeout=1.0)

        self._executor.shutdown(wait=True)


# Global hub instance
_global_hub: TNFRComputationalHub | None = None


def get_computational_hub() -> TNFRComputationalHub:
    """Get global computational hub."""
    global _global_hub

    if _global_hub is None:
        _global_hub = TNFRComputationalHub()

    return _global_hub


def execute_unified_computation(
    operation: str,
    graph: Any,
    engine_type: EngineType = EngineType.UNIFIED_BACKEND,
    **kwargs,
) -> ComputationResult:
    """Convenience function for unified computation."""
    hub = get_computational_hub()

    request = ComputationRequest(
        engine_type=engine_type, operation=operation, graph=graph, parameters=kwargs
    )

    return hub.execute_computation_sync(request)


def batch_execute_computations(
    requests: list, max_parallel: int = 4
) -> dict[str, ComputationResult]:
    """Execute multiple computations in parallel."""
    hub = get_computational_hub()

    # Submit all requests
    request_ids = []
    for req in requests:
        req_id = hub.submit_computation(req)
        request_ids.append(req_id)

    # Collect results
    results = {}
    for req_id in request_ids:
        result = hub.get_result(req_id, timeout=300.0)
        results[req_id] = result

    return results
