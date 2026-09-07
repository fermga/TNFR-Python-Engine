"""GPU compatibility facade with explicit backend provenance.

The facade discovers optional JAX, Torch, and CuPy devices, validates finite
array inputs, records observed resource telemetry, and provides an explicit CPU
fallback policy. Canonical graph pressure and canonical AL/RA graph commits use
the shared CPU implementations until an accelerated implementation has matching
directed, weighted, phase-gated, and transactional semantics.

Device availability alone is not performance evidence. Unknown resource fields
remain ``None`` and no speedup is inferred without a measured accelerated run.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import psutil

from ...config import get_config

# Unified mathematics backend integration
from ...mathematics.backend import get_backend
from ...mathematics.unified_numerical import np

logger = logging.getLogger(__name__)


@dataclass
class GPUDeviceInfo:
    """Unified GPU device information."""

    device_id: int
    name: str
    backend: str  # "jax", "torch", "cupy"
    total_memory_mb: float | None
    free_memory_mb: float | None
    utilization_percent: float | None
    compute_capability: str | None = None
    is_available: bool = True


@dataclass
class GPUOperationResult:
    """Unified result container for GPU operations."""

    # Core results
    result_data: np.ndarray
    backend_used: str

    # Performance metrics
    computation_time_ms: float
    memory_usage_mb: float

    # Optional
    device_used: str | None = None
    gpu_utilization: float | None = None

    # Execution details
    fallback_used: bool = False
    memory_transferred_mb: float = 0.0

    # Quality indicators
    precision: str = "float32"
    convergence_achieved: bool | None = None

    # Telemetry
    operation_metadata: dict[str, Any] = field(default_factory=dict)


class GraphDeltaNFRResult(dict[Any, float]):
    """Mapping result with honest backend provenance for graph ΔNFR reads."""

    def __init__(
        self,
        values: dict[Any, float],
        *,
        backend_used: str,
        fallback_used: bool,
    ) -> None:
        super().__init__(values)
        self.backend_used = backend_used
        self.fallback_used = fallback_used


def _finite_real_array(value: Any, label: str) -> np.ndarray:
    """Materialize one array-like input as finite binary64 real values."""

    try:
        raw = np.asarray(value)
        object_view = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must contain finite real values") from exc
    if raw.dtype.kind not in "iuf" or any(
        isinstance(item, (bool, np.bool_)) for item in object_view.flat
    ):
        raise TypeError(f"{label} must contain finite real values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must contain finite real values") from exc
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{label} must contain finite real values")
    return array


def _normalize_dense_nodal_inputs(
    adjacency: Any,
    epi: Any,
    vf: Any,
    phase: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and materialize the complete dense nodal input exactly once."""

    weights = _finite_real_array(adjacency, "adjacency")
    field = _finite_real_array(epi, "EPI")
    capacity = _finite_real_array(vf, "nu_f")
    phase_field = _finite_real_array(phase, "phase")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if field.ndim != 1 or field.shape[0] != weights.shape[0]:
        raise ValueError("EPI must be a vector aligned with adjacency")
    for label, vector in (("nu_f", capacity), ("phase", phase_field)):
        if vector.ndim != 1 or vector.shape[0] != field.shape[0]:
            raise ValueError(f"{label} must be a vector aligned with EPI")
    if bool(np.any(weights < 0.0)):
        raise ValueError("adjacency conductance must be nonnegative")
    if bool(np.any(capacity < 0.0)):
        raise ValueError("nu_f must be nonnegative")
    return weights, field, capacity, phase_field


def _dense_epi_pressure_from_arrays(
    weights: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """Return row-normalized neighbour mean minus self from validated arrays."""

    pressure = np.zeros_like(field, dtype=float)
    if field.size == 0:
        return pressure
    row_scale = np.max(weights, axis=1)
    active = row_scale > 0.0
    if not bool(np.any(active)):
        return pressure
    normalized = np.zeros_like(weights, dtype=float)
    normalized[active] = weights[active] / row_scale[active, None]
    row_sum = np.sum(normalized[active], axis=1)
    neighbor_mean = (normalized[active] @ field) / row_sum
    pressure[active] = neighbor_mean - field[active]
    return pressure


@dataclass
class UnifiedGPUConfig:
    """Configuration for unified GPU system."""

    # Backend preferences
    preferred_backend: str = "torch"  # "jax", "torch", "cupy", "auto"
    enable_gpu_acceleration: bool = True
    auto_backend_selection: bool = True

    # Memory management
    max_memory_usage_percent: float = 80.0
    memory_cleanup_threshold: float = 90.0
    enable_memory_pooling: bool = True

    # Device selection
    device_selection_strategy: str = (
        "memory_optimal"  # "memory_optimal", "compute_optimal", "round_robin"
    )
    multi_gpu_enabled: bool = True

    # Fallback settings
    enable_cpu_fallback: bool = True
    fallback_threshold_nodes: int = 50000

    # Performance tuning
    batch_size_optimization: bool = True
    precision_scaling: str = "auto"  # "auto", "float32", "float64"

    # Monitoring
    enable_profiling: bool = False
    log_memory_usage: bool = True


class TNFRUnifiedGPUSystem:
    """Route optional array operations and expose their actual provenance.

    The system owns device discovery, resource observations, backend selection,
    cleanup, and explicitly configured CPU recovery. Graph-aware structural
    operations retain the canonical CPU realization described by the module
    contract until semantic parity is demonstrated by executable tests.
    """

    def __init__(self, config: UnifiedGPUConfig | None = None):
        """Initialize unified GPU system with configuration."""
        self.config = config or UnifiedGPUConfig()

        # Get global configuration integration
        self.global_config = get_config()

        # Initialize backend system
        self.math_backend = get_backend()

        # Device management
        self._available_devices: list[GPUDeviceInfo] = []
        self._current_device: GPUDeviceInfo | None = None
        self._device_load_balance: dict[int, float] = {}

        # Memory management
        self._memory_pools: dict[str, Any] = {}
        self._active_allocations: dict[str, float] = {}

        # Performance tracking
        self._operation_stats = {
            "total_operations": 0,
            "gpu_operations": 0,
            "cpu_fallbacks": 0,
            "memory_errors": 0,
            "average_gpu_time_ms": 0.0,
        }

        # Initialize GPU backends and devices
        self._initialize_gpu_backends()
        self._detect_available_devices()

        if self.config.log_memory_usage:
            logger.info(
                "Initialized unified GPU system with %d devices",
                len(self._available_devices),
            )

    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return len(self._available_devices) > 0 and self.config.enable_gpu_acceleration

    def _initialize_gpu_backends(self) -> None:
        """Initialize available GPU backends through mathematics backend."""
        try:
            # Check what backends are available through unified system
            backend_info = self.math_backend.get_backend_info()

            # Check for 'accelerated' (standard) or 'supports_gpu' (legacy)
            is_accelerated = backend_info.get("accelerated", False) or backend_info.get(
                "supports_gpu", False
            )

            if is_accelerated:
                logger.info(f"GPU backend available: {backend_info['name']}")
            else:
                logger.warning("No GPU backend available through mathematics backend")

        except Exception as e:
            logger.error(f"Failed to initialize GPU backends: {e}")

    @staticmethod
    def _read_gpu_memory_mb(
        backend_info: dict[str, Any], field: str
    ) -> float | None:
        """Return an observed finite nonnegative memory value, if present."""

        value = backend_info.get(field)
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            memory_mb = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(memory_mb) or memory_mb < 0.0:
            return None
        return memory_mb

    @staticmethod
    def _read_gpu_utilization(backend_info: dict[str, Any]) -> float | None:
        """Return a validated observed percentage or None when unavailable."""

        value = backend_info.get("utilization_percent")
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        try:
            utilization = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(utilization) or not 0.0 <= utilization <= 100.0:
            return None
        return utilization

    def _detect_available_devices(self) -> None:
        """Detect available GPU devices across all backends."""
        devices = []

        # Try to get devices through mathematics backend
        try:
            backend_info = self.math_backend.get_backend_info()

            # Check for 'accelerated' (standard) or 'supports_gpu' (legacy)
            is_accelerated = backend_info.get("accelerated", False) or backend_info.get(
                "supports_gpu", False
            )

            if is_accelerated:
                # Create device info based on backend
                device = GPUDeviceInfo(
                    device_id=0,
                    name=backend_info.get("device_name", "GPU Device"),
                    backend=backend_info["name"],
                    total_memory_mb=self._read_gpu_memory_mb(
                        backend_info, "total_memory_mb"
                    ),
                    free_memory_mb=self._read_gpu_memory_mb(
                        backend_info, "free_memory_mb"
                    ),
                    utilization_percent=self._read_gpu_utilization(backend_info),
                    is_available=True,
                )
                devices.append(device)

        except Exception as e:
            logger.warning(f"Could not detect GPU devices: {e}")

        self._available_devices = devices

        # Select initial device
        if devices:
            self._current_device = self._select_optimal_device()

    def _select_optimal_device(self) -> GPUDeviceInfo | None:
        """Select optimal GPU device based on strategy."""
        if not self._available_devices:
            return None

        strategy = self.config.device_selection_strategy

        if strategy == "memory_optimal":
            # Prefer the largest observed capacity; unknown readings sort last.
            return max(
                self._available_devices,
                key=lambda device: (
                    device.free_memory_mb is not None,
                    (
                        device.free_memory_mb
                        if device.free_memory_mb is not None
                        else float("-inf")
                    ),
                ),
            )

        elif strategy == "compute_optimal":
            # Prefer the lowest observed utilization; unknown readings sort last.
            return min(
                self._available_devices,
                key=lambda device: (
                    device.utilization_percent is None,
                    (
                        device.utilization_percent
                        if device.utilization_percent is not None
                        else float("inf")
                    ),
                ),
            )

        elif strategy == "round_robin":
            # Round-robin selection with load balancing
            device_loads = [
                (d.device_id, self._device_load_balance.get(d.device_id, 0.0))
                for d in self._available_devices
            ]
            device_id = min(device_loads, key=lambda x: x[1])[0]
            return next(d for d in self._available_devices if d.device_id == device_id)

        else:
            # Default: first available device
            return self._available_devices[0]

    def compute_delta_nfr_gpu(
        self,
        adjacency: Any,
        epi: Any,
        vf: Any,
        phase: Any,
        **kwargs: Any,
    ) -> GPUOperationResult:
        """Compute the canonical dense EPI-channel structural pressure.

        This compatibility entry point keeps its historical signature, but νf
        and phase are inputs to other nodal channels and are not folded into
        EPI pressure.  The current backend-specific kernel used a noncanonical
        unnormalized sum, so this path deliberately reports a canonical CPU
        fallback until an accelerated kernel has exact semantic parity.

        Parameters
        ----------
        adjacency : array-like
            Square nonnegative conductance matrix.
        epi : array-like
            Finite real EPI structural configuration values.
        vf : array-like
            Structural frequency values (νf), accepted for API compatibility
            and kept separate from pressure.
        phase : array-like
            Phase values (φ/θ), accepted for API compatibility and kept
            separate from the pure EPI channel.
        **kwargs
            Additional computation parameters

        Returns
        -------
        GPUOperationResult
            Unified result with ``D^-1 W EPI - EPI`` and backend provenance.
        """
        import time

        start_time = time.perf_counter()
        weights, field, capacity, phase_field = _normalize_dense_nodal_inputs(
            adjacency, epi, vf, phase
        )
        result_data = _dense_epi_pressure_from_arrays(weights, field)
        node_count = len(result_data)

        stats = getattr(self, "_operation_stats", None)
        if isinstance(stats, dict):
            stats["total_operations"] = stats.get("total_operations", 0) + 1
            stats["cpu_fallbacks"] = stats.get("cpu_fallbacks", 0) + 1
        computation_time = (time.perf_counter() - start_time) * 1000
        return GPUOperationResult(
            result_data=result_data,
            backend_used="canonical-cpu",
            device_used=None,
            computation_time_ms=computation_time,
            memory_usage_mb=self._estimate_memory_usage(
                weights, field, capacity, phase_field
            ),
            fallback_used=True,
            precision="float64",
            gpu_utilization=None,
            convergence_achieved=None,
            operation_metadata={
                "operation": "delta_nfr_epi_channel",
                "nodes": node_count,
                "semantics": "D^-1 W EPI - EPI",
                "nu_f_applied": False,
                "phase_applied": False,
                "ignored_options": tuple(sorted(kwargs)),
            },
        )

    def compute_structural_fields(
        self, graph_data: np.ndarray, **kwargs: Any
    ) -> GPUOperationResult:
        """Reject the legacy array-only tetrad placeholder.

        Φ_s, |∇φ|, K_φ and ξ_C require graph topology, distances and field
        provenance that an anonymous matrix cannot supply.  Call the canonical
        graph functions in :mod:`tnfr.physics.fields` instead.
        """
        raise NotImplementedError(
            "Array-only structural fields cannot represent the canonical TNFR "
            "tetrad; use tnfr.physics.fields with the source graph"
        )

    def compute_delta_nfr_from_graph(self, graph: Any) -> GraphDeltaNFRResult:
        """Read the canonical EPI-channel ΔNFR from a TNFR graph.

        The dense low-level GPU kernel predates the graph transport contract
        and cannot represent all directed, weighted and parallel-edge
        conventions.  Until that kernel has semantic parity, this public graph
        adapter delegates to the shared CPU structural-diffusion oracle.  The
        returned mapping retains ``backend_used`` and ``fallback_used`` so the
        caller can distinguish this compatibility fallback from acceleration.

        Parameters
        ----------
        graph : TNFRGraph
            Network graph with TNFR attributes

        Returns
        -------
        GraphDeltaNFRResult
            Node-to-pressure mapping with backend provenance.  Values are the
            pure EPI channel ``-L_rw @ EPI``; νf remains the separate capacity
            multiplier in the nodal equation.
        """
        from ...physics.structural_diffusion import (
            structural_diffusion_operator,
            structural_field,
        )

        nodes, laplacian = structural_diffusion_operator(graph)
        epi_field = structural_field(graph, nodes)
        pressure = -(laplacian @ epi_field)
        stats = getattr(self, "_operation_stats", None)
        if isinstance(stats, dict):
            stats["total_operations"] = stats.get("total_operations", 0) + 1
            stats["cpu_fallbacks"] = stats.get("cpu_fallbacks", 0) + 1
        result = GraphDeltaNFRResult(
            {node: float(value) for node, value in zip(nodes, pressure)},
            backend_used="canonical-cpu",
            fallback_used=True,
        )
        self._last_graph_delta_nfr_result = result
        return result

    def execute_with_fallback(
        self,
        operation: Callable[..., GPUOperationResult],
        *args: Any,
        cpu_fallback: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> GPUOperationResult:
        """Execute a GPU callable and, when declared, a distinct CPU callable.

        The successful-call path is backward compatible with the former API.
        A failed GPU callable is never invoked a second time under a CPU label.
        Callers that require recovery must provide cpu_fallback explicitly.
        """

        if cpu_fallback is operation:
            raise ValueError("cpu_fallback must be distinct from the GPU operation")
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            fallback_enabled = bool(
                getattr(
                    getattr(self, "config", None),
                    "enable_cpu_fallback",
                    True,
                )
            )
            if not fallback_enabled:
                logger.warning(
                    "GPU operation failed and CPU fallback is disabled: %s", exc
                )
                raise
            if cpu_fallback is None:
                logger.warning(
                    "GPU operation failed and no distinct CPU fallback was "
                    "provided: %s",
                    exc,
                )
                raise

            logger.warning(
                "GPU operation failed; executing declared CPU fallback: %s", exc
            )
            stats = getattr(self, "_operation_stats", None)
            if isinstance(stats, dict):
                stats["cpu_fallbacks"] = stats.get("cpu_fallbacks", 0) + 1
            return self._execute_cpu_fallback(cpu_fallback, *args, **kwargs)

    def execute_with_gpu_fallback(
        self,
        gpu_fn: Callable[..., Any],
        cpu_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Execute distinct GPU/CPU callables under the configured policy."""

        if cpu_fn is gpu_fn:
            raise ValueError("cpu_fn must be distinct from gpu_fn")
        try:
            return gpu_fn(*args, **kwargs), "gpu"
        except Exception as exc:
            if not self.config.enable_cpu_fallback:
                logger.warning(
                    "GPU execution failed and CPU fallback is disabled: %s", exc
                )
                raise
            logger.warning(
                "GPU execution failed; executing declared CPU fallback: %s", exc
            )
            stats = getattr(self, "_operation_stats", None)
            if isinstance(stats, dict):
                stats["cpu_fallbacks"] = stats.get("cpu_fallbacks", 0) + 1
            return cpu_fn(*args, **kwargs), "cpu"

    def has_gpu_backend(self) -> bool:
        """Check if GPU backend is available (compatibility alias)."""
        return self.is_available

    def _can_handle_gpu_operation(self, *arrays: np.ndarray) -> bool:
        """Check if GPU can handle the operation based on memory constraints."""
        if not self._available_devices or not self.config.enable_gpu_acceleration:
            return False

        # Estimate memory requirement
        total_memory_needed = sum(array.nbytes for array in arrays) / (
            1024 * 1024
        )  # MB

        # Check against current device memory
        if self._current_device:
            available_memory = self._current_device.free_memory_mb
            if available_memory is None:
                return False
            memory_threshold = available_memory * (
                self.config.max_memory_usage_percent / 100.0
            )

            return total_memory_needed <= memory_threshold

        return False

    def _compute_delta_nfr_cpu(
        self,
        adjacency: Any,
        epi: Any,
        vf: Any,
        phase: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Delegate legacy fallback calls to the canonical dense EPI kernel."""

        weights, field, _capacity, _phase = _normalize_dense_nodal_inputs(
            adjacency, epi, vf, phase
        )
        return _dense_epi_pressure_from_arrays(weights, field)

    def _compute_structural_fields_cpu(
        self, graph_data: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Reject the removed zero-tetrad compatibility placeholder."""
        raise NotImplementedError(
            "Canonical structural fields require a graph; use tnfr.physics.fields"
        )

    def _execute_cpu_fallback(
        self, operation: Callable, *args: Any, **kwargs: Any
    ) -> GPUOperationResult:
        """Execute generic CPU fallback operation."""
        import time

        start_time = time.perf_counter()

        result_data = operation(*args, **kwargs)
        computation_time = (time.perf_counter() - start_time) * 1000

        if isinstance(result_data, GPUOperationResult):
            return replace(
                result_data,
                fallback_used=True,
                operation_metadata={
                    **result_data.operation_metadata,
                    "execution": "declared_cpu_fallback",
                },
            )
        return GPUOperationResult(
            result_data=result_data,
            backend_used="cpu_fallback",
            computation_time_ms=computation_time,
            memory_usage_mb=self._estimate_memory_usage(*args),
            fallback_used=True,
            gpu_utilization=None,
            convergence_achieved=None,
            operation_metadata={"execution": "declared_cpu_fallback"},
        )

    def _estimate_memory_usage(self, *arrays: Any) -> float:
        """Estimate materialized input storage in MiB."""

        total_bytes = 0
        for value in arrays:
            nbytes = getattr(value, "nbytes", None)
            if nbytes is None:
                try:
                    nbytes = np.asarray(value).nbytes
                except (TypeError, ValueError, OverflowError):
                    continue
            total_bytes += int(nbytes)
        return total_bytes / (1024 * 1024)

    def _get_current_gpu_utilization(self) -> float | None:
        """Return the observed GPU utilization, if one is available."""

        if self._current_device:
            return self._current_device.utilization_percent
        return None

    def cleanup_memory(self) -> None:
        """Clean up GPU memory and resources."""
        try:
            # Trigger garbage collection
            gc.collect()

            # Clear memory pools if available
            if hasattr(self.math_backend, "clear_memory"):
                self.math_backend.clear_memory()

            # Reset allocation tracking
            self._active_allocations.clear()

            logger.info("GPU memory cleanup completed")

        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")

    def get_device_info(self) -> list[GPUDeviceInfo]:
        """Get information about available GPU devices."""
        # Refresh device information
        self._detect_available_devices()
        return self._available_devices.copy()

    def get_memory_info(self) -> dict[str, Any]:
        """Get detailed memory usage information."""
        info = {
            "total_devices": len(self._available_devices),
            "current_device": (
                self._current_device.name if self._current_device else None
            ),
            "active_allocations": self._active_allocations.copy(),
            "system_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
        }

        if self._current_device:
            info.update(
                {
                    "gpu_total_memory_mb": self._current_device.total_memory_mb,
                    "gpu_free_memory_mb": self._current_device.free_memory_mb,
                    "gpu_utilization_percent": self._current_device.utilization_percent,
                }
            )

        return info

    def get_performance_stats(self) -> dict[str, Any]:
        """Get GPU system performance statistics."""
        stats = self._operation_stats.copy()

        if stats["total_operations"] > 0:
            stats["gpu_success_rate"] = (
                stats["gpu_operations"] / stats["total_operations"]
            ) * 100.0
            stats["cpu_fallback_rate"] = (
                stats["cpu_fallbacks"] / stats["total_operations"]
            ) * 100.0
        else:
            stats["gpu_success_rate"] = 0.0
            stats["cpu_fallback_rate"] = 0.0

        return stats

    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return (
            len(self._available_devices) > 0
            and self.config.enable_gpu_acceleration
            and self._current_device is not None
        )

    def set_device(self, device_id: int) -> bool:
        """set active GPU device by ID."""
        device = next(
            (d for d in self._available_devices if d.device_id == device_id), None
        )

        if device:
            self._current_device = device
            logger.info(f"Switched to GPU device {device_id}: {device.name}")
            return True

        logger.warning(f"Device {device_id} not available")
        return False


# ============================================================================
# PUBLIC API - Unified GPU Interface
# ============================================================================

# Global unified GPU system instance
_unified_gpu_system: TNFRUnifiedGPUSystem | None = None


def get_unified_gpu_system(
    config: UnifiedGPUConfig | None = None,
) -> TNFRUnifiedGPUSystem:
    """Get or create global unified GPU system.

    This provides a singleton interface for all TNFR GPU operations
    to eliminate redundant system creation across modules.

    Parameters
    ----------
    config : UnifiedGPUConfig, optional
        Configuration for system (only used on first call)

    Returns
    -------
    TNFRUnifiedGPUSystem
        Global unified GPU system instance
    """
    global _unified_gpu_system

    if _unified_gpu_system is None:
        _unified_gpu_system = TNFRUnifiedGPUSystem(config)
        logger.info("Created global unified GPU system")

    return _unified_gpu_system


# Convenience functions for direct GPU operations
def compute_unified_delta_nfr(
    adjacency: Any,
    epi: Any,
    vf: Any,
    phase: Any,
    **kwargs: Any,
) -> GPUOperationResult:
    """Compute canonical dense EPI pressure with explicit backend provenance."""
    return get_unified_gpu_system().compute_delta_nfr_gpu(
        adjacency, epi, vf, phase, **kwargs
    )


def compute_unified_structural_fields(
    graph_data: np.ndarray, **kwargs: Any
) -> GPUOperationResult:
    """Reject the legacy array-only structural-tetrad placeholder."""
    return get_unified_gpu_system().compute_structural_fields(graph_data, **kwargs)


def cleanup_unified_gpu_memory() -> None:
    """Clean up unified GPU memory - convenience function."""
    if _unified_gpu_system is not None:
        _unified_gpu_system.cleanup_memory()


def get_unified_gpu_stats() -> dict[str, Any]:
    """Get unified GPU system statistics - convenience function."""
    if _unified_gpu_system is not None:
        return {
            "performance": _unified_gpu_system.get_performance_stats(),
            "memory": _unified_gpu_system.get_memory_info(),
            "devices": [
                device.__dict__ for device in _unified_gpu_system.get_device_info()
            ],
        }
    return {"status": "system_not_initialized"}


def execute_with_gpu_fallback(
    gpu_fn: Callable[..., Any], cpu_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, str]:
    """Execute with GPU fallback (compatibility wrapper).

    Parameters
    ----------
    gpu_fn : Callable
        GPU operation to attempt
    cpu_fn : Callable
        CPU fallback operation
    *args, **kwargs
        Operation arguments

    Returns
    -------
    tuple[Any, str]
        (result, backend_used)
    """
    return get_unified_gpu_system().execute_with_gpu_fallback(
        gpu_fn, cpu_fn, *args, **kwargs
    )
