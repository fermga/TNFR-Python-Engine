"""Truthful facade for one-dimensional FFT operations on ordered sequences.

This module does not perform a Graph Fourier Transform. Its inputs are plain
one-dimensional arrays and compute_spectral_convolution returns circular
sequence convolution through numpy.fft. Graph-Laplacian decomposition and
the physical-space nodal product nu_f * delta_nfr remain separate APIs.

The historical advanced and distributed engines expose graph-oriented
methods. They are used here only if they implement the requested sequence
protocol; otherwise provenance records an explicit NumPy fallback.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol

from ...config import get_config
from ...dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from ...dynamics.distributed_fft import DistributedFFTEngine
from ...dynamics.fft_backend import FFTBackendCapabilities
from ...dynamics.multi_modal_cache import cache_signature_digest
from ...errors import TNFRValueError
from ...mathematics.unified_numerical import np
from .unified_gpu_system import get_unified_gpu_system

logger = logging.getLogger(__name__)

_SUPPORTED_PRECISIONS = frozenset({"float32", "float64"})


def _estimate_owned_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate owned Python and array payload bytes without double counting."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _estimate_owned_bytes(key, seen) + _estimate_owned_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_estimate_owned_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return size + _estimate_owned_bytes(vars(value), seen)
    return size


def _validate_sequence_values(data: np.ndarray, *, label: str) -> None:
    """Reject nonnumeric, boolean, NaN, and infinite sequence coordinates."""

    if data.dtype.kind not in "iufc" or not np.all(np.isfinite(data)):
        raise TNFRValueError(
            f"{label} must contain only finite numeric scalars",
            context={"dtype": str(data.dtype), "shape": data.shape},
            suggestion="Remove booleans, strings, NaNs, and infinities.",
        )


@dataclass
class UnifiedFFTConfig:
    """Configuration for the unified sequence-FFT engine."""

    preferred_backend: str = "advanced"
    auto_backend_selection: bool = True
    enable_gpu_acceleration: bool = True

    cache_spectral_decompositions: bool = True
    max_cache_size_mb: float = 512.0
    enable_precision_scaling: bool = True

    distribute_threshold_nodes: int = 10000
    max_workers: int = 4

    spectral_precision: str = "float64"
    convergence_tolerance: float = 1e-12

    profile_operations: bool = False
    log_backend_selection: bool = True


@dataclass
class UnifiedFFTResult:
    """Result container shared by the sequence-FFT operations."""

    spectral_data: np.ndarray
    frequencies: np.ndarray
    backend_used: str

    computation_time_ms: float
    memory_usage_mb: float
    cache_hit: bool = False

    spectral_precision: str = "float64"
    convergence_achieved: bool | None = None

    harmonic_analysis: dict[str, Any] | None = None
    coherence_matrix: np.ndarray | None = None
    phase_relationships: dict[str, float] | None = None
    operation_metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedFFTBackend(Protocol):
    """Protocol implemented by ordered-sequence FFT backends."""

    def get_capabilities(self) -> FFTBackendCapabilities:
        """Return backend capabilities."""
        ...

    def compute_fft(self, data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
        """Compute a one-dimensional FFT."""
        ...

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute circular convolution for equally sized sequences."""
        ...


class TNFRUnifiedFFTEngine:
    """Single access point for one-dimensional sequence FFT operations."""

    def __init__(self, config: UnifiedFFTConfig | None = None):
        self.config = config or UnifiedFFTConfig()
        self._validate_config()

        self.gpu_manager = get_unified_gpu_system()
        self._advanced_engine: TNFRAdvancedFFTEngine | None = None
        self._distributed_engine: DistributedFFTEngine | None = None
        self._basic_backends: dict[str, UnifiedFFTBackend] = {}

        self._spectral_cache: OrderedDict[str, UnifiedFFTResult] = OrderedDict()
        self._cache_lock = RLock()
        self._cache_sizes: dict[str, int] = {}
        self._cache_bytes = 0
        self._max_cache_bytes = int(self.config.max_cache_size_mb * 1024 * 1024)
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "oversize_skips": 0,
        }

        self.global_config = get_config()
        if self.config.log_backend_selection:
            logger.info("Initialized unified FFT engine with config: %s", self.config)

    def _validate_config(self) -> None:
        """Validate options that alter numerical results or resource bounds."""

        if self.config.preferred_backend not in {"advanced", "distributed", "basic"}:
            raise TNFRValueError(
                "preferred_backend must name an available FFT backend",
                context={"preferred_backend": self.config.preferred_backend},
                suggestion="Use 'advanced', 'distributed', or 'basic'.",
            )
        if self.config.spectral_precision not in _SUPPORTED_PRECISIONS:
            raise TNFRValueError(
                "spectral_precision must be 'float32' or 'float64'",
                context={"spectral_precision": self.config.spectral_precision},
            )
        cache_size = self.config.max_cache_size_mb
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, (int, float))
            or not math.isfinite(float(cache_size))
            or float(cache_size) < 0.0
        ):
            raise TNFRValueError(
                "max_cache_size_mb must be finite and nonnegative",
                context={"max_cache_size_mb": cache_size},
            )
        tolerance = self.config.convergence_tolerance
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) <= 0.0
        ):
            raise TNFRValueError(
                "convergence_tolerance must be finite and positive",
                context={"convergence_tolerance": tolerance},
            )

    def _effective_precision(self, *arrays: np.ndarray) -> str:
        """Resolve the actual arithmetic precision under the configured policy."""

        if self.config.enable_precision_scaling:
            return self.config.spectral_precision
        low_precision = {
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("complex64"),
        }
        if arrays and all(np.asarray(array).dtype in low_precision for array in arrays):
            return "float32"
        return "float64"

    def _get_cache_key(self, data: Any, operation: str, **kwargs: Any) -> str:
        """Return a type-, shape-, and parameter-aware operation key."""

        return cache_signature_digest(
            {
                "schema": "unified_sequence_fft_v2",
                "operation": operation,
                "payload": data,
                "parameters": kwargs,
            }
        )

    def _check_cache(self, cache_key: str) -> UnifiedFFTResult | None:
        if not self.config.cache_spectral_decompositions:
            return None
        with self._cache_lock:
            if cache_key in self._spectral_cache:
                self._cache_stats["hits"] += 1
                self._spectral_cache.move_to_end(cache_key)
                result = deepcopy(self._spectral_cache[cache_key])
                result.cache_hit = True
                return result
            self._cache_stats["misses"] += 1
        return None

    def _store_cache(self, cache_key: str, result: UnifiedFFTResult) -> None:
        if (
            not self.config.cache_spectral_decompositions
            or self._max_cache_bytes <= 0
        ):
            return

        stored = deepcopy(result)
        stored.cache_hit = False
        size = sys.getsizeof(cache_key) + _estimate_owned_bytes(stored)
        with self._cache_lock:
            if size > self._max_cache_bytes:
                self._cache_stats["oversize_skips"] += 1
                return

            if cache_key in self._spectral_cache:
                self._cache_bytes -= self._cache_sizes.pop(cache_key)
                del self._spectral_cache[cache_key]

            while (
                self._spectral_cache
                and self._cache_bytes + size > self._max_cache_bytes
            ):
                oldest_key, _ = self._spectral_cache.popitem(last=False)
                self._cache_bytes -= self._cache_sizes.pop(oldest_key)
                self._cache_stats["evictions"] += 1

            self._spectral_cache[cache_key] = stored
            self._cache_sizes[cache_key] = size
            self._cache_bytes += size
    def _select_backend(self, data_shape: tuple[int, ...], operation: str) -> str:
        """Select only a backend that implements the sequence protocol."""

        if not self.config.auto_backend_selection:
            return self.config.preferred_backend

        n_elements = int(np.prod(data_shape))
        candidate = "basic"
        if n_elements > self.config.distribute_threshold_nodes:
            candidate = "distributed"
        elif (
            n_elements > 1000
            and self.config.enable_gpu_acceleration
            and self.gpu_manager.has_gpu_backend()
        ):
            candidate = "advanced"

        if candidate == "basic":
            return candidate
        method = (
            "compute_fft" if operation == "fft" else "compute_spectral_convolution"
        )
        engine = self._get_backend_engine(candidate)
        return candidate if callable(getattr(engine, method, None)) else "basic"

    def _get_backend_engine(
        self, backend_name: str, spectral_precision: str | None = None
    ) -> Any:
        if backend_name == "advanced":
            if self._advanced_engine is None:
                self._advanced_engine = TNFRAdvancedFFTEngine()
            return self._advanced_engine
        if backend_name == "distributed":
            if self._distributed_engine is None:
                self._distributed_engine = DistributedFFTEngine()
            return self._distributed_engine
        if backend_name == "basic":
            precision = spectral_precision or self.config.spectral_precision
            key = f"basic:{precision}"
            if key not in self._basic_backends:
                self._basic_backends[key] = _BasicNumpyFFTBackend(precision)
            return self._basic_backends[key]
        raise TNFRValueError(
            f"Unknown FFT backend: {backend_name}",
            context={
                "requested": backend_name,
                "available": ["advanced", "distributed", "basic"],
            },
            suggestion="Use 'advanced', 'distributed', or 'basic'.",
        )

    def _resolve_sequence_backend(
        self,
        requested_backend: str,
        operation: str,
        spectral_precision: str,
    ) -> tuple[UnifiedFFTBackend, str, str | None]:
        candidate = self._get_backend_engine(requested_backend, spectral_precision)
        method = (
            "compute_fft" if operation == "fft" else "compute_spectral_convolution"
        )
        if callable(getattr(candidate, method, None)):
            return candidate, requested_backend, None
        fallback = self._get_backend_engine("basic", spectral_precision)
        return fallback, "basic", requested_backend

    def compute_fft(
        self, data: np.ndarray, backend: str | None = None, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute a finite one-dimensional sequence FFT with truthful telemetry."""

        data = np.asarray(data)
        if data.ndim != 1 or data.size == 0:
            raise TNFRValueError(
                "Sequence FFT requires a non-empty one-dimensional array",
                context={"shape": data.shape},
                suggestion=(
                    "Pass a non-empty vector; use graph-spectral APIs for graph "
                    "signals."
                ),
            )
        _validate_sequence_values(data, label="FFT input")

        precision = self._effective_precision(data)
        requested = backend or self._select_backend(data.shape, "fft")
        cache_key = self._get_cache_key(
            data,
            "fft",
            requested_backend=requested,
            spectral_precision=precision,
            options=kwargs,
        )
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        engine, actual_name, fallback_from = self._resolve_sequence_backend(
            requested, "fft", precision
        )
        started = time.perf_counter()
        try:
            result = engine.compute_fft(data, **kwargs)
        except Exception as exc:
            logger.error("FFT computation failed with backend %s: %s", actual_name, exc)
            if actual_name != "basic":
                return self.compute_fft(data, backend="basic", **kwargs)
            raise

        result.computation_time_ms = (time.perf_counter() - started) * 1000.0
        result.convergence_achieved = None
        result.operation_metadata.update(
            {
                "requested_backend": requested,
                "actual_backend": result.backend_used,
                "backend_fallback": fallback_from is not None,
                "input_dtype": str(data.dtype),
                "effective_precision": precision,
                "precision_scaling_enabled": self.config.enable_precision_scaling,
                "convergence_applicable": False,
                "convergence_tolerance_used": None,
            }
        )
        if fallback_from is not None:
            result.operation_metadata["fallback_from"] = fallback_from

        self._store_cache(cache_key, result)
        return result

    def compute_harmonic_analysis(
        self, epi_data: np.ndarray, frequencies: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Rank FFT coefficient amplitudes against supplied frequency labels."""

        frequencies = np.asarray(frequencies)
        if frequencies.shape != np.asarray(epi_data).shape:
            raise TNFRValueError(
                "Frequency labels must match the input sequence",
                context={
                    "data_shape": np.asarray(epi_data).shape,
                    "frequencies_shape": frequencies.shape,
                },
                suggestion="Provide one frequency label per input sample.",
            )
        _validate_sequence_values(frequencies, label="Frequency labels")
        result = self.compute_fft(epi_data, **kwargs)
        result.harmonic_analysis = self._extract_harmonics(
            result.spectral_data, frequencies
        )
        return result

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute circular convolution of two equally sized ordered sequences."""

        signal1 = np.asarray(signal1)
        signal2 = np.asarray(signal2)
        if (
            signal1.ndim != 1
            or signal2.ndim != 1
            or signal1.shape != signal2.shape
            or signal1.size == 0
        ):
            raise TNFRValueError(
                "Sequence convolution requires equal non-empty vectors",
                context={
                    "signal1_shape": signal1.shape,
                    "signal2_shape": signal2.shape,
                },
                suggestion="Pass equal one-dimensional vectors with positive length.",
            )
        _validate_sequence_values(signal1, label="First convolution input")
        _validate_sequence_values(signal2, label="Second convolution input")

        options = dict(kwargs)
        forced_backend = options.pop("backend", None)
        precision = self._effective_precision(signal1, signal2)
        requested = forced_backend or self._select_backend(
            signal1.shape, "convolution"
        )
        cache_key = self._get_cache_key(
            (signal1, signal2),
            "convolution",
            requested_backend=requested,
            spectral_precision=precision,
            options=options,
        )
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        engine, _, fallback_from = self._resolve_sequence_backend(
            requested, "convolution", precision
        )
        started = time.perf_counter()
        result = engine.compute_spectral_convolution(signal1, signal2, **options)
        result.computation_time_ms = (time.perf_counter() - started) * 1000.0
        result.convergence_achieved = None
        result.operation_metadata.update(
            {
                "requested_backend": requested,
                "actual_backend": result.backend_used,
                "backend_fallback": fallback_from is not None,
                "semantics": "circular_sequence_convolution",
                "input_dtypes": [str(signal1.dtype), str(signal2.dtype)],
                "effective_precision": precision,
                "precision_scaling_enabled": self.config.enable_precision_scaling,
                "convergence_applicable": False,
                "convergence_tolerance_used": None,
            }
        )
        if fallback_from is not None:
            result.operation_metadata["fallback_from"] = fallback_from

        self._store_cache(cache_key, result)
        return result

    def compute_cross_spectral_coherence(
        self, first: np.ndarray, second: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Return modewise normalized cross-power for two sequences.

        A single transform pair does not estimate statistical coherence. Its
        per-bin normalization reduces exactly to common nonzero spectral
        support. The compatibility method name is retained while metadata
        states that restricted scope.
        """

        if np.asarray(first).shape != np.asarray(second).shape:
            raise TNFRValueError(
                "Cross-power inputs must have the same sequence shape",
                context={
                    "first_shape": np.asarray(first).shape,
                    "second_shape": np.asarray(second).shape,
                },
                suggestion="Align both ordered sequences before comparison.",
            )

        started = time.perf_counter()
        fft1 = self.compute_fft(first, **kwargs)
        fft2 = self.compute_fft(second, **kwargs)
        coherence = self._compute_coherence_matrix(
            fft1.spectral_data, fft2.spectral_data
        )
        frequencies = np.array(fft1.frequencies, copy=True)
        payload_mb = (coherence.nbytes + frequencies.nbytes) / (1024 * 1024)
        precision = (
            fft1.spectral_precision
            if fft1.spectral_precision == fft2.spectral_precision
            else f"{fft1.spectral_precision}+{fft2.spectral_precision}"
        )
        return UnifiedFFTResult(
            spectral_data=coherence,
            frequencies=frequencies,
            backend_used=(
                fft1.backend_used
                if fft1.backend_used == fft2.backend_used
                else f"{fft1.backend_used}+{fft2.backend_used}"
            ),
            computation_time_ms=(time.perf_counter() - started) * 1000.0,
            memory_usage_mb=payload_mb,
            spectral_precision=precision,
            convergence_achieved=None,
            coherence_matrix=coherence,
            operation_metadata={
                "operation": "modewise_normalized_cross_power",
                "reduced_semantics": "common_nonzero_spectral_support",
                "input_backends": [fft1.backend_used, fft2.backend_used],
                "zero_power_value": 0.0,
                "convergence_applicable": False,
                "memory_semantics": "returned_array_payload",
            },
        )

    @staticmethod
    def _extract_harmonics(
        spectral_data: np.ndarray, frequencies: np.ndarray
    ) -> dict[str, Any]:
        magnitudes = np.abs(spectral_data)
        peak_indices = np.argsort(magnitudes)[-10:]
        dominant = float(magnitudes[peak_indices[-1]])
        return {
            "fundamental_freq": frequencies[peak_indices[-1]],
            "harmonic_freqs": frequencies[peak_indices].tolist(),
            "harmonic_amplitudes": magnitudes[peak_indices].tolist(),
            "total_harmonic_distortion": (
                float(np.sum(magnitudes[peak_indices[:-1]]) / dominant)
                if dominant > 0.0
                else 0.0
            ),
        }

    @staticmethod
    def _compute_coherence_matrix(
        fft1: np.ndarray, fft2: np.ndarray
    ) -> np.ndarray:
        """Compute scale-invariant normalized cross-power at every FFT bin."""

        first_nonzero = np.abs(fft1) > 0.0
        second_nonzero = np.abs(fft2) > 0.0
        return np.asarray(first_nonzero & second_nonzero, dtype=float)

    def get_cache_statistics(self) -> dict[str, Any]:
        with self._cache_lock:
            stats = dict(self._cache_stats)
            cache_size = len(self._spectral_cache)
            cache_bytes = self._cache_bytes
        total_requests = stats["hits"] + stats["misses"]
        hit_rate = stats["hits"] / total_requests * 100.0 if total_requests else 0.0
        return {
            **stats,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": cache_size,
            "cache_memory_estimate_mb": cache_bytes / (1024 * 1024),
            "cache_limit_mb": self._max_cache_bytes / (1024 * 1024),
            "cache_memory_semantics": "owned_cached_payload_estimate",
        }

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._spectral_cache.clear()
            self._cache_sizes.clear()
            self._cache_bytes = 0
            self._cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "oversize_skips": 0,
            }
        logger.info("Cleared unified FFT cache")
    def get_backend_info(self) -> dict[str, Any]:
        backends: dict[str, Any] = {}
        for name in ("advanced", "distributed", "basic"):
            try:
                engine = self._get_backend_engine(name)
                if hasattr(engine, "get_capabilities"):
                    backends[name] = {
                        **engine.get_capabilities().__dict__,
                        "sequence_fft_supported": callable(
                            getattr(engine, "compute_fft", None)
                        ),
                        "sequence_convolution_supported": callable(
                            getattr(engine, "compute_spectral_convolution", None)
                        ),
                    }
                else:
                    backends[name] = {
                        "status": "available",
                        "capabilities": "unknown",
                    }
            except Exception as exc:
                backends[name] = {"status": "unavailable", "error": str(exc)}
        return {
            "backends": backends,
            "config": self.config.__dict__,
            "gpu_available": self.gpu_manager.has_gpu_backend(),
            "cache_stats": self.get_cache_statistics(),
        }


class _BasicNumpyFFTBackend:
    """NumPy sequence backend with an explicit arithmetic precision."""

    def __init__(self, precision: str):
        if precision not in _SUPPORTED_PRECISIONS:
            raise TNFRValueError(
                "Basic FFT precision must be 'float32' or 'float64'",
                context={"spectral_precision": precision},
            )
        self.precision = precision
        self.real_dtype = np.dtype(precision)
        self.complex_dtype = np.dtype(
            "complex64" if precision == "float32" else "complex128"
        )

    def _cast(self, data: np.ndarray, *, label: str) -> np.ndarray:
        dtype = self.complex_dtype if np.iscomplexobj(data) else self.real_dtype
        cast = np.asarray(data, dtype=dtype)
        if not np.all(np.isfinite(cast)):
            raise TNFRValueError(
                f"{label} cannot be represented at {self.precision} precision",
                context={"input_dtype": str(np.asarray(data).dtype)},
                suggestion="Use float64 precision or reduce the input magnitude.",
            )
        return cast

    def get_capabilities(self) -> FFTBackendCapabilities:
        return FFTBackendCapabilities(
            backend_name="numpy_basic",
            max_nodes=None,
            precision=self.precision,
            supports_distributed=False,
            extra={"library": "numpy.fft", "sequence_protocol": True},
        )

    def compute_fft(self, data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
        del kwargs
        started = time.perf_counter()
        working = self._cast(data, label="FFT input")
        spectral = np.asarray(np.fft.fft(working), dtype=self.complex_dtype)
        if not np.all(np.isfinite(spectral)):
            raise TNFRValueError(
                "FFT output is nonfinite at the selected precision",
                context={"spectral_precision": self.precision},
                suggestion="Use float64 precision or reduce the input magnitude.",
            )
        frequencies = np.asarray(
            np.fft.fftfreq(len(working)), dtype=self.real_dtype
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        payload_mb = (spectral.nbytes + frequencies.nbytes) / (1024 * 1024)
        return UnifiedFFTResult(
            spectral_data=spectral,
            frequencies=frequencies,
            backend_used="basic",
            computation_time_ms=elapsed,
            memory_usage_mb=payload_mb,
            spectral_precision=self.precision,
            convergence_achieved=None,
            operation_metadata={
                "backend": "numpy",
                "algorithm": "fft",
                "spectral_dtype": str(spectral.dtype),
                "frequency_dtype": str(frequencies.dtype),
                "memory_semantics": "returned_array_payload",
            },
        )

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        del kwargs
        started = time.perf_counter()
        first = self._cast(signal1, label="First convolution input")
        second = self._cast(signal2, label="Second convolution input")
        first_fft = np.fft.fft(first)
        second_fft = np.fft.fft(second)
        inverse = np.fft.ifft(first_fft * second_fft)
        if np.iscomplexobj(signal1) or np.iscomplexobj(signal2):
            convolution = np.asarray(inverse, dtype=self.complex_dtype)
        else:
            convolution = np.asarray(np.real(inverse), dtype=self.real_dtype)
        if not np.all(np.isfinite(convolution)):
            raise TNFRValueError(
                "Convolution output is nonfinite at the selected precision",
                context={"spectral_precision": self.precision},
                suggestion="Use float64 precision or reduce the input magnitude.",
            )
        frequencies = np.asarray(
            np.fft.fftfreq(len(first)), dtype=self.real_dtype
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        payload_mb = (convolution.nbytes + frequencies.nbytes) / (1024 * 1024)
        return UnifiedFFTResult(
            spectral_data=convolution,
            frequencies=frequencies,
            backend_used="basic",
            computation_time_ms=elapsed,
            memory_usage_mb=payload_mb,
            spectral_precision=self.precision,
            convergence_achieved=None,
            operation_metadata={
                "backend": "numpy",
                "algorithm": "fft_circular_convolution",
                "semantics": "circular_sequence_convolution",
                "spectral_dtype": str(convolution.dtype),
                "frequency_dtype": str(frequencies.dtype),
                "memory_semantics": "returned_array_payload",
            },
        )


_unified_fft_engine: TNFRUnifiedFFTEngine | None = None


def get_unified_fft_engine(
    config: UnifiedFFTConfig | None = None,
) -> TNFRUnifiedFFTEngine:
    """Return the process-wide sequence-FFT engine."""

    global _unified_fft_engine
    if _unified_fft_engine is None:
        _unified_fft_engine = TNFRUnifiedFFTEngine(config)
        logger.info("Created global unified FFT engine")
    return _unified_fft_engine


def compute_unified_fft(data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
    """Compute a sequence FFT through the shared engine."""

    return get_unified_fft_engine().compute_fft(data, **kwargs)


def compute_unified_spectral_convolution(
    signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
) -> UnifiedFFTResult:
    """Compute circular sequence convolution through the shared engine."""

    return get_unified_fft_engine().compute_spectral_convolution(
        signal1, signal2, **kwargs
    )


def clear_unified_fft_cache() -> None:
    """Clear the process-wide sequence-FFT cache if initialized."""

    if _unified_fft_engine is not None:
        _unified_fft_engine.clear_cache()


def get_unified_fft_stats() -> dict[str, Any]:
    """Return process-wide FFT backend and cache telemetry."""

    if _unified_fft_engine is not None:
        return _unified_fft_engine.get_backend_info()
    return {"status": "engine_not_initialized"}
