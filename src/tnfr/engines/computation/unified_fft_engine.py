"""TNFR Unified FFT Engine - Single Access Point for All Spectral Operations.

CONSOLIDATION ACHIEVEMENT: This module unifies all TNFR FFT implementations
under a single coherent interface following nodal equation dynamics principles.

Unified Architecture:
- Single entry point for all FFT operations across TNFR
- Intelligent backend selection (GPU, CPU, distributed)
- Automatic performance optimization and caching
- Consistent spectral arithmetic interface
- Unified error handling and telemetry

Theoretical Foundation:
The nodal equation ∂EPI/∂t = νf · ΔNFR(t) exhibits natural spectral structure
when transformed to frequency domain, enabling fast convolution-based computation
of structural dynamics via Graph Fourier Transform.

Mathematical Operations:
1. Spectral Convolution: Fast ΔNFR computation O(N log N)
2. Harmonic Analysis: Multi-scale resonance detection
3. Coherence Spectroscopy: Phase relationship analysis
4. Adaptive Filtering: Noise reduction via spectral masks
5. Cross-Spectral Analysis: Multi-graph coherence measurement

Backend Routing:
- Advanced operations → advanced_fft_arithmetic.py
- Distributed workloads → distributed_fft.py
- Basic transforms → fft_backend.py
- GPU acceleration → unified_gpu_manager.py integration

Performance Benefits:
- Eliminates redundant FFT engine instantiation
- Unified caching across all spectral operations
- Automatic precision and backend optimization
- Consistent memory management via GPU manager

Status: UNIFIED FFT CONSOLIDATION - All spectral operations centralized
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...config import get_config

# Import existing FFT components for consolidation
from ...dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from ...dynamics.distributed_fft import DistributedFFTEngine
from ...dynamics.fft_backend import FFTBackendCapabilities
from ...errors import TNFRValueError
from ...mathematics.unified_numerical import np
from .unified_gpu_system import get_unified_gpu_system

logger = logging.getLogger(__name__)


@dataclass
class UnifiedFFTConfig:
    """Configuration for unified FFT engine."""

    # Backend selection
    preferred_backend: str = "advanced"  # "advanced", "distributed", "basic"
    auto_backend_selection: bool = True
    enable_gpu_acceleration: bool = True

    # Performance tuning
    cache_spectral_decompositions: bool = True
    max_cache_size_mb: int = 512
    enable_precision_scaling: bool = True

    # Distributed settings
    distribute_threshold_nodes: int = 10000
    max_workers: int = 4

    # Quality settings
    spectral_precision: str = "float64"
    convergence_tolerance: float = 1e-12

    # Debug and telemetry
    profile_operations: bool = False
    log_backend_selection: bool = True


@dataclass
class UnifiedFFTResult:
    """Unified result container for all FFT operations."""

    # Core results
    spectral_data: np.ndarray
    frequencies: np.ndarray
    backend_used: str

    # Performance metrics
    computation_time_ms: float
    memory_usage_mb: float
    cache_hit: bool = False

    # Quality indicators
    spectral_precision: str = "float64"
    convergence_achieved: bool = True

    # Advanced results (optional)
    harmonic_analysis: dict[str, Any] | None = None
    coherence_matrix: np.ndarray | None = None
    phase_relationships: dict[str, float] | None = None

    # Telemetry
    operation_metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedFFTBackend(Protocol):
    """Protocol for unified FFT backend implementations."""

    def get_capabilities(self) -> FFTBackendCapabilities:
        """Return backend capabilities and limits."""
        ...

    def compute_fft(self, data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
        """Compute FFT using this backend."""
        ...

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute spectral convolution efficiently."""
        ...


class TNFRUnifiedFFTEngine:
    """Unified FFT Engine - Single Access Point for All TNFR Spectral Operations.

    ARCHITECTURE: This engine consolidates all TNFR FFT implementations under
    a unified interface with intelligent backend routing and performance optimization.

    Usage:
        # Single entry point for all FFT operations
        engine = TNFRUnifiedFFTEngine()

        # Automatic backend selection
        result = engine.compute_fft(data)

        # Advanced spectral analysis
        result = engine.compute_harmonic_analysis(epi_data, frequencies)

        # Multi-graph coherence
        result = engine.compute_cross_spectral_coherence(graph1, graph2)

    Benefits:
        - Eliminates FFT backend redundancy across codebase
        - Unified caching and performance optimization
        - Consistent error handling and telemetry
        - Automatic GPU/CPU backend selection
        - Integrated with unified config system
    """

    def __init__(self, config: UnifiedFFTConfig | None = None):
        """Initialize unified FFT engine with configuration."""
        self.config = config or UnifiedFFTConfig()

        # Initialize GPU manager for acceleration
        self.gpu_manager = get_unified_gpu_system()

        # Initialize backend engines
        self._advanced_engine: TNFRAdvancedFFTEngine | None = None
        self._distributed_engine: DistributedFFTEngine | None = None
        self._basic_backends: dict[str, UnifiedFFTBackend] = {}

        # Caching system
        self._spectral_cache: dict[str, UnifiedFFTResult] = {}
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}

        # Get global config integration
        self.global_config = get_config()

        if self.config.log_backend_selection:
            logger.info(f"Initialized unified FFT engine with config: {self.config}")

    def _get_cache_key(self, data: np.ndarray, operation: str, **kwargs: Any) -> str:
        """Generate cache key for spectral operations."""
        # Create deterministic hash from data and parameters
        data_hash = hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()[:16]
        params_str = ",".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return f"{operation}:{data_hash}:{params_str}"

    def _check_cache(self, cache_key: str) -> UnifiedFFTResult | None:
        """Check spectral operation cache."""
        if not self.config.cache_spectral_decompositions:
            return None

        if cache_key in self._spectral_cache:
            self._cache_stats["hits"] += 1
            result = self._spectral_cache[cache_key]
            result.cache_hit = True
            return result

        self._cache_stats["misses"] += 1
        return None

    def _store_cache(self, cache_key: str, result: UnifiedFFTResult) -> None:
        """Store result in spectral cache with size management."""
        if not self.config.cache_spectral_decompositions:
            return

        # Simple cache size management (evict oldest on overflow)
        max_entries = (
            self.config.max_cache_size_mb * 1024 * 1024 // (8 * 1024)
        )  # Rough estimate

        if len(self._spectral_cache) >= max_entries:
            # Remove oldest entry
            oldest_key = next(iter(self._spectral_cache))
            del self._spectral_cache[oldest_key]
            self._cache_stats["evictions"] += 1

        self._spectral_cache[cache_key] = result

    def _select_backend(self, data_shape: tuple[int, ...], operation: str) -> str:
        """Intelligent backend selection based on data and operation."""
        if not self.config.auto_backend_selection:
            return self.config.preferred_backend

        n_elements = np.prod(data_shape)

        # Use distributed backend for large workloads
        if n_elements > self.config.distribute_threshold_nodes:
            return "distributed"

        # Use advanced backend for moderate workloads with GPU
        if (
            n_elements > 1000
            and self.config.enable_gpu_acceleration
            and self.gpu_manager.has_gpu_backend()
        ):
            return "advanced"

        # Use basic backend for simple operations
        return "basic"

    def _get_backend_engine(self, backend_name: str) -> UnifiedFFTBackend:
        """Get or create backend engine."""
        if backend_name == "advanced":
            if self._advanced_engine is None:
                self._advanced_engine = TNFRAdvancedFFTEngine()
            return self._advanced_engine

        elif backend_name == "distributed":
            if self._distributed_engine is None:
                self._distributed_engine = DistributedFFTEngine()
            return self._distributed_engine

        elif backend_name == "basic":
            if "basic" not in self._basic_backends:
                # Create basic NumPy-based backend
                self._basic_backends["basic"] = _BasicNumpyFFTBackend()
            return self._basic_backends["basic"]

        else:
            raise TNFRValueError(
                f"Unknown FFT backend: {backend_name}",
                context={
                    "requested": backend_name,
                    "available": ["advanced", "distributed", "basic"],
                },
                suggestion="Use 'advanced', 'distributed', or 'basic'.",
            )

    def compute_fft(
        self, data: np.ndarray, backend: str | None = None, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute FFT with automatic backend selection and caching.

        Parameters
        ----------
        data : np.ndarray
            Input data for FFT computation
        backend : str, optional
            Force specific backend ("advanced", "distributed", "basic")
        **kwargs
            Additional parameters for FFT computation

        Returns
        -------
        UnifiedFFTResult
            Unified FFT result with performance metrics and metadata
        """
        import time

        # Generate cache key
        cache_key = self._get_cache_key(data, "fft", **kwargs)

        # Check cache first
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Select backend
        selected_backend = backend or self._select_backend(data.shape, "fft")

        # Get backend engine
        engine = self._get_backend_engine(selected_backend)

        # Execute FFT with timing
        start_time = time.perf_counter()

        try:
            # Use GPU manager for acceleration if available
            if self.config.enable_gpu_acceleration and selected_backend != "basic":
                result = self.gpu_manager.execute_with_gpu_fallback(
                    lambda: engine.compute_fft(data, **kwargs),
                    fallback=lambda: self._get_backend_engine("basic").compute_fft(
                        data, **kwargs
                    ),
                )
            else:
                result = engine.compute_fft(data, **kwargs)

            # Update timing
            result.computation_time_ms = (time.perf_counter() - start_time) * 1000
            result.backend_used = selected_backend

            # Store in cache
            self._store_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"FFT computation failed with backend {selected_backend}: {e}")
            # Fallback to basic backend
            if selected_backend != "basic":
                return self.compute_fft(data, backend="basic", **kwargs)
            raise

    def compute_harmonic_analysis(
        self, epi_data: np.ndarray, frequencies: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute harmonic analysis of EPI structural evolution."""
        # Use advanced engine for harmonic analysis
        engine = self._get_backend_engine("advanced")

        if hasattr(engine, "compute_harmonic_analysis"):
            return engine.compute_harmonic_analysis(epi_data, frequencies, **kwargs)
        else:
            # Fallback: compute FFT and extract harmonics
            fft_result = self.compute_fft(epi_data, **kwargs)
            # Add harmonic analysis to result
            fft_result.harmonic_analysis = self._extract_harmonics(
                fft_result.spectral_data, frequencies
            )
            return fft_result

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute spectral convolution for ΔNFR operations."""
        # Select backend based on signal size
        backend = self._select_backend(signal1.shape, "convolution")
        engine = self._get_backend_engine(backend)

        return engine.compute_spectral_convolution(signal1, signal2, **kwargs)

    def compute_cross_spectral_coherence(
        self, graph1_data: np.ndarray, graph2_data: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute cross-spectral coherence between graph structures."""
        # Advanced cross-spectral analysis
        engine = self._get_backend_engine("advanced")

        # Compute individual FFTs
        fft1 = self.compute_fft(graph1_data, **kwargs)
        fft2 = self.compute_fft(graph2_data, **kwargs)

        # Compute cross-spectral coherence
        coherence_matrix = self._compute_coherence_matrix(
            fft1.spectral_data, fft2.spectral_data
        )

        # Create unified result
        result = UnifiedFFTResult(
            spectral_data=coherence_matrix,
            frequencies=fft1.frequencies,
            backend_used="advanced",
            computation_time_ms=fft1.computation_time_ms + fft2.computation_time_ms,
            memory_usage_mb=fft1.memory_usage_mb + fft2.memory_usage_mb,
            coherence_matrix=coherence_matrix,
            operation_metadata={"operation": "cross_spectral_coherence"},
        )

        return result

    def _extract_harmonics(
        self, spectral_data: np.ndarray, frequencies: np.ndarray
    ) -> dict[str, Any]:
        """Extract harmonic components from spectral data."""
        # Find peak frequencies
        magnitudes = np.abs(spectral_data)
        peak_indices = np.argsort(magnitudes)[-10:]  # Top 10 peaks

        harmonics = {
            "fundamental_freq": frequencies[peak_indices[-1]],
            "harmonic_freqs": frequencies[peak_indices].tolist(),
            "harmonic_amplitudes": magnitudes[peak_indices].tolist(),
            "total_harmonic_distortion": np.sum(magnitudes[peak_indices[:-1]])
            / magnitudes[peak_indices[-1]],
        }

        return harmonics

    def _compute_coherence_matrix(
        self, fft1: np.ndarray, fft2: np.ndarray
    ) -> np.ndarray:
        """Compute coherence matrix between two spectral signals."""
        # Cross-power spectral density
        cross_psd = fft1 * np.conj(fft2)

        # Auto-power spectral densities
        psd1 = fft1 * np.conj(fft1)
        psd2 = fft2 * np.conj(fft2)

        # Coherence = |cross_psd|^2 / (psd1 * psd2)
        coherence = np.abs(cross_psd) ** 2 / (np.abs(psd1) * np.abs(psd2) + 1e-12)

        return coherence

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get FFT cache performance statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (
            (self._cache_stats["hits"] / total_requests * 100.0)
            if total_requests > 0
            else 0.0
        )

        return {
            **self._cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._spectral_cache),
            "cache_memory_estimate_mb": len(self._spectral_cache)
            * 0.1,  # Rough estimate
        }

    def clear_cache(self) -> None:
        """Clear spectral operation cache."""
        self._spectral_cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("Cleared unified FFT cache")

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about available backends and their capabilities."""
        backends = {}

        for backend_name in ["advanced", "distributed", "basic"]:
            try:
                engine = self._get_backend_engine(backend_name)
                if hasattr(engine, "get_capabilities"):
                    backends[backend_name] = engine.get_capabilities().__dict__
                else:
                    backends[backend_name] = {
                        "status": "available",
                        "capabilities": "unknown",
                    }
            except Exception as e:
                backends[backend_name] = {"status": "unavailable", "error": str(e)}

        return {
            "backends": backends,
            "config": self.config.__dict__,
            "gpu_available": self.gpu_manager.has_gpu_backend(),
            "cache_stats": self.get_cache_statistics(),
        }


class _BasicNumpyFFTBackend:
    """Basic NumPy-based FFT backend for fallback operations."""

    def get_capabilities(self) -> FFTBackendCapabilities:
        """Return basic NumPy backend capabilities."""
        return FFTBackendCapabilities(
            backend_name="numpy_basic",
            max_nodes=None,
            precision="float64",
            supports_distributed=False,
            extra={"library": "numpy.fft"},
        )

    def compute_fft(self, data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
        """Compute FFT using NumPy."""
        import time

        start_time = time.perf_counter()

        # Compute FFT
        spectral_data = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data))

        computation_time = (time.perf_counter() - start_time) * 1000

        return UnifiedFFTResult(
            spectral_data=spectral_data,
            frequencies=frequencies,
            backend_used="basic",
            computation_time_ms=computation_time,
            memory_usage_mb=data.nbytes / (1024 * 1024),
            spectral_precision="float64",
            operation_metadata={"backend": "numpy", "algorithm": "fft"},
        )

    def compute_spectral_convolution(
        self, signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
    ) -> UnifiedFFTResult:
        """Compute spectral convolution using NumPy."""
        import time

        start_time = time.perf_counter()

        # Compute convolution via FFT
        fft1 = np.fft.fft(signal1)
        fft2 = np.fft.fft(signal2)
        convolution = np.fft.ifft(fft1 * fft2)
        frequencies = np.fft.fftfreq(len(signal1))

        computation_time = (time.perf_counter() - start_time) * 1000

        return UnifiedFFTResult(
            spectral_data=convolution,
            frequencies=frequencies,
            backend_used="basic",
            computation_time_ms=computation_time,
            memory_usage_mb=(signal1.nbytes + signal2.nbytes) / (1024 * 1024),
            spectral_precision="float64",
            operation_metadata={
                "backend": "numpy",
                "algorithm": "spectral_convolution",
            },
        )


# ============================================================================
# PUBLIC API - Unified FFT Interface
# ============================================================================

# Global unified FFT engine instance
_unified_fft_engine: TNFRUnifiedFFTEngine | None = None


def get_unified_fft_engine(
    config: UnifiedFFTConfig | None = None,
) -> TNFRUnifiedFFTEngine:
    """Get or create global unified FFT engine.

    This provides a singleton interface for all TNFR FFT operations
    to eliminate redundant engine creation across modules.

    Parameters
    ----------
    config : UnifiedFFTConfig, optional
        Configuration for engine (only used on first call)

    Returns
    -------
    TNFRUnifiedFFTEngine
        Global unified FFT engine instance
    """
    global _unified_fft_engine

    if _unified_fft_engine is None:
        _unified_fft_engine = TNFRUnifiedFFTEngine(config)
        logger.info("Created global unified FFT engine")

    return _unified_fft_engine


# Convenience functions for direct FFT operations
def compute_unified_fft(data: np.ndarray, **kwargs: Any) -> UnifiedFFTResult:
    """Compute FFT using unified engine - convenience function."""
    return get_unified_fft_engine().compute_fft(data, **kwargs)


def compute_unified_spectral_convolution(
    signal1: np.ndarray, signal2: np.ndarray, **kwargs: Any
) -> UnifiedFFTResult:
    """Compute spectral convolution using unified engine - convenience function."""
    return get_unified_fft_engine().compute_spectral_convolution(
        signal1, signal2, **kwargs
    )


def clear_unified_fft_cache() -> None:
    """Clear unified FFT cache - convenience function."""
    if _unified_fft_engine is not None:
        _unified_fft_engine.clear_cache()


def get_unified_fft_stats() -> dict[str, Any]:
    """Get unified FFT engine statistics - convenience function."""
    if _unified_fft_engine is not None:
        return _unified_fft_engine.get_backend_info()
    return {"status": "engine_not_initialized"}
