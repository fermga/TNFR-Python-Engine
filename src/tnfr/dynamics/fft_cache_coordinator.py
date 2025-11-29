"""TNFR FFT cache coordination utilities.

This module centralizes FFT arithmetic caching so spectral computations reuse
repo-wide cache layers and stay aligned with the nodal equation
∂EPI/∂t = νf · ΔNFR(t). It bridges:

- Repo hierarchical cache (``tnfr.utils.cache``)
- Unified multi-modal cache (spectral ↔ structural ↔ operator data)
- Structural coherence cache (topology fingerprints)

Capabilities
------------
* Spectral basis reuse keyed by graph signature
* FFT kernel memoization (convolution/filter windows)
* Registration of FFT arithmetic outputs for later reuse
* Telemetry on cache efficiency for diagnostics

The coordinator is intentionally lightweight so engines (FFT dynamics,
advanced FFT arithmetic, emergent optimizers) can share cached artifacts
without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import time

import numpy as np

try:  # Optional dependency
    import networkx as nx  # noqa: F401
    HAS_NETWORKX = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HAS_NETWORKX = False

try:
    from ..mathematics.spectral import get_laplacian_spectrum
    HAS_SPECTRAL = True
except ImportError:  # pragma: no cover - handled upstream
    HAS_SPECTRAL = False

try:
    from ..utils.cache import cache_tnfr_computation, CacheLevel
    _CORE_CACHE_AVAILABLE = True
except ImportError:  # pragma: no cover - cache infra optional in some builds
    _CORE_CACHE_AVAILABLE = False

# PHASE 6 EXTENDED: Canonical constants for FFT cache coordination
from ..constants.canonical import (
    PI,                                    # π ≈ 3.1416 (3.0 → canonical)
    OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,   # e·φ/π ≈ 1.4048 (1.5 → canonical)
    FFT_OPT_SEQUENTIAL_IMPROVEMENT_CANONICAL,  # φ·γ/(π·e) ≈ 0.1095 (mathematical importance scaling)
)

try:
    from .multi_modal_cache import get_unified_cache, CacheEntryType
    HAS_UNIFIED_CACHE = True
except ImportError:  # pragma: no cover
    HAS_UNIFIED_CACHE = False
    CacheEntryType = None  # type: ignore

try:
    from .structural_cache import get_structural_cache
    HAS_STRUCTURAL_CACHE = True
except ImportError:  # pragma: no cover
    HAS_STRUCTURAL_CACHE = False


@dataclass
class SpectralBasis:
    """Cached Laplacian eigensystem."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    signature: str
    computed_at: float


@dataclass
class FFTCacheStats:
    """Telemetry counters exposed to higher-level engines."""

    spectral_requests: int = 0
    spectral_hits: int = 0
    spectral_misses: int = 0
    kernel_hits: int = 0
    kernel_misses: int = 0
    registered_results: int = 0


class FFTCacheCoordinator:
    """Bridge between FFT arithmetic and repo-wide cache subsystems."""

    def __init__(self) -> None:
        self._spectral_cache: Dict[str, SpectralBasis] = {}
        self._kernel_cache: Dict[str, Any] = {}
        self._stats = FFTCacheStats()
        self._unified_cache = get_unified_cache() if HAS_UNIFIED_CACHE else None
        self._structural_cache = (
            get_structural_cache() if HAS_STRUCTURAL_CACHE else None
        )

    # ------------------------------------------------------------------
    # Spectral basis management
    # ------------------------------------------------------------------
    def get_spectral_basis(
        self,
        G: Any,
        *,
        force_recompute: bool = False,
    ) -> SpectralBasis:
        """Return (and cache) Laplacian eigensystem for *G*."""

        if not HAS_SPECTRAL or G is None:
            raise ValueError("Spectral analysis unavailable in current environment")

        signature = self._graph_signature(G)
        self._stats.spectral_requests += 1

        if not force_recompute and signature in self._spectral_cache:
            self._stats.spectral_hits += 1
            return self._spectral_cache[signature]

        self._stats.spectral_misses += 1
        eigenvalues, eigenvectors = self._load_spectrum_with_repo_cache(G, signature)

        # Also register inside unified cache so other engines can reuse instantly
        if self._unified_cache is not None:
            self._unified_cache.get(
                CacheEntryType.SPECTRAL_DECOMPOSITION,
                G,
                parameters={"signature": signature},
                computation_func=lambda: (eigenvalues, eigenvectors),
                mathematical_importance=PI,
            )

        basis = SpectralBasis(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            signature=signature,
            computed_at=time.time(),
        )
        self._spectral_cache[signature] = basis
        return basis

    def register_spectral_state(self, G: Any, spectral_state: Any) -> None:
        """Optional helper for storing richer spectral states in unified cache."""

        if self._unified_cache is None or spectral_state is None:
            return

        self._unified_cache.get(
            CacheEntryType.NODAL_STATE,
            G,
            parameters={"kind": "spectral_state"},
            computation_func=lambda: spectral_state,
            mathematical_importance=OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,
        )

    # ------------------------------------------------------------------
    # FFT kernel + result registration
    # ------------------------------------------------------------------
    def get_kernel(
        self,
        G: Any,
        kernel_name: str,
        builder: Callable[[], Any],
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return cached FFT kernel (window, filter, etc.)."""

        signature = self._graph_signature(G)
        params_str = self._serialize_params(kernel_params)
        cache_key = f"{signature}:{kernel_name}:{params_str}"

        if cache_key in self._kernel_cache:
            self._stats.kernel_hits += 1
            return self._kernel_cache[cache_key]

        kernel = builder()
        self._kernel_cache[cache_key] = kernel
        self._stats.kernel_misses += 1

        if self._unified_cache is not None:
            combined_params = {"kernel": kernel_name, **(kernel_params or {})}
            self._unified_cache.get(
                CacheEntryType.FFT_OPERATION,
                G,
                parameters=combined_params,
                computation_func=lambda: kernel,
                mathematical_importance=OPT_ORCH_ARITHMETIC_BOOST_CANONICAL,
            )

        return kernel

    def register_fft_result(
        self,
        G: Any,
        fft_result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register FFT arithmetic outputs for cross-engine reuse."""

        self._stats.registered_results += 1

        if self._unified_cache is None:
            return

        params = {"operation": getattr(fft_result, "operation", "fft")}
        if metadata:
            params.update(metadata)

        self._unified_cache.get(
            CacheEntryType.FFT_OPERATION,
            G,
            parameters=params,
            computation_func=lambda: fft_result,
            mathematical_importance=FFT_OPT_SEQUENTIAL_IMPROVEMENT_CANONICAL,
        )

    # ------------------------------------------------------------------
    # Diagnostics / maintenance
    # ------------------------------------------------------------------
    def reset_local(self) -> None:
        """Clear local (in-process) caches without touching global layers."""

        self._spectral_cache.clear()
        self._kernel_cache.clear()
        self._stats = FFTCacheStats()

    def get_stats(self) -> Dict[str, int]:
        """Expose telemetry counters for testing and monitoring."""

        return {
            "spectral_requests": self._stats.spectral_requests,
            "spectral_hits": self._stats.spectral_hits,
            "spectral_misses": self._stats.spectral_misses,
            "kernel_hits": self._stats.kernel_hits,
            "kernel_misses": self._stats.kernel_misses,
            "registered_results": self._stats.registered_results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _graph_signature(self, G: Any) -> str:
        if self._unified_cache is not None:
            return self._unified_cache.compute_graph_signature(G)
        if self._structural_cache is not None:
            return self._structural_cache.get_topology_hash(G)
        return f"graph_{id(G)}"

    def _serialize_params(self, params: Optional[Dict[str, Any]]) -> str:
        if not params:
            return "default"
        return "|".join(f"{k}={v}" for k, v in sorted(params.items()))

    def _load_spectrum_with_repo_cache(
        self,
        G: Any,
        signature: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_SPECTRAL:
            raise RuntimeError("Spectral backends unavailable")
        return self._compute_spectrum_repo_cached(G, graph_signature=signature)

    def _compute_spectrum_repo_cached(
        self,
        G: Any,
        graph_signature: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del graph_signature  # Stable cache key already encodes signature
        return get_laplacian_spectrum(G)


_global_fft_cache: Optional[FFTCacheCoordinator] = None


def get_fft_cache_coordinator() -> FFTCacheCoordinator:
    """Return process-wide FFT cache coordinator instance."""

    global _global_fft_cache
    if _global_fft_cache is None:
        _global_fft_cache = FFTCacheCoordinator()
    return _global_fft_cache


if _CORE_CACHE_AVAILABLE:
    FFTCacheCoordinator._compute_spectrum_repo_cached = cache_tnfr_computation(  # type: ignore[attr-defined]
        level=CacheLevel.GRAPH_STRUCTURE,
        dependencies={"graph_topology"},
    )(FFTCacheCoordinator._compute_spectrum_repo_cached)  # type: ignore[attr-defined]
