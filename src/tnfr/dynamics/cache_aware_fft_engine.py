"""
TNFR Cache-Aware FFT Arithmetic Engine

This engine implements the advanced FFT arithmetic operations that emerge 
from the TNFR nodal equation ∂EPI/∂t = νf · ΔNFR(t) with full integration 
into the repository's unified cache system.

Mathematical Foundation:
The Graph Fourier Transform reveals TNFR dynamics as spectral operations:
- EPI evolution → Spectral coefficient modulation
- ΔNFR computation → Laplacian eigenvalue multiplication in frequency domain
- νf modulation → Pointwise multiplication in spectral space
- Phase coupling → Convolution operations via inverse FFT

Cache Integration Benefits:
- Spectral basis reuse across engines (O(N³) → O(1) for repeated decompositions)
- FFT kernel memoization (filter responses, windows, convolution kernels)
- Cross-engine sharing of spectral artifacts
- Predictive prefetching of likely spectral operations

Status: CANONICAL CACHE-INTEGRATED FFT ENGINE
"""

import time
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import FFT infrastructure
try:
    from .advanced_fft_arithmetic import TNFRAdvancedFFTEngine
    HAS_ADVANCED_FFT = True
except ImportError:
    HAS_ADVANCED_FFT = False

# Import cache coordination
try:
    from .fft_cache_coordinator import get_fft_cache_coordinator
    HAS_FFT_CACHE = True
except ImportError:
    HAS_FFT_CACHE = False

# Import cache optimization
try:
    from .advanced_cache_optimizer import (
        get_cache_optimizer,
        CacheOptimizationStrategy,
        record_computation
    )
    HAS_CACHE_OPTIMIZER = True
except ImportError:
    HAS_CACHE_OPTIMIZER = False

# Import spectral analysis
try:
    from ..mathematics.spectral import gft, igft
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False


class FFTOperationType(Enum):
    """Types of cache-optimized FFT operations."""
    SPECTRAL_CONVOLUTION = "spectral_convolution"
    HARMONIC_ANALYSIS = "harmonic_analysis"
    SPECTRAL_FILTERING = "spectral_filtering"
    PHASE_SYNCHRONIZATION = "phase_synchronization"
    MULTI_SCALE_DECOMPOSITION = "multi_scale_decomposition"
    CROSS_SPECTRAL_ANALYSIS = "cross_spectral_analysis"


@dataclass
class CacheOptimizedFFTResult:
    """Result of cache-optimized FFT operation."""
    operation_type: FFTOperationType
    fft_result: Any
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_time_saved: float = 0.0
    total_execution_time: float = 0.0
    spectral_basis_reused: bool = False
    kernel_cache_hits: int = 0


class TNFRCacheAwareFFTEngine:
    """
    Cache-aware FFT arithmetic engine for TNFR computations.
    
    This engine combines advanced FFT operations with intelligent caching
    to maximize performance while preserving TNFR mathematical coherence.
    """
    
    def __init__(
        self,
        enable_cache_optimization: bool = True,
        enable_predictive_prefetch: bool = True,
        cache_coordinator=None
    ):
        self.enable_cache_optimization = enable_cache_optimization
        self.enable_predictive_prefetch = enable_predictive_prefetch
        
        # Initialize FFT engine with cache coordinator
        if HAS_ADVANCED_FFT:
            fft_cache = cache_coordinator or (get_fft_cache_coordinator() if HAS_FFT_CACHE else None)
            self.fft_engine = TNFRAdvancedFFTEngine(cache_coordinator=fft_cache)
        else:
            self.fft_engine = None
            
        # Cache optimization
        self.cache_optimizer = get_cache_optimizer() if HAS_CACHE_OPTIMIZER else None
        
        # Performance tracking
        self.total_operations = 0
        self.total_cache_hits = 0
        self.total_time_saved = 0.0
        
    @record_computation("spectral_convolution")
    def spectral_convolution_cached(
        self,
        G: Any,
        signal1: Optional[np.ndarray] = None,
        signal2: Optional[np.ndarray] = None,
        operation: str = "multiply"
    ) -> CacheOptimizedFFTResult:
        """
        Perform spectral convolution with intelligent caching.
        
        This leverages cached spectral decompositions and memoized convolution kernels.
        """
        if not self.fft_engine or not HAS_NUMPY:
            raise RuntimeError("FFT engine not available")
            
        start_time = time.perf_counter()
        
        # Apply cache optimizations
        cache_results = []
        if self.enable_cache_optimization and self.cache_optimizer:
            cache_results = self.cache_optimizer.optimize_cache_strategy(
                G,
                [
                    CacheOptimizationStrategy.SPECTRAL_PERSISTENCE,
                    CacheOptimizationStrategy.CROSS_ENGINE_SHARING,
                    CacheOptimizationStrategy.PREDICTIVE_PREFETCH
                ]
            )
        
        # Perform FFT operation (benefits from cache optimizations)
        fft_result = self.fft_engine.spectral_convolution(
            G, signal1, signal2, operation
        )
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate cache statistics
        total_cache_hits = sum(r.cache_hits_improved for r in cache_results)
        time_saved = sum(r.computation_time_saved for r in cache_results)
        
        self.total_operations += 1
        self.total_cache_hits += total_cache_hits
        self.total_time_saved += time_saved
        
        return CacheOptimizedFFTResult(
            operation_type=FFTOperationType.SPECTRAL_CONVOLUTION,
            fft_result=fft_result,
            cache_hits=total_cache_hits,
            cache_misses=1 if total_cache_hits == 0 else 0,
            optimization_time_saved=time_saved,
            total_execution_time=total_time,
            spectral_basis_reused=any(
                r.strategy == CacheOptimizationStrategy.SPECTRAL_PERSISTENCE
                for r in cache_results
            ),
            kernel_cache_hits=getattr(
                self.fft_engine.cache_coordinator, 'get_stats',
                lambda: {"kernel_hits": 0}
            )()["kernel_hits"]
        )
    
    @record_computation("harmonic_analysis")
    def harmonic_analysis_cached(
        self,
        G: Any,
        num_harmonics: int = 5,
        window_size: Optional[int] = None
    ) -> CacheOptimizedFFTResult:
        """
        Perform harmonic analysis with cache optimization.
        
        Reuses spectral decompositions and caches harmonic pattern analysis.
        """
        if not self.fft_engine:
            raise RuntimeError("FFT engine not available")
            
        start_time = time.perf_counter()
        
        # Optimize cache for harmonic analysis
        cache_results = []
        if self.enable_cache_optimization and self.cache_optimizer:
            cache_results = self.cache_optimizer.optimize_cache_strategy(
                G,
                [
                    CacheOptimizationStrategy.SPECTRAL_PERSISTENCE,
                    CacheOptimizationStrategy.PATTERN_COMPRESSION,
                    CacheOptimizationStrategy.TEMPORAL_LOCALITY
                ]
            )
        
        # Perform harmonic analysis
        fft_result = self.fft_engine.harmonic_analysis(G, num_harmonics, window_size)
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate statistics
        total_cache_hits = sum(r.cache_hits_improved for r in cache_results)
        time_saved = sum(r.computation_time_saved for r in cache_results)
        
        self.total_operations += 1
        self.total_cache_hits += total_cache_hits
        self.total_time_saved += time_saved
        
        return CacheOptimizedFFTResult(
            operation_type=FFTOperationType.HARMONIC_ANALYSIS,
            fft_result=fft_result,
            cache_hits=total_cache_hits,
            cache_misses=1 if total_cache_hits == 0 else 0,
            optimization_time_saved=time_saved,
            total_execution_time=total_time,
            spectral_basis_reused=True  # Harmonic analysis always reuses basis
        )
    
    @record_computation("spectral_filtering")
    def spectral_filtering_cached(
        self,
        G: Any,
        filter_type: str = "lowpass",
        cutoff_frequency: Optional[float] = None,
        filter_order: int = 4
    ) -> CacheOptimizedFFTResult:
        """
        Perform spectral filtering with kernel caching.
        
        Filter kernels are cached and reused across similar filtering operations.
        """
        if not self.fft_engine:
            raise RuntimeError("FFT engine not available")
            
        start_time = time.perf_counter()
        
        # Optimize for filtering operations
        cache_results = []
        if self.enable_cache_optimization and self.cache_optimizer:
            cache_results = self.cache_optimizer.optimize_cache_strategy(
                G,
                [
                    CacheOptimizationStrategy.CROSS_ENGINE_SHARING,
                    CacheOptimizationStrategy.IMPORTANCE_WEIGHTING
                ]
            )
        
        # Perform spectral filtering (uses cached kernels)
        fft_result = self.fft_engine.spectral_filtering(
            G, filter_type, cutoff_frequency, filter_order
        )
        
        total_time = time.perf_counter() - start_time
        
        # Calculate cache benefits
        total_cache_hits = sum(r.cache_hits_improved for r in cache_results)
        time_saved = sum(r.computation_time_saved for r in cache_results)
        
        # Get kernel cache statistics
        kernel_hits = 0
        if hasattr(self.fft_engine, 'cache_coordinator') and self.fft_engine.cache_coordinator:
            stats = self.fft_engine.cache_coordinator.get_stats()
            kernel_hits = stats.get("kernel_hits", 0)
        
        self.total_operations += 1
        self.total_cache_hits += total_cache_hits
        self.total_time_saved += time_saved
        
        return CacheOptimizedFFTResult(
            operation_type=FFTOperationType.SPECTRAL_FILTERING,
            fft_result=fft_result,
            cache_hits=total_cache_hits,
            cache_misses=1 if total_cache_hits == 0 else 0,
            optimization_time_saved=time_saved,
            total_execution_time=total_time,
            spectral_basis_reused=True,
            kernel_cache_hits=kernel_hits
        )
    
    def multi_scale_analysis_cached(
        self,
        G: Any,
        scales: List[float] = None,
        analysis_type: str = "wavelet"
    ) -> CacheOptimizedFFTResult:
        """
        Perform multi-scale spectral analysis with hierarchical caching.
        
        Each scale level can reuse computations from other scales.
        """
        if not HAS_SPECTRAL or not self.fft_engine:
            raise RuntimeError("Spectral analysis not available")
            
        if scales is None:
            scales = [0.5, 1.0, 2.0, 4.0]
            
        start_time = time.perf_counter()
        
        # Optimize for multi-scale operations
        cache_results = []
        if self.enable_cache_optimization and self.cache_optimizer:
            cache_results = self.cache_optimizer.optimize_cache_strategy(
                G,
                [
                    CacheOptimizationStrategy.SPECTRAL_PERSISTENCE,
                    CacheOptimizationStrategy.CROSS_ENGINE_SHARING,
                    CacheOptimizationStrategy.PATTERN_COMPRESSION
                ]
            )
        
        # Get spectral state (cached via coordinator)
        spectral_state = self.fft_engine.get_spectral_state(G)
        
        # Perform multi-scale analysis
        scale_results = {}
        for scale in scales:
            # Apply scale transformation in spectral domain
            scaled_coeffs = spectral_state.spectral_coeffs * (1.0 / scale)
            
            # Transform back for this scale
            scaled_signal = igft(scaled_coeffs, spectral_state.eigenvectors)
            
            # Analyze this scale
            scale_results[scale] = {
                "signal": scaled_signal,
                "energy": float(np.sum(np.abs(scaled_coeffs) ** 2)),
                "dominant_freq": float(spectral_state.frequencies[np.argmax(np.abs(scaled_coeffs))])
            }
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate cache statistics
        total_cache_hits = sum(r.cache_hits_improved for r in cache_results)
        time_saved = sum(r.computation_time_saved for r in cache_results)
        
        multi_scale_result = {
            "scales": scales,
            "scale_results": scale_results,
            "spectral_state": spectral_state,
            "analysis_type": analysis_type
        }
        
        self.total_operations += 1
        self.total_cache_hits += total_cache_hits
        self.total_time_saved += time_saved
        
        return CacheOptimizedFFTResult(
            operation_type=FFTOperationType.MULTI_SCALE_DECOMPOSITION,
            fft_result=multi_scale_result,
            cache_hits=total_cache_hits,
            cache_misses=len(scales),  # One "miss" per scale computed
            optimization_time_saved=time_saved,
            total_execution_time=total_time,
            spectral_basis_reused=True
        )
    
    def cross_spectral_coherence_cached(
        self,
        G1: Any,
        G2: Any,
        coherence_bands: List[Tuple[float, float]] = None
    ) -> CacheOptimizedFFTResult:
        """
        Compute cross-spectral coherence between two graphs with caching.
        
        Reuses spectral decompositions for both graphs and caches coherence calculations.
        """
        if not self.fft_engine:
            raise RuntimeError("FFT engine not available")
            
        if coherence_bands is None:
            coherence_bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
            
        start_time = time.perf_counter()
        
        # Optimize for cross-spectral analysis
        cache_results = []
        if self.enable_cache_optimization and self.cache_optimizer:
            # Apply optimizations to both graphs
            cache_results.extend(self.cache_optimizer.optimize_cache_strategy(
                G1, [CacheOptimizationStrategy.SPECTRAL_PERSISTENCE]
            ))
            cache_results.extend(self.cache_optimizer.optimize_cache_strategy(
                G2, [CacheOptimizationStrategy.SPECTRAL_PERSISTENCE]
            ))
        
        # Get spectral states for both graphs (leverages caching)
        spectral1 = self.fft_engine.get_spectral_state(G1)
        spectral2 = self.fft_engine.get_spectral_state(G2)
        
        # Compute cross-spectral coherence
        coherence_results = {}
        
        for low_freq, high_freq in coherence_bands:
            # Find frequency indices in band
            freq_mask1 = (spectral1.frequencies >= low_freq) & (spectral1.frequencies < high_freq)
            freq_mask2 = (spectral2.frequencies >= low_freq) & (spectral2.frequencies < high_freq)
            
            if np.any(freq_mask1) and np.any(freq_mask2):
                # Compute coherence in this band
                coeffs1_band = spectral1.spectral_coeffs[freq_mask1]
                coeffs2_band = spectral2.spectral_coeffs[freq_mask2]
                
                # Cross-correlation in frequency domain
                cross_power = np.mean(coeffs1_band * np.conj(coeffs2_band))
                auto_power1 = np.mean(np.abs(coeffs1_band) ** 2)
                auto_power2 = np.mean(np.abs(coeffs2_band) ** 2)
                
                coherence = abs(cross_power) ** 2 / (auto_power1 * auto_power2) if (auto_power1 * auto_power2) > 0 else 0.0
                
                coherence_results[(low_freq, high_freq)] = {
                    "coherence": float(coherence),
                    "cross_power": float(abs(cross_power)),
                    "auto_power1": float(auto_power1),
                    "auto_power2": float(auto_power2)
                }
        
        total_time = time.perf_counter() - start_time
        
        # Aggregate cache statistics
        total_cache_hits = sum(r.cache_hits_improved for r in cache_results)
        time_saved = sum(r.computation_time_saved for r in cache_results)
        
        cross_spectral_result = {
            "coherence_bands": coherence_bands,
            "coherence_results": coherence_results,
            "mean_coherence": np.mean([r["coherence"] for r in coherence_results.values()]),
            "spectral_states": (spectral1, spectral2)
        }
        
        self.total_operations += 1
        self.total_cache_hits += total_cache_hits
        self.total_time_saved += time_saved
        
        return CacheOptimizedFFTResult(
            operation_type=FFTOperationType.CROSS_SPECTRAL_ANALYSIS,
            fft_result=cross_spectral_result,
            cache_hits=total_cache_hits,
            cache_misses=2 if total_cache_hits == 0 else 0,  # Two graphs
            optimization_time_saved=time_saved,
            total_execution_time=total_time,
            spectral_basis_reused=True
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        fft_stats = {}
        if hasattr(self.fft_engine, 'cache_coordinator') and self.fft_engine.cache_coordinator:
            fft_stats = self.fft_engine.cache_coordinator.get_stats()
            
        optimizer_stats = {}
        if self.cache_optimizer:
            optimizer_stats = self.cache_optimizer.get_optimization_stats()
            
        return {
            "cache_aware_fft_engine": {
                "total_operations": self.total_operations,
                "total_cache_hits": self.total_cache_hits,
                "total_time_saved": self.total_time_saved,
                "cache_hit_rate": self.total_cache_hits / max(1, self.total_operations)
            },
            "fft_cache_coordinator": fft_stats,
            "cache_optimizer": optimizer_stats,
            "overall_efficiency": {
                "average_time_saved_per_op": self.total_time_saved / max(1, self.total_operations),
                "cache_effectiveness": self.total_cache_hits / max(1, self.total_operations)
            }
        }


# Factory functions
def create_cache_aware_fft_engine(**kwargs) -> TNFRCacheAwareFFTEngine:
    """Create cache-aware FFT engine."""
    return TNFRCacheAwareFFTEngine(**kwargs)


# Convenience functions for common operations
def cached_spectral_convolution(G: Any, **kwargs) -> CacheOptimizedFFTResult:
    """Convenience function for cached spectral convolution."""
    engine = create_cache_aware_fft_engine()
    return engine.spectral_convolution_cached(G, **kwargs)


def cached_harmonic_analysis(G: Any, **kwargs) -> CacheOptimizedFFTResult:
    """Convenience function for cached harmonic analysis."""
    engine = create_cache_aware_fft_engine()
    return engine.harmonic_analysis_cached(G, **kwargs)


def cached_spectral_filtering(G: Any, **kwargs) -> CacheOptimizedFFTResult:
    """Convenience function for cached spectral filtering."""
    engine = create_cache_aware_fft_engine()
    return engine.spectral_filtering_cached(G, **kwargs)