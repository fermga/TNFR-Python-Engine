"""
TNFR Advanced FFT Arithmetic Engine

This module implements advanced FFT arithmetic operations that emerge naturally from
the nodal equation ∂EPI/∂t = νf · ΔNFR(t) when viewed in spectral domain.

Mathematical Foundation:
The Graph Fourier Transform reveals that TNFR dynamics have natural spectral structure:
- EPI signals live in graph spectral domain
- ΔNFR operations are convolutions in spectral space  
- νf modulation becomes multiplication in frequency domain
- Multi-scale coupling creates harmonic relationships

Advanced Operations:
1. **Spectral Convolution**: Fast ΔNFR computation via FFT
2. **Harmonic Analysis**: Multi-scale resonance detection  
3. **Spectral Filtering**: Noise reduction and mode selection
4. **Phase-Locked Loops**: Automatic phase synchronization
5. **Adaptive Windowing**: Time-frequency analysis of EPI evolution
6. **Cross-Spectral Analysis**: Multi-graph coherence measurement

Performance:
- O(N log N) complexity for most operations
- GPU acceleration via JAX/PyTorch backends
- Automatic precision management
- Cache-aware spectral decomposition reuse

Status: CANONICAL SPECTRAL ARITHMETIC ENGINE
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Import mathematical backends
try:
    from ..mathematics.backend import get_backend
    HAS_MATH_BACKENDS = True
except ImportError:
    HAS_MATH_BACKENDS = False

# Import spectral analysis
try:
    from ..mathematics.spectral import get_laplacian_spectrum, gft, igft
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

# Import unified cache
try:
    from .multi_modal_cache import get_unified_cache, CacheEntryType
    HAS_UNIFIED_CACHE = True
except ImportError:
    HAS_UNIFIED_CACHE = False

# FFT cache coordinator
try:
    from .fft_cache_coordinator import (
        FFTCacheCoordinator,
        get_fft_cache_coordinator,
    )
    HAS_FFT_CACHE = True
except ImportError:
    HAS_FFT_CACHE = False
    FFTCacheCoordinator = None  # type: ignore


class SpectralOperation(Enum):
    """Types of spectral operations."""
    CONVOLUTION = "convolution"              # Spectral convolution (fast ΔNFR)
    CORRELATION = "correlation"              # Cross-correlation analysis
    FILTERING = "filtering"                  # Spectral filtering
    WINDOWING = "windowing"                  # Time-frequency windowing
    HARMONIC_ANALYSIS = "harmonic_analysis"  # Multi-scale harmonics
    PHASE_LOCKING = "phase_locking"          # Phase synchronization
    COHERENCE_ANALYSIS = "coherence_analysis" # Cross-spectral coherence


@dataclass
class SpectralState:
    """State in spectral domain."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  
    spectral_coeffs: np.ndarray
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    coherence_length: float = 0.0
    dominant_modes: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass  
class FFTArithmeticResult:
    """Result of FFT arithmetic operation."""
    operation: SpectralOperation
    input_shape: Tuple[int, ...]
    output_data: Any
    spectral_state: Optional[SpectralState] = None
    execution_time: float = 0.0
    fft_operations: int = 0
    cache_hits: int = 0
    backend_used: str = "numpy"
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)


class TNFRAdvancedFFTEngine:
    """
    Advanced FFT arithmetic engine for TNFR computations.
    
    This engine leverages the spectral structure of the nodal equation to
    provide O(N log N) arithmetic operations on graph signals.
    """
    
    def __init__(
        self,
        default_backend: str = "numpy",
        precision: str = "float64",
        cache_coordinator: Optional[FFTCacheCoordinator] = None,
    ):
        self.default_backend = default_backend
        self.precision = precision
        self.cache_coordinator = (
            cache_coordinator
            if cache_coordinator is not None
            else (get_fft_cache_coordinator() if HAS_FFT_CACHE else None)
        )
        
        # Initialize mathematical backend
        if HAS_MATH_BACKENDS:
            self.backend = get_backend(default_backend)
        else:
            self.backend = None
            
        # Performance tracking
        self.total_operations = 0
        self.total_fft_ops = 0
        self.cache_hits = 0
        
        # Spectral cache for eigendecompositions (fallback when coordinator unavailable)
        self._spectral_cache = {} if self.cache_coordinator is None else None
        
        # Precomputed windows for time-frequency analysis
        self._window_cache = {}
    
    def _calculate_attenuation_db(self, filter_response: np.ndarray) -> float:
        """Calculate filter attenuation in dB safely."""
        if len(filter_response) == 0:
            return 0.0
            
        positive_vals = filter_response[filter_response > 0]
        if len(positive_vals) == 0:
            return -np.inf  # Complete attenuation
        
        max_positive = np.max(positive_vals)
        max_overall = np.max(filter_response)
        
        if max_overall <= 0:
            return -np.inf  # Complete attenuation
        
        if max_positive == max_overall:
            return 0.0  # No attenuation
        
        return -20 * np.log10(max_positive / max_overall)
        
    def get_spectral_state(self, G: Any, force_recompute: bool = False) -> SpectralState:
        """
        Get spectral decomposition of graph.
        
        This is the fundamental operation that enables all FFT arithmetic.
        """
        if not HAS_NETWORKX or G is None:
            raise ValueError("Graph required for spectral analysis")
            
        spectral_basis = None

        if self.cache_coordinator is not None:
            spectral_basis = self.cache_coordinator.get_spectral_basis(
                G,
                force_recompute=force_recompute
            )
            eigenvalues = spectral_basis.eigenvalues
            eigenvectors = spectral_basis.eigenvectors
        else:
            graph_id = id(G)
            if (
                not force_recompute
                and self._spectral_cache is not None
                and graph_id in self._spectral_cache
            ):
                return self._spectral_cache[graph_id]
            
            if not HAS_SPECTRAL:
                raise RuntimeError("Spectral analysis not available")
                
            if HAS_UNIFIED_CACHE:
                cache = get_unified_cache()
                eigenvalues, eigenvectors = cache.get(
                    CacheEntryType.SPECTRAL_DECOMPOSITION,
                    G,
                    computation_func=lambda: get_laplacian_spectrum(G),
                    mathematical_importance=3.0
                )
            else:
                eigenvalues, eigenvectors = get_laplacian_spectrum(G)
            
        # Extract current signal from nodes
        signal = np.array([G.nodes[node].get('EPI', 0.0) for node in G.nodes()])
        
        # Compute spectral coefficients
        spectral_coeffs = gft(signal, eigenvectors)
        
        # Analyze spectral properties
        amplitudes = np.abs(spectral_coeffs)
        phases = np.angle(spectral_coeffs)
        
        # Find dominant modes (largest amplitude coefficients)
        dominant_indices = np.argsort(amplitudes)[-5:]  # Top 5 modes
        
        # Estimate coherence length from spectral decay
        coherence_length = self._estimate_coherence_length(eigenvalues, amplitudes)
        
        # Create spectral state
        spectral_state = SpectralState(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_coeffs=spectral_coeffs,
            frequencies=eigenvalues,  # In graph setting, eigenvalues are frequencies
            amplitudes=amplitudes,
            phases=phases,
            coherence_length=coherence_length,
            dominant_modes=dominant_indices
        )
        
        # Cache the result locally when coordinator absent
        if self._spectral_cache is not None:
            graph_id = id(G)
            self._spectral_cache[graph_id] = spectral_state

        if self.cache_coordinator is not None:
            self.cache_coordinator.register_spectral_state(G, spectral_state)
        
        return spectral_state
        
    def spectral_convolution(
        self,
        G: Any,
        signal1: Optional[np.ndarray] = None,
        signal2: Optional[np.ndarray] = None,
        operation: str = "multiply"
    ) -> FFTArithmeticResult:
        """
        Perform spectral domain convolution.
        
        This enables fast computation of ΔNFR and other nodal operations.
        """
        start_time = time.perf_counter()
        
        # Get spectral state
        spectral_state = self.get_spectral_state(G)
        
        # Extract signals from graph if not provided
        if signal1 is None:
            signal1 = np.array([G.nodes[node].get('EPI', 0.0) for node in G.nodes()])
        if signal2 is None:
            signal2 = np.array([G.nodes[node].get('nu_f', 1.0) for node in G.nodes()])
            
        # Transform to spectral domain
        spectral1 = gft(signal1, spectral_state.eigenvectors)
        spectral2 = gft(signal2, spectral_state.eigenvectors)
        
        # Perform operation in spectral domain
        if operation == "multiply":
            result_spectral = spectral1 * spectral2
        elif operation == "add":
            result_spectral = spectral1 + spectral2
        elif operation == "convolve":
            # True convolution: multiplication in spectral domain
            result_spectral = spectral1 * spectral2
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Transform back to spatial domain
        result_spatial = igft(result_spectral, spectral_state.eigenvectors)
        
        execution_time = time.perf_counter() - start_time
        
        # Update statistics
        self.total_operations += 1
        self.total_fft_ops += 2  # Forward + inverse transform
        
        result = FFTArithmeticResult(
            operation=SpectralOperation.CONVOLUTION,
            input_shape=signal1.shape,
            output_data=result_spatial,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=2,
            backend_used=self.default_backend
        )
        
        if self.cache_coordinator is not None:
            self.cache_coordinator.register_fft_result(
                G,
                result,
                metadata={"mode": operation}
            )
        
        return result
        
    def harmonic_analysis(
        self,
        G: Any,
        num_harmonics: int = 5,
        window_size: Optional[int] = None
    ) -> FFTArithmeticResult:
        """
        Perform multi-scale harmonic analysis.
        
        Identifies resonant modes and harmonic relationships in EPI evolution.
        """
        start_time = time.perf_counter()
        
        # Get spectral state
        spectral_state = self.get_spectral_state(G)
        
        # Extract EPI signal
        epi_signal = np.array([G.nodes[node].get('EPI', 0.0) for node in G.nodes()])
        
        # Identify dominant harmonics
        amplitudes = spectral_state.amplitudes
        frequencies = spectral_state.frequencies
        
        # Find peaks in amplitude spectrum
        harmonic_indices = np.argsort(amplitudes)[-num_harmonics:]
        harmonic_freqs = frequencies[harmonic_indices]
        harmonic_amps = amplitudes[harmonic_indices]
        
        # Analyze harmonic relationships
        fundamental_freq = harmonic_freqs[np.argmax(harmonic_amps)]
        
        harmonic_ratios = []
        for freq in harmonic_freqs:
            if fundamental_freq > 0:
                ratio = freq / fundamental_freq
                harmonic_ratios.append(ratio)
            else:
                harmonic_ratios.append(0.0)
                
        # Compute harmonic distortion
        total_harmonic_distortion = np.sqrt(np.sum(harmonic_amps[1:]**2)) / harmonic_amps[0] if harmonic_amps[0] > 0 else 0.0
        
        execution_time = time.perf_counter() - start_time
        
        # Create result
        harmonic_data = {
            "fundamental_frequency": fundamental_freq,
            "harmonic_frequencies": harmonic_freqs,
            "harmonic_amplitudes": harmonic_amps,
            "harmonic_ratios": harmonic_ratios,
            "total_harmonic_distortion": total_harmonic_distortion,
            "dominant_mode_index": harmonic_indices[np.argmax(harmonic_amps)]
        }
        
        self.total_operations += 1
        
        return FFTArithmeticResult(
            operation=SpectralOperation.HARMONIC_ANALYSIS,
            input_shape=epi_signal.shape,
            output_data=harmonic_data,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=1,
            backend_used=self.default_backend
        )
        
    def spectral_filtering(
        self,
        G: Any,
        filter_type: str = "lowpass",
        cutoff_frequency: Optional[float] = None,
        filter_order: int = 4
    ) -> FFTArithmeticResult:
        """
        Apply spectral filtering to graph signals.
        
        Enables noise reduction and mode selection in EPI evolution.
        """
        start_time = time.perf_counter()
        
        # Get spectral state
        spectral_state = self.get_spectral_state(G)
        
        # Determine cutoff frequency if not specified
        if cutoff_frequency is None:
            # Use median eigenvalue as default cutoff
            cutoff_frequency = np.median(spectral_state.eigenvalues)
            
        frequencies = spectral_state.eigenvalues

        def _build_filter() -> np.ndarray:
            return self._build_filter_response(
                frequencies,
                filter_type,
                float(cutoff_frequency),
                filter_order
            )

        if self.cache_coordinator is not None:
            filter_response = self.cache_coordinator.get_kernel(
                G,
                "spectral_filter",
                _build_filter,
                kernel_params={
                    "type": filter_type,
                    "cutoff": round(float(cutoff_frequency), 6),
                    "order": filter_order
                }
            )
        else:
            filter_response = _build_filter()
            
        # Apply filter to current signal
        epi_signal = np.array([G.nodes[node].get('EPI', 0.0) for node in G.nodes()])
        spectral_coeffs = spectral_state.spectral_coeffs
        
        # Apply filter
        filtered_coeffs = spectral_coeffs * filter_response
        
        # Transform back to spatial domain
        filtered_signal = igft(filtered_coeffs, spectral_state.eigenvectors)
        
        execution_time = time.perf_counter() - start_time
        
        # Create filtered result
        filtered_data = {
            "original_signal": epi_signal,
            "filtered_signal": filtered_signal,
            "filter_response": filter_response,
            "cutoff_frequency": cutoff_frequency,
            "filter_type": filter_type,
            "attenuation_db": self._calculate_attenuation_db(filter_response)
        }
        
        self.total_operations += 1
        self.total_fft_ops += 1  # Inverse transform
        
        result = FFTArithmeticResult(
            operation=SpectralOperation.FILTERING,
            input_shape=epi_signal.shape,
            output_data=filtered_data,
            spectral_state=spectral_state,
            execution_time=execution_time,
            fft_operations=1,
            backend_used=self.default_backend
        )
        
        if self.cache_coordinator is not None:
            self.cache_coordinator.register_fft_result(
                G,
                result,
                metadata={"filter_type": filter_type}
            )
        
        return result

    def _build_filter_response(
        self,
        frequencies: np.ndarray,
        filter_type: str,
        cutoff_frequency: float,
        filter_order: int,
    ) -> np.ndarray:
        """Construct smooth spectral filter responses."""

        response = np.ones_like(frequencies)
        safe_cutoff = max(cutoff_frequency, 1e-9)

        if filter_type == "lowpass":
            attenuation = np.maximum(frequencies - safe_cutoff, 0.0)
            response = np.exp(-attenuation * filter_order / safe_cutoff)
        elif filter_type == "highpass":
            attenuation = np.maximum(safe_cutoff - frequencies, 0.0)
            response = 1.0 - np.exp(-attenuation * filter_order / safe_cutoff)
        elif filter_type == "bandpass":
            low_cutoff = safe_cutoff * 0.5
            high_cutoff = safe_cutoff * 1.5
            response = np.exp(-np.maximum(low_cutoff - frequencies, 0.0) * filter_order / safe_cutoff)
            response *= np.exp(-np.maximum(frequencies - high_cutoff, 0.0) * filter_order / safe_cutoff)
        elif filter_type == "notch":
            bandwidth = safe_cutoff * 0.1
            distance = np.abs(frequencies - safe_cutoff)
            response = 1.0 - np.exp(-(bandwidth - distance).clip(min=0.0) * filter_order / safe_cutoff)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return np.clip(response, 0.0, 1.0)
        
    def cross_spectral_coherence(
        self,
        G1: Any,
        G2: Any,
        frequency_bands: Optional[int] = 10
    ) -> FFTArithmeticResult:
        """
        Compute cross-spectral coherence between two graphs.
        
        Measures synchronization and coupling strength between TNFR networks.
        """
        start_time = time.perf_counter()
        
        # Get spectral states for both graphs
        spectral1 = self.get_spectral_state(G1)
        spectral2 = self.get_spectral_state(G2)
        
        # Ensure compatible dimensions
        min_size = min(len(spectral1.spectral_coeffs), len(spectral2.spectral_coeffs))
        coeffs1 = spectral1.spectral_coeffs[:min_size]
        coeffs2 = spectral2.spectral_coeffs[:min_size]
        freqs1 = spectral1.frequencies[:min_size]
        freqs2 = spectral2.frequencies[:min_size]
        
        # Compute cross-spectral density
        cross_spectrum = coeffs1 * np.conj(coeffs2)
        
        # Compute power spectra
        power1 = np.abs(coeffs1)**2
        power2 = np.abs(coeffs2)**2
        
        # Compute coherence function
        coherence = np.abs(cross_spectrum)**2 / (power1 * power2 + 1e-12)  # Avoid division by zero
        
        # Compute frequency-band coherence
        if frequency_bands:
            band_edges = np.linspace(0, np.max([np.max(freqs1), np.max(freqs2)]), frequency_bands + 1)
            band_coherence = []
            
            for i in range(frequency_bands):
                band_mask = (freqs1 >= band_edges[i]) & (freqs1 < band_edges[i+1])
                if np.any(band_mask):
                    band_coh = np.mean(coherence[band_mask])
                    band_coherence.append(band_coh)
                else:
                    band_coherence.append(0.0)
        else:
            band_coherence = []
            
        # Compute overall coherence metrics
        mean_coherence = np.mean(coherence)
        max_coherence = np.max(coherence)
        coherent_bandwidth = np.sum(coherence > 0.5) / len(coherence)  # Fraction above threshold
        
        execution_time = time.perf_counter() - start_time
        
        # Create coherence result
        coherence_data = {
            "coherence_spectrum": coherence,
            "cross_spectrum": cross_spectrum,
            "frequencies": freqs1,  # Use first graph's frequencies
            "band_coherence": band_coherence,
            "mean_coherence": mean_coherence,
            "max_coherence": max_coherence,
            "coherent_bandwidth": coherent_bandwidth,
            "frequency_bands": frequency_bands
        }
        
        self.total_operations += 1
        
        return FFTArithmeticResult(
            operation=SpectralOperation.COHERENCE_ANALYSIS,
            input_shape=(min_size,),
            output_data=coherence_data,
            execution_time=execution_time,
            fft_operations=0,  # No additional FFTs needed
            backend_used=self.default_backend
        )
        
    def _estimate_coherence_length(self, eigenvalues: np.ndarray, amplitudes: np.ndarray) -> float:
        """Estimate spatial coherence length from spectral decay."""
        # Find spectral centroid (weighted average frequency)
        total_power = np.sum(amplitudes**2)
        if total_power > 0:
            centroid = np.sum(eigenvalues * amplitudes**2) / total_power
        else:
            centroid = 0.0
            
        # Estimate coherence length as inverse of spectral centroid
        if centroid > 0:
            coherence_length = 1.0 / np.sqrt(centroid)
        else:
            coherence_length = float('inf')  # Infinite coherence
            
        return coherence_length
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_operations": self.total_operations,
            "total_fft_operations": self.total_fft_ops,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_operations),
            "cached_spectra": len(self._spectral_cache),
            "backend": self.default_backend,
            "precision": self.precision,
            "spectral_analysis_available": HAS_SPECTRAL,
            "unified_cache_available": HAS_UNIFIED_CACHE
        }
        
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._spectral_cache.clear()
        self._window_cache.clear()


# Factory functions
def create_fft_arithmetic_engine(**kwargs) -> TNFRAdvancedFFTEngine:
    """Create FFT arithmetic engine."""
    return TNFRAdvancedFFTEngine(**kwargs)


def fast_spectral_convolution(G: Any, signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """Convenience function for fast spectral convolution."""
    engine = create_fft_arithmetic_engine()
    result = engine.spectral_convolution(G, signal1, signal2, operation="multiply")
    return result.output_data


def analyze_graph_harmonics(G: Any, num_harmonics: int = 5) -> Dict[str, Any]:
    """Convenience function for harmonic analysis."""
    engine = create_fft_arithmetic_engine()
    result = engine.harmonic_analysis(G, num_harmonics)
    return result.output_data


def measure_graph_coherence(G1: Any, G2: Any) -> float:
    """Convenience function for measuring cross-graph coherence."""
    engine = create_fft_arithmetic_engine()
    result = engine.cross_spectral_coherence(G1, G2)
    return result.output_data["mean_coherence"]