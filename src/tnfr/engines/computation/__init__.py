"""TNFR Computation Engines

High-performance computing backends for TNFR operations.
Includes GPU acceleration, FFT processing, and parallel computation.

Main Classes:
- TNFRUnifiedGPUSystem: Centralized GPU-accelerated TNFR computations
- TNFRUnifiedFFTEngine: Consolidated FFT processing with intelligent backend selection

Usage:
```python
from tnfr.engines.computation import get_unified_gpu_system, get_unified_fft_engine
gpu_system = get_unified_gpu_system()
fft_engine = get_unified_fft_engine()
result = fft_engine.compute_fft(data)
```
"""

try:
    from .unified_fft_engine import (
        TNFRUnifiedFFTEngine,
        UnifiedFFTConfig,
        UnifiedFFTResult,
        get_unified_fft_engine,
        compute_unified_fft,
        compute_unified_spectral_convolution,
        clear_unified_fft_cache,
        get_unified_fft_stats
    )
    from .unified_gpu_system import (
        TNFRUnifiedGPUSystem,
        UnifiedGPUConfig,
        GPUOperationResult,
        get_unified_gpu_system,
        compute_unified_delta_nfr,
        compute_unified_structural_fields,
        cleanup_unified_gpu_memory,
        get_unified_gpu_stats
    )
    
    __all__ = [
        # Unified FFT Engine
        "TNFRUnifiedFFTEngine",
        "UnifiedFFTConfig",
        "UnifiedFFTResult",
        "get_unified_fft_engine",
        "compute_unified_fft",
        "compute_unified_spectral_convolution",
        "clear_unified_fft_cache",
        "get_unified_fft_stats",
        # Unified GPU System (consolidates all GPU functionality)
        "TNFRUnifiedGPUSystem",
        "UnifiedGPUConfig",
        "GPUOperationResult",
        "get_unified_gpu_system",
        "compute_unified_delta_nfr",
        "compute_unified_structural_fields",
        "cleanup_unified_gpu_memory",
        "get_unified_gpu_stats"
    ]
except ImportError:
    __all__ = []

try:
    from .fft_engine import FFTEngine
    __all__.append("FFTEngine")
except ImportError:
    pass
