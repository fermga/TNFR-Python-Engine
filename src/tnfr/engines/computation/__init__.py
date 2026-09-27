"""TNFR Computation Engines

Computation adapters for ordered-sequence FFTs and graph dynamics.
Backend availability and actual acceleration are reported by each operation;
importing this namespace does not establish a speedup or a dynamical theorem.

Main Classes:
- TNFRUnifiedGPUSystem: Backend selection and CPU/GPU provenance
- TNFRUnifiedFFTEngine: Consolidated FFT processing with intelligent backend selection
- FFTDynamicsEngine: Compatibility export of the shared graph-dynamics owner

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
        clear_unified_fft_cache,
        compute_unified_fft,
        compute_unified_spectral_convolution,
        get_unified_fft_engine,
        get_unified_fft_stats,
    )
    from .unified_gpu_system import (
        GPUOperationResult,
        TNFRUnifiedGPUSystem,
        UnifiedGPUConfig,
        cleanup_unified_gpu_memory,
        compute_unified_delta_nfr,
        compute_unified_structural_fields,
        get_unified_gpu_stats,
        get_unified_gpu_system,
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
        "get_unified_gpu_stats",
    ]
except ImportError:
    __all__ = []

try:
    from .fft_engine import FFTDynamicsEngine

    __all__.append("FFTDynamicsEngine")
except ImportError:
    pass
