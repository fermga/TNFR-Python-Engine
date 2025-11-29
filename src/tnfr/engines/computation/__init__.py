"""TNFR Computation Engines

High-performance computing backends for TNFR operations.
Includes GPU acceleration, FFT processing, and parallel computation.

Main Classes:
- GPUEngine: GPU-accelerated TNFR computations
- FFTEngine: Fast Fourier Transform processing

Usage:
```python
from tnfr.engines.computation import GPUEngine, FFTEngine
gpu_engine = GPUEngine()
result = gpu_engine.compute_field_tetrad(network)
```
"""

try:
    from .gpu_engine import GPUEngine
    __all__ = ["GPUEngine"]
except ImportError:
    __all__ = []

try:
    from .fft_engine import FFTEngine
    __all__.append("FFTEngine")
except ImportError:
    pass
