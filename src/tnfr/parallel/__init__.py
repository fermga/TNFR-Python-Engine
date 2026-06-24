"""Parallel and distributed computation engines for TNFR networks.

This module provides parallelization strategies that respect TNFR's structural
coherence while enabling efficient computation on large networks. All parallel
implementations preserve the canonical nodal equation and operator closure.

Key components:
- FractalPartitioner: Partitions networks using coherence-based communities
- TNFRParallelEngine: Multiprocessing/threading engine for medium networks
- TNFRDistributedEngine: Optional Ray/Dask backend for massive networks
- TNFRGPUEngine: Optional GPU acceleration via JAX/CuPy
- TNFRAutoScaler: Recommends optimal execution strategy
- ParallelExecutionMonitor: Real-time performance tracking

All engines maintain TNFR invariants:
- EPI changes only via structural operators
- νf expressed in Hz_str (structural hertz)
- ΔNFR semantics preserved (not reinterpreted as ML gradient)
- Operator closure maintained
- Phase synchrony verification
- Operational fractality preserved
"""

from __future__ import annotations

from .auto_scaler import TNFRAutoScaler
from .engine import TNFRParallelEngine
from .monitoring import ParallelExecutionMonitor, PerformanceMetrics

# Import all core components
from .partitioner import FractalPartitioner

__all__ = (
    "FractalPartitioner",
    "TNFRParallelEngine",
    "TNFRAutoScaler",
    "ParallelExecutionMonitor",
    "PerformanceMetrics",
)

# Optional distributed backends
try:
    pass

    __all__ = __all__ + ("TNFRDistributedEngine",)
except ImportError:
    pass

# Optional GPU backend
try:
    # GPU functionality moved to tnfr.engines.computation.unified_gpu_system
    pass
except ImportError:
    pass
