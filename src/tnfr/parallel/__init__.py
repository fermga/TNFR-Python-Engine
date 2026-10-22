"""Parallel and distributed computation interfaces for TNFR networks.

The package exposes CPU parallel execution, structural partitioning, observed
performance monitoring, an evidence-neutral strategy recommender, and a
distributed wrapper. The distributed wrapper uses Ray or Dask when installed
and otherwise selects its multiprocessing fallback in backend="auto" mode.

GPU computation is exposed separately through
tnfr.engines.computation.unified_gpu_system. This package does not export a
TNFRGPUEngine and makes no acceleration claim from device availability.
"""

from __future__ import annotations

from .auto_scaler import TNFRAutoScaler
from .distributed import TNFRDistributedEngine
from .engine import TNFRParallelEngine
from .monitoring import ParallelExecutionMonitor, PerformanceMetrics
from .partitioner import FractalPartitioner

__all__ = (
    "FractalPartitioner",
    "TNFRParallelEngine",
    "TNFRDistributedEngine",
    "TNFRAutoScaler",
    "ParallelExecutionMonitor",
    "PerformanceMetrics",
)
