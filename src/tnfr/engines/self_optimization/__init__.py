"""TNFR Self-Optimization Engine

Configured computation selection using retained performance evidence.
Recommendations do not constitute an emergent selection law or guarantee
improvement for an unmeasured workload.

Main Classes:
- TNFRSelfOptimizingEngine: Core optimization engine

Usage:
```python
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
engine = TNFRSelfOptimizingEngine()
report = engine.optimize_automatically(network, dry_run=True)
```
"""

from .engine import (
    LearningStrategy,
    OptimizationExperience,
    OptimizationObjective,
    OptimizationPolicy,
    SelfOptimizationResult,
    TNFRSelfOptimizingEngine,
    auto_optimize_tnfr_computation,
    create_self_optimizing_engine,
)

__all__ = [
    "TNFRSelfOptimizingEngine",
    "OptimizationObjective",
    "LearningStrategy",
    "OptimizationExperience",
    "OptimizationPolicy",
    "SelfOptimizationResult",
    "create_self_optimizing_engine",
    "auto_optimize_tnfr_computation",
]
