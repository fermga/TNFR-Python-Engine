"""Self-optimization engine wrapper.

This module re-exports the canonical implementation from
``tnfr.dynamics.self_optimizing_engine`` so that legacy imports from the
``tnfr.engines`` namespace remain valid without duplicating code.
"""

from ...dynamics.self_optimizing_engine import (
    TNFRSelfOptimizingEngine,
    OptimizationObjective,
    LearningStrategy,
    OptimizationExperience,
    OptimizationPolicy,
    SelfOptimizationResult,
    create_self_optimizing_engine,
    auto_optimize_tnfr_computation,
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
