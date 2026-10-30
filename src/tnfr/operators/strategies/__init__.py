"""Operator strategy scaffolding exports."""

from .defaults import ensure_default_strategies_registered
from .strategy import (
    BackendName,
    FailureRisk,
    OperationResult,
    OperatorName,
    OperatorStrategy,
    PartitionBlock,
    PreparedBlock,
    ResourceEstimate,
    StrategyContext,
    StrategyFactory,
    StrategyRegistrationError,
    StrategyRegistry,
    StructuralFields,
)

ensure_default_strategies_registered()

__all__ = [
    "BackendName",
    "FailureRisk",
    "OperationResult",
    "OperatorName",
    "OperatorStrategy",
    "PartitionBlock",
    "PreparedBlock",
    "ResourceEstimate",
    "StrategyContext",
    "StrategyFactory",
    "StrategyRegistrationError",
    "StrategyRegistry",
    "StructuralFields",
    "ensure_default_strategies_registered",
]
