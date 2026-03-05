"""Operator strategy scaffolding exports."""

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
from .defaults import ensure_default_strategies_registered

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
