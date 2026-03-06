"""Operator strategy scaffolding for AL/IL/RA/SHA streaming execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

OperatorName = Literal["AL", "IL", "RA", "SHA"]
BackendName = Literal["cpu", "gpu", "remote"]
FailureRisk = Literal["low", "medium", "high"]
PartitionBlock = Any
PreparedBlock = Any

@dataclass(frozen=True)
class StructuralFields:
    phi_s: float
    phase_gradient: float
    phase_curvature: float
    coherence_length: float

@dataclass(frozen=True)
class StrategyContext:
    partition_id: str
    operator_sequence_position: int
    structural_fields: StructuralFields
    dispatcher_capabilities: Mapping[str, Any]
    backend: BackendName
    block_size: int
    boundary_overlap: int
    seed: int

@dataclass(frozen=True)
class ResourceEstimate:
    memory_bytes: int
    time_ms: float
    delta_nfr: float
    phi_s_drift: float
    failure_risk: FailureRisk

@dataclass
class OperationResult:
    block: PartitionBlock
    telemetry: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    proof_hash: str = ""

class OperatorStrategy(Protocol):
    operator: OperatorName

    def supports(self, ctx: StrategyContext) -> bool:
        ...

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        ...

    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock:
        ...

    def apply(self, prepared: PreparedBlock) -> OperationResult:
        ...

    def cleanup(self, prepared: PreparedBlock) -> None:
        ...

StrategyFactory = Callable[[], OperatorStrategy]

class StrategyRegistrationError(RuntimeError):
    """Raised when invalid or duplicate strategy registrations occur."""

class StrategyRegistry:
    """In-memory registry for operator strategies."""

    _registry: MutableMapping[OperatorName, dict[str, StrategyFactory]] = {
        "AL": {},
        "IL": {},
        "RA": {},
        "SHA": {},
    }

    @classmethod
    def register(cls, *, operator: OperatorName, name: str, factory: StrategyFactory) -> None:
        name = name.strip()
        if not name:
            raise StrategyRegistrationError("Strategy name cannot be empty")
        if operator not in cls._registry:
            raise StrategyRegistrationError(f"Unsupported operator '{operator}'")
        bucket = cls._registry[operator]
        if name in bucket:
            raise StrategyRegistrationError(f"Strategy '{name}' already registered for {operator}")
        bucket[name] = factory

    @classmethod
    def get(cls, operator: OperatorName) -> Mapping[str, StrategyFactory]:
        if operator not in cls._registry:
            raise StrategyRegistrationError(f"Unsupported operator '{operator}'")
        return dict(cls._registry[operator])

    @classmethod
    def create(cls, *, operator: OperatorName, name: str) -> OperatorStrategy:
        bucket = cls.get(operator)
        try:
            factory = bucket[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise StrategyRegistrationError(
                f"Strategy '{name}' not registered for {operator}"
            ) from exc
        return factory()

    @classmethod
    def available(cls) -> Mapping[OperatorName, list[str]]:
        return {op: sorted(names.keys()) for op, names in cls._registry.items()}

    @classmethod
    def clear(cls) -> None:
        for bucket in cls._registry.values():
            bucket.clear()
