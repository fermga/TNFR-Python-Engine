"""Reference CPU operator strategies for streaming partitions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from .strategy import (
    BackendName,
    OperationResult,
    OperatorName,
    PartitionBlock,
    PreparedBlock,
    ResourceEstimate,
    StrategyContext,
    StrategyFactory,
    StrategyRegistry,
)

# ---------------------------------------------------------------------------
# Phase gradient risk thresholds
# ---------------------------------------------------------------------------
_PHASE_GRADIENT_MEDIUM_RISK = 0.5
_PHASE_GRADIENT_HIGH_RISK = 0.9

@dataclass
class _PreparedBlock:
    ctx: StrategyContext
    block: PartitionBlock
    operator: OperatorName
    strategy_name: str

class _BaseCpuStrategy:
    """Utility mixin implementing the OperatorStrategy protocol."""

    operator: OperatorName
    name: str = "cpu-default"
    max_block_size: int = 1_000_000
    backend: BackendName = "cpu"
    memory_per_node: int = 128
    time_per_node_ms: float = 0.02
    delta_nfr_factor: float = 1e-4
    phi_s_drift_factor: float = 5e-5

    def supports(self, ctx: StrategyContext) -> bool:  # pragma: no cover - trivial
        return ctx.backend == self.backend and ctx.block_size <= self.max_block_size

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        memory = max(4096, int(ctx.block_size * self.memory_per_node))
        time_ms = max(0.1, ctx.block_size * self.time_per_node_ms)
        delta_nfr = self.delta_nfr_factor * ctx.block_size
        phi_drift = self.phi_s_drift_factor * ctx.block_size
        # When coherence is high, risk is lower.
        risk = "low"
        if ctx.structural_fields.phase_gradient > _PHASE_GRADIENT_MEDIUM_RISK:
            risk = "medium"
        if ctx.structural_fields.phase_gradient > _PHASE_GRADIENT_HIGH_RISK:
            risk = "high"
        return ResourceEstimate(
            memory_bytes=memory,
            time_ms=time_ms,
            delta_nfr=delta_nfr,
            phi_s_drift=phi_drift,
            failure_risk=risk,
        )

    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock:
        return _PreparedBlock(ctx=ctx, block=block, operator=self.operator, strategy_name=self.name)

    def apply(self, prepared: _PreparedBlock) -> OperationResult:
        telemetry = {
            "operator": prepared.operator,
            "strategy": prepared.strategy_name,
            "partition_id": prepared.ctx.partition_id,
            "block_size": prepared.ctx.block_size,
            "boundary_overlap": prepared.ctx.boundary_overlap,
        }
        seed = f"{prepared.ctx.partition_id}:{prepared.ctx.operator_sequence_position}:{prepared.operator}".encode()
        proof_hash = hashlib.sha3_256(seed).hexdigest()
        return OperationResult(block=prepared.block, telemetry=telemetry, proof_hash=proof_hash)

    def cleanup(self, prepared: PreparedBlock) -> None:  # pragma: no cover - no resources
        return None

class CpuEmissionStrategy(_BaseCpuStrategy):
    operator: OperatorName = "AL"
    memory_per_node = 192
    time_per_node_ms = 0.03
    delta_nfr_factor = 1.5e-4
    phi_s_drift_factor = 7.5e-5

class CpuCoherenceStrategy(_BaseCpuStrategy):
    operator: OperatorName = "IL"
    memory_per_node = 160
    time_per_node_ms = 0.025
    delta_nfr_factor = -8e-5
    phi_s_drift_factor = 4e-5

class CpuResonanceStrategy(_BaseCpuStrategy):
    operator: OperatorName = "RA"
    memory_per_node = 200
    time_per_node_ms = 0.035
    delta_nfr_factor = 2e-4
    phi_s_drift_factor = 6e-5

class CpuSilenceStrategy(_BaseCpuStrategy):
    operator: OperatorName = "SHA"
    memory_per_node = 64
    time_per_node_ms = 0.01
    delta_nfr_factor = -1.2e-4
    phi_s_drift_factor = 2e-5

_DEFAULT_STRATEGIES: Mapping[OperatorName, tuple[StrategyFactory, ...]] = {
    "AL": (CpuEmissionStrategy,),
    "IL": (CpuCoherenceStrategy,),
    "RA": (CpuResonanceStrategy,),
    "SHA": (CpuSilenceStrategy,),
}

def ensure_default_strategies_registered() -> None:
    for operator, factories in _DEFAULT_STRATEGIES.items():
        existing = StrategyRegistry.get(operator)
        for factory in factories:
            name = getattr(factory, "name", "cpu-default")
            if name in existing:
                continue
            StrategyRegistry.register(operator=operator, name=name, factory=factory)  # type: ignore[arg-type]
    # no-op if already registered
