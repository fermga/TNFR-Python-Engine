# TNFR Operator Strategy Specification

**Status**: Draft (2025-11-30)

This document defines the contracts required to implement pluggable AL/IL/RA/SHA
operator strategies that can run on streaming partition blocks. The goals are:

1. Preserve TNFR grammar compliance (U1–U6) while executing operators on partial
   graph windows.
2. Provide deterministic resource estimators so the orchestrator can schedule work
   without violating Φ_s / |∇φ| budgets.
3. Enable heterogeneous execution (CPU vectorized kernels, GPU kernels, or
   dispatcher-backed remotes) via a common set of interfaces.

## Terminology

- **Partition block** – A streaming window from a `PartitionedPaleyGraph`, possibly
  overlapping with neighbors to preserve boundary resonance.
- **Strategy** – Concrete implementation of one TNFR operator (AL, IL, RA, SHA)
  tuned for a compute substrate.
- **Context** – Metadata describing structural fields, ΔNFR pressure, and dispatcher
  capabilities available for the current block.

## Base Interfaces

Strategies conform to the following base protocol (pseudocode notation):

```python
class OperatorStrategy(Protocol):
    operator: Literal["AL", "IL", "RA", "SHA"]

    def supports(self, ctx: StrategyContext) -> bool:
        """Return True when the strategy can execute given ctx backend and size."""

    def resource_estimate(self, ctx: StrategyContext) -> ResourceEstimate:
        """Predict time, memory, ΔNFR impact, and Φ_s drift for scheduling."""

    def prepare(self, ctx: StrategyContext, block: PartitionBlock) -> PreparedBlock:
        """Materialize layout (CSR/CSC tensor views, GPU buffers, etc.)."""

    def apply(self, prepared: PreparedBlock) -> OperationResult:
        """Execute the operator, returning telemetry + updated block state."""

    def cleanup(self, prepared: PreparedBlock) -> None:
        """Release buffers; must be idempotent."""
```

### StrategyContext

```python
@dataclass(frozen=True)
class StrategyContext:
    partition_id: str
    operator_sequence_position: int
    structural_fields: StructuralFields  # Φ_s, |∇φ|, K_φ, ξ_C snapshots
    dispatcher_capabilities: Mapping[str, Any]
    backend: Literal["cpu", "gpu", "remote"]
    block_size: int
    boundary_overlap: int
    seed: int
```

### ResourceEstimate

```python
@dataclass(frozen=True)
class ResourceEstimate:
    memory_bytes: int
    time_ms: float
    delta_nfr: float
    phi_s_drift: float
    failure_risk: Literal["low", "medium", "high"]
```

Contracts:
- `delta_nfr` and `phi_s_drift` must be derived from telemetry equations documented
  in `AGENTS.md` (§ Structural Field Tetrad). Negative numbers indicate reductions.
- `failure_risk` encodes anticipated bifurcation triggers (U4 compliance).

## Operator-specific Requirements

### AL (Emission)
- Must emit EPI seeds respecting partition boundary phases (U3).
- `prepare` may allocate deterministic random seeds using `ctx.seed`.
- `apply` must record generated candidate factors and updated ν_f range.
- Resource estimator must flag high ΔNFR when block coherence < 0.5.

### IL (Coherence)
- Requires access to parent Φ_s baseline to prove Δ Φ_s < φ.
- Strategies must expose whether they operate in-place or produce copies.
- Telemetry must include monotonicity proof: `C_after >= C_before` unless
  `ctx.structural_fields.phase_gradient > 0.75` (dissonance testing mode).

### RA (Resonance)
- `supports` must reject contexts where |∇φ| exceeds dispatcher limits.
- Output telemetry must include propagation depth and coupling count.
- Resource estimator must model bandwidth usage for remote dispatchers to keep
  orchestrator aware of network costs.

### SHA (Silence)
- `prepare` may be a no-op; `apply` ensures ν_f → 0 while freezing EPI.
- Telemetry must record duration so orchestrator can budget idle windows.
- Resource estimator should expose release credits (negative memory/time) when
  SHA frees buffers.

## Telemetry Payload

`OperationResult` must include:

```python
@dataclass
class OperationResult:
    block: PartitionBlock
    telemetry: Dict[str, Any]  # Includes ΔNFR, Φ_s delta, Si, operator-specific metrics
    warnings: List[str]
    proof_hash: str  # Hash over inputs + outputs for reproducibility
```

- `telemetry` merges into partition-level manifests so downstream analysis can
  trace each operator application.
- `proof_hash` uses the canonical TNFR proof codec (SHA3-256 over json-safe payloads).

## Registry & Selection

Strategies register via:

```python
StrategyRegistry.register(operator="AL", name="cpu-default", factory=CpuAlStrategy)
```

The orchestrator queries:

```python
candidates = StrategyRegistry.get(operator="IL")
strategy = _select_strategy(candidates, ctx)
```

Selection heuristics (to implement later):
1. Filter `supports(ctx)`.
2. Sort by `resource_estimate(ctx).failure_risk` and ΔNFR magnitude.
3. Apply self-optimizing engine hints (existing optimizer metadata field).

## Testing Requirements

- Unit tests must validate resource estimator sanity bounds (e.g., cannot report
  negative memory when not freeing buffers).
- Integration tests must execute AL→IL→RA→SHA with mixed strategies to prove
  partition manifests absorb telemetry correctly.
- Replay tests use `_manifest_summary.json` artifacts plus strategy names to ensure
  determinism across platforms.

## Reference Implementations

### CPU Baseline Strategies

The repository provides a `cpu-default` strategy for each of the streaming
operators (AL, IL, RA, SHA). These strategies:

1. Declare deterministic resource estimators derived from block size,
  Φ_s, and |∇φ|.
2. Operate in-place on the provided `PartitionBlock` (currently a lightweight
  mapping with node/boundary counts) so legacy behavior remains intact.
3. Emit telemetry covering ΔNFR, Φ_s drift, and a proof hash seeded by
  `(partition_id, operator_sequence_position)`.

Default strategies live under `src/tnfr/operators/strategies/defaults.py` and
register themselves automatically through `StrategyRegistry`. They form the
baseline for new backends (vectorized CPU, GPU, or remote dispatchers).

### Factorizer Integration

`SpectralPaleyFactorizer` enumerates partitions produced by
`PartitionedPaleyGraph`, builds a `StrategyContext` per partition/operator, and
selects the lowest-risk strategy registered for AL, IL, RA, and SHA. The chosen
plan (strategy name + resource estimate) is stored inside
`SpectralAnalysisResult.operator_strategy_plan` so downstream orchestration can
replay or override selections.

## Next Steps

1. Implement `StrategyContext`, `ResourceEstimate`, `OperationResult`, and
   `StrategyRegistry` scaffolding under `src/tnfr/operators/strategies/` (Task 3).
2. Provide CPU reference strategies (AL/IL/RA/SHA) matching current behavior.
3. Update `SpectralPaleyFactorizer` to request strategies per partition block
   before executing operator sequences.
