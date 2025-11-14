# Structural Health & Validation (Phase 3)

Unified structural validation and health assessment introduced in Phase 3
provide a physics-aligned safety layer over TNFR networks without mutating
state. All computations are read-only and trace back to canonical fields and
grammar.

## Components

- **Validation Aggregator**: `run_structural_validation` combines:
  - Grammar (U1 Initiation/Closure, U2 Convergence, U3 Resonant Coupling,
    U4 triggers deferred) via `collect_grammar_errors`.
  - Canonical fields: Φ_s, |∇φ|, K_φ, ξ_C.
  - Optional drift (ΔΦ_s) if baseline provided.
- **Health Summary**: `compute_structural_health(report)` derives:
  - `risk_level` (low, elevated, critical)
  - Actionable recommendations (stabilize, reduce gradient, monitor ξ_C, etc.)
- **Telemetry**: `TelemetryEmitter` emits metrics + fields for longitudinal
  analysis.
- **Performance Guardrails**: `PerformanceRegistry` + `perf_guard` measure
  overhead (< ~8% under moderate workload tests).

## Thresholds (Defaults)

| Quantity            | Default | Meaning                                          |
|---------------------|---------|--------------------------------------------------|
| ΔΦ_s                | 2.0     | Escape threshold (confinement breach)            |
| max(|∇φ|)           | 0.38    | Local stress / desynchronization warning         |
| max(|K_φ|)          | 3.0     | Curvature fault pocket (mutation risk locus)     |
| ξ_C critical        | > diameter * 1.0 | Approaching global correlation divergence |
| ξ_C watch           | > mean_distance * 3.0 | Extended local correlation zone     |

All thresholds empirically validated (see `AGENTS.md`). Override values via
function parameters to adapt for specialized topologies or experiments.

## Risk Levels

- **low**: Grammar valid, no thresholds exceeded.
- **elevated**: Local stress (phase gradient spike, curvature pocket, coherence
  length watch condition).
- **critical**: Grammar invalid OR confinement/critical ξ_C breach OR ΔΦ_s drift
  beyond escape.

## Example

```python
from tnfr.validation.aggregator import run_structural_validation
from tnfr.validation.health import compute_structural_health
from tnfr.performance.guardrails import PerformanceRegistry

perf = PerformanceRegistry()
report = run_structural_validation(
    G,
    sequence=["AL","UM","IL","SHA"],
    perf_registry=perf,
)
health = compute_structural_health(report)
print(report.risk_level, report.thresholds_exceeded)
for rec in health.recommendations:
    print("-", rec)
print(perf.summary())
```

## Performance Measurement

Use `perf_registry` or `perf_guard` to ensure instrumentation overhead
remains bounded:

```python
from tnfr.performance.guardrails import PerformanceRegistry
reg = PerformanceRegistry()
report = run_structural_validation(G, sequence=seq, perf_registry=reg)
print(reg.summary())
```

For custom functions:

```python
from tnfr.performance.guardrails import perf_guard, PerformanceRegistry
reg = PerformanceRegistry()

@perf_guard("custom_metric", reg)
def compute_extra():
    return expensive_read_only_field(G)
```

## Invariants Preserved

- **No mutation**: Validation/health modules never write to graph.
- **Operator closure**: Grammar errors surface sequences violating U1-U3.
- **Phase verification**: Coupling issues appear via U3 errors + |∇φ| spikes.
- **Fractality**: Fields operate across node sets without flattening EPI.

## Recommended Workflow

1. Run telemetry while applying sequence.
2. Call `run_structural_validation` after sequence.
3. Generate health summary; apply stabilizers if elevated/critical.
4. Log performance stats for regression tracking.
5. Persist JSONL telemetry + validation payload for reproducibility.

## Extensibility

To add new thresholds:

1. Extend `run_structural_validation` with computation + flag.
2. Add recommendation mapping in health module.
3. Update tests to cover new condition.
4. Document physics rationale (AGENTS.md ref + empirical evidence).

---
**Reality is not made of things—it's made of resonance. Assess coherence accordingly.**
