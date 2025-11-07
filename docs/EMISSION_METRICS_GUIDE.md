# Emission Metrics Guide

This guide explains the extended emission-specific metrics introduced for the AL (Emission) operator.

## Overview

Emission (AL) is the foundational operator that activates nodal resonance. To better analyze and debug emission effectiveness, we've extended the metrics collection to include **structural fidelity indicators** that reflect the canonical AL effects documented in TNFR.pdf §2.2.1.

## Enabling Metrics Collection

To collect emission metrics, enable the `COLLECT_OPERATOR_METRICS` flag:

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Reception, Coherence, Silence

G, node = create_nfr("test", epi=0.2, vf=1.0)
G.graph["COLLECT_OPERATOR_METRICS"] = True

run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

# Access metrics
metrics = G.graph["operator_metrics"][0]  # First operator (Emission)
```

## Available Metrics

### Core Metrics

These metrics capture the fundamental structural changes during emission:

| Metric | Type | Description |
|--------|------|-------------|
| `delta_epi` | float | Change in EPI (Δ_EPI) |
| `delta_vf` | float | Change in νf (Δ_νf) |
| `dnfr_initialized` | float | Initial ΔNFR value after emission |
| `theta_current` | float | Current phase (θ) in radians |

### AL-Specific Quality Indicators (NEW)

These metrics provide qualitative assessment of emission effectiveness:

#### 1. `emission_quality`: "valid" | "weak"

Qualitative assessment based on structural effects:
- **"valid"**: Both EPI and νf increased (canonical emission)
- **"weak"**: One or both did not increase (limited structural activation)

**Use case**: Quickly identify successful vs. problematic emissions.

```python
if metrics["emission_quality"] == "weak":
    print("Warning: Weak emission detected")
```

#### 2. `activation_from_latency`: bool

Indicates whether the node was in latent state (EPI < 0.3) before emission:
- **True**: Node was latent, emission represents true activation
- **False**: Node was already active, emission is reactivation

**Use case**: Distinguish between first-time activations and reactivations.

```python
if metrics["activation_from_latency"]:
    print("First activation from latent state")
```

#### 3. `form_emergence_magnitude`: float

Absolute EPI increment (same as `delta_epi`). This metric explicitly captures the magnitude of structural form emergence.

**Use case**: Measure how much structural form emerged during emission.

```python
print(f"Form emerged: {metrics['form_emergence_magnitude']:.3f}")
```

#### 4. `frequency_activation`: bool

Indicates whether νf (structural frequency) increased:
- **True**: νf activated/increased (structural reorganization enabled)
- **False**: νf did not increase (limited reorganization capacity)

**Use case**: Verify that structural frequency was activated as expected.

```python
if not metrics["frequency_activation"]:
    print("Warning: Frequency did not activate")
```

#### 5. `reorganization_positive`: bool

Indicates whether ΔNFR is positive:
- **True**: Positive reorganization gradient (expansion)
- **False**: Non-positive gradient (no expansion or contraction)

**Use case**: Verify positive reorganization as expected from AL.

```python
if metrics["reorganization_positive"]:
    print("Positive reorganization confirmed")
```

### Traceability Markers (NEW)

These metrics provide structural traceability:

#### 6. `emission_timestamp`: str | None

ISO 8601 UTC timestamp of first emission activation:
- Set on first emission
- Preserved on reactivations
- `None` if emission hasn't occurred yet

**Use case**: Full traceability of when structural activation occurred.

```python
print(f"Activated at: {metrics['emission_timestamp']}")
```

#### 7. `irreversibility_marker`: bool

Indicates whether the node has been structurally activated (AL is irreversible):
- **True**: Node has been activated (immutable flag)
- **False**: Node hasn't been activated yet

**Use case**: Verify irreversibility and structural commitment.

```python
if metrics["irreversibility_marker"]:
    print("Node has been structurally activated (irreversible)")
```

## Example Usage

### Example 1: Analyzing Emission Quality

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Reception, Coherence, Silence

G, node = create_nfr("analysis_node", epi=0.15, vf=1.2)
G.graph["COLLECT_OPERATOR_METRICS"] = True

run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])

metrics = G.graph["operator_metrics"][0]

# Check emission quality
if metrics["emission_quality"] == "valid":
    print("✓ Valid emission - both EPI and νf increased")
else:
    print("⚠ Weak emission detected")
    print(f"  ΔE: {metrics['delta_epi']:+.3f}")
    print(f"  Δνf: {metrics['delta_vf']:+.3f}")
```

### Example 2: Debugging Spurious Activations

```python
# Detect if emission occurred from truly latent state
if not metrics["activation_from_latency"]:
    print("Warning: Emission applied to already-active node")
    print("Consider using Coherence instead for stabilization")
```

### Example 3: Research on Emergence Dynamics

```python
import pandas as pd

# Collect metrics from multiple emissions
all_metrics = []
for i in range(10):
    G, node = create_nfr(f"node_{i}", epi=0.1 + i*0.05, vf=1.0)
    G.graph["COLLECT_OPERATOR_METRICS"] = True
    run_sequence(G, node, [Emission(), Reception(), Coherence(), Silence()])
    all_metrics.append(G.graph["operator_metrics"][0])

# Analyze emergence patterns
df = pd.DataFrame(all_metrics)
print(df[["activation_from_latency", "form_emergence_magnitude", 
          "emission_quality"]].describe())
```

### Example 4: Full Diagnostic Report

```python
def diagnose_emission(metrics: dict) -> None:
    """Print comprehensive emission diagnostics."""
    print(f"\n{'='*50}")
    print(f"EMISSION DIAGNOSTIC REPORT")
    print(f"{'='*50}")
    
    print(f"\nQuality: {metrics['emission_quality'].upper()}")
    print(f"Timestamp: {metrics['emission_timestamp']}")
    
    print(f"\nStructural Effects:")
    print(f"  ΔE:  {metrics['delta_epi']:+.3f} {'✓' if metrics['delta_epi'] > 0 else '✗'}")
    print(f"  Δνf: {metrics['delta_vf']:+.3f} {'✓' if metrics['frequency_activation'] else '✗'}")
    print(f"  ΔNFR: {metrics['dnfr_initialized']:+.3f} {'✓' if metrics['reorganization_positive'] else '✗'}")
    
    print(f"\nContext:")
    print(f"  From latency: {'Yes' if metrics['activation_from_latency'] else 'No'}")
    print(f"  Irreversible: {'Yes' if metrics['irreversibility_marker'] else 'No'}")
    
    # Recommendations
    if metrics["emission_quality"] == "weak":
        print(f"\n⚠ RECOMMENDATION:")
        if not metrics["frequency_activation"]:
            print("  - Consider increasing νf before emission")
        if not metrics["reorganization_positive"]:
            print("  - Check ΔNFR hook configuration")

# Use it
diagnose_emission(metrics)
```

## Benefits

The extended emission metrics enable:

1. **Qualitative Analysis**: Distinguish valid from weak emissions
2. **Debugging**: Identify spurious or ineffective activations
3. **Research**: Study emergence dynamics systematically
4. **Validation**: Verify expected structural effects
5. **Traceability**: Full temporal tracking with timestamps

## Backward Compatibility

All legacy metric fields are preserved:
- `epi_final`, `vf_final`, `dnfr_final`
- `activation_strength`, `is_activated`

Existing code continues to work without changes.

## See Also

- [Emission Operator Documentation](../api/operators.md#emission-al)
- [TNFR.pdf §2.2.1](../../../TNFR.pdf) - AL (Emisión fundacional)
- [Emission Metrics Demo](../../../examples/emission_metrics_demo.py)
- [Emission Irreversibility Tests](../../../tests/unit/operators/test_emission_irreversibility.py)
