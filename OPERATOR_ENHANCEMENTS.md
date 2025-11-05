# Operator Enhancements: Preconditions and Metrics

This document describes the enhanced operator implementation with precondition validation and metrics collection capabilities added to the TNFR engine.

## Overview

All 13 structural operators now have:
1. **Specific precondition validators** - Ensure operators are applied in valid structural states
2. **Operator-specific metrics** - Collect telemetry about structural effects
3. **Backward compatibility** - New features are opt-in via graph configuration flags

## Quick Start

### Enable Precondition Validation

```python
import networkx as nx
from tnfr.operators.definitions import Emission, Dissonance
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY

# Create graph and enable validation
G = nx.DiGraph()
G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True

# Add a node with high EPI
G.add_node("n1", **{EPI_PRIMARY: 0.9, VF_PRIMARY: 1.0})

# Emission requires low EPI - this will raise an error
try:
    Emission()(G, "n1")
except OperatorPreconditionError as e:
    print(f"Precondition failed: {e}")
    # Output: "Emission: Node already active (EPI=0.900 >= 0.800)"
```

### Enable Metrics Collection

```python
import networkx as nx
from tnfr.operators.definitions import Coherence
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY, DNFR_PRIMARY

# Create graph and enable metrics
G = nx.DiGraph()
G.graph["COLLECT_OPERATOR_METRICS"] = True

# Add a node
G.add_node("n1", **{EPI_PRIMARY: 0.5, VF_PRIMARY: 1.0, DNFR_PRIMARY: 0.15})

# Apply operator (requires full TNFR dynamics setup)
# After operator application, metrics are available:
# metrics = G.graph["operator_metrics"][-1]
# print(f"ΔNFR reduction: {metrics['dnfr_reduction']}")
# print(f"Stability gain: {metrics['stability_gain']}")
```

## Operator-Specific Details

### AL - Emission

**Preconditions:**
- EPI must be below activation threshold (default: 0.8)
- Node must be in latent or low-activation state

**Metrics:**
- `delta_epi`: Change in Primary Information Structure
- `activation_strength`: Magnitude of activation
- `is_activated`: Boolean indicating if node crossed activation threshold

```python
G.graph["AL_MAX_EPI_FOR_EMISSION"] = 0.7  # Customize threshold
```

### EN - Reception

**Preconditions:**
- Node must have neighbors to receive energy from

**Metrics:**
- `delta_epi`: Change in EPI
- `neighbor_count`: Number of coupled neighbors
- `neighbor_epi_mean`: Average neighbor EPI
- `integration_strength`: Effectiveness of energy integration

### IL - Coherence

**Preconditions:**
- ΔNFR must be > 0 (something to stabilize)

**Metrics:**
- `dnfr_reduction`: Amount ΔNFR decreased
- `stability_gain`: Improvement in structural stability
- `is_stabilized`: Boolean indicating if ΔNFR < threshold

```python
G.graph["IL_MIN_DNFR"] = 1e-5  # Customize minimum ΔNFR threshold
```

### OZ - Dissonance

**Preconditions:**
- Structural frequency (νf) must be > minimum threshold (default: 0.01)

**Metrics:**
- `dnfr_increase`: Amount ΔNFR increased
- `bifurcation_risk`: Boolean indicating potential phase transition
- `dissonance_level`: Magnitude of structural tension
- `theta_shift`: Phase change magnitude

```python
G.graph["OZ_MIN_VF"] = 0.02  # Customize minimum νf
G.graph["OZ_BIFURCATION_THRESHOLD"] = 0.5  # Customize bifurcation threshold
```

### UM - Coupling

**Preconditions:**
- Graph must have other nodes to couple with

**Metrics:**
- `theta_shift`: Phase alignment change
- `phase_alignment`: Synchrony with neighbors (0.0 to 1.0)
- `neighbor_count`: Number of coupled nodes

### RA - Resonance

**Preconditions:**
- Node must have neighbors for resonance propagation

**Metrics:**
- `delta_epi`: EPI change via resonance
- `resonance_strength`: Propagation effectiveness
- `propagation_successful`: Boolean indicating effective resonance

### SHA - Silence

**Preconditions:**
- Structural frequency must be > minimum to reduce

**Metrics:**
- `vf_reduction`: Amount νf decreased
- `epi_preservation`: Verification EPI remained stable
- `is_silent`: Boolean indicating νf < threshold

```python
G.graph["SHA_MIN_VF"] = 0.02  # Customize minimum νf
```

### VAL - Expansion

**Preconditions:**
- Structural frequency must be below maximum threshold

**Metrics:**
- `vf_increase`: Amount νf increased
- `expansion_factor`: Multiplicative change in νf
- `delta_epi`: Associated EPI change

```python
G.graph["VAL_MAX_VF"] = 15.0  # Customize maximum νf
```

### NUL - Contraction

**Preconditions:**
- Structural frequency must be > minimum threshold

**Metrics:**
- `vf_decrease`: Amount νf decreased
- `contraction_factor`: Multiplicative change in νf
- `delta_epi`: Associated EPI change

```python
G.graph["NUL_MIN_VF"] = 0.05  # Customize minimum νf
```

### THOL - Self-organization

**Preconditions:**
- EPI must exceed minimum threshold for nested structure formation (default: 0.3)

**Metrics:**
- `nested_epi_count`: Number of sub-EPIs generated
- `cascade_active`: Boolean indicating self-organization cascade
- `delta_epi`, `delta_vf`: Changes in primary variables

```python
G.graph["THOL_MIN_EPI"] = 0.4  # Customize minimum EPI
```

### ZHIR - Mutation

**Preconditions:**
- Structural frequency must be > minimum for meaningful phase transition

**Metrics:**
- `theta_shift`: Magnitude of phase change
- `phase_change`: Boolean indicating significant structural transition
- `delta_epi`: Associated form change

```python
G.graph["ZHIR_MIN_VF"] = 0.1  # Customize minimum νf
```

### NAV - Transition

**Preconditions:**
- Structural frequency must be > minimum for controlled handoff

**Metrics:**
- `dnfr_change`: ΔNFR rebalancing magnitude
- `transition_complete`: Boolean indicating ΔNFR aligned with νf
- `theta_shift`: Phase adjustment during transition

```python
G.graph["NAV_MIN_VF"] = 0.02  # Customize minimum νf
```

### REMESH - Recursivity

**Preconditions:**
- Network must have minimum number of nodes (default: 2)

**Metrics:**
- `fractal_depth`: Number of recursive pattern echoes
- `multi_scale_active`: Boolean indicating fractal propagation
- `delta_epi`, `delta_vf`: Changes in primary variables

```python
G.graph["REMESH_MIN_NODES"] = 5  # Customize minimum network size
```

## Advanced Usage

### Per-Operator Control

Enable validation or metrics for specific operator calls:

```python
# Enable validation for this call only
Emission()(G, "n1", validate_preconditions=True)

# Enable metrics for this call only
Coherence()(G, "n1", collect_metrics=True)
```

### Analyzing Collected Metrics

```python
# Enable global metrics collection
G.graph["COLLECT_OPERATOR_METRICS"] = True

# Run operator sequence...

# Analyze collected metrics
for metric in G.graph["operator_metrics"]:
    op_name = metric["operator"]
    glyph = metric["glyph"]
    print(f"{op_name} ({glyph}):")
    print(f"  ΔEPI: {metric.get('delta_epi', 'N/A')}")
    print(f"  Δνf: {metric.get('delta_vf', 'N/A')}")
```

### Custom Thresholds

All thresholds are configurable via graph metadata:

```python
G.graph.update({
    # Emission thresholds
    "AL_MAX_EPI_FOR_EMISSION": 0.75,
    
    # Dissonance thresholds
    "OZ_MIN_VF": 0.015,
    "OZ_BIFURCATION_THRESHOLD": 0.6,
    
    # Self-organization thresholds
    "THOL_MIN_EPI": 0.35,
    
    # ... and so on for all operators
})
```

## TNFR Canonical Invariants

These enhancements maintain all TNFR canonical invariants:

1. ✅ **EPI as coherent form** - Operators only change EPI via structural transformations
2. ✅ **Structural units** - νf expressed in Hz_str, validated and tracked
3. ✅ **ΔNFR semantics** - Metrics track reorganization rate changes
4. ✅ **Operator closure** - Preconditions ensure valid operator composition
5. ✅ **Phase check** - Coupling validates phase synchrony explicitly
6. ✅ **Node birth/collapse** - Emission validates activation conditions
7. ✅ **Operational fractality** - Self-organization tracks nested EPIs
8. ✅ **Controlled determinism** - Metrics enable structural traceability
9. ✅ **Structural metrics** - C(t), Si, phase, νf all tracked
10. ✅ **Domain neutrality** - Thresholds are configurable, no hard-coded assumptions

## Backward Compatibility

By default, operators behave exactly as before. The new features are opt-in:

```python
# Traditional usage - no changes required
G = nx.DiGraph()
G.add_node("n1", **{EPI_PRIMARY: 0.5})
Emission()(G, "n1")  # Works as before

# Enhanced usage - explicit opt-in
G.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
G.graph["COLLECT_OPERATOR_METRICS"] = True
```

## Testing

Run the operator enhancement tests:

```bash
pytest tests/unit/operators/test_operator_enhancements.py -v
```

All existing tests continue to pass, ensuring backward compatibility.
