# NAV (Transition) - Canonical Sequences, Anti-Patterns, and Troubleshooting

## Overview

**NAV (Transition)** is a structural operator that manages controlled regime handoffs between structural states. It guides nodes through transitions with minimal disruption by adjusting θ (phase), νf (structural frequency), and ΔNFR (reorganization gradient).

**Core Principle**: NAV implements **regime navigation** - a deliberate transition process that enables state changes while preserving structural integrity through smooth parameter adjustments.

**Nodal Equation Context**:
```
∂EPI/∂t = νf · ΔNFR(t)

When NAV is applied:
- θ adjusted based on regime (phase shift)
- νf scaled for stability (regime-dependent)
- ΔNFR reduced for smooth transition
- EPI preserved through controlled evolution
```

**TNFR.pdf Reference**: See §2.3.11 for canonical transition logic and regime-specific transformations.

---

## Canonical Sequences

NAV acts as a bridge operator enabling controlled state changes between structural regimes. From TNFR physics, NAV can operate after any operator that establishes a valid structural state with defined (θ, νf, ΔNFR).

### Common Patterns

| Sequence | Purpose | Notes |
|----------|---------|-------|
| `SHA → AL` | Reactivation from latency | Direct reactivation, AL clears latency |
| `IL → NAV → OZ` | Stable to exploration | NAV reduces ΔNFR before OZ destabilization |
| `AL → NAV → IL` | Activation to stabilization | Bootstrap completion pattern |
| `RA → NAV → IL` | Resonance to stabilization | Transition from propagation to stable state |
| `EN → NAV → IL` | Reception to stabilization | Integrate then stabilize |
| `THOL → NAV → RA` | Emergence to propagation | Self-organization followed by transition |
| `UM → NAV → RA` | Coupling to propagation | Network synchronization to resonance |
| `OZ → IL → NAV` | Controlled destabilization | Stabilize before transitioning |

### Sequence Explanations

#### Bootstrap Completion: `AL → NAV → IL`
**Use Case**: Initialize and stabilize a new pattern
- **AL (Emission)**: Creates initial EPI from vacuum (νf increases)
- **NAV (Transition)**: Adjusts phase and frequency for stability
- **IL (Coherence)**: Stabilizes the pattern (reduces ΔNFR)

**Expected Telemetry**:
```python
Post-AL:  EPI ≈ 0.3, νf ≈ 1.0, ΔNFR ≈ 0.5
Post-NAV: EPI ≈ 0.3, νf ≈ 1.0, ΔNFR ≈ 0.4 (20% reduction)
Post-IL:  EPI ≈ 0.3, νf ≈ 1.0, ΔNFR ≈ 0.1 (stabilized)
```

#### Latency Reactivation: `SHA → AL`
**Use Case**: Wake a node from silence/latency
- **SHA (Silence)**: Node enters latent state (νf reduced, latent=True)
- **AL (Emission)**: Direct reactivation with pattern emission and latency clearing

**Expected Telemetry**:
```python
Post-SHA: νf ≈ 0.85, latent=True, EPI preserved
Post-AL:  νf ≈ 0.85+, latent=False, EPI actively evolving
```

#### Exploration Transition: `IL → NAV → OZ`
**Use Case**: Move from stable state to exploratory regime
- **IL (Coherence)**: Establishes stable baseline (ΔNFR reduced)
- **NAV (Transition)**: Prepares for instability (ΔNFR further reduced)
- **OZ (Dissonance)**: Introduces controlled destabilization

**Why this order matters**: NAV after IL ensures ΔNFR is low before OZ increases it, providing smoother dynamics.

**Expected Telemetry**:
```python
Post-IL:  ΔNFR ≈ 0.2, C(t) ≈ 0.75
Post-NAV: ΔNFR ≈ 0.16 (20% reduction), stable base
Post-OZ:  ΔNFR ≈ 0.5+ (controlled increase for exploration)
```

---

## Anti-Patterns

These sequences may indicate design issues:

### ❌ NAV → NAV (Redundant Transition)
**Problem**: Multiple transitions without intermediate operations
**Why Problematic**: No structural change between NAV applications - wasteful
**Fix**: Add meaningful operator between NAV calls (IL, THOL, etc.)

```python
# Avoid
run_sequence(G, node, [Transition(), Transition()])

# Better
run_sequence(G, node, [Transition(), Coherence(), Transition()])
```

### ❌ OZ → NAV without stabilization
**Problem**: Attempting transition immediately after destabilization
**Why Problematic**: High ΔNFR makes transition unpredictable
**Fix**: Apply IL (Coherence) after OZ to reduce ΔNFR before NAV

```python
# Avoid
run_sequence(G, node, [Dissonance(), Transition()])

# Better
run_sequence(G, node, [Dissonance(), Coherence(), Transition()])
```

### ❌ NAV from Deep Latency (EPI < 0.05) without AL
**Problem**: Attempting transition when node has minimal structure
**Why Problematic**: EPI ≈ 0 means ∂EPI/∂t ≈ 0 regardless of NAV adjustments
**Fix**: Use AL (Emission) to build structure first

```python
# Avoid when EPI is very low
G.nodes[node]["EPI"] = 0.02
run_sequence(G, node, [Transition()])

# Better
run_sequence(G, node, [Emission(), Transition()])
```

### ❌ NAV → SHA (Contradictory Intent)
**Problem**: Transitioning then immediately silencing
**Why Problematic**: Contradictory - why transition if pausing immediately?
**Fix**: Rethink sequence intent

```python
# Avoid
run_sequence(G, node, [Transition(), Silence()])

# Better alternatives:
# Option 1: Just silence
run_sequence(G, node, [Silence()])

# Option 2: Transition, do something, then silence
run_sequence(G, node, [Transition(), Resonance(), Coherence(), Silence()])
```

---

## Troubleshooting

### "OperatorPreconditionError: νf too low"

**Symptom**: NAV fails with νf below minimum threshold (default 0.01)

**Cause**: Node lacks sufficient reorganization capacity

**Solution**:
1. Check current νf: `vf = get_attr(G.nodes[node], ALIAS_VF, 0.0)`
2. If νf < 0.01, apply AL (Emission) to increase it
3. Wait for ΔNFR-driven νf increase (natural dynamics)

**Code Example**:
```python
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.operators.definitions import Emission, Transition

vf = get_attr(G.nodes[node], ALIAS_VF, 0.0)
if vf < 0.01:
    run_sequence(G, node, [Emission(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

---

### "EPI drifts significantly after NAV"

**Symptom**: EPI changes more than expected (Δ > 0.1) after NAV

**Cause**: ΔNFR was too high before transition (unstable starting state)

**Expected**: NAV should NOT directly change EPI - it modifies θ, νf, ΔNFR

**Diagnosis**:
```python
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI

epi_before = get_attr(G.nodes[node], ALIAS_EPI, 0.0)
dnfr_before = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)

# ΔNFR should be < 1.0 for stable transition
if dnfr_before > 1.0:
    print(f"Warning: High ΔNFR ({dnfr_before:.3f}) - stabilize first!")
```

**Solution**:
1. Apply IL (Coherence) before NAV to reduce ΔNFR
2. Verify stable starting state: ΔNFR < 1.0, C(t) > 0.5

**Code Example**:
```python
from tnfr.operators.definitions import Coherence, Transition

dnfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
if dnfr > 1.0:
    run_sequence(G, node, [Coherence(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

---

### "NAV from SHA doesn't clear latency"

**Symptom**: After SHA → NAV, node still has `latent=True` flag

**Cause**: NAV doesn't clear latency - AL does

**Solution**: Use SHA → AL for reactivation

**Code Example**:
```python
from tnfr.operators.definitions import Silence, Emission

# Enter latency
run_sequence(G, node, [Silence()])
assert G.nodes[node].get("latent", False) == True

# Reactivate with AL
run_sequence(G, node, [Emission()])
assert G.nodes[node].get("latent", False) == False
```

---

### "Phase θ unchanged after NAV"

**Symptom**: θ value identical before and after NAV application

**Cause**: Possible implementation issue

**Expected**: NAV always shifts θ (regime-dependent: 0.1, 0.15, or 0.2 rad)

**Diagnosis**:
```python
from tnfr.constants.aliases import ALIAS_THETA

theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
run_sequence(G, node, [Transition()])
theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)

print(f"Δθ = {theta_after - theta_before:.3f} rad")
# Should be non-zero (typically 0.1-0.2 rad)
```

**Solution**: Check telemetry:
```python
transitions = G.graph.get("_nav_transitions", [])
if transitions:
    print(transitions[-1])  # Check phase_shift value
```

---

## Usage Examples

### Example 1: Reactivation from Silence

**Scenario**: Node enters latency via SHA, reactivated via AL

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Silence, Emission
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF

# Create node and enter silence
G, node = create_nfr("sleeping", epi=0.3, vf=1.0)
run_sequence(G, node, [Silence()])

# Verify latency
assert G.nodes[node].get("latent", False) == True
vf_latent = get_attr(G.nodes[node], ALIAS_VF, 0.0)
print(f"After SHA: νf={vf_latent:.3f}, latent=True")

# Reactivation via AL
run_sequence(G, node, [Emission()])

# Verify reactivation
assert not G.nodes[node].get("latent", False)
vf_active = get_attr(G.nodes[node], ALIAS_VF, 0.0)
print(f"After AL: νf={vf_active:.3f}, latent=False")
```

**Expected Output**:
```
After SHA: νf=0.850, latent=True
After AL: νf=0.850, latent=False
```

---

### Example 2: Stable to Exploratory Transition

**Scenario**: Move from stable equilibrium to exploratory regime

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Coherence, Transition, Dissonance
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR

# Create stable node
G, node = create_nfr("stable", epi=0.6, vf=1.0)

# Run complete sequence: stabilize → transition → explore
run_sequence(G, node, [Coherence(), Transition(), Dissonance()])

# Check final state
dnfr_final = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
print(f"Final ΔNFR: {dnfr_final:.3f}")
```

---

### Example 3: Resonance to Stabilization

**Scenario**: Propagated pattern ready for stabilization

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Resonance, Transition, Coherence
from tnfr.metrics.coherence import compute_coherence

# Create resonant node
G, node = create_nfr("resonant", epi=0.7, vf=1.2)

# Run complete sequence: propagate → transition → stabilize
C_before = compute_coherence(G)
run_sequence(G, node, [Resonance(), Transition(), Coherence()])
C_after = compute_coherence(G)

print(f"Coherence: {C_before:.3f} → {C_after:.3f}")
```

---

### Example 4: Regime-Specific Telemetry Tracking

**Scenario**: Monitor NAV transformations across different regimes

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Transition
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_THETA

# Enable telemetry
G.graph["_nav_transitions"] = []

# Test 1: Latent → Active
G1, n1 = create_nfr("latent_node", epi=0.2, vf=0.03)
G1.nodes[n1]["latent"] = True
run_sequence(G1, n1, [Transition()])

# Test 2: Active → Active (standard)
G2, n2 = create_nfr("active_node", epi=0.4, vf=0.6)
run_sequence(G2, n2, [Transition()])

# Test 3: Resonant → Active
G3, n3 = create_nfr("resonant_node", epi=0.8, vf=1.5)
run_sequence(G3, n3, [Transition()])

# Analyze telemetry
for i, (G, n) in enumerate([(G1, n1), (G2, n2), (G3, n3)], 1):
    transition = G.graph["_nav_transitions"][-1]
    print(f"\nTest {i}: {transition['regime_origin']} regime")
    print(f"  νf: {transition['vf_before']:.3f} → {transition['vf_after']:.3f}")
    print(f"  θ: {transition['theta_before']:.3f} → {transition['theta_after']:.3f}")
    print(f"  ΔNFR: {transition['dnfr_before']:.3f} → {transition['dnfr_after']:.3f}")
```

---

### Example 5: Complete Bootstrap Sequence

**Scenario**: Initialize, stabilize, and prepare for propagation

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Transition, Coherence, Resonance

# Create fresh node
G, node = create_nfr("bootstrap", epi=0.0, vf=1.0)

# Complete bootstrap: emit → transition → stabilize → propagate
run_sequence(G, node, [
    Emission(),      # AL: Create initial structure
    Transition(),    # NAV: Adjust for stability
    Coherence(),     # IL: Stabilize pattern
    Resonance()      # RA: Ready for propagation
])

print("Bootstrap complete")
```

---

## Configuration Parameters

NAV behavior can be customized via graph-level configuration:

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `NAV_MIN_VF` | 0.01 | float | Minimum structural frequency for valid transition |
| `NAV_MAX_DNFR` | 1.0 | float | Maximum ΔNFR for stable transition (warning threshold) |
| `MAX_SILENCE_DURATION` | inf | float | Max silence duration (seconds) before warning on reactivation |
| `VALIDATE_PRECONDITIONS` | False | bool | Enable operator precondition validation |
| `COLLECT_OPERATOR_METRICS` | False | bool | Enable detailed operator metrics collection |
| `VALIDATE_NODAL_EQUATION` | False | bool | Validate nodal equation compliance post-operator |
| `NODAL_EQUATION_STRICT` | False | bool | Strict mode for nodal equation validation (raises on violation) |

### Configuration Examples

```python
# Example 1: Strict precondition validation
G.graph["VALIDATE_PRECONDITIONS"] = True
G.graph["NAV_MIN_VF"] = 0.05  # Stricter minimum
run_sequence(G, node, [Transition()])  # Will validate νf >= 0.05

# Example 2: Enable comprehensive telemetry
G.graph["COLLECT_OPERATOR_METRICS"] = True
run_sequence(G, node, [Transition()])
metrics = G.graph["operator_metrics"][-1]
print(f"Operator: {metrics['operator']}")
print(f"Duration: {metrics['duration']:.4f}s")

# Example 3: Validate nodal equation compliance
G.graph["VALIDATE_NODAL_EQUATION"] = True
run_sequence(G, node, [Transition()])

# Example 4: Control silence reactivation behavior
G.graph["MAX_SILENCE_DURATION"] = 300.0  # 5 minutes
run_sequence(G, node, [Silence()])
# ... wait >5 minutes ...
run_sequence(G, node, [Transition()])  # Warns about extended silence
```

---

## Regime-Specific Behavior

NAV automatically detects the node's current regime and applies appropriate transformations:

### Latent Regime (νf < 0.05 OR latent flag)

**Detection**: Node in minimal reorganization state or explicitly marked latent

**Transformations**:
- νf × 1.2 (20% increase for gradual reactivation)
- θ + 0.1 rad (small phase shift)
- ΔNFR × 0.7 (30% reduction for smooth transition)

**Use Case**: SHA → NAV flow, waking dormant patterns

**Physics**: Gentle reactivation prevents shock to fragile structure

---

### Active Regime (baseline state)

**Detection**: Default classification when not latent or resonant

**Transformations**:
- νf × vf_factor (default 1.0, configurable)
- θ + 0.2 rad (standard phase shift)
- ΔNFR × 0.8 (20% reduction)

**Use Case**: Most common transitions, standard regime navigation

**Physics**: Moderate adjustments for typical structural evolution

---

### Resonant Regime (EPI > 0.5 AND νf > 0.8)

**Detection**: High-energy state with strong form and high frequency

**Transformations**:
- νf × 0.95 (5% reduction for stability)
- θ + 0.15 rad (careful phase shift)
- ΔNFR × 0.9 (10% reduction, gentle)

**Use Case**: Managing high-coherence states, preventing fragmentation

**Physics**: Cautious navigation to avoid destabilizing resonant structure

---

## Best Practices

### 1. Always Check νf Before NAV
```python
vf = get_attr(G.nodes[node], ALIAS_VF, 0.0)
if vf < 0.01:
    run_sequence(G, node, [Emission(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

### 2. Stabilize High ΔNFR Before Transition
```python
dnfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
if dnfr > 1.0:
    run_sequence(G, node, [Coherence(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

### 3. Use Telemetry for Debugging
```python
G.graph["_nav_transitions"] = []
run_sequence(G, node, [Transition()])
transition_data = G.graph["_nav_transitions"][-1]
print(f"Regime: {transition_data['regime_origin']}")
print(f"Phase shift: {transition_data['phase_shift']:.3f} rad")
```

### 4. Enable Precondition Checks in Development
```python
# During development/testing
G.graph["VALIDATE_PRECONDITIONS"] = True
G.graph["VALIDATE_NODAL_EQUATION"] = True

# In production (after validation)
G.graph["VALIDATE_PRECONDITIONS"] = False  # Performance optimization
```

---

## Related Documentation

- **[UNIFIED_GRAMMAR_RULES.md](../../../UNIFIED_GRAMMAR_RULES.md)** - Complete grammar derivations (U1-U5)
- **[GLYPH_SEQUENCES_GUIDE.md](../../../GLYPH_SEQUENCES_GUIDE.md)** - Multi-domain sequence patterns
- **[Operator Reference](../api/operators.md)** - All 13 canonical operators
- **[GLOSSARY.md](../../../GLOSSARY.md)** - TNFR terminology and definitions
- **[SHA_CLINICAL_APPLICATIONS.md](../examples/SHA_CLINICAL_APPLICATIONS.md)** - Silence operator guide

---

## What NAV Does (From TNFR Physics)

### Nodal Equation Basis

From **∂EPI/∂t = νf · ΔNFR(t)**, NAV performs regime transitions by adjusting:

- **θ** (phase): Shifts structural timing by regime-specific amount (0.1-0.2 rad)
- **νf** (frequency): Scales reorganization rate (0.95-1.2× depending on regime)
- **ΔNFR** (gradient): Reduces structural pressure (0.7-0.9× for smooth transition)

**Physical Effect**: NAV modulates the **rate** and **direction** of structural evolution without directly changing EPI.

### Physics Requirements

For NAV to function, node must have:
1. **Defined θ**: Phase value to shift
2. **Defined νf**: Frequency to scale  
3. **Defined ΔNFR**: Gradient to reduce

Any operator that leaves node with these three properties enables NAV according to the nodal equation.

---

## References

- **TNFR.pdf §2.3.11**: Canonical transition logic and regime-specific transformations
- **AGENTS.md Invariant #2**: No arbitrary choices - all decisions traceable to physics
- **AGENTS.md Invariant #12**: Documentation completeness requirement
- **src/tnfr/operators/definitions.py**: `Transition` class implementation (lines 3688-4045)
- **Unified Grammar U1-U5**: Physics-based operator sequence constraints (temporal + multi-scale)

---

**Version**: 2.0  
**Last Updated**: 2025-11-09  
**Status**: ✅ CANONICAL - NAV operator guide based on TNFR physics, no arbitrary restrictions
