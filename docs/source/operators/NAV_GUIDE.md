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

NAV acts as a bridge operator enabling controlled state changes. The following sequences represent canonical patterns validated by TNFR grammar (U1-U4).

| Sequence | Purpose | Grammar Status | Notes |
|----------|---------|----------------|-------|
| `SHA → NAV → AL` | Reactivation from latency | ✅ Supported | Requires `_handle_latency_transition` (implemented) |
| `IL → NAV → OZ` | Stable to exploration | ✅ Supported | NAV reduces ΔNFR before OZ destabilization |
| `AL → NAV → IL` | Activation to stabilization | ✅ Supported | Common bootstrap completion pattern |
| `NAV → ZHIR` | Enable mutation | ⚠️ Requires U4b check | ZHIR needs prior IL + recent destabilizer |
| `THOL → NAV → RA` | Emergence to propagation | ✅ Supported | NAV prepares emergent structure for resonance |
| `OZ → IL → NAV` | Controlled destabilization | ✅ Supported | IL stabilizes before NAV transition |
| `UM → NAV → RA` | Coupling to propagation | ✅ Supported | NAV smooths transition to resonance |
| `EN → NAV → IL` | Reception to stabilization | ✅ Supported | Integrate then stabilize received patterns |

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

#### Latency Reactivation: `SHA → NAV → AL`
**Use Case**: Wake a node from silence/latency
- **SHA (Silence)**: Node enters latent state (νf → 0, latent=True)
- **NAV (Transition)**: Clears latency flag, gradual νf increase (×1.2)
- **AL (Emission)**: Full reactivation with pattern emission

**Expected Telemetry**:
```python
Post-SHA: νf ≈ 0.05 (near-zero), latent=True, EPI preserved
Post-NAV: νf ≈ 0.06 (20% increase), latent=False, θ +0.1 rad
Post-AL:  νf ≈ 1.0+, EPI actively evolving
```

#### Exploration Transition: `IL → NAV → OZ`
**Use Case**: Move from stable state to exploratory regime
- **IL (Coherence)**: Establishes stable baseline (ΔNFR reduced)
- **NAV (Transition)**: Prepares for instability (ΔNFR further reduced)
- **OZ (Dissonance)**: Introduces controlled destabilization

**Why this order matters**: NAV after IL ensures ΔNFR is low before OZ increases it. Direct `IL → OZ` would work, but `IL → NAV → OZ` provides smoother dynamics.

**Expected Telemetry**:
```python
Post-IL:  ΔNFR ≈ 0.2, C(t) ≈ 0.75
Post-NAV: ΔNFR ≈ 0.16 (20% reduction), stable base
Post-OZ:  ΔNFR ≈ 0.5+ (controlled increase for exploration)
```

---

## Anti-Patterns

These sequences violate TNFR physics or grammar constraints and should be avoided:

### ❌ NAV → NAV (Redundant Transition)
**Problem**: Multiple transitions without intermediate stabilization
**Why Invalid**: No structural change between NAV applications - wasteful and potentially destabilizing
**Fix**: Add stabilizer (IL, THOL) or other meaningful operator between NAV calls

```python
# Bad
run_sequence(G, node, [Transition(), Transition()])

# Good
run_sequence(G, node, [Transition(), Coherence(), Transition()])
```

### ❌ OZ → NAV (High ΔNFR Transition)
**Problem**: Attempting transition immediately after destabilization
**Why Invalid**: High ΔNFR makes transition chaotic and unpredictable
**Physics**: NAV expects ΔNFR < 1.0 for stable regime handoff
**Fix**: Apply IL (Coherence) after OZ to reduce ΔNFR before NAV

```python
# Bad - chaotic transition
run_sequence(G, node, [Dissonance(), Transition()])

# Good - stabilize first
run_sequence(G, node, [Dissonance(), Coherence(), Transition()])
```

### ❌ NAV from Deep Latency (EPI < 0.05) without AL
**Problem**: Attempting transition when node has minimal structure
**Why Invalid**: NAV assumes viable EPI to transition; deep latency needs regeneration
**Physics**: EPI ≈ 0 means ∂EPI/∂t ≈ 0 regardless of NAV adjustments
**Fix**: Use AL (Emission) before or after NAV to build structure

```python
# Bad - insufficient structure
G.nodes[node]["EPI"] = 0.02
run_sequence(G, node, [Transition()])

# Good - regenerate then transition
run_sequence(G, node, [Transition(), Emission()])
# Or: Emission first, then transition
run_sequence(G, node, [Emission(), Transition()])
```

### ❌ NAV → SHA (Contradictory Intent)
**Problem**: Transitioning then immediately silencing
**Why Invalid**: Contradictory intent - why transition if pausing immediately?
**Structural Logic**: NAV prepares for activity; SHA freezes activity
**Fix**: Rethink sequence intent or add intermediate operators

```python
# Bad - contradictory
run_sequence(G, node, [Transition(), Silence()])

# Better alternatives:
# Option 1: Just silence (no transition needed)
run_sequence(G, node, [Silence()])

# Option 2: Transition, do something, then silence
run_sequence(G, node, [Transition(), Resonance(), Coherence(), Silence()])
```

### ❌ NAV → ZHIR without Prior IL
**Problem**: Attempting mutation without stable base
**Why Invalid**: Violates U4b (ZHIR requires prior IL within sequence)
**Grammar**: ZHIR contract requires stabilized structure before phase transformation
**Fix**: Always apply IL before ZHIR-enabling NAV

```python
# Bad - violates U4b
run_sequence(G, node, [Transition(), Mutation()])

# Good - satisfies U4b
run_sequence(G, node, [Coherence(), Transition(), Mutation()])
```

---

## Troubleshooting

### "OperatorPreconditionError: νf too low"

**Symptom**: NAV fails with νf below minimum threshold (default 0.01)

**Cause**: Node lacks sufficient reorganization capacity for transition

**Solution**:
1. Check current νf: `vf = get_attr(G.nodes[node], ALIAS_VF, 0.0)`
2. If νf < 0.01, apply AL (Emission) to increase it
3. If in latency, use SHA → NAV → AL sequence
4. Wait for ΔNFR-driven νf increase (natural dynamics)

**Code Example**:
```python
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.operators.definitions import Emission, Transition

vf = get_attr(G.nodes[node], ALIAS_VF, 0.0)
if vf < 0.01:
    # Boost νf before transition
    run_sequence(G, node, [Emission(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

---

### "NAV applied but node behavior unchanged"

**Symptom**: NAV completes without error, but θ, νf, ΔNFR show no changes

**Cause**: Regime detection may classify node as "active" with no override parameters

**Diagnosis**:
1. Check telemetry: `G.graph.get("_nav_transitions", [])`
2. Verify regime classification: `_detect_regime(G, node)`
3. Inspect νf, EPI values to understand regime

**Solution**:
- If intentional (active regime, standard behavior), NAV is working correctly
- If unexpected, provide override parameters:
  ```python
  run_sequence(G, node, [Transition(phase_shift=0.3, vf_factor=1.1)])
  ```
- Enable metrics collection to track changes:
  ```python
  G.graph["COLLECT_OPERATOR_METRICS"] = True
  run_sequence(G, node, [Transition()])
  print(G.graph["operator_metrics"][-1])
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
3. Use NAV_MAX_DNFR config to enforce threshold:
   ```python
   G.graph["NAV_MAX_DNFR"] = 0.8  # Strict threshold
   ```

**Code Example**:
```python
# Check and stabilize before transition
from tnfr.operators.definitions import Coherence, Transition

dnfr = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
if dnfr > 1.0:
    # Stabilize first
    run_sequence(G, node, [Coherence(), Transition()])
else:
    run_sequence(G, node, [Transition()])
```

---

### "NAV from SHA doesn't clear latency"

**Symptom**: After SHA → NAV, node still has `latent=True` flag

**Cause**: Implementation bug or incorrect sequence

**Expected Behavior**: NAV's `_handle_latency_transition` should clear:
- `latent` flag
- `latency_start_time`
- `preserved_epi`

**Diagnosis**:
```python
# Check latency state before and after
print("Before NAV:", G.nodes[node].get("latent", False))
run_sequence(G, node, [Transition()])
print("After NAV:", G.nodes[node].get("latent", False))
```

**Solution**:
1. Verify NAV implementation includes `_handle_latency_transition` (it does in current codebase)
2. Ensure SHA properly set latency flag:
   ```python
   run_sequence(G, node, [Silence()])
   assert G.nodes[node].get("latent", False) == True
   ```
3. Check for custom hooks or overrides that might interfere

---

### "Phase θ unchanged after NAV"

**Symptom**: θ value identical before and after NAV application

**Cause**: Regime-specific phase shift not applied (implementation issue or override)

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

**Solution**:
1. Verify `_apply_structural_transition` is being called
2. Check for `phase_shift=0.0` override in kwargs
3. Enable validation:
   ```python
   G.graph["VALIDATE_PRECONDITIONS"] = True
   ```
4. Inspect telemetry:
   ```python
   transitions = G.graph.get("_nav_transitions", [])
   print(transitions[-1])  # Check phase_shift value
   ```

---

### "OperatorPreconditionError: Invalid regime transition"

**Symptom**: Custom validation rejects NAV application

**Cause**: Sequence violates grammar constraints (e.g., U4b for NAV → ZHIR)

**Solution**:
1. Review sequence against unified grammar (U1-U4)
2. Check specific constraint:
   - **U1a**: EPI=0 requires generator first
   - **U1b**: Sequence needs closure operator
   - **U2**: Destabilizers need stabilizers
   - **U3**: Coupling/resonance needs phase check
   - **U4b**: ZHIR needs prior IL + recent destabilizer
3. Add missing operators:
   ```python
   # Bad: NAV → ZHIR without context
   run_sequence(G, node, [Transition(), Mutation()])
   
   # Good: Satisfy U4b
   run_sequence(G, node, [Coherence(), Dissonance(), Transition(), Mutation()])
   ```

**Diagnostic Tool**:
```python
from tnfr.operators.grammar import validate_grammar

sequence = [Transition(), Mutation()]
is_valid = validate_grammar(sequence, epi_initial=0.5)
if not is_valid:
    print("Sequence violates grammar constraints")
```

---

## Usage Examples

### Example 1: Reactivation from Silence

**Scenario**: Node enters latency via SHA, later reactivated via NAV → AL

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Silence, Transition, Emission
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_VF

# Create node and enter silence
G, node = create_nfr("sleeping", epi=0.3, vf=1.0)
run_sequence(G, node, [Silence()])

# Verify latency
assert G.nodes[node].get("latent", False) == True
vf_latent = get_attr(G.nodes[node], ALIAS_VF, 0.0)
assert vf_latent < 0.1, "νf should be near-zero in latency"

# Reactivation via NAV → AL
run_sequence(G, node, [Transition(), Emission()])

# Verify reactivation
assert not G.nodes[node].get("latent", False), "Latency should be cleared"
vf_active = get_attr(G.nodes[node], ALIAS_VF, 0.0)
assert vf_active > 0.1, "νf should increase after reactivation"

print(f"Reactivation successful: νf {vf_latent:.3f} → {vf_active:.3f}")
```

**Expected Output**:
```
Reactivation successful: νf 0.050 → 1.200
```

---

### Example 2: Stable to Exploratory Transition

**Scenario**: Move from stable equilibrium to exploratory regime safely

```python
from tnfr.operators.definitions import Coherence, Transition, Dissonance
from tnfr.constants.aliases import ALIAS_DNFR

# Create stable node
G, node = create_nfr("stable", epi=0.6, vf=1.0)
run_sequence(G, node, [Coherence()])

# Check stability
dnfr_stable = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
print(f"Stable ΔNFR: {dnfr_stable:.3f}")

# Transition to exploration (safe via NAV)
run_sequence(G, node, [Transition(), Dissonance()])

# Verify controlled destabilization
dnfr_explore = get_attr(G.nodes[node], ALIAS_DNFR, 0.0)
print(f"Exploratory ΔNFR: {dnfr_explore:.3f}")
assert dnfr_explore > dnfr_stable, "ΔNFR should increase for exploration"
```

**Expected Output**:
```
Stable ΔNFR: 0.150
Exploratory ΔNFR: 0.520
```

---

### Example 3: Emergence to Propagation

**Scenario**: Self-organized pattern ready for network propagation

```python
from tnfr.operators.definitions import SelfOrganization, Transition, Resonance
from tnfr.metrics.coherence import compute_coherence

# Create emergent node
G, node = create_nfr("emergent", epi=0.7, vf=1.2)
run_sequence(G, node, [SelfOrganization()])

# Verify emergence (sub-EPIs created)
assert "sub_epis" in G.nodes[node], "THOL should create sub-EPIs"

# Transition to propagation
C_before = compute_coherence(G)
run_sequence(G, node, [Transition(), Resonance()])
C_after = compute_coherence(G)

print(f"Coherence: {C_before:.3f} → {C_after:.3f}")
assert C_after >= C_before, "Resonance should maintain/increase coherence"
```

**Expected Output**:
```
Coherence: 0.720 → 0.765
```

---

### Example 4: Regime-Specific Telemetry Tracking

**Scenario**: Monitor NAV transformations across different regimes

```python
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

**Expected Output**:
```
Test 1: latent regime
  νf: 0.030 → 0.036
  θ: 0.000 → 0.100
  ΔNFR: 0.000 → 0.000

Test 2: active regime
  νf: 0.600 → 0.600
  θ: 0.000 → 0.200
  ΔNFR: 0.000 → 0.000

Test 3: resonant regime
  νf: 1.500 → 1.425
  θ: 0.000 → 0.150
  ΔNFR: 0.000 → 0.000
```

---

### Example 5: Safe Mutation Preparation

**Scenario**: Prepare node for ZHIR (Mutation) with proper grammar compliance

```python
from tnfr.operators.definitions import Coherence, Dissonance, Transition, Mutation
from tnfr.operators.grammar import validate_grammar

# Build grammar-compliant sequence
sequence = [
    Coherence(),    # U4b: Prior IL required
    Dissonance(),   # U4b: Recent destabilizer (~3 ops before ZHIR)
    Transition(),   # Enable mutation regime
    Mutation(),     # ZHIR: Phase transformation
]

# Validate before execution
is_valid = validate_grammar(sequence, epi_initial=0.5)
assert is_valid, "Sequence must satisfy unified grammar"

# Execute sequence
G, node = create_nfr("mutable", epi=0.5, vf=1.0)
theta_before = get_attr(G.nodes[node], ALIAS_THETA, 0.0)

run_sequence(G, node, sequence)

theta_after = get_attr(G.nodes[node], ALIAS_THETA, 0.0)
print(f"Phase transformation: θ {theta_before:.3f} → {theta_after:.3f} rad")
```

**Expected Output**:
```
Phase transformation: θ 0.000 → 1.257 rad
```

---

## Configuration Parameters

NAV behavior can be customized via graph-level configuration:

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `NAV_MIN_VF` | 0.01 | float | Minimum structural frequency for valid transition |
| `NAV_MAX_DNFR` | 1.0 | float | Maximum ΔNFR for stable transition (warning threshold) |
| `NAV_STRICT_SEQUENCE_CHECK` | False | bool | Enable strict sequence validation warnings |
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
print(f"ΔNFR change: {metrics['dnfr_change']:.3f}")

# Example 3: Validate nodal equation compliance
G.graph["VALIDATE_NODAL_EQUATION"] = True
G.graph["NODAL_EQUATION_STRICT"] = True
run_sequence(G, node, [Transition()])  # Raises if ∂EPI/∂t ≠ νf·ΔNFR

# Example 4: Control silence reactivation behavior
G.graph["MAX_SILENCE_DURATION"] = 300.0  # 5 minutes
run_sequence(G, node, [Silence()])
# ... wait >5 minutes ...
run_sequence(G, node, [Transition()])  # Warns about extended silence
```

### Accessing Configuration

```python
# Check current configuration
nav_min_vf = G.graph.get("NAV_MIN_VF", 0.01)
print(f"Current NAV_MIN_VF: {nav_min_vf}")

# Temporarily override for single operation
run_sequence(G, node, [Transition(phase_shift=0.25, vf_factor=1.05)])

# Batch configuration
config = {
    "VALIDATE_PRECONDITIONS": True,
    "COLLECT_OPERATOR_METRICS": True,
    "NAV_MAX_DNFR": 0.8,
}
G.graph.update(config)
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

### 4. Validate Grammar Before Complex Sequences
```python
from tnfr.operators.grammar import validate_grammar

sequence = [Coherence(), Transition(), Mutation()]
if not validate_grammar(sequence, epi_initial=0.5):
    # Adjust sequence to satisfy constraints
    sequence = [Coherence(), Dissonance(), Transition(), Mutation()]
```

### 5. Enable Precondition Checks in Development
```python
# During development/testing
G.graph["VALIDATE_PRECONDITIONS"] = True
G.graph["VALIDATE_NODAL_EQUATION"] = True

# In production (after validation)
G.graph["VALIDATE_PRECONDITIONS"] = False  # Performance optimization
```

---

## Related Documentation

- **[UNIFIED_GRAMMAR_RULES.md](../../../UNIFIED_GRAMMAR_RULES.md)** - Complete grammar derivations (U1-U4)
- **[GLYPH_SEQUENCES_GUIDE.md](../../../GLYPH_SEQUENCES_GUIDE.md)** - Multi-domain sequence patterns
- **[Operator Reference](../api/operators.md)** - All 13 canonical operators
- **[GLOSSARY.md](../../../GLOSSARY.md)** - TNFR terminology and definitions
- **[SHA_CLINICAL_APPLICATIONS.md](../examples/SHA_CLINICAL_APPLICATIONS.md)** - Silence operator guide (template)

---

## References

- **TNFR.pdf §2.3.11**: Canonical transition logic and regime-specific transformations
- **AGENTS.md Invariant #12**: Documentation completeness requirement
- **src/tnfr/operators/definitions.py**: `Transition` class implementation (lines 3688-4045)
- **Unified Grammar U1-U4**: Physics-based operator sequence constraints

---

**Version**: 1.0  
**Last Updated**: 2025-11-09  
**Status**: ✅ CANONICAL - Complete NAV operator guide
