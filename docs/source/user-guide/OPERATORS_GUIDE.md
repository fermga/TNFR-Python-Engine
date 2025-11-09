# Operators Guide: The 13 Structural Operators

[Home](../home.md) › Operators Guide

This guide provides a comprehensive reference for TNFR's 13 canonical structural operators. These operators are the **only valid way** to modify networks in TNFR, ensuring all changes are traceable, coherent, and reproducible.

> **📖 Visual Guide**: For detailed visual explanations with ASCII diagrams, flow charts, and interactive examples, see the [Operators Visual Guide](../api/OPERATORS_VISUAL_GUIDE.md).

## Overview

Structural operators are **resonant transformations** that reorganize coherence while preserving TNFR invariants. Think of them as **musical gestures** rather than mechanical operations - each operator creates a specific type of structural change that respects the natural resonance of the system.

### Why Operators?

In TNFR, every change must go through an operator. This ensures:
- ✅ **Traceability**: Every reorganization is observable
- ✅ **Coherence**: Changes preserve structural integrity
- ✅ **Reproducibility**: Same conditions → same outcomes
- ✅ **Composability**: Operators combine into complex sequences

### Operator Closure

All operator compositions yield valid TNFR states. Any new function must map to existing operators or be defined as one.

## The 13 Canonical Operators

### 1. Emission (AL) 🎵

**Function**: Initiates a resonant pattern

**Effect**: 
- Increases νf (structural frequency)
- Creates positive ΔNFR
- Starts pattern propagation

**When to use**: 
- Starting new patterns
- Launching trajectories
- Activating dormant nodes

**Mathematical representation**: φ(νf, θ)

**Example**:
```python
from tnfr.operators import Emission

# Initiate emission from a node
Emission()(G, source_node)

# Check increased frequency
print(f"νf: {G.nodes[source_node]['nf']:.2f} Hz_str")
```

**Contracts**:
- Must increase νf
- Creates positive ΔNFR
- Does not break existing couplings

---

### 2. Reception (EN) 📡

**Function**: Receives and integrates external patterns

**Effect**:
- Updates EPI based on incoming resonance
- Adjusts phase to synchronize
- Integrates network information

**When to use**:
- Gathering information from neighbors
- Network listening
- Synchronizing with environment

**Mathematical representation**: ∫ ψ(x, t) dx

**Example**:
```python
from tnfr.operators import Reception

# Node receives from neighbors
Reception()(G, target_node)

# Check updated EPI
print(f"New EPI: {G.nodes[target_node]['epi']}")
```

**Contracts**:
- Preserves node identity
- Updates EPI coherently
- Maintains phase consistency

---

### 3. Coherence (IL) 🔒

**Function**: Stabilizes structural form

**Effect**:
- Increases C(t) (total coherence)
- Reduces |ΔNFR|
- Consolidates structure

**When to use**:
- After changes, to consolidate
- Stabilizing unstable networks
- Reducing reorganization pressure

**Mathematical representation**: ∂EPI/∂t → 0 when ΔNFR → 0

**Example**:
```python
from tnfr.operators import Coherence
from tnfr.metrics import total_coherence

# Stabilize network
C_before = total_coherence(G)
Coherence()(G)
C_after = total_coherence(G)

print(f"Coherence: {C_before:.3f} → {C_after:.3f}")
```

**Contracts**:
- Must not decrease C(t) (except controlled dissonance)
- Reduces ΔNFR
- Preserves network topology

---

### 4. Dissonance (OZ) ⚡

**Function**: Introduces controlled instability

**Effect**:
- Increases |ΔNFR|
- May trigger bifurcation if ∂²EPI/∂t² > τ
- Creates exploration space

**When to use**:
- Breaking out of local optima
- Exploration and innovation
- Testing network resilience

**Mathematical representation**: ΔNFR(t) > νf

**Example**:
```python
from tnfr.operators import Dissonance

# Introduce controlled instability
Dissonance()(G, intensity=0.5)

# Monitor for bifurcation
for node in G.nodes():
    delta_nfr = G.nodes[node]['delta_nfr']
    if abs(delta_nfr) > threshold:
        print(f"Node {node} approaching bifurcation")
```

**Contracts**:
- Must increase |ΔNFR|
- May trigger bifurcation (by design)
- Must remain controllable

---

### 5. Coupling (UM) 🔗

**Function**: Creates structural links between nodes and synchronizes their dynamics

**Effect**:
- Phase synchronization: φᵢ(t) ≈ φⱼ(t)
- Structural frequency synchronization: νf,ᵢ → νf,ⱼ
- ΔNFR reduction through mutual stabilization
- Information exchange enabled
- Network connectivity increased

**When to use**:
- Network formation
- Connecting isolated nodes
- Creating communication pathways
- Synchronizing reorganization rates between coupled systems
- Stabilizing reorganization pressure in coupled networks

**Mathematical representation**: 
- Phase: φᵢ(t) ≈ φⱼ(t)
- Frequency: νf,ᵢ(t+1) = νf,ᵢ(t) + k_vf · (⟨νf,neighbors⟩ - νf,ᵢ(t))
- ΔNFR stabilization: ΔNFR(t+1) = ΔNFR(t) · (1 - k_dnfr · alignment)
  - Where alignment ∈ [0, 1] is based on phase coherence with neighbors
  - Higher alignment → stronger ΔNFR reduction

**Example**:
```python
from tnfr.operators import apply_glyph
from tnfr.alias import set_vf, get_attr
from tnfr.constants.aliases import ALIAS_VF, ALIAS_DNFR

# Initialize nodes with different structural frequencies and reorganization pressures
set_vf(G, node1, 1.0)  # Slow reorganization
set_vf(G, node2, 5.0)  # Fast reorganization
G.nodes[node1]['dnfr'] = 0.8  # High reorganization pressure
G.nodes[node2]['dnfr'] = 0.3

# Couple nodes - synchronizes phase, frequency, AND reduces ΔNFR
apply_glyph(G, node1, "UM")

# Check synchronization and stabilization
phase1 = G.nodes[node1]['theta']
phase2 = G.nodes[node2]['theta']
vf1 = get_attr(G.nodes[node1], ALIAS_VF, 0.0)
vf2 = get_attr(G.nodes[node2], ALIAS_VF, 0.0)
dnfr1 = get_attr(G.nodes[node1], ALIAS_DNFR, 0.0)

print(f"Phase difference: {abs(phase1 - phase2):.3f} rad")
print(f"Frequency difference: {abs(vf1 - vf2):.3f} Hz_str")
print(f"ΔNFR after coupling: {dnfr1:.3f} (reduced by mutual stabilization)")
```

**Configuration**:
- `UM_theta_push`: Phase synchronization strength (default: 0.25)
- `UM_vf_sync`: Frequency synchronization strength (default: 0.10)
- `UM_dnfr_reduction`: ΔNFR reduction factor (default: 0.15)
- `UM_BIDIRECTIONAL`: Enable bidirectional phase sync (default: True)
- `UM_SYNC_VF`: Enable frequency synchronization (default: True)
- `UM_STABILIZE_DNFR`: Enable ΔNFR stabilization (default: True)
- `UM_FUNCTIONAL_LINKS`: Create edges based on compatibility (default: True)

**Contracts**:
- Must verify phase compatibility before coupling
- Synchronizes both θ and νf when enabled
- Reduces ΔNFR proportionally to phase alignment
- Creates bidirectional connection
- Preserves existing couplings
- Respects nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

**Structural Invariants**:
- **⚠️ CRITICAL**: UM **NEVER** modifies EPI directly
- EPI identity is preserved during all coupling operations
- Only θ (phase), νf (frequency), and ΔNFR are modified by UM
- Any EPI change during a sequence with UM must come from:
  - Other operators (Emission, Reception, etc.)
  - Natural evolution via nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
  - Never from UM itself
- Theoretical basis: Coupling creates structural links through phase synchronization (φᵢ(t) ≈ φⱼ(t)), not through information transfer or EPI modification
- Implementation guarantee: `_op_UM` function does not touch EPI attributes

**Notes**:
According to TNFR canonical theory, coupling synchronizes not only phases but also
structural frequencies, and produces a stabilizing effect that reduces reorganization
pressure (ΔNFR) through mutual stabilization. This ensures that coupled nodes:
- Converge their reorganization rates (essential for sustained resonance)
- Experience reduced structural instability when well-aligned
- Maintain coherence through phase-dependent stabilization

This mechanism is observed in systems such as:
- Synchronized biological rhythms (heartbeats, neural oscillations)
- Coherent network evolution (social dynamics, collective behavior)
- Coupled oscillator systems (phase-locked loops, synchronized clocks)
- Multi-agent coordination (swarm intelligence, distributed systems)

---

### 6. Resonance (RA) 🌊

**Function**: Amplifies and propagates patterns

**Effect**:
- Increases effective coupling
- Preserves EPI identity during propagation
- Strengthens coherent patterns

**When to use**:
- Pattern reinforcement
- Spreading coherence through network
- Amplifying weak signals

**Mathematical representation**: EPIₙ → EPIₙ₊₁

**Example**:
```python
from tnfr.operators import Resonance

# Propagate pattern from source
Resonance()(G, source_node, radius=2)

# Check propagation
for neighbor in G.neighbors(source_node):
    coupling_strength = G[source_node][neighbor].get('coupling', 0)
    print(f"Coupling with {neighbor}: {coupling_strength:.3f}")
```

**Contracts**:
- Must preserve EPI identity
- Increases effective coupling
- Does not introduce noise

---

### 7. Silence (SHA) 🔇

**Function**: Temporarily freezes evolution

**Effect**:
- Sets νf ≈ 0
- EPI remains unchanged
- Pauses reorganization

**When to use**:
- Observation windows
- Synchronization pauses
- Stabilizing before measurement

**Mathematical representation**: νf ≈ 0 ⇒ ∂EPI/∂t ≈ 0

**Example**:
```python
from tnfr.operators import Silence

# Freeze node evolution
Silence()(G, node, duration=10)

# Verify frozen state
nf = G.nodes[node]['nf']
print(f"Structural frequency: {nf:.6f} Hz_str (≈0)")
```

**Contracts**:
- EPI must remain invariant
- νf → 0
- Can be reversed

---

### 8. Expansion (VAL) 📈

**Function**: Increases structural complexity

**Effect**:
- EPI dimensionality grows
- Adds degrees of freedom
- Elaborates structure

**When to use**:
- Adding capabilities
- Increasing expressiveness
- Growing network capacity

**Mathematical representation**: EPI → k·EPI, k ∈ ℕ⁺

**Example**:
```python
from tnfr.operators import Expansion

# Expand node structure
dim_before = len(G.nodes[node]['epi'])
Expansion()(G, node, factor=1.5)
dim_after = len(G.nodes[node]['epi'])

print(f"Dimensionality: {dim_before} → {dim_after}")
```

**Contracts**:
- Must increase EPI dimensionality
- Preserves existing structure
- Maintains coherence

---

### 9. Contraction (NUL) 📉

**Function**: Reduces structural complexity

**Effect**:
- EPI dimensionality decreases
- Removes degrees of freedom
- Simplifies structure

**When to use**:
- Simplification
- Focusing on essentials
- Reducing computational cost

**Mathematical representation**: ‖EPI′‖ ≥ τ (reduced support)

**Example**:
```python
from tnfr.operators import Contraction

# Contract node structure
Contraction()(G, node, factor=0.7)

# Verify reduced complexity
print(f"New dimensionality: {len(G.nodes[node]['epi'])}")
```

**Contracts**:
- Must decrease EPI dimensionality
- Preserves core structure
- Maintains minimum threshold τ

---

### 10. Self-organization (THOL) 🌱

**Function**: Spontaneous pattern formation

**Effect**:
- Creates sub-EPIs
- Preserves global form
- Enables emergent structure

**When to use**:
- Emergent structure formation
- Hierarchical organization
- Fractalization

**Mathematical representation**: ∂²EPI/∂t² > τ

**Example**:
```python
from tnfr.operators import SelfOrganization

# Enable self-organization
SelfOrganization()(G, threshold=0.5)

# Check for sub-structures
for node in G.nodes():
    if 'sub_epi' in G.nodes[node]:
        print(f"Node {node} has {len(G.nodes[node]['sub_epi'])} sub-EPIs")
```

**Contracts**:
- May create sub-EPIs
- Preserves global form
- Maintains operational fractality

---

### 11. Mutation (ZHIR) 🧬

**Function**: Phase transformation

**Effect**:
- θ → θ′ when structural threshold crossed
- Qualitative state change
- Phase transition

**When to use**:
- Qualitative state changes
- Phase transitions
- Transformative reorganization

**Mathematical representation**: θ → θ′ if ΔEPI/Δt > ξ

**Example**:
```python
from tnfr.operators import Mutation

# Trigger mutation if threshold exceeded
phase_before = G.nodes[node]['phase']
Mutation()(G, node, threshold=0.8)
phase_after = G.nodes[node]['phase']

if phase_after != phase_before:
    print(f"Phase mutation: {phase_before:.3f} → {phase_after:.3f} rad")
```

**Contracts**:
- Changes phase only if threshold exceeded
- Preserves EPI integrity
- Maintains network coherence

---

### 12. Transition (NAV) ➡️

**Function**: Movement between structural states

**Effect**:
- Controlled EPI evolution along path
- Guided trajectory
- Smooth state changes

**When to use**:
- Trajectory navigation
- Guided change
- State interpolation

**Mathematical representation**: Triggers creative thresholds (ΔNFR ≈ νf)

**Example**:
```python
from tnfr.operators import Transition

# Navigate between states
Transition()(G, node, target_state={'epi': target_epi}, steps=10)

# Monitor transition progress
current_epi = G.nodes[node]['epi']
distance = np.linalg.norm(current_epi - target_epi)
print(f"Distance to target: {distance:.3f}")
```

**Contracts**:
- Must follow valid path
- Preserves network integrity
- Maintains coherence during transition

---

### 13. Recursivity (REMESH) 🔄

**Function**: Nested operator application

**Effect**:
- Maintains operational fractality
- Applies operators at multiple scales
- Preserves hierarchical structure

**When to use**:
- Multi-scale operations
- Hierarchical coherence
- Nested transformations

**Mathematical representation**: EPI(t) = EPI(t − τ)

**Example**:
```python
from tnfr.operators import Recursivity

# Apply operators recursively
Recursivity()(G, [Coherence(), Resonance()], depth=3)

# Verify multi-scale coherence
for level in range(3):
    C_level = total_coherence(G, level=level)
    print(f"Coherence at level {level}: {C_level:.3f}")
```

**Contracts**:
- Maintains operational fractality
- Preserves structure at each level
- Does not exceed stack limits

---

## Operator Sequences

Operators are rarely used alone. They combine into **sequences** that create complex behaviors.

### Common Sequences

#### Bootstrap Sequence
```python
from tnfr import run_sequence
from tnfr.operators import Emission, Coupling, Coherence

# Start a new node
ops = [Emission(), Coupling(), Coherence()]
run_sequence(G, node, ops)
```

**Purpose**: Initialize and stabilize a new node

#### Stabilize Sequence
```python
ops = [Coherence(), Silence()]
run_sequence(G, node, ops)
```

**Purpose**: Freeze current state for observation

#### Explore Sequence
```python
ops = [Dissonance(), Mutation(), Coherence()]
run_sequence(G, node, ops)
```

**Purpose**: Try new configurations while maintaining coherence

#### Propagate Sequence
```python
ops = [Resonance(), Coupling()]
run_sequence(G, node, ops)
```

**Purpose**: Spread patterns through the network

### Sequence Design Principles

1. **Start with emission** for new patterns
2. **End with coherence** to stabilize
3. **Use dissonance carefully** - always follow with coherence
4. **Verify phase** before coupling
5. **Monitor metrics** (C(t), Si, ΔNFR) throughout

## Operator Grammar

Operators follow a compositional grammar:

```
Sequence := Operator+ Coherence
Operator := Emission | Reception | Coupling | Resonance | ...
Safe_Exploration := Dissonance Operator* Coherence
Multi_Scale := Recursivity(Sequence)
```

See [Glyph Sequences Guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md) for comprehensive patterns.

## Monitoring Operator Effects

### Before/After Metrics

```python
from tnfr.metrics import total_coherence, sense_index

# Before
C_before = total_coherence(G)
Si_before = sense_index(G)

# Apply operator
operator(G)

# After
C_after = total_coherence(G)
Si_after = sense_index(G)

print(f"C(t): {C_before:.3f} → {C_after:.3f}")
print(f"Si: {Si_before:.3f} → {Si_after:.3f}")
```

### Telemetry Tracing

```python
from tnfr.telemetry import enable_tracing

# Enable detailed logging
enable_tracing()

# Operators will now log all changes
operator(G)

# Review trace
from tnfr.telemetry import get_trace
for event in get_trace():
    print(f"{event['operator']}: {event['effect']}")
```

## Best Practices

### DO:
- ✅ Always end sequences with Coherence
- ✅ Monitor C(t) and Si after each operator
- ✅ Use Dissonance carefully and controllably
- ✅ Verify phase before Coupling
- ✅ Trace operator sequences for debugging

### DON'T:
- ❌ Apply operators without monitoring effects
- ❌ Use Dissonance without following Coherence
- ❌ Skip phase verification
- ❌ Create arbitrary mutations outside operators
- ❌ Ignore warning signs (low C(t), high |ΔNFR|)

## Troubleshooting

### Network coherence drops after operator
- **Likely cause**: Dissonance without stabilization
- **Solution**: Apply Coherence operator
- **Prevention**: Always end with Coherence

### Nodes won't couple
- **Likely cause**: Phase mismatch
- **Solution**: Check phase difference, use Reception first
- **Prevention**: Verify phase compatibility

### Bifurcation occurs unexpectedly
- **Likely cause**: Excessive ΔNFR from Dissonance
- **Solution**: Reduce dissonance intensity, apply Coherence
- **Prevention**: Monitor |ΔNFR| carefully

See [Troubleshooting Guide](TROUBLESHOOTING.md) for more.

## Advanced Topics

### Custom Operator Composition

```python
def custom_sequence(G, node):
    """Custom operator sequence for specific use case."""
    Emission()(G, node)
    for neighbor in G.neighbors(node):
        Coupling()(G, node, neighbor)
    Resonance()(G, node)
    Coherence()(G)
```

### Conditional Operators

```python
def adaptive_operator(G, node):
    """Apply operator based on current state."""
    C_t = total_coherence(G)
    
    if C_t < 0.3:
        # Low coherence - stabilize
        Coherence()(G)
    elif C_t > 0.8:
        # High coherence - explore
        Dissonance()(G, intensity=0.3)
        Coherence()(G)
    else:
        # Moderate coherence - propagate
        Resonance()(G, node)
```

## See Also

- **[API Reference](../api/operators.md)** - Complete operator API
- **[Glyph Sequences Guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)** - Canonical patterns
- **[Metrics Interpretation](METRICS_INTERPRETATION.md)** - Understanding effects
- **[Troubleshooting](TROUBLESHOOTING.md)** - Solving common issues
- **[Examples](../examples/README.md)** - Practical operator usage

---

**Next**: Learn how to interpret metrics in [Metrics Interpretation Guide](METRICS_INTERPRETATION.md) →
