# Metrics Interpretation Guide

[Home](../home.md) › Metrics Interpretation

This guide explains how to interpret and use TNFR's key metrics: Total Coherence (C(t)), Sense Index (Si), structural frequency (νf), phase (φ), and reorganization gradient (ΔNFR).

## Overview

TNFR provides precise, observable metrics to measure network health and evolution. Understanding these metrics is essential for:
- Monitoring network stability
- Detecting bifurcations and collapses
- Optimizing operator sequences
- Debugging structural issues

## Core Metrics

### Total Coherence: C(t)

**What it measures**: Global network stability at time t

**Analogy**: Like measuring the clarity of a choir's harmony. High C(t) = clear, stable patterns. Low C(t) = chaotic, fragmented noise.

**Range**: 0.0 (total chaos) to 1.0 (perfect coherence)

**Interpretation**:
- **C(t) > 0.7**: 🟢 Strong coherence, stable patterns
- **0.5 < C(t) < 0.7**: 🟡 Moderate coherence, generally stable
- **0.3 < C(t) < 0.5**: 🟠 Weak coherence, at risk
- **C(t) < 0.3**: 🔴 Critical - fragmentation likely

**How to measure**:
```python
from tnfr.metrics import total_coherence

C_t = total_coherence(G)
print(f"Total coherence: {C_t:.3f}")
```

**What influences C(t)**:
- Network topology (more coupling → higher C(t))
- Phase synchronization (aligned phases → higher C(t))
- Structural frequencies (moderate νf → higher C(t))
- Recent operators (Coherence ↑, Dissonance ↓)

**Typical evolution**:
```
Network creation:    C(t) ≈ 0.3-0.5 (initial chaos)
After Coherence():   C(t) ↑ 0.6-0.8 (stabilization)
After Dissonance():  C(t) ↓ 0.3-0.5 (exploration)
After Resonance():   C(t) ↑ 0.5-0.7 (propagation)
```

**When to act**:
- **C(t) dropping**: Apply Coherence operator
- **C(t) too high (>0.9)**: May indicate over-stabilization, consider Dissonance
- **C(t) oscillating**: Normal during exploration, monitor amplitude

---

### Sense Index: Si

**What it measures**: Capacity to generate stable reorganization patterns

**Analogy**: Like measuring a musician's skill. High Si = can improvise while maintaining harmony. Low Si = changes lead to chaos.

**Range**: 0.0 (unstable) to 1.0+ (highly stable)

**Interpretation**:
- **Si > 0.8**: 🟢 Excellent reorganization stability
- **0.6 < Si < 0.8**: 🟡 Good stability, safe to apply operators
- **0.4 < Si < 0.6**: 🟠 Moderate stability, careful with Dissonance
- **Si < 0.4**: 🔴 Warning - changes may cause bifurcation

**How to measure**:
```python
from tnfr.metrics import sense_index

# Network-wide
Si = sense_index(G)
print(f"Network sense index: {Si:.3f}")

# Per-node
for node in G.nodes():
    Si_node = sense_index(G, node=node)
    print(f"Node {node} Si: {Si_node:.3f}")
```

**What influences Si**:
- **ΔNFR**: Lower gradient → higher Si
- **νf**: Moderate frequency → higher Si
- **Phase dispersion**: Lower spread → higher Si
- **Network topology**: Better coupling → higher Si

**Formula (simplified)**:
```
Si ≈ 1 / (1 + |ΔNFR| + phase_variance)
```

**Typical values**:
```
Stable network:           Si ≈ 0.7-0.9
Exploring network:        Si ≈ 0.4-0.6
Pre-bifurcation:         Si < 0.3
After Self-organization: Si ↑ 0.6-0.8
```

**When to act**:
- **Si dropping rapidly**: Apply Coherence immediately
- **Si < 0.4**: Avoid Dissonance, stabilize first
- **Si very high (>0.95)**: May indicate stagnation, consider exploration

---

### Structural Frequency: νf

**What it measures**: Rate at which a node reorganizes its internal structure

**Units**: Hz_str (structural hertz) - NOT physical frequency!

**Analogy**: Like a heart rate, but for structural change. Higher νf = faster reorganization.

**Range**: 0.0+ Hz_str (no theoretical upper limit)

**Typical values**:
- **Active node**: νf ≈ 1.0-5.0 Hz_str
- **Stable node**: νf ≈ 0.1-1.0 Hz_str
- **Dormant node**: νf ≈ 0.01-0.1 Hz_str
- **Frozen node**: νf ≈ 0.0 Hz_str (via Silence operator)
- **Collapsing node**: νf → 0

**How to measure**:
```python
# Single node
nf = G.nodes[node]['nf']
print(f"Node {node} νf: {nf:.2f} Hz_str")

# All nodes
for node in G.nodes():
    nf = G.nodes[node]['nf']
    print(f"Node {node}: {nf:.2f} Hz_str")

# Average network frequency
avg_nf = sum(G.nodes[n]['nf'] for n in G.nodes()) / G.number_of_nodes()
print(f"Average νf: {avg_nf:.2f} Hz_str")
```

**What influences νf**:
- **Emission**: Increases νf
- **Silence**: Sets νf ≈ 0
- **Coupling**: May synchronize νf between nodes
- **Dissonance**: May increase νf variability

**Interpretation by range**:
```
νf > 5.0:    Very active, may be unstable
νf 1.0-5.0:  Normal active reorganization
νf 0.1-1.0:  Stable, measured evolution
νf < 0.1:    Low activity, approaching dormancy
νf ≈ 0:      Frozen or collapsing
```

**When to act**:
- **νf → 0**: Node collapsing, apply Emission
- **νf too high**: May indicate instability, apply Coherence
- **νf dispersion high**: Network desynchronized, use Coupling

---

### Phase: φ (theta)

**What it measures**: Relative timing/synchrony of a node with its neighbors

**Units**: Radians (0 to 2π)

**Analogy**: Like dancers in choreography - need to be in sync to create coherent performance.

**Range**: 0 to 2π radians (or -π to π)

**Interpretation**:
- **Phase aligned** (Δφ < 0.5 rad): Strong coupling possible
- **Phase misaligned** (Δφ > 1.5 rad): Weak/no coupling
- **Phase variance low**: Network synchronized
- **Phase variance high**: Network fragmented

**How to measure**:
```python
# Single node
phase = G.nodes[node]['phase']
print(f"Node {node} phase: {phase:.3f} rad")

# Phase difference between nodes
phase1 = G.nodes[node1]['phase']
phase2 = G.nodes[node2]['phase']
delta_phase = abs(phase1 - phase2)
print(f"Phase difference: {delta_phase:.3f} rad")

# Network phase coherence (Kuramoto order parameter)
from tnfr.metrics import phase_coherence
R = phase_coherence(G)
print(f"Phase coherence: {R:.3f}")  # 0.0-1.0
```

**Phase Coherence (Kuramoto)**:
- **R > 0.7**: Strong synchronization
- **0.3 < R < 0.7**: Partial synchronization
- **R < 0.3**: Weak synchronization

**What influences phase**:
- **Coupling**: Synchronizes phases
- **Reception**: Adjusts phase to match inputs
- **Mutation**: May shift phase
- **Network topology**: Better connected → more synchronized

**When to act**:
- **Large Δφ before Coupling**: Use Reception first to align
- **Phase variance increasing**: Apply Coupling or Coherence
- **Phase frozen**: Check if Silence was applied

---

### Reorganization Gradient: ΔNFR

**What it measures**: Internal pressure driving structural change

**Analogy**: Like water pressure that drives flow. ΔNFR measures "structural pressure."

**Range**: Unbounded (typically -10 to +10)

**Interpretation**:
- **ΔNFR > 0**: Expansion pressure (growth)
- **ΔNFR < 0**: Contraction pressure (simplification)
- **|ΔNFR| large**: High reorganization demand
- **ΔNFR ≈ 0**: Equilibrium, stable state

**Sign meaning**:
```
ΔNFR > +2:  Strong expansion, risk of instability
ΔNFR +0.5 to +2: Healthy growth
ΔNFR -0.5 to +0.5: Equilibrium
ΔNFR -2 to -0.5: Healthy contraction
ΔNFR < -2: Strong contraction, risk of collapse
```

**How to measure**:
```python
# Per-node ΔNFR
delta_nfr = G.nodes[node].get('delta_nfr', 0)
print(f"Node {node} ΔNFR: {delta_nfr:+.3f}")

# Identify high-pressure nodes
for node in G.nodes():
    dnfr = G.nodes[node].get('delta_nfr', 0)
    if abs(dnfr) > 2.0:
        print(f"⚠️ Node {node} high pressure: {dnfr:+.3f}")
```

**What influences ΔNFR**:
- **Dissonance**: Increases |ΔNFR|
- **Coherence**: Reduces |ΔNFR|
- **Network mismatch**: High coupling with phase mismatch → high |ΔNFR|
- **Topology changes**: Adding/removing connections affects ΔNFR

**Relationship to nodal equation**:
```
∂EPI/∂t = νf · ΔNFR(t)

If νf = 2.0 Hz_str and ΔNFR = +1.5:
  → ∂EPI/∂t = 3.0 (rapid expansion)

If νf = 0.5 Hz_str and ΔNFR = +1.5:
  → ∂EPI/∂t = 0.75 (slow expansion)
```

**When to act**:
- **|ΔNFR| > 3**: Apply Coherence to stabilize
- **ΔNFR ≈ 0 everywhere**: May indicate stagnation
- **ΔNFR oscillating**: Normal during exploration

---

## Metric Relationships

### How metrics interact:

**C(t) and Si**:
- Usually correlated: High C(t) → High Si
- Exception: Over-stabilization (high C(t), low Si)

**νf and ΔNFR**:
- Combined in nodal equation: ∂EPI/∂t = νf · ΔNFR
- High νf + high ΔNFR = rapid change
- Low νf + high ΔNFR = constrained change

**Phase and C(t)**:
- Aligned phases → Higher C(t)
- Phase coherence ≈ C(t) (approximate)

**Si and ΔNFR**:
- Inversely related: High |ΔNFR| → Low Si
- Si measures stability under reorganization pressure

### Healthy Network Profile:

```
C(t): 0.5-0.7      (moderate coherence)
Si: 0.6-0.8        (good stability)
νf: 0.5-2.0 Hz_str (active but stable)
Phase coherence: >0.5 (synchronized)
|ΔNFR|: <2.0       (manageable pressure)
```

## Monitoring Workflows

### Basic Monitoring

```python
from tnfr.metrics import total_coherence, sense_index

def monitor_network(G):
    """Basic network health check."""
    C_t = total_coherence(G)
    Si = sense_index(G)
    
    print(f"C(t): {C_t:.3f}")
    print(f"Si: {Si:.3f}")
    
    if C_t < 0.3:
        print("⚠️ Low coherence - apply Coherence operator")
    if Si < 0.4:
        print("⚠️ Low sense index - avoid Dissonance")
    if C_t > 0.5 and Si > 0.6:
        print("✓ Network healthy")
```

### Detailed Monitoring

```python
def detailed_monitor(G):
    """Comprehensive network analysis."""
    from tnfr.metrics import total_coherence, sense_index, phase_coherence
    
    # Global metrics
    C_t = total_coherence(G)
    Si = sense_index(G)
    R = phase_coherence(G)
    
    # Per-node statistics
    nf_values = [G.nodes[n]['nf'] for n in G.nodes()]
    dnfr_values = [G.nodes[n].get('delta_nfr', 0) for n in G.nodes()]
    
    avg_nf = sum(nf_values) / len(nf_values)
    max_dnfr = max(abs(d) for d in dnfr_values)
    
    print(f"╔══ Network Health ══╗")
    print(f"║ C(t): {C_t:.3f}         ║")
    print(f"║ Si:   {Si:.3f}         ║")
    print(f"║ R:    {R:.3f}         ║")
    print(f"╟────────────────────╢")
    print(f"║ Avg νf: {avg_nf:.2f} Hz   ║")
    print(f"║ Max |ΔNFR|: {max_dnfr:.2f}  ║")
    print(f"╚════════════════════╝")
    
    # Warnings
    issues = []
    if C_t < 0.3: issues.append("Low coherence")
    if Si < 0.4: issues.append("Low sense index")
    if R < 0.3: issues.append("Poor phase sync")
    if max_dnfr > 3.0: issues.append("High reorganization pressure")
    
    if issues:
        print(f"⚠️ Issues: {', '.join(issues)}")
    else:
        print("✓ All metrics healthy")
```

### Time-Series Monitoring

```python
def monitor_evolution(G, operators, steps=100):
    """Monitor metrics over time during operator sequence."""
    import matplotlib.pyplot as plt
    
    C_history = []
    Si_history = []
    
    for step in range(steps):
        # Apply operators
        for op in operators:
            op(G)
        
        # Record metrics
        C_history.append(total_coherence(G))
        Si_history.append(sense_index(G))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    ax1.plot(C_history)
    ax1.set_ylabel('C(t)')
    ax1.set_title('Total Coherence Evolution')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    ax1.legend()
    
    ax2.plot(Si_history)
    ax2.set_ylabel('Si')
    ax2.set_xlabel('Step')
    ax2.set_title('Sense Index Evolution')
    ax2.axhline(y=0.4, color='r', linestyle='--', label='Warning')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## Diagnostic Patterns

### Pattern 1: Declining Coherence

**Symptoms**: C(t) dropping over time

**Likely causes**:
- Excessive Dissonance
- Network desynchronization
- Node collapses (νf → 0)

**Solution**:
```python
# Stabilize network
from tnfr.operators import Coherence, Coupling

Coherence()(G)  # Global stabilization
for node in low_coherence_nodes:
    for neighbor in G.neighbors(node):
        Coupling()(G, node, neighbor)  # Restore connections
```

### Pattern 2: Low Sense Index

**Symptoms**: Si < 0.4, especially if C(t) > 0.5

**Likely causes**:
- High reorganization pressure (|ΔNFR| large)
- Phase dispersion
- Frequency variability

**Solution**:
```python
# Reduce pressure and synchronize
from tnfr.operators import Coherence, Reception

Coherence()(G)  # Reduce ΔNFR
for node in high_pressure_nodes:
    Reception()(G, node)  # Align phases
```

### Pattern 3: Oscillating Metrics

**Symptoms**: C(t) and Si oscillate regularly

**Interpretation**: Normal during exploration with Dissonance

**When to worry**: If oscillation amplitude increases

**Solution**: If concerning, reduce Dissonance intensity

### Pattern 4: Flatlined Metrics

**Symptoms**: C(t), Si unchanging despite operators

**Likely causes**:
- Over-stabilization
- All nodes in Silence
- Network disconnected

**Solution**:
```python
# Reactivate network
from tnfr.operators import Emission, Dissonance

# Wake up dormant nodes
for node in dormant_nodes:
    Emission()(G, node)

# Introduce exploration
Dissonance()(G, intensity=0.3)
```

## Best Practices

### DO:
- ✅ Monitor metrics before and after each operator
- ✅ Track time-series for trends
- ✅ Set thresholds based on your domain
- ✅ Log metrics for reproducibility
- ✅ Use multiple metrics together (not just one)

### DON'T:
- ❌ Rely on single metric alone
- ❌ Ignore warning signs (low C(t), low Si)
- ❌ Compare metrics across different networks without context
- ❌ Expect perfect coherence (C(t)=1.0) - often unrealistic
- ❌ Forget to account for domain-specific thresholds

## Domain-Specific Interpretations

Different domains may have different healthy ranges:

### Biological Systems
```
C(t): 0.4-0.6  (natural variability)
Si: 0.5-0.7    (adaptive capacity)
νf: 0.5-2.0    (moderate dynamics)
```

### Social Networks
```
C(t): 0.3-0.5  (diversity maintained)
Si: 0.4-0.6    (change tolerance)
νf: 1.0-3.0    (active interaction)
```

### Technical Systems
```
C(t): 0.6-0.8  (higher stability needed)
Si: 0.7-0.9    (predictable behavior)
νf: 0.1-1.0    (controlled change)
```

## See Also

- **[Operators Guide](OPERATORS_GUIDE.md)** - How operators affect metrics
- **[Troubleshooting](TROUBLESHOOTING.md)** - Fixing metric issues
- **[API Reference](../api/telemetry.md)** - Metrics API details
- **[Theory](../foundations.md)** - Mathematical foundations

---

**Next**: Learn how to solve common problems in [Troubleshooting Guide](TROUBLESHOOTING.md) →
