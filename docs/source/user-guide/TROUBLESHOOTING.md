# Troubleshooting Guide

[Home](../index.md) › Troubleshooting

This guide helps you diagnose and solve common problems when working with TNFR networks.

## Quick Diagnostic Checklist

Use this checklist to quickly identify issues:

```
□ Check C(t): Is it < 0.3? → Low coherence problem
□ Check Si: Is it < 0.4? → Stability problem  
□ Check νf: Any nodes with νf ≈ 0? → Node collapse
□ Check phase: High variance? → Synchronization problem
□ Check |ΔNFR|: Any > 3.0? → High pressure problem
□ Check connectivity: Disconnected components? → Topology problem
```

## Common Problems

### Problem 1: Low Network Coherence (C(t) < 0.3)

**Symptoms**:
- C(t) below 0.3
- Network feels "fragmented"
- Operators have little effect

**Possible Causes**:
1. Insufficient coupling between nodes
2. Phase desynchronization
3. Recent Dissonance without stabilization
4. Nodes collapsing (νf → 0)

**Diagnosis**:
```python
from tnfr.metrics import total_coherence, phase_coherence

C_t = total_coherence(G)
R = phase_coherence(G)

print(f"C(t): {C_t:.3f}")
print(f"Phase coherence: {R:.3f}")
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# Check for disconnected components
import networkx as nx
components = list(nx.connected_components(G.to_undirected()))
print(f"Connected components: {len(components)}")

# Check for collapsed nodes
collapsed = [n for n in G.nodes() if G.nodes[n]['nf'] < 0.01]
print(f"Collapsed nodes: {len(collapsed)}")
```

**Solutions**:

**Solution 1**: Apply global Coherence
```python
from tnfr.operators import Coherence

Coherence()(G)
print(f"After Coherence: C(t) = {total_coherence(G):.3f}")
```

**Solution 2**: Increase coupling
```python
from tnfr.operators import Coupling

# Couple low-coherence nodes
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 3:  # Insufficient connections
        # Find nearest nodes by phase
        candidates = [n for n in G.nodes() if n != node and n not in neighbors]
        if candidates:
            nearest = min(candidates, key=lambda n: abs(G.nodes[n]['phase'] - G.nodes[node]['phase']))
            Coupling()(G, node, nearest)
```

**Solution 3**: Revive collapsed nodes
```python
from tnfr.operators import Emission

for node in collapsed:
    Emission()(G, node)
```

---

### Problem 2: Low Sense Index (Si < 0.4)

**Symptoms**:
- Si below 0.4
- Network unstable under changes
- Bifurcations occur unexpectedly

**Possible Causes**:
1. High reorganization pressure (|ΔNFR| large)
2. High phase dispersion
3. Frequency variability too high

**Diagnosis**:
```python
from tnfr.metrics import sense_index

Si = sense_index(G)
print(f"Si: {Si:.3f}")

# Check ΔNFR distribution
dnfr_values = [G.nodes[n].get('delta_nfr', 0) for n in G.nodes()]
max_dnfr = max(abs(d) for d in dnfr_values)
avg_dnfr = sum(abs(d) for d in dnfr_values) / len(dnfr_values)

print(f"Max |ΔNFR|: {max_dnfr:.3f}")
print(f"Avg |ΔNFR|: {avg_dnfr:.3f}")

# Check phase variance
import numpy as np
phases = [G.nodes[n]['phase'] for n in G.nodes()]
phase_var = np.var(phases)
print(f"Phase variance: {phase_var:.3f}")
```

**Solutions**:

**Solution 1**: Reduce reorganization pressure
```python
from tnfr.operators import Coherence

# Apply coherence to reduce ΔNFR
Coherence()(G)
```

**Solution 2**: Synchronize phases
```python
from tnfr.operators import Reception

# Align phases through reception
for node in G.nodes():
    Reception()(G, node)
```

**Solution 3**: Stabilize frequencies
```python
# Reduce frequency dispersion
nf_values = [G.nodes[n]['nf'] for n in G.nodes()]
target_nf = sum(nf_values) / len(nf_values)

for node in G.nodes():
    current_nf = G.nodes[node]['nf']
    if abs(current_nf - target_nf) > 1.0:
        # Gradually adjust toward average
        G.nodes[node]['nf'] = 0.7 * current_nf + 0.3 * target_nf
```

---

### Problem 3: Node Collapse (νf → 0)

**Symptoms**:
- Individual nodes with νf ≈ 0
- Node stops responding to operators
- Gradual loss of network functionality

**Possible Causes**:
1. Insufficient energy/activation
2. Isolated from network
3. Excessive Silence operator
4. Natural decay without maintenance

**Diagnosis**:
```python
# Identify collapsing nodes
threshold = 0.1
collapsing = []

for node in G.nodes():
    nf = G.nodes[node]['nf']
    if nf < threshold:
        neighbors = list(G.neighbors(node))
        collapsing.append((node, nf, len(neighbors)))

print(f"Collapsing nodes: {len(collapsing)}")
for node, nf, degree in collapsing:
    print(f"  Node {node}: νf={nf:.4f}, degree={degree}")
```

**Solutions**:

**Solution 1**: Reactivate with Emission
```python
from tnfr.operators import Emission

for node, _, _ in collapsing:
    Emission()(G, node)
    print(f"Revived node {node}: νf={G.nodes[node]['nf']:.3f}")
```

**Solution 2**: Increase connectivity
```python
from tnfr.operators import Coupling

for node, _, degree in collapsing:
    if degree < 2:  # Isolated or barely connected
        # Connect to active neighbors
        active_nodes = [n for n in G.nodes() if G.nodes[n]['nf'] > 1.0]
        if active_nodes:
            target = active_nodes[0]
            Coupling()(G, node, target)
```

**Solution 3**: Remove if truly dead
```python
# Only if revival fails and node is truly non-functional
truly_dead = [n for n in G.nodes() if G.nodes[n]['nf'] < 0.001]
if truly_dead:
    print(f"Removing {len(truly_dead)} dead nodes")
    G.remove_nodes_from(truly_dead)
```

---

### Problem 4: Nodes Won't Couple

**Symptoms**:
- Coupling operator has no effect
- Connections don't form
- Phase mismatch warnings

**Possible Causes**:
1. Phase mismatch too large
2. Nodes in Silence (νf ≈ 0)
3. Invalid coupling conditions

**Diagnosis**:
```python
# Check phase compatibility
def check_coupling_compatibility(G, node1, node2):
    phase1 = G.nodes[node1]['phase']
    phase2 = G.nodes[node2]['phase']
    nf1 = G.nodes[node1]['nf']
    nf2 = G.nodes[node2]['nf']
    
    phase_diff = abs(phase1 - phase2)
    
    print(f"Node {node1}: φ={phase1:.3f}, νf={nf1:.3f}")
    print(f"Node {node2}: φ={phase2:.3f}, νf={nf2:.3f}")
    print(f"Phase difference: {phase_diff:.3f} rad")
    
    if phase_diff > 1.5:
        print("⚠️ Phase mismatch too large")
    if nf1 < 0.1 or nf2 < 0.1:
        print("⚠️ One or both nodes inactive")

check_coupling_compatibility(G, node1, node2)
```

**Solutions**:

**Solution 1**: Align phases first
```python
from tnfr.operators import Reception

# Use Reception to align phases
Reception()(G, node1)
Reception()(G, node2)

# Then couple
from tnfr.operators import Coupling
Coupling()(G, node1, node2)
```

**Solution 2**: Activate silent nodes
```python
from tnfr.operators import Emission

# Reactivate if in Silence
if G.nodes[node1]['nf'] < 0.1:
    Emission()(G, node1)
if G.nodes[node2]['nf'] < 0.1:
    Emission()(G, node2)

# Then couple
Coupling()(G, node1, node2)
```

**Solution 3**: Use Resonance for indirect coupling
```python
from tnfr.operators import Resonance

# Propagate pattern from one to the other
Resonance()(G, node1, radius=2)
# This may create indirect pathway
```

---

### Problem 5: Unexpected Bifurcation

**Symptoms**:
- Network suddenly fragments
- New patterns emerge unexpectedly
- ∂²EPI/∂t² > threshold

**Possible Causes**:
1. Excessive Dissonance
2. |ΔNFR| too large
3. Phase transition threshold crossed
4. Legitimate self-organization

**Diagnosis**:
```python
# Check if bifurcation was triggered
def check_bifurcation_risk(G):
    risks = []
    
    for node in G.nodes():
        dnfr = G.nodes[node].get('delta_nfr', 0)
        nf = G.nodes[node]['nf']
        
        # High reorganization rate
        reorg_rate = nf * abs(dnfr)
        
        if reorg_rate > 5.0:
            risks.append((node, reorg_rate, dnfr, nf))
    
    if risks:
        print(f"⚠️ {len(risks)} nodes at bifurcation risk:")
        for node, rate, dnfr, nf in risks:
            print(f"  Node {node}: rate={rate:.2f}, ΔNFR={dnfr:+.2f}, νf={nf:.2f}")
    else:
        print("✓ No bifurcation risk detected")
    
    return risks

check_bifurcation_risk(G)
```

**Solutions**:

**Solution 1**: If unwanted, stabilize immediately
```python
from tnfr.operators import Coherence

# Emergency stabilization
Coherence()(G)
Coherence()(G)  # Apply twice for strong effect
```

**Solution 2**: Reduce Dissonance intensity
```python
# If using Dissonance, reduce intensity
from tnfr.operators import Dissonance

# Instead of:
# Dissonance()(G, intensity=1.0)

# Use:
Dissonance()(G, intensity=0.3)
Coherence()(G)  # Always follow with Coherence
```

**Solution 3**: If legitimate, accept and observe
```python
# If bifurcation is desired (exploration, self-organization)
from tnfr.operators import SelfOrganization

# Allow system to self-organize
SelfOrganization()(G)

# Monitor new structure
print(f"New C(t): {total_coherence(G):.3f}")
print(f"New Si: {sense_index(G):.3f}")
```

---

### Problem 6: Slow Performance

**Symptoms**:
- Operations take long time
- Large networks sluggish
- Memory usage high

**Possible Causes**:
1. Using NumPy backend on large networks
2. Dense connectivity (O(N²) edges)
3. Inefficient operator sequences
4. No caching enabled

**Diagnosis**:
```python
import time
import tnfr

# Check network size
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
print(f"Density: {density:.3f}")

# Check backend
print(f"Backend: {tnfr.get_backend()}")

# Time an operation
start = time.time()
from tnfr.operators import Coherence
Coherence()(G)
elapsed = time.time() - start
print(f"Coherence took: {elapsed:.3f}s")
```

**Solutions**:

**Solution 1**: Use JAX backend for GPU acceleration
```bash
# Install JAX
pip install tnfr[compute-jax]
```

```python
import tnfr

# Switch to JAX backend
tnfr.set_backend('jax')
print(f"Using backend: {tnfr.get_backend()}")
```

**Solution 2**: Reduce network density
```python
# Remove weak couplings
edges_to_remove = []
for u, v in G.edges():
    strength = G[u][v].get('coupling', 1.0)
    if strength < 0.1:  # Weak coupling threshold
        edges_to_remove.append((u, v))

print(f"Removing {len(edges_to_remove)} weak edges")
G.remove_edges_from(edges_to_remove)
```

**Solution 3**: Enable caching
```bash
# Install caching support
pip install tnfr[orjson]
```

**Solution 4**: Use sparse networks
```python
# When creating networks, keep connectivity low
G = tnfr.create_network(nodes=1000, connectivity=0.05)  # 5% connectivity
```

---

### Problem 7: Operators Have No Effect

**Symptoms**:
- Applying operators doesn't change metrics
- Network appears "frozen"
- C(t), Si remain constant

**Possible Causes**:
1. All nodes in Silence
2. Over-stabilized network (C(t) ≈ 1.0)
3. Disconnected network
4. Bug in operator application

**Diagnosis**:
```python
# Check if nodes are active
active_count = sum(1 for n in G.nodes() if G.nodes[n]['nf'] > 0.1)
print(f"Active nodes: {active_count}/{G.number_of_nodes()}")

# Check coherence
C_t = total_coherence(G)
print(f"C(t): {C_t:.3f}")

# Check connectivity
import networkx as nx
is_connected = nx.is_connected(G.to_undirected())
print(f"Connected: {is_connected}")

# Try operator and measure
C_before = total_coherence(G)
from tnfr.operators import Emission
Emission()(G, list(G.nodes())[0])
C_after = total_coherence(G)
print(f"C(t) change: {C_before:.3f} → {C_after:.3f}")
```

**Solutions**:

**Solution 1**: Wake up frozen network
```python
from tnfr.operators import Emission, Dissonance

# Reactivate all nodes
for node in G.nodes():
    if G.nodes[node]['nf'] < 0.1:
        Emission()(G, node)

# Introduce some exploration
Dissonance()(G, intensity=0.3)
```

**Solution 2**: If over-stabilized, introduce Dissonance
```python
from tnfr.operators import Dissonance

# Shake up over-stable network
if total_coherence(G) > 0.9:
    Dissonance()(G, intensity=0.5)
```

**Solution 3**: Reconnect network
```python
from tnfr.operators import Coupling
import networkx as nx

# Find disconnected components
components = list(nx.connected_components(G.to_undirected()))
if len(components) > 1:
    print(f"Found {len(components)} disconnected components")
    
    # Connect components
    for i in range(len(components) - 1):
        node1 = list(components[i])[0]
        node2 = list(components[i+1])[0]
        Coupling()(G, node1, node2)
```

---

## Debugging Techniques

### Enable Telemetry

```python
from tnfr.telemetry import enable_tracing, get_trace

# Enable detailed logging
enable_tracing()

# Apply operators
from tnfr.operators import Coherence
Coherence()(G)

# Review what happened
for event in get_trace():
    print(f"{event['time']}: {event['operator']} on {event['node']}")
    print(f"  Effect: {event['effect']}")
```

### Validate Network

```python
from tnfr.validation import validate_network

# Check for structural problems
issues = validate_network(G)

if issues:
    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"  {issue['type']}: {issue['description']}")
else:
    print("✓ Network structure valid")
```

### Step-by-Step Execution

```python
from tnfr.metrics import total_coherence, sense_index

def debug_operator_sequence(G, operators):
    """Execute operators one at a time with diagnostics."""
    for i, op in enumerate(operators):
        print(f"\n=== Step {i+1}: {op.__class__.__name__} ===")
        
        # Before
        C_before = total_coherence(G)
        Si_before = sense_index(G)
        
        # Apply
        op(G)
        
        # After
        C_after = total_coherence(G)
        Si_after = sense_index(G)
        
        # Report
        print(f"C(t): {C_before:.3f} → {C_after:.3f} ({C_after-C_before:+.3f})")
        print(f"Si:   {Si_before:.3f} → {Si_after:.3f} ({Si_after-Si_before:+.3f})")
```

### Snapshot Comparison

```python
import copy

# Take snapshot before operation
G_before = copy.deepcopy(G)

# Apply operators
from tnfr.operators import Dissonance, Coherence
Dissonance()(G, intensity=0.5)
Coherence()(G)

# Compare snapshots
print("Node-by-node changes:")
for node in G.nodes():
    nf_before = G_before.nodes[node]['nf']
    nf_after = G.nodes[node]['nf']
    if abs(nf_after - nf_before) > 0.1:
        print(f"  Node {node}: νf {nf_before:.3f} → {nf_after:.3f}")
```

## Getting More Help

If these solutions don't resolve your issue:

1. **Check the FAQ**: [FAQ](../getting-started/FAQ.md)
2. **Review Examples**: [Examples](../examples/README.md)
3. **API Documentation**: [API Reference](../api/overview.md)
4. **Open an Issue**: [GitHub Issues](https://github.com/fermga/TNFR-Python-Engine/issues)

When reporting issues, include:
- TNFR version: `python -c "import tnfr; print(tnfr.__version__)"`
- Network size and topology
- Operator sequence applied
- Metrics before/after
- Full error message (if any)

## See Also

- **[Operators Guide](OPERATORS_GUIDE.md)** - Understanding operator effects
- **[Metrics Interpretation](METRICS_INTERPRETATION.md)** - Reading network health
- **[FAQ](../getting-started/FAQ.md)** - Common questions
- **[API Reference](../api/overview.md)** - Complete API documentation

---

**Still stuck?** Open an issue on [GitHub](https://github.com/fermga/TNFR-Python-Engine/issues) with details about your problem.
