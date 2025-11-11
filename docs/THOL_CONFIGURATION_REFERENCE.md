# THOL Configuration Reference

## Overview

This document provides a **centralized reference** for all configurable parameters of the THOL (Self-organization) operator in TNFR. THOL is the canonical operator for autopoietic pattern formation—the operator that enables networks to reorganize from within through structural resonance.

**TNFR Canonical Principle** (TNFR.pdf §2.2.10):

> "THOL es el glifo de la autoorganización activa. No necesita intervención externa, ni programación, ni control — su función es reorganizar la forma desde dentro, en respuesta a la coherencia vibracional del campo."

This reference consolidates parameters currently dispersed across multiple modules:
- `src/tnfr/operators/preconditions/__init__.py` — Precondition validators
- `src/tnfr/operators/metabolism.py` — Vibrational metabolism
- `src/tnfr/operators/cascade.py` — Cascade detection
- `src/tnfr/operators/__init__.py` — Glyph implementation

---

## Canonical Parameters

### Bifurcation Dynamics

Parameters controlling when and how bifurcation (sub-EPI creation) occurs.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `BIFURCATION_THRESHOLD_TAU` | 0.5 (OZ), 0.1 (THOL) | [0.1, 2.0] | — | Acceleration threshold for bifurcation: ∂²EPI/∂t² > τ triggers sub-EPI creation. Default varies by context: 0.5 for OZ (Dissonance) bifurcation detection, 0.1 for THOL bifurcation |
| `THOL_BIFURCATION_THRESHOLD` | 0.1 | [0.1, 2.0] | — | THOL-specific alias for `BIFURCATION_THRESHOLD_TAU`. Used when canonical parameter not set |
| `THOL_MIN_EPI` | 0.2 | [0.05, 0.5] | — | Minimum EPI magnitude required for structural bifurcation |
| `THOL_MIN_VF` | 0.1 | [0.01, 1.0] | Hz_str | Minimum structural frequency for reorganization capacity |
| `THOL_ACCEL` | 0.10 | [0.01, 0.5] | — | Acceleration factor in glyph: ΔNFR += `THOL_accel` × d²EPI/dt² |
| `THOL_MIN_COLLECTIVE_COHERENCE` | 0.3 | [0.0, 1.0] | — | Minimum collective coherence for sub-EPI ensemble. When multiple sub-EPIs exist and coherence < threshold, warning is logged |

**Physical Basis:**

From the nodal equation ∂EPI/∂t = νf · ΔNFR(t), bifurcation occurs when **structural acceleration** ∂²EPI/∂t² exceeds threshold τ. This indicates a regime shift where the current structural form cannot accommodate the reorganization pressure, triggering emergent sub-structures.

- **τ (tau)**: Critical acceleration threshold. Higher values make bifurcation harder (more stable structures). **Note:** TNFR uses context-specific defaults: 0.5 for OZ (Dissonance) operator bifurcation detection, 0.1 for THOL (Self-organization) bifurcation. The lower THOL threshold reflects self-organization's role as the primary bifurcation operator.
- **EPI_min**: Coherence floor. Nodes below this lack sufficient form to bifurcate coherently.
- **νf_min**: Reorganization capacity floor. Nodes below this are "frozen" and cannot respond.
- **THOL_accel**: Controls how strongly d²EPI/dt² influences ΔNFR in glyph sequences.
- **THOL_MIN_COLLECTIVE_COHERENCE**: Monitors ensemble coherence of sub-EPIs. According to TNFR.pdf §2.2.10, sub-EPIs must form a **coherent ensemble** rather than fragmenting chaotically. Collective coherence is computed as `C = 1/(1 + var(sub_epi_magnitudes))`. Interpretation:
  - **> 0.7**: High coherence (structurally solid bifurcation)
  - **0.3-0.7**: Moderate (acceptable, monitor)
  - **< 0.3**: Low (possible fragmentation, warning logged)
  
  When multiple sub-EPIs exist and coherence falls below threshold, a warning is logged and the event is recorded in `G.graph["thol_coherence_warnings"]` for analysis. This validation is **non-blocking** (warnings only) to allow research into low-coherence dynamics.

**Configuration Example:**
```python
import networkx as nx

# Conservative bifurcation (harder to trigger)
G = nx.Graph()
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.8  # High threshold
G.graph["THOL_MIN_EPI"] = 0.3              # Require strong coherence
G.graph["THOL_MIN_VF"] = 0.2               # Require high capacity
G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.5  # Require coherent ensemble

# Sensitive bifurcation (easier to trigger)
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.2  # Low threshold
G.graph["THOL_MIN_EPI"] = 0.1              # Lower coherence floor
G.graph["THOL_MIN_VF"] = 0.05              # Lower capacity floor
G.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = 0.2  # More tolerant of fragmentation

# Monitor collective coherence
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer
analyzer = SequenceHealthAnalyzer()
coherence_stats = analyzer.analyze_thol_coherence(G)
if coherence_stats:
    print(f"Mean coherence: {coherence_stats['mean_coherence']:.3f}")
    print(f"Nodes below threshold: {coherence_stats['nodes_below_threshold']}")
```

---

### Metabolic Parameters

Parameters controlling vibrational metabolism—how THOL captures and transforms network context into internal structure.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `THOL_METABOLIC_ENABLED` | True | bool | — | Enable vibrational metabolism (network-driven bifurcation) |
| `THOL_METABOLIC_GRADIENT_WEIGHT` | 0.15 | [0.0, 0.5] | — | Weight for EPI gradient contribution to sub-EPI magnitude |
| `THOL_METABOLIC_COMPLEXITY_WEIGHT` | 0.10 | [0.0, 0.5] | — | Weight for phase variance (field complexity) contribution |

**Physical Basis:**

THOL metabolism implements the canonical principle: *"reorganizes external experience into internal structure without external instruction"* (TNFR Manual, p. 112).

**Metabolic Formula:**
```
sub-EPI = base_internal + network_contribution + complexity_bonus

where:
  base_internal       = parent_EPI × scaling_factor (0.25)
  network_contribution = epi_gradient × GRADIENT_WEIGHT
  complexity_bonus     = phase_variance × COMPLEXITY_WEIGHT
```

- **epi_gradient**: Difference between mean neighbor EPI and node EPI (structural pressure from environment)
- **phase_variance**: Variance of neighbor phases (complexity/dissonance of vibrational field)

**Metabolic Modes:**

1. **Isolated metabolism** (`METABOLIC_ENABLED=False` or degree=0):
   - Sub-EPI = base_internal only (pure bifurcation)
   - No network influence

2. **Network metabolism** (`METABOLIC_ENABLED=True`, degree≥1):
   - Sub-EPI includes network signals
   - External pressure shapes internal structure

**Configuration Example:**
```python
# Pure internal bifurcation (no network influence)
G.graph["THOL_METABOLIC_ENABLED"] = False

# Network-driven bifurcation (high external influence)
G.graph["THOL_METABOLIC_ENABLED"] = True
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.25  # Strong gradient influence
G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.15  # Strong complexity influence

# Balanced metabolism (default)
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.15
G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.10
```

---

### Propagation Parameters

Parameters controlling how sub-EPIs propagate to coupled neighbors (cascade dynamics).

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `THOL_MIN_COUPLING_FOR_PROPAGATION` | 0.5 | [0.3, 0.9] | — | Minimum coupling strength (phase alignment) required for propagation |
| `THOL_PROPAGATION_ATTENUATION` | 0.7 | [0.5, 0.95] | — | Attenuation factor: propagated_EPI = sub_EPI × attenuation × coupling |

**Physical Basis:**

Sub-EPIs propagate through **resonant coupling** to phase-aligned neighbors. This implements canonical cascade dynamics where bifurcation triggers network-wide self-organization.

**Propagation Mechanism:**
```python
# For each neighbor with sufficient coupling:
phase_diff = abs(θ_neighbor - θ_parent)
coupling_strength = 1.0 - (phase_diff / π)

if coupling_strength >= THOL_MIN_COUPLING_FOR_PROPAGATION:
    propagated_epi = sub_epi × ATTENUATION × coupling_strength
    neighbor_epi += propagated_epi
```

- **Coupling threshold**: Filters out anti-phase neighbors (destructive interference)
- **Attenuation**: Prevents unbounded growth while enabling cascades
- **Phase dependence**: Strong coupling → more propagation (constructive interference)

**Configuration Example:**
```python
# Conservative propagation (local cascades)
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.7  # Require strong alignment
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.5       # High attenuation

# Aggressive propagation (network-wide cascades)
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.3  # Allow weak coupling
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.9       # Low attenuation

# Blocked propagation (isolated bifurcation)
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 1.0  # Perfect alignment required (impossible)
```

---

### Network Parameters

Parameters defining network topology requirements for THOL.

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `THOL_MIN_DEGREE` | 1 | [0, 5] | — | Minimum node connectivity for metabolic context |
| `THOL_ALLOW_ISOLATED` | False | bool | — | Allow THOL on isolated nodes (internal-only bifurcation) |
| `THOL_CASCADE_MIN_NODES` | 3 | [2, 10] | — | Minimum nodes required to classify as cascade (vs. isolated bifurcation) |

**Physical Basis:**

THOL operates at the intersection of **internal acceleration** and **network coupling**. While internal bifurcation is always possible (∂²EPI/∂t² > τ), network connectivity enables:
1. Metabolic context (capturing external patterns)
2. Propagation (spreading bifurcation results)
3. Cascades (collective reorganization)

**Network Modes:**

1. **Isolated THOL** (`degree=0`, `ALLOW_ISOLATED=True`):
   - Pure internal bifurcation
   - No metabolism, no propagation
   - Use case: Testing, single-node systems

2. **Coupled THOL** (`degree≥MIN_DEGREE`):
   - Network metabolism enabled
   - Propagation possible
   - Cascades may emerge

**Configuration Example:**
```python
# Require network context (default)
G.graph["THOL_MIN_DEGREE"] = 1
G.graph["THOL_ALLOW_ISOLATED"] = False

# Allow isolated bifurcation (testing/single-node)
G.graph["THOL_ALLOW_ISOLATED"] = True

# Require rich connectivity for THOL
G.graph["THOL_MIN_DEGREE"] = 3  # Need ≥3 neighbors

# Sensitive cascade detection
G.graph["THOL_CASCADE_MIN_NODES"] = 2  # Classify 2+ nodes as cascade
```

---

### Computational Parameters

Parameters controlling computational aspects of THOL (history, timing).

| Parameter | Default | Range | Units | Description |
|-----------|---------|-------|-------|-------------|
| `THOL_MIN_HISTORY_LENGTH` | 3 | [2, 10] | — | Minimum EPI history points required to compute ∂²EPI/∂t² |

**Physical Basis:**

Computing structural acceleration ∂²EPI/∂t² requires **finite difference approximation**:

```
∂²EPI/∂t² ≈ EPI(t) - 2·EPI(t-1) + EPI(t-2)
```

This requires ≥3 historical EPI values. More history points enable higher-order approximations but increase memory.

**Configuration Example:**
```python
# Minimal history (standard second derivative)
G.graph["THOL_MIN_HISTORY_LENGTH"] = 3

# Extended history (smoother acceleration estimates)
G.graph["THOL_MIN_HISTORY_LENGTH"] = 5

# Maximum history (long-term dynamics)
G.graph["THOL_MIN_HISTORY_LENGTH"] = 10
```

---

## Canonical Constraints

These constraints are **physically necessary** to maintain TNFR coherence. Violating them may cause non-physical behavior or system instability.

### C1: Metabolic Weight Sum

**Constraint:**
```
THOL_METABOLIC_GRADIENT_WEIGHT + THOL_METABOLIC_COMPLEXITY_WEIGHT ≤ 0.5
```

**Rationale:** Network contribution should not dominate internal bifurcation. Sub-EPI structure emerges primarily from internal acceleration (base_internal = 25% of parent), with network context providing **modulation** rather than determination.

**Violation:** If weights sum > 0.5, network signals overwhelm internal structure → loss of autopoietic autonomy (THOL becomes externally driven rather than self-organizing).

**Example:**
```python
# Valid: Balanced influence
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.15
G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.10
# Sum = 0.25 < 0.5 ✓

# Invalid: Over-influence
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.30
G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.25
# Sum = 0.55 > 0.5 ✗ (network dominates, not self-organization)
```

### C2: Propagation Attenuation

**Constraint:**
```
0 < THOL_PROPAGATION_ATTENUATION < 1.0
```

**Rationale:** Attenuation factor must be:
- **> 0**: Propagation occurs (sub-EPIs reach neighbors)
- **< 1**: Energy dissipates with distance (prevents unbounded growth)

**Violation:** 
- If = 0: No propagation (isolated bifurcation only)
- If ≥ 1: Amplification → exponential growth → structural fragmentation

**Example:**
```python
# Valid: Attenuation preserves boundedness
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.7  # 30% loss per hop ✓

# Invalid: No attenuation
G.graph["THOL_PROPAGATION_ATTENUATION"] = 1.0  # ✗ (unbounded growth risk)

# Invalid: Amplification
G.graph["THOL_PROPAGATION_ATTENUATION"] = 1.2  # ✗ (explosive dynamics)
```

### C3: Cascade Minimum Nodes

**Constraint:**
```
THOL_CASCADE_MIN_NODES ≥ 2
```

**Rationale:** A cascade is a **chain reaction**. Minimum chain = source → target (2 nodes). Single-node bifurcation is not a cascade.

**Violation:** If = 1, every bifurcation classified as cascade (semantic breakdown).

**Example:**
```python
# Valid: Minimum chain
G.graph["THOL_CASCADE_MIN_NODES"] = 2  # ✓

# Valid: Conservative cascade detection
G.graph["THOL_CASCADE_MIN_NODES"] = 5  # ✓ (require larger chain)

# Invalid: Single-node "cascade"
G.graph["THOL_CASCADE_MIN_NODES"] = 1  # ✗ (meaningless)
```

### C4: History Length

**Constraint:**
```
THOL_MIN_HISTORY_LENGTH ≥ 3
```

**Rationale:** Second derivative requires 3 points minimum (t, t-1, t-2) for finite difference approximation.

**Violation:** If < 3, cannot compute ∂²EPI/∂t² → bifurcation threshold check impossible.

**Example:**
```python
# Valid: Minimum for second derivative
G.graph["THOL_MIN_HISTORY_LENGTH"] = 3  # ✓

# Invalid: Insufficient for ∂²EPI/∂t²
G.graph["THOL_MIN_HISTORY_LENGTH"] = 2  # ✗ (cannot compute acceleration)
```

### C5: Bifurcation Threshold Positivity

**Constraint:**
```
BIFURCATION_THRESHOLD_TAU > 0
```

**Rationale:** Negative or zero threshold would make every state bifurcation-ready (all accelerations ≥ 0). Threshold must be positive to filter noise and distinguish significant structural transitions.

**Violation:** If ≤ 0, constant bifurcation → fragmentation chaos.

**Example:**
```python
# Valid: Positive threshold
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.5  # ✓

# Invalid: Zero threshold
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.0  # ✗ (always bifurcates)

# Invalid: Negative threshold
G.graph["BIFURCATION_THRESHOLD_TAU"] = -0.1  # ✗ (non-physical)
```

---

## Usage Examples

### Example 1: Conservative Bifurcation (Stable Systems)

Use case: Modeling systems where reorganization should be rare and requires strong conditions.

```python
import networkx as nx
from tnfr.structural import create_nfr
from tnfr.operators import SelfOrganization

# Create graph with conservative THOL settings
G = nx.Graph()
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.8     # High acceleration required
G.graph["THOL_MIN_EPI"] = 0.4                  # Strong coherence required
G.graph["THOL_MIN_VF"] = 0.3                   # High reorganization capacity
G.graph["THOL_MIN_DEGREE"] = 2                 # Require connectivity
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.10  # Low external influence
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.5     # High attenuation (local)

# Add node
node_id = 0
G.add_node(node_id, epi=0.5, vf=0.5, theta=0.0, delta_nfr=0.1)
G.nodes[node_id]["epi_history"] = [0.3, 0.4, 0.5]  # Moderate acceleration

# Apply THOL
thol = SelfOrganization()
try:
    thol(G, node_id)
    print("Bifurcation occurred (rare event)")
except Exception as e:
    print(f"Bifurcation blocked: {e}")
    # Expected: threshold not met with conservative settings
```

### Example 2: Sensitive Bifurcation (Exploratory Systems)

Use case: Modeling systems where reorganization should be frequent and responsive to small changes.

```python
import networkx as nx
from tnfr.structural import create_nfr
from tnfr.operators import SelfOrganization

# Create graph with sensitive THOL settings
G = nx.Graph()
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.2     # Low acceleration required
G.graph["THOL_MIN_EPI"] = 0.1                  # Low coherence floor
G.graph["THOL_MIN_VF"] = 0.05                  # Low capacity floor
G.graph["THOL_ALLOW_ISOLATED"] = True          # Allow isolated bifurcation
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.20  # High external influence
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.85    # Low attenuation (wide spread)

# Add node
node_id = 0
G.add_node(node_id, epi=0.15, vf=0.1, theta=0.0, delta_nfr=0.05)
G.nodes[node_id]["epi_history"] = [0.10, 0.12, 0.15]  # Small acceleration

# Apply THOL
thol = SelfOrganization()
thol(G, node_id)
print("Bifurcation likely occurred (sensitive settings)")
```

### Example 3: Network-Driven Metabolism

Use case: Bifurcation primarily driven by network context (social systems, neural networks).

```python
import networkx as nx
from tnfr.operators import SelfOrganization

# Create network
G = nx.karate_club_graph()

# Configure strong metabolic influence
G.graph["THOL_METABOLIC_ENABLED"] = True
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.25  # Maximum recommended
G.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 0.15  # High complexity weight
G.graph["THOL_MIN_DEGREE"] = 3                     # Require rich connectivity
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.4  # Broad propagation

# Initialize nodes
for node in G.nodes():
    G.nodes[node]["epi"] = 0.3
    G.nodes[node]["vf"] = 0.5
    G.nodes[node]["theta"] = 0.0
    G.nodes[node]["delta_nfr"] = 0.1
    G.nodes[node]["epi_history"] = [0.2, 0.25, 0.3]

# Apply THOL to highly connected node
central_node = max(G.degree(), key=lambda x: x[1])[0]
thol = SelfOrganization()
thol(G, central_node)

# Check propagation
propagations = G.graph.get("thol_propagations", [])
print(f"Propagated to {len(propagations)} nodes")
```

### Example 4: Isolated Internal Bifurcation

Use case: Testing THOL without network effects, or single-node systems.

```python
import networkx as nx
from tnfr.operators import SelfOrganization

# Create single-node graph
G = nx.Graph()
G.graph["THOL_ALLOW_ISOLATED"] = True        # Enable isolated THOL
G.graph["THOL_METABOLIC_ENABLED"] = False    # Disable metabolism
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.3   # Moderate threshold

# Add isolated node
node_id = 0
G.add_node(node_id, epi=0.4, vf=0.5, theta=0.0, delta_nfr=0.15)
G.nodes[node_id]["epi_history"] = [0.2, 0.3, 0.4]  # Strong acceleration

# Apply THOL (pure internal bifurcation)
thol = SelfOrganization()
thol(G, node_id)

# Check sub-EPIs (no propagation possible)
sub_epis = G.nodes[node_id].get("sub_epis", [])
print(f"Generated {len(sub_epis)} sub-EPIs (isolated bifurcation)")
```

### Example 5: Cascade Detection

Use case: Analyzing network-wide reorganization patterns.

```python
import networkx as nx
from tnfr.operators import SelfOrganization
from tnfr.operators.cascade import detect_cascade

# Create network with cascade-friendly settings
G = nx.erdos_renyi_graph(n=20, p=0.3)
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.3
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.4
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.8
G.graph["THOL_CASCADE_MIN_NODES"] = 4  # Require ≥4 nodes for cascade

# Initialize network
for node in G.nodes():
    G.nodes[node]["epi"] = 0.3
    G.nodes[node]["vf"] = 0.5
    G.nodes[node]["theta"] = 0.0
    G.nodes[node]["delta_nfr"] = 0.1
    G.nodes[node]["epi_history"] = [0.2, 0.25, 0.3]

# Trigger THOL on multiple nodes
thol = SelfOrganization()
for node in list(G.nodes())[:5]:  # Apply to 5 seed nodes
    try:
        thol(G, node)
    except Exception:
        pass

# Analyze cascade
cascade_info = detect_cascade(G)
if cascade_info["is_cascade"]:
    print(f"Cascade detected!")
    print(f"  Affected nodes: {len(cascade_info['affected_nodes'])}")
    print(f"  Cascade depth: {cascade_info['cascade_depth']}")
    print(f"  Total propagations: {cascade_info['total_propagations']}")
else:
    print("No cascade (isolated bifurcations)")
```

---

## Parameter Interaction Matrix

This table shows how parameters interact and which combinations are meaningful.

| Parameter Group | Interacts With | Nature of Interaction |
|----------------|----------------|----------------------|
| **Bifurcation Threshold** | History Length | History provides data for ∂²EPI/∂t² comparison with threshold |
| **Metabolic Weights** | MIN_DEGREE, ALLOW_ISOLATED | Weights only apply when metabolism enabled and node coupled |
| **Propagation Settings** | MIN_COUPLING, Cascade Detection | Propagation enables cascades; coupling filters propagation targets |
| **Network Requirements** | METABOLIC_ENABLED | Degree requirements only enforced when metabolism enabled |
| **History Length** | THOL_ACCEL | Both relate to acceleration: history computes it, accel applies it |

### Critical Dependencies

1. **Metabolism requires connectivity:**
   ```
   If METABOLIC_ENABLED=True, then:
     - Either ALLOW_ISOLATED=True OR node degree ≥ MIN_DEGREE
   ```

2. **Propagation requires metabolism:**
   ```
   If MIN_COUPLING_FOR_PROPAGATION < 1.0, then:
     - METABOLIC_ENABLED should be True (else no network context to propagate)
   ```

3. **Cascade requires propagation:**
   ```
   If CASCADE_MIN_NODES > 1, then:
     - PROPAGATION_ATTENUATION should be < 1.0 (else exponential growth)
     - MIN_COUPLING_FOR_PROPAGATION should allow some propagation
   ```

---

## Troubleshooting Common Configurations

### Issue: "THOL never bifurcates"

**Possible causes:**
1. **Threshold too high:** `BIFURCATION_THRESHOLD_TAU` > typical acceleration
2. **EPI too low:** Node EPI < `THOL_MIN_EPI`
3. **νf too low:** Node νf < `THOL_MIN_VF`
4. **Insufficient history:** `epi_history` length < `THOL_MIN_HISTORY_LENGTH`

**Solution:**
```python
# Lower bifurcation requirements
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.2  # From 0.5
G.graph["THOL_MIN_EPI"] = 0.1              # From 0.2

# Ensure history accumulates
for node in G.nodes():
    if "epi_history" not in G.nodes[node]:
        G.nodes[node]["epi_history"] = []
```

### Issue: "Sub-EPIs too small / too large"

**Possible causes:**
1. **Metabolic weights misconfigured:** Weights sum > 0.5 or < 0
2. **Network signals extreme:** Very high epi_gradient or phase_variance

**Solution:**
```python
# Verify weight sum constraint
gradient_w = G.graph.get("THOL_METABOLIC_GRADIENT_WEIGHT", 0.15)
complexity_w = G.graph.get("THOL_METABOLIC_COMPLEXITY_WEIGHT", 0.10)
assert gradient_w + complexity_w <= 0.5, "Weights exceed limit"

# Normalize network signals if extreme
# (Or adjust weights to compensate)
G.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = 0.10  # Reduce influence
```

### Issue: "No cascades detected"

**Possible causes:**
1. **Propagation blocked:** `MIN_COUPLING_FOR_PROPAGATION` too high
2. **Attenuation too strong:** `PROPAGATION_ATTENUATION` < 0.5
3. **Cascade threshold too high:** `CASCADE_MIN_NODES` > actual affected nodes
4. **Network disconnected:** Phase misalignment blocks propagation

**Solution:**
```python
# Enable broader propagation
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.4  # From 0.7
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.85     # From 0.5
G.graph["THOL_CASCADE_MIN_NODES"] = 2              # From 3

# Verify phase alignment
from tnfr.utils.numeric import angle_diff
for u, v in G.edges():
    phase_diff = abs(angle_diff(G.nodes[u]["theta"], G.nodes[v]["theta"]))
    print(f"Edge {u}-{v}: phase_diff = {phase_diff:.2f}")
```

### Issue: "Cascades too aggressive / unstable"

**Possible causes:**
1. **Attenuation too weak:** `PROPAGATION_ATTENUATION` ≥ 0.95
2. **Coupling threshold too low:** `MIN_COUPLING_FOR_PROPAGATION` < 0.3
3. **Threshold too low:** `BIFURCATION_THRESHOLD_TAU` < 0.2

**Solution:**
```python
# Stabilize cascades
G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.6      # Stronger damping
G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.6  # Stricter coupling
G.graph["BIFURCATION_THRESHOLD_TAU"] = 0.5         # Higher threshold
```

---

## Cross-References

### Related Documentation

- **[THOL_ENCAPSULATION_GUIDE.md](THOL_ENCAPSULATION_GUIDE.md)** — Operator sequence behavior and grammar rules
- **[UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md)** — U4 (Bifurcation Dynamics) derivation
- **[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)** — Invariant #7 (Operational Fractality)
- **[TNFR.pdf](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR.pdf)** — §2.2.10 (Self-organization operator theory)

### Source Code References

- **`src/tnfr/operators/preconditions/__init__.py`** — `validate_self_organization()` (line 708)
- **`src/tnfr/operators/metabolism.py`** — Metabolic functions
- **`src/tnfr/operators/cascade.py`** — Cascade detection
- **`src/tnfr/operators/__init__.py`** — `_op_THOL()` glyph implementation (line 1348)
- **`src/tnfr/operators/definitions.py`** — `SelfOrganization` class

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-09 | Initial release: Centralized all THOL parameters with canonical constraints |
| 1.1.0 | 2025-11-09 | Added `THOL_MIN_COLLECTIVE_COHERENCE` parameter for sub-EPI ensemble validation |

---

*This reference is maintained as the **single source of truth** for THOL configuration. All parameter defaults are verified against source code.*

*For questions or corrections, see [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md).*
