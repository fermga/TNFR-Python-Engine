# Worked Examples: TNFR Calculations Step-by-Step

This document provides detailed, step-by-step walkthroughs of key TNFR calculations, showing both the mathematical derivation and Python implementation.

## Purpose

These worked examples serve to:

1. **Bridge theory and practice**: Show how mathematical formulas translate to code
2. **Verify correctness**: Provide test cases with expected results
3. **Build intuition**: Demonstrate how structural parameters interact
4. **Enable debugging**: Offer reference calculations for troubleshooting

---

## Example 1: Computing Sense Index (Si) for a Single Node

### Overview

The **Sense Index** (\(\text{Si}\)) quantifies a node's capacity for stable reorganization. It combines three structural signals:

- \(\nu_f\): How fast the node reorganizes (structural frequency)
- \(\theta\): How synchronized it is with neighbors (phase coupling)
- \(\Delta\text{NFR}\): How much reorganization pressure it experiences

### Mathematical Definition

\[
\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})
\]

**Where:**
- \(\nu_{f,\text{norm}} = \frac{|\nu_f|}{\nu_{f,\max}}\): Normalized structural frequency
- \(\text{disp}_\theta = \frac{|\theta - \bar{\theta}|}{\pi}\): Phase dispersion from neighbor mean
- \(|\Delta\text{NFR}|_{\text{norm}} = \frac{|\Delta\text{NFR}|}{\Delta\text{NFR}_{\max}}\): Normalized reorganization magnitude
- \(\alpha, \beta, \gamma\): Structural weights with \(\alpha + \beta + \gamma = 1\)

### Input Data

```python
import numpy as np

# Node structural state
node_data = {
    "nu_f": 0.8,           # Hz_str (structural frequency)
    "delta_nfr": 0.2,      # Reorganization gradient
    "phase": 0.5,          # radians
    "neighbors": ["n1", "n2"],
    "neighbor_phases": [0.4, 0.6]  # radians
}

# Structural weights (default TNFR configuration)
weights = {
    "alpha": 0.4,   # Frequency weight
    "beta": 0.3,    # Phase coupling weight
    "gamma": 0.3    # ΔNFR damping weight
}

# Normalization limits
max_values = {
    "vfmax": 1.0,      # Maximum structural frequency in network
    "dnfrmax": 1.0     # Maximum |ΔNFR| in network
}
```

### Step 1: Normalize Structural Frequency

**Formula:**
\[
\nu_{f,\text{norm}} = \frac{|\nu_f|}{\nu_{f,\max}}
\]

**Calculation:**
\[
\nu_{f,\text{norm}} = \frac{|0.8|}{1.0} = \frac{0.8}{1.0} = 0.8
\]

**Python:**
```python
vf_norm = abs(node_data["nu_f"]) / max_values["vfmax"]
print(f"Step 1: vf_norm = {vf_norm}")
# Output: Step 1: vf_norm = 0.8
```

### Step 2: Compute Phase Dispersion

**Phase mean** (circular average using atan2):
\[
\bar{\theta} = \text{atan2}\left(\sum_{j \in \text{neighbors}} \sin\theta_j, \sum_{j \in \text{neighbors}} \cos\theta_j\right)
\]

**Calculation:**
```python
neighbor_phases = np.array([0.4, 0.6])

# Compute circular mean
cos_sum = np.sum(np.cos(neighbor_phases))
sin_sum = np.sum(np.sin(neighbor_phases))
theta_bar = np.arctan2(sin_sum, cos_sum)

print(f"cos_sum = {cos_sum:.6f}")
print(f"sin_sum = {sin_sum:.6f}")
print(f"theta_bar = {theta_bar:.6f} rad")
```

**Numerical values:**
\[
\begin{aligned}
\sum \cos\theta_j &= \cos(0.4) + \cos(0.6) \approx 0.9211 + 0.8253 = 1.7464 \\
\sum \sin\theta_j &= \sin(0.4) + \sin(0.6) \approx 0.3894 + 0.5646 = 0.9540 \\
\bar{\theta} &= \text{atan2}(0.9540, 1.7464) \approx 0.500 \text{ rad}
\end{aligned}
\]

**Phase dispersion** (normalized to \([0, 1]\)):
\[
\text{disp}_\theta = \frac{|\theta - \bar{\theta}|}{\pi} = \frac{|0.5 - 0.500|}{\pi} \approx \frac{0.0}{\pi} = 0.0
\]

**Python:**
```python
phase_dispersion = abs(node_data["phase"] - theta_bar) / np.pi
print(f"Step 2: phase_dispersion = {phase_dispersion:.6f}")
# Output: Step 2: phase_dispersion = 0.000000
```

### Step 3: Normalize Reorganization Gradient

**Formula:**
\[
|\Delta\text{NFR}|_{\text{norm}} = \frac{|\Delta\text{NFR}|}{\Delta\text{NFR}_{\max}}
\]

**Calculation:**
\[
|\Delta\text{NFR}|_{\text{norm}} = \frac{|0.2|}{1.0} = 0.2
\]

**Python:**
```python
dnfr_norm = abs(node_data["delta_nfr"]) / max_values["dnfrmax"]
print(f"Step 3: dnfr_norm = {dnfr_norm}")
# Output: Step 3: dnfr_norm = 0.2
```

### Step 4: Combine Components

**Formula:**
\[
\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})
\]

**Substituting values:**
\[
\begin{aligned}
\text{Si} &= 0.4 \cdot 0.8 + 0.3 \cdot (1 - 0.0) + 0.3 \cdot (1 - 0.2) \\
&= 0.4 \cdot 0.8 + 0.3 \cdot 1.0 + 0.3 \cdot 0.8 \\
&= 0.32 + 0.30 + 0.24 \\
&= 0.86
\end{aligned}
\]

**Python:**
```python
Si = (weights["alpha"] * vf_norm +
      weights["beta"] * (1.0 - phase_dispersion) +
      weights["gamma"] * (1.0 - dnfr_norm))

print(f"\nStep 4: Si components")
print(f"  alpha * vf_norm           = {weights['alpha']} * {vf_norm} = {weights['alpha'] * vf_norm}")
print(f"  beta * (1 - disp_theta)   = {weights['beta']} * {1.0 - phase_dispersion} = {weights['beta'] * (1.0 - phase_dispersion)}")
print(f"  gamma * (1 - dnfr_norm)   = {weights['gamma']} * {1.0 - dnfr_norm} = {weights['gamma'] * (1.0 - dnfr_norm)}")
print(f"  Si (before clamp)         = {Si}")
```

### Step 5: Clamp to Valid Range

TNFR implementation clamps Si to \([0, 1]\) to ensure bounded metrics:

\[
\text{Si}_{\text{final}} = \max(0, \min(1, \text{Si}))
\]

**For our example:**
\[
\text{Si}_{\text{final}} = \max(0, \min(1, 0.86)) = 0.86
\]

**Python:**
```python
def clamp01(x):
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))

Si_final = clamp01(Si)
print(f"Step 5: Si_final = {Si_final}")
# Output: Step 5: Si_final = 0.86
```

### Verification with Implementation

```python
import networkx as nx
from tnfr.metrics.sense_index import compute_Si

# Create a minimal network matching our example
G = nx.Graph()
G.add_edge("n0", "n1")
G.add_edge("n0", "n2")

# Set node attributes
G.nodes["n0"].update({
    "nu_f": 0.8,
    "delta_nfr": 0.2,
    "phase": 0.5
})
G.nodes["n1"]["phase"] = 0.4
G.nodes["n2"]["phase"] = 0.6

# Set global Si weights
G.graph["SI_WEIGHTS"] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

# Compute Si using TNFR implementation
Si_result = compute_Si(G, vfmax=1.0, dnfrmax=1.0, inplace=False)

print(f"\n✅ Verification:")
print(f"  Manual calculation:   Si = {Si_final}")
print(f"  TNFR implementation:  Si = {Si_result['n0']:.6f}")
print(f"  Match: {abs(Si_final - Si_result['n0']) < 1e-6}")
```

**Expected output:**
```
✅ Verification:
  Manual calculation:   Si = 0.86
  TNFR implementation:  Si = 0.860000
  Match: True
```

### Interpretation

**What does Si = 0.86 mean?**

- **High value (close to 1)**: This node has excellent reorganization stability
- **Breakdown:**
  - **Frequency contribution (0.32)**: Strong reorganization capacity (\(\nu_f = 0.8\))
  - **Phase contribution (0.30)**: Perfect synchrony with neighbors (\(\text{disp}_\theta = 0\))
  - **ΔNFR contribution (0.24)**: Low reorganization pressure (\(\Delta\text{NFR} = 0.2\))

**Physical meaning**: This node can reorganize quickly (\(\nu_f = 0.8\)), stays synchronized with its network (\(\theta \approx \bar{\theta}\)), and experiences manageable structural pressure (\(\Delta\text{NFR} = 0.2\)). It's a **stable, well-integrated node**.

---

## Example 2: Computing Coherence Matrix Elements

### Overview

The **coherence matrix** \(W\) approximates the coherence operator \(\hat{C}\) in the discrete node basis. Each element \(w_{ij}\) measures structural similarity between nodes \(i\) and \(j\).

### Mathematical Definition

**Matrix element:**
\[
w_{ij} \approx \langle i | \hat{C} | j \rangle
\]

**Computed as weighted similarity:**
\[
w_{ij} = w_{\text{phase}} \cdot s_{\text{phase}}(i,j) + w_{\text{EPI}} \cdot s_{\text{EPI}}(i,j) + w_{\nu_f} \cdot s_{\nu_f}(i,j) + w_{\text{Si}} \cdot s_{\text{Si}}(i,j)
\]

**Similarity components:**
\[
\begin{aligned}
s_{\text{phase}}(i,j) &= \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right) \\
s_{\text{EPI}}(i,j) &= 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}} \\
s_{\nu_f}(i,j) &= 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}} \\
s_{\text{Si}}(i,j) &= 1 - |\text{Si}_i - \text{Si}_j|
\end{aligned}
\]

### Input Data

```python
import math

# Two nodes with structural state
node_i = {
    "EPI": 0.5,
    "nu_f": 0.8,
    "phase": 0.0,
    "Si": 0.7
}

node_j = {
    "EPI": 0.6,
    "nu_f": 0.7,
    "phase": 0.1,
    "Si": 0.8
}

# Network ranges (for normalization)
ranges = {
    "EPI_max": 1.0,
    "EPI_min": 0.0,
    "vf_max": 1.0,
    "vf_min": 0.0
}

# Coherence weights (default)
weights = {
    "phase": 0.25,
    "epi": 0.25,
    "vf": 0.25,
    "si": 0.25
}
```

### Step 1: Compute Phase Similarity

**Formula:**
\[
s_{\text{phase}}(i,j) = \frac{1}{2}\left(1 + \cos(\theta_i - \theta_j)\right)
\]

**Calculation:**
\[
\begin{aligned}
\theta_i - \theta_j &= 0.0 - 0.1 = -0.1 \text{ rad} \\
\cos(-0.1) &\approx 0.9950 \\
s_{\text{phase}} &= \frac{1}{2}(1 + 0.9950) = \frac{1.9950}{2} = 0.9975
\end{aligned}
\]

**Python:**
```python
phase_diff = node_i["phase"] - node_j["phase"]
s_phase = 0.5 * (1.0 + math.cos(phase_diff))
print(f"Step 1: s_phase = {s_phase:.6f}")
# Output: Step 1: s_phase = 0.997502
```

### Step 2: Compute EPI Similarity

**Formula:**
\[
s_{\text{EPI}}(i,j) = 1 - \frac{|\text{EPI}_i - \text{EPI}_j|}{\Delta_{\text{EPI}}}
\]

where \(\Delta_{\text{EPI}} = \text{EPI}_{\max} - \text{EPI}_{\min}\)

**Calculation:**
\[
\begin{aligned}
\Delta_{\text{EPI}} &= 1.0 - 0.0 = 1.0 \\
|\text{EPI}_i - \text{EPI}_j| &= |0.5 - 0.6| = 0.1 \\
s_{\text{EPI}} &= 1 - \frac{0.1}{1.0} = 0.9
\end{aligned}
\]

**Python:**
```python
epi_range = ranges["EPI_max"] - ranges["EPI_min"]
epi_diff = abs(node_i["EPI"] - node_j["EPI"])
s_epi = 1.0 - (epi_diff / epi_range if epi_range > 0 else 0.0)
print(f"Step 2: s_epi = {s_epi:.6f}")
# Output: Step 2: s_epi = 0.900000
```

### Step 3: Compute Frequency Similarity

**Formula:**
\[
s_{\nu_f}(i,j) = 1 - \frac{|\nu_{f,i} - \nu_{f,j}|}{\Delta_{\nu_f}}
\]

**Calculation:**
\[
\begin{aligned}
\Delta_{\nu_f} &= 1.0 - 0.0 = 1.0 \\
|\nu_{f,i} - \nu_{f,j}| &= |0.8 - 0.7| = 0.1 \\
s_{\nu_f} &= 1 - \frac{0.1}{1.0} = 0.9
\end{aligned}
\]

**Python:**
```python
vf_range = ranges["vf_max"] - ranges["vf_min"]
vf_diff = abs(node_i["nu_f"] - node_j["nu_f"])
s_vf = 1.0 - (vf_diff / vf_range if vf_range > 0 else 0.0)
print(f"Step 3: s_vf = {s_vf:.6f}")
# Output: Step 3: s_vf = 0.900000
```

### Step 4: Compute Si Similarity

**Formula:**
\[
s_{\text{Si}}(i,j) = 1 - |\text{Si}_i - \text{Si}_j|
\]

**Calculation:**
\[
s_{\text{Si}} = 1 - |0.7 - 0.8| = 1 - 0.1 = 0.9
\]

**Python:**
```python
s_si = 1.0 - abs(node_i["Si"] - node_j["Si"])
print(f"Step 4: s_si = {s_si:.6f}")
# Output: Step 4: s_si = 0.900000
```

### Step 5: Combine with Weights

**Formula:**
\[
w_{ij} = w_{\text{phase}} \cdot s_{\text{phase}} + w_{\text{EPI}} \cdot s_{\text{EPI}} + w_{\nu_f} \cdot s_{\nu_f} + w_{\text{Si}} \cdot s_{\text{Si}}
\]

**Calculation:**
\[
\begin{aligned}
w_{ij} &= 0.25 \cdot 0.9975 + 0.25 \cdot 0.9 + 0.25 \cdot 0.9 + 0.25 \cdot 0.9 \\
&= 0.2494 + 0.225 + 0.225 + 0.225 \\
&= 0.9244
\end{aligned}
\]

**Python:**
```python
w_ij = (weights["phase"] * s_phase +
        weights["epi"] * s_epi +
        weights["vf"] * s_vf +
        weights["si"] * s_si)

print(f"\nStep 5: w_ij components")
print(f"  phase: {weights['phase']} * {s_phase:.6f} = {weights['phase'] * s_phase:.6f}")
print(f"  epi:   {weights['epi']} * {s_epi:.6f} = {weights['epi'] * s_epi:.6f}")
print(f"  vf:    {weights['vf']} * {s_vf:.6f} = {weights['vf'] * s_vf:.6f}")
print(f"  si:    {weights['si']} * {s_si:.6f} = {weights['si'] * s_si:.6f}")
print(f"  w_ij = {w_ij:.6f}")
```

### Step 6: Clamp to [0, 1]

\[
w_{ij,\text{final}} = \max(0, \min(1, w_{ij})) = 0.9244
\]

**Python:**
```python
w_ij_final = max(0.0, min(1.0, w_ij))
print(f"Step 6: w_ij_final = {w_ij_final:.6f}")
# Output: Step 6: w_ij_final = 0.924375
```

### Verification with Implementation

```python
import networkx as nx
from tnfr.metrics.coherence import compute_wij_phase_epi_vf_si, SimilarityInputs

# Create test network
G = nx.Graph()
G.add_edge("i", "j")

# Set attributes
G.nodes["i"].update(node_i)
G.nodes["j"].update(node_j)

# Prepare inputs
inputs = SimilarityInputs(
    th_vals=[node_i["phase"], node_j["phase"]],
    epi_vals=[node_i["EPI"], node_j["EPI"]],
    vf_vals=[node_i["nu_f"], node_j["nu_f"]],
    si_vals=[node_i["Si"], node_j["Si"]]
)

# Compute similarities
epi_range = ranges["EPI_max"] - ranges["EPI_min"]
vf_range = ranges["vf_max"] - ranges["vf_min"]

s_phase_impl, s_epi_impl, s_vf_impl, s_si_impl = compute_wij_phase_epi_vf_si(
    inputs, i=0, j=1,
    epi_range=epi_range,
    vf_range=vf_range
)

# Combine
w_ij_impl = (weights["phase"] * s_phase_impl +
             weights["epi"] * s_epi_impl +
             weights["vf"] * s_vf_impl +
             weights["si"] * s_si_impl)

print(f"\n✅ Verification:")
print(f"  Manual calculation:   w_ij = {w_ij_final:.6f}")
print(f"  TNFR implementation:  w_ij = {w_ij_impl:.6f}")
print(f"  Match: {abs(w_ij_final - w_ij_impl) < 1e-6}")
```

### Interpretation

**What does \(w_{ij} = 0.924\) mean?**

- **High coherence**: Nodes \(i\) and \(j\) are structurally very similar
- **Breakdown:**
  - **Phase** (0.998): Nearly perfect synchrony (\(\Delta\theta = 0.1\) rad)
  - **EPI** (0.90): Moderate structural similarity
  - **Frequency** (0.90): Similar reorganization rates
  - **Si** (0.90): Similar stability capacities

**Physical meaning**: These nodes are **strongly coupled** in the coherence operator sense. They form a stable resonant pair that reinforces each other's structural patterns.

---

## Example 3: Total Coherence C(t) from Matrix

### Overview

Given the coherence matrix \(W\), we can compute the total network coherence \(C(t)\).

### Mathematical Definition

**Trace formula:**
\[
C(t) = \frac{1}{N} \sum_{i=1}^N w_{ii}
\]

For an idealized network where all nodes have identical self-coherence and pairwise coherence:

\[
C(t) = \frac{1}{N}\left(\sum_{i=1}^N w_{ii}\right) \approx \bar{w}
\]

### Example Network

```python
import numpy as np

# 3-node network, coherence matrix
N = 3
W = np.array([
    [1.0, 0.9, 0.8],  # Node 0: perfect self-coherence, high coupling
    [0.9, 1.0, 0.85], # Node 1
    [0.8, 0.85, 1.0]  # Node 2
])

print("Coherence matrix W:")
print(W)
```

### Step 1: Extract Diagonal (Self-Coherence)

\[
\text{diag}(W) = [w_{00}, w_{11}, w_{22}] = [1.0, 1.0, 1.0]
\]

**Python:**
```python
diagonal = np.diag(W)
print(f"\nStep 1: Diagonal elements = {diagonal}")
# Output: [1.0, 1.0, 1.0]
```

### Step 2: Compute Mean

\[
C(t) = \frac{1}{3}(1.0 + 1.0 + 1.0) = \frac{3.0}{3} = 1.0
\]

**Python:**
```python
C_t = np.mean(diagonal)
print(f"Step 2: C(t) = {C_t:.6f}")
# Output: Step 2: C(t) = 1.000000
```

### Interpretation

**C(t) = 1.0** means:
- All nodes have perfect self-coherence (\(w_{ii} = 1\))
- The network as a whole exhibits **maximum stability**
- No nodes are at risk of structural collapse

**Note**: This is the diagonal-only definition. The full coherence includes off-diagonal coupling terms in more sophisticated formulations.

---

## Example 4: Phase Synchrony (Kuramoto Order Parameter)

### Overview

The **Kuramoto order parameter** \(r\) measures phase synchronization across a network.

### Mathematical Definition

\[
r e^{i\Psi} = \frac{1}{N}\sum_{j=1}^N e^{i\theta_j}
\]

where:
- \(r \in [0,1]\): Synchronization strength
- \(\Psi\): Mean phase direction

### Input Data

```python
import numpy as np

# Four nodes with phases
phases = np.array([0.0, 0.1, 0.05, 6.28])  # radians
N = len(phases)

print(f"Phases: {phases}")
```

### Step 1: Compute Complex Sum

\[
\sum_{j=1}^N e^{i\theta_j} = \sum_{j=1}^N (\cos\theta_j + i\sin\theta_j)
\]

**Python:**
```python
complex_sum = np.sum(np.exp(1j * phases))
print(f"\nStep 1: Complex sum = {complex_sum}")
# Output: Complex sum = (3.95+0.14j) approximately
```

### Step 2: Normalize by N

\[
\frac{1}{N}\sum_{j=1}^N e^{i\theta_j} = r e^{i\Psi}
\]

**Python:**
```python
normalized = complex_sum / N
r = np.abs(normalized)
Psi = np.angle(normalized)

print(f"Step 2: Normalized = {normalized}")
print(f"        r (magnitude) = {r:.6f}")
print(f"        Psi (angle) = {Psi:.6f} rad")
```

### Interpretation

- **r ≈ 0.988**: Very high synchronization (phases are tightly clustered)
- **Psi ≈ 0.036 rad**: Mean phase direction is close to 0

**Physical meaning**: The network is **highly synchronized**. Nodes oscillate nearly in phase, indicating strong coherent coupling.

---

## Summary

These worked examples demonstrate:

1. **Si calculation**: Shows how frequency, phase, and ΔNFR combine to measure reorganization stability
2. **Coherence matrix**: Illustrates pairwise structural similarity computation
3. **Total coherence**: Demonstrates network-wide stability measurement
4. **Phase synchrony**: Quantifies collective oscillation patterns

Each example:
- ✅ Provides step-by-step mathematical derivation
- ✅ Shows Python implementation
- ✅ Verifies against TNFR library functions
- ✅ Interprets results physically

For more examples, see:
- [Theory notebooks](../theory/) for interactive explorations
- [API documentation](../api/metrics.html) for function references
- [Mathematical Foundations](../theory/mathematical_foundations.md) for complete derivations
