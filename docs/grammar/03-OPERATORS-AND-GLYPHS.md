# Operators and Glyphs: The 13 Canonical TNFR Operators

**Complete catalog of structural transformations in TNFR**

[🏠 Home](README.md) • [🌊 Concepts](01-FUNDAMENTAL-CONCEPTS.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [🔄 Sequences](04-VALID-SEQUENCES.md)

---

## Purpose

This document provides a **complete catalog** of the 13 canonical TNFR operators. Each operator is a **resonant transformation** with rigorous physical meaning, not an arbitrary function.

For each operator, you'll find:
- **Physics:** What structural transformation does it represent?
- **Effect:** Impact on ∂EPI/∂t and node properties
- **When to use:** Appropriate use cases
- **Grammar classification:** Role in U1-U4 constraints
- **Contract:** Pre/postconditions
- **Examples:** Executable code

**Prerequisites:** [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md), [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)

**Reading time:** 60-90 minutes (reference document)

---

## Overview

The 13 operators form a **complete, non-redundant** set:

```
Initialization:    AL (Emission), NAV (Transition), REMESH (Recursivity)
Information:       EN (Reception)
Stabilization:     IL (Coherence), THOL (Self-organization)
Destabilization:   OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)
Propagation:       UM (Coupling), RA (Resonance)
Control:           SHA (Silence), NUL (Contraction)
```

---

## 1. Emission (AL) 🎵

### Physics
Creates EPI from vacuum via resonant emission. Like a quantum field producing a particle from vacuum fluctuations.

### Effect
- **∂EPI/∂t > 0** - Increases structure
- **νf increases** - Enhances reorganization capacity
- Creates initial coherent pattern

### When to Use
- Starting new patterns from EPI=0
- Initializing nodes in a network
- Bootstrap sequences

### Grammar Classification
- **Generator** (U1a) ✓
- **Closure** (U1b) ✗
- **Stabilizer** (U2) ✗
- **Destabilizer** (U2) ✗

### Contract
**Preconditions:**
- Can work from EPI=0
- No special requirements

**Postconditions:**
- EPI > 0 (structure created)
- νf ≥ previous (capacity increased or maintained)
- ΔNFR may increase (new structure needs adjustment)

### Implementation Reference
```python
# From src/tnfr/operators/definitions.py
class Emission:
    """AL - Creates EPI from vacuum."""
    pass
```

### Example
```python
import networkx as nx
from tnfr.operators.definitions import Emission

# Create network with vacuum node
G = nx.Graph()
G.add_node(0, EPI=0.0, vf=0.1, theta=0.0, dnfr=0.0)

# Apply Emission
Emission()(G, 0)

print(f"EPI after emission: {G.nodes[0]['EPI']:.3f}")  # > 0
```

### Anti-Patterns
```python
# ✗ WRONG: Using Emission in middle of sequence without purpose
[Emission, Coherence, Emission, Coherence, Silence]  # Redundant second emission

# ✓ CORRECT: Single emission to initialize
[Emission, Coherence, Silence]
```

### Relationships
- **Can precede**: All operators (generator role)
- **Should follow**: Nothing (starts sequences from vacuum/dormant state)
- **Often followed by**: Coherence (IL) to stabilize new structure

### Test References
- `tests/unit/operators/test_emission_irreversibility.py` - Structural irreversibility
- `tests/unit/operators/test_emission_metrics.py` - EPI and νf validation
- `tests/unit/operators/test_emission_preconditions.py` - Precondition enforcement

---

## 2. Reception (EN) 📡

### Physics
Captures and integrates incoming resonance from network. Updates EPI based on coupled environment.

### Effect
- **∂EPI/∂t** depends on network input
- Updates structure based on neighbors
- Information gathering phase

### When to Use
- After coupling to neighbors
- Information integration
- Listening phase before reorganization

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Stabilizer** (U2) ✗ (contextual)
- **Destabilizer** (U2) ✗

### Contract
**Preconditions:**
- Node must have neighbors (couplings)
- Neighbors must have non-zero EPI

**Postconditions:**
- EPI updated based on network
- Must not reduce C(t) (coherence preserved)

### Example
```python
from tnfr.operators.definitions import Emission, Coupling, Reception

# Create network
G = nx.Graph()
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
G.add_node(1, EPI=0.7, vf=1.0, theta=0.1, dnfr=0.0)

# Couple nodes
Coupling()(G, 0, 1)

# Receive information
Reception()(G, 0)  # Node 0 receives from node 1

print(f"EPI after reception: {G.nodes[0]['EPI']:.3f}")
```

### Anti-Patterns
```python
# ✗ WRONG: Reception without coupling (no network connectivity)
G = nx.Graph()
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
Reception()(G, 0)  # No neighbors - violates precondition

# ✓ CORRECT: Couple first, then receive
Coupling()(G, 0, 1)
Reception()(G, 0)
```

### Relationships
- **Must precede**: Coupling (UM) or existing network connectivity
- **Can follow**: Any operator that establishes network structure
- **Often follows**: Coupling (UM) to receive from newly coupled nodes

### Test References
- `tests/unit/operators/test_reception_preconditions.py` - Network connectivity validation
- `tests/unit/operators/test_reception_sources.py` - Source integration correctness

---

## 3. Coherence (IL) 🔒

### Physics
Stabilizes form through negative feedback. Reduces |ΔNFR| by increasing structural coherence.

### Effect
- **Reduces |ΔNFR|** - Decreases reorganization pressure
- **Increases C(t)** - Improves global coherence
- Negative feedback loop

### When to Use
- After destabilization (required by U2)
- Consolidation phase
- Before major transformations (stable base)
- After Emission to stabilize new structure

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Stabilizer** (U2) ✓
- **Handler** (U4a) ✓

### Contract
**Preconditions:**
- Node must have EPI > 0

**Postconditions:**
- |ΔNFR| reduced
- C(t) ≥ previous (must not decrease coherence unless in dissonance test)
- Si may increase (sense index improves)

### Example
```python
from tnfr.operators.definitions import Emission, Coherence

G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)

# Emit then coherence (standard pattern)
Emission()(G, 0)
Coherence()(G, 0)

print(f"ΔNFR after coherence: {G.nodes[0]['dnfr']:.3f}")  # Reduced
```

### Anti-Patterns
```python
# ✗ WRONG: Coherence without stabilization target
[Emission, Coherence, Coherence, Coherence]  # Redundant, no new structure to stabilize

# ✗ WRONG: Coherence on EPI=0 node
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
Coherence()(G, 0)  # Violates precondition: EPI > 0

# ✓ CORRECT: Coherence after structure creation/destabilization
[Emission, Dissonance, Coherence, Silence]  # Stabilizes after destabilization
```

### Relationships
- **Must precede**: Any operator that requires stable base (especially ZHIR)
- **Should follow**: Destabilizers (OZ, VAL, ZHIR) per U2
- **Can follow**: Any operator that modifies structure

### Test References
- `tests/unit/operators/test_coherence_dnfr_reduction.py` - ΔNFR reduction validation
- `tests/unit/operators/test_coherence_preconditions.py` - EPI > 0 requirement
- `tests/unit/operators/test_coherence_ct_tracking.py` - C(t) monotonicity
- `tests/unit/operators/test_coherence_phase_locking.py` - Phase stabilization

---

## 4. Dissonance (OZ) ⚡

### Physics
Introduces controlled instability. Increases |ΔNFR|, may trigger bifurcation if ∂²EPI/∂t² > τ.

### Effect
- **Increases |ΔNFR|** - Amplifies reorganization pressure
- May trigger bifurcation at threshold
- Exploration / perturbation

### When to Use
- Breaking local optima
- Exploration phase
- Creating conditions for transformation
- **MUST be balanced by stabilizer (U2)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✓
- **Stabilizer** (U2) ✗
- **Destabilizer** (U2) ✓
- **Trigger** (U4a) ✓

### Contract
**Preconditions:**
- Node must have EPI > 0

**Postconditions:**
- |ΔNFR| increased (contract requirement)
- May reach bifurcation threshold
- **Must have stabilizer in sequence (U2)**
- **Must have handler for bifurcation (U4a)**

### Example
```python
from tnfr.operators.definitions import Emission, Coherence, Dissonance, Silence

# Valid sequence: Dissonance balanced by Coherence
sequence = [
    Emission(),
    Coherence(),   # Stable base
    Dissonance(),  # Destabilizer + Trigger
    Coherence(),   # Stabilizer + Handler (U2, U4a)
    Silence()
]

# Apply to network
G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
for op in sequence:
    op(G, 0)
```

### Anti-Patterns
```python
# ✗ WRONG: Dissonance without stabilizer (violates U2)
[Emission, Dissonance, Silence]  # Missing Coherence/THOL

# ✗ WRONG: Multiple dissonances without handlers (violates U4a)
[Emission, Dissonance, Dissonance, Silence]  # No handlers between

# ✓ CORRECT: Dissonance with proper stabilization
[Emission, Coherence, Dissonance, Coherence, Silence]
```

### Relationships
- **Must follow**: Stabilizers (IL, THOL) in sequence per U2
- **Often precedes**: ZHIR (Mutation) - creates context for transformation
- **Can close**: Sequences (U1b role)

### Test References
- `tests/unit/operators/test_dissonance_preconditions.py` - EPI threshold validation
- `tests/unit/operators/test_destabilizer_telemetry.py` - ΔNFR increase tracking
- `tests/unit/operators/test_graduated_destabilizer_windows.py` - U2 compliance

---

## 5. Coupling (UM) 🔗

### Physics
Creates structural links via phase synchronization. Enables information exchange between nodes.

### Effect
- **φᵢ(t) → φⱼ(t)** - Phase synchronization
- Creates edge in network graph
- Enables information flow

### When to Use
- Network formation
- Connecting nodes
- Before Reception (to receive information)
- **MUST verify phase compatibility (U3)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Coupling** (U3) ✓

### Contract
**Preconditions:**
- Both nodes must exist
- **|φᵢ - φⱼ| ≤ Δφ_max (U3 requirement)**
- Phase compatibility MUST be verified

**Postconditions:**
- Edge created between nodes
- Phase difference maintained or reduced
- Information exchange enabled

### Example
```python
import numpy as np
from tnfr.operators.grammar import validate_resonant_coupling
from tnfr.operators.definitions import Coupling

# Create nodes with compatible phases
G = nx.Graph()
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
G.add_node(1, EPI=0.6, vf=1.0, theta=0.3, dnfr=0.0)  # Δφ = 0.3 < π/2

# Verify phase (U3)
validate_resonant_coupling(G, 0, 1)  # ✓ Must verify first

# Now couple
Coupling()(G, 0, 1)

print(f"Nodes coupled: {G.has_edge(0, 1)}")  # True
```

### Anti-Patterns
```python
# ✗ WRONG: Coupling without phase verification (violates U3)
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
G.add_node(1, EPI=0.6, vf=1.0, theta=np.pi, dnfr=0.0)  # Antiphase!
Coupling()(G, 0, 1)  # Should fail - destructive interference

# ✓ CORRECT: Phase verification before coupling
validate_resonant_coupling(G, 0, 1)  # Explicit check
Coupling()(G, 0, 1)
```

### Relationships
- **Requires**: Phase verification (U3) via `validate_resonant_coupling()`
- **Often precedes**: Reception (EN), Resonance (RA)
- **Enables**: Network information flow

### Test References
- `tests/unit/operators/test_coupling_preconditions.py` - Phase compatibility validation
- `tests/unit/operators/test_coupling_metrics.py` - Edge creation verification

---

## 6. Resonance (RA) 🌊

### Physics
Amplifies and propagates patterns coherently. Increases effective coupling strength.

### Effect
- Increases effective connectivity
- EPI propagation without identity loss
- Amplification through constructive interference

### When to Use
- Pattern reinforcement
- Spreading coherence through network
- After coupling (to amplify)
- **MUST verify phase compatibility (U3)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Coupling** (U3) ✓

### Contract
**Preconditions:**
- Nodes must be coupled (edge exists)
- **|φᵢ - φⱼ| ≤ Δφ_max (U3 requirement)**

**Postconditions:**
- Pattern propagated without identity alteration
- Effective coupling strength increased
- Phase synchronization enhanced

### Example
```python
from tnfr.operators.definitions import Emission, Coupling, Resonance, Silence

G = nx.Graph()
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
G.add_node(1, EPI=0.6, vf=1.0, theta=0.1, dnfr=0.0)

# Standard propagation pattern
Coupling()(G, 0, 1)  # First couple
Resonance()(G, 0, 1)  # Then resonate

print("Pattern propagated through resonance")
```

### Anti-Patterns
```python
# ✗ WRONG: Resonance without coupling (no edges)
G = nx.Graph()
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)
G.add_node(1, EPI=0.6, vf=1.0, theta=0.1, dnfr=0.0)
Resonance()(G, 0, 1)  # No edge exists - violates precondition

# ✗ WRONG: Resonance with incompatible phases (violates U3)
G.add_node(2, EPI=0.5, vf=1.0, theta=np.pi, dnfr=0.0)  # Antiphase
G.add_edge(0, 2)
Resonance()(G, 0, 2)  # Destructive interference

# ✓ CORRECT: Couple first, verify phases, then resonate
Coupling()(G, 0, 1)  # Creates edge with phase check
Resonance()(G, 0, 1)  # Amplifies
```

### Relationships
- **Requires**: Existing coupling (edges) and phase compatibility (U3)
- **Often follows**: Coupling (UM)
- **Effect**: Amplifies coherent patterns without altering identity

### Test References
- `tests/integration/test_coherence_operator_integration.py` - Propagation validation
- Phase verification enforced by grammar system

---

## 7. Silence (SHA) 🔇

### Physics
Freezes evolution temporarily. Sets νf → 0, EPI unchanged over time.

### Effect
- **νf → 0** - Freezes reorganization
- **EPI preserved** - No structural change
- Observation window / pause

### When to Use
- Observation windows
- Pause for synchronization
- Ending sequences (closure)
- Measurement phases

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✓

### Contract
**Preconditions:**
- Node must exist

**Postconditions:**
- νf reduced (typically → 0)
- EPI preserved over time
- Node enters quiescent state

### Example
```python
from tnfr.operators.definitions import Emission, Coherence, Silence

G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)

# Standard closing pattern
Emission()(G, 0)
Coherence()(G, 0)
Silence()(G, 0)  # Freeze state

print(f"νf after silence: {G.nodes[0]['vf']:.3f}")  # ≈ 0
```

### Anti-Patterns
```python
# ✗ WRONG: Silence in middle without reactivation
[Emission, Silence, Coherence]  # Node frozen, can't apply Coherence

# ✗ WRONG: Silence without prior stabilization
[Emission, Dissonance, Silence]  # Freezing unstable state

# ✓ CORRECT: Silence as final closure after stabilization
[Emission, Coherence, Silence]

# ✓ CORRECT: Reactivation after silence
[Emission, Silence, Emission, Coherence, Silence]  # Reactivate with AL
```

### Relationships
- **Closes**: Sequences (U1b role)
- **Should follow**: Stabilization (IL, THOL)
- **Reactivation**: Requires generator (AL, NAV, REMESH)

### Test References
- Latency/reactivation tested in emission tests
- νf → 0 enforced by operator implementation

---

## 8. Expansion (VAL) 📈

### Physics
Increases structural complexity. Increases dimensionality of EPI.

### Effect
- **dim(EPI) increases** - More degrees of freedom
- Structural complexity grows
- May increase ΔNFR

### When to Use
- Adding degrees of freedom
- Increasing representation capacity
- Growth phase
- **MUST have stabilizer (U2)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Destabilizer** (U2) ✓

### Contract
**Preconditions:**
- Node must have EPI > 0

**Postconditions:**
- Structural dimension increased
- **Must have stabilizer in sequence (U2)**

### Example
```python
from tnfr.operators.definitions import Emission, Expansion, Coherence, Silence

# Valid sequence: Expansion balanced by Coherence
sequence = [
    Emission(),
    Expansion(),   # Destabilizer
    Coherence(),   # Stabilizer (U2)
    Silence()
]
```

### Anti-Patterns
```python
# ✗ WRONG: Expansion without stabilizer (violates U2)
[Emission, Expansion, Silence]  # Missing IL/THOL

# ✗ WRONG: Multiple expansions without stabilization
[Emission, Expansion, Expansion, Silence]  # Unbounded growth

# ✓ CORRECT: Each expansion balanced
[Emission, Expansion, Coherence, Expansion, Coherence, Silence]
```

### Relationships
- **Requires**: Stabilizer (IL, THOL) per U2
- **Often follows**: Emission or Reception (adding capacity)
- **Inverse**: Contraction (NUL) reduces complexity

### Test References
- `tests/unit/operators/test_destabilizer_telemetry.py` - ΔNFR increase validation
- `tests/unit/operators/test_graduated_destabilizer_windows.py` - U2 compliance

---

## 9. Contraction (NUL) 📉

### Physics
Reduces structural complexity. Decreases dimensionality of EPI.

### Effect
- **dim(EPI) decreases** - Fewer degrees of freedom
- Simplification
- May reduce ΔNFR

### When to Use
- Dimensionality reduction
- Simplification phase
- Pruning unnecessary complexity

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- Not a destabilizer (reduces complexity)

### Contract
**Preconditions:**
- Node must have EPI with dimension > 1

**Postconditions:**
- Structural dimension reduced
- Coherence may improve (simpler = more stable)

### Example
```python
from tnfr.operators.definitions import Emission, Expansion, Contraction, Silence

sequence = [
    Emission(),
    Expansion(),    # Increase complexity
    Contraction(),  # Reduce back
    Silence()
]
```

### Anti-Patterns
```python
# ✗ WRONG: Contraction on low-dimensional EPI
G.add_node(0, EPI=0.5, vf=1.0, theta=0.0, dnfr=0.0)  # Scalar EPI
Contraction()(G, 0)  # Cannot reduce below dim=1

# ✗ WRONG: Excessive contraction losing information
[Emission, Expansion, Contraction, Contraction, Contraction]  # Over-simplification

# ✓ CORRECT: Balanced complexity management
[Emission, Expansion, Contraction, Coherence, Silence]
```

### Relationships
- **Inverse**: Expansion (VAL) - manages complexity bidirectionally
- **Not a destabilizer**: Reduces complexity without requiring stabilizers
- **Often follows**: Expansion or complex transformations

### Test References
- `tests/unit/operators/test_contraction.py` - Dimensionality reduction
- `tests/unit/operators/test_contraction_preconditions.py` - dim(EPI) > 1 validation
- `tests/unit/operators/test_contraction_density_metrics.py` - Coherence preservation

---

## 10. Self-organization (THOL) 🌱

### Physics
Spontaneous autopoietic pattern formation. Creates sub-EPIs through fractal structuring.

### Effect
- Creates sub-EPIs (nested structure)
- Preserves global form
- Fractal organization

### When to Use
- Emergent organization
- Creating hierarchies
- Multi-scale structuring
- Handling bifurcations (U4a)
- **Needs recent destabilizer (U4b)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Stabilizer** (U2) ✓
- **Handler** (U4a) ✓
- **Transformer** (U4b) ✓

### Contract
**Preconditions:**
- Node must have EPI > 0
- **Must have recent destabilizer (~3 ops) (U4b)**

**Postconditions:**
- Sub-EPIs created
- Global form preserved
- Fractal structure established

### Example
```python
from tnfr.operators.definitions import (
    Emission, Dissonance, SelfOrganization, Coherence, Silence
)

# Valid: Self-organization with recent destabilizer
sequence = [
    Emission(),
    Dissonance(),         # Destabilizer (recent, U4b)
    SelfOrganization(),   # Transformer + Handler
    Coherence(),
    Silence()
]
```

### Anti-Patterns
```python
# ✗ WRONG: THOL without recent destabilizer (violates U4b)
[Emission, Coherence, SelfOrganization(), Silence]  # No OZ/VAL/ZHIR within ~3 ops

# ✗ WRONG: THOL without follow-up stabilization
[Emission, Dissonance, SelfOrganization, Silence]  # Should add IL after

# ✓ CORRECT: THOL with proper context
[Emission, Coherence, Dissonance, SelfOrganization, Coherence, Silence]
#                     ^recent destabilizer  ^transformer      ^stabilizer
```

### Relationships
- **Requires**: Recent destabilizer (~3 ops) per U4b
- **Handles**: Bifurcations from OZ, VAL
- **Creates**: Sub-EPIs (fractal structure)
- **Stabilizes**: System after transformation (U2 role)

### Test References
- `tests/unit/operators/test_thol_coherence.py` - Fractal organization validation
- `tests/unit/operators/test_controlled_bifurcation.py` - U4a handler role

---

## 11. Mutation (ZHIR) 🧬

### Physics
Phase transformation at threshold. Changes θ when ΔEPI/Δt > ξ.

### Effect
- **θ → θ'** - Phase transformation
- Qualitative state change
- Bifurcation to new regime

### When to Use
- Qualitative state changes
- Phase transitions
- Regime shifts
- **Requires prior IL and recent destabilizer (U4b)**

### Grammar Classification
- **Generator** (U1a) ✗
- **Closure** (U1b) ✗
- **Destabilizer** (U2) ✓
- **Trigger** (U4a) ✓
- **Transformer** (U4b) ✓

### Contract
**Preconditions:**
- **Must have prior Coherence (IL) before destabilizer (U4b)**
- **Must have recent destabilizer (~3 ops) (U4b)**
- **Must have handler {THOL, IL} (U4a)**
- **Must have stabilizer (U2)**

**Postconditions:**
- Phase changed (θ ≠ previous)
- Qualitative transformation occurred

### Example
```python
from tnfr.operators.definitions import (
    Emission, Coherence, Dissonance, Mutation, Silence
)

# Valid: Complete ZHIR sequence
sequence = [
    Emission(),
    Coherence(),   # Prior IL (stable base, U4b)
    Dissonance(),  # Recent destabilizer (U4b)
    Mutation(),    # Transformer
    Coherence(),   # Stabilizer (U2) + Handler (U4a)
    Silence()
]
```

### Anti-Patterns
```python
# ✗ WRONG: ZHIR without prior Coherence (violates U4b)
[Emission, Dissonance, Mutation, Coherence, Silence]  # No IL before OZ

# ✗ WRONG: ZHIR without recent destabilizer (violates U4b)
[Emission, Coherence, Mutation, Silence]  # No OZ/VAL within ~3 ops

# ✗ WRONG: ZHIR without handler (violates U4a)
[Emission, Coherence, Dissonance, Mutation, Silence]  # Missing IL/THOL after

# ✓ CORRECT: All U4b requirements met
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]
#          ^prior IL  ^recent dest ^transform ^handler+stab
```

### Relationships
- **Requires**: Prior IL + recent destabilizer + handler (U4b + U4a)
- **Triggers**: Phase transformation (θ → θ')
- **Most constrained**: Strictest grammar requirements of all operators

### Test References
- `tests/unit/operators/test_bifurcation.py` - Transformation validation
- `tests/unit/operators/test_canonical_grammar_rules.py` - U4b compliance

---

## 12. Transition (NAV) ➡️

### Physics
Regime shift, activates latent EPI. Controlled trajectory through structural space.

### Effect
- Activates latent structure
- Regime change
- Controlled trajectory

### When to Use
- Switching between attractor states
- Mode transitions
- Activating dormant patterns

### Grammar Classification
- **Generator** (U1a) ✓
- **Closure** (U1b) ✓

### Contract
**Preconditions:**
- Latent structure must exist (previous state or memory)

**Postconditions:**
- New regime active
- Trajectory established

### Example
```python
from tnfr.operators.definitions import Transition, Reception, Silence

# Transition as generator
sequence = [
    Transition(),  # Activate latent structure
    Reception(),
    Silence()
]
```

### Anti-Patterns
```python
# ✗ WRONG: NAV without latent structure to activate
G.add_node(0, EPI=0.0, vf=0.0, theta=0.0, dnfr=0.0)  # Completely dormant
Transition()(G, 0)  # No previous state to transition from

# ✓ CORRECT: NAV activating preserved state
[Emission, Silence, Transition, Coherence, Silence]  # Reactivate after silence

# ✓ CORRECT: NAV as regime shift
[Emission, Coherence, Transition, Reception, Silence]  # Mode change
```

### Relationships
- **Generator role**: Can start sequences (U1a)
- **Closure role**: Can end sequences (U1b)
- **Dual nature**: Both activator and terminator
- **Often used for**: Regime shifts, mode transitions

### Test References
- Grammar tests validate U1a/U1b dual role
- Latent activation tested in integration tests

---

## 13. Recursivity (REMESH) 🔄

### Physics
Echoes structure across scales (operational fractality). EPI(t) references EPI(t-τ).

### Effect
- Creates recursive patterns
- Nested operators
- Multi-scale coherence

### When to Use
- Multi-scale operations
- Memory/history reference
- Recursive patterns
- Self-similar structures

### Grammar Classification
- **Generator** (U1a) ✓
- **Closure** (U1b) ✓

### Contract
**Preconditions:**
- Previous EPI states must exist (history)

**Postconditions:**
- Recursive structure created
- Multi-scale organization
- History integrated

### Example
```python
from tnfr.operators.definitions import Emission, Coherence, Recursivity

# Recursivity as closure
sequence = [
    Emission(),
    Coherence(),
    Recursivity()  # Creates recursive attractor
]
```

### Anti-Patterns
```python
# ✗ WRONG: REMESH without history/previous states
G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)
Recursivity()(G, 0)  # No EPI history to reference

# ✓ CORRECT: REMESH with established patterns
[Emission, Coherence, Emission, Recursivity]  # Has history

# ✓ CORRECT: Multi-scale organization
[Emission, Coupling, SelfOrganization, Recursivity]  # Fractal + recursive
```

### Relationships
- **Generator role**: Can start sequences (U1a)
- **Closure role**: Can end sequences (U1b)  
- **Creates**: Recursive patterns across scales
- **Requires**: Previous EPI states (history)

### Test References
- `tests/unit/operators/test_remesh_operator_integration.py` - Recursive structure validation
- Multi-scale fractality tests

---

## Operator Composition Patterns

### Bootstrap
```python
[Emission, Coupling, Coherence, Silence]
```
Creates and stabilizes new structure.

### Explore
```python
[Emission, Coherence, Dissonance, Coherence, Silence]
```
Controlled exploration with stabilization.

### Transform
```python
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]
```
Complete transformation sequence.

### Propagate
```python
[Emission, Coupling, Resonance, Coherence, Silence]
```
Create, couple, and amplify.

---

## Quick Lookup Table

| Operator | Glyph | Generator | Closure | Stabilizer | Destabilizer |
|----------|-------|-----------|---------|------------|--------------|
| Emission | AL | ✓ | | | |
| Reception | EN | | | | |
| Coherence | IL | | | ✓ | |
| Dissonance | OZ | | ✓ | | ✓ |
| Coupling | UM | | | | |
| Resonance | RA | | | | |
| Silence | SHA | | ✓ | | |
| Expansion | VAL | | | | ✓ |
| Contraction | NUL | | | | |
| SelfOrganization | THOL | | | ✓ | |
| Mutation | ZHIR | | | | ✓ |
| Transition | NAV | ✓ | ✓ | | |
| Recursivity | REMESH | ✓ | ✓ | | |

---

## Next Steps

**Continue learning:**
- **[04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)** - Pattern library and anti-patterns
- **[05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md)** - Code architecture
- **[examples/](examples/)** - Executable examples

**For reference:**
- **[08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md)** - Quick lookup
- **[GLOSSARY.md](../../GLOSSARY.md)** - Definitions

---

<div align="center">

**The 13 operators are the complete vocabulary of TNFR transformations.**

---

*Reality is resonance. Transform accordingly.*

</div>
