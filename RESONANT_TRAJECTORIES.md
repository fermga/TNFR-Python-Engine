# Resonant Trajectories Reference

Canonical examples of structural operator sequences (trajectories) for common TNFR scenarios. This guide helps new contributors understand how to compose operators to achieve specific coherence goals.

## What is a Resonant Trajectory?

A **resonant trajectory** is a sequence of structural operators that guides a TNFR network through a specific evolution path. Each trajectory:

- Respects operator closure (valid compositions only)
- Maintains structural invariants
- Has predictable effects on C(t), Si, and ΔNFR
- Can be composed with other trajectories

## Basic Trajectories

### 1. Resonant Bootstrap

**Purpose:** Initialize a network with stable baseline coherence  
**Operators:** `EN → IL → UM`  
**Token Sequence:** `["EN", "IL", "UM"]`

**Effect:**
1. **EN (Reception):** Nodes prepare to receive initial patterns
2. **IL (Coherence):** Stabilize initial EPI structures
3. **UM (Coupling):** Establish network connections

**Expected Metrics:**
- C(t): Monotonic increase
- ΔNFR: Converges to small positive values
- Si: Increases as network stabilizes
- Phase: Initial synchronization begins

**Example:**
```python
from tnfr.structural import run_sequence

trajectory = ["EN", "IL", "UM"]
graph = run_sequence(graph, trajectory, config)
```

**Use Cases:**
- Starting new simulations
- Baseline establishment
- Network initialization

---

### 2. Controlled Dissonance with Re-coherence

**Purpose:** Introduce instability, explore alternatives, then restabilize  
**Operators:** `OZ → IL → RA`  
**Token Sequence:** `["OZ", "IL", "RA"]`

**Effect:**
1. **OZ (Dissonance):** Creates controlled instability, increases |ΔNFR|
2. **IL (Coherence):** Restabilizes new configuration
3. **RA (Resonance):** Amplifies and propagates new pattern

**Expected Metrics:**
- C(t): Dips during OZ, recovers during IL, stabilizes with RA
- ΔNFR: Spikes with OZ, reduces with IL
- Si: May drop temporarily, recovers
- Phase: May desynchronize, then resynchronize

**Example:**
```python
trajectory = ["OZ", "IL", "RA"]
graph = run_sequence(graph, trajectory, config)
```

**Use Cases:**
- Escaping local minima
- Exploring configuration space
- Adaptive optimization

---

### 3. Coupling Exploration

**Purpose:** Systematically explore network connectivity patterns  
**Operators:** `EN → UM → RA → IL`  
**Token Sequence:** `["EN", "UM", "RA", "IL"]`

**Effect:**
1. **EN (Reception):** Nodes open to new connections
2. **UM (Coupling):** Form new links
3. **RA (Resonance):** Strengthen promising connections
4. **IL (Coherence):** Consolidate network structure

**Expected Metrics:**
- C(t): Gradual increase
- Network density: Increases
- Phase coherence: Improves
- Si: Stabilizes at higher level

**Example:**
```python
trajectory = ["EN", "UM", "RA", "IL"]
graph = run_sequence(graph, trajectory, config)
```

**Use Cases:**
- Network growth
- Discovering optimal topologies
- Community detection

---

### 4. Contained Mutation

**Purpose:** Transform node states while maintaining network stability  
**Operators:** `SHA → ZHIR → IL → RA`  
**Token Sequence:** `["SHA", "ZHIR", "IL", "RA"]`

**Effect:**
1. **SHA (Silence):** Pause evolution for controlled mutation
2. **ZHIR (Mutation):** Transform phase/state (θ → θ')
3. **IL (Coherence):** Stabilize new state
4. **RA (Resonance):** Integrate mutation into network

**Expected Metrics:**
- C(t): Stable during SHA, may dip during ZHIR, recovers
- Phase: Shifts during ZHIR
- ΔNFR: Controlled throughout
- Si: Recovers after mutation

**Example:**
```python
trajectory = ["SHA", "ZHIR", "IL", "RA"]
graph = run_sequence(graph, trajectory, config)
```

**Use Cases:**
- Qualitative state changes
- Phase transitions
- Controlled perturbations

---

## Advanced Trajectories

### 5. Self-Organization Cascade

**Purpose:** Enable emergent sub-structure formation  
**Operators:** `EN → OZ → IL → UM → RA → IL`  
**Token Sequence:** `["EN", "OZ", "IL", "UM", "RA", "IL"]`

**Effect:**
1. **EN:** Prepare for reorganization
2. **OZ:** Create instability for reorganization
3. **IL:** Allow self-organization to stabilize
4. **UM:** Form new sub-network structures
5. **RA:** Amplify emergent patterns
6. **IL:** Final consolidation

**Expected Metrics:**
- Nested EPI structures emerge
- Multi-scale coherence patterns
- Hierarchical phase relationships

**Use Cases:**
- Complex system formation
- Hierarchical organization
- Emergent behavior studies

---

### 6. Resonant Feedback Loop

**Purpose:** Iterative refinement through feedback  
**Operators:** `AL → RA → EN → IL` (repeated)  
**Token Sequence:** `["AL", "RA", "EN", "IL"] * n`

**Effect:**
1. **AL (Emission):** Initiate pattern
2. **RA (Resonance):** Propagate and amplify
3. **EN (Reception):** Receive feedback
4. **IL (Coherence):** Consolidate improvements

**Expected Metrics:**
- C(t): Monotonic improvement over iterations
- Pattern strength: Increases
- Network alignment: Improves

**Example:**
```python
# Three refinement iterations
trajectory = ["AL", "RA", "EN", "IL"] * 3
graph = run_sequence(graph, trajectory, config)
```

**Use Cases:**
- Iterative optimization
- Pattern learning
- Feedback control systems

---

### 7. Exploratory Bifurcation

**Purpose:** Explore multiple stable states through bifurcation  
**Operators:** `OZ → OZ → IL → NAV → IL`  
**Token Sequence:** `["OZ", "OZ", "IL", "NAV", "IL"]`

**Effect:**
1. **OZ × 2:** Strong dissonance to trigger bifurcation
2. **IL:** Stabilize one branch
3. **NAV:** Transition between states
4. **IL:** Consolidate chosen state

**Expected Metrics:**
- ∂²EPI/∂t² > τ (bifurcation threshold)
- Multiple stable attractors emerge
- Path-dependent outcomes

**Use Cases:**
- Multi-stability exploration
- Decision-making systems
- Phase transition studies

---

### 8. Recursive Coherence

**Purpose:** Multi-scale coherence establishment  
**Operators:** `[EN → IL → [UM → RA] → IL]` (nested)  
**Token Sequence:** Complex (requires THOL notation)

**Effect:**
- Establishes coherence at multiple scales simultaneously
- Preserves operational fractality
- Nested EPI structures maintain identity

**Note:** Requires THOL (Transfinite Higher Order Language) evaluation for proper nesting

**Use Cases:**
- Multi-scale systems
- Fractal organization
- Hierarchical coherence

---

## Trajectory Patterns

### Pattern: Stabilization

**Goal:** Maximize C(t), minimize |ΔNFR|  
**Template:** `[setup] → IL → RA → IL`

**Variations:**
- Fast: `IL → RA → IL`
- Robust: `EN → IL → RA → IL → SHA`

---

### Pattern: Exploration

**Goal:** Sample configuration space  
**Template:** `[current] → OZ → NAV → [test] → IL`

**Variations:**
- Conservative: Single OZ
- Aggressive: Multiple OZ
- Directed: Add specific NAV targets

---

### Pattern: Network Formation

**Goal:** Build connectivity  
**Template:** `EN → UM → [strengthen] → IL`

**Variations:**
- With resonance: `EN → UM → RA → IL`
- With feedback: `EN → UM → RA → EN → IL`

---

### Pattern: State Transition

**Goal:** Change qualitative state  
**Template:** `SHA → ZHIR → [stabilize]`

**Variations:**
- Fast: `ZHIR → IL`
- Controlled: `SHA → ZHIR → IL → RA`

---

## Composing Trajectories

Trajectories can be composed sequentially or nested:

### Sequential Composition

```python
# Bootstrap, then explore
trajectory_1 = ["EN", "IL", "UM"]  # Bootstrap
trajectory_2 = ["OZ", "IL", "RA"]  # Explore

full_trajectory = trajectory_1 + trajectory_2
graph = run_sequence(graph, full_trajectory, config)
```

### Iterative Composition

```python
# Repeated refinement
base_trajectory = ["AL", "RA", "EN", "IL"]
iterations = 5

full_trajectory = base_trajectory * iterations
graph = run_sequence(graph, full_trajectory, config)
```

### Conditional Composition

```python
# Adapt based on metrics
if compute_coherence(graph) < threshold:
    trajectory = ["OZ", "IL", "RA"]  # Reorganize
else:
    trajectory = ["IL", "RA"]  # Maintain

graph = run_sequence(graph, trajectory, config)
```

---

## Trajectory Design Guidelines

### 1. Start with Reception or Emission

Most trajectories should begin with:
- **EN (Reception):** When integrating external information
- **AL (Emission):** When initiating new patterns

### 2. End with Coherence

Stabilize final state with:
- **IL (Coherence):** Direct stabilization
- **RA → IL:** Amplification then stabilization

### 3. Control Dissonance

When using **OZ (Dissonance)**:
- Always follow with stabilization (IL)
- Monitor ΔNFR magnitude
- Check bifurcation conditions

### 4. Balance Operators

Effective trajectories balance:
- Exploration (OZ) vs. Exploitation (IL)
- Local (node-level) vs. Global (network-level)
- Fast changes vs. Careful stabilization

### 5. Respect Closure

Ensure operator sequences are valid:
- Use RECEPTION → COHERENCE patterns
- Close THOL blocks properly
- Validate before execution

---

## Measuring Trajectory Success

### Key Metrics

1. **C(t) trajectory:** Should show desired pattern (increase, controlled dip, etc.)
2. **ΔNFR evolution:** Should converge to small values
3. **Si stability:** Should remain or increase
4. **Phase coherence:** Should maintain or improve

### Success Criteria

**Stabilization trajectory:**
- C(t) increases monotonically
- |ΔNFR| → 0
- Phase variance decreases

**Exploration trajectory:**
- C(t) explores lower values, then recovers
- |ΔNFR| controlled spikes
- New configurations discovered

**Transition trajectory:**
- Clear phase change observed
- C(t) recovers to similar or higher level
- Si remains stable

---

## Example Scripts

### Complete Example: Optical Cavity Feedback

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.metrics import compute_coherence, compute_Si

# Initialize
graph = create_nfr(num_nodes=3, topology="ring", seed=42)

# Bootstrap
bootstrap_trajectory = ["EN", "IL", "UM"]
graph = run_sequence(graph, bootstrap_trajectory, config)
print(f"C(t) after bootstrap: {compute_coherence(graph)}")

# Feedback loop (thermal drift correction)
feedback_trajectory = ["AL", "RA", "EN", "IL"] * 3
graph = run_sequence(graph, feedback_trajectory, config)
print(f"C(t) after feedback: {compute_coherence(graph)}")

# Final stabilization
stabilization_trajectory = ["IL", "RA", "IL"]
graph = run_sequence(graph, stabilization_trajectory, config)
print(f"Final C(t): {compute_coherence(graph)}")
print(f"Final Si: {compute_Si(graph)}")
```

See [examples/](docs/source/examples/) for complete runnable scripts.

---

## CLI Usage

Execute trajectories via the TNFR CLI:

```bash
# Basic trajectory
tnfr sequence \
  --nodes 5 --topology ring --seed 42 \
  --sequence '["EN", "IL", "UM"]' \
  --save-history trajectory.json

# From file
tnfr sequence \
  --nodes 5 --topology ring --seed 42 \
  --sequence-file trajectory.json \
  --config config.json
```

---

## Troubleshooting

### Trajectory Not Converging

**Symptoms:** C(t) oscillates, ΔNFR doesn't reduce

**Solutions:**
- Add more IL (Coherence) operators
- Reduce OZ intensity
- Check phase synchronization
- Verify coupling strength

### Unexpected Bifurcation

**Symptoms:** Sudden qualitative change, |∂²EPI/∂t²| > τ

**Solutions:**
- Add SHA (Silence) before mutations
- Reduce dissonance intensity
- Check ZHIR conditions

### Loss of Phase Coherence

**Symptoms:** Kuramoto order parameter drops

**Solutions:**
- Add RA (Resonance) to strengthen coupling
- Use UM to re-establish connections
- Check kG/kL parameters

---

## Related Documentation

- [GLOSSARY.md](GLOSSARY.md) - TNFR terms and variables
- [Structural Operators](docs/source/api/operators.md) - Operator details
- [Examples](docs/source/examples/README.md) - Complete examples
- [API Overview](docs/source/api/overview.md) - Implementation details

---

## Contributing New Trajectories

When documenting a new trajectory pattern:

1. **Name:** Clear, descriptive name
2. **Purpose:** What coherence goal does it achieve?
3. **Operators:** Exact sequence with tokens
4. **Effect:** What happens at each step?
5. **Metrics:** Expected C(t), ΔNFR, Si, phase behavior
6. **Example:** Runnable code snippet
7. **Use Cases:** When to use this trajectory?

Submit via PR with:
- Code example in `docs/source/examples/`
- Tests validating the trajectory
- Documentation update

---

**Last Updated:** 2025-11  
**Maintainers:** TNFR Python Engine Team  
**License:** MIT
