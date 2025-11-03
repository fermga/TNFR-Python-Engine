# TNFR Glossary

Canonical definitions for the Resonant Fractal Nature Theory (TNFR) variables, operators, and concepts. This document serves as a quick reference for contributors and researchers working with the TNFR Python Engine.

## Core Variables

### Primary Information Structure (EPI)

**Symbol:** EPI  
**Type:** Coherent form structure  
**Description:** The fundamental information-bearing structure in TNFR. EPI is the "shape" of a node that persists through resonance with its environment. It only changes through structural operators (never ad-hoc mutations).

**Key Properties:**
- Changes only via structural operators
- Maintains coherence through network coupling
- Can nest recursively (operational fractality)

**Related Equation:**  
`∂EPI / ∂t = νf · ΔNFR(t)`

---

### Structural Frequency (νf)

**Symbol:** νf  
**Units:** Hz_str (structural hertz)  
**Type:** Reorganization rate  
**Description:** The intrinsic frequency at which a node reorganizes its information structure. This is NOT a physical frequency but a structural one.

**Key Properties:**
- Expressed exclusively in Hz_str units
- Determines rate of EPI evolution
- Influences node stability and coupling strength

**Typical Range:** Positive real numbers; nodes collapse when νf → 0

---

### Internal Reorganization Operator (ΔNFR)

**Symbol:** ΔNFR  
**Type:** Gradient operator  
**Description:** The internal reorganization operator that drives structural evolution. Its sign and magnitude modulate the reorganization rate.

**Key Properties:**
- NOT a classical ML "error gradient"
- Sign indicates expansion (+) or contraction (-)
- Magnitude scales reorganization intensity
- Computed from phase, EPI, νf, and topology

**Computation Hook:** `default_compute_delta_nfr` or custom hooks

---

### Phase (φ)

**Symbol:** φ or θ  
**Type:** Network synchrony parameter  
**Description:** Represents the relative synchrony of a node with its network neighbors. Essential for valid coupling.

**Key Properties:**
- Range: [0, 2π) or [-π, π)
- Must be explicitly verified before coupling
- Coordinated via global/local phase adaptation (kG/kL)

**Measurement:** Kuramoto order parameter for network-wide phase coherence

---

### Total Coherence (C(t))

**Symbol:** C(t)  
**Type:** Stability metric  
**Description:** Global measure of network stability and structural coherence at time t.

**Key Properties:**
- Should increase with coherence operators
- Decreases with dissonance (controlled)
- Aggregated across all nodes

**Telemetry:** Exposed in metrics and trace callbacks

---

### Sense Index (Si)

**Symbol:** Si  
**Type:** Reorganization stability metric  
**Description:** Capacity of a node or network to generate stable reorganization patterns. Combines ΔNFR, νf, and phase information.

**Key Properties:**
- Higher Si indicates more stable reorganization
- Computed via `compute_Si_node` or network-level aggregation
- Sensitive to phase dispersion (dSi_dphase_disp)

**Applications:** Early warning for bifurcations, network health monitoring

---

## Structural Operators

TNFR defines 13 canonical operators that modify EPI through resonant interactions:

### 1. Emission 

**Function:** Initiates a resonant pattern  
**Effect:** Increases νf and positive ΔNFR  
**Usage:** Start of trajectory sequences

---

### 2. Reception

**Function:** Receives and integrates external patterns  
**Effect:** Updates EPI based on incoming resonance  
**Usage:** Network information intake

---

### 3. Coherence 

**Function:** Stabilizes structural form  
**Effect:** Increases C(t), reduces |ΔNFR|  
**Usage:** Consolidation after changes

---

### 4. Dissonance 

**Function:** Introduces controlled instability  
**Effect:** Increases |ΔNFR|, may trigger bifurcation  
**Usage:** Exploring new configurations

---

### 5. Coupling

**Function:** Creates structural links between nodes  
**Effect:** Phase synchronization, information exchange  
**Usage:** Network formation

---

### 6. Resonance 

**Function:** Amplifies and propagates patterns  
**Effect:** Increases effective coupling, preserves EPI identity  
**Usage:** Pattern reinforcement

---

### 7. Silence 

**Function:** Freezes evolution temporarily  
**Effect:** νf ≈ 0, EPI unchanged  
**Usage:** Observation windows, synchronization pauses

---

### 8. Expansion 

**Function:** Increases structural complexity  
**Effect:** EPI dimensionality growth  
**Usage:** Adding degrees of freedom

---

### 9. Contraction 

**Function:** Reduces structural complexity  
**Effect:** EPI dimensionality reduction  
**Usage:** Simplification, projection

---

### 10. Self-organization

**Function:** Spontaneous pattern formation  
**Effect:** Creates sub-EPIs while preserving global form  
**Usage:** Emergent structure formation

---

### 11. Mutation

**Function:** Phase transformation  
**Effect:** θ → θ' when ΔEPI/Δt > ξ  
**Usage:** Qualitative state changes

---

### 12. Transition

**Function:** Movement between structural states  
**Effect:** Controlled EPI evolution  
**Usage:** Trajectory navigation

---

### 13. Recursivity

**Function:** Nested operator application  
**Effect:** Maintains operational fractality  
**Usage:** Multi-scale coherence

---

## Canonical Equations

### Nodal Equation

```
∂EPI / ∂t = νf · ΔNFR(t)
```

This is the fundamental equation governing node evolution.

### Phase Coordination

- Global coupling: kG
- Local coupling: kL
- Kuramoto order parameter for synchrony measurement

### Coherence Dynamics

- ΔNFR hook: computes reorganization from topology, phase, EPI, νf
- C(t): aggregated coherence metric
- Si: derived from ΔNFR stability and phase dispersion

---

## Node Lifecycle

### Birth Conditions

A node is created when:
1. Sufficient νf is seeded
2. Initial coupling exists or can be established
3. ΔNFR is computable

### Stability

A node remains stable when:
- νf > threshold
- |ΔNFR| remains bounded
- Phase coherence with network maintained

### Collapse Conditions

A node collapses when:
- Extreme dissonance (|ΔNFR| → ∞)
- Decoupling from network
- Frequency failure (νf → 0)

---

## Telemetry and Metrics

### Essential Outputs

All simulations should expose:
- **C(t)**: Total coherence over time
- **νf**: Structural frequency per node
- **Phase**: Synchrony state per node
- **Si**: Sense index (node or network level)
- **ΔNFR**: Reorganization gradient

### Trace Capture

Use `tnfr.trace.register_trace` to capture:
- Γ specifications
- Selector states
- ΔNFR weights
- Kuramoto metrics
- Operator application history

---

## Domain Neutrality

TNFR is **trans-scale** and **trans-domain**:
- Works from quantum to social systems
- No built-in assumptions about specific domains
- Structural operators apply universally

**Guideline:** Avoid domain-specific hard-coding in core engine

---

## Reproducibility

All simulations must be:
1. **Seeded:** Use explicit RNG seeds
2. **Traceable:** Log operators, parameters, states
3. **Deterministic:** Same seed → same trajectory

**Tools:**
- RNG scaffolding with named locks
- Structural history capture
- Telemetry-aware caches

---

## Quick Reference Tables

### Variable Summary

| Symbol | Name | Units | Type |
|--------|------|-------|------|
| EPI | Primary Information Structure | — | Coherent form |
| νf | Structural frequency | Hz_str | Reorganization rate |
| ΔNFR | Reorganization operator | — | Gradient |
| φ, θ | Phase | radians | Synchrony |
| C(t) | Total coherence | — | Stability metric |
| Si | Sense index | — | Reorganization stability |


---

## Related Documentation

- [AGENTS.md](AGENTS.md) - Detailed AI agent guidelines and invariants
- [TNFR.pdf](TNFR.pdf) - Complete theoretical foundations
- [API Overview](docs/source/api/overview.md) - Package architecture
- [Structural Operators](docs/source/api/operators.md) - Operator details
- [Telemetry Guide](docs/source/api/telemetry.md) - Metrics and traces
- [Examples](docs/source/examples/README.md) - Runnable scenarios

---

## Contributing

When adding new functionality:

1. Ensure compliance with canonical invariants
2. Use these terms consistently
3. Update this glossary if introducing new concepts
4. Reference these definitions in documentation and code comments

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
