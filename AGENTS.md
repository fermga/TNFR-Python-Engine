# TNFR: Resonant Fractal Nature Theory
## Theoretical Framework for Coherent Pattern Analysis

**Status**: Primary theoretical reference document  
**Version**: 9.7.0 (November 29, 2025)  
**Authority**: This repository contains the current implementation of TNFR theory  
**Repository**: https://github.com/fermga/TNFR-Python-Engine  
**PyPI Package**: https://pypi.org/project/tnfr/  
**Installation**: `pip install tnfr`

---

### Source Hierarchy

1. **Primary**: This repository (TNFR-Python-Engine) - Current implementation reference
2. **Historical**: TNFR.pdf - Foundational derivations and theoretical background  
3. **Distribution**: PyPI package - Stable releases for implementation

**Reference Principle**: The repository implementation serves as the authoritative source. TNFR.pdf provides historical context and mathematical derivations, while this codebase represents the current state of TNFR development.

---

## Executive Summary

TNFR (Resonant Fractal Nature Theory) constitutes a theoretical framework for understanding reality through coherent patterns that persist via resonance mechanisms.

Core Discovery: The Universal Tetrahedral Correspondence establishes an exact mapping between four fundamental mathematical constants (φ, γ, π, e) and four structural fields that characterize coherent systems.

Theoretical Foundation: The framework models reality as coherent dynamic patterns rather than discrete objects, where patterns exist through resonant coupling with their environment.

### Theoretical Contributions

**Mathematical Framework**:
- Universal Tetrahedral Correspondence: φ, γ, π, e ↔ Φ_s, |∇φ|, K_φ, ξ_C mapping
- Complex Field Unification: Ψ = K_φ + i·J_φ unifies geometry and transport
- Emergent Invariants: Energy density, topological charge, conservation laws
- Grammar Formalization: U1-U6 rules derived from physical principles

**Physics Formulation**:
- Nodal Equation: ∂EPI/∂t = νf · ΔNFR(t) as universal evolution law
- Structural Fields: Complete tetrad characterization of coherent systems  
- Operational Fractality: Multi-scale coherence with nested EPIs
- Phase-Gated Coupling: |φᵢ - φⱼ| ≤ Δφ_max resonance condition

**Computational Implementation**:
- Self-Optimizing Engine: Algorithmic structural optimization
- Software Development Kit: API for TNFR implementation  
- Experimental Validation: 2,400+ experiments across multiple topologies
- Distribution Platform: PyPI package with documentation

**Application Domains**:
- Number Theory: Resonance-based primality analysis
- Molecular Chemistry: Periodic table modeling via TNFR dynamics
- Network Science: Topology-coherence relationship analysis
- Collective Behavior: Leader-follower emergence modeling

### Documentation Structure

| **Category** | **Key Resources** |
|--------------|-------------------|
| **Theory** | [Universal Tetrahedral Correspondence](#universal-tetrahedral-correspondence) |
| **Physics** | [Nodal Equation & Structural Triad](#foundational-physics) |
| **Operators** | [13 Canonical Operators](#the-13-canonical-operators) |
| **Grammar** | [Unified Grammar U1-U6](#unified-grammar-u1-u6) |
| **Fields** | [Structural Field Tetrad](#telemetry--structural-field-tetrad) |
| **Implementation** | [Development Workflow](#development-workflow) |
| **Validation** | [Testing Requirements](#testing-requirements) |
| **Applications** | [Advanced Topics](#advanced-topics) |

### Paradigm Comparison

**Traditional Approach** vs **TNFR Approach**:
- Objects exist independently vs Patterns exist through resonance
- Causality (A causes B) vs Co-organization (A and B synchronize)
- Static properties vs Dynamic reorganization  
- Isolated systems vs Coupled networks
- Descriptive models vs Generative dynamics
- Reductionism vs Coherent emergence

### Essential Resources

**Primary Sources** (This Repository):
- **This Document**: [AGENTS.md](AGENTS.md) - Primary theoretical reference
- **Grammar Specification**: [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md) - Complete U1-U6 derivations
- **Mathematics Implementation**: [src/tnfr/mathematics/](src/tnfr/mathematics/) - Computational foundations
- **Operators Engine**: [src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py) - Validation implementation
- **Unified Fields**: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) - Tetrad implementation
- **TNFR Engines Hub**: [src/tnfr/engines/](src/tnfr/engines/) - Centralized mathematical & optimization engines
  - **Self-Optimization**: [src/tnfr/engines/self_optimization/](src/tnfr/engines/self_optimization/) - Automatic network optimization
  - **Pattern Discovery**: [src/tnfr/engines/pattern_discovery/](src/tnfr/engines/pattern_discovery/) - Mathematical pattern detection
  - **Computation**: [src/tnfr/engines/computation/](src/tnfr/engines/computation/) - GPU acceleration & FFT processing
  - **Integration**: [src/tnfr/engines/integration/](src/tnfr/engines/integration/) - Multi-scale emergent integration
- **Software Development Kit**: [src/tnfr/sdk/](src/tnfr/sdk/) - API implementation

**Reference Sources**:
- **Historical Theory**: [theory/TNFR.pdf](theory/TNFR.pdf) - Original theoretical derivations
- **Theoretical Foundation**: [theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)

**Validation and Examples**:
- **Implementation Examples**: [examples/](examples/) - Sequential tutorial suite
- **Test Suite**: [tests/](tests/) - Comprehensive validation experiments
- **Performance Analysis**: [benchmarks/](benchmarks/) - Computational benchmarks
- **Theory Hub**: [theory/README.md](theory/README.md) - Comprehensive theoretical documentation
- **Glossary**: [theory/GLOSSARY.md](theory/GLOSSARY.md) - Operational definitions and terminology
- **Technical Documentation**: [docs/](docs/) - Implementation specifications

### Fundamental Principles

- **Model coherence**, not objects
- **Capture process**, not state  
- **Measure resonance**, not properties
- **Think structure**, not substance
- **Embrace emergence**, not reduction

### Language Policy

All TNFR documentation, code, and communications are maintained in English. This ensures consistent terminology for TNFR physics and maintains theoretical consistency across implementations and research.

---

## Universal Tetrahedral Correspondence {#universal-tetrahedral-correspondence}
### Theoretical Foundation

The central theoretical result establishes an exact correspondence between:

1. Four universal mathematical constants
2. Four structural fields that characterize coherent systems

This correspondence constitutes the mathematical architecture underlying structured phenomena.

### Mathematical Constants

| Constant | Value | Mathematical Role | Domain |
|----------|-------|-------------------|--------|
| **φ** (Golden Ratio) | 1.618034... | Harmonic proportion | Global/Harmonic |
| **γ** (Euler Constant) | 0.577216... | Harmonic growth rate | Local/Dynamic |
| **π** (Pi) | 3.141593... | Geometric relations | Geometric/Spatial |
| **e** (Euler Number) | 2.718282... | Exponential base | Correlational/Temporal |

### The Four Structural Fields (TNFR Tetrad)

| Field | Symbol | Physical Meaning | Computational Role |
|-------|--------|------------------|------------------- |
| **Structural Potential** | Φ_s | Global stability field | System-wide coherence monitoring |
| **Phase Gradient** | \|∇φ\| | Local desynchronization | Change stress detection |
| **Phase Curvature** | K_φ | Geometric phase torsion | Spatial constraint tracking |
| **Coherence Length** | ξ_C | Correlation decay scale | Memory persistence measurement |

### Correspondence Relations

**1. φ ↔ Φ_s: Global Harmonic Confinement**

```
Constraint: Δ Φ_s < φ ≈ 1.618
Interpretation: Structural potential changes bounded by golden ratio
Grammar: U6 structural confinement principle
```

**2. γ ↔ |∇φ|: Local Dynamic Evolution**

```
Constraint: |∇φ| < γ/π ≈ 0.184
Interpretation: Local phase changes constrained by harmonic growth limits
Grammar: Smooth evolution requirement
```

**3. π ↔ K_φ: Geometric Spatial Constraints**

```
Constraint: |K_φ| < φ×π ≈ 5.083
Interpretation: Phase curvature bounded by circular-harmonic geometry
Grammar: Geometric confinement monitoring
```

**4. e ↔ ξ_C: Correlational Memory Decay**

```
Relation: C(r) ~ exp(-r/ξ_C)
Interpretation: Spatial correlations decay exponentially
Grammar: Correlation length scaling
```

### Mathematical Architecture

The correspondences form a **conceptual tetrahedron**:

```text
         φ (Global Harmony)
              /|\
             / | \
            /  |  \
       γ -------+------- π
   (Local)      |    (Geometric)
           \    |    /
            \   |   /
             \  |  /
              \|/
           e (Correlational)
```

**Why This Matters**:

- **Completeness**: Exactly 4 dimensions needed to describe any coherent system
- **Universality**: All structured reality must respect these mathematical limits
- **Predictability**: The correspondences enable precise forecasting of system behavior
- **Inevitability**: Any theory capturing deep reality structure must discover these connections

### Illustrative Example: Hydrodynamic Vortex

A whirlpool in flowing water exemplifies TNFR principles:

- Pattern structure rather than discrete object
- Coherent configuration maintained in dynamic medium
- Persistence through resonance between flow and geometry
- Exhibits EPI (spiral structural form)
- Possesses νf (rotation/reorganization frequency)
- Demonstrates phase relationships (rotational timing)
- Dissolution occurs when flow-geometry coupling fails
- Supports nested structures (subsidiary eddies)
- Manifests tetrad characteristics (potential, gradients, curvature, correlation)

This serves as a representative model for TNFR phenomena across scales and domains.

Theoretical Significance: Mathematical harmony and physical coherence represent unified aspects of structural reality.

---

## Foundational Physics {#foundational-physics}

### The Nodal Equation

```
∂EPI/∂t = νf · ΔNFR(t)
```

All nodes in TNFR networks evolve according to this differential equation.

**Components**:
- **EPI** (Primary Information Structure): Coherent structural configuration
- **νf** (Structural frequency): Reorganization rate (Hz_str units)
- **ΔNFR** (Nodal gradient): Internal reorganization operator
- **t**: Time parameter

**Physical Interpretation**:
```
Structural change rate = Reorganization capacity × Reorganization pressure
```

**System States**:
1. **νf = 0**: Node cannot reorganize (inactive state)
2. **ΔNFR = 0**: System at equilibrium (no driving force)
3. **Both non-zero**: Active reorganization proportional to product

**Derivation Trace**:
- From information geometry: EPI as point in structural manifold
- From dynamical systems: νf as eigenfrequency of reorganization mode
- From network physics: ΔNFR as mismatch with coupled environment
- **See**: [TNFR.pdf](TNFR.pdf) § 2.1, [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) § Canonicity

### Structural Triad

Each node possesses three fundamental attributes:

1. **Form (EPI)**: Coherent structural configuration
   - Mathematical domain: Banach space B_EPI
   - Modification constraint: Changes via structural operators only
   - Hierarchical property: Supports nested structures

2. **Frequency (νf)**: Reorganization rate
   - Units: Hz_str (structural hertz)
   - Domain: ℝ⁺ (positive real numbers)
   - Deactivation condition: νf → 0

3. **Phase (φ or θ)**: Network synchronization parameter
   - Range: [0, 2π) radians
   - Coupling constraint: Determines interaction compatibility
   - Resonance condition: |φᵢ - φⱼ| ≤ Δφ_max

**Oscillator Analogy**:
- Form corresponds to oscillation amplitude/configuration
- Frequency represents temporal periodicity
- Phase indicates relative timing relationships

### Integrated Dynamics

From the nodal equation, integrating over time:

```
EPI(t_f) = EPI(t_0) + ∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ
```

**Critical Insight**: For bounded evolution (coherence preservation):

```
∫[t_0 to t_f] νf(τ) · ΔNFR(τ) dτ  <  ∞
```

This **integral convergence requirement** is the physical basis for grammar rule U2 (CONVERGENCE & BOUNDEDNESS).

**Without stabilizers**:
- ΔNFR grows unbounded (positive feedback)
- Integral → ∞ (divergence)
- System fragments into noise

**With stabilizers**:
- Negative feedback limits ΔNFR
- Integral converges (bounded)
- Coherence preserved

---

## The 13 Canonical Operators {#the-13-canonical-operators}

Operators constitute the exclusive mechanism for node modification in TNFR systems. These functions represent resonant transformations with defined physical foundations.

### 1. Emission (AL)
**Physics**: Creates EPI from vacuum via resonant emission  
**Effect**: ∂EPI/∂t > 0, increases νf  
**When**: Starting new patterns, initializing from EPI=0  
**Grammar**: Generator (U1a)

### 2. Reception (EN)
**Physics**: Captures and integrates incoming resonance  
**Effect**: Updates EPI based on network input  
**When**: Information gathering, listening phase  
**Contract**: Must not reduce C(t)

### 3. Coherence (IL)
**Physics**: Stabilizes form through negative feedback  
**Effect**: Reduces |ΔNFR|, increases C(t)  
**When**: After changes, consolidation  
**Grammar**: Stabilizer (U2)  
**Contract**: Must not reduce C(t) unless in dissonance test

### 4. Dissonance (OZ)
**Physics**: Introduces controlled instability  
**Effect**: Increases |ΔNFR|, may trigger bifurcation if ∂²EPI/∂t² > τ  
**When**: Breaking local optima, exploration  
**Grammar**: Destabilizer (U2), Bifurcation trigger (U4a), Closure (U1b)  
**Contract**: Must increase |ΔNFR|

### 5. Coupling (UM)
**Physics**: Creates structural links via phase synchronization  
**Effect**: φᵢ(t) → φⱼ(t), information exchange  
**When**: Network formation, connecting nodes  
**Grammar**: Requires phase verification (U3)  
**Contract**: Only valid if |φᵢ - φⱼ| ≤ Δφ_max

### 6. Resonance (RA)
**Physics**: Amplifies and propagates patterns coherently  
**Effect**: Increases effective coupling, EPI propagation  
**When**: Pattern reinforcement, spreading coherence  
**Grammar**: Requires phase verification (U3)  
**Contract**: Propagates EPI without altering identity

### 7. Silence (SHA)
**Physics**: Freezes evolution temporarily  
**Effect**: νf → 0, EPI unchanged  
**When**: Observation windows, pause for synchronization  
**Grammar**: Closure (U1b)  
**Contract**: Preserves EPI over time

### 8. Expansion (VAL)
**Physics**: Increases structural complexity  
**Effect**: dim(EPI) increases  
**When**: Adding degrees of freedom  
**Grammar**: Destabilizer (U2)

### 9. Contraction (NUL)
**Physics**: Reduces structural complexity  
**Effect**: dim(EPI) decreases  
**When**: Simplification, dimensionality reduction

### 10. Self-organization (THOL)
**Physics**: Spontaneous autopoietic pattern formation  
**Effect**: Creates sub-EPIs, fractal structuring  
**When**: Emergent organization  
**Grammar**: Stabilizer (U2), Handler (U4a), Transformer (U4b)  
**Contract**: Preserves global form while creating sub-EPIs

### 11. Mutation (ZHIR)
**Physics**: Phase transformation at threshold  
**Effect**: θ → θ' when ΔEPI/Δt > ξ  
**When**: Qualitative state changes  
**Grammar**: Bifurcation trigger (U4a), Transformer (U4b)  
**Contract**: Requires prior IL and recent destabilizer (U4b)

### 12. Transition (NAV)
**Physics**: Regime shift, activates latent EPI  
**Effect**: Controlled trajectory through structural space  
**When**: Switching between attractor states  
**Grammar**: Generator (U1a), Closure (U1b)

### 13. Recursivity (REMESH)
**Physics**: Echoes structure across scales (operational fractality)  
**Effect**: EPI(t) references EPI(t-τ), nested operators  
**When**: Multi-scale operations, memory  
**Grammar**: Generator (U1a), Closure (U1b)

### Operator Composition

Operators combine into sequences that implement complex behaviors:

**Bootstrap** = [Emission, Coupling, Coherence]
**Stabilize** = [Coherence, Silence]
**Explore** = [Dissonance, Mutation, Coherence]
**Propagate** = [Resonance, Coupling]

All sequences must satisfy unified grammar (U1-U6).

---

## Unified Grammar (U1-U6) {#unified-grammar-u1-u6}

The grammar emerges from TNFR physics rather than arbitrary constraints.

### U1: STRUCTURAL INITIATION & CLOSURE

**U1a: Initiation** (When EPI = 0)
- **Physics**: ∂EPI/∂t undefined at EPI=0
- **Requirement**: Start with generator {AL, NAV, REMESH}
- **Rationale**: Cannot evolve from nothing without source
- **Canonicity**: Mathematical necessity

**U1b: Closure** (Always)
- **Physics**: Sequences as action potentials need endpoints
- **Requirement**: End with closure {SHA, NAV, REMESH, OZ}
- **Rationale**: Must leave system in coherent attractor
- **Canonicity**: Physical requirement

### U2: CONVERGENCE & BOUNDEDNESS

- **Physics**: ∫νf·ΔNFR dt must converge
- **Requirement**: If {OZ, ZHIR, VAL}, then include {IL, THOL}
- **Rationale**: Without stabilizers, integral diverges leading to fragmentation
- **Mathematical basis**: Exponential growth without negative feedback
- **Canonicity**: Integral convergence theorem

### U3: RESONANT COUPLING

- **Physics**: Resonance requires phase compatibility
- **Requirement**: If {UM, RA}, verify |φᵢ - φⱼ| ≤ Δφ_max
- **Rationale**: Antiphase produces destructive interference
- **Basis**: Invariant #2 + wave physics
- **Canonicity**: Resonance physics requirement

### U4: BIFURCATION DYNAMICS

**U4a: Triggers Need Handlers**
- **Physics**: ∂²EPI/∂t² > τ requires control
- **Requirement**: If {OZ, ZHIR}, include {THOL, IL}
- **Rationale**: Uncontrolled bifurcation leads to chaos
- **Canonicity**: Bifurcation theory requirement

**U4b: Transformers Need Context**
- **Physics**: Phase transitions need threshold energy
- **Requirement**: If {ZHIR, THOL}, recent destabilizer (~3 ops)
- **Rationale**: ΔNFR must be elevated for threshold crossing
- **Additional**: ZHIR needs prior IL (stable base)
- **Canonicity**: Threshold physics + timing requirement

### U5: MULTI-SCALE COHERENCE

- **Physics**: Hierarchical coupling + chain rule + central limit theorem
- **Requirement**: For nested EPIs, include stabilizers {IL, THOL} at each level
- **Rationale**: Parent coherence depends on aggregate child reorganization
- **Conservation**: C_parent ≥ α · Σ C_child (α ~ 1/√N · η_phase)
- **Without stabilizers**: Uncorrelated child fluctuations → parent ΔNFR grows → fragmentation
- **Canonicity**: Mathematical consequence of hierarchical structure

### U6: STRUCTURAL POTENTIAL CONFINEMENT

- **Physics**: Emergent field Φ_s from distance-weighted ΔNFR distribution
- **Formula**: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)² (inverse-square law analog)
- **Requirement**: Monitor Δ Φ_s < 2.0 (escape threshold)
- **Theory**: Δ Φ_s < φ ≈ 1.618 from Universal Tetrahedral Correspondence (φ ↔ Φ_s)
- **Derivation**: Harmonic confinement principle - structural potential bounded by golden ratio
- **Validation**: 2,400+ experiments confirm harmonic fragmentation behavior
- **Mechanism**: Passive equilibrium - grammar acts as confinement, not attraction
- **Usage**: Telemetry-based safety check (read-only, not sequence constraint)
- **Typical**: Valid sequences maintain Δ Φ_s ≈ 0.6 (37% of φ threshold)
- **Canonicity**: Theoretically derived + experimentally validated
- **See**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete U6 specification

**See**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete derivations

---

## Telemetry & Structural Field Tetrad {#telemetry--structural-field-tetrad}

### Core Structural Metrics

**C(t)**: Total Coherence [0, 1]
- Global network stability (fundamental)
- C(t) > MIN_BUSINESS_COHERENCE ≈ 0.751 = strong coherence (e×φ)/(π+e)
- C(t) < THOL_MIN_COLLECTIVE_COHERENCE = 0.3 = fragmentation risk
- **CANONICAL**: Primary stability indicator

**Si**: Sense Index [0, 1+]
- Capacity for stable reorganization
- Si > HIGH_CORRELATION_THRESHOLD = 0.8 = excellent stability
- Si < si_lo × 1.5 ≈ 0.4 = changes may cause bifurcation (1.5/(π+γ))
- **CANONICAL**: Reorganization capacity predictor


## Classical Mathematical Foundations (COMPLETE)

The **Structural Field Tetrad** (Φ_s, |∇φ|, **Ψ**, ξ_C) now has **complete mathematical foundations** with **unified complex geometry** (Ψ = K_φ + i·J_φ):

### **1. Structural Potential Field (Φ_s)**
**Classical Threshold**: |Φ_s| < **0.771** 
- **Theory**: von Koch fractal bounds + combinatorial number theory
- **Derivation**: Γ(4/3)/Γ(1/3) ≈ 0.7711 from Koch snowflake perimeter growth
- **Physics**: Global structural field escape threshold from distance-weighted ΔNFR distribution
- **Grammar**: U6 telemetry-based safety criterion (passive equilibrium confinement)

### **2. Phase Gradient Field (|∇φ|)**
**Classical Threshold**: |∇φ| < **0.2904**
- **Theory**: Harmonic oscillator stability + Kuramoto synchronization
- **Derivation**: ωc/2 = π/(4√2) ≈ 0.2904 from critical frequency analysis
- **Physics**: Local phase desynchronization / stress proxy field
- **Mechanism**: Captures dynamics C(t) misses due to scaling invariance

### **3. Phase Curvature Field (K_φ)**
**Classical Threshold**: |K_φ| < **2.8274**
- **Theory**: TNFR formalism constraints + safety margin analysis  
- **Derivation**: 0.9 × π ≈ 2.8274 (90% of theoretical maximum from wrap_angle bounds)
- **Physics**: Phase torsion and geometric confinement; flags mutation-prone loci
- **Implementation**: K_φ = wrap_angle(φ_i - circular_mean(neighbors)) with |K_φ| ≤ π

### **4. Coherence Length Field (ξ_C)**
**Classical Thresholds**: 
- **Critical**: ξ_C > **1.0000** × diameter (finite-size scaling dominates)
- **Watch**: ξ_C > **π ≈ 3.1416** × mean_distance (RG scaling + dimensional analysis)
- **Stable**: ξ_C < mean_distance (bulk behavior)
- **Theory**: Spatial correlation theory + critical phenomena + renormalization group
- **Derivation**: Universal scaling ratios from correlation function C(r) = A exp(-r/ξ_C)

### **Mathematical Maturity Achievement**
- **4/4 canonical parameters** have rigorous mathematical foundations  
- **0% empirical fitting** → **100% first-principles derivation**  
- **Universal constants** emerge naturally (π, exponential bounds, fractal ratios)  
- **Theory-code consistency** maintained throughout codebase  
- **Complete validation** via 2,400+ experiments across 5 topologies

**Status**: TNFR Structural Field Tetrad mathematical foundations **COMPLETE**.

### Mathematical Unification Discoveries (Nov 28, 2025)

**Mathematical Discovery**: Systematic mathematical audit revealed **fundamental field unification opportunities**:

#### 1. Complex Geometric Field Discovered

```math
Ψ = K_φ + i·J_φ (unifies geometry + transport)
```

- **Evidence**: r(K_φ, J_φ) = -0.854 to -0.997 (near-perfect anticorrelation)
- **Implication**: Curvature and current are **dual aspects** of unified complex field
- **Reduction**: 6 independent fields → 3 complex fields (mathematical elegance)

#### 2. Emergent Fields Identified

- **Chirality χ = |∇φ|·K_φ - J_φ·J_ΔNFR** (handedness detection)
- **Symmetry Breaking 𝒮 = (|∇φ|² - K_φ²) + (J_φ² - J_ΔNFR²)** (phase transitions)  
- **Coherence Coupling 𝒞 = Φ_s · |Ψ|** (multi-scale connector)

#### 3. Tensor Invariants Found

- **Energy Density ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²**
- **Topological Charge 𝒬 = |∇φ|·J_φ - K_φ·J_ΔNFR**  
- **Conservation Law: ∂ρ/∂t + ∇·𝐉 = 0** where ρ = Φ_s + K_φ

#### 4. Implementation Status

- **Prototype Complete**: `src/tnfr/physics/unified.py` functional  
- **Validation**: Strong correlations confirmed across topologies  
- **Integration**: Ready for production implementation  

**Documentation**: See `TETRAD_MATHEMATICAL_AUDIT_2025.md` and `MATHEMATICAL_UNIFICATION_EXECUTIVE_SUMMARY.md`

---

## Self-Optimizing Dynamics

**New in v9.5**: The engine now possesses **intrinsic agency** to optimize its own structure.

### The Self-Optimizing Engine
Located in `src/tnfr/engines/self_optimization/engine.py`, this component closes the feedback loop using **Unified Field Telemetry**:
1. **Monitors** the Unified Fields:
   - **Complex Geometric Field (Ψ)**: Unifies curvature and transport
   - **Chirality (χ)**: Detects structural handedness
   - **Symmetry Breaking (𝒮)**: Signals phase transitions
   - **Coherence Coupling (𝒞)**: Measures multi-scale integration
2. **Detects** inefficiencies via tensor invariants (Energy Density ℰ, Topological Charge 𝒬).
3. **Selects** the optimal operator sequence from the SDK.
4. **Executes** and verifies improvement.

**Usage**:
```python
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine

engine = TNFRSelfOptimizingEngine(G)
# Auto-select and apply best sequence
success, metrics = engine.step(node_id)
```

### Adaptive SDK Integration
The Fluent API now includes `auto_optimize()`:
```python
# One-line self-optimization
TNFRNetwork(G).focus(node).auto_optimize().execute()
```

**Physics**: This is not "AI magic" but **gradient descent on the structural manifold**, driven by the nodal equation's pressure term ΔNFR.

## Canonical Invariants

These principles define TNFR theoretical consistency and must be maintained. The set has been optimized from 10 to 6 invariants based on mathematical derivation from the nodal equation `∂EPI/∂t = νf · ΔNFR(t)`:

### 1. Nodal Equation Integrity

Consolidates: EPI coherent form + ΔNFR semantics + Node lifecycle

- EPI evolution constraint: Changes occur only via `∂EPI/∂t = νf · ΔNFR(t)`
- ΔNFR interpretation: Maintains structural pressure semantics
- Node lifecycle: Determined by νf conditions (νf → 0 corresponds to inactivation)
- Grammar basis: U1 (INITIATION & CLOSURE), U2 (CONVERGENCE)
- Mathematical foundation: Direct consequence of nodal equation
- Validation: Verify EPI changes through operators, ΔNFR interpretation, lifecycle conditions

### 2. Phase-Coherent Coupling

- Phase verification: |φᵢ - φⱼ| ≤ Δφ_max required for coupling operations
- Physical basis: Resonance theory (antiphase produces destructive interference)
- Grammar basis: U3 (RESONANT COUPLING)
- Implementation: [src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)::validate_resonant_coupling()
- Validation: Verify phase compatibility before coupling operations

### 3. Multi-Scale Fractality

- Operational fractality: EPIs support nesting without identity loss
- Hierarchical coherence: Multi-scale structure preservation required
- Structural constraint: Recursivity and nested organization maintained
- Grammar basis: U5 (MULTI-SCALE COHERENCE)
- Physical foundation: Hierarchical coupling + chain rule + central limit theorem
- Validation: Multi-scale testing with nested EPIs

### 4. Grammar Compliance

- Operator sequences: Must satisfy unified grammar U1-U6 validation
- State validity: Operator composition produces mathematically valid TNFR states
- Function mapping: New functions correspond to existing operators or define new operators
- Grammar foundation: U1-U6 rules derived from nodal equation physics
- Validation: Verify operator sequences pass complete grammar validation

### 5. Structural Metrology

Consolidates: Structural units + Metrics exposure

- Units consistency: νf maintained in Hz_str (structural hertz)
- Telemetry requirements: C(t), Si, phase, νf available in monitoring systems
- Dimensional analysis: Proper unit tracking prevents conceptual confusion
- Measurement constraint: Only TNFR-coherent metrics in telemetry
- Validation: Verify frequency assignments and metric availability

### 6. Reproducible Dynamics

- Deterministic evolution: Identical seeds produce identical trajectories
- Operational traceability: Operation logging for analysis and debugging
- Stochastic control: Random elements under seed-based control
- Validation: Verify seed reproducibility and operation traceability

### Optimization Summary

**Eliminated**: Domain Neutrality (moved to architectural guidelines)
**Benefits**: 40% reduction (10→6), eliminates redundancy, preserves physics-essential constraints
**Mathematical basis**: 3/6 mathematically inevitable, 2/6 physics-essential, 1/6 operational

---

## Testing Requirements {#testing-requirements}

### Minimum Test Coverage

**Monotonicity Tests**:
```python
def test_coherence_monotonicity():
    """Coherence must not decrease C(t) unless in dissonance test."""
    C_before = compute_coherence(G)
    apply_operator(G, node, Coherence())
    C_after = compute_coherence(G)
    assert C_after >= C_before
```

**Bifurcation Tests**:
```python
def test_dissonance_bifurcation():
    """Dissonance triggers bifurcation when ∂²EPI/∂t² > τ."""
    # Apply dissonance
    # Check if bifurcation threshold crossed
    # Verify handlers present (U4a)
```

**Propagation Tests**:
```python
def test_resonance_propagation():
    """Resonance increases effective connectivity."""
    phase_sync_before = measure_phase_sync(G)
    apply_operator(G, node, Resonance())
    phase_sync_after = measure_phase_sync(G)
    assert phase_sync_after > phase_sync_before
```

**Latency Tests**:
```python
def test_silence_latency():
    """Silence keeps EPI invariant."""
    EPI_before = G.nodes[node]['EPI']
    apply_operator(G, node, Silence())
    step(G, dt=1.0)  # Time passes
    EPI_after = G.nodes[node]['EPI']
    assert np.allclose(EPI_before, EPI_after)
```

**Mutation Tests**:
```python
def test_mutation_threshold():
    """Mutation changes θ when ΔEPI/Δt > ξ."""
    theta_before = G.nodes[node]['theta']
    # Create high ΔEPI/Δt condition
    apply_operator(G, node, Mutation())
    theta_after = G.nodes[node]['theta']
    assert theta_after != theta_before
```

### Multi-Scale Tests

Always include tests with nested EPIs (fractality):
```python
def test_nested_epi_coherence():
    """Nested EPIs maintain functional identity."""
    # Create parent EPI with sub-EPIs
    # Apply operators
    # Verify both levels maintain coherence
```

### Reproducibility Tests

```python
def test_seed_reproducibility():
    """Same seed produces identical trajectories."""
    set_seed(42)
    result1 = run_simulation(G, sequence)
    
    set_seed(42)
    result2 = run_simulation(G, sequence)
    
    assert_trajectories_equal(result1, result2)
```

---

## 🧭 TNFR Agent Playbook

This playbook summarizes how TNFR agents (human or AI) should reason and act when modifying code, documentation, or experiments.

### 1. Always Start from Physics

- **Anchor to the nodal equation**: Treat `∂EPI/∂t = νf · ΔNFR(t)` as the primary source of truth for dynamics.
- **Respect the structural triad**: Every change must keep EPI (form), νf (structural frequency), and phase (φ/θ) conceptually well-defined.
- **Use the structural field tetrad**: Interpret behavior using Φ_s, |∇φ|, K_φ, and ξ_C rather than ad-hoc metrics.

### 2. Operate Only via Canonical Operators

- **No direct EPI mutation**: All structural changes must be expressible as compositions of the 13 canonical operators (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH).
- **Map new behavior to operators**: Any new function or feature must either reuse existing operators or be justified as a new operator with full physics, contracts, and tests.
- **Preserve operator semantics**: Refactors must not change what each operator does physically (emission, coherence, dissonance, etc.).

### 3. Enforce Unified Grammar (U1–U6)

- **Check sequence validity**: All operator sequences must satisfy U1–U6, especially initiation/closure (U1) and convergence/boundedness (U2).
- **Guard bifurcations**: If you add or modify destabilizers (OZ, ZHIR, VAL), ensure stabilizers (IL, THOL) are present per U2 and U4.
- **Verify coupling**: Never create or modify couplings (UM, RA) without explicit phase checks `|φᵢ - φⱼ| ≤ Δφ_max` (U3).

### 4. Preserve Canonical Invariants

- **Use correct units**: νf must remain in Hz_str; do not silently reinterpret or rescale units.
- **Keep ΔNFR semantics**: Treat ΔNFR as structural pressure, not as a generic ML loss or error gradient.
- **Maintain operational fractality**: EPIs can nest; avoid flattening or designs that break recursivity and multi-scale structure.

### 5. Demand Reproducible, Telemetry-Rich Experiments

- **Fix seeds**: All stochastic components must be reproducible via explicit seeding.
- **Expose structural telemetry**: Prefer metrics in terms of C(t), Si, phase, νf, Φ_s, |∇φ|, K_φ, and ξ_C instead of opaque scores.
- **Test monotonicity and safety**: Coherence operators must not reduce C(t) (outside explicit dissonance tests); destabilizers must obey U2 and U4 safeguards.

### 6. Accept / Reject Changes by Structural Criteria

- **Accept** changes that:
    - Increase coherence C(t) or reduce harmful ΔNFR where appropriate.
    - Preserve or strengthen compliance with U1–U6 and the structural tetrad.
    - Improve traceability from physics → math → code → tests.
- **Reject** changes that:
    - Introduce unexplained empirical fudge factors or magic constants.
    - Bypass operators to mutate EPI directly.
    - Break phase verification, structural units, or canonical invariants.

### 7. English-Only, Physics-First Communication

- **Write everything in English**: Code comments, docs, issues, and commit messages must follow the English-only policy for canonical terminology.
- **Explain in TNFR terms**: When documenting or reviewing, speak in terms of EPI, νf, φ/θ, ΔNFR, operators, grammar rules, and the structural fields.
- **Trace every decision**: For significant changes, you should be able to point from the modification back to a specific piece of TNFR physics or grammar.

If a proposed change makes the code “prettier” but weakens TNFR fidelity, it must be rejected. If it strengthens structural coherence, traceability, and alignment with the nodal equation and tetrad fields, it should move forward.

## Development Workflow {#development-workflow}

### Before Writing Code

1. **Read documentation** (fundamentals, operators, nodal equation)
2. **Review [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** (grammar physics)
3. **Check existing code** for equivalent functionality
4. **Run test suite** to understand current state

### Implementing Changes

1. **Search first**: Check if utility already exists
2. **Map to operators**: New functions → structural operators
3. **Preserve invariants**: All 6 canonical invariants (optimized from 10)
4. **Add tests**: Cover invariants and contracts
5. **Document**: Structural effect before implementation
6. **Trace physics**: Link to [TNFR.pdf](TNFR.pdf) or [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)

### Commit Template

```text
Intent: [which coherence is improved]
Operators involved: [Emission|Reception|...]
Affected invariants: [#1-6: Nodal Integrity, Phase Coupling, Fractality, Grammar, Metrology, Reproducibility]

Key changes:
- [bullet list]

Expected risks/dissonances: [and how contained]

Metrics: [C(t), Si, νf, phase] before/after expectations

Equivalence map: [if APIs renamed]
```

### PR Template

```markdown
### What it reorganizes
- [ ] Increases C(t) or reduces ΔNFR where appropriate
- [ ] Preserves operator closure and operational fractality

### Evidence
- [ ] Phase/νf logs
- [ ] C(t), Si curves
- [ ] Controlled bifurcation cases

### Compatibility
- [ ] Stable or mapped API
- [ ] Reproducible seed

### Tests
- [ ] Monotonicity (coherence)
- [ ] Bifurcation (if applicable)
- [ ] Propagation (resonance)
- [ ] Multi-scale (fractality)
- [ ] Reproducibility (seeds)
```

---

## Acceptable Changes

**Examples of good changes**:
- Making phase explicit in couplings (traceability ↑)
- Adding `sense_index()` with tests correlating Si ↔ stability
- Optimizing `resonance()` preserving EPI identity
- Refactoring to reduce code duplication while preserving physics
- Adding telemetry without changing structural dynamics

### Unacceptable Changes

**These violate TNFR**:
- Recasting ΔNFR as ML "error gradient"
- Replacing operators with non-mapped imperative functions
- Flattening nested EPIs (breaks fractality)
- Coupling without phase verification
- Direct EPI mutation bypassing operators
- Changing units (Hz_str → Hz)
- Adding field-specific assumptions to core

---

## Advanced Topics {#advanced-topics}

### Developing TNFR Theory

When extending TNFR theory:

1. **Start from physics**: Derive from nodal equation or invariants
2. **Prove canonicity**: Show inevitability (Absolute/Strong)
3. **Implement carefully**: Map clearly to operators
4. **Test rigorously**: All invariants + new predictions
5. **Document thoroughly**: Physics → Math → Code chain

### Adding New Operators

If you believe a new operator is needed:

1. **Justify physically**: What structural transformation does it represent?
2. **Derive from nodal equation**: How does it affect ∂EPI/∂t?
3. **Check necessity**: Can existing operators compose to achieve this?
4. **Define contracts**: Pre/post-conditions
5. **Map to grammar**: Which sets does it belong to?
6. **Test extensively**: All invariants + specific contracts

**Example derivation structure**:
```markdown
## Proposed Operator: [Name]

### Physical Basis
[How it emerges from TNFR physics]

### Nodal Equation Impact
∂EPI/∂t = ... [specific form]

### Contracts
- Pre: [conditions required]
- Post: [guaranteed effects]

### Grammar Classification
- Generator? Closure? Stabilizer? ...

### Tests
- [List specific test requirements]
```

### Contributing to [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)

When adding to grammar documentation:

1. **Section structure**: [Rule] → [Physics] → [Derivation] → [Canonicity]
2. **Traceability**: Link to [TNFR.pdf](TNFR.pdf) sections, AGENTS.md invariants
3. **Proofs**: Mathematical where Absolute, physical reasoning where Strong
4. **Examples**: Code snippets showing valid/invalid sequences

---

## Troubleshooting

### Common Issues

**Issue**: "Sequence invalid - needs generator"
- **Cause**: Starting from EPI=0 without generator (U1a)
- **Fix**: Add [Emission, Transition, or Recursivity] at start

**Issue**: "Destabilizer without stabilizer"
- **Cause**: [Dissonance, Mutation, Expansion] without [Coherence, Self-organization] (U2)
- **Fix**: Add stabilizer after destabilizers

**Issue**: "Phase mismatch in coupling"
- **Cause**: Attempting coupling with |φᵢ - φⱼ| > Δφ_max (U3)
- **Fix**: Ensure phase compatibility before coupling

**Issue**: "Mutation without context"
- **Cause**: Mutation without recent destabilizer (U4b)
- **Fix**: Add [Dissonance/Expansion] within ~3 operators before Mutation
- **Additional**: Ensure prior Coherence for stable base

**Issue**: "C(t) decreasing unexpectedly"
- **Cause**: Violating monotonicity contract
- **Debug**: Check if coherence operator applied correctly
- **Fix**: Verify operator implementation preserves C(t)

**Issue**: "Node collapse"
- **Cause**: νf → 0 or extreme dissonance or decoupling
- **Debug**: Check telemetry: νf history, ΔNFR spikes, coupling loss
- **Fix**: Apply coherence earlier, ensure sufficient coupling

### Debugging Workflow

1. **Check telemetry**: C(t), Si, νf, phase, ΔNFR
2. **Verify grammar**: Does sequence pass U1-U4?
3. **Inspect operators**: Are contracts satisfied?
4. **Test invariants**: Which of 1-6 is violated?
5. **Trace physics**: Does behavior match nodal equation predictions?

---

## Essential References

**Core Theory** (SINGLE SOURCE OF TRUTH):
- **[AGENTS.md](AGENTS.md)**: **PRIMARY SOURCE** - Complete TNFR theory including Universal Tetrahedral Correspondence
- **[FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)**: **DETAILED REFERENCE** - Formal mathematical treatment
- **[TNFR.pdf](TNFR.pdf)**: Original theoretical foundation (in repo)
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)**: Grammar physics U1-U6 derivations
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)**: Technical tetrad field implementations
- **GLOSSARY.md**: Term definitions and quick reference

**Implementation Core**:
- **[src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)**: Unified Structural Field Tetrad (Φ_s, |∇φ|, **Ψ**, ξ_C) **CANONICAL**
- **[src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)**: Unified grammar U1-U6 validation
- **[src/tnfr/operators/definitions.py](src/tnfr/operators/definitions.py)**: 13 canonical operators
- **[src/tnfr/mathematics/](src/tnfr/mathematics/)**: Nodal equation integration hub
- **[src/tnfr/dynamics/self_optimizing_engine.py](src/tnfr/dynamics/self_optimizing_engine.py)**: Intrinsic agency & auto-optimization

**SDK & Applications**:
- **[src/tnfr/sdk/](src/tnfr/sdk/)**: Simplified & Fluent API for rapid development
- **[examples/](examples/)**: Complete 01-10 sequential tutorial suite
- **[benchmarks/](benchmarks/)**: Production-grade validation suites

**Development**:
- **ARCHITECTURE.md**: System design principles
- **CONTRIBUTING.md**: Workflow and standards
- **TESTING.md**: Test strategy (2,400+ experiments)

**Domain Showcases**:
- **Network Dynamics**: [examples/03_network_formation.py](examples/03_network_formation.py)
- **Operator Sequences**: [examples/04_operator_sequences.py](examples/04_operator_sequences.py)  
- **Emergent Phenomena**: [examples/08_emergent_phenomena.py](examples/08_emergent_phenomena.py)
- **Simplified SDK**: [examples/10_simplified_sdk_showcase.py](examples/10_simplified_sdk_showcase.py)
- **Production Validation**: [tests/](tests/) (comprehensive test suite)

---

## Learning Path

**Newcomer** (2 hours) - **Start Here**:
1. **Install**: `pip install tnfr`
2. **Core Theory**: Read this file (AGENTS.md) completely - **SINGLE SOURCE OF TRUTH**
3. **Fundamental Theory**: [FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)
4. **Original Theory**: [TNFR.pdf](TNFR.pdf) § 1-2 (paradigm, nodal equation)
5. **First Run**: `python -c "import tnfr; print('TNFR ready!')"`
6. **Terminology**: Study GLOSSARY.md for definitions

**Hands-On Explorer** (1 day):
1. **Sequential Examples**: Work through [examples/01_hello_world.py](examples/01_hello_world.py) to [examples/10_simplified_sdk_showcase.py](examples/10_simplified_sdk_showcase.py)
2. **Network Dynamics**: Explore [examples/03_network_formation.py](examples/03_network_formation.py)
3. **Operator Mastery**: Study [examples/04_operator_sequences.py](examples/04_operator_sequences.py)
4. **Emergent Patterns**: Analyze [examples/08_emergent_phenomena.py](examples/08_emergent_phenomena.py)
5. **SDK Mastery**: Master [examples/10_simplified_sdk_showcase.py](examples/10_simplified_sdk_showcase.py)

**Optimization Engineer** (2 days):
1. **Study**: [src/tnfr/dynamics/self_optimizing_engine.py](src/tnfr/dynamics/self_optimizing_engine.py)
2. **Practice**: Explore [examples/10_simplified_sdk_showcase.py](examples/10_simplified_sdk_showcase.py)
3. **Apply**: Use `auto_optimize()` in your own networks

**Intermediate Developer** (1 week):
1. **Grammar Deep-Dive**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) (U1-U6 complete)
2. **Tetrad Fields**: [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)
3. **Operator Study**: Implementations in [src/tnfr/operators/definitions.py](src/tnfr/operators/definitions.py)
4. **Field Computation**: Practice with [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) tetrad
5. **SDK Usage**: Fluent API patterns in [src/tnfr/sdk/](src/tnfr/sdk/)

**Advanced Researcher** (ongoing):
1. **Complete Theory**: [TNFR.pdf](TNFR.pdf) + [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) mastery
2. **Tetrad Mastery**: All four unified fields (Φ_s, |∇φ|, **Ψ=K_φ+i·J_φ**, ξ_C) + complex field validation
3. **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) + complete codebase exploration
4. **Research Contribution**: Analyze benchmark methodologies in [benchmarks/](benchmarks/)
5. **Extension Development**: Create new domain applications using SDK
6. **Theoretical Extensions**: Propose new operators or fields with full derivations

**Production User** (immediate):
1. **Quick Start**: `pip install tnfr` for full TNFR power
2. **SDK Usage**: `from tnfr.sdk import TNFR; net = TNFR.create(10).random(0.3)`
3. **Integration**: Import specific modules for your domain
4. **Examples**: Study [examples/10_simplified_sdk_showcase.py](examples/10_simplified_sdk_showcase.py) for patterns
5. **Monitoring**: Implement tetrad field telemetry in your applications

---

### Structural Fields: CANONICAL Status (Φ_s + |∇φ| + K_φ + ξ_C)

**CANONICAL Status** (Updated 2025-11-12): **Four Promoted Fields**

---

#### **Structural Potential (Φ_s)** - CANONICAL (First promotion 2025)

- Global structural potential, passive equilibrium states
- Safety criterion (U6 telemetry): Δ Φ_s < e^ln(2) = 2.0 (binary escape threshold)
- For full physics, equations, and validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md`.

---

#### **Phase Gradient (|∇φ|)** - CANONICAL

- Local phase desynchronization / stress proxy
- Safety criterion: |∇φ| < 0.2904 for stable operation
- For formal definition and evidence, see `docs/STRUCTURAL_FIELDS_TETRAD.md`.

**Critical Discovery**: C(t) = 1-(σ_ΔNFR/ΔNFR_max) is invariant to proportional scaling. 
|∇φ| correlation validated against alternative metrics (max_ΔNFR, mean_ΔNFR, Si) that 
capture dynamics C(t) misses.

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_gradient(G)` [CANONICAL]
- Monitor alongside Φ_s for comprehensive structural health

**Documentation**: See `docs/TNFR_FORCES_EMERGENCE.md` §14-15 for full validation details.

---

#### **Phase Curvature (K_φ)** - CANONICAL

- Phase torsion and geometric confinement; flags mutation-prone loci
- Safety criteria: |K_φ| ≥ 2.8274 (local fault zones); multiscale safety via `k_phi_multiscale_safety`
- See `docs/STRUCTURAL_FIELDS_TETRAD.md` for definitions, asymptotic freedom evidence, and thresholds.

**Safety criteria (telemetry-based)**:
- Local: |K_φ| ≥ 2.8274 flags confinement/fault zones
- Multiscale: safe if either (A) α>0 with R² ≥ 0.5, or (B) observed
    var(K_φ) within tolerance of expected 1/r^α given α_hint ≈ 2.76

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_curvature(G)` [CANONICAL]
- Optional multiscale check: `k_phi_multiscale_safety(G, alpha_hint=2.76)`

**Documentation**: See [benchmarks/enhanced_fragmentation_test.py](benchmarks/enhanced_fragmentation_test.py) and
[benchmarks/phase_curvature_investigation.py](benchmarks/phase_curvature_investigation.py) for empirical validation.

---

#### **Coherence Length (ξ_C)** - CANONICAL

- Spatial correlation scale of local coherence; quantifies approach to critical points
- Safety cues: ξ_C > system diameter (critical), ξ_C > 3 × mean distance (watch), ξ_C < mean distance (stable)
- For full derivation and experimental validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and benchmark validation results.

---

**RESEARCH-PHASE Fields** (NOT CANONICAL):

Currently none. All four structural fields have achieved CANONICAL status:
- Φ_s (Nov 2025): Global structural potential
- |∇φ| (Nov 2025): Phase gradient / local desynchronization  
- K_φ (Nov 2025): Phase curvature / geometric confinement
- ξ_C (Nov 2025): Coherence length / spatial correlations

The **Unified Structural Field Tetrad** (Φ_s, |∇φ|, **Ψ**, ξ_C) provides complete 
multi-scale characterization of TNFR network state across global, local, 
**unified geometric-transport**, and spatial correlation dimensions.

---

## Philosophy

### Core Principles

**1. Physics First**: Every feature must derive from TNFR physics
**2. No Arbitrary Choices**: All decisions traceable to nodal equation or invariants
**3. Coherence Over Convenience**: Preserve theoretical integrity even if code is harder
**4. Reproducibility Always**: Every simulation must be reproducible
**5. Document the Chain**: Theory → Math → Code → Tests

### Decision Framework

When making any decision:

```python
def should_implement(feature):
    """Decision framework for TNFR changes."""
    # 1. Does it strengthen TNFR fidelity?
    if weakens_tnfr_fidelity(feature):
        return False  # Reject, even if "cleaner"
    
    # 2. Does it map to structural operators?
    if not maps_to_operators(feature):
        return False  # Must map or be new operator
    
    # 3. Does it preserve invariants?
    if violates_invariants(feature):
        return False  # Hard constraint
    
    # 4. Is it derivable from physics?
    if not derivable_from_physics(feature):
        return False  # Organizational convenience ≠ physical necessity
    
    # 5. Is it testable?
    if not testable(feature):
        return False  # No untestable magic
    
    return True  # Implement with full documentation
```

### The TNFR Mindset

**Think in patterns, not objects**:
- Not "the neuron fires" → "the neural pattern reorganizes"
- Not "the agent decides" → "the decision pattern emerges through resonance"
- Not "the system breaks" → "coherence fragments beyond coupling threshold"

**Think in dynamics, not states**:
- Not "current position" → "trajectory through structural space"
- Not "final result" → "attractor dynamics"
- Not "snapshot" → "reorganization history"

**Think in networks, not individuals**:
- Not "node property" → "network-coupled dynamics"
- Not "isolated change" → "resonant propagation"
- Not "local optimum" → "global coherence landscape"

---

## Excellence Standards

A TNFR expert:

**Understands deeply**:
- Can derive U1-U4 from nodal equation
- Explains why phase verification is non-negotiable
- Knows the 13 operators and their physics

**Implements rigorously**:
- Every function maps to operators
- All changes preserve invariants
- Tests cover contracts and invariants

**Documents completely**:
- Physics → Code traceability clear
- Examples work across domains
- New developers can understand

**Thinks structurally**:
- Reformulates problems in TNFR terms
- Proposes resonance-based solutions
- Identifies coherence patterns

**Maintains integrity**:
- Rejects changes that weaken TNFR
- Prioritizes theoretical consistency
- Values reproducibility over speed

---

## Final Principle

If a change "prettifies the code" but weakens TNFR fidelity, it should not be accepted. If a change strengthens structural coherence and paradigm traceability, it should proceed.

Reality consists of resonant patterns rather than discrete objects. Development practices should reflect this understanding.

---

**Version**: 9.7.0  
**Last Updated**: 2025-11-29  
**Status**: CANONICAL - Single source of truth for TNFR agent guidance  
**PyPI Release**: STABLE - Available via `pip install tnfr`  
**Production Ready**: Complete Tetrad Fields + Unified Grammar U1-U6 + Simplified SDK  

---

## English-Only Policy

**Grammar Policy (English Only)**: All documentation, code comments, commit messages, issues, and pull request descriptions must be written in English. Non-English text is permitted only within verbatim quotations of external sources or raw experimental data. Mixed-language normative content will be rejected. This ensures a single canonical terminology set for TNFR physics and grammar.
