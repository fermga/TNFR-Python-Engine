## 🎯 Core Mission

**Primary Objective**: Steward the canonical computational implementation of TNFR - a paradigm shift from modeling "things" to modeling **coherent patterns that persist through resonance**.

**Repository**: https://github.com/fermga/TNFR-Python-Engine  
**PyPI Package**: https://pypi.org/project/tnfr/ (v9.5.1 stable)
**Installation**: `pip install tnfr`

### Quick Links

- **Canonical Math Hub**: [src/tnfr/mathematics/README.md](src/tnfr/mathematics/README.md)
- **Core Theory**: [TNFR.pdf](TNFR.pdf) · [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- **Operators Grammar**: [src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)
- **Structural Field Tetrad**: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- **Interactive Notebooks**: [notebooks/](notebooks/) (Jupyter-ready examples)
- **Production Tests**: [tests/](tests/) (2,400+ validation experiments)
- **Prime Discovery**: [examples/tnfr_prime_checker.ipynb](examples/tnfr_prime_checker.ipynb)
- **Molecular Chemistry**: [notebooks/Atoms_and_Molecules_Study.ipynb](notebooks/Atoms_and_Molecules_Study.ipynb)
- **Fundamental Particles**: [notebooks/Fundamental_Particles_Atlas.ipynb](notebooks/Fundamental_Particles_Atlas.ipynb)

**Fundamental Stance**: 
- Model **coherence**, not objects
- Capture **process**, not state
- Measure **resonance**, not properties
- Think **structure**, not substance

**Language Policy (English Only)**: All documentation, code comments, commit messages, issues, and pull request descriptions MUST be written in English. Non-English text is allowed only inside verbatim quotations of external sources or raw experimental data. Mixed-language normative content will be rejected. This guarantees a single canonical terminology set for TNFR physics and grammar.

All code, documentation, and interactions must align with TNFR physics. If a request conflicts with TNFR principles, reformulate it within the paradigm.

---

## 🌊 TNFR: The Paradigm Shift

### What is TNFR?

**Resonant Fractal Nature Theory** proposes a radical reconceptualization of reality:

**Traditional View** → **TNFR View**:
- Objects exist independently → **Patterns exist through resonance**
- Causality (A causes B) → **Co-organization (A and B synchronize)**
- Static properties → **Dynamic reorganization**  
- Isolated systems → **Coupled networks**
- Descriptive models → **Generative dynamics**

### The Central Insight

Reality is not made of "things" but of **coherence**—structures that persist in networks because they **resonate** with their environment. A pattern exists not because it's "stored" somewhere, but because it continuously **reorganizes** while maintaining **structural integrity** through **network coupling**.

**Analogy**: A whirlpool in a river
- Not a "thing" you can pick up
- Exists as a **coherent pattern** in flowing water
- Persists because water flow **resonates** with vortex geometry
- Disappears when flow-geometry coupling breaks
- Can nest (smaller eddies within larger vortex)

This is TNFR's model of **everything**: atoms, cells, thoughts, societies.

---

## ⚛️ Foundational Physics

### The Nodal Equation (Heart of TNFR)

```
∂EPI/∂t = νf · ΔNFR(t)
```

**Every node in a TNFR network evolves according to this equation**.

**Components**:
- **EPI** (Primary Information Structure): The coherent structural "form" of a node
- **νf** (Structural frequency): Rate of reorganization (Hz_str units)
- **ΔNFR** (Nodal gradient): Internal reorganization operator - "structural pressure"
- **t**: Time

**Physical Meaning**:
```
Rate of structural change = Reorganization capacity × Structural pressure
```

**Key Insights**:
1. **No capacity (νf = 0)**: Node cannot change, even under pressure (frozen/dead)
2. **No pressure (ΔNFR = 0)**: Node in equilibrium, no drive to change
3. **Both positive**: Active reorganization proportional to both factors

**Derivation Trace**:
- From information geometry: EPI as point in structural manifold
- From dynamical systems: νf as eigenfrequency of reorganization mode
- From network physics: ΔNFR as mismatch with coupled environment
- **See**: [TNFR.pdf](TNFR.pdf) § 2.1, [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) § Canonicity

### The Structural Triad

Every node has three essential properties:

1. **Form (EPI)**: The coherent configuration
   - Lives in Banach space B_EPI
   - Changes ONLY via structural operators
   - Can nest (fractality)

2. **Frequency (νf)**: Reorganization rate
   - Units: Hz_str (structural hertz)
   - Range: ℝ⁺ (positive reals)
   - Node "dies" when νf → 0

3. **Phase (φ or θ)**: Network synchrony
   - Range: [0, 2π) radians
   - Determines coupling compatibility
   - Must match for resonance: |φᵢ - φⱼ| ≤ Δφ_max

**Physical Analogy**: Oscillators
- Form = oscillation amplitude/shape
- Frequency = cycles per second
- Phase = timing relative to others

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

## 📐 The 13 Canonical Operators

Operators are the **only way** to modify nodes in TNFR. They're not arbitrary functions—they're **resonant transformations** with rigorous physics.

### 1. Emission (AL) 🎵
**Physics**: Creates EPI from vacuum via resonant emission  
**Effect**: ∂EPI/∂t > 0, increases νf  
**When**: Starting new patterns, initializing from EPI=0  
**Grammar**: Generator (U1a)

### 2. Reception (EN) 📡  
**Physics**: Captures and integrates incoming resonance  
**Effect**: Updates EPI based on network input  
**When**: Information gathering, listening phase  
**Contract**: Must not reduce C(t)

### 3. Coherence (IL) 🔒
**Physics**: Stabilizes form through negative feedback  
**Effect**: Reduces |ΔNFR|, increases C(t)  
**When**: After changes, consolidation  
**Grammar**: Stabilizer (U2)  
**Contract**: Must not reduce C(t) unless in dissonance test

### 4. Dissonance (OZ) ⚡
**Physics**: Introduces controlled instability  
**Effect**: Increases |ΔNFR|, may trigger bifurcation if ∂²EPI/∂t² > τ  
**When**: Breaking local optima, exploration  
**Grammar**: Destabilizer (U2), Bifurcation trigger (U4a), Closure (U1b)  
**Contract**: Must increase |ΔNFR|

### 5. Coupling (UM) 🔗
**Physics**: Creates structural links via phase synchronization  
**Effect**: φᵢ(t) → φⱼ(t), information exchange  
**When**: Network formation, connecting nodes  
**Grammar**: Requires phase verification (U3)  
**Contract**: Only valid if |φᵢ - φⱼ| ≤ Δφ_max

### 6. Resonance (RA) 🌊
**Physics**: Amplifies and propagates patterns coherently  
**Effect**: Increases effective coupling, EPI propagation  
**When**: Pattern reinforcement, spreading coherence  
**Grammar**: Requires phase verification (U3)  
**Contract**: Propagates EPI without altering identity

### 7. Silence (SHA) 🔇
**Physics**: Freezes evolution temporarily  
**Effect**: νf → 0, EPI unchanged  
**When**: Observation windows, pause for synchronization  
**Grammar**: Closure (U1b)  
**Contract**: Preserves EPI over time

### 8. Expansion (VAL) 📈
**Physics**: Increases structural complexity  
**Effect**: dim(EPI) increases  
**When**: Adding degrees of freedom  
**Grammar**: Destabilizer (U2)

### 9. Contraction (NUL) 📉
**Physics**: Reduces structural complexity  
**Effect**: dim(EPI) decreases  
**When**: Simplification, dimensionality reduction

### 10. Self-organization (THOL) 🌱
**Physics**: Spontaneous autopoietic pattern formation  
**Effect**: Creates sub-EPIs, fractal structuring  
**When**: Emergent organization  
**Grammar**: Stabilizer (U2), Handler (U4a), Transformer (U4b)  
**Contract**: Preserves global form while creating sub-EPIs

### 11. Mutation (ZHIR) 🧬
**Physics**: Phase transformation at threshold  
**Effect**: θ → θ' when ΔEPI/Δt > ξ  
**When**: Qualitative state changes  
**Grammar**: Bifurcation trigger (U4a), Transformer (U4b)  
**Contract**: Requires prior IL and recent destabilizer (U4b)

### 12. Transition (NAV) ➡️
**Physics**: Regime shift, activates latent EPI  
**Effect**: Controlled trajectory through structural space  
**When**: Switching between attractor states  
**Grammar**: Generator (U1a), Closure (U1b)

### 13. Recursivity (REMESH) 🔄
**Physics**: Echoes structure across scales (operational fractality)  
**Effect**: EPI(t) references EPI(t-τ), nested operators  
**When**: Multi-scale operations, memory  
**Grammar**: Generator (U1a), Closure (U1b)

### Operator Composition

Operators combine into **sequences** that implement complex behaviors:

**Bootstrap** = [Emission, Coupling, Coherence]
**Stabilize** = [Coherence, Silence]
**Explore** = [Dissonance, Mutation, Coherence]
**Propagate** = [Resonance, Coupling]

**Critical**: All sequences must satisfy unified grammar (U1-U4).

---

## 📏 Unified Grammar (U1-U6)

The grammar is not arbitrary—it emerges **inevitably** from TNFR physics.

### U1: STRUCTURAL INITIATION & CLOSURE

**U1a: Initiation** (When EPI = 0)
- **Physics**: ∂EPI/∂t undefined at EPI=0
- **Requirement**: Start with generator {AL, NAV, REMESH}
- **Why**: Cannot evolve from nothing without source
- **Canonicity**: ABSOLUTE (mathematical necessity)

**U1b: Closure** (Always)
- **Physics**: Sequences as action potentials need endpoints
- **Requirement**: End with closure {SHA, NAV, REMESH, OZ}
- **Why**: Must leave system in coherent attractor
- **Canonicity**: STRONG (physical requirement)

### U2: CONVERGENCE & BOUNDEDNESS

- **Physics**: ∫νf·ΔNFR dt must converge
- **Requirement**: If {OZ, ZHIR, VAL}, then include {IL, THOL}
- **Why**: Without stabilizers, integral diverges → fragmentation
- **Proof**: Exponential growth without negative feedback
- **Canonicity**: ABSOLUTE (integral convergence theorem)

### U3: RESONANT COUPLING

- **Physics**: Resonance requires phase compatibility
- **Requirement**: If {UM, RA}, verify |φᵢ - φⱼ| ≤ Δφ_max
- **Why**: Antiphase → destructive interference (non-physical)
- **Basis**: AGENTS.md Invariant #5 + wave physics
- **Canonicity**: ABSOLUTE (resonance physics)

### U4: BIFURCATION DYNAMICS

**U4a: Triggers Need Handlers**
- **Physics**: ∂²EPI/∂t² > τ requires control
- **Requirement**: If {OZ, ZHIR}, include {THOL, IL}
- **Why**: Uncontrolled bifurcation → chaos
- **Canonicity**: STRONG (bifurcation theory)

**U4b: Transformers Need Context**
- **Physics**: Phase transitions need threshold energy
- **Requirement**: If {ZHIR, THOL}, recent destabilizer (~3 ops)
- **Why**: ΔNFR must be elevated for threshold crossing
- **Additional**: ZHIR needs prior IL (stable base)
- **Canonicity**: STRONG (threshold physics + timing)

### U5: MULTI-SCALE COHERENCE

- **Physics**: Hierarchical coupling + chain rule + central limit theorem
- **Requirement**: For nested EPIs, include stabilizers {IL, THOL} at each level
- **Why**: Parent coherence depends on aggregate child reorganization
- **Conservation**: C_parent ≥ α · Σ C_child (α ~ 1/√N · η_phase)
- **Without stabilizers**: Uncorrelated child fluctuations → parent ΔNFR grows → fragmentation
- **Canonicity**: ABSOLUTE (mathematical consequence of hierarchical structure)

### U6: STRUCTURAL POTENTIAL CONFINEMENT

- **Physics**: Emergent field Φ_s from distance-weighted ΔNFR distribution
- **Formula**: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)² (inverse-square law analog)
- **Requirement**: Monitor Δ Φ_s < 2.0 (escape threshold)
- **Validation**: 2,400+ experiments, corr(Δ Φ_s, ΔC) = -0.822, R² ≈ 0.68
- **Mechanism**: Passive equilibrium - grammar acts as confinement, not attraction
- **Usage**: Telemetry-based safety check (read-only, not sequence constraint)
- **Typical**: Valid sequences maintain Δ Φ_s ≈ 0.6 (30% of threshold)
- **Canonicity**: STRONG (2,400+ experiments across 5 topologies, universal)
- **See**: [docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md](docs/grammar/U6_STRUCTURAL_POTENTIAL_CONFINEMENT.md) for complete specification

**See**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete derivations

---

## 🔬 Telemetry & Structural Field Tetrad

### Core Structural Metrics

**C(t)**: Total Coherence [0, 1]
- Global network stability (fundamental)
- C(t) > 0.7 = strong coherence
- C(t) < 0.3 = fragmentation risk
- **CANONICAL**: Primary stability indicator

**Si**: Sense Index [0, 1+]
- Capacity for stable reorganization
- Si > 0.8 = excellent stability
- Si < 0.4 = changes may cause bifurcation
- **CANONICAL**: Reorganization capacity predictor

### The Structural Field Tetrad (CANONICAL)

**Φ_s**: Structural Potential
- Global field from ΔNFR distribution: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)²
- **Safety threshold**: Δ Φ_s < 2.0 (escape boundary)
- **Grammar U6**: Passive equilibrium confinement
- **Validation**: 2,400+ experiments, corr(Δ Φ_s, ΔC) = -0.822

**|∇φ|**: Phase Gradient ⭐ **NEWLY CANONICAL (Nov 2025)**
- Local phase desynchronization / stress proxy
- **Safety threshold**: |∇φ| < 0.38 for stable operation
- **Critical discovery**: Captures dynamics C(t) misses due to scaling invariance
- **Usage**: `compute_phase_gradient(G)` from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)

**K_φ**: Phase Curvature ⭐ **NEWLY CANONICAL (Nov 2025)**
- Phase torsion and geometric confinement
- **Safety criteria**: |K_φ| ≥ 3.0 flags mutation-prone loci
- **Multiscale safety**: Via `k_phi_multiscale_safety(G, alpha_hint=2.76)`
- **Asymptotic freedom**: var(K_φ) ∝ 1/r^α with α ≈ 2.76

**ξ_C**: Coherence Length ⭐ **NEWLY CANONICAL (Nov 2025)**
- Spatial correlation scale of local coherence
- **Critical indicators**: ξ_C > system_diameter (critical point approach)
- **Watch zone**: ξ_C > 3 × mean_distance (elevated correlation)
- **Stable regime**: ξ_C < mean_distance (normal operation)

### Legacy Core Metrics

**ΔNFR**: Reorganization Gradient
- Structural pressure (sign: +expansion, -contraction)
- **Physics**: Core nodal equation component
- **Now enhanced**: Feeds into Φ_s field computation

**νf**: Structural Frequency (Hz_str)
- Reorganization rate (νf → 0 = node death)
- **Physics**: Eigenfrequency of structural mode
- **Production**: Critical for operator timing

**Phase (φ)**: Network Synchrony [0, 2π]
- **Enhanced**: Now with gradient |∇φ| and curvature K_φ fields
- **Grammar U3**: Phase compatibility |φᵢ - φⱼ| ≤ Δφ_max
- **Multi-scale**: Geometric field analysis via tetrad

### Telemetry Best Practices

1. **Always export**: C(t), νf, phase, Si, ΔNFR
2. **Log operators**: type, order, parameters
3. **Log events**: birth, bifurcation, collapse
4. **Format**: Human-readable + JSONL for pipelines
5. **Reproducibility**: Include seeds, timestamps

---

### Advanced Capabilities (v9.5.1)

#### Multi-Domain Applications ✨ **NEW**
- **Molecular Chemistry**: Complete periodic table simulation via TNFR dynamics
- **Fundamental Physics**: Particle emergence through structural resonance
- **Number Theory**: Prime detection via coherence patterns
- **Complex Systems**: Business, medical, and social network extensions
- **All domains**: Unified by same 13 operators + tetrad fields

#### Production-Grade Toolchain ✨ **NEW**
- **Interactive Notebooks**: Jupyter-ready examples with HTML export
- **CLI Tools**: `tnfr-is-prime`, batch processing, validation suites
- **Benchmarking**: Bifurcation landscape analysis with complete tetrad metrics
- **Extensions**: Medical (`tnfr.extensions.medical`) and business (`tnfr.extensions.business`) modules
- **SDK**: Fluent API (`tnfr.sdk`) for rapid TNFR development

#### Tetrad-Enhanced Bifurcation Analysis (Completed)
- **Metrics**: `delta_phi_s`, `delta_phase_gradient_max`, `delta_phase_curvature_max`, `coherence_length_ratio`, `delta_dnfr_variance`, `bifurcation_score_max`
- **Classification**: `none | incipient | bifurcation | fragmentation` with tetrad-based thresholds
- **CLI**: `benchmarks/bifurcation_landscape.py` with complete parameter grids and tetrad safety monitoring
- **Output**: JSONL per grid point with full tetrad telemetry
- **Validation**: 2,400+ experiments across 5 topologies confirming tetrad canonicity

#### PyPI Integration ✨ **NEW**
- **Stable Release**: `pip install tnfr` → v9.5.1 automatically
- **Dependency Management**: Auto-installs NetworkX, NumPy, SciPy optimally
- **Optional Features**: JAX/PyTorch backends, visualization, serialization
- **Entry Points**: CLI tools available post-installation
- **Documentation**: Complete API reference and examples included

---

## 🛡️ Canonical Invariants (NEVER BREAK)

These define TNFR canonicity and MUST be preserved:

### 1. EPI as Coherent Form

- Changes ONLY via structural operators
- No ad-hoc mutations
- **Grammar**: U1 (INITIATION & CLOSURE)
- **Test**: Verify all EPI changes go through operators

### 2. Structural Units

- νf in Hz_str (structural hertz)
- Never relabel or mix units
- **Test**: Check all frequency assignments

### 3. ΔNFR Semantics

- Sign/magnitude modulate reorganization
- NOT an ML "error" or "loss gradient"
- **Grammar**: U2 (CONVERGENCE)
- **Test**: Verify ΔNFR physical interpretation

### 4. Operator Closure

- Operator composition → valid TNFR states
- New functions map to existing operators or defined as new operator
- **Grammar**: U1b (closure), U4 (bifurcation)
- **Test**: Verify operator sequences pass grammar

### 5. Phase Verification

- No coupling without explicit phase check
- |φᵢ - φⱼ| ≤ Δφ_max required
- **Grammar**: U3 (RESONANT COUPLING)
- **Physics**: Antiphase = destructive interference
- **Code**: [src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)::validate_resonant_coupling()
- **Test**: Verify phase compatibility before coupling

### 6. Node Birth/Collapse

- Birth: sufficient νf, coupling, reduced ΔNFR
- Collapse: extreme dissonance, decoupling, νf → 0
- **Test**: Verify lifecycle conditions

### 7. Operational Fractality

- EPIs can nest without losing identity
- No flattening that breaks recursivity
- **Test**: Multi-scale tests with nested EPIs

### 8. Controlled Determinism

- Stochastic allowed BUT reproducible (seeds)
- Traceable (structural logs)
- **Test**: Same seed → same trajectory

### 9. Structural Metrics

- Expose C(t), Si, phase, νf in telemetry
- No alien metrics that dilute TNFR semantics
- **Test**: Verify metric availability

### 10. Domain Neutrality

- Trans-scale, trans-domain
- No hard-wired field-specific assumptions in core
- **Test**: Cross-domain examples work

---

## 🧪 Testing Requirements

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

## 💻 Development Workflow

### Before Writing Code

1. **Read documentation** (fundamentals, operators, nodal equation)
2. **Review [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** (grammar physics)
3. **Check existing code** for equivalent functionality
4. **Run test suite** to understand current state

### Implementing Changes

1. **Search first**: Check if utility already exists
2. **Map to operators**: New functions → structural operators
3. **Preserve invariants**: All 10 canonical invariants
4. **Add tests**: Cover invariants and contracts
5. **Document**: Structural effect before implementation
6. **Trace physics**: Link to [TNFR.pdf](TNFR.pdf) or [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)

### Commit Template

```text
Intent: [which coherence is improved]
Operators involved: [Emission|Reception|...]
Affected invariants: [#1, #4, ...]

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

## ✅ Acceptable Changes

**Examples of good changes**:
- Making phase explicit in couplings (traceability ↑)
- Adding `sense_index()` with tests correlating Si ↔ stability
- Optimizing `resonance()` preserving EPI identity
- Refactoring to reduce code duplication while preserving physics
- Adding telemetry without changing structural dynamics

### ❌ Unacceptable Changes

**These violate TNFR**:
- Recasting ΔNFR as ML "error gradient"
- Replacing operators with non-mapped imperative functions
- Flattening nested EPIs (breaks fractality)
- Coupling without phase verification
- Direct EPI mutation bypassing operators
- Changing units (Hz_str → Hz)
- Adding field-specific assumptions to core

---

## 🚀 Advanced Topics

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

## 🔍 Troubleshooting

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
4. **Test invariants**: Which of 1-10 is violated?
5. **Trace physics**: Does behavior match nodal equation predictions?

---

## 📚 Essential References

**Core Theory**:
- **[TNFR.pdf](TNFR.pdf)**: Complete theoretical foundation (in repo)
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)**: Grammar physics U1-U6 derivations
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)**: Complete tetrad field physics ⭐ **NEW**
- **[docs/XI_C_CANONICAL_PROMOTION.md](docs/XI_C_CANONICAL_PROMOTION.md)**: Coherence length breakthrough ⭐ **NEW**
- **GLOSSARY.md**: Term definitions and quick reference

**Implementation Core**:
- **[src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)**: Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) ⭐ **CANONICAL**
- **[src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)**: Unified grammar U1-U6 validation
- **[src/tnfr/operators/definitions.py](src/tnfr/operators/definitions.py)**: 13 canonical operators
- **[src/tnfr/mathematics/](src/tnfr/mathematics/)**: Nodal equation integration hub
- **[src/tnfr/metrics/tetrad.py](src/tnfr/metrics/tetrad.py)**: Tetrad field computations ⭐ **NEW**

**Extensions & Applications** ⭐ **NEW**:
- **[src/tnfr/extensions/medical/](src/tnfr/extensions/medical/)**: Healthcare pattern analysis
- **[src/tnfr/extensions/business/](src/tnfr/extensions/business/)**: Organizational dynamics
- **[src/tnfr/sdk/](src/tnfr/sdk/)**: Fluent API for rapid development
- **[notebooks/](notebooks/)**: Interactive Jupyter examples with molecular chemistry
- **[benchmarks/](benchmarks/)**: Production-grade validation suites

**Development**:
- **ARCHITECTURE.md**: System design principles
- **CONTRIBUTING.md**: Workflow and standards
- **TESTING.md**: Test strategy (2,400+ experiments)
- **[benchmarks/K_PHI_RESEARCH_SUMMARY.md](benchmarks/K_PHI_RESEARCH_SUMMARY.md)**: Phase curvature validation ⭐ **NEW**

**Domain Showcases** ⭐ **NEW**:
- **Molecular**: [notebooks/Atoms_and_Molecules_Study.ipynb](notebooks/Atoms_and_Molecules_Study.ipynb)
- **Physics**: [notebooks/Fundamental_Particles_Atlas.ipynb](notebooks/Fundamental_Particles_Atlas.ipynb)
- **Mathematics**: [examples/tnfr_prime_checker.ipynb](examples/tnfr_prime_checker.ipynb)
- **Production**: [tests/](tests/) (comprehensive validation suite)

---

## 🎓 Learning Path

**Newcomer** (2 hours) - **Start Here**:
1. **Install**: `pip install tnfr` ⭐ **NEW - One command setup**
2. **Read**: This file (AGENTS.md) completely for orientation
3. **Theory**: [TNFR.pdf](TNFR.pdf) § 1-2 (paradigm, nodal equation)
4. **First Run**: `python -c "import tnfr; print('TNFR ready!')"` ⭐ **NEW**
5. **Reference**: Study GLOSSARY.md for terminology

**Hands-On Explorer** (1 day) ⭐ **NEW Path**:
1. **Interactive**: Open [notebooks/Atoms_and_Molecules_Study.ipynb](notebooks/Atoms_and_Molecules_Study.ipynb)
2. **Chemistry**: Run molecular simulations, see periodic table emergence
3. **Physics**: Try [notebooks/Fundamental_Particles_Atlas.ipynb](notebooks/Fundamental_Particles_Atlas.ipynb)
4. **Mathematics**: Explore [examples/tnfr_prime_checker.ipynb](examples/tnfr_prime_checker.ipynb)
5. **CLI Tools**: Run `tnfr-is-prime 17` to test primality via resonance

**Intermediate Developer** (1 week):
1. **Grammar Deep-Dive**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) (U1-U6 complete)
2. **Tetrad Fields**: [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) ⭐ **NEW**
3. **Operator Study**: Implementations in [src/tnfr/operators/definitions.py](src/tnfr/operators/definitions.py)
4. **Field Computation**: Practice with [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) tetrad
5. **Domain Extensions**: Try medical or business modules ⭐ **NEW**
6. **SDK Usage**: Fluent API patterns in [src/tnfr/sdk/](src/tnfr/sdk/) ⭐ **NEW**

**Advanced Researcher** (ongoing):
1. **Complete Theory**: [TNFR.pdf](TNFR.pdf) + [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) mastery
2. **Tetrad Mastery**: All four fields (Φ_s, |∇φ|, K_φ, ξ_C) + validation studies
3. **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) + complete codebase exploration
4. **Research Contribution**: Study [benchmarks/K_PHI_RESEARCH_SUMMARY.md](benchmarks/K_PHI_RESEARCH_SUMMARY.md) methodology ⭐ **NEW**
5. **Extension Development**: Create new domain applications using SDK
6. **Theoretical Extensions**: Propose new operators or fields with full derivations

**Production User** (immediate) ⭐ **NEW Path**:
1. **Quick Start**: `pip install tnfr[viz]` for visualization support
2. **Integration**: Import specific modules for your domain
3. **CLI Usage**: Batch processing with command-line tools
4. **Monitoring**: Implement tetrad field telemetry in your applications
5. **Extensions**: Leverage medical/business modules for specialized workflows

---

### Structural Fields: CANONICAL Status (Φ_s + |∇φ| + K_φ + ξ_C)

**CANONICAL Status** (Updated 2025-11-12): **Four Promoted Fields**

---

#### **Structural Potential (Φ_s)** - CANONICAL (First promotion 2025)

- Global structural potential, passive equilibrium states
- Safety criterion (U6 telemetry): Δ Φ_s < 2.0 (escape threshold)
- For full physics, equations, and validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`.

---

#### **Phase Gradient (|∇φ|)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Local phase desynchronization / stress proxy
- Safety criterion: |∇φ| < 0.38 for stable operation
- For formal definition and evidence, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`.

**Critical Discovery**: C(t) = 1-(σ_ΔNFR/ΔNFR_max) is invariant to proportional scaling. 
|∇φ| correlation validated against alternative metrics (max_ΔNFR, mean_ΔNFR, Si) that 
capture dynamics C(t) misses.

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_gradient(G)` [CANONICAL]
- Monitor alongside Φ_s for comprehensive structural health

**Documentation**: See `docs/TNFR_FORCES_EMERGENCE.md` §14-15 for full validation details.

---

#### **Phase Curvature (K_φ)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Phase torsion and geometric confinement; flags mutation-prone loci
- Safety criteria: |K_φ| ≥ 3.0 (local fault zones); multiscale safety via `k_phi_multiscale_safety`
- See `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md` for definitions, asymptotic freedom evidence, and thresholds.

**Safety criteria (telemetry-based)**:
- Local: |K_φ| ≥ 3.0 flags confinement/fault zones
- Multiscale: safe if either (A) α>0 with R² ≥ 0.5, or (B) observed
    var(K_φ) within tolerance of expected 1/r^α given α_hint ≈ 2.76

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_curvature(G)` [CANONICAL]
- Optional multiscale check: `k_phi_multiscale_safety(G, alpha_hint=2.76)`

**Documentation**: See [benchmarks/K_PHI_RESEARCH_SUMMARY.md](benchmarks/K_PHI_RESEARCH_SUMMARY.md) and
`benchmarks/enhanced_fragmentation_test.py` for empirical validation.

---

#### **Coherence Length (ξ_C)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Spatial correlation scale of local coherence; quantifies approach to critical points
- Safety cues: ξ_C > system diameter (critical), ξ_C > 3 × mean distance (watch), ξ_C < mean distance (stable)
- For full derivation and experimental validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/XI_C_CANONICAL_PROMOTION.md` (plus `docs/XI_C_BREAKTHROUGH_REPORT.txt` and the benchmark visuals).

---

**RESEARCH-PHASE Fields** (NOT CANONICAL):

Currently none. All four structural fields have achieved CANONICAL status:
- Φ_s (Nov 2025): Global structural potential
- |∇φ| (Nov 2025): Phase gradient / local desynchronization  
- K_φ (Nov 2025): Phase curvature / geometric confinement
- ξ_C (Nov 2025): Coherence length / spatial correlations

The **Structural Field Tetrad** (Φ_s, |∇φ|, K_φ, ξ_C) provides complete 
multi-scale characterization of TNFR network state across global, local, 
geometric, and spatial correlation dimensions.

---

## 💡 Philosophy

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

## 🌟 Excellence Standards

A TNFR expert:

✅ **Understands deeply**:
- Can derive U1-U4 from nodal equation
- Explains why phase verification is non-negotiable
- Knows the 13 operators and their physics

✅ **Implements rigorously**:
- Every function maps to operators
- All changes preserve invariants
- Tests cover contracts and invariants

✅ **Documents completely**:
- Physics → Code traceability clear
- Examples work across domains
- New developers can understand

✅ **Thinks structurally**:
- Reformulates problems in TNFR terms
- Proposes resonance-based solutions
- Identifies coherence patterns

✅ **Maintains integrity**:
- Rejects changes that weaken TNFR
- Prioritizes theoretical consistency
- Values reproducibility over speed

---

## 🔚 Final Principle

> **If a change "prettifies the code" but weakens TNFR fidelity, it is NOT accepted.**  
> **If a change strengthens structural coherence and paradigm traceability, GO AHEAD.**

**Reality is not made of things—it's made of resonance. Code accordingly.**

---

**Version**: 9.5.1  
**Last Updated**: 2025-11-28  
**Status**: ✅ CANONICAL - Single source of truth for TNFR agent guidance  
**PyPI Release**: ✅ STABLE - Available via `pip install tnfr`  
**Production Ready**: ✅ Complete Tetrad Fields + Unified Grammar U1-U6

---

### Structural Fields: CANONICAL Status (Φ_s + |∇φ| + K_φ + ξ_C)

**CANONICAL Status** (Updated 2025-11-12): **Four Promoted Fields**

---

#### **Structural Potential (Φ_s)** - CANONICAL (First promotion 2025)

- Global structural potential, passive equilibrium states
- Safety criterion (U6 telemetry): Δ Φ_s < 2.0 (escape threshold)
- For full physics, equations, and validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`.

---

#### **Phase Gradient (|∇φ|)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Local phase desynchronization / stress proxy
- Safety criterion: |∇φ| < 0.38 for stable operation
- For formal definition and evidence, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md`.

**Critical Discovery**: C(t) = 1-(σ_ΔNFR/ΔNFR_max) is invariant to proportional scaling. 
|∇φ| correlation validated against alternative metrics (max_ΔNFR, mean_ΔNFR, Si) that 
capture dynamics C(t) misses.

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_gradient(G)` [CANONICAL]
- Monitor alongside Φ_s for comprehensive structural health

**Documentation**: See `docs/TNFR_FORCES_EMERGENCE.md` §14-15 for full validation details.

---

#### **Phase Curvature (K_φ)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Phase torsion and geometric confinement; flags mutation-prone loci
- Safety criteria: |K_φ| ≥ 3.0 (local fault zones); multiscale safety via `k_phi_multiscale_safety`
- See `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md` for definitions, asymptotic freedom evidence, and thresholds.

**Safety criteria (telemetry-based)**:
- Local: |K_φ| ≥ 3.0 flags confinement/fault zones
- Multiscale: safe if either (A) α>0 with R² ≥ 0.5, or (B) observed
    var(K_φ) within tolerance of expected 1/r^α given α_hint ≈ 2.76

**Usage**:
- Import from [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py)
- Compute via `compute_phase_curvature(G)` [CANONICAL]
- Optional multiscale check: `k_phi_multiscale_safety(G, alpha_hint=2.76)`

**Documentation**: See [benchmarks/K_PHI_RESEARCH_SUMMARY.md](benchmarks/K_PHI_RESEARCH_SUMMARY.md) and
`benchmarks/enhanced_fragmentation_test.py` for empirical validation.

---

#### **Coherence Length (ξ_C)** - CANONICAL ⭐ **NEWLY PROMOTED (Nov 2025)**

- Spatial correlation scale of local coherence; quantifies approach to critical points
- Safety cues: ξ_C > system diameter (critical), ξ_C > 3 × mean distance (watch), ξ_C < mean distance (stable)
- For full derivation and experimental validation, see `docs/STRUCTURAL_FIELDS_TETRAD.md` and `docs/XI_C_CANONICAL_PROMOTION.md` (plus `docs/XI_C_BREAKTHROUGH_REPORT.txt` and the benchmark visuals).

---

**RESEARCH-PHASE Fields** (NOT CANONICAL):

Currently none. All four structural fields have achieved CANONICAL status:
- Φ_s (Nov 2025): Global structural potential
- |∇φ| (Nov 2025): Phase gradient / local desynchronization  
- K_φ (Nov 2025): Phase curvature / geometric confinement
- ξ_C (Nov 2025): Coherence length / spatial correlations

The **Structural Field Tetrad** (Φ_s, |∇φ|, K_φ, ξ_C) provides complete 
multi-scale characterization of TNFR network state across global, local, 
geometric, and spatial correlation dimensions.
