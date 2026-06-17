# TNFR API Contracts and Structural Invariants

**Status**: ✅ **ACTIVE** - Complete operator specifications  
**Version**: 3.0.0 (Operator contracts centralized in `operator_contracts.py`)  
**Last Updated**: June 17, 2026  

## Purpose

This document formalizes the **structural invariants** and **API contracts** for key TNFR functions, inspired by the 13 canonical structural operators. Each contract specifies preconditions, postconditions, and the structural effects on the Primary Information Structure (EPI), structural frequency (νf), phase (θ), and internal reorganization operator (ΔNFR).

### 🎯 What This Document Provides

1. **Formal Operator Specifications**: Mathematical contracts for all 13 canonical operators
2. **Implementation Guidelines**: How to implement operators while preserving TNFR physics
3. **Optimization Patterns**: Common patterns for improving operator sequences
4. **Debugging Reference**: Contract violations and how to fix them
5. **Extension Framework**: How to add new operators while maintaining canonicity

---

## Module Organization

### Core Modules

- **`tnfr.operators`**: Structural operators implementing TNFR canonical grammar
- **`tnfr.utils`**: Utility functions for caching, graph operations, and data normalization
- **`tnfr.dynamics`**: Evolution and dynamics computation
- **`tnfr.structural`**: Node creation and network initialization

---

## Structural Operators

> **Single source of truth.** The canonical contract of every operator is
> owned by [`tnfr.operators.operator_contracts`](../src/tnfr/operators/operator_contracts.py)
> (`OPERATOR_CONTRACTS`), derived from the nodal-equation ground-truth effect
> of each glyph plus TNFR.pdf §2.2.1. The proactive audit
> (`tnfr.physics.integrity.audit_operator_contracts`), the reactive monitor
> (`POSTCONDITIONS`), and the introspection metadata all derive from / are
> pinned to this spec. This table is a generated view; do not hand-edit
> contracts here — change the spec.

At the public level the **English structural-operator name** is canonical
(Emission, Reception, …); the glyph code (AL, EN, …) is the internal symbol.

### Canonical contract table

Each operator's primary effect lands on exactly one **nodal-equation channel**
(`∂EPI/∂t = νf · ΔNFR`). This channel partition simultaneously *is* the
dual-lever (examples 37/130), the tetrad driver (example 39), and the
number-theory grading (example 147). A second, orthogonal **scale** axis
(grammar rule U5, operational fractality) separates the twelve node-scale
operators from the single network-scale operator, Recursivity (REMESH).

| # | Operator | Glyph | Channel | Scale | Direction | Postcondition |
|---|----------|-------|---------|-------|-----------|---------------|
| 1 | **Emission** | AL | EPI (form) | node | increase | EPI not decreased (∂EPI/∂t ≥ 0) |
| 2 | **Reception** | EN | EPI (form) | node | reorganize | C(t) not decreased (coherent integration) |
| 3 | **Resonance** | RA | EPI (form) | node | reorganize | EPI structural identity (sign/kind) preserved |
| 4 | **Silence** | SHA | νf (capacity) | node | decrease | νf not increased (freeze) |
| 5 | **Expansion** | VAL | νf (capacity) | node | increase | νf not decreased (capacity added) |
| 6 | **Contraction** | NUL | νf (capacity) | node | decrease | νf not increased (capacity removed) |
| 7 | **Coupling** | UM | θ (phase) | node | reorganize | |ΔNFR| not increased (mutual stabilization) |
| 8 | **Mutation** | ZHIR | θ (phase) | node | transform | θ transformed (θ → θ') |
| 9 | **Coherence** | IL | ΔNFR (pressure) | node | decrease | |ΔNFR| not increased and C(t) not decreased |
| 10 | **Dissonance** | OZ | ΔNFR (pressure) | node | increase | |ΔNFR| not decreased |
| 11 | **SelfOrganization** | THOL | ΔNFR (pressure) | node | increase | C(t) not catastrophic (≥ 90%, global form preserved) |
| 12 | **Transition** | NAV | ΔNFR (pressure) | node | reorganize | state changed (νf, θ, or ΔNFR) |
| 13 | **Recursivity** | REMESH | EPI (form) | network | reorganize | node-level advisory; network effect = EPI mixed toward temporal/multi-scale history |

### Per-operator detail

#### Emission (AL) — `tnfr.operators.definitions.Emission`

- **Purpose**: Activates an EPI from a latent state (founding emission).
- **Nodal channel**: EPI (form) (increase)
- **Scale**: node
- **Postcondition** (measured context: network): EPI not decreased (∂EPI/∂t ≥ 0)
- **Nodal expression**: `A'L ⇒ ∂EPI/∂t > 0, νf ≈ ν₀⁺`
- **Reference**: TNFR.pdf §2.2.1 (1) A'L — Emisión fundacional

#### Reception (EN) — `tnfr.operators.definitions.Reception`

- **Purpose**: Integrates an external emission, reorganizing EPI coherently.
- **Nodal channel**: EPI (form) (reorganize)
- **Scale**: node
- **Postcondition** (measured context: network): C(t) not decreased (coherent integration)
- **Nodal expression**: `E'N ⇒ input coherente → modulación de Wᵢ(t)`
- **Reference**: TNFR.pdf §2.2.1 (2) E'N — Recepción estructural

#### Resonance (RA) — `tnfr.operators.definitions.Resonance`

- **Purpose**: Propagates an EPI across couplings without altering identity.
- **Nodal channel**: EPI (form) (reorganize)
- **Scale**: node
- **Postcondition** (measured context: identity): EPI structural identity (sign/kind) preserved
- **Nodal expression**: `R'A ⇒ propagación de EPI con νf amplificada`
- **Reference**: TNFR.pdf §2.2.1 R'A — Resonancia

#### Silence (SHA) — `tnfr.operators.definitions.Silence`

- **Purpose**: Freezes evolution by driving νf → 0 (structural latency).
- **Nodal channel**: νf (capacity) (decrease)
- **Scale**: node
- **Postcondition** (measured context: network): νf not increased (freeze)
- **Nodal expression**: `SH'A ⇒ νf → 0 ⇒ ∂EPI/∂t → 0`
- **Reference**: TNFR.pdf §2.2.1 SH'A — Silencio

#### Expansion (VAL) — `tnfr.operators.definitions.Expansion`

- **Purpose**: Adds reorganization capacity (νf), raising structural complexity.
- **Nodal channel**: νf (capacity) (increase)
- **Scale**: node
- **Postcondition** (measured context: network): νf not decreased (capacity added)
- **Nodal expression**: `VA'L ⇒ νf ↑ (complejidad estructural)`
- **Reference**: TNFR.pdf §2.2.1 VA'L — Expansión

#### Contraction (NUL) — `tnfr.operators.definitions.Contraction`

- **Purpose**: Removes capacity (νf ↓) and concentrates pressure (ΔNFR ↑).
- **Nodal channel**: νf (capacity) (decrease)
- **Scale**: node
- **Postcondition** (measured context: network): νf not increased (capacity removed)
- **Nodal expression**: `NU'L ⇒ νf ↓, ΔNFR densificada`
- **Reference**: TNFR.pdf §2.2.1 NU'L — Contracción

#### Coupling (UM) — `tnfr.operators.definitions.Coupling`

- **Purpose**: Synchronizes phase with neighbours (φᵢ ≈ φⱼ), reducing pressure.
- **Nodal channel**: θ (phase) (reorganize)
- **Scale**: node
- **Postcondition** (measured context: network): |ΔNFR| not increased (mutual stabilization)
- **Nodal expression**: `U'M ⇒ φᵢ(t) → φⱼ(t) (sincronización de fase)`
- **Reference**: TNFR.pdf §2.2.1 U'M — Acoplamiento

#### Mutation (ZHIR) — `tnfr.operators.definitions.Mutation`

- **Purpose**: Transforms the phase regime θ → θ' at a structural threshold.
- **Nodal channel**: θ (phase) (transform)
- **Scale**: node
- **Postcondition** (measured context: phase): θ transformed (θ → θ')
- **Nodal expression**: `Z'HIR ⇒ θ → θ' cuando ΔEPI/Δt > ξ`
- **Reference**: TNFR.pdf §2.2.1 Z'HIR — Mutación

#### Coherence (IL) — `tnfr.operators.definitions.Coherence`

- **Purpose**: Stabilizes form by reducing |ΔNFR| (negative feedback).
- **Nodal channel**: ΔNFR (pressure) (decrease)
- **Scale**: node
- **Postcondition** (measured context: network): |ΔNFR| not increased and C(t) not decreased
- **Nodal expression**: `I'L ⇒ ∂Wᵢ/∂t → 0, νf = const`
- **Reference**: TNFR.pdf §2.2.1 (3) I'L — Coherencia estructural

#### Dissonance (OZ) — `tnfr.operators.definitions.Dissonance`

- **Purpose**: Injects controlled instability, raising |ΔNFR| (may bifurcate).
- **Nodal channel**: ΔNFR (pressure) (increase)
- **Scale**: node
- **Postcondition** (measured context: node): |ΔNFR| not decreased
- **Nodal expression**: `O'Z ⇒ |ΔNFR| ↑ (puede gatillar ∂²EPI/∂t² > τ)`
- **Reference**: TNFR.pdf §2.2.1 O'Z — Disonancia

#### SelfOrganization (THOL) — `tnfr.operators.definitions.SelfOrganization`

- **Purpose**: Autopoietic structuring: spawns sub-EPIs, preserves global form.
- **Nodal channel**: ΔNFR (pressure) (increase)
- **Scale**: node
- **Postcondition** (measured context: network): C(t) not catastrophic (≥ 90%, global form preserved)
- **Nodal expression**: `T'HOL ⇒ ΔNFR += κ·∂²EPI/∂t² (sub-EPIs)`
- **Reference**: TNFR.pdf §2.2.1 T'HOL — Autoorganización

#### Transition (NAV) — `tnfr.operators.definitions.Transition`

- **Purpose**: Controlled regime shift, retargeting ΔNFR toward a νf-aligned state.
- **Nodal channel**: ΔNFR (pressure) (reorganize)
- **Scale**: node
- **Postcondition** (measured context: state): state changed (νf, θ, or ΔNFR)
- **Nodal expression**: `NA'V ⇒ ΔNFR → νf-aligned (transición de régimen)`
- **Reference**: TNFR.pdf §2.2.1 NA'V — Transición

#### Recursivity (REMESH) — `tnfr.operators.definitions.Recursivity`

- **Purpose**: Echoes the form (EPI) across time and scale (operational fractality, U5).
- **Nodal channel**: EPI (form) (reorganize)
- **Scale**: network — operational fractality (U5)
- **Postcondition** (measured context: advisory): node-level advisory; network effect = EPI mixed toward temporal/multi-scale history
- **Nodal expression**: `RE'MESH ⇒ EPI_new = (1-α)²·EPI(t) + α(1-α)·EPI(t-τ_l) + α·EPI(t-τ_g)`
- **Reference**: TNFR.pdf §2.2.1 RE'MESH — Recursividad

---

## Utility Functions Contracts

### `tnfr.utils.cache.cached_node_list`

**Purpose**: Return cached node list with version tracking.

**Contract**:
- **Preconditions**:
  - `G` implements GraphLike protocol
  - Graph has version tracking enabled
- **Postconditions**:
  - Returns list of nodes consistent with graph version
  - Cache invalidated on graph mutation (version increment)
  - Thread-safe access via locking
- **Invariants**:
  - ✓ Deterministic: same graph version → same node list
  - ✓ Cache hit improves performance without semantic change

---

### `tnfr.utils.graph.mark_dnfr_prep_dirty`

**Purpose**: Invalidate ΔNFR preparation cache.

**Contract**:
- **Preconditions**:
  - `G` has graph metadata accessible
- **Postconditions**:
  - `G.graph["_dnfr_prep_dirty"] = True`
  - Next ΔNFR computation will rebuild cache
- **Invariants**:
  - ✓ Structural consistency: dirty flag triggers recomputation
  - ✓ No data loss on invalidation

---

### `tnfr.utils.numeric.clamp`

**Purpose**: Constrain value within bounds.

**Contract**:
- **Preconditions**:
  - `value`, `min_val`, `max_val` are comparable
  - `min_val ≤ max_val`
- **Postconditions**:
  - `result = max(min_val, min(value, max_val))`
  - `min_val ≤ result ≤ max_val`
- **Invariants**:
  - ✓ Idempotent: `clamp(clamp(x, a, b), a, b) = clamp(x, a, b)`
  - ✓ Deterministic

---

## Canonical Invariants (Global)

These invariants apply across all TNFR operations:

1. **EPI as coherent form**: EPI changes only via structural operators; no ad-hoc mutations
2. **Structural units**: νf expressed in Hz_str (structural hertz)
3. **ΔNFR semantics**: Sign and magnitude modulate reorganization rate
4. **Operator closure**: Composition yields valid TNFR states
5. **Phase check**: Coupling requires explicit phase verification
6. **Node birth/collapse**: Minimal conditions (sufficient νf, coupling, reduced ΔNFR)
7. **Operational fractality**: EPIs nest without losing functional identity
8. **Controlled determinism**: Reproducible with seeds, traceable with structural logs
9. **Structural metrics**: Expose C(t), Si, phase, νf in telemetry
10. **Domain neutrality**: Trans-scale, trans-domain engine

---

## Usage Example

```python
from tnfr.operators.definitions import Emission, Coherence, Coupling
from tnfr.structural import create_nfr, run_sequence

# Create a node with initial conditions
G, node = create_nfr("seed", epi=0.1, vf=1.0, theta=0.0)

# Apply operators respecting structural invariants
sequence = [
    Emission(),    # ✓ Increases EPI, preserves νf, θ, ΔNFR
    Coherence(),   # ✓ Dampens ΔNFR, increases C(t)
    Coupling(),    # ✓ Synchronizes phase with neighbors
]

run_sequence(G, node, sequence)

# Verify invariants
assert G.nodes[node]["epi"] >= 0.1  # Emission increased EPI
assert abs(G.nodes[node]["dnfr"]) < 1.0  # Coherence dampened ΔNFR
```

---

## Testing Contracts

All contracts should be validated by:
1. **Unit tests**: Verify pre/postconditions for each operator
2. **Integration tests**: Ensure operator closure and composition
3. **Property tests**: Use Hypothesis to check invariants across parameter spaces
4. **Regression tests**: Prevent contract violations on refactoring

See `tests/integration/test_additional_critical_paths.py` for examples.

---

## Module Dependency Analysis and Coupling Assessment

### Utils Package Dependency Graph

The `tnfr.utils` package exhibits a well-structured dependency hierarchy with no circular runtime dependencies:

```
init.py (foundational - logging, lazy imports)
  ↓
numeric.py, chunks.py (pure mathematical functions, no TNFR imports)
  ↓
data.py (depends on: numeric, init)
  ↓
graph.py (depends on: types only)
  ↓
io.py (depends on: init)
  ↓
cache.py (depends on: locking, types, init, graph, io)
  ↓
callbacks.py (depends on: constants, locking, init, data, types)
```

### Cross-Module Import Analysis

**Verified No Circular Imports**: Analysis confirmed no runtime circular dependencies. The apparent bidirectional reference between `init.py` and `cache.py` uses `TYPE_CHECKING` guards, preventing runtime circular import issues.

#### Legitimate Cross-Dependencies

1. **`cache.py` → `graph.py`**: 
   - Purpose: ΔNFR preparation state management
   - Functions: `get_graph()`, `mark_dnfr_prep_dirty()`
   - Justification: Cache invalidation must coordinate with graph mutation tracking

2. **`cache.py` → `init.py`**:
   - Purpose: Logging and lazy numpy backend loading
   - Functions: `get_logger()`, `get_numpy()`
   - Justification: Domain-neutral backend selection (INVARIANT #10)

3. **`cache.py` → `io.py`**:
   - Purpose: Deterministic serialization for cache keys
   - Functions: `json_dumps()`
   - Justification: Reproducible cache key generation (INVARIANT #8)

4. **`data.py` → `numeric.py`**:
   - Purpose: Compensated summation for coherence calculations
   - Functions: `kahan_sum_nd()`
   - Justification: Numerical stability for C(t) computation

5. **`callbacks.py` → `data.py`**:
   - Purpose: Callback argument validation
   - Functions: `is_non_string_sequence()`
   - Justification: Type checking for structural event data

**Assessment**: All cross-module imports serve specific structural purposes aligned with TNFR invariants. No unnecessary coupling detected.

### Compatibility Shims

**`callback_utils.py`**: Deprecated compatibility shim that redirects to `utils.callbacks`. This module:
- Emits `DeprecationWarning` on import
- Provides backward compatibility during migration period
- Documented for future removal
- **Recommendation**: Remove in next major version after migration period

### Module Coupling Metrics

| Module | Internal Imports | External TNFR Imports | Coupling Score |
|--------|------------------|----------------------|----------------|
| `numeric.py` | 0 | 0 | **Low** ✓ |
| `chunks.py` | 0 | 0 | **Low** ✓ |
| `init.py` | 1 (cache - TYPE_CHECKING only) | 0 | **Low** ✓ |
| `graph.py` | 0 | 2 (types) | **Low** ✓ |
| `io.py` | 4 (init) | 0 | **Low** ✓ |
| `data.py` | 3 (numeric, init) | 0 | **Moderate** ✓ |
| `callbacks.py` | 2 (init, data) | 3 (constants, locking, types) | **Moderate** ✓ |
| `cache.py` | 5 (graph, init, io) | 2 (locking, types) | **Moderate** ✓ |

**✓ All coupling levels are appropriate for module responsibilities.**

### Linting Results Summary

Flake8 analysis identified only minor style issues:
- Blank lines with whitespace (W293)
- Module imports not at top (E402) - intentional for lazy loading
- Unused TYPE_CHECKING imports (F401) - required for type hints
- Minor spacing issues (E302, E305)

**No structural anti-patterns, circular imports, or dangerous coupling detected.**

### Recommendations

1. **Keep Current Structure**: The dependency hierarchy is clean and well-organized
2. **Remove `callback_utils.py`**: After deprecation period expires
3. **Document Import Rationale**: Comments added where non-obvious

---

## 🚀 Advanced Optimization Patterns

### Pattern 1: Therapeutic Transformation Cycle

**Purpose**: Transform unstable or problematic sequences into stable therapeutic patterns.

**Template**:
```python
# Before: Unstable transformation (health ≈ 0.60)
unstable = [EMISSION, DISSONANCE, MUTATION, COHERENCE]

# After: Full therapeutic cycle (health ≈ 0.89)
therapeutic = [
    EMISSION,           # AL: Initialize therapeutic space
    RECEPTION,          # EN: Gather contextual information
    COHERENCE,          # IL: Establish baseline stability
    DISSONANCE,         # OZ: Controlled destabilization
    SELF_ORGANIZATION,  # THOL: Emergent reorganization
    COHERENCE,          # IL: Consolidate transformation
    SILENCE             # SHA: Integration period
]
```

**Key Improvements**:
- **+0.29 health improvement** through proper stabilization
- **Grammar compliance**: U2 (destabilizer + stabilizer), U1b (proper closure)
- **Pattern recognition**: Canonical therapeutic sequence

### Pattern 2: Enhanced Activation with Amplification

**Purpose**: Upgrade basic activation patterns with resonant amplification.

**Template**:
```python
# Before: Basic activation (health ≈ 0.66)
basic = [EMISSION, RECEPTION, COHERENCE, SILENCE]

# After: Enhanced with resonance (health ≈ 0.85)
enhanced = [
    EMISSION,    # AL: Seed activation
    RECEPTION,   # EN: Information gathering
    COHERENCE,   # IL: Baseline stability
    RESONANCE,   # RA: Amplify coherent patterns
    SILENCE      # SHA: Preserve amplified state
]
```

**Key Improvements**:
- **+0.19 health improvement** through resonant amplification
- **Enhanced pattern**: Basic activation → Activation with amplification
- **Preserved simplicity**: Minimal operator addition

### Pattern 3: Exploration with Controlled Bifurcation

**Purpose**: Enable safe exploration through controlled destabilization and recovery.

**Template**:
```python
controlled_exploration = [
    EMISSION,           # AL: Initialize exploration space
    RECEPTION,          # EN: Gather environmental context
    COHERENCE,          # IL: Establish safety baseline
    EXPANSION,          # VAL: Increase exploration volume
    COHERENCE,          # IL: Stabilize expansion (U2)
    DISSONANCE,         # OZ: First exploration wave
    MUTATION,           # ZHIR: Alternative trajectory (U4b)
    COHERENCE,          # IL: Checkpoint stability
    DISSONANCE,         # OZ: Second exploration wave
    SELF_ORGANIZATION,  # THOL: Emergent structure (U4a)
    COHERENCE,          # IL: Final consolidation
    SILENCE             # SHA: Integration
]
```

**Key Features**:
- **Multi-wave exploration**: Two destabilization cycles
- **Safety checkpoints**: Coherence after each major operation
- **Grammar compliance**: U2, U4a, U4b all satisfied
- **Health target**: > 0.68 (moderate due to exploration complexity)

### Pattern 4: Network Synchronization Protocol

**Purpose**: Establish coherent coupling across network nodes with phase verification.

**Template**:
```python
network_sync = [
    EMISSION,     # AL: Initialize local coherence
    RECEPTION,    # EN: Sense network state
    COHERENCE,    # IL: Local stabilization
    COUPLING,     # UM: Form phase-verified links (U3)
    RESONANCE,    # RA: Network-wide amplification
    COUPLING,     # UM: Strengthen synchronization
    RESONANCE,    # RA: Reinforce coherent patterns
    COHERENCE,    # IL: Network consolidation
    SILENCE       # SHA: Maintain synchrony
]
```

**Critical Requirements**:
- **Phase verification**: Each COUPLING must satisfy |φᵢ - φⱼ| ≤ Δφ_max (U3)
- **Network topology**: Requires connected network with multiple nodes
- **Health target**: > 0.80 for strong network coherence

### Pattern 5: Compression and Essence Extraction

**Purpose**: Simplify complex structures while preserving essential information.

**Template**:
```python
compression = [
    EMISSION,      # AL: Initialize from complex state
    RECEPTION,     # EN: Understand full complexity
    COHERENCE,     # IL: Baseline stability
    EXPANSION,     # VAL: Temporarily increase complexity
    COHERENCE,     # IL: Stabilize expansion (U2)
    CONTRACTION,   # NUL: First compression wave
    COHERENCE,     # IL: Stabilize compression
    CONTRACTION,   # NUL: Second compression wave
    COHERENCE,     # IL: Final consolidation
    SILENCE        # SHA: Preserve essence
]
```

**Key Features**:
- **Expand then compress**: Counter-intuitive but effective pattern
- **Multiple compression waves**: Progressive simplification
- **Essence preservation**: Each contraction followed by stabilization

---

## 🔬 Optimization Strategies

### Strategy 1: Health-Driven Optimization

```python
def optimize_by_health(sequence, target_health=0.80):
    """Iterative health improvement strategy."""
    current = sequence.copy()
    
    while compute_health(current) < target_health:
        # 1. Check grammar violations
        violations = validate_grammar(current)
        if violations:
            current = fix_grammar_violations(current, violations)
            continue
        
        # 2. Apply common improvements
        if needs_stabilization(current):
            current = add_stabilizers(current)
        elif lacks_amplification(current):
            current = add_resonance(current)
        elif missing_closure(current):
            current = fix_closure(current)
        else:
            break  # No obvious improvements
    
    return current
```

### Strategy 2: Pattern-Based Enhancement

```python
def enhance_with_patterns(sequence):
    """Upgrade sequences using canonical patterns."""
    pattern_type = classify_sequence(sequence)
    
    if pattern_type == "basic_activation":
        return upgrade_to_enhanced_activation(sequence)
    elif pattern_type == "unstable_transformation":
        return convert_to_therapeutic(sequence)
    elif pattern_type == "incomplete_exploration":
        return complete_exploration_cycle(sequence)
    else:
        return apply_generic_improvements(sequence)
```

### Strategy 3: Grammar-First Optimization

```python
def grammar_compliant_optimization(sequence):
    """Ensure grammar compliance while optimizing."""
    # 1. Fix U1 violations (initiation/closure)
    if not starts_with_generator(sequence):
        sequence = [EMISSION] + sequence
    if not ends_with_closure(sequence):
        sequence = sequence + [SILENCE]
    
    # 2. Fix U2 violations (convergence/boundedness)
    sequence = balance_destabilizers(sequence)
    
    # 3. Fix U3 violations (resonant coupling)
    sequence = verify_phase_compatibility(sequence)
    
    # 4. Fix U4 violations (bifurcation dynamics)
    sequence = handle_bifurcation_triggers(sequence)
    
    return sequence
```

---

## 📊 Performance Optimization Guidelines

### Memory Optimization

1. **Use inplace operations** when possible: `compute_si(G, inplace=True)`
2. **Enable node caching** for repeated calculations: `TNFR_CACHE_ENABLED=1`
3. **Batch operations** on multiple nodes to amortize overhead
4. **Profile memory usage** with `OperatorMetrics` collection

### Computational Optimization

1. **Backend selection**: Choose appropriate backend for network size
   - **Numpy**: Networks < 100 nodes
   - **Numba**: Networks 100-1000 nodes  
   - **GPU**: Networks > 1000 nodes

2. **Operator composition**: Combine compatible operators to reduce overhead

3. **Sequence caching**: Cache validated sequences to avoid re-validation

### Network Architecture Optimization

1. **Topology awareness**: Choose operators based on network topology
   - **Dense networks**: Favor coupling and resonance
   - **Sparse networks**: Use expansion carefully
   - **Scale-free**: Focus on hub stabilization

2. **Load balancing**: Distribute operations across high-degree nodes

3. **Hierarchical processing**: Use operational fractality for multi-scale networks

---

## 🛠️ Extension Framework

### Adding Custom Operators

To add new operators while maintaining canonicity:

```python
from tnfr.operators import Operator
from tnfr.operators.glyphs import Glyph

class CustomOperator(Operator):
    """Custom operator following TNFR physics."""
    
    name: ClassVar[str] = "custom_name"
    glyph: ClassVar[Glyph] = Glyph.CUSTOM  # Define new glyph
    
    def _validate_preconditions(self, G, node):
        """Validate custom preconditions."""
        # Implement physics-based validation
        pass
    
    def _apply_operation(self, G, node):
        """Apply custom structural transformation."""
        # Must preserve nodal equation: ∂EPI/∂t = νf · ΔNFR
        pass
    
    def _collect_metrics(self, G, node, state_before):
        """Collect custom metrics."""
        # Return metrics dict
        pass
```

### Grammar Integration

New operators must be classified in grammar rules:

- **Generators** (U1a): Can start sequences from EPI=0
- **Closures** (U1b): Can end sequences safely
- **Stabilizers** (U2): Reduce ΔNFR, increase C(t)
- **Destabilizers** (U2): Increase ΔNFR, require stabilizers
- **Coupling-based** (U3): Require phase verification
- **Bifurcation triggers** (U4): Need careful handling

### Testing Requirements

All new operators must pass:

1. **Physics compliance**: Preserve nodal equation
2. **Grammar validation**: Follow U1-U6 rules
3. **Invariant preservation**: Maintain 10 canonical invariants
4. **Health metrics**: Provide predictable health impact
5. **Multi-topology testing**: Work across different network topologies

---

## 📚 Related Documentation

- **[AGENTS.md](../AGENTS.md)**: Complete TNFR theory and canonical invariants
- **[grammar/PHYSICS_VERIFICATION.md](grammar/PHYSICS_VERIFICATION.md)**: U1–U6 grammar derivations
- **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)**: Dissonance-based patterns
- **[examples/01_foundations/04_operator_sequences.py](../examples/01_foundations/04_operator_sequences.py)**: Live operator examples
- **[src/tnfr/operators/](../src/tnfr/operators/)**: Implementation source code
4. **Maintain TYPE_CHECKING Guards**: For forward references without runtime cycles
5. **Continue Modular Design**: New utilities should follow same layered approach

---

## References

- `AGENTS.md`: Agent instructions for maintaining TNFR fidelity
- `TNFR.pdf`: Base paradigm document
- `tnfr.operators`: Operator implementations
- `tnfr.validation`: Grammar and precondition checking
