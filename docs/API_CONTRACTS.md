# TNFR API Contracts and Structural Invariants

**Status**: ✅ **ACTIVE** - Complete operator specifications  
**Version**: 2.1.0 (Enhanced with optimization patterns)  
**Last Updated**: November 29, 2025  

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

### 1. Emission (AL) — `tnfr.operators.definitions.Emission`

**Structural Function**: Seeds coherence by projecting the emission pattern.

**Operator**: AL (Glyph.AL)

**Contract**:
- **Preconditions**:
  - Node has valid EPI, νf, θ, ΔNFR attributes
  - EPI is finite and real-valued
- **Postconditions**:
  - `EPI_new = EPI_old + AL_boost` where `AL_boost > 0`
  - `νf`, `θ`, `ΔNFR` remain unchanged
  - Node coherence `C(t)` increases or remains stable
- **Structural Effect**: Increases Primary Information Structure without altering temporal cadence
- **TNFR Invariants**:
  - ✓ Preserves operator closure: Emission → valid TNFR state
  - ✓ Maintains phase coherence: `θ` unchanged
  - ✓ Conserves structural frequency: `νf` unchanged

**Implementation**: `tnfr.operators._op_AL`

---

### 2. Reception (EN) — `tnfr.operators.definitions.Reception`

**Structural Function**: Stabilizes inbound energy to strengthen receptivity.

**Operator**: EN (Glyph.EN)

**Contract**:
- **Preconditions**:
  - Node has neighbors accessible via `neighbors()` method
  - Each neighbor has valid EPI attribute
- **Postconditions**:
  - `EPI_new = (1 - EN_mix) * EPI_old + EN_mix * EPI_bar` where `EPI_bar` is neighbor mean
  - `0 ≤ EN_mix ≤ 1` (default: 0.25)
  - `epi_kind` updated to dominant neighbor kind if `|EPI_neighbor| > |EPI_node|`
  - `νf`, `θ`, `ΔNFR` remain unchanged
- **Structural Effect**: Harmonizes node EPI with neighborhood field
- **TNFR Invariants**:
  - ✓ Coupling requires phase verification (implicit in neighbor mean)
  - ✓ Propagates EPI without identity loss
  - ✓ Preserves structural frequency

**Implementation**: `tnfr.operators._op_EN`

---

### 3. Coherence (IL) — `tnfr.operators.definitions.Coherence`

**Structural Function**: Reinforces structural alignment by compressing ΔNFR drift.

**Operator**: IL (Glyph.IL)

**Contract**:
- **Preconditions**:
  - Node has ΔNFR attribute (default: 0.0 if missing)
- **Postconditions**:
  - `ΔNFR_new = IL_dnfr_factor * ΔNFR_old` where `0 < IL_dnfr_factor < 1` (default: 0.7)
  - `|ΔNFR_new| ≤ |ΔNFR_old|` (monotonic decrease unless factor > 1)
  - Sign of ΔNFR preserved: `sign(ΔNFR_new) = sign(ΔNFR_old)`
  - `EPI`, `νf`, `θ` remain unchanged
  - Total coherence `C(t)` increases or remains stable
- **Structural Effect**: Dampens internal reorganization, increasing stability
- **TNFR Invariants**:
  - ✓ Coherence application must not reduce `C(t)` (unless in controlled dissonance test)
  - ✓ Preserves operator closure
  - ✓ Maintains phase and frequency integrity

**Implementation**: `tnfr.operators._op_IL`

---

### 4. Dissonance (OZ) — `tnfr.operators.definitions.Dissonance`

**Structural Function**: Injects controlled dissonance to probe structural robustness.

**Operator**: OZ (Glyph.OZ)

**Contract**:
- **Preconditions**:
  - Node has ΔNFR attribute
  - Node has access to graph metadata for noise mode configuration
- **Postconditions**:
  - **Noise mode** (`OZ_NOISE_MODE=True`): `ΔNFR_new = ΔNFR_old + jitter` where `jitter ~ N(0, OZ_SIGMA²)`
  - **Amplification mode** (`OZ_NOISE_MODE=False`): `ΔNFR_new = OZ_dnfr_factor * ΔNFR_old` where `OZ_dnfr_factor > 1` (default: 1.3)
  - `|ΔNFR_new| > |ΔNFR_old|` (typically, increases internal reorganization)
  - `EPI`, `νf`, `θ` remain unchanged
  - May trigger bifurcation if `∂²EPI/∂t² > τ`
- **Structural Effect**: Increases ΔNFR to test bifurcation thresholds
- **TNFR Invariants**:
  - ✓ Dissonance increases `|ΔNFR|`
  - ✓ May trigger bifurcation at configurable threshold
  - ✓ Preserves EPI, νf, θ

**Implementation**: `tnfr.operators._op_OZ`

---

### 5. Coupling (UM) — `tnfr.operators.definitions.Coupling`

**Structural Function**: Synchronizes phase and optionally creates functional links.

**Operator**: UM (Glyph.UM)

**Contract**:
- **Preconditions**:
  - Node has θ (phase) attribute
  - Node has neighbors accessible via `neighbors()` method
  - Each neighbor has θ attribute
- **Postconditions**:
  - `θ_new = θ_old + UM_theta_push * angle_diff(θ_neighbor_mean, θ_old)`
  - `0 ≤ UM_theta_push ≤ 1` (default: 0.25)
  - By default (`UM_FUNCTIONAL_LINKS=True`), may add edges based on:
    - Phase similarity: `1 - |Δθ|/π`
    - EPI similarity: `1 - |EPI_i - EPI_j| / (|EPI_i| + |EPI_j| + ε)`
    - Sense index similarity: `1 - |Si_i - Si_j|`
    - Edge added if `compatibility ≥ UM_COMPAT_THRESHOLD` (default: 0.75)
  - Set `UM_FUNCTIONAL_LINKS=False` to disable link creation (edge case: phase-only sync)
  - `EPI`, `νf`, `ΔNFR` remain unchanged
- **Structural Effect**: Aligns node phase with neighbor mean, enables structural coupling
- **TNFR Invariants**:
  - ✓ No coupling without explicit phase verification
  - ✓ Phase synchrony increases effective coupling
  - ✓ Preserves structural frequency and EPI

**Implementation**: `tnfr.operators._op_UM`

---

### 6. Resonance (RA) — `tnfr.operators.definitions.Resonance`

**Structural Function**: Propagates coherent energy through the network.

**Operator**: RA (Glyph.RA)

**Contract**:
- **Preconditions**:
  - Node has EPI, epi_kind attributes
  - Node has neighbors with EPI attributes
- **Postconditions**:
  - `EPI_new = (1 - RA_epi_diff) * EPI_old + RA_epi_diff * EPI_bar`
  - `0 ≤ RA_epi_diff ≤ 1` (default: 0.15)
  - `epi_kind` updated to dominant neighbor kind if applicable
  - `νf`, `θ`, `ΔNFR` remain unchanged
- **Structural Effect**: Diffuses EPI along existing couplings
- **TNFR Invariants**:
  - ✓ Resonance increases effective connectivity (measured via phase)
  - ✓ Propagates EPI without altering identity
  - ✓ Preserves νf and phase

**Implementation**: `tnfr.operators._op_RA`

---

### 7. Silence (SHA) — `tnfr.operators.definitions.Silence`

**Structural Function**: Temporarily suspends structural evolution by reducing νf.

**Operator**: SHA (Glyph.SHA)

**Contract**:
- **Preconditions**:
  - Node has νf (structural frequency) attribute
- **Postconditions**:
  - `νf_new = SHA_vf_factor * νf_old` where `0 < SHA_vf_factor < 1` (default: 0.9015 = 1 - γ/(π+e))
  - `νf_new < νf_old` (monotonic decrease)
  - `EPI`, `θ`, `ΔNFR` remain unchanged
  - Node enters reduced evolution state (`νf ≈ 0` limit)
- **Structural Effect**: Decelerates temporal cadence without structural change
- **TNFR Invariants**:
  - ✓ Silence freezes evolution (`νf → 0`) without EPI loss
  - ✓ Latency: EPI remains invariant over `t + Δt`
  - ✓ Preserves phase and ΔNFR

**Implementation**: `tnfr.operators._op_SHA`

---

### 8. Expansion (VAL) — `tnfr.operators.definitions.Expansion`

**Structural Function**: Accelerates structural frequency to expand temporal cadence.

**Operator**: VAL (Glyph.VAL)

**Contract**:
- **Preconditions**:
  - Node has νf attribute
- **Postconditions**:
  - `νf_new = VAL_scale * νf_old` where `VAL_scale > 1` (default: 1.0676 = 1 + γ/(π×e))
  - `νf_new > νf_old` (monotonic increase)
  - `EPI`, `θ`, `ΔNFR` remain unchanged
- **Structural Effect**: Increases reorganization rate
- **TNFR Invariants**:
  - ✓ Expansion increases νf
  - ✓ Preserves EPI, phase, ΔNFR

**Implementation**: `tnfr.operators._op_scale` via `_make_scale_op(Glyph.VAL)`

---

### 9. Contraction (NUL) — `tnfr.operators.definitions.Contraction`

**Structural Function**: Decelerates structural frequency to contract temporal cadence.

**Operator**: NUL (Glyph.NUL)

**Contract**:
- **Preconditions**:
  - Node has νf attribute
- **Postconditions**:
  - `νf_new = NUL_scale * νf_old` where `0 < NUL_scale < 1` (default: 0.9015 = 1 - γ/(π+e))
  - `νf_new < νf_old` (monotonic decrease)
  - `EPI`, `θ`, `ΔNFR` remain unchanged
- **Structural Effect**: Decreases reorganization rate
- **TNFR Invariants**:
  - ✓ Contraction decreases νf
  - ✓ Preserves EPI, phase, ΔNFR

**Implementation**: `tnfr.operators._op_scale` via `_make_scale_op(Glyph.NUL)`

---

### 10. Self-Organization (THOL) — `tnfr.operators.definitions.SelfOrganization`

**Structural Function**: Injects EPI curvature into ΔNFR to trigger self-organization.

**Operator**: THOL (Glyph.THOL)

**Contract**:
- **Preconditions**:
  - Node has ΔNFR attribute
  - Node has d2EPI (second derivative of EPI) attribute
- **Postconditions**:
  - `ΔNFR_new = ΔNFR_old + THOL_accel * d2EPI`
  - `THOL_accel > 0` (default: 0.10)
  - `EPI`, `νf`, `θ` remain unchanged
  - May create sub-EPIs while preserving global form (operational fractality)
- **Structural Effect**: Accelerates structural rearrangement via curvature
- **TNFR Invariants**:
  - ✓ Self-organization may create sub-EPIs
  - ✓ Preserves global form (operational fractality)
  - ✓ Bifurcation triggered if `∂²EPI/∂t² > τ`

**Implementation**: `tnfr.operators._op_THOL`

---

### 11. Mutation (ZHIR) — `tnfr.operators.definitions.Mutation`

**Structural Function**: Enacts discrete structural transition via phase shift.

**Operator**: ZHIR (Glyph.ZHIR)

**Contract**:
- **Preconditions**:
  - Node has θ (phase) attribute
- **Postconditions**:
  - `θ_new = θ_old + ZHIR_theta_shift`
  - `ZHIR_theta_shift` configurable (default: π/2)
  - Phase change occurs if `ΔEPI/Δt > ξ` (limits configurable)
  - `EPI`, `νf`, `ΔNFR` remain unchanged
- **Structural Effect**: Rotates phase to encode state transition
- **TNFR Invariants**:
  - ✓ Mutation changes θ respecting threshold ξ
  - ✓ Preserves EPI, νf, ΔNFR

**Implementation**: `tnfr.operators._op_ZHIR`

---

### 12. Transition (NAV) — `tnfr.operators.definitions.Transition`

**Structural Function**: Rebalances ΔNFR towards νf-aligned target.

**Operator**: NAV (Glyph.NAV)

**Contract**:
- **Preconditions**:
  - Node has ΔNFR, νf attributes
- **Postconditions**:
  - **Strict mode** (`NAV_STRICT=True`): `ΔNFR_new = vf + jitter`
  - **Default mode**: `ΔNFR_new = (1-η)*ΔNFR_old + η*sign(ΔNFR_old)*vf + jitter`
  - `0 ≤ η ≤ 1` (default: 0.5)
  - Jitter configurable via `NAV_jitter` (default: 0.05)
  - `EPI`, `θ` remain unchanged; `νf` used as reference
- **Structural Effect**: Redirects ΔNFR with optional exploration
- **TNFR Invariants**:
  - ✓ Preserves EPI and phase
  - ✓ Uses νf as reference without direct modification

**Implementation**: `tnfr.operators._op_NAV`

---

### 13. Recursivity (REMESH)

**Structural Function**: Advisory for network-scale remeshing.

**Operator**: REMESH (Glyph.REMESH)

**Contract**:
- **Preconditions**:
  - Node has access to graph metadata
  - Glyph history tracking enabled
- **Postconditions**:
  - Node-level `EPI`, `νf`, `θ`, `ΔNFR` unchanged
  - Advisory recorded in glyph history
  - `_remesh_warn_step` updated to current step
- **Structural Effect**: Signals orchestrator for global remesh when stability conditions met
- **TNFR Invariants**:
  - ✓ Operational fractality: EPIs nest without losing identity
  - ✓ No direct node mutation

**Implementation**: `tnfr.operators._op_REMESH`

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
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Grammar validation error solutions
- **[CANONICAL_OZ_SEQUENCES.md](CANONICAL_OZ_SEQUENCES.md)**: Dissonance-based patterns
- **[examples/04_operator_sequences.py](../examples/04_operator_sequences.py)**: Live operator examples
- **[src/tnfr/operators/](../src/tnfr/operators/)**: Implementation source code
4. **Maintain TYPE_CHECKING Guards**: For forward references without runtime cycles
5. **Continue Modular Design**: New utilities should follow same layered approach

---

## References

- `AGENTS.md`: Agent instructions for maintaining TNFR fidelity
- `TNFR.pdf`: Base paradigm document
- `tnfr.operators`: Operator implementations
- `tnfr.validation`: Grammar and precondition checking
