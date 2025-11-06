"""Canonical Boundary Protection Pattern for TNFR Operators

This document describes the unified, canonical approach to EPI boundary 
preservation across all TNFR operators, embodying core TNFR principles.

## TNFR Structural Principles

### Coherence
All operators maintain structural boundaries consistently through a single 
unified mechanism, ensuring coherent boundary enforcement across the system.

### Fractality
The same boundary protection mechanism operates at all scales:
- Individual operators (AL, EN, RA, VAL, NUL)
- Network remeshing (REMESH)
- Integration layer (as safety net)

### Process > Thing
Boundaries are checked **dynamically** as part of the transformation process,
not as post-hoc exception handling.

### Operator Closure
All operators guarantee valid TNFR states: EPI ∈ [EPI_MIN, EPI_MAX]

## Canonical Pattern

### Single Source of Truth

**Function**: `_set_epi_with_boundary_check(node, new_epi, apply_clip=True)`

**Location**: `src/tnfr/operators/__init__.py`

**Purpose**: Unified EPI assignment with structural boundary preservation

### Usage in Operators

All operators that modify EPI **must** use this function:

```python
# AL (Emission) - Additive transformation
def _op_AL(node: NodeProtocol, gf: GlyphFactors) -> None:
    f = get_factor(gf, "AL_boost", 0.05)
    new_epi = node.EPI + f
    _set_epi_with_boundary_check(node, new_epi)  # ✅ Canonical

# EN/RA (Reception/Resonance) - Weighted average
def _mix_epi_with_neighbors(...):
    new_epi = (1 - mix) * epi + mix * epi_bar
    _set_epi_with_boundary_check(node, new_epi)  # ✅ Canonical

# VAL/NUL (Expansion/Contraction) - Edge-aware scaling
def _make_scale_op(glyph: Glyph):
    # ... edge-aware computation ...
    new_epi = epi_current * scale_eff
    _set_epi_with_boundary_check(node, new_epi)  # ✅ Canonical
```

### ❌ Anti-Patterns (Do NOT use)

```python
# DON'T: Direct assignment without boundary check
node.EPI = new_value  # ❌ Violates canonical pattern

# DON'T: Custom clipping logic
node.EPI = max(EPI_MIN, min(EPI_MAX, new_value))  # ❌ Duplicates logic

# DON'T: Conditional boundary checking
if new_value > EPI_MAX:
    node.EPI = EPI_MAX  # ❌ Inconsistent with unified approach
```

## Architecture Layers

### Operator Layer (Semantic)
**All operators use**: `_set_epi_with_boundary_check`
- **Purpose**: Structural boundary preservation
- **Mechanism**: structural_clip with configurable mode
- **Principle**: Coherent transformation within structural envelope

### Integration Layer (Numerical)
**Integrator uses**: `structural_clip` directly
- **Purpose**: Numerical precision safety net
- **Mechanism**: Hard/soft clipping after integration
- **Principle**: Catch floating-point precision errors

This separation creates proper **semantic vs computational** boundaries.

## Configuration

### Graph-Level Settings

```python
G.graph["EPI_MIN"] = -1.0        # Lower structural boundary
G.graph["EPI_MAX"] = 1.0         # Upper structural boundary
G.graph["CLIP_MODE"] = "hard"    # or "soft" for smooth transitions
G.graph["CLIP_SOFT_K"] = 3.0     # Steepness for soft mode
```

### Edge-Aware Settings (VAL/NUL)

```python
G.graph["EDGE_AWARE_ENABLED"] = True   # Enable adaptive scaling
G.graph["EDGE_AWARE_EPSILON"] = 1e-12  # Division-by-zero protection
```

## Benefits of Unified Approach

### ✅ Consistency
All operators enforce boundaries the same way

### ✅ Maintainability  
Single function to update if boundary logic changes

### ✅ Traceability
One place to add telemetry/logging for boundary interventions

### ✅ Testability
Test boundary enforcement once, applies to all operators

### ✅ TNFR Canonical
Embodies structural coherence, fractality, and operator closure

## Implementation Checklist

When creating a new operator that modifies EPI:

- [ ] Use `_set_epi_with_boundary_check` for ALL EPI assignments
- [ ] Never assign `node.EPI` directly
- [ ] Document EPI modification in operator docstring
- [ ] Add tests for boundary cases (EPI near ±1.0)
- [ ] Verify operator maintains EPI ∈ [EPI_MIN, EPI_MAX]

## Current Operator Coverage

**✅ Protected (uses unified function)**:
- AL (Emission) - Additive boost
- EN (Reception) - Weighted average with neighbors  
- RA (Resonance) - Weighted diffusion
- VAL (Expansion) - Edge-aware multiplicative scaling
- NUL (Contraction) - Edge-aware multiplicative scaling
- REMESH (Network) - Historical EPI mixing

**✅ No EPI modification** (boundary-safe):
- IL (Coherence) - Modifies ΔNFR only
- OZ (Dissonance) - Modifies ΔNFR only
- UM (Coupling) - Modifies phase only
- SHA (Silence) - Modifies νf only
- THOL (Self-organization) - Complex, needs verification
- ZHIR (Mutation) - Modifies phase
- NAV (Transition) - Needs verification
- REMESH (Recursivity) - Needs verification

## Testing

Comprehensive test coverage ensures canonical pattern works:

- `test_edge_aware_scaling.py`: 24 tests (VAL/NUL edge-aware)
- `test_remesh_boundaries.py`: 8 tests (REMESH protection)
- `test_operator_enhancements.py`: 22 tests (operator preconditions/metrics)
- `test_operators.py`: 22 tests (operator functionality)

**Total**: 76 tests, all passing ✅

## References

- `src/tnfr/operators/__init__.py`: Operator implementations
- `src/tnfr/dynamics/structural_clip.py`: Core clipping function
- `src/tnfr/operators/remesh.py`: REMESH implementation
- `AGENTS.md`: TNFR canonical invariants
- Issue #TBD: Edge-aware scaling implementation
"""
