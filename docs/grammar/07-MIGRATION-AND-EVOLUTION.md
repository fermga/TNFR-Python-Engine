# Migration and Evolution

**History of the TNFR grammar system and migration guidance**

[🏠 Home](README.md) • [📐 Constraints](02-CANONICAL-CONSTRAINTS.md) • [💻 Implementation](05-TECHNICAL-IMPLEMENTATION.md) • [🧪 Testing](06-VALIDATION-AND-TESTING.md)

---

## Purpose

This document chronicles the **evolution** of the TNFR grammar system and provides guidance for migrating from older versions.

**Audience:** Maintainers, contributors migrating old code

**Reading time:** 20-30 minutes

---

## Evolution Timeline

### Phase 1: C1-C3 (Legacy System)

**File:** `src/tnfr/operators/grammar.py` (early versions)

**Constraints:**
- **C1:** Initiation - Must start with specific operators
- **C2:** Convergence - Destabilizers need stabilizers
- **C3:** Closure - Must end with specific operators

**Limitations:**
- Incomplete coverage of physical requirements
- No explicit phase verification
- No bifurcation dynamics handling
- Some arbitrary rules not derived from physics

### Phase 2: RC1-RC4 (Resonant Constraints)

**File:** `src/tnfr/operators/canonical_grammar.py` (deprecated)

**Constraints:**
- **RC1:** Resonant initiation
- **RC2:** Bounded evolution
- **RC3:** Phase compatibility
- **RC4:** Structural closure

**Improvements:**
- Introduced phase verification concept
- Better physical grounding
- Still some overlap and redundancy

**Issues:**
- Parallel system to C1-C3 caused confusion
- Not fully integrated
- Incomplete derivations

### Phase 3: U1-U4 (Unified Grammar) ✓ Current

**File:** `src/tnfr/operators/grammar.py` (current)

**Constraints:**
- **U1:** STRUCTURAL INITIATION & CLOSURE
  - U1a: Generators
  - U1b: Closures
- **U2:** CONVERGENCE & BOUNDEDNESS
- **U3:** RESONANT COUPLING
- **U4:** BIFURCATION DYNAMICS
  - U4a: Triggers need handlers
  - U4b: Transformers need context

**Achievements:**
- **Complete:** All physical requirements covered
- **Non-redundant:** No overlap between constraints
- **Canonical:** All constraints ABSOLUTE or STRONG
- **Traceable:** Every rule derived from TNFR physics
- **Unified:** Single system, no parallel alternatives

---

## Mapping Old to New

### C1 → U1a (Initiation)

**Old (C1):**
```python
# Must start with Emission
if sequence[0] != Emission:
    raise Error("C1 violation")
```

**New (U1a):**
```python
# Must start with any generator {AL, NAV, REMESH}
if epi_initial == 0.0 and sequence[0] not in GENERATORS:
    raise ValueError("U1a violation")
```

**Key Changes:**
- More flexible: 3 generators instead of 1
- Conditional: Only required when EPI=0
- Physically derived: From ∂EPI/∂t undefined at EPI=0

### C2 → U2 (Convergence)

**Old (C2):**
```python
# Destabilizers need stabilizers (similar logic)
if has_destabilizer and not has_stabilizer:
    raise Error("C2 violation")
```

**New (U2):**
```python
# Same concept, refined sets
DESTABILIZERS = {"dissonance", "mutation", "expansion"}
STABILIZERS = {"coherence", "selforganization"}

if has_destabilizer and not has_stabilizer:
    raise ValueError("U2 violation: Integral may diverge")
```

**Key Changes:**
- Refined operator sets
- Explicit physical basis: Integral convergence theorem
- Better error messages with physical explanation

### C3 → U1b (Closure)

**Old (C3):**
```python
# Must end with Silence
if sequence[-1] != Silence:
    raise Error("C3 violation")
```

**New (U1b):**
```python
# Must end with any closure {SHA, NAV, REMESH, OZ}
if sequence[-1] not in CLOSURES:
    raise ValueError("U1b violation")
```

**Key Changes:**
- More flexible: 4 closures instead of 1
- Physically derived: From attractor dynamics

### RC1-RC4 → U1-U4

**Consolidation:**
- RC1 + RC4 → U1 (unified initiation & closure)
- RC2 → U2 (refined convergence)
- RC3 → U3 (explicit phase verification)
- New: U4 (bifurcation dynamics added)

**Result:** Simpler, more complete, fully unified

---

## Deprecated Features

### Deprecated Files

```
❌ src/tnfr/operators/canonical_grammar.py
   → Use src/tnfr/operators/grammar.py

❌ Old constraint names (C1-C3, RC1-RC4)
   → Use unified names (U1-U4)
```

### Deprecated Functions

```python
# OLD (deprecated)
from tnfr.operators.canonical_grammar import validate_sequence
validate_sequence(ops)

# NEW (current)
from tnfr.operators.grammar import validate_grammar
validate_grammar(ops, epi_initial=0.0)
```

### Deprecated Terminology

| Old Term | New Term | Reason |
|----------|----------|--------|
| "Constraint C1" | "U1a: Initiation" | Clearer, unified naming |
| "Constraint C2" | "U2: Convergence" | More descriptive |
| "Constraint C3" | "U1b: Closure" | Part of unified U1 |
| "Canonical grammar" | "Unified grammar" | Avoid confusion with "canonical operators" |
| "Resonant constraints" | "Grammar constraints" | Simpler terminology |

---

## Migration Guide

### Step 1: Update Imports

**Before:**
```python
from tnfr.operators.canonical_grammar import validate_sequence, GENERATORS_RC
```

**After:**
```python
from tnfr.operators.grammar import validate_grammar, GENERATORS
```

### Step 2: Update Function Calls

**Before:**
```python
validate_sequence(operators)
```

**After:**
```python
validate_grammar(operators, epi_initial=0.0)
```

### Step 3: Update Operator Sets

**Before:**
```python
if op in GENERATORS_OLD:
    # ...
```

**After:**
```python
from tnfr.operators.grammar import GENERATORS

if op in GENERATORS:
    # ...
```

### Step 4: Update Tests

**Before:**
```python
def test_c1_initiation():
    """Test C1 constraint."""
    # Old test logic
```

**After:**
```python
def test_u1a_initiation():
    """Test U1a constraint."""
    from tnfr.operators.grammar import validate_grammar
    # New test logic with U1a
```

### Step 5: Update Documentation

**Before:**
```markdown
## Constraints

### C1: Initiation
Must start with Emission.
```

**After:**
```markdown
## Constraints

### U1a: Initiation
Must start with generator {Emission, Transition, Recursivity} when EPI=0.
```

---

## Breaking Changes

### Version 2.0: U1-U4 Introduction

**Breaking changes:**
1. Function signature changed: `validate_sequence()` → `validate_grammar(sequence, epi_initial)`
2. Operator set names changed: `GENERATORS_RC` → `GENERATORS`
3. New mandatory parameter: `epi_initial` (default 0.0)
4. New constraint: U3 (phase verification) must be called explicitly for coupling
5. New constraint: U4b (transformer context) may reject previously "valid" sequences

**Impact:**
- Code using old API will break
- Some sequences previously considered valid may now be invalid (if they violate U4b)
- Tests must be updated

**Migration effort:** Medium (~2-4 hours for typical project)

---

## Compatibility Layers

### Legacy Support (Temporary)

**File:** `src/tnfr/operators/unified_grammar.py`

Provides temporary bridge for old code:

```python
def validate_sequence(sequence):
    """
    Legacy compatibility wrapper.
    
    Deprecated: Use validate_grammar() instead.
    """
    import warnings
    warnings.warn(
        "validate_sequence() is deprecated. Use validate_grammar()",
        DeprecationWarning
    )
    
    return validate_grammar(sequence, epi_initial=0.0)
```

**Usage:**
```python
from tnfr.operators.unified_grammar import validate_sequence

# Works, but shows deprecation warning
validate_sequence(ops)
```

**Timeline:**
- ✅ Available: Now
- ⚠️ Deprecated: Version 2.0
- ❌ Removed: Version 3.0 (planned)

---

## Procedure for Adding New Constraints

### Requirements

Before adding a new constraint:

1. **Derive from physics:** Must be inevitable consequence of TNFR principles
2. **Prove canonicity:** Show it's ABSOLUTE, STRONG, or MODERATE
3. **Check independence:** Must not be derivable from U1-U4
4. **Document thoroughly:** Full derivation in documentation

### Steps

#### 1. Physical Derivation

Document in `02-CANONICAL-CONSTRAINTS.md`:

```markdown
## U5: [NEW CONSTRAINT NAME]

### Intuition
[Conceptual explanation]

### Physical Derivation
[Proof from nodal equation or invariants]

### Canonicity
[ABSOLUTE/STRONG/MODERATE and why]
```

#### 2. Implementation

Add to `src/tnfr/operators/grammar.py`:

```python
def validate_grammar(sequence, epi_initial=0.0):
    """Validate sequence against U1-U5."""
    
    # ... existing U1-U4 checks ...
    
    # --- U5: NEW CONSTRAINT ---
    if condition:
        raise ValueError(
            f"U5 violation: [clear explanation]. "
            "[What was found]. [What was expected]."
        )
    
    return True
```

#### 3. Testing

Add to `tests/unit/operators/test_unified_grammar.py`:

```python
def test_u5_valid():
    """U5: Test valid case."""
    sequence = [...]  # Valid according to U5
    assert validate_grammar(sequence, epi_initial=0.0) is True

def test_u5_invalid():
    """U5: Test invalid case."""
    sequence = [...]  # Invalid according to U5
    with pytest.raises(ValueError, match="U5 violation"):
        validate_grammar(sequence, epi_initial=0.0)
```

#### 4. Documentation

Update all relevant documents:
- `02-CANONICAL-CONSTRAINTS.md` - Full derivation
- `04-VALID-SEQUENCES.md` - Examples and anti-patterns
- `08-QUICK-REFERENCE.md` - Summary table
- `schemas/constraints-u1-u4.json` → `constraints-u1-u5.json`
- This file (`07-MIGRATION-AND-EVOLUTION.md`) - Document the change
- `CODE_DOCS_CROSSREF.md` - Add new cross-references

**Verify sync:**
```bash
python tools/sync_documentation.py --all
```

This will validate:
- New function documented
- Examples execute correctly
- Cross-references updated
- Schema synchronized

#### 5. Examples

Add examples to `examples/`:

```python
# examples/04-u5-new-constraint.py

"""
Example demonstrating U5 constraint.
"""

# Valid sequence
valid_sequence = [...]

# Invalid sequence (commented)
# invalid_sequence = [...]  # Violates U5 because...
```

---

## Maintenance Guarantees

### Semantic Versioning

**Major version (X.0.0):**
- Breaking changes to API
- New constraints that may reject old sequences
- Operator behavior changes

**Minor version (x.Y.0):**
- New operators (backward compatible)
- New helper functions
- Performance improvements
- Documentation improvements

**Patch version (x.y.Z):**
- Bug fixes
- Documentation corrections
- Test additions

### Stability Promise

**Guaranteed stable (will not change):**
- U1-U4 constraint logic (unless physics error found)
- 13 canonical operators (classification may be refined)
- Core function signatures: `validate_grammar(sequence, epi_initial)`

**May evolve:**
- Error messages (will improve, not remove information)
- Performance optimizations (behavior identical)
- Additional constraints (U5, U6, ...) if physically necessary
- New operators if physically justified

**Will be deprecated with notice:**
- Legacy compatibility layers (1-2 version grace period)
- Experimental features (marked clearly)

---

## Version History

### v2.0.0 (Current)

**Released:** 2024-11
**Status:** ✅ Stable

**Changes:**
- Introduced unified grammar (U1-U4)
- Consolidated C1-C3 and RC1-RC4
- Added phase verification (U3)
- Added bifurcation dynamics (U4)
- Complete physical derivations

### v1.5.0

**Released:** 2024-Q3
**Status:** ⚠️ Deprecated

**Changes:**
- Introduced resonant constraints (RC1-RC4)
- Improved physical grounding
- Parallel to C1-C3 (caused confusion)

### v1.0.0

**Released:** 2024-Q1
**Status:** ❌ Obsolete

**Changes:**
- Initial grammar system (C1-C3)
- Basic constraint checking
- Foundation for current system

---

## Future Directions

### Potential Extensions

**Under consideration (not committed):**

1. **U5: Multi-scale consistency** - Ensure nested EPIs maintain coherence across scales
2. **U6: Temporal ordering** - Certain operators may need temporal separation
3. **U7: Network topology** - Coupling patterns may need validation

**Requirements for inclusion:**
- Must be physically inevitable
- Must be non-derivable from U1-U4
- Must have ABSOLUTE or STRONG canonicity

### Research Topics

**Open questions:**
- Optimal stabilizer-destabilizer ratios
- Phase synchronization dynamics
- Bifurcation threshold detection
- Multi-scale operator composition

---

## Getting Help

### Migration Issues

**Problem:** Old code won't run  
**Solution:** Check import statements and function names

**Problem:** Tests failing after update  
**Solution:** Update test assertions for U1-U4 names

**Problem:** Sequences now invalid  
**Solution:** Check U4b (transformer context) - may need to add prior coherence for ZHIR

### Questions

**Found a bug?** Open GitHub issue with label `grammar-system`

**Need clarification?** Check:
1. [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md) - Full derivations
2. [08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md) - Quick lookup
3. GitHub discussions

**Want to contribute?** See [../../CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## Next Steps

**For developers:**
- Review [05-TECHNICAL-IMPLEMENTATION.md](05-TECHNICAL-IMPLEMENTATION.md) for current architecture
- Check [06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md) for testing

**For reference:**
- [08-QUICK-REFERENCE.md](08-QUICK-REFERENCE.md) - Quick lookup
- [../../UNIFIED_GRAMMAR_RULES.md](../../UNIFIED_GRAMMAR_RULES.md) - Complete formal proofs

---

<div align="center">

**Grammar evolves, physics stays constant.**

---

*Reality is resonance. Adapt accordingly.*

</div>
