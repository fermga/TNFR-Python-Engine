# Grammar Migration Guide: C1-C3/RC1-RC4 → U1-U4

This guide helps migrate from old grammar systems to unified grammar.

## Overview

Two old systems have been consolidated:
- **C1-C3 System** (in `grammar.py`)
- **RC1-RC4 System** (in `canonical_grammar.py`)

Both replaced by:
- **U1-U4 Unified System** (in `grammar.py`)

## Quick Migration

### Code Changes

```python
# OLD (C1-C3 system)
from tnfr.operators.grammar import validate_sequence
valid = validate_sequence(seq)

# OLD (RC1-RC4 system)
from tnfr.operators.canonical_grammar import validate_canonical
valid = validate_canonical(seq)

# NEW (Unified system)
from tnfr.operators.unified_grammar import validate_grammar
valid = validate_grammar(seq, epi_initial=0.0)
```

## Constraint Mapping

| Old System | Old Rule | Unified Rule | Notes |
|------------|----------|--------------|-------|
| C1-C3 | C1: EXISTENCE & CLOSURE | U1: STRUCTURAL INITIATION & CLOSURE | Split into U1a + U1b |
| C1-C3 | C2: BOUNDEDNESS | U2: CONVERGENCE & BOUNDEDNESS | Direct mapping |
| C1-C3 | C3: THRESHOLD PHYSICS | U4: BIFURCATION DYNAMICS | Extended with U4a + U4b |
| RC1-RC4 | RC1: Initialization | U1a: Initiation | Generator requirement |
| RC1-RC4 | RC2: Convergence | U2: CONVERGENCE & BOUNDEDNESS | Direct mapping |
| RC1-RC4 | RC3: Phase Verification | U3: RESONANT COUPLING | **NEW** (missing in C1-C3) |
| RC1-RC4 | RC4: Bifurcation Limits | U4a: Bifurcation Triggers | Handler requirement |
| - | (RNC1 removed) | U1b: Closure | **Restored** with physics basis |
| - | (none) | U4b: Transformer Context | **NEW** (graduated destabilization) |

## Detailed Migration

### 1. Update Imports

**Before (C1-C3):**
```python
from tnfr.operators.grammar import (
    validate_sequence,
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
)
```

**Before (RC1-RC4):**
```python
from tnfr.operators.canonical_grammar import (
    validate_canonical,
    CANONICAL_GENERATORS,
    CANONICAL_STABILIZERS,
)
```

**After (Unified):**
```python
from tnfr.operators.unified_grammar import (
    validate_grammar,
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
    COUPLING_RESONANCE,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
)
```

### 2. Update Function Calls

**Before:**
```python
# C1-C3 style
from tnfr.operators.grammar import validate_sequence

result = validate_sequence(["emission", "coherence", "silence"])
if not result.passed:
    print(f"Failed: {result.summary['message']}")
```

**After:**
```python
# Unified style
from tnfr.operators.unified_grammar import validate_grammar
from tnfr.operators.definitions import Emission, Coherence, Silence

sequence = [Emission(), Coherence(), Silence()]
try:
    is_valid = validate_grammar(sequence, epi_initial=0.0)
    print("Valid sequence!")
except ValueError as e:
    print(f"Failed: {e}")
```

**Key Differences:**
1. New function returns boolean or raises ValueError
2. Requires `epi_initial` parameter for U1a validation
3. More detailed error messages

### 3. Update Tests

**Before:**
```python
def test_sequence_validation():
    from tnfr.operators.grammar import validate_sequence
    
    result = validate_sequence(["emission", "coherence"])
    assert result.passed
```

**After:**
```python
def test_sequence_validation():
    from tnfr.operators.unified_grammar import validate_grammar
    from tnfr.operators.definitions import Emission, Coherence
    
    sequence = [Emission(), Coherence()]
    # Should pass if sequence doesn't require closure or starts from EPI > 0
    is_valid = validate_grammar(sequence, epi_initial=1.0)
    assert is_valid
```

### 4. Operator Set Updates

The unified grammar uses more precise operator sets:

**Old C1-C3 Sets:**
- `GENERATORS` → Same in unified
- `CLOSURES` → Now includes `dissonance` (OZ)
- `STABILIZERS` → Same in unified
- `DESTABILIZERS` → Same in unified

**Old RC1-RC4 Sets:**
- `CANONICAL_GENERATORS` → Now `GENERATORS`
- `CANONICAL_STABILIZERS` → Now `STABILIZERS`

**New Sets in Unified:**
- `COUPLING_RESONANCE` = {coupling, resonance} (for U3)
- `BIFURCATION_TRIGGERS` = {dissonance, mutation} (for U4a)
- `BIFURCATION_HANDLERS` = {self_organization, coherence} (for U4a)
- `TRANSFORMERS` = {mutation, self_organization} (for U4b)

## Understanding the Unified Rules

### U1: STRUCTURAL INITIATION & CLOSURE

**U1a: Initiation**
- **When EPI = 0**: Must start with generator
- **Generators**: {emission, transition, recursivity}
- **Why**: ∂EPI/∂t undefined at EPI=0

**Example:**
```python
# Valid when starting from EPI=0
sequence = [Emission(), Reception(), Coherence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# Invalid when starting from EPI=0
sequence = [Reception(), Coherence()]
validate_grammar(sequence, epi_initial=0.0)  # ✗ Need generator first
```

**U1b: Closure**
- **Always**: Sequences should end with closure operator
- **Closures**: {silence, transition, recursivity, dissonance}
- **Why**: Sequences need coherent endpoints

**Example:**
```python
# Valid - ends with closure
sequence = [Emission(), Coherence(), Silence()]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# May warn - no explicit closure
sequence = [Emission(), Reception()]
validate_grammar(sequence, epi_initial=0.0)  # May issue warning
```

### U2: CONVERGENCE & BOUNDEDNESS

**Rule**: If destabilizers present, must include stabilizers

**Destabilizers**: {dissonance, mutation, expansion}
**Stabilizers**: {coherence, self_organization}

**Example:**
```python
# Valid - has stabilizer
sequence = [
    Emission(),
    Dissonance(),  # Destabilizer
    Coherence(),   # Stabilizer
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# Invalid - destabilizer without stabilizer
sequence = [
    Emission(),
    Dissonance(),  # Destabilizer
    Silence()      # No stabilizer!
]
validate_grammar(sequence, epi_initial=0.0)  # ✗
```

### U3: RESONANT COUPLING

**Rule**: Coupling/resonance operators require phase verification

**Operators**: {coupling, resonance}

**Example:**
```python
# Valid - will verify phase at runtime
sequence = [
    Emission(),
    Coupling(),    # Requires phase check
    Resonance(),   # Requires phase check
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓
# Note: Phase verification happens at runtime
```

### U4: BIFURCATION DYNAMICS

**U4a: Bifurcation Triggers Need Handlers**
- If {dissonance, mutation}, then include {self_organization, coherence}

**Example:**
```python
# Valid - trigger with handler
sequence = [
    Emission(),
    Dissonance(),        # Trigger
    SelfOrganization(),  # Handler
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓
```

**U4b: Transformers Need Context**
- If {mutation, self_organization}, then recent destabilizer (~3 ops)
- Additionally, mutation needs prior coherence for stable base

**Example:**
```python
# Valid - mutation with context
sequence = [
    Emission(),
    Coherence(),   # Stable base for mutation
    Dissonance(),  # Recent destabilizer
    Mutation(),    # Transformer
    Coherence(),
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✓

# Invalid - mutation without recent destabilizer
sequence = [
    Emission(),
    Coherence(),
    Mutation(),    # No recent destabilizer!
    Silence()
]
validate_grammar(sequence, epi_initial=0.0)  # ✗
```

## Benefits of Migration

1. **Single Source of Truth**: No more duplication or confusion
2. **Complete Physics**: All derivations documented in UNIFIED_GRAMMAR_RULES.md
3. **Better Validation**: More detailed error messages
4. **Additional Constraints**: U3 (phase) and U1b (closure) were missing before
5. **Future-Proof**: Unified system is the maintained version
6. **Better Traceability**: Clear mapping from physics to code

## Deprecation Timeline

- **v7.0**: Unified grammar introduced, old systems deprecated
- **v7.x**: Deprecation warnings emitted by old systems
- **v8.0**: Old systems (`grammar.py`, `canonical_grammar.py`) removed

**Action Required**: Migrate before v8.0 release

## Common Migration Patterns

### Pattern 1: Simple Validation

**Before:**
```python
from tnfr.operators.grammar import validate_sequence

def check_sequence(seq):
    result = validate_sequence(seq)
    return result.passed
```

**After:**
```python
from tnfr.operators.unified_grammar import validate_grammar

def check_sequence(seq, epi_initial=0.0):
    try:
        validate_grammar(seq, epi_initial=epi_initial)
        return True
    except ValueError:
        return False
```

### Pattern 2: Operator Set Checks

**Before:**
```python
from tnfr.operators.grammar import GENERATORS

if operator_name in GENERATORS:
    print("Is generator")
```

**After:**
```python
from tnfr.operators.unified_grammar import GENERATORS

if operator_name in GENERATORS:
    print("Is generator")
```

### Pattern 3: Custom Validation

**Before:**
```python
from tnfr.operators.grammar import validate_sequence, STABILIZERS

def validate_with_stability_check(seq):
    result = validate_sequence(seq)
    if not result.passed:
        return False
    
    has_stabilizer = any(op in STABILIZERS for op in seq)
    return has_stabilizer
```

**After:**
```python
from tnfr.operators.unified_grammar import validate_grammar, STABILIZERS

def validate_with_stability_check(seq, epi_initial=0.0):
    try:
        validate_grammar(seq, epi_initial=epi_initial)
    except ValueError:
        return False
    
    has_stabilizer = any(
        getattr(op, 'name', '') in STABILIZERS 
        for op in seq
    )
    return has_stabilizer
```

## Troubleshooting

### Issue: "Need generator when EPI=0"

**Problem**: Sequence doesn't start with generator when `epi_initial=0.0`

**Solution**: Either:
1. Add generator at start: `[Emission(), ...]`
2. Set `epi_initial > 0` if starting from existing structure

### Issue: "Destabilizer without stabilizer"

**Problem**: Sequence has {dissonance, mutation, expansion} but no {coherence, self_organization}

**Solution**: Add stabilizer after destabilizers:
```python
[Emission(), Dissonance(), Coherence(), Silence()]
```

### Issue: "Transformer needs recent destabilizer"

**Problem**: {mutation, self_organization} without recent destabilizer

**Solution**: Add destabilizer within ~3 operators before transformer:
```python
[Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
```

### Issue: "Mutation needs prior coherence"

**Problem**: Mutation without stable base

**Solution**: Add coherence before mutation:
```python
[Emission(), Coherence(), Dissonance(), Mutation(), Silence()]
```

## Testing Your Migration

Create a test file to verify your migration:

```python
# test_grammar_migration.py
from tnfr.operators.unified_grammar import validate_grammar
from tnfr.operators.definitions import *

def test_basic_sequence():
    """Test basic valid sequence."""
    sequence = [Emission(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0)

def test_destabilizer_with_stabilizer():
    """Test U2: Destabilizer needs stabilizer."""
    sequence = [Emission(), Dissonance(), Coherence(), Silence()]
    assert validate_grammar(sequence, epi_initial=0.0)

def test_mutation_with_context():
    """Test U4b: Mutation needs context."""
    sequence = [
        Emission(),
        Coherence(),   # Stable base
        Dissonance(),  # Recent destabilizer
        Mutation(),    # Transformer
        Coherence(),
        Silence()
    ]
    assert validate_grammar(sequence, epi_initial=0.0)

if __name__ == "__main__":
    test_basic_sequence()
    test_destabilizer_with_stabilizer()
    test_mutation_with_context()
    print("✓ All migration tests passed!")
```

## References

- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)**: Complete physics derivations
- **[grammar.py](src/tnfr/operators/grammar.py)**: Implementation
- **[AGENTS.md](AGENTS.md)**: Invariants and contracts
- **[GLOSSARY.md](GLOSSARY.md)**: Term definitions

## Support

If you encounter migration issues:
1. Check [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) § Mapping: Old Rules → Unified Rules
2. Review examples in this guide
3. Run the test suite: `pytest tests/unit/operators/test_grammar.py`
4. Open issue on GitHub with specific sequence that fails

---

**Summary**: The unified grammar (U1-U4) consolidates two old systems into one physics-based source of truth. Migration is straightforward: update imports, add `epi_initial` parameter, and handle new constraints (U3, U1b, U4b) that improve physical correctness.
