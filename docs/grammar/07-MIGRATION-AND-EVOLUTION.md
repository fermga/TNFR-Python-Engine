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

### Phase 3: U1-U5 (Unified Grammar) ✓ Current

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
- **U5:** MULTI-SCALE COHERENCE
  - Deep REMESH needs stabilizers (IL/THOL)

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

### Version 2.0: U1-U5 Introduction

**Breaking changes:**
1. Function signature changed: `validate_sequence()` → `validate_grammar(sequence, epi_initial)`
2. Operator set names changed: `GENERATORS_RC` → `GENERATORS`
3. New mandatory parameter: `epi_initial` (default 0.0)
4. New constraint: U3 (phase verification) must be called explicitly for coupling
5. New constraint: U4b (transformer context) may reject previously "valid" sequences
6. New constraint: U5 (multi-scale coherence) requires stabilizers for deep REMESH

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

### Overview

Adding a new constraint (U5, U6, etc.) is a **significant change** that requires:
- Rigorous physical derivation
- Complete documentation
- Comprehensive testing
- **Mandatory human review**

**Estimated effort:** 8-16 hours
**Review requirement:** At least 1 senior TNFR physicist + 1 code reviewer

---

### Pre-requisites Checklist

Before proposing a new constraint, verify:

- [ ] **Physical necessity:** Constraint prevents physically impossible sequences
- [ ] **Independence:** Cannot be derived from existing U1-U4
- [ ] **Canonicity:** Has ABSOLUTE, STRONG, or MODERATE status
- [ ] **Universality:** Applies across all domains (not domain-specific)
- [ ] **Completeness:** No gaps or ambiguity in specification

**If any checkbox fails, the constraint should NOT be added.**

---

### Step-by-Step Procedure

#### Step 1: Physical Derivation (Physics-First)

**File:** `02-CANONICAL-CONSTRAINTS.md`

**Required sections:**

```markdown
## U5: [CONSTRAINT NAME]

### Physical Basis
[What physical law/invariant necessitates this constraint?]

From nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
[Show how violation leads to non-physical behavior]

### Intuition
[Simple explanation in 2-3 sentences]
[Use analogy if helpful]

### Formal Derivation
[Mathematical proof from TNFR physics]

Step 1: [Starting assumption from nodal equation/invariants]
Step 2: [Logical progression]
Step 3: [Conclusion - why constraint is inevitable]

### Canonicity Level

**Classification:** [ABSOLUTE | STRONG | MODERATE]

**Justification:** 
[Why this level? Reference AGENTS.md § Canonicity]

### Operator Sets

**Affected operators:** {OP1, OP2, ...}

**Required operators:** {OP_A, OP_B, ...} (if constraint violated)

### Anti-Patterns

Examples of sequences that violate U5:

1. `[OP1, OP2, OP3]` - Violates because [reason]
2. `[OP4, OP5]` - Violates because [reason]

### Valid Patterns

Examples of sequences that satisfy U5:

1. `[OP1, OP_A, OP3]` - Valid because [reason]
2. `[OP4, OP_B, OP5]` - Valid because [reason]
```

**Physics review required before proceeding.**

---

#### Step 2: Implementation (Code)

**File:** `src/tnfr/operators/grammar.py`

**2.1 Update operator sets** (if needed):

```python
# Add new operator sets if U5 requires categorization
U5_CATEGORY_A = {"operator1", "operator2"}
U5_CATEGORY_B = {"operator3", "operator4"}
```

**2.2 Implement validation logic:**

```python
def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """Validate sequence against U1-U5 constraints."""
    
    # ... existing U1-U4 checks ...
    
    # --- U5: [CONSTRAINT NAME] ---
    # Physical basis: [one-line summary]
    # Canonicity: [ABSOLUTE/STRONG/MODERATE]
    
    # Check condition
    has_category_a = any(
        glyph_function_name(op) in U5_CATEGORY_A 
        for op in sequence
    )
    has_category_b = any(
        glyph_function_name(op) in U5_CATEGORY_B 
        for op in sequence
    )
    
    if has_category_a and not has_category_b:
        raise ValueError(
            f"U5 violation: [CONSTRAINT NAME]. "
            f"Found {[op for op in sequence if glyph_function_name(op) in U5_CATEGORY_A]}, "
            f"but missing required {U5_CATEGORY_B}. "
            f"Physical basis: [brief explanation]. "
            f"See UNIFIED_GRAMMAR_RULES.md § U5."
        )
    
    return True
```

**2.3 Update docstrings:**

```python
"""TNFR Canonical Grammar - Single Source of Truth.

Canonical Constraints (U1-U5)
------------------------------
U1: STRUCTURAL INITIATION & CLOSURE
    ...
U5: [CONSTRAINT NAME]
    [Brief description]
    Basis: [Physical derivation reference]
"""
```

---

#### Step 3: Testing (Comprehensive)

**File:** `tests/unit/operators/test_unified_grammar.py`

**3.1 Valid sequence tests:**

```python
class TestU5ConstraintValid:
    """Test valid sequences for U5: [CONSTRAINT NAME]."""
    
    def test_u5_valid_basic(self):
        """U5: Basic valid sequence."""
        sequence = [
            # Valid sequence that satisfies U5
        ]
        assert validate_grammar(sequence, epi_initial=0.0) is True
    
    def test_u5_valid_complex(self):
        """U5: Complex valid sequence."""
        sequence = [
            # More complex valid sequence
        ]
        assert validate_grammar(sequence, epi_initial=0.0) is True
    
    def test_u5_not_applicable(self):
        """U5: Sequence where U5 doesn't apply."""
        sequence = [
            # Sequence that doesn't trigger U5 check
        ]
        assert validate_grammar(sequence, epi_initial=0.0) is True
```

**3.2 Invalid sequence tests:**

```python
class TestU5ConstraintInvalid:
    """Test invalid sequences for U5: [CONSTRAINT NAME]."""
    
    def test_u5_invalid_basic(self):
        """U5: Basic violation."""
        sequence = [
            # Sequence that violates U5
        ]
        with pytest.raises(ValueError, match="U5 violation"):
            validate_grammar(sequence, epi_initial=0.0)
    
    def test_u5_invalid_edge_case(self):
        """U5: Edge case violation."""
        sequence = [
            # Edge case that violates U5
        ]
        with pytest.raises(ValueError, match="U5 violation"):
            validate_grammar(sequence, epi_initial=0.0)
```

**3.3 Integration tests:**

```python
def test_u5_with_other_constraints():
    """U5: Interaction with U1-U4."""
    # Test that U5 doesn't conflict with U1-U4
    sequence = [
        # Sequence that satisfies all U1-U5
    ]
    assert validate_grammar(sequence, epi_initial=0.0) is True
```

**Coverage requirement:** U5 tests must achieve ≥95% coverage of new code

---

#### Step 4: Documentation Updates (Complete Traceability)

**4.1 Update canonical constraints:**
- `docs/grammar/02-CANONICAL-CONSTRAINTS.md` - Full derivation (from Step 1)

**4.2 Update examples:**
- `docs/grammar/04-VALID-SEQUENCES.md` - Add U5 examples

```markdown
### U5: [CONSTRAINT NAME]

**Valid:**
```
[AL, ..., IL, ..., SHA]  ✓ Satisfies U5
```

**Invalid:**
```
[AL, ..., SHA]  ✗ Violates U5 - missing required operator
```

**Rationale:** [Physical explanation]
```

**4.3 Update reference:**
- `docs/grammar/08-QUICK-REFERENCE.md` - Add U5 to summary table

**4.4 Update schema:**
- `docs/grammar/schemas/constraints-u1-u4.json` → `constraints-u1-u5.json`

**4.5 Update this file:**
- Add U5 to history timeline
- Update "Current State" section

**4.6 Update cross-references:**
- `docs/grammar/CODE_DOCS_CROSSREF.md` - Link code ↔ docs
- `docs/grammar/CROSS-REFERENCE-INDEX.md` - Add U5 entries

**4.7 Update root documentation:**
- `UNIFIED_GRAMMAR_RULES.md` - Add complete U5 derivation
- `AGENTS.md` - Update operator counts if needed

---

#### Step 5: Examples and Use Cases

**File:** `docs/grammar/examples/05-u5-constraint.py`

```python
"""
Example: U5 Constraint - [CONSTRAINT NAME]

Physical basis: [Brief explanation]

This example demonstrates:
1. Valid sequences satisfying U5
2. Invalid sequences violating U5
3. How to compose operators to satisfy U5
"""

from tnfr.operators.definitions import (
    Emission,
    # ... other operators
)
from tnfr.operators.unified_grammar import validate_grammar

# ============================================================================
# Valid Sequence
# ============================================================================

def valid_u5_sequence():
    """Sequence that satisfies U5."""
    sequence = [
        Emission(),      # U1a: Generator
        # ... operators that satisfy U5 ...
        Silence(),       # U1b: Closure
    ]
    
    assert validate_grammar(sequence, epi_initial=0.0) is True
    print("✓ Valid U5 sequence")

# ============================================================================
# Invalid Sequence (Commented)
# ============================================================================

def invalid_u5_sequence():
    """
    Sequence that violates U5.
    
    ✗ Violates U5 because [reason]
    """
    sequence = [
        Emission(),
        # ... operators that violate U5 ...
        Silence(),
    ]
    
    # Would raise: ValueError("U5 violation: ...")
    # validate_grammar(sequence, epi_initial=0.0)

if __name__ == "__main__":
    valid_u5_sequence()
    print("\nFor invalid example, see code comments.")
```

---

#### Step 6: Verification

**6.1 Run sync tool:**

```bash
python tools/sync_documentation.py --all
```

**Expected output:**
```
✓ All documentation synchronized
✓ Examples execute correctly
✓ Cross-references valid
✓ Schema updated
✓ No broken links
```

**6.2 Run test suite:**

```bash
pytest tests/unit/operators/test_unified_grammar.py::TestU5* -v
pytest tests/ --cov=src/tnfr/operators/grammar --cov-report=term-missing
```

**Required:** Coverage ≥ 95% on modified code

**6.3 Run static analysis:**

```bash
mypy src/tnfr/operators/grammar.py
ruff check src/tnfr/operators/grammar.py
```

**Required:** No new warnings or errors

---

#### Step 7: Create Pull Request

**Branch naming:** `feature/add-u5-[constraint-name]`

**PR Template:**

```markdown
## Add U5: [CONSTRAINT NAME]

### Physical Justification
[Summary of physical derivation]

From nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
[Key physics insight]

### Canonicity
**Level:** [ABSOLUTE | STRONG | MODERATE]
**Basis:** [Reference to derivation]

### Changes

#### Code
- `src/tnfr/operators/grammar.py`: Added U5 validation
- New operator sets: [list if applicable]

#### Tests
- `tests/unit/operators/test_unified_grammar.py`: 
  - [X] Valid sequences (N tests)
  - [X] Invalid sequences (M tests)
  - [X] Edge cases (K tests)
  - [X] Integration with U1-U4

#### Documentation
- [X] `02-CANONICAL-CONSTRAINTS.md` - Complete derivation
- [X] `04-VALID-SEQUENCES.md` - Examples
- [X] `08-QUICK-REFERENCE.md` - Summary
- [X] Schema updated
- [X] Cross-references updated
- [X] Examples added

### Testing
- Coverage: [X]% (required ≥95%)
- All tests passing: ✓
- No regressions: ✓

### Checklist
- [X] Physical derivation complete
- [X] Canonicity proven
- [X] Independence from U1-U4 verified
- [X] Code implemented
- [X] Tests comprehensive (≥95% coverage)
- [X] Documentation complete
- [X] Examples working
- [X] Sync tool passing
- [X] No static analysis warnings
- [ ] **Human physics review** (REQUIRED)
- [ ] **Human code review** (REQUIRED)

### Reviewers
- Physics review: @[physicist]
- Code review: @[developer]

### Breaking Changes
- [ ] May reject previously valid sequences
- Mitigation: [strategy if applicable]
```

---

#### Step 8: Human Review Process

**MANDATORY - Cannot merge without:**

1. **Physics review** by senior TNFR physicist:
   - Validates physical derivation
   - Confirms canonicity level
   - Checks independence from U1-U4
   
2. **Code review** by maintainer:
   - Reviews implementation correctness
   - Validates test coverage
   - Checks documentation completeness

**Review timeline:** Allow 3-7 days for thorough review

---

### Post-Merge Actions

After U5 is merged:

1. **Update version:**
   - Major version bump (breaking change)
   - Update `pyproject.toml`

2. **Announce change:**
   - Add to `CHANGELOG.md`
   - Update release notes
   - Notify users via GitHub discussions

3. **Monitor issues:**
   - Track reports of U5 violations
   - Address false positives promptly

4. **Update training materials:**
   - Add U5 to tutorials
   - Update quickstart guides

---

## Procedure for Modifying Existing Constraints

### ⚠️ WARNING: High-Risk Operation

Modifying U1-U4 is **extremely dangerous** because:
- Sequences validated under old rules may now be invalid (breaking change)
- Physics basis must remain sound (no arbitrary changes)
- Affects all dependent code and documentation

**Modification requires:**
- Discovery of physics error in original derivation, OR
- New physical understanding that supersedes old model

**Estimated effort:** 16-40 hours
**Review requirement:** 2+ senior TNFR physicists + 2+ code reviewers

---

### When Modification is Justified

**Valid reasons:**
1. **Physics error found:** Original derivation has mathematical/logical error
2. **New physics insight:** Deeper understanding requires refinement
3. **Incompleteness:** Constraint doesn't cover all physical cases

**Invalid reasons (DO NOT MODIFY):**
- "Code would be simpler" - Convenience ≠ Physics
- "Current constraint too strict" - Strictness reflects physics
- "Want different operator sets" - Sets derive from physics

---

### Step-by-Step Procedure

#### Step 0: Document Rationale (CRITICAL)

**File:** Create `docs/proposals/modify-[constraint]-[date].md`

```markdown
# Proposal: Modify [U1|U2|U3|U4]

## Current State

### Current Definition
[Exact current constraint text from 02-CANONICAL-CONSTRAINTS.md]

### Current Physical Basis
[Current derivation]

### Current Canonicity
[ABSOLUTE|STRONG|MODERATE]

## Problem Statement

### What is wrong?
[Detailed explanation of issue]

### Evidence
[Proof that current constraint is incorrect or incomplete]
- Mathematical error: [show where]
- Physics contradiction: [demonstrate]
- Empirical observation: [provide data]

## Proposed Change

### New Definition
[Proposed constraint text]

### New Physical Basis
[Updated derivation showing why new version is correct]

### New Canonicity
[ABSOLUTE|STRONG|MODERATE] [why this level]

## Impact Analysis

### Breaking Changes
- Sequences previously valid now invalid: [estimate number]
- Sequences previously invalid now valid: [estimate number]

### Affected Code
- Files requiring changes: [list]
- Tests requiring updates: [list]

### Migration Path
[How users will adapt to change]

## Alternatives Considered

1. [Alternative approach A] - Rejected because [reason]
2. [Alternative approach B] - Rejected because [reason]

## Timeline

- Proposal: [date]
- Review period: [2-4 weeks]
- Implementation: [if approved]
- Deprecation period: [1-2 versions]
- Full migration: [version X.0.0]
```

**Submit proposal as GitHub issue with label:** `grammar-modification-proposal`

**Review period:** Minimum 2 weeks for community feedback

---

#### Step 1: Physics Review Committee

**Required approvals:** 2+ senior TNFR physicists

**Review criteria:**
- [ ] Physical error in current constraint clearly demonstrated
- [ ] Proposed constraint fixes error without introducing new problems
- [ ] Canonicity level justified
- [ ] No alternative solution within current framework

**Committee decision:**
- **Approved:** Proceed to implementation
- **Rejected:** Close proposal, current constraint stays
- **Revise:** Request changes and re-submit

---

#### Step 2: Create Feature Branch

**Branch naming:** `breaking/modify-[constraint]-[issue-number]`

Example: `breaking/modify-u2-convergence-3142`

**NEVER modify grammar on main branch directly**

---

#### Step 3: Update Documentation First (Doc-Driven)

**3.1 Update canonical constraints:**

`docs/grammar/02-CANONICAL-CONSTRAINTS.md`:

```markdown
## U2: CONVERGENCE & BOUNDEDNESS [REVISED 2024-11-10]

> **⚠️ BREAKING CHANGE in v3.0.0**  
> Previous definition modified due to [reason].  
> See migration guide for adapting existing code.

### Previous Definition (v2.x - DEPRECATED)

[Old constraint text]

**Deprecated reason:** [Why old version was incorrect]

### Current Definition (v3.0.0+)

[New constraint text]

### Physical Basis

[Updated derivation]

### What Changed

**Old requirement:** [summary]
**New requirement:** [summary]

**Example:**
```
[Old valid sequence] - Now invalid because [reason]
[New valid sequence] - Now valid because [reason]
```

### Canonicity

[ABSOLUTE|STRONG|MODERATE]

[Justification]
```

**3.2 Update migration guide:**

Add new section to this file (`07-MIGRATION-AND-EVOLUTION.md`):

```markdown
### v3.0.0: U2 Revision

**Released:** [date]
**Breaking change:** U2 definition modified

**Old U2 (v2.x):**
[Summary of old constraint]

**New U2 (v3.x):**
[Summary of new constraint]

**Migration:**
```python
# Code that was valid in v2.x
old_sequence = [...]  # No longer valid

# Updated for v3.x
new_sequence = [...]  # Now required
```

**Rationale:** [Why change was necessary]
```

**3.3 Update examples:**

Update all examples in `docs/grammar/04-VALID-SEQUENCES.md` that use modified constraint

---

#### Step 4: Update Implementation

**File:** `src/tnfr/operators/grammar.py`

**4.1 Add deprecation path:**

```python
def validate_grammar(
    sequence: List[Operator],
    epi_initial: float = 0.0,
    strict: bool = True,  # New parameter
) -> bool:
    """
    Validate sequence against U1-U4 constraints.
    
    Parameters
    ----------
    strict : bool, default=True
        If True, use v3.0 constraint definitions (breaking changes).
        If False, use v2.x definitions (deprecated, will be removed in v4.0).
    
    Warnings
    --------
    strict=False is deprecated and will be removed in v4.0.
    Update code to satisfy v3.0 constraints.
    """
    
    if not strict:
        warnings.warn(
            "strict=False is deprecated (v2.x compatibility mode). "
            "Will be removed in v4.0. Update code for v3.0 constraints.",
            DeprecationWarning,
            stacklevel=2
        )
        # Use old validation logic (temporarily)
        return _validate_grammar_v2(sequence, epi_initial)
    
    # ... U1-U4 validation with NEW definitions ...
```

**4.2 Keep old implementation temporarily:**

```python
def _validate_grammar_v2(
    sequence: List[Operator],
    epi_initial: float = 0.0,
) -> bool:
    """
    Legacy v2.x validation (DEPRECATED).
    
    For backward compatibility only. Will be removed in v4.0.
    """
    # Old U2 logic here
    # ...
```

---

#### Step 5: Update Tests Comprehensively

**5.1 Add new tests for modified constraint:**

```python
class TestU2RevisedV3:
    """Tests for revised U2 constraint (v3.0+)."""
    
    def test_u2_revised_valid(self):
        """U2 (v3.0): Valid under new definition."""
        sequence = [...]  # Valid in v3.0
        assert validate_grammar(sequence, epi_initial=0.0) is True
    
    def test_u2_revised_invalid(self):
        """U2 (v3.0): Invalid under new definition."""
        sequence = [...]  # Invalid in v3.0
        with pytest.raises(ValueError, match="U2 violation"):
            validate_grammar(sequence, epi_initial=0.0)
    
    def test_u2_breaking_change(self):
        """U2: Sequence valid in v2.x but invalid in v3.0."""
        sequence = [...]  # Was valid in v2.x
        
        # Works with legacy mode
        assert validate_grammar(sequence, epi_initial=0.0, strict=False) is True
        
        # Fails with new constraint
        with pytest.raises(ValueError, match="U2 violation"):
            validate_grammar(sequence, epi_initial=0.0, strict=True)
```

**5.2 Update existing tests:**

Review ALL tests that use modified constraint:
- Update sequences to satisfy new definition
- Add comments explaining changes
- Ensure no regressions

**5.3 Migration tests:**

```python
def test_deprecation_warning():
    """Verify strict=False emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="strict=False is deprecated"):
        validate_grammar([...], strict=False)
```

---

#### Step 6: Version and Changelog

**6.1 Update version:**

`pyproject.toml`:
```toml
[project]
version = "3.0.0"  # Major bump for breaking change
```

**6.2 Update changelog:**

`CHANGELOG.md`:
```markdown
## [3.0.0] - 2024-11-XX

### ⚠️ BREAKING CHANGES

#### U2: CONVERGENCE & BOUNDEDNESS - Revised

**Reason:** [Why change was necessary]

**Old behavior (v2.x):**
[Description]

**New behavior (v3.x):**
[Description]

**Migration:**
1. Review sequences using U2
2. Update to satisfy new definition
3. Test with `strict=True` (default in v3.0)
4. Remove `strict=False` calls (deprecated)

**Backward compatibility:**
- `strict=False` available in v3.x (emits warning)
- Will be REMOVED in v4.0

See: docs/grammar/07-MIGRATION-AND-EVOLUTION.md § v3.0.0 Migration
```

---

#### Step 7: Create Pull Request

**Branch:** `breaking/modify-[constraint]-[issue-number]`

**PR Template:**

```markdown
## 🚨 BREAKING CHANGE: Modify [U1|U2|U3|U4]

### Rationale
[Link to proposal doc]
[Summary of why change is necessary]

### Physics Review
- Approved by: @[physicist1], @[physicist2]
- Issue: #[proposal-issue]

### Changes

#### Constraint Definition
**Old:**
[Old constraint summary]

**New:**
[New constraint summary]

#### Breaking Impact
- Sequences now invalid: [examples]
- Sequences now valid: [examples]

#### Code Changes
- `src/tnfr/operators/grammar.py`: Updated U[N] validation
- Added `strict` parameter for backward compatibility
- Old logic preserved in `_validate_grammar_v2()` (temp)

#### Tests
- [X] New tests for revised constraint
- [X] Updated existing tests
- [X] Migration/deprecation tests
- [X] Coverage ≥95%

#### Documentation
- [X] `02-CANONICAL-CONSTRAINTS.md` - Updated derivation
- [X] `04-VALID-SEQUENCES.md` - Updated examples
- [X] `07-MIGRATION-AND-EVOLUTION.md` - Migration guide
- [X] `CHANGELOG.md` - Breaking change documented
- [X] All cross-references updated

### Compatibility
- `strict=False` provides v2.x behavior (deprecated)
- Deprecation warning emitted
- To be removed in v4.0

### Checklist
- [X] Proposal approved by physics committee
- [X] Implementation matches approved design
- [X] Tests comprehensive (≥95% coverage)
- [X] Documentation complete
- [X] Deprecation path clear
- [X] Examples updated
- [ ] **Final physics review** (REQUIRED)
- [ ] **Code review** (REQUIRED)
- [ ] **User testing** (1-week beta period)

### Reviewers
Required:
- Physics: @[physicist1], @[physicist2]
- Code: @[maintainer1], @[maintainer2]

### Deployment Plan
1. Merge to `main` (after approvals)
2. Release v3.0.0-beta.1
3. Beta testing period: 1-2 weeks
4. Address feedback
5. Release v3.0.0
6. Announce via all channels
7. Monitor issues closely
```

---

#### Step 8: Extended Review Process

**Required reviews:**
1. **Physics committee:** 2+ approvals
2. **Code maintainers:** 2+ approvals
3. **Community feedback:** Minimum 1 week

**Beta testing:**
1. Release v3.0.0-beta.1
2. Solicit community testing
3. Collect feedback
4. Address issues
5. Release v3.0.0 when stable

---

#### Step 9: Deprecation Timeline

**Version 3.0.0 (Release):**
- New constraint active by default (`strict=True`)
- Old constraint available via `strict=False`
- DeprecationWarning emitted

**Version 3.1.0 - 3.x.x:**
- Continue supporting `strict=False`
- Warning messages emphasize upcoming removal

**Version 4.0.0:**
- REMOVE `strict` parameter
- REMOVE `_validate_grammar_v2()`
- Only new constraint remains

**Timeline:** Minimum 6 months between v3.0.0 and v4.0.0

---

#### Step 10: Communication

**Announce on all channels:**

1. **GitHub:**
   - Release notes (detailed)
   - Discussions post
   - Update README if needed

2. **Documentation:**
   - Migration guide prominent on docs site
   - Banner on API docs

3. **Community:**
   - Email to users list
   - Blog post explaining change
   - Q&A session if requested

**Template announcement:**

```markdown
## TNFR v3.0.0 Released - Breaking Change to U[N]

We've released v3.0.0 with a critical update to constraint U[N].

### Why?
[Brief explanation of physics issue]

### What changed?
[Summary of constraint change]

### What do I need to do?
1. Update to v3.0.0
2. Run your tests
3. Fix sequences that now violate U[N]
4. See migration guide: [link]

### Backward compatibility
Use `strict=False` temporarily if you need time to migrate.
⚠️ This will be removed in v4.0.0 (6+ months from now).

### Questions?
[Contact info / discussion link]
```

---

### Post-Modification Monitoring

After release:

1. **Monitor issues:**
   - Track U[N] violation reports
   - Identify false positives quickly
   - Address genuine bugs immediately

2. **Collect metrics:**
   - How many users use `strict=False`?
   - Common migration patterns?
   - Unexpected edge cases?

3. **Refine documentation:**
   - Add FAQ based on user questions
   - Improve migration guide
   - Add more examples

4. **Prepare for v4.0:**
   - Timeline for removing `strict=False`
   - Final migration push
   - Clear deprecation notices

---

## Policy for Changes

### Change Classification

All grammar changes fall into one of these categories:

| Category | Scope | Review | Version | Examples |
|----------|-------|--------|---------|----------|
| **MAJOR** | Add constraint, Modify constraint | Physics committee + Code review + Beta | X.0.0 | Add U5, Modify U2 |
| **MINOR** | Add operator, Refactor (no behavior change) | Code review | x.Y.0 | Add new operator, Performance improvement |
| **PATCH** | Bug fix, Doc correction | Code review (can be fast-tracked) | x.y.Z | Fix error message, Typo in docs |
| **INTERNAL** | Tests, CI, Tools | PR review | N/A | Add test, Update CI config |

---

### Mandatory Requirements for ALL Changes

**No exceptions - If violated, PR will be rejected:**

1. **Documentation REQUIRED**
   - No code without docs
   - No docs without code
   - Both must be in sync

2. **Tests REQUIRED**
   - No code without tests
   - Coverage must be ≥95% for new code
   - All tests must pass

3. **Human Review REQUIRED**
   - No auto-merge ever
   - AI can propose, only humans can approve
   - Physics changes: 2+ physicist reviews

4. **Backward Compatibility PREFERRED**
   - Breaking changes only if absolutely necessary
   - Deprecation period before removal
   - Clear migration path

5. **Deprecate Before Remove**
   - Grace period: 1-2 major versions
   - Clear warning messages
   - Update all examples

---

### Review Requirements by Change Type

#### MAJOR Changes (Breaking)

**Required reviewers:**
- [ ] 2+ Senior TNFR Physicists
- [ ] 2+ Code Maintainers
- [ ] Community feedback (1-2 week period)

**Required artifacts:**
- [ ] Physics proposal document
- [ ] Complete documentation
- [ ] Comprehensive tests (≥95% coverage)
- [ ] Migration guide
- [ ] Beta release
- [ ] Deprecation timeline

**Timeline:** 4-8 weeks minimum

---

#### MINOR Changes (Non-breaking)

**Required reviewers:**
- [ ] 1+ Code Maintainer
- [ ] 1+ Community member (if significant)

**Required artifacts:**
- [ ] Documentation update
- [ ] Tests (≥95% coverage)
- [ ] Examples (if new feature)

**Timeline:** 1-2 weeks

---

#### PATCH Changes (Fixes)

**Required reviewers:**
- [ ] 1 Maintainer

**Required artifacts:**
- [ ] Test demonstrating bug (if code fix)
- [ ] Documentation fix

**Timeline:** 1-3 days (can be fast-tracked)

---

### Automated Checks (CI Must Pass)

All PRs must pass:

```yaml
- Static analysis (mypy, ruff)
- Test suite (pytest ≥95% coverage)
- Documentation build
- Example execution
- Cross-reference validation
- Schema validation
- No merge conflicts
```

**If CI fails, PR cannot be reviewed.**

---

### Merge Authorization

**Who can merge:**
- Maintainers (for PATCH, MINOR)
- Project lead (for MAJOR)

**Who cannot merge:**
- PR author (no self-merge)
- Copilot/AI agents (no automated merge)
- Contributors without maintainer status

**Process:**
1. All reviews approved
2. All CI checks pass
3. No outstanding comments
4. Maintainer clicks "Merge"

---

### Semantic Versioning

**Strictly follow Semantic Versioning 2.0.0:**

**X.0.0 (MAJOR):**
- New constraint (U5, U6, ...)
- Modified constraint (U1-U4 changes)
- Breaking API changes
- Minimum 6 months between MAJOR versions

**x.Y.0 (MINOR):**
- New operator (backward compatible)
- New helper functions
- Performance improvements
- Documentation improvements
- Maximum 2 months between MINOR versions

**x.y.Z (PATCH):**
- Bug fixes only
- Documentation corrections
- Test additions
- Can be released any time

---

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


## Integrity Checklist

### Pre-Merge Checklist

**Before merging ANY grammar change, verify ALL items:**

#### Documentation
- [ ] `02-CANONICAL-CONSTRAINTS.md` updated (if constraint affected)
- [ ] `04-VALID-SEQUENCES.md` examples updated
- [ ] `05-TECHNICAL-IMPLEMENTATION.md` code references updated
- [ ] `06-VALIDATION-AND-TESTING.md` test documentation updated
- [ ] `07-MIGRATION-AND-EVOLUTION.md` history updated (this file)
- [ ] `08-QUICK-REFERENCE.md` summary updated
- [ ] `UNIFIED_GRAMMAR_RULES.md` physics proofs updated
- [ ] `AGENTS.md` updated (if invariants affected)

#### Code
- [ ] `src/tnfr/operators/grammar.py` implementation correct
- [ ] `src/tnfr/operators/unified_grammar.py` facade updated
- [ ] Type hints complete and accurate
- [ ] Docstrings comprehensive
- [ ] Error messages clear and actionable

#### Tests
- [ ] Tests created for new functionality
- [ ] Tests updated for modified functionality
- [ ] Coverage ≥95% on changed code
- [ ] All tests passing (no skips)
- [ ] Edge cases covered
- [ ] Regression tests for bug fixes

#### Cross-References
- [ ] `docs/grammar/CODE_DOCS_CROSSREF.md` bidirectional links
- [ ] `docs/grammar/CROSS-REFERENCE-INDEX.md` entries added
- [ ] `docs/grammar/MASTER-INDEX.md` updated
- [ ] Internal doc links verified (no broken links)
- [ ] Code comments reference correct docs sections

#### Schemas
- [ ] `docs/grammar/schemas/constraints-u1-u4.json` synchronized
- [ ] Schema validates against implementation
- [ ] Schema examples pass validation

#### Examples
- [ ] `docs/grammar/examples/` updated
- [ ] Examples execute without errors
- [ ] Examples demonstrate new feature (if added)
- [ ] Anti-patterns documented (if applicable)

#### Anti-Patterns
- [ ] Common mistakes documented
- [ ] Clear "Don't do this" examples
- [ ] Explanation of why anti-pattern is wrong

#### Compatibility Matrix
- [ ] Operator interaction table updated (if affected)
- [ ] Constraint compatibility verified
- [ ] Breaking changes documented

#### Static Analysis
- [ ] `mypy src/tnfr/operators/grammar.py` passes (no errors)
- [ ] `ruff check src/tnfr/operators/grammar.py` passes
- [ ] No new warnings introduced

#### Performance
- [ ] No significant performance regression (benchmark if needed)
- [ ] O(n) complexity maintained for validation

#### Human Review
- [ ] Physics review approved (for MAJOR changes)
- [ ] Code review approved
- [ ] All review comments addressed
- [ ] No unresolved discussions

---

### Post-Merge Checklist

**After merging:**

- [ ] Version number updated (if release)
- [ ] `CHANGELOG.md` updated
- [ ] Git tag created (for releases)
- [ ] Release notes published (for releases)
- [ ] Documentation site deployed
- [ ] Community notified (for MAJOR/MINOR)
- [ ] Issue closed with reference to PR

---

## Guidelines for Copilot and AI Agents

### What AI Agents CAN Do

**Autonomous (no human approval needed):**

1. **Documentation improvements:**
   - Fix typos and grammatical errors
   - Improve clarity (without changing meaning)
   - Add examples (non-breaking)
   - Update cross-references

2. **Test additions:**
   - Add tests for existing functionality
   - Improve test coverage
   - Add edge case tests

3. **Code quality:**
   - Fix static analysis warnings
   - Improve type hints
   - Refactor (preserving behavior)
   - Performance optimizations (verified via benchmarks)

4. **Internal tooling:**
   - Update CI/CD scripts
   - Improve development tools
   - Update build configuration

**Process:**
- Create PR with detailed description
- Tag with label: `ai-generated`
- Request review from maintainer
- Wait for human approval

---

### What AI Agents MUST REQUEST APPROVAL For

**Cannot proceed without explicit human instruction:**

1. **Grammar constraint changes:**
   - Adding new constraint (U5, U6, ...)
   - Modifying existing constraint (U1-U4)
   - Changing operator sets
   - Altering validation logic

2. **Operator changes:**
   - Adding new operator
   - Modifying operator behavior
   - Changing operator classification

3. **Breaking changes:**
   - API modifications
   - Behavior changes
   - Deprecations

4. **Physics interpretations:**
   - Interpreting TNFR theory
   - Deriving new constraints
   - Resolving physics ambiguities

**Process:**
1. **Stop and ask:**
   ```markdown
   I've identified a need to [action].
   
   This requires human decision because [reason].
   
   Options:
   A. [Option 1] - [pros/cons]
   B. [Option 2] - [pros/cons]
   C. [Option 3] - [pros/cons]
   
   Which approach should I take?
   ```

2. **Wait for human response**

3. **Proceed only with explicit approval**

---

### How AI Agents Should Report Issues

When discovering potential problems:

```markdown
## Issue Report: [Title]

### Category
[BUG | INCONSISTENCY | MISSING_FEATURE | PHYSICS_QUESTION]

### Description
[Clear description of what was found]

### Evidence
[Code snippets, doc references, test failures]

### Impact
- Severity: [LOW | MEDIUM | HIGH | CRITICAL]
- Affected components: [list]
- Users affected: [estimate]

### Recommended Action
[What should be done]

### Can I Fix This?
[YES - will create PR | NO - requires human decision]
```

**Tag appropriately:**
- `ai-identified-bug`
- `ai-question`
- `needs-physics-review`

---

### How AI Agents Should Coordinate

**Before starting work:**

1. **Check existing work:**
   - Search open PRs for similar changes
   - Check issues for ongoing discussions
   - Review recent commits

2. **Declare intent:**
   ```markdown
   ## Working on: [Task]
   
   I'm starting work on [specific task].
   
   Estimated completion: [time]
   
   Will update when:
   - [ ] Documentation complete
   - [ ] Tests complete
   - [ ] PR ready for review
   ```

3. **Avoid conflicts:**
   - Don't work on same file as open PR
   - Don't contradict recent changes
   - Coordinate with other agents if multiple active

---

### Code of Conduct for AI Agents

**Core principles:**

1. **Humility:**
   - "I don't know" is acceptable
   - Ask questions when uncertain
   - Defer to human expertise

2. **Transparency:**
   - Clearly mark AI-generated content
   - Document reasoning
   - Reference sources

3. **Caution:**
   - Prefer conservative changes
   - Avoid clever tricks
   - Maintain readability

4. **Rigor:**
   - Test thoroughly
   - Document completely
   - Verify all claims

5. **Respect for physics:**
   - TNFR physics is law
   - Never contradict established physics
   - Question understanding, not theory

---

### Copilot-Specific Guidelines

**When working on this repository:**

1. **Always read first:**
   - Start with `AGENTS.md`
   - Read relevant doc sections
   - Understand context before coding

2. **Physics-first mindset:**
   - Every change must have physics basis
   - Reference nodal equation when relevant
   - Think in terms of resonance/coherence

3. **Documentation-driven:**
   - Write docs before code
   - Ensure docs and code match
   - Update examples

4. **Test-driven:**
   - Write test first
   - Implement to pass test
   - Add edge cases

5. **Incremental:**
   - Small PRs (< 500 lines preferred)
   - One logical change per PR
   - Easy to review

---

### When to Escalate

**Immediately escalate (ask human) if:**

- ❌ Physics derivation unclear
- ❌ Multiple valid interpretations possible
- ❌ Breaking change seems necessary
- ❌ Conflict with existing constraints
- ❌ Test failure unexplained
- ❌ Documentation contradictory
- ❌ Performance regression significant
- ❌ Security concern identified

**How to escalate:**
```markdown
🚨 ESCALATION REQUIRED 🚨

## Issue
[Brief description]

## Why This Needs Human Decision
[Explanation]

## Context
[Relevant background]

## Attempted Solutions
[What I tried]

## Blocking
[What I cannot proceed with]

## Urgency
[LOW | MEDIUM | HIGH | CRITICAL]
```

---

## Roadmap: Future Constraints

### Potential Future Constraints

These are **SPECULATIVE** - not committed for inclusion.
Each would require full physics derivation and review process.

---

### U5: Multi-Scale Coherence (Under Consideration)

**Concept:** Nested EPIs must maintain coherence across scales

**Physical Basis (Preliminary):**
```
For nested structure: EPI_parent contains {EPI_child_1, EPI_child_2, ...}

Coherence requirement:
  C(EPI_parent) ≥ α · ∑ C(EPI_child_i)

Where α is scale factor (0.5-0.9 typically)

If violated → Fractal structure fragments
```

**Constraint (Hypothetical):**
```
If REMESH(EPI_parent, depth > 1):
  Then apply IL or THOL at parent level
  
Reason: Stabilize multi-scale structure
```

**Status:** 
- Canonicity: Possibly STRONG (from operational fractality)
- Research needed: Determine α, optimal stabilization
- Expected: v4.0+ (12+ months)

---

### U6: Temporal Ordering (Under Consideration)

**Concept:** Certain operators require minimum temporal separation

**Physical Basis (Preliminary):**
```
From bifurcation theory:
  After ZHIR or OZ, system needs relaxation time τ_relax
  
  If next destabilizer applied before τ_relax:
    → Chaotic dynamics (loss of coherence)
    
τ_relax ≈ 1/(2πνf) [one structural period]
```

**Constraint (Hypothetical):**
```
If OZ or ZHIR at position i:
  Then no {OZ, ZHIR, VAL} at position i+1, i+2
  
Reason: Allow structural relaxation
```

**Status:**
- Canonicity: Possibly MODERATE (from bifurcation theory)
- Research needed: Determine τ_relax precisely
- Expected: v5.0+ (24+ months)

---

### U7: Network Topology Constraints (Research Stage)

**Concept:** Coupling patterns must respect network structure

**Physical Basis (Very Preliminary):**
```
Coupling via UM or RA requires:
  Network path exists between nodes
  Phase propagation path ≤ max_path_length
  
Otherwise: Coupling is non-physical (action at distance)
```

**Constraint (Very Hypothetical):**
```
If UM(node_i, node_j):
  Then verify network_distance(i, j) ≤ coupling_range
  
Reason: Physical coupling requires local interaction
```

**Status:**
- Canonicity: Unknown (research in progress)
- Research needed: Fundamental review of coupling physics
- Expected: v6.0+ (36+ months) or never

---

### Why These Are Not U5-U7 Yet

**Requirements for inclusion:**

1. **Physical inevitability:** Must be proven from TNFR principles
2. **Independence:** Cannot be derived from U1-U4
3. **Universality:** Must apply across all domains
4. **Testability:** Must be verifiable experimentally
5. **Canonicity:** Must achieve ABSOLUTE or STRONG status

**Current status:**
- U5: Closest to meeting requirements (60% confidence)
- U6: Plausible but needs more physics work (40% confidence)
- U7: Speculative, may not be necessary (20% confidence)

---

### Community Input Welcome

Have an idea for future constraint?

**Process:**
1. Open GitHub discussion (not issue yet)
2. Title: "[RESEARCH] Potential constraint: [Name]"
3. Provide:
   - Physical intuition
   - Preliminary derivation (if any)
   - Example sequences that would be affected
   - Why current U1-U4 insufficient

**We will:**
- Engage in physics discussion
- Evaluate physical basis
- Determine if research should continue
- Possibly form working group

**Note:** Most ideas will be rejected. This is good! U1-U4 is complete for current physics understanding. Only add constraint if absolutely necessary.

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
