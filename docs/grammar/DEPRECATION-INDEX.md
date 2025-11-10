# Grammar System Deprecation Index

**Purpose**: Track all deprecated grammar components and their replacements

**Status**: ✅ Active - Updated for v2.0.0

**Last Updated**: 2024-11-10

---

## Overview

This document catalogs all deprecated grammar-related components, provides their replacements, and outlines removal timelines.

**Deprecation Policy**:
- Grace period: 1-2 major versions
- Clear warnings in code and docs
- Migration path provided
- Final removal announced 6+ months in advance

---

## Deprecated Documentation

### Root-Level Files

#### GRAMMAR_MIGRATION_GUIDE.md

- **Status**: ⚠️ DEPRECATED (v2.0.0)
- **Superseded by**: `docs/grammar/07-MIGRATION-AND-EVOLUTION.md`
- **Reason**: Consolidated into canonical grammar documentation
- **Removal planned**: v4.0.0
- **Action required**: Update links to point to new location
- **Migration**: All content migrated to canonical docs

---

### Legacy Documentation References

#### C1-C3 Constraint Names

- **Status**: ❌ OBSOLETE (v2.0.0)
- **Replaced by**: U1-U4 unified constraints
- **Reason**: Non-unified, incomplete physics coverage
- **Removal**: Already removed from canonical docs
- **Kept**: Historical references in migration guide only

**Mapping:**
- C1 (EXISTENCE & CLOSURE) → U1 (STRUCTURAL INITIATION & CLOSURE)
- C2 (BOUNDEDNESS) → U2 (CONVERGENCE & BOUNDEDNESS)
- C3 (THRESHOLD PHYSICS) → U4 (BIFURCATION DYNAMICS)

#### RC1-RC4 Constraint Names

- **Status**: ❌ OBSOLETE (v2.0.0)
- **Replaced by**: U1-U4 unified constraints
- **Reason**: Parallel system causing confusion
- **Removal**: Already removed from canonical docs
- **Kept**: Historical references in migration guide only

**Mapping:**
- RC1 (Initialization) → U1a (Initiation)
- RC2 (Convergence) → U2 (CONVERGENCE & BOUNDEDNESS)
- RC3 (Phase Verification) → U3 (RESONANT COUPLING)
- RC4 (Bifurcation Limits) → U4a (Bifurcation Triggers)

---

## Deprecated Code

### Modules

#### src/tnfr/operators/canonical_grammar.py

- **Status**: ❌ REMOVED (v2.0.0)
- **Replaced by**: `src/tnfr/operators/grammar.py` + `unified_grammar.py`
- **Reason**: Parallel implementation consolidated
- **Removal**: Already removed
- **Migration**: Use `from tnfr.operators.unified_grammar import validate_grammar`

---

### Functions

#### validate_sequence() [deprecated signature]

```python
# OLD (DEPRECATED)
from tnfr.operators.grammar import validate_sequence
result = validate_sequence(operators)

# NEW (CURRENT)
from tnfr.operators.unified_grammar import validate_grammar
result = validate_grammar(operators, epi_initial=0.0)
```

- **Status**: ⚠️ DEPRECATED (v2.0.0)
- **Replaced by**: `validate_grammar(sequence, epi_initial=0.0)`
- **Reason**: New signature adds required EPI parameter
- **Removal planned**: v4.0.0
- **Warning**: DeprecationWarning emitted
- **Migration**: Add `epi_initial` parameter (default 0.0)

#### validate_canonical()

```python
# OLD (DEPRECATED)
from tnfr.operators.canonical_grammar import validate_canonical
result = validate_canonical(operators)

# NEW (CURRENT)
from tnfr.operators.unified_grammar import validate_grammar
result = validate_grammar(operators, epi_initial=0.0)
```

- **Status**: ❌ REMOVED (v2.0.0)
- **Replaced by**: `validate_grammar()`
- **Reason**: Module removed, functionality consolidated
- **Removal**: Already removed
- **Migration**: Use unified grammar validation

---

### Variables and Constants

#### GENERATORS_RC, STABILIZERS_RC, etc.

```python
# OLD (DEPRECATED)
from tnfr.operators.canonical_grammar import GENERATORS_RC

# NEW (CURRENT)
from tnfr.operators.unified_grammar import GENERATORS
```

- **Status**: ❌ REMOVED (v2.0.0)
- **Replaced by**: Sets without `_RC` suffix
- **Reason**: Naming confusion with dual systems
- **Removal**: Already removed
- **Migration**: Remove `_RC` suffix from imports

---

## Deprecated Tests

### Test Files

#### tests/unit/operators/test_canonical_grammar_legacy.py

- **Status**: ✅ KEPT (with deprecation notice)
- **Purpose**: Test legacy compatibility layer
- **Reason**: Ensures backward compatibility works
- **Removal planned**: v4.0.0 (when compatibility layer removed)
- **Note**: Emits DeprecationWarnings by design

#### tests/unit/operators/test_grammar_c1_c3_deprecation.py

- **Status**: ✅ KEPT (with deprecation notice)
- **Purpose**: Test C1-C3 to U1-U4 migration warnings
- **Reason**: Validates deprecation path
- **Removal planned**: v4.0.0
- **Note**: Tests that old names emit warnings

---

## Deprecated Examples

### Example Files

#### examples/grammar_migration_example.py

- **Status**: ✅ KEPT (marked as deprecated)
- **Purpose**: Show C1-C3/RC1-RC4 to U1-U4 migration
- **Reason**: Educational, helps users migrate
- **Removal planned**: v4.0.0
- **Note**: File header clearly marked as deprecated
- **Replacement**: See `docs/grammar/examples/` for canonical examples

---

## Terminology Changes

### Deprecated Terms

| Old Term | New Term | Status | Notes |
|----------|----------|--------|-------|
| "Canonical grammar" | "Unified grammar" | ⚠️ TRANSITIONING | Avoid confusion with "canonical operators" |
| "Constraint C1" | "U1a: Initiation" | ❌ OBSOLETE | Use unified naming |
| "Constraint C2" | "U2: Convergence" | ❌ OBSOLETE | More descriptive name |
| "Constraint C3" | "U1b: Closure" | ❌ OBSOLETE | Part of unified U1 |
| "Resonant constraints" | "Grammar constraints" | ⚠️ TRANSITIONING | Simpler terminology |
| "RC1-RC4" | "U1-U4" | ❌ OBSOLETE | Unified system |

---

## Migration Timeline

### v2.0.0 (Released: 2024-11)

**Introduced:**
- Unified grammar (U1-U4)
- New `validate_grammar()` function
- Consolidated documentation

**Deprecated:**
- Old constraint names (C1-C3, RC1-RC4)
- `validate_sequence()` old signature
- `canonical_grammar.py` module

**Removed:**
- Parallel RC1-RC4 implementation
- Redundant operator sets

---

### v3.0.0 (Planned: 2025-Q2)

**Will introduce:**
- Possible new constraints (if physics requires)
- Enhanced error messages

**Will deprecate:**
- Remaining compatibility layers (if any)
- Old example files

**May remove:**
- None (grace period continues)

---

### v4.0.0 (Planned: 2025-Q4)

**Will remove:**
- All v2.0.0 deprecated items
- `validate_sequence()` old signature
- Legacy test files
- Migration example files
- `GRAMMAR_MIGRATION_GUIDE.md` (root level)

**Breaking changes:**
- Old API no longer available
- Must use U1-U4 names exclusively

---

## Deprecation Warnings

### How Warnings Are Emitted

All deprecated functions emit Python `DeprecationWarning`:

```python
import warnings

def old_function():
    warnings.warn(
        "old_function() is deprecated. Use new_function() instead.",
        DeprecationWarning,
        stacklevel=2
    )
```

**To see warnings during testing:**
```bash
pytest -W default::DeprecationWarning
```

**To convert warnings to errors (strict mode):**
```bash
pytest -W error::DeprecationWarning
```

---

## Migration Checklist

### For Code

- [ ] Update imports to use `unified_grammar`
- [ ] Replace `validate_sequence()` with `validate_grammar()`
- [ ] Add `epi_initial` parameter to validation calls
- [ ] Update operator set names (remove `_RC` suffixes)
- [ ] Replace C1-C3/RC1-RC4 references with U1-U4
- [ ] Run tests with warnings visible
- [ ] Fix all DeprecationWarnings

### For Documentation

- [ ] Update links to point to `docs/grammar/07-MIGRATION-AND-EVOLUTION.md`
- [ ] Replace old constraint names with U1-U4
- [ ] Remove references to `canonical_grammar.py`
- [ ] Update code examples to use new API
- [ ] Add migration notes if creating new docs

### For Tests

- [ ] Update test imports
- [ ] Replace old constraint names in assertions
- [ ] Update test docstrings with new terminology
- [ ] Ensure tests pass without warnings (unless testing deprecation)

---

## Support and Questions

### Getting Help

**Found deprecated code?**
1. Check this index for replacement
2. See `docs/grammar/07-MIGRATION-AND-EVOLUTION.md` for details
3. Open GitHub issue if unclear

**Need migration assistance?**
1. Review migration guide
2. Check examples in `docs/grammar/examples/`
3. Ask in GitHub Discussions

**Reporting deprecation issues:**
- Label: `grammar-system`, `deprecation`
- Template: Include old code + error message
- Response time: 1-2 business days

---

## Deprecation Policy

### Principles

1. **Clear warnings**: All deprecated items emit warnings
2. **Grace period**: Minimum 1 major version (6+ months)
3. **Migration path**: Always provide replacement
4. **Documentation**: Always update docs simultaneously
5. **No surprise removal**: Announce 6+ months before removal

### Exception: Critical Bugs

Physics errors or security issues may require faster removal:
- Immediate patch release
- Extended grace period if possible
- Clear announcement and migration guide
- Direct user notification if breaking

---

## Version-Specific Notes

### v2.0.0 Notes

**Why so many changes?**
- Consolidated two parallel systems (C1-C3, RC1-RC4)
- Physics derivations incomplete in old system
- Missing phase verification (U3)
- Unclear bifurcation dynamics (U4)

**What stayed the same?**
- Core operators (AL, EN, IL, etc.)
- Physical principles (nodal equation)
- API surface (new parameters, not removed functions)

**Impact:**
- Most code works without changes
- Some sequences now invalid (correctly so, per physics)
- Better error messages help fix issues

---

## Future Deprecations

### None Currently Planned

**As of v2.0.0:**
- U1-U4 is considered stable
- No further deprecations planned
- API is finalized

**Possible future deprecations:**
- If physics error found in U1-U4 (requires review committee)
- If new constraint (U5+) requires API change
- Announced 12+ months in advance

---

## Appendix: Complete Mapping

### Constraint Mapping (v1.x → v2.x)

```
C1: EXISTENCE & CLOSURE
├─→ U1a: Initiation (generators)
└─→ U1b: Closure (closures)

C2: BOUNDEDNESS
└─→ U2: CONVERGENCE & BOUNDEDNESS (stabilizers)

C3: THRESHOLD PHYSICS
└─→ U4: BIFURCATION DYNAMICS
    ├─→ U4a: Triggers need handlers
    └─→ U4b: Transformers need context

RC1: Initialization
└─→ U1a: Initiation

RC2: Convergence
└─→ U2: CONVERGENCE & BOUNDEDNESS

RC3: Phase Verification
└─→ U3: RESONANT COUPLING (NEW)

RC4: Bifurcation Limits
└─→ U4a: Triggers need handlers
```

### Function Mapping

```
validate_sequence(ops)
└─→ validate_grammar(ops, epi_initial=0.0)

validate_canonical(ops)
└─→ validate_grammar(ops, epi_initial=0.0)

CanonicalGrammarValidator.validate()
└─→ GrammarValidator.validate()  (in unified_grammar)
```

### Import Mapping

```
from tnfr.operators.grammar import validate_sequence
└─→ from tnfr.operators.unified_grammar import validate_grammar

from tnfr.operators.canonical_grammar import *
└─→ from tnfr.operators.unified_grammar import *

from tnfr.operators.grammar import GENERATORS
└─→ from tnfr.operators.unified_grammar import GENERATORS
(No change, but now unified)
```

---

<div align="center">

**Deprecations are transitions, not endings.**

---

*When in doubt, check the migration guide.*

</div>
