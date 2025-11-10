# Documentation Synchronization Tool

**Purpose**: Centralized tool to maintain synchronization between TNFR grammar implementation and documentation.

## Usage

### Full Synchronization (Default)

```bash
python tools/sync_documentation.py
# or
python tools/sync_documentation.py --all
```

This performs:
1. ✅ Audit of `grammar.py` (functions, operator sets)
2. ✅ Docstring quality analysis
3. ✅ Cross-reference checking
4. ✅ Example validation (execution tests)
5. ✅ Schema validation

### Specific Tasks

**Audit only:**
```bash
python tools/sync_documentation.py --audit
```

**Validate examples only:**
```bash
python tools/sync_documentation.py --validate
```

## Output

### Console Report

```
======================================================================
TNFR Documentation Synchronization - Full Sync
======================================================================

[1/5] Auditing grammar.py...
  ✓ Found 8 operator sets
  ✓ Audited 17 functions

[2/5] Analyzing docstrings...
  ✓ Analyzed 9 critical functions

[3/5] Checking cross-references...
  ✓ Found 35 cross-references

[4/5] Validating examples...
    Testing u1-initiation-closure-examples.py... ✓
    Testing u2-convergence-examples.py... ✓
    (...)

[5/5] Validating schema...
  ✓ Schema contains 13 operators

======================================================================
SYNCHRONIZATION REPORT
======================================================================

📊 SUMMARY
----------------------------------------------------------------------
Functions: 17/17 documented (100.0%)
Examples: 8/8 passing (100.0%)
Cross-references: 35

✅ SYNC COMPLETE: All critical checks passed
```

### JSON Report

Detailed report saved to: `docs/grammar/SYNC_REPORT.json`

```json
{
  "functions": {
    "validate_grammar": {
      "signature": "...",
      "docstring": "...",
      "has_docstring": true
    },
    ...
  },
  "operator_sets": {
    "GENERATORS": ["emission", "transition", "recursivity"],
    ...
  },
  "examples": {
    "u1-initiation-closure-examples.py": {
      "executes": true,
      "exit_code": 0
    },
    ...
  },
  "cross_references": [...],
  "issues": [...]
}
```

## What It Checks

### 1. Function Audit

- ✅ All public functions have docstrings
- ✅ Docstrings include Parameters/Returns sections
- ✅ Function signatures match documentation

**Functions audited:**
- `validate_grammar()`
- `GrammarValidator.validate()`
- `GrammarValidator.validate_initiation()`
- `GrammarValidator.validate_closure()`
- `GrammarValidator.validate_convergence()`
- `GrammarValidator.validate_resonant_coupling()`
- `GrammarValidator.validate_bifurcation_triggers()`
- `GrammarValidator.validate_transformer_context()`
- `GrammarValidator.validate_remesh_amplification()`
- Plus helper functions

### 2. Operator Sets

- ✅ All 8 operator sets defined
- ✅ Sets match TNFR physics
- ✅ No duplicates or conflicts

**Sets checked:**
- GENERATORS
- CLOSURES
- STABILIZERS
- DESTABILIZERS
- COUPLING_RESONANCE
- BIFURCATION_TRIGGERS
- BIFURCATION_HANDLERS
- TRANSFORMERS

### 3. Examples

Tests that all example files execute successfully:

- `u1-initiation-closure-examples.py`
- `u2-convergence-examples.py`
- `u3-resonant-coupling-examples.py`
- `u4-bifurcation-examples.py`
- `01-basic-bootstrap.py`
- `02-intermediate-exploration.py`
- `03-advanced-bifurcation.py`
- `all-operators-catalog.py`

**Failures reported with:**
- Exit code
- Error message
- Severity level

### 4. Cross-References

Scans documentation for:
- References to `grammar.py`
- References to specific functions
- References from examples to code

**Reported:**
- Doc → Code references
- Code → Doc references
- Bidirectional mappings

### 5. Schema Validation

Checks `canonical-operators.json`:
- All operators in code are in schema
- All operators in schema exist in code
- Classifications match operator sets

## Integration

### Pre-Commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: sync-docs
      name: Check documentation sync
      entry: python tools/sync_documentation.py --all
      language: system
      pass_filenames: false
```

### CI/CD

Add to GitHub Actions:

```yaml
- name: Validate Documentation Sync
  run: python tools/sync_documentation.py --all
```

### Development Workflow

**Before committing grammar changes:**

```bash
# 1. Make changes to grammar.py
vim src/tnfr/operators/grammar.py

# 2. Update docstrings if needed
# 3. Run sync check
python tools/sync_documentation.py --all

# 4. Fix any issues reported
# 5. Commit
git add src/tnfr/operators/grammar.py
git commit -m "Update grammar validation"
```

## Issue Reporting

### Severity Levels

**HIGH**: Must fix before release
- Missing docstrings on critical functions
- Examples that fail to execute
- Missing schema file

**MEDIUM**: Should fix soon
- Incomplete docstrings (missing Parameters/Returns)
- Partial cross-references

**LOW**: Optional improvements
- Extra operators in schema (deprecated but not removed)
- Minor documentation inconsistencies

### Example Issues

```json
{
  "type": "example_failure",
  "file": "u3-resonant-coupling-examples.py",
  "error": "AttributeError: 'Coupling' object has no...",
  "severity": "high"
}
```

```json
{
  "type": "incomplete_docstring",
  "function": "validate_grammar",
  "missing": "parameters",
  "severity": "medium"
}
```

## Extending the Tool

### Add New Checks

```python
def check_new_aspect(self):
    """Add custom validation."""
    print("  Checking new aspect...")
    
    # Perform checks
    if issue_found:
        self.audit_report["issues"].append({
            "type": "new_issue_type",
            "description": "...",
            "severity": "medium"
        })
    
    print(f"  ✓ Check complete")
```

Then add to `run_full_sync()`:

```python
def run_full_sync(self):
    # ... existing checks ...
    
    print("\n[6/6] Checking new aspect...")
    self.check_new_aspect()
```

## Maintenance

**Run regularly:**
- After changing `grammar.py`
- After updating documentation
- After modifying examples
- Before releases

**Keep up to date:**
- Update critical function list as needed
- Add new example files to validation
- Adjust severity levels based on impact

---

**Last Updated**: 2025-11-10  
**Version**: 1.0  
**Status**: ✅ Production Ready
