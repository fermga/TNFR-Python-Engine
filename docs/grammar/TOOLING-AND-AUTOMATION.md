# TNFR Grammar Tooling and Automation

**Complete guide to validation scripts, testing tools, and automation**

[🏠 Home](README.md) • [📚 Documentation](01-FUNDAMENTAL-CONCEPTS.md) • [🧪 Testing](06-VALIDATION-AND-TESTING.md)

---

## Purpose

This document describes all tools, scripts, and automation available for working with the TNFR grammar system. These tools ensure consistency, validate correctness, and streamline development workflows.

---

## Available Tools

### 1. Documentation Synchronization Tool

**Location**: `tools/sync_documentation.py`

**Purpose**: Maintains synchronization between grammar implementation and documentation.

**Documentation**: [tools/README_SYNC.md](../../tools/README_SYNC.md)

#### Usage

```bash
# Full synchronization (recommended)
python tools/sync_documentation.py --all

# Specific checks
python tools/sync_documentation.py --audit        # Audit code only
python tools/sync_documentation.py --validate     # Validate examples only
python tools/sync_documentation.py --crossref     # Check cross-references only
```

#### What It Checks

1. ✅ **Function Audit**: All public functions documented
2. ✅ **Operator Sets**: All 8 operator classifications present
3. ✅ **Example Execution**: All 8 examples run successfully
4. ✅ **Cross-References**: Links between docs and code
5. ✅ **Schema Validation**: JSON schemas match implementation

#### Output

- **Console**: Human-readable progress and summary
- **JSON Report**: `docs/grammar/SYNC_REPORT.json` with detailed results

#### When to Run

- ✅ After modifying `grammar.py`
- ✅ After updating documentation
- ✅ After adding/modifying examples
- ✅ Before committing grammar changes
- ✅ In CI/CD pipeline (automated)

---

### 2. Example Health Auditor

**Location**: `tools/audit_example_health.py`

**Purpose**: Deep analysis of example code quality and execution.

#### Usage

```bash
# Audit all examples
python tools/audit_example_health.py

# Audit specific example
python tools/audit_example_health.py docs/grammar/examples/01-basic-bootstrap.py

# Generate detailed report
python tools/audit_example_health.py --report audit_report.json
```

#### What It Checks

1. **Syntax**: Python syntax validity
2. **Imports**: All dependencies available
3. **Execution**: Runs without errors
4. **Output**: Produces expected telemetry
5. **Style**: PEP 8 compliance (optional)
6. **Documentation**: Comments and docstrings

#### Output

```
======================================================================
Example Health Audit Report
======================================================================

File: 01-basic-bootstrap.py
Status: ✅ HEALTHY

Checks:
  ✓ Syntax valid
  ✓ Imports available
  ✓ Executes successfully (exit code 0)
  ✓ Produces output (1247 bytes)
  ✓ Contains explanatory comments
  ✓ Demonstrates grammar rules clearly

Warnings:
  ⚠ Could add more inline comments (line 45-60)

Recommendations:
  • Consider adding telemetry output example
  • Could demonstrate error handling
```

---

### 3. Sequence Explorer

**Location**: `tools/sequence_explorer.py`

**Purpose**: Interactive tool for exploring valid operator sequences.

#### Usage

```bash
# Launch interactive explorer
python tools/sequence_explorer.py

# Generate all valid sequences of length N
python tools/sequence_explorer.py --generate --length 3

# Test specific sequence
python tools/sequence_explorer.py --test "emission,coherence,silence"

# Find sequences with specific operators
python tools/sequence_explorer.py --contains mutation --max-length 5
```

#### Features

1. **Interactive Mode**: Build sequences step by step with validation
2. **Generation Mode**: Enumerate all valid sequences up to length N
3. **Testing Mode**: Validate custom sequences
4. **Search Mode**: Find sequences containing specific operators
5. **Explain Mode**: Detailed explanation of why sequences pass/fail

#### Example Session

```
$ python tools/sequence_explorer.py

TNFR Sequence Explorer
======================================================================

Current sequence: []
EPI initial: 0.0

Available operators:
  1. Emission (AL)      - Generator
  2. Transition (NAV)   - Generator, Closure
  3. Recursivity (REMESH) - Generator, Closure

Select operator (1-3) or 'q' to quit: 1

Sequence: [Emission]
Status: ⚠ Needs closure

Available operators:
  1. Reception (EN)
  2. Coherence (IL)
  3. Dissonance (OZ)
  4. Silence (SHA)      - Closure
  ...

Select operator: 4

Sequence: [Emission, Silence]
Status: ⚠ Unstable (no stabilization)

Recommendation: Add Coherence before Silence

Continue? (y/n): y
...
```

---

### 4. TNFR Generator

**Location**: `tools/tnfr_generate`

**Purpose**: Code generation for common TNFR patterns.

#### Usage

```bash
# Generate boilerplate for new operator
./tools/tnfr_generate operator --name NewOperator --type stabilizer

# Generate test skeleton
./tools/tnfr_generate test --constraint U2 --operator coherence

# Generate example template
./tools/tnfr_generate example --pattern exploration

# Generate documentation stub
./tools/tnfr_generate docs --operator mutation
```

#### Templates Available

1. **Operator**: Complete operator class with contracts
2. **Test**: Unit test with common assertions
3. **Example**: Executable example with documentation
4. **Documentation**: Markdown stub with standard sections

---

### 5. Schema Validator

**Location**: Embedded in `tools/sync_documentation.py`

**Purpose**: Validates JSON schemas against implementation.

#### Usage

```bash
# Validate all schemas
python tools/sync_documentation.py --validate-schemas

# Check specific schema
python -c "
import json
from src.tnfr.operators.grammar import GENERATORS
schema = json.load(open('docs/grammar/schemas/canonical-operators.json'))
# Validation logic here
"
```

#### Schemas

1. **canonical-operators.json**: All 13 operators with metadata
2. **constraints-u1-u4.json**: Formal constraint definitions
3. **valid-sequences.json**: Catalog of canonical patterns

---

## Testing Infrastructure

### Unit Tests

**Location**: `tests/unit/operators/test_unified_grammar.py`

**Run all grammar tests**:
```bash
pytest tests/unit/operators/test_unified_grammar.py -v
```

**Run specific constraint tests**:
```bash
# U1 tests only
pytest tests/unit/operators/test_unified_grammar.py -k "u1"

# U2 convergence tests
pytest tests/unit/operators/test_unified_grammar.py -k "convergence"

# Bifurcation tests
pytest tests/unit/operators/test_unified_grammar.py -k "bifurcation"
```

**Run with coverage**:
```bash
pytest tests/unit/operators/test_unified_grammar.py --cov=src/tnfr/operators/grammar --cov-report=html
```

### Integration Tests

**Location**: `tests/integration/`

**Run integration tests**:
```bash
pytest tests/integration/ -v
```

### Example Tests

**All examples as tests**:
```bash
# Run all examples
for f in docs/grammar/examples/*.py; do python "$f"; done

# Check exit codes
for f in docs/grammar/examples/*.py; do
  python "$f" > /dev/null 2>&1 && echo "✓ $f" || echo "✗ $f FAILED"
done
```

---

## CI/CD Integration

### GitHub Actions

**Location**: `.github/workflows/`

#### Grammar Validation Workflow

```yaml
name: Validate Grammar System

on:
  push:
    paths:
      - 'src/tnfr/operators/grammar.py'
      - 'docs/grammar/**'
      - 'tests/unit/operators/test_unified_grammar.py'
  pull_request:
    paths:
      - 'src/tnfr/operators/**'
      - 'docs/grammar/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run grammar tests
        run: |
          pytest tests/unit/operators/test_unified_grammar.py -v
      
      - name: Validate documentation sync
        run: |
          python tools/sync_documentation.py --all
      
      - name: Validate examples
        run: |
          for f in docs/grammar/examples/*.py; do
            python "$f" || exit 1
          done
      
      - name: Check schema validity
        run: |
          python -c "import json; json.load(open('docs/grammar/schemas/canonical-operators.json'))"
```

---

## Pre-Commit Hooks

### Setup

**Install pre-commit**:
```bash
pip install pre-commit
pre-commit install
```

**Configuration** (`.pre-commit-config.yaml`):
```yaml
repos:
  - repo: local
    hooks:
      # Grammar validation
      - id: validate-grammar
        name: Validate TNFR Grammar
        entry: python tools/sync_documentation.py --audit
        language: system
        files: ^src/tnfr/operators/grammar\.py$
        pass_filenames: false
      
      # Example validation
      - id: validate-examples
        name: Validate Grammar Examples
        entry: python tools/sync_documentation.py --validate
        language: system
        files: ^docs/grammar/examples/.*\.py$
        pass_filenames: false
      
      # Schema validation
      - id: validate-schemas
        name: Validate JSON Schemas
        entry: python -m json.tool
        language: system
        files: ^docs/grammar/schemas/.*\.json$
```

---

## Development Workflows

### Workflow 1: Adding a New Operator

```bash
# 1. Generate operator skeleton
./tools/tnfr_generate operator --name MyOperator --type stabilizer

# 2. Implement operator in src/tnfr/operators/definitions.py
vim src/tnfr/operators/definitions.py

# 3. Add to appropriate operator set in grammar.py
vim src/tnfr/operators/grammar.py

# 4. Generate test skeleton
./tools/tnfr_generate test --operator myoperator

# 5. Write tests
vim tests/unit/operators/test_myoperator.py

# 6. Run tests
pytest tests/unit/operators/test_myoperator.py -v

# 7. Update documentation
vim docs/grammar/03-OPERATORS-AND-GLYPHS.md

# 8. Update schema
vim docs/grammar/schemas/canonical-operators.json

# 9. Run full validation
python tools/sync_documentation.py --all

# 10. Commit changes
git add .
git commit -m "feat: Add MyOperator stabilizer"
```

### Workflow 2: Modifying Grammar Constraint

```bash
# 1. Update constraint logic in grammar.py
vim src/tnfr/operators/grammar.py

# 2. Update corresponding tests
vim tests/unit/operators/test_unified_grammar.py

# 3. Run tests
pytest tests/unit/operators/test_unified_grammar.py -k "new_constraint" -v

# 4. Update documentation
vim docs/grammar/02-CANONICAL-CONSTRAINTS.md

# 5. Update schema
vim docs/grammar/schemas/constraints-u1-u4.json

# 6. Update affected examples
vim docs/grammar/examples/relevant-example.py

# 7. Validate all examples still work
for f in docs/grammar/examples/*.py; do python "$f"; done

# 8. Run full sync check
python tools/sync_documentation.py --all

# 9. Commit with detailed message
git commit -m "feat: Update U2 convergence constraint logic

- Modified convergence checking algorithm
- Added test cases for edge conditions
- Updated documentation with new examples
- Verified all existing examples still pass"
```

### Workflow 3: Adding Example

```bash
# 1. Generate example template
./tools/tnfr_generate example --pattern bifurcation

# 2. Write example code
vim docs/grammar/examples/04-my-example.py

# 3. Test example executes
python docs/grammar/examples/04-my-example.py

# 4. Add to example README
vim docs/grammar/examples/README.md

# 5. Run audit
python tools/audit_example_health.py docs/grammar/examples/04-my-example.py

# 6. Verify in sync tool
python tools/sync_documentation.py --validate

# 7. Commit
git add docs/grammar/examples/04-my-example.py
git commit -m "docs: Add bifurcation example"
```

---

## Troubleshooting

### Common Issues

#### Issue: "Sync tool reports missing function"

**Cause**: New function added to `grammar.py` without docstring

**Solution**:
```python
def my_new_function(param1, param2):
    """
    Brief description.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
    
    Returns
    -------
    type
        Description
    """
    # implementation
```

#### Issue: "Example validation fails"

**Cause**: Example has syntax error or missing import

**Solution**:
```bash
# Run example directly to see error
python docs/grammar/examples/failing-example.py

# Check imports
python -c "from src.tnfr.operators.grammar import *"

# Validate syntax
python -m py_compile docs/grammar/examples/failing-example.py
```

#### Issue: "Schema validation fails"

**Cause**: Schema doesn't match implementation

**Solution**:
```bash
# Check which operators are missing
python -c "
from src.tnfr.operators.grammar import GENERATORS, CLOSURES
import json
schema = json.load(open('docs/grammar/schemas/canonical-operators.json'))
schema_ops = {op['name'] for op in schema['operators']}
code_ops = GENERATORS | CLOSURES  # etc.
print('Missing:', code_ops - schema_ops)
"

# Update schema manually or regenerate
```

---

## Best Practices

### Development

1. ✅ **Run sync tool before committing**
   ```bash
   python tools/sync_documentation.py --all
   ```

2. ✅ **Test examples after grammar changes**
   ```bash
   for f in docs/grammar/examples/*.py; do python "$f" || exit 1; done
   ```

3. ✅ **Keep documentation synchronized**
   - Update docs when updating code
   - Update code when updating docs
   - Use sync tool to verify

4. ✅ **Write tests for new features**
   - Unit tests for operators
   - Integration tests for constraints
   - Example code as executable tests

### Documentation

1. ✅ **Use cross-references consistently**
   ```markdown
   [Grammar Implementation](../../src/tnfr/operators/grammar.py)
   [Fundamental Concepts](01-FUNDAMENTAL-CONCEPTS.md)
   ```

2. ✅ **Include executable code snippets**
   ```python
   # This should be runnable
   from tnfr.operators.grammar import validate_grammar
   from tnfr.operators.definitions import Emission, Silence
   
   sequence = [Emission(), Silence()]
   is_valid = validate_grammar(sequence, epi_initial=0.0)
   ```

3. ✅ **Document expected output**
   ```
   Expected output:
   ✓ Valid sequence
   C(t) = 0.85
   ```

### Testing

1. ✅ **Test positive and negative cases**
   ```python
   def test_valid_sequence():
       # Should pass
       assert validate_grammar([Emission(), Silence()], 0.0)
   
   def test_invalid_sequence():
       # Should raise ValueError
       with pytest.raises(ValueError):
           validate_grammar([Coherence(), Silence()], 0.0)
   ```

2. ✅ **Use descriptive test names**
   ```python
   def test_u1a_requires_generator_when_epi_zero():
       """U1a: Starting from EPI=0 requires generator."""
       # ...
   ```

3. ✅ **Test invariants explicitly**
   ```python
   def test_coherence_monotonicity():
       """Coherence must not decrease C(t)."""
       C_before = compute_coherence(G)
       apply_coherence(G, node)
       C_after = compute_coherence(G)
       assert C_after >= C_before
   ```

---

## Tool Reference

### Quick Command Reference

```bash
# Documentation sync (full)
python tools/sync_documentation.py --all

# Test all grammar
pytest tests/unit/operators/test_unified_grammar.py -v

# Test single constraint
pytest tests/unit/operators/test_unified_grammar.py -k "u2_convergence" -v

# Run all examples
for f in docs/grammar/examples/*.py; do python "$f"; done

# Audit specific example
python tools/audit_example_health.py docs/grammar/examples/01-basic-bootstrap.py

# Explore sequences interactively
python tools/sequence_explorer.py

# Generate operator skeleton
./tools/tnfr_generate operator --name MyOp --type stabilizer

# Validate schema
python -m json.tool docs/grammar/schemas/canonical-operators.json > /dev/null

# Check test coverage
pytest tests/unit/operators/ --cov=src/tnfr/operators --cov-report=term-missing
```

---

## Metrics and Monitoring

### Quality Metrics

Track these metrics to ensure system health:

| Metric | Command | Target |
|--------|---------|--------|
| Test Coverage | `pytest --cov --cov-report=term` | > 95% |
| Documentation Coverage | `python tools/sync_documentation.py --audit` | 100% |
| Example Success Rate | `python tools/sync_documentation.py --validate` | 100% |
| Link Integrity | Manual check or custom script | 100% |
| Schema Validity | `python -m json.tool <schema>` | Valid JSON |

### Continuous Monitoring

```bash
# Weekly health check script
#!/bin/bash
echo "=== TNFR Grammar System Health Check ==="
echo ""

echo "1. Running full test suite..."
pytest tests/unit/operators/test_unified_grammar.py -v --tb=short

echo ""
echo "2. Validating documentation sync..."
python tools/sync_documentation.py --all

echo ""
echo "3. Testing all examples..."
for f in docs/grammar/examples/*.py; do
  python "$f" > /dev/null 2>&1 && echo "✓ $f" || echo "✗ $f FAILED"
done

echo ""
echo "4. Checking schemas..."
for f in docs/grammar/schemas/*.json; do
  python -m json.tool "$f" > /dev/null && echo "✓ $f" || echo "✗ $f INVALID"
done

echo ""
echo "=== Health Check Complete ==="
```

---

## Resources

### Documentation
- **Main README**: [docs/grammar/README.md](README.md)
- **Sync Tool Guide**: [tools/README_SYNC.md](../../tools/README_SYNC.md)
- **Testing Guide**: [docs/grammar/06-VALIDATION-AND-TESTING.md](06-VALIDATION-AND-TESTING.md)

### Code
- **Sync Tool**: `tools/sync_documentation.py`
- **Example Auditor**: `tools/audit_example_health.py`
- **Sequence Explorer**: `tools/sequence_explorer.py`
- **Generator**: `tools/tnfr_generate`

### Support
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Documentation**: This file and README.md

---

<div align="center">

**Automation enables consistency. Consistency enables trust.**

---

*Reality is resonance. Validate accordingly.*

</div>
