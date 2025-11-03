# Type Stub Automation Workflow

> **Complete guide for automated `.pyi` stub file generation and synchronization**

---

## Overview

This project uses **automated stub generation** to maintain type safety and prevent drift between Python implementations (`.py`) and type stubs (`.pyi`). The automation ensures:

1. **All Python modules have corresponding type stubs**
2. **Stub files stay synchronized with implementations**
3. **Changes are validated in CI/CD pipeline**
4. **Pre-commit hooks prevent accidental drift**

---

## Quick Commands

```bash
# Generate missing stub files
make stubs

# Check for missing stubs (exit code 1 if any missing)
make stubs-check

# Check if stubs are synchronized (exit code 1 if outdated)
make stubs-check-sync

# Regenerate outdated stub files
make stubs-sync

# Display all available commands
make help
```

---

## How It Works

### 1. Stub Generation Script

**Location**: `scripts/generate_stubs.py`

The script uses `mypy.stubgen` to automatically generate `.pyi` stub files from Python implementations:

```python
# Usage examples
python scripts/generate_stubs.py                    # Generate missing stubs
python scripts/generate_stubs.py --check            # Check for missing stubs
python scripts/generate_stubs.py --check-sync       # Check synchronization
python scripts/generate_stubs.py --sync             # Regenerate outdated stubs
python scripts/generate_stubs.py --dry-run          # Preview without changes
```

### 2. Pre-commit Hook

**Location**: `.pre-commit-config.yaml`

Runs automatically before each commit to prevent committing code without stubs:

```yaml
- id: check-stubs
  name: Check .pyi stub files
  entry: python scripts/generate_stubs.py --check
  language: system
  pass_filenames: false
  files: ^src/tnfr/.*\.py$
```

**Behavior:**
- Triggered on changes to `.py` files in `src/tnfr/`
- Checks if corresponding `.pyi` files exist
- Blocks commit if stubs are missing
- Provides clear error message with fix command

### 3. CI/CD Integration

**Location**: `.github/workflows/type-check.yml`

Validates stub completeness and synchronization in CI pipeline:

```yaml
- name: Check stub files exist
  run: python scripts/generate_stubs.py --check

- name: Check stub file synchronization
  run: python scripts/generate_stubs.py --check-sync
```

**Behavior:**
- Runs on all PRs and pushes to main/master
- First check: Ensures all `.py` files have `.pyi` stubs
- Second check: Ensures `.pyi` files are not older than `.py` files
- Blocks PR merge if checks fail

---

## Workflow Scenarios

### Scenario 1: Creating a New Module

When you create a new Python module:

```bash
# 1. Create your module
touch src/tnfr/mymodule.py
# ... implement your code ...

# 2. Generate stub file
make stubs
# Creates: src/tnfr/mymodule.pyi

# 3. Review and commit
git add src/tnfr/mymodule.py src/tnfr/mymodule.pyi
git commit -m "feat: add new mymodule"
```

### Scenario 2: Modifying an Existing Module

When you modify a Python file:

```bash
# 1. Make your changes
vim src/tnfr/mathematics/operators_factory.py

# 2. Check if stub needs regeneration
make stubs-check-sync

# 3. If outdated, regenerate
make stubs-sync

# 4. Review changes and commit
git add src/tnfr/mathematics/operators_factory.py
git add src/tnfr/mathematics/operators_factory.pyi
git commit -m "feat: enhance operator factory validation"
```

### Scenario 3: Pre-commit Hook Blocked Your Commit

If pre-commit hook reports missing stubs:

```bash
# Hook output:
# Error: Python file without corresponding .pyi stub
#   - src/tnfr/newmodule.py
# 
# Run 'python scripts/generate_stubs.py' to generate missing stubs

# Solution:
make stubs
git add src/tnfr/newmodule.pyi
git commit  # Retry commit
```

### Scenario 4: CI Check Failed

If CI reports stub issues:

```bash
# CI output:
# Found 2 outdated stub files:
#   - src/tnfr/module1.py
#   - src/tnfr/module2.py
# 
# Run 'python scripts/generate_stubs.py --sync' to update outdated stubs

# Solution:
git fetch origin
git checkout your-branch
make stubs-sync
git add src/tnfr/module1.pyi src/tnfr/module2.pyi
git commit -m "chore: synchronize type stubs"
git push
```

---

## Stub Generation Details

### What Gets Generated

For a Python file `module.py`, `stubgen` generates `module.pyi` containing:

1. **Function signatures** with type annotations
2. **Class definitions** with method signatures
3. **Module-level constants** and variables
4. **Import statements** for types
5. **`__all__` export list** if defined

### Example

**Input (`operators_factory.py`):**

```python
def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = 0.1,
) -> CoherenceOperator:
    """Return a Hermitian positive semidefinite CoherenceOperator."""
    # ... implementation ...
    return operator
```

**Generated (`operators_factory.pyi`):**

```python
import numpy as np
from .operators import CoherenceOperator

__all__ = ['make_coherence_operator', 'make_frequency_operator']

def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = 0.1
) -> CoherenceOperator: ...
```

### Excluded Files

The following are automatically excluded from stub generation:

- Private modules (starting with `_`)
- `__pycache__` directories
- Test files (typically in `tests/`)
- `__init__.py` files (handled specially)

---

## Synchronization Logic

### Timestamp-Based Detection

The script compares modification times:

```python
py_mtime = py_file.stat().st_mtime
pyi_mtime = pyi_file.stat().st_mtime

# Outdated if .py modified > 1 second after .pyi
if py_mtime - pyi_mtime > 1.0:
    # Regenerate stub
```

**Tolerance**: 1 second to avoid false positives from filesystem precision.

### Why Synchronization Matters

Outdated stubs can cause:

1. **Type checking errors**: Mypy sees wrong signatures
2. **IDE confusion**: Autocomplete shows incorrect types
3. **Documentation drift**: Stubs serve as documentation
4. **Runtime surprises**: Expectations don't match reality

---

## Troubleshooting

### Problem: Stub generation fails

```bash
✗ Failed to generate stub for tnfr.mymodule
  Error: No module named 'tnfr.mymodule'
```

**Causes:**
- Module not installed
- Import errors in module
- Syntax errors preventing parsing

**Solution:**
```bash
# Install package in development mode
pip install -e .

# Check for import errors
python -c "import tnfr.mymodule"

# Check syntax
python -m py_compile src/tnfr/mymodule.py
```

### Problem: Stubs appear outdated but aren't

```bash
Found 5 outdated stub files:
  - src/tnfr/module.py
```

**Cause:** Filesystem timestamp inconsistencies

**Solution:**
```bash
# Touch .pyi files to update timestamps
find src/tnfr -name "*.pyi" -exec touch {} \;

# Verify synchronization
make stubs-check-sync
```

### Problem: Pre-commit hook too slow

**Cause:** Hook runs on every commit

**Solution:**
```bash
# Skip hook for emergency commits
git commit --no-verify -m "hotfix: critical bug"

# But remember to fix stubs later!
make stubs-sync
git commit -m "chore: synchronize stubs"
```

### Problem: Merge conflict in .pyi file

**Cause:** Both branches modified the same module

**Solution:**
```bash
# Resolve conflict in .py file first
vim src/tnfr/module.py
git add src/tnfr/module.py

# Regenerate stub from resolved .py
make stubs-sync

# Complete merge
git add src/tnfr/module.pyi
git commit
```

---

## Best Practices

### Do's ✓

1. **Always generate stubs after adding new modules**
   ```bash
   make stubs
   ```

2. **Regenerate stubs after significant signature changes**
   ```bash
   make stubs-sync
   ```

3. **Review generated stubs before committing**
   ```bash
   git diff src/tnfr/module.pyi
   ```

4. **Run full check before pushing**
   ```bash
   make stubs-check-sync
   mypy src/tnfr
   ```

5. **Keep .py and .pyi files in sync in commits**
   ```bash
   git add src/tnfr/module.py src/tnfr/module.pyi
   ```

### Don'ts ✗

1. **Don't manually edit generated stubs**
   - Use type annotations in `.py` instead
   - Regenerate stubs to reflect changes

2. **Don't skip pre-commit hooks habitually**
   - Leads to drift accumulation
   - CI will catch it anyway

3. **Don't commit without checking synchronization**
   ```bash
   # Bad: Commit only .py
   git add src/tnfr/module.py
   git commit
   
   # Good: Check and update stubs
   make stubs-check-sync
   make stubs-sync
   git add src/tnfr/module.py src/tnfr/module.pyi
   git commit
   ```

4. **Don't ignore stub generation errors**
   - Fix the underlying issue
   - Errors indicate problems with the code

---

## Advanced Usage

### Generate Stubs for Specific Modules

```bash
# Generate stub for single module
stubgen -p tnfr.mathematics.operators_factory -o src

# Generate stubs for package
stubgen -p tnfr.mathematics -o src
```

### Custom Stub Generation

For special cases requiring manual stubs:

1. Create stub manually in correct location
2. Document why automatic generation doesn't work
3. Add special handling in `scripts/generate_stubs.py` if needed

Example comment in manual stub:
```python
# NOTE: Manual stub - automatic generation fails due to dynamic __getattr__
```

### Dry-Run Mode

Preview what would be generated without making changes:

```bash
# See what would be generated
python scripts/generate_stubs.py --dry-run

# See what would be synchronized
python scripts/generate_stubs.py --sync --dry-run
```

---

## Integration with Other Tools

### Mypy

Stubs are automatically discovered by mypy:

```bash
# Type check using stubs
mypy src/tnfr
```

### IDEs

Most Python IDEs (PyCharm, VSCode) automatically use `.pyi` files for:
- Autocomplete
- Type hints
- Quick documentation
- Refactoring

### Documentation

Sphinx and other doc generators can use stubs:

```python
# In conf.py
autodoc_typehints = 'description'
```

---

## Monitoring & Metrics

Track stub coverage:

```bash
# Count .py files
PY_COUNT=$(find src/tnfr -name "*.py" ! -name "__*" | wc -l)

# Count .pyi files
PYI_COUNT=$(find src/tnfr -name "*.pyi" ! -name "__*" | wc -l)

# Coverage percentage
echo "Stub coverage: $((PYI_COUNT * 100 / PY_COUNT))%"
```

---

## Future Enhancements

Potential improvements to the automation:

1. **Automated PR creation**: Bot creates PR when stubs drift
2. **Incremental generation**: Only regenerate changed modules
3. **Quality checks**: Validate stub quality beyond existence
4. **Documentation integration**: Generate API docs from stubs
5. **Performance optimization**: Cache stubgen results

---

## Related Documentation

- [Factory Patterns Guide](FACTORY_PATTERNS.md) - Factory design patterns requiring stubs
- [Factory Quick Reference](FACTORY_QUICK_REFERENCE.md) - Quick guide including stub generation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - General contribution guidelines
- [Type-check workflow](../.github/workflows/type-check.yml) - CI configuration

---

## Support

If you encounter issues with stub generation:

1. Check this documentation
2. Review existing GitHub issues
3. Run diagnostics:
   ```bash
   make stubs-check
   make stubs-check-sync
   mypy src/tnfr
   ```
4. Open an issue with:
   - Error message
   - Command that failed
   - Python version
   - Module causing issues
