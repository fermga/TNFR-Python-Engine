# TNFR Development Workflow

> **Complete guide for contributors: workflows, best practices, and CI/CD integration**

This guide provides a comprehensive overview of development workflows for TNFR Python Engine contributors, from initial setup through PR submission.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Workflow Patterns](#workflow-patterns)
4. [Code Quality](#code-quality)
5. [Documentation](#documentation)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Release Process](#release-process)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

```bash
# Required
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- git
- pip

# Recommended
- pyenv (for managing Python versions)
- direnv (for managing environment variables)
- pre-commit (installed automatically with dev dependencies)
```

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/fermga/TNFR-Python-Engine.git
cd TNFR-Python-Engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in editable mode with all dev dependencies
pip install -e .[dev-full]

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify installation
python -c "import tnfr; print(tnfr.__version__)"
pytest tests/ -k "test_basic" --collect-only
make help
```

### Repository Structure

```
TNFR-Python-Engine/
├── src/tnfr/               # Source code
│   ├── operators/          # Structural operators (Emission, Reception, etc.)
│   ├── mathematics/        # Factory functions, generators
│   ├── metrics/            # C(t), Si, phase calculations
│   ├── dynamics/           # ΔNFR computation
│   ├── utils/              # Utilities (cache, validation, etc.)
│   ├── backends/           # Math backends (NumPy, JAX, PyTorch)
│   └── ...
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── ci/                 # CI-specific tests
├── docs/                   # Documentation
│   ├── source/             # Sphinx/MkDocs source
│   │   ├── getting-started/
│   │   ├── user-guide/
│   │   ├── advanced/       # ← Your consolidated guides live here
│   │   └── ...
│   └── ...
├── scripts/                # Development scripts
│   ├── generate_stubs.py   # Type stub generation
│   ├── verify_internal_references.py
│   └── ...
├── .github/                # GitHub Actions workflows
├── pyproject.toml          # Project configuration
├── Makefile                # Common tasks
└── README.md
```

---

## Development Environment

### Development Dependencies

```bash
# Core development tools
pip install -e .[dev-core]

# Full development suite (recommended)
pip install -e .[dev-full]

# Specific tool groups
pip install -e .[typecheck]    # mypy, type stubs
pip install -e .[test-all]     # pytest and all plugins
pip install -e .[docs]         # Sphinx, MkDocs
pip install -e .[lint]         # Linters and formatters
```

### Environment Variables

Create `.env` file for local configuration:

```bash
# .env (copy from .env.example)
TNFR_BACKEND=numpy          # or jax, torch
TNFR_CACHE_SIZE=128         # Cache entries
TNFR_LOG_LEVEL=INFO         # DEBUG, INFO, WARNING, ERROR
TNFR_PROFILE=false          # Enable profiling
```

### Editor Configuration

#### VS Code

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

#### PyCharm

1. Open Settings → Project → Python Interpreter
2. Add interpreter from `venv/bin/python`
3. Enable pytest as test runner
4. Configure mypy as external tool

---

## Workflow Patterns

### Feature Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes
# Edit files...

# 3. Run tests frequently
pytest tests/ -n auto

# 4. Generate/update stubs if you modified .py files
make stubs-sync

# 5. Check code quality
make stubs-check-sync    # Verify stubs
mypy src/tnfr            # Type check
pytest tests/ --cov=tnfr # Test with coverage

# 6. Commit changes
git add .
git commit -m "Add feature: description"
# Pre-commit hooks run automatically

# 7. Push and create PR
git push origin feature/my-new-feature
# Create PR on GitHub
```

### Bug Fix Workflow

```bash
# 1. Create bug fix branch
git checkout -b fix/issue-123

# 2. Write failing test first (TDD)
# Edit tests/unit/test_operators.py
def test_bug_123():
    """Reproduce bug #123."""
    # Code that demonstrates the bug
    assert False, "Bug not fixed yet"

# 3. Verify test fails
pytest tests/unit/test_operators.py::test_bug_123

# 4. Fix the bug
# Edit src/tnfr/operators/coherence.py

# 5. Verify test passes
pytest tests/unit/test_operators.py::test_bug_123

# 6. Run full test suite
pytest tests/ -n auto

# 7. Commit and push
git add .
git commit -m "Fix #123: description of fix"
git push origin fix/issue-123
```

### Documentation Workflow

```bash
# 1. Create documentation branch
git checkout -b docs/improve-operator-guide

# 2. Edit documentation
# Edit docs/source/user-guide/OPERATORS_GUIDE.md

# 3. Build documentation locally
make docs
# Or for MkDocs:
mkdocs serve

# 4. Review in browser
# Sphinx: open docs/_build/html/index.html
# MkDocs: http://127.0.0.1:8000

# 5. Check internal references
make verify-refs

# 6. Commit and push
git add docs/
git commit -m "Docs: improve operator guide"
git push origin docs/improve-operator-guide
```

### Factory Function Development

```bash
# 1. Plan your factory
# - Decide naming: make_*, build_*, or create_*
# - Define inputs and outputs
# - List validation requirements
# - Document structural invariants

# 2. Create stub implementation
# Edit src/tnfr/mathematics/operators_factory.py
def make_new_operator(dim: int, *, param: float = 1.0) -> NewOperator:
    """Create validated new operator.
    
    Parameters
    ----------
    dim : int
        Dimensionality.
    param : float, optional
        Parameter description (default: 1.0).
    
    Returns
    -------
    NewOperator
        Validated operator instance.
    """
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    
    # TODO: Implementation
    raise NotImplementedError

# 3. Generate stub file
make stubs

# 4. Write tests
# Edit tests/mathematics/test_factory_patterns.py
class TestMakeNewOperator:
    def test_valid_construction(self):
        op = make_new_operator(dim=5)
        assert op.shape == (5, 5)
    
    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            make_new_operator(dim=0)

# 5. Implement factory
# Follow template from ARCHITECTURE_GUIDE.md

# 6. Run tests iteratively
pytest tests/mathematics/test_factory_patterns.py::TestMakeNewOperator -vv

# 7. Verify all checks pass
make stubs-check-sync
pytest tests/mathematics/test_factory_patterns.py
mypy src/tnfr/mathematics/operators_factory.py
```

---

## Code Quality

### Pre-commit Hooks

Automatically run on `git commit`:

```yaml
# .pre-commit-config.yaml (excerpt)
repos:
  - repo: local
    hooks:
      - id: check-stubs
        name: Check for missing .pyi stub files
        entry: make stubs-check
        language: system
        pass_filenames: false
        always_run: true
```

**Hooks include**:
- Stub file checks
- Code formatting (if configured)
- Trailing whitespace removal
- YAML validation
- Large file prevention

### Manual Quality Checks

```bash
# Type checking
mypy src/tnfr

# Stub validation
make stubs-check       # Check for missing stubs
make stubs-check-sync  # Check for outdated stubs

# Testing
pytest tests/ -n auto               # All tests
pytest tests/ --cov=tnfr            # With coverage
pytest tests/ -m "not slow"         # Fast tests only

# Documentation
make docs                           # Build Sphinx docs
make verify-refs                    # Check internal references
```

### Code Style Guidelines

#### Naming Conventions

```python
# Factory functions
def make_operator_name(...)      # Creates objects
def build_generator_name(...)    # Builds data structures
def create_nfr_name(...)         # Creates nodes/networks

# Private functions
def _internal_helper(...)        # Leading underscore

# Constants
MAX_ITERATIONS = 100             # UPPER_CASE
DEFAULT_TOLERANCE = 1e-9

# Type aliases
GraphLike = Union[nx.Graph, TNFRGraph]
```

#### Docstring Style

Use NumPy-style docstrings:

```python
def compute_metric(G, nodes, *, threshold=0.5):
    """Compute structural metric for specified nodes.
    
    Parameters
    ----------
    G : TNFRGraph
        Network graph with TNFR attributes.
    nodes : list of str
        Node identifiers to compute metric for.
    threshold : float, optional
        Minimum threshold for inclusion (default: 0.5).
    
    Returns
    -------
    dict
        Mapping from node ID to metric value.
    
    Raises
    ------
    ValueError
        If graph lacks required attributes.
    
    Notes
    -----
    This function preserves structural invariants and does not
    modify the input graph.
    
    Examples
    --------
    >>> G = create_test_graph()
    >>> metrics = compute_metric(G, ['n1', 'n2'])
    >>> assert all(0 <= v <= 1 for v in metrics.values())
    """
```

#### Type Annotations

```python
from typing import Optional, Union, Dict, List
import networkx as nx
import numpy as np

# Function signatures
def process_graph(
    G: nx.Graph,
    nodes: List[str],
    *,
    weights: Optional[np.ndarray] = None,
    backend: str = 'numpy',
) -> Dict[str, float]:
    """Process graph nodes."""
    ...

# Type aliases for clarity
NodeID = str
Weight = float
GraphLike = Union[nx.Graph, 'TNFRGraph']
```

---

## Documentation

### Documentation Types

1. **API Reference** (`docs/source/api/`)
   - Auto-generated from docstrings
   - One file per module/category
   - Examples included

2. **User Guides** (`docs/source/user-guide/`)
   - How-to guides for common tasks
   - Tutorial-style with examples
   - Operator guides, troubleshooting

3. **Advanced Guides** (`docs/source/advanced/`)
   - Architecture guide (factory patterns, dependencies)
   - Performance optimization
   - Testing strategies
   - Development workflow

4. **Theory** (`docs/source/theory/`)
   - Mathematical foundations
   - Jupyter notebooks
   - Operator derivations

### Building Documentation

```bash
# Sphinx documentation
make docs
open docs/_build/html/index.html

# MkDocs documentation
mkdocs serve
# Visit http://127.0.0.1:8000

# Verify internal references
make verify-refs
make verify-refs-verbose  # Detailed output
```

### Documentation Guidelines

**DO**:
- ✅ Use concrete examples
- ✅ Show both valid and invalid usage
- ✅ Include expected outputs
- ✅ Link to related documentation
- ✅ Document TNFR-specific semantics (νf, phase, ΔNFR)
- ✅ Update when APIs change

**DON'T**:
- ❌ Copy-paste code without testing
- ❌ Use generic placeholder examples
- ❌ Forget to update cross-references
- ❌ Skip documenting edge cases
- ❌ Use jargon without explanation

---

## CI/CD Pipeline

### GitHub Actions Workflows

The project uses multiple CI workflows:

#### 1. Main CI Workflow (`.github/workflows/ci.yml`)

```yaml
jobs:
  test:
    # Run tests on Python 3.9-3.13
    # Generate coverage reports
  
  type-check:
    # Verify type stubs
    # Run mypy
  
  docs:
    # Build documentation
    # Verify no broken links
```

#### 2. Security Workflow

```yaml
jobs:
  bandit:
    # Static security analysis
  
  dependency-audit:
    # Check for vulnerable dependencies
  
  codeql:
    # CodeQL security scanning
```

#### 3. Release Workflow

```yaml
jobs:
  publish:
    # Build distributions
    # Publish to PyPI
    # Create GitHub release
```

### CI Stages

```
┌─────────────────┐
│  Push to GitHub │
└────────┬────────┘
         │
         ├─── Type Check ────────────┐
         │    - Stub validation       │
         │    - mypy                  │
         │                            │
         ├─── Test (3.9-3.13) ───────┤
         │    - Unit tests            │
         │    - Integration tests     │
         │    - Coverage report       │
         │                            │
         ├─── Docs ──────────────────┤
         │    - Build Sphinx          │
         │    - Verify references     │
         │                            │
         ├─── Security ──────────────┤
         │    - Bandit                │
         │    - Dependency audit      │
         │    - CodeQL                │
         │                            │
         └────────┬─────────────────┘
                  │
         ┌────────▼────────┐
         │  All Checks Pass │
         └─────────────────┘
```

### Local CI Simulation

Run the same checks locally before pushing:

```bash
# Type checks
make stubs-check
make stubs-check-sync
mypy src/tnfr

# Tests
pytest tests/ -n auto --cov=tnfr

# Documentation
make docs
make verify-refs

# Security (optional)
bandit -r src -ll -f json -o bandit.json
python tools/bandit_to_sarif.py bandit.json bandit.sarif

# All-in-one pre-push check
./scripts/pre-push-check.sh  # If available
```

---

## Release Process

### Version Numbering

TNFR follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

Examples:
1.0.0  - First stable release
1.1.0  - New feature, backward compatible
1.1.1  - Bug fix, backward compatible
2.0.0  - Breaking change
```

### Release Checklist

```bash
# 1. Ensure clean state
git checkout main
git pull origin main
git status  # Should be clean

# 2. Update version
# Edit pyproject.toml or use tool
bumpversion minor  # or major, patch

# 3. Update changelog
# Edit CHANGELOG.md or use towncrier
towncrier build --version 1.2.0

# 4. Run full test suite
pytest tests/ -n auto
make docs
make verify-refs

# 5. Create release commit
git add .
git commit -m "Release 1.2.0"

# 6. Tag release
git tag -a v1.2.0 -m "Release version 1.2.0"

# 7. Push to GitHub
git push origin main
git push origin v1.2.0

# 8. GitHub Actions will:
#    - Run all CI checks
#    - Build distributions
#    - Publish to PyPI (if configured)
#    - Create GitHub release
```

### Changelog Fragments

Use towncrier for managing changelog entries:

```bash
# Create fragment for new feature
echo "Add new operator feature" > docs/changelog.d/123.feature.md

# Types: .feature, .bugfix, .doc, .removal, .misc

# Build changelog
towncrier build --version 1.2.0

# Preview without modifying files
towncrier build --draft --version 1.2.0
```

---

## Troubleshooting

### Common Issues

#### Issue: Import errors after installation

**Solution**:
```bash
# Reinstall in editable mode
pip install -e .

# Or with dev dependencies
pip install -e .[dev-full]

# Verify installation
python -c "import tnfr; print(tnfr.__version__)"
```

#### Issue: Pre-commit hooks failing

**Solution**:
```bash
# Update hooks to latest version
pre-commit autoupdate

# Run manually to see detailed errors
pre-commit run --all-files --verbose

# Skip hooks for emergency commit (use sparingly!)
git commit --no-verify -m "Emergency fix"
```

#### Issue: Tests passing locally but failing in CI

**Possible causes**:
1. Different Python version
2. Missing seed for random operations
3. Platform-specific behavior
4. Missing test dependency

**Diagnosis**:
```bash
# Match CI Python version
pyenv install 3.12
pyenv local 3.12

# Run tests exactly as CI does
pytest tests/ -n auto --cov=tnfr --cov-report=xml

# Check for missing seeds
grep -r "random" tests/ | grep -v "seed"
```

#### Issue: Documentation build fails

**Solution**:
```bash
# Install docs dependencies
pip install -e .[docs]

# Clean build directory
rm -rf docs/_build

# Build with verbose output
sphinx-build -v docs/source docs/_build/html

# Check for missing references
make verify-refs-verbose
```

### Getting Help

1. **Check existing documentation**:
   - This guide
   - [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
   - [TESTING_STRATEGIES.md](TESTING_STRATEGIES.md)
   - [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)

2. **Search GitHub Issues**:
   - Known issues and solutions
   - Feature discussions

3. **Ask in Discussions**:
   - Questions about development
   - Feature proposals
   - Best practices

4. **Open an Issue**:
   - Bug reports
   - Feature requests
   - Documentation improvements

---

## Best Practices Summary

### DO:
- ✅ Write tests before code (TDD)
- ✅ Generate stubs for all .py files
- ✅ Use explicit seeds for reproducibility
- ✅ Document TNFR-specific semantics
- ✅ Run pre-push checks locally
- ✅ Keep commits focused and atomic
- ✅ Update documentation with code changes

### DON'T:
- ❌ Skip writing tests
- ❌ Commit without running pre-commit hooks
- ❌ Break structural invariants
- ❌ Push untested code
- ❌ Ignore CI failures
- ❌ Commit generated files (unless .pyi stubs)
- ❌ Make breaking changes without discussion

---

## Quick Reference

### Essential Commands

```bash
# Setup
pip install -e .[dev-full]
pre-commit install

# Development
pytest tests/ -n auto              # Run tests
make stubs-sync                    # Update stubs
mypy src/tnfr                      # Type check
make docs                          # Build docs

# Quality
make stubs-check-sync              # Verify stubs
pytest tests/ --cov=tnfr           # Coverage
make verify-refs                   # Check doc links

# Help
make help                          # Available make targets
pytest --help                      # Pytest options
mypy --help                        # Mypy options
```

### File Locations

- Source code: `src/tnfr/`
- Tests: `tests/`
- Documentation: `docs/source/`
- Scripts: `scripts/`
- Configuration: `pyproject.toml`
- CI workflows: `.github/workflows/`

---

## See Also

- [Architecture Guide](ARCHITECTURE_GUIDE.md) - Factory patterns and dependencies
- [Testing Strategies](TESTING_STRATEGIES.md) - Testing best practices
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md) - Optimization techniques
- [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) - Contribution guidelines
- [README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md) - Project overview

---

**Last Updated**: 2025-11-06  
**Status**: Active - consolidated workflow documentation  
**Maintenance**: Update when processes change, review quarterly
