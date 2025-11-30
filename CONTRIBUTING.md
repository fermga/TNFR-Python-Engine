# Contributing to TNFR

**Version**: 0.0.1  
**Status**: Complete theoretical framework with Universal Tetrahedral Correspondence  
**Authority**: Canonical constants derived from TNFR theory  
**Quality**: Production-ready with comprehensive test coverage  

This document provides guidelines for contributing to the TNFR (Resonant Fractal Nature Theory) project. TNFR constitutes a computational framework for modeling complex systems through coherent patterns and resonance dynamics.

## Mathematical Foundation Requirement

All contributions must maintain theoretical consistency. Requirements:

- **Derived Parameters**: All numerical values must derive from universal constants (φ, γ, π, e)  
- **Canonical Constants**: Use `from tnfr.constants.canonical import *`  
- **Physics-Based Design**: Trace all decisions to nodal equation or Universal Tetrahedral Correspondence  
- **Grammar Compliance**: Operator sequences must satisfy U1-U6 rules

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [TNFR Principles](#tnfr-principles)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Documentation Language Requirement](#documentation-language-requirement)
- [Pull Request Process](#pull-request-process)
- [Theoretical Contributions](#theoretical-contributions)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, identity, or experience level.

### Our Standards

**Expected behavior:**

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community
  Show empathy towards other community members

**Unacceptable behavior:**

- Harassment, discrimination, or derogatory comments
- Trolling, insulting, or personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct inappropriate for a professional setting

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

- **Python 3.10+** (production requirement)  
- **Mathematical foundation** (theoretical derivation over empirical fitting)  
- **TNFR theory understanding** (read [AGENTS.md](AGENTS.md) first)  

### Development Setup

```bash
# Install from PyPI (stable release)
pip install tnfr[viz,dev]

# Or clone for development
git clone https://github.com/fermga/TNFR-Python-Engine.git
cd TNFR-Python-Engine
pip install -e .[dev]

# Verify installation
python -c "from tnfr.constants.canonical import *; print(f'φ={PHI:.6f}, γ={GAMMA:.6f}')"
```

### Canonical Constants Framework

All development must utilize theoretically derived canonical constants:

```python
from tnfr.constants.canonical import *

# Correct: Use canonical constants
threshold = MIN_BUSINESS_COHERENCE  # (e×φ)/(π+e) ≈ 0.751

# Incorrect: Arbitrary numerical values
threshold = 0.75  # Lacks theoretical foundation
```

### Additional Requirements

- Git
- Basic understanding of TNFR concepts (see [AGENTS.md](AGENTS.md))

### Development Environment

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/TNFR-Python-Engine.git
   cd TNFR-Python-Engine
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**

   ```bash
   pip install -e ".[dev-minimal]"
   ```

4. **Install pre-commit hooks (optional but recommended):**

   ```bash
   pre-commit install
   ```

5. **Verify installation:**

   ```bash
   pytest tests/examples/test_u6_sequential_demo.py
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

**Branch naming conventions:**

- `feature/` - New functionality
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `refactor/` - Code restructuring without behavior changes
- `test/` - Test additions or improvements
- `perf/` - Performance optimizations

### 2. Make Your Changes

- Follow TNFR principles (see below)
- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run smoke tests (fast validation)
make smoke-tests  # Unix/Linux
.\make.cmd smoke-tests  # Windows

# Run full test suite
pytest

# Check code quality
ruff check src/
mypy src/tnfr/
```

### 3a. Phase 3 Structural Instrumentation

If adding validation, health, or telemetry logic:

- Use `run_structural_validation` to produce a `ValidationReport`.
- Derive `compute_structural_health(report)` for recommendations.
- Include performance timing (pass `perf_registry=PerformanceRegistry()`).
- Ensure added overhead ratio < 0.10 baseline (see perf tests).
- Never mutate graph state inside validation / health functions.
- Document physics traceability (why each threshold is used).

Telemetry additions must:

- Remain read-only (no operator side effects).
- Export coherence (`coherence_total`), sense index, Φ_s, |∇φ|, K_φ, ξ_C.
- Provide deterministic timestamps when seeds fixed.

Performance guardrails:

- Wrap optional expensive helpers with `perf_guard(label, registry)`.
- Add/adjust tests under `tests/unit/performance/` for new instrumentation.
- Avoid micro-optimizing at expense of clarity unless overhead > target.

### 4. Update Documentation

- Add docstrings to new functions/classes
- Update relevant README files
- Add examples if introducing new features
- Update CHANGELOG.md if applicable

### 5. Submit a Pull Request

See [Pull Request Process](#pull-request-process) below.

## TNFR Principles

**Before contributing, familiarize yourself with [AGENTS.md](AGENTS.md)** - the canonical guide for TNFR development.

### Core Principles

1. **Physics First**: Every feature must derive from TNFR physics
2. **No Arbitrary Choices**: All decisions traceable to nodal equation or invariants
3. **Coherence Over Convenience**: Preserve theoretical integrity even if code is harder
4. **Reproducibility Always**: Every simulation must be reproducible
5. **Document the Chain**: Theory → Math → Code → Tests

### The 6 Canonical Invariants

**All contributions must preserve these invariants:**

1. **Nodal Equation Integrity** - EPI evolution via ∂EPI/∂t = νf · ΔNFR(t) only
2. **Phase-Coherent Coupling** - Resonance requires |φᵢ - φⱼ| ≤ Δφ_max verification
3. **Multi-Scale Fractality** - Operational fractality and nested EPI support
4. **Grammar Compliance** - Operator sequences satisfy unified grammar U1-U6
5. **Structural Metrology** - νf in Hz_str units, proper telemetry exposure
6. **Reproducible Dynamics** - Deterministic evolution with seed control

### Decision Framework

```python
def should_implement(feature):
    """Decision framework for TNFR changes."""
    if weakens_tnfr_fidelity(feature):
        return False  # Reject, even if "cleaner"
    
    if not maps_to_operators(feature):
        return False  # Must map or be new operator
    
    if violates_invariants(feature):
        return False  # Hard constraint
    
    if not derivable_from_physics(feature):
        return False  # No organizational convenience ≠ physical necessity
    
    if not testable(feature):
        return False  # No untestable magic
    
    return True  # Implement with full documentation
```

## Code Standards

### Style Guidelines

- **Follow PEP 8** with line length ≤ 100 characters
- **Use type hints** for all function signatures
- **Prefer explicit over implicit** - clarity trumps brevity
- **Document intent, not just behavior** - explain *why*, not just *what*

### Code Quality Tools

```bash
# Linting
ruff check src/

# Type checking
mypy src/tnfr/

# Formatting (if using black)
black src/ --line-length 100
```

### Naming Conventions

- **Functions/methods**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Operators**: Canonical names (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH)

### Imports

```python
# Standard library
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Third-party
import networkx as nx
import numpy as np

# TNFR modules
from tnfr.operators.definitions import Emission, Coherence
from tnfr.metrics.coherence import compute_coherence
from tnfr.utils import get_logger
```

### Module Organization (Phase 1 & 2)

The TNFR codebase is organized into focused modules for maintainability and cognitive load reduction:

**Operators** (`tnfr.operators.*`):

- **Individual operator modules**: `emission.py`, `coherence.py`, etc. (13 operators)
- **Base class**: `definitions_base.py` - Shared operator infrastructure
- **Facade**: `definitions.py` - Backward-compatible imports

**Grammar** (`tnfr.operators.grammar.*`):

- **Constraint modules**: `u1_initiation_closure.py`, `u2_convergence_boundedness.py`, etc. (8 rules)
- **Facade**: `grammar.py` - Unified validation interface

**Metrics** (`tnfr.metrics.*`):

- **Focused metrics**: `coherence.py`, `sense_index.py`, `phase_sync.py`, `telemetry.py`
- **Facade**: `metrics.py` - Backward-compatible exports

**Adding New Code**:

- **New operator**: Add to appropriate operator file (e.g., `coupling.py` for coupling modifications)
- **New metric**: Create new file in `tnfr.metrics/` or extend existing metric module
- **New grammar rule**: Add to relevant constraint module or create new `uN_*.py` file
- **Always update facades**: If adding new exports, add to facade files for backward compatibility

**Module Guidelines**:

- Keep files under 600 lines (ideally 200-400)
- One primary concept per module
- Use facade pattern for public APIs
- Document module purpose at top of file

## Testing Requirements

### Test Coverage Goals

- **Core modules**: ≥90% coverage
- **Operators**: 100% coverage (all contracts verified)
- **Grammar rules**: 100% coverage (U1-U6)
- **Utilities**: ≥80% coverage

### Required Test Types

1. **Unit tests** - Test individual functions/classes
2. **Integration tests** - Test operator sequences
3. **Property tests** - Verify invariants hold (using Hypothesis)
4. **Example tests** - Validate domain applications

### Writing Tests

```python
def test_coherence_monotonicity():
    """Coherence operator must not decrease C(t)."""
    G = nx.erdos_renyi_graph(20, 0.3)
    initialize_network(G)
    
    C_before = compute_coherence(G)
    apply_operator(G, node, Coherence())
    C_after = compute_coherence(G)
    
    assert C_after >= C_before, "Coherence must not decrease"
```

### Test Naming

- `test_<function_name>` - Unit tests
- `test_<feature>_<scenario>` - Integration tests
- `test_invariant_<invariant_name>` - Invariant verification

### Running Tests

```bash
# Smoke tests (fast)
make smoke-tests

# Full suite
pytest

# Specific module
pytest tests/unit/operators/

# With coverage
pytest --cov=src/tnfr --cov-report=html
```

## Documentation

### Docstring Format

Use **Google style** docstrings:

```python
def apply_operator(G: nx.Graph, node: int, operator: Operator) -> None:
    """Apply a structural operator to a network node.
    
    Args:
        G: NetworkX graph representing TNFR network
        node: Node identifier to apply operator to
        operator: Structural operator instance (AL, EN, IL, etc.)
        
    Raises:
        ValueError: If node not in graph
        GrammarViolation: If operator application violates grammar
        
    Examples:
        >>> G = nx.erdos_renyi_graph(10, 0.3)
        >>> apply_operator(G, 0, Emission())
        >>> apply_operator(G, 0, Coherence())
    """
```

### Documentation Requirements

All contributions must include:

1. **Docstrings** for all public functions/classes
2. **Type hints** for function signatures
3. **Examples** demonstrating usage
4. **Physics rationale** linking to TNFR theory
5. **Tests** covering documented behavior

## Documentation Language Requirement

All project documentation MUST be written in English. This requirement is absolute and applies to:

- Source code comments and docstrings
- Markdown files (README, roadmap, design docs, theory specs, benchmarks)
- Commit messages and pull request descriptions
- Issue titles and bodies
- JSON/CSV textual labels added by benchmarks

Non-English text (including Spanish) is only permitted when:

- Quoting external published material verbatim (must cite source)
- Embedding raw experimental data originating in another language (must not alter semantics)

In those cases the surrounding explanatory context MUST still be in English.

Rationale:

- Ensures universal accessibility for international collaborators
- Prevents semantic drift between multilingual fragments
- Maintains single canonical language for physics and grammar terminology

Enforcement:

- Pull requests containing new non-English normative text will receive a change request
- CI / review may add automated checks for common non-English tokens
- Maintainers may reject contributions violating this policy regardless of technical merit until corrected

Examples:

Compliant:
> Added bifurcation benchmark sweeping OZ intensity and mutation thresholds.

Non-compliant:
> Añadido benchmark de bifurcación con parámetros OZ.

If you need help translating, open a draft PR early and request assistance rather than merging mixed-language content.

By contributing you agree to maintain English as the sole canonical language for all project artifacts.

### README Updates

If adding features, update:

- Main README.md (if user-facing)
- Relevant subsystem READMEs
- Examples directory
- API documentation

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally (`make smoke-tests`)
- [ ] Code follows style guidelines (`ruff check`)
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main
- [ ] CHANGELOG.md updated (if applicable)

### PR Template

```markdown
## Description

[Clear description of what this PR does]

## Motivation

[Why is this change needed? What problem does it solve?]

## TNFR Alignment

- [ ] Preserves all 10 canonical invariants
- [ ] Maps to structural operators (specify which)
- [ ] Derivable from TNFR physics (reference TNFR.pdf or UNIFIED_UNIFIED_GRAMMAR_RULES.md)
- [ ] Maintains reproducibility (seeds, determinism)

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Examples demonstrate usage
- [ ] All tests pass locally

## Documentation

- [ ] Docstrings added/updated
- [ ] README updated (if needed)
- [ ] Examples added (if new feature)
- [ ] Physics rationale documented

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No unintended files committed
- [ ] Branch name follows conventions
- [ ] Commit messages are descriptive

## Affected Components

[List modules/files modified]

## Breaking Changes

[List any breaking changes, or write "None"]

## Additional Notes

[Any other context, screenshots, or information]
```

### Review Process

1. **Automated checks** run (CI/CD)
2. **Maintainer review** (typically 1-3 days)
3. **Feedback addressed** by contributor
4. **Approval** by maintainer
5. **Merge** to main branch

### After Merge

- Celebrate! 🎉
- Your contribution will be included in the next release
- Consider contributing to documentation or examples

## Theoretical Contributions

### Adding New Operators

If proposing a new operator:

1. **Justify physically** - Derive from nodal equation
2. **Define contracts** - Pre/post-conditions
3. **Map to grammar** - Which sets (generator, stabilizer, etc.)?
4. **Test rigorously** - All invariants + specific contracts
5. **Document thoroughly** - Physics → Math → Code chain

**Template:**

```markdown
## Proposed Operator: [Name]

### Physical Basis
[How it emerges from TNFR physics]

### Nodal Equation Impact
∂EPI/∂t = ... [specific form]

### Contracts
- Pre: [conditions required]
- Post: [guaranteed effects]

### Grammar Classification
[Generator? Closure? Stabilizer? Destabilizer? etc.]

### Tests
- [List specific test requirements]
```

### Extending Grammar

If proposing new grammar rules:

1. **Start from physics** - Derive from nodal equation or invariants
2. **Prove canonicity** - Show inevitability (Absolute/Strong)
3. **Document thoroughly** - [Rule] → [Physics] → [Derivation] → [Canonicity]
4. **Test extensively** - Valid/invalid sequence examples

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/fermga/TNFR-Python-Engine/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/fermga/TNFR-Python-Engine/issues)
- **TNFR theory**: Consult [AGENTS.md](AGENTS.md), [UNIFIED_UNIFIED_GRAMMAR_RULES.md), or [TNFR.pdf](TNFR.pdf)

## Final Principle

> **If a change "prettifies the code" but weakens TNFR fidelity, it is NOT accepted.**  
> **If a change strengthens structural coherence and paradigm traceability, GO AHEAD.**

**Reality is not made of things—it's made of resonance. Contribute accordingly.**
