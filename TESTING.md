# TNFR Testing Guide

This document describes the testing strategy, infrastructure, and best practices for the TNFR Python Engine. It consolidates information about test organization, coverage requirements, and structural fidelity validation.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Structural Fidelity Tests](#structural-fidelity-tests)
- [Backend Selection](#backend-selection)
- [Coverage Requirements](#coverage-requirements)
- [Test Development Guidelines](#test-development-guidelines)

## Testing Philosophy

TNFR tests validate **structural coherence** first, implementation details second. Every test must:

1. **Preserve TNFR Invariants**: Verify canonical constraints (see ARCHITECTURE.md §3)
2. **Test Structural Behavior**: Focus on coherence, phase, frequency, not implementation
3. **Maintain Reproducibility**: Use seeds, validate determinism
4. **Guard Regressions**: Performance, accuracy, and API stability

Tests are **not** responsible for:
- Fixing unrelated pre-existing failures
- Optimizing code that already passes
- Validating framework internals (NetworkX, NumPy)

## Test Organization

The test suite is organized by concern and scope:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── utils.py                 # Test utilities (module clearing, etc.)
├── unit/                    # Unit tests for individual modules
│   ├── test_cache.py
│   ├── test_dynamics.py
│   ├── test_operators.py
│   ├── test_structural.py
│   └── ...
├── integration/             # Integration tests for subsystems
│   ├── test_glyph_sequences.py
│   ├── test_operator_chains.py
│   └── test_telemetry_pipeline.py
├── property/                # Property-based tests (Hypothesis)
│   └── test_tnfr_invariants.py
├── stress/                  # Stress and scale tests
│   └── test_large_networks.py
├── performance/             # Performance regression tests
│   └── test_dnfr_pipeline.py
├── mathematics/             # Mathematical backend tests
│   └── test_epi.py
└── cli/                     # CLI interface tests
    └── test_cli.py
```

### Key Test Files

| File | Purpose | Markers |
|------|---------|---------|
| `test_extreme_cases.py` | Boundary value testing | `unit` |
| `test_glyph_sequences.py` | Grammar and sequence validation | `integration` |
| `test_tnfr_invariants.py` | Property-based invariant checks | `property`, `slow` |
| `test_dnfr_pipeline.py` | Performance regression guards | `performance`, `slow` |
| `test_trace.py` | Telemetry and debugging utilities | `unit` |

## Running Tests

### Basic Test Execution

```bash
# Run all tests (excluding slow/benchmarks by default)
pytest

# Run specific test category
pytest tests/unit
pytest tests/integration

# Run with coverage report
pytest --cov=tnfr --cov-report=html

# Run verbose with output
pytest -v -s
```

### Artifact Guards

Telemetry dashboards are part of the reproducibility surface. A lightweight regression test (`tests/test_precision_walk_dashboard_artifact.py`) ensures `benchmarks/results/precision_walk_dashboard.json` is valid JSON (no `NaN` literals) and remains loadable by downstream tooling. It now runs automatically via:

```bash
# Curated smoke bundle (includes dashboard JSON guard)
make smoke-tests

# Run the guard directly when iterating on telemetry scripts
pytest tests/test_precision_walk_dashboard_artifact.py
```

### Test Markers

Tests are marked for selective execution:

```bash
# Run only fast tests (default in CI)
pytest -m "not slow"

# Run slow/property-based tests
pytest -m slow

# Run performance regression tests
pytest -m performance tests/performance

# Run backend-specific tests
pytest -m numpy_only
pytest -m requires_jax
```

### Configuration

Default pytest options are configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-m", "not slow",          # Skip slow tests by default
    "--benchmark-skip",        # Skip benchmarks by default
    "--strict-markers",
    "--tb=short",
]
markers = [
    "slow: marks tests as slow (deselected by default)",
    "performance: marks performance regression tests",
    "numpy_only: requires NumPy backend",
    "requires_jax: requires JAX backend",
    "requires_torch: requires PyTorch backend",
]
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual modules and functions in isolation.

**Coverage Areas**:

- Structural operators (emission, reception, coherence, etc.)
- Dynamics (ΔNFR computation, nodal equation integration)
- Cache layers (Shelve, Redis, Memory)
- Validation (sequence grammar, graph state)
- Telemetry (coherence, sense index, traces)

**Example**:

```python
def test_coherence_operator_stabilizes_epi(simple_graph):
    """Coherence operator should increase stability."""
    G, node = simple_graph
    initial_coherence = compute_coherence(G)
    
    apply_operator(G, node, "coherence")
    
    final_coherence = compute_coherence(G)
    assert final_coherence >= initial_coherence
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test interactions between subsystems and operator sequences.

**Coverage Areas**:

- Canonical glyph sequences (emission→reception→coherence)
- Operator chaining and graph state evolution
- Telemetry pipeline end-to-end
- Validation and execution coordination

**Example**:

```python
def test_canonical_sequence_execution():
    """Test canonical TNFR sequence executes without errors."""
    G, node = create_nfr("test", epi=1.0, vf=1.0)
    sequence = ["emission", "reception", "coherence", "coupling"]
    
    result = run_sequence(G, node, sequence)
    
    assert result["success"]
    assert compute_coherence(G) > 0
```

### 3. Property-Based Tests (`tests/property/`)

**Purpose**: Use Hypothesis to generate test cases validating TNFR invariants.

**Coverage Areas**:

- Operator invariants (bounds preservation, no NaN/inf)
- Coherence monotonicity
- Phase coupling symmetry
- Frequency scaling laws

**Example**:

```python
@given(
    epi=st.floats(min_value=0.1, max_value=10.0),
    vf=st.floats(min_value=0.1, max_value=10.0),
)
def test_coherence_operator_preserves_bounds(epi, vf):
    """Coherence operator should never produce NaN or infinite values."""
    G, node = create_nfr("prop_test", epi=epi, vf=vf)
    
    apply_operator(G, node, "coherence")
    
    epi_after = G.nodes[node]["epi"]
    assert np.isfinite(epi_after).all()
```

### 4. Stress Tests (`tests/stress/`)

**Purpose**: Validate behavior under extreme conditions and large scales.

**Coverage Areas**:

- Large networks (1000+ nodes)
- Long sequences (100+ operators)
- Memory usage patterns
- Parallel execution stability

**Example**:

```python
@pytest.mark.slow
def test_large_network_coherence():
    """Test coherence computation on large networks."""
    G = nx.DiGraph()
    for i in range(1000):
        G.add_node(i, epi=np.random.rand(10), vf=1.0, phase=0.0)
    
    coherence = compute_coherence(G)
    
    assert 0 <= coherence <= 1.0
    assert np.isfinite(coherence)
```

### 5. Performance Tests (`tests/performance/`)

**Purpose**: Guard against performance regressions in critical paths.

**Coverage Areas**:

- ΔNFR computation pipeline
- Alias cache effectiveness
- Trigonometric metric calculations
- Sense index computation

**Execution**:
```bash
pytest -m performance tests/performance
```

## Structural Fidelity Tests

## Invariant Tests

These tests validate TNFR canonical invariants. For complete invariant definitions and physics, see **[AGENTS.md § Canonical Invariants](AGENTS.md#-canonical-invariants-never-break)**.

### Invariant 1: EPI Changes Only Through Operators

```python
def test_epi_only_changes_via_operators():
    """EPI should only change through structural operators."""
    G, node = create_nfr("test", epi=1.0, vf=1.0)
    initial_epi = G.nodes[node]["epi"].copy()
    
    # Direct mutation should not happen
    # Only operator application should change EPI
    apply_operator(G, node, "coherence")
    
    assert not np.array_equal(G.nodes[node]["epi"], initial_epi)
```

### Invariant 2: Structural Frequency in Hz_str

```python
def test_vf_preserved_in_hz_str():
    """Structural frequency should remain in Hz_str units."""
    G, node = create_nfr("test", epi=1.0, vf=2.5)
    
    apply_operator(G, node, "mutation")
    
    vf = G.nodes[node]["vf"]
    assert isinstance(vf, (int, float))
    assert vf > 0  # Positive Hz_str
```

### Invariant 5: Phase Check Before Coupling

```python
def test_coupling_requires_phase_check():
    """Coupling should verify phase synchrony."""
    G = nx.DiGraph()
    n1 = G.add_node(1, epi=1.0, vf=1.0, phase=0.0)
    n2 = G.add_node(2, epi=1.0, vf=1.0, phase=np.pi)
    
    # Should validate phase compatibility
    with pytest.raises(ValidationError):
        apply_operator(G, n1, "coupling", target=n2)
```

### Invariant 8: Reproducible Simulations

```python
def test_deterministic_with_seed():
    """Same seed should produce same results."""
    seed = 42
    
    # Run 1
    G1, node1 = create_nfr("test", epi=1.0, vf=1.0, seed=seed)
    run_sequence(G1, node1, ["emission", "coherence"])
    result1 = compute_coherence(G1)
    
    # Run 2
    G2, node2 = create_nfr("test", epi=1.0, vf=1.0, seed=seed)
    run_sequence(G2, node2, ["emission", "coherence"])
    result2 = compute_coherence(G2)
    
    assert result1 == result2
```

## Backend Selection

TNFR supports multiple mathematical backends (NumPy, JAX, PyTorch). Tests can specify backend requirements:

### Environment Variable

```bash
# Set backend before running tests
export TNFR_MATH_BACKEND=numpy  # or jax, torch
pytest tests/mathematics
```

### Command Line

```bash
# Use pytest option
pytest tests/mathematics --math-backend=torch
```

### In Tests

```python
@pytest.mark.requires_jax
def test_jax_specific_feature():
    """This test requires JAX backend."""
    from tnfr.mathematics import get_backend
    backend = get_backend()
    assert backend.name == "jax"
```

### Backend Availability

When a requested backend is unavailable, pytest automatically skips backend-specific tests while continuing with NumPy fallback.

## Coverage Requirements

### Minimum Coverage Targets

- **Overall**: 80% line coverage
- **Core modules** (structural, dynamics, operators): 90%
- **Telemetry** (metrics, trace): 85%
- **Utilities** (cache, validation): 80%

### Critical Paths (Must be 100% covered)

- Nodal equation integration (`dynamics/integrators.py`)
- Operator registry and dispatch (`operators/registry.py`)
- Canonical sequence validation (`validation/__init__.py`)
- ΔNFR computation (`dynamics/dnfr.py`)

### Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=tnfr --cov-report=html

# Open report
open htmlcov/index.html
```

## Test Development Guidelines

### 1. Test Isolation

Use `clear_test_module()` utility for test independence:

```python
from tests.utils import clear_test_module

def test_fresh_import():
    """Test with fresh module state."""
    clear_test_module('tnfr.utils.io')
    import tnfr.utils.io  # Fresh import
```

**Note**: Module path checking (`'module' in sys.modules`) may trigger CodeQL warnings. This is legitimate test infrastructure, not URL validation. See ARCHITECTURE.md §"Test isolation and module management".

### 2. Fixture Usage

Prefer pytest fixtures for common setup:

```python
@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    G = nx.DiGraph()
    node = G.add_node(1, epi=np.array([1.0, 2.0]), vf=1.0, phase=0.0)
    return G, node

def test_with_fixture(simple_graph):
    G, node = simple_graph
    # Test implementation
```

### 3. Parametrize for Variants

Use `pytest.mark.parametrize` for multiple test cases:

```python
@pytest.mark.parametrize("operator", [
    "emission", "reception", "coherence", "dissonance"
])
def test_operator_preserves_structure(operator):
    """All operators should preserve graph structure."""
    G, node = create_nfr("test", epi=1.0, vf=1.0)
    apply_operator(G, node, operator)
    
    assert node in G.nodes
```

### 4. Document Test Intent

Every test should have a clear docstring explaining:

- What structural behavior is being validated
- Which TNFR invariant(s) are checked
- Why the test matters for coherence

```python
def test_resonance_propagates_coherence():
    """
    Resonance operator should propagate coherence to coupled nodes
    without altering EPI identity (Invariant 1, 4).
    
    This test validates that resonance maintains operational
    fractality while increasing network coupling.
    """
    # Test implementation
```

### 5. Avoid Brittleness

- Don't assert exact floating-point equality (use `np.allclose`)
- Don't depend on internal implementation details
- Don't test framework internals (NetworkX, NumPy)
- Focus on structural semantics, not code paths

### 6. Performance Awareness

Mark slow tests appropriately:

```python
@pytest.mark.slow
def test_expensive_computation():
    """This test takes >1 second to run."""
    # Long-running test
```

## Test Maintenance

### Pre-Existing Failures

The repository tracks known test failures that are not regressions:

- **Import errors**: 3 known (require infrastructure changes)
- **Backend compatibility**: Some tests may fail with specific backends
- **Platform-specific**: Windows vs Unix path handling

These are documented and should not block PR approval unless your changes introduce new failures.

### Continuous Integration

All PRs must:

1. Pass the default test suite (`pytest -m "not slow"`)
2. Maintain or improve coverage
3. Not introduce new failures (beyond pre-existing)
4. Pass all structural fidelity tests

### Test Optimization

When optimizing tests:

1. Consolidate redundant tests across directories
2. Use shared fixtures to reduce duplication
3. Parametrize instead of copying test functions
4. Profile slow tests and optimize data generation

## Debugging Tests

### Verbose Output

```bash
# Show print statements and detailed assertions
pytest -v -s tests/unit/test_operators.py

# Show local variables on failure
pytest --showlocals
```

### Run Single Test

```bash
# Run specific test function
pytest tests/unit/test_cache.py::test_shelve_layer_stores_data

# Run test by keyword match
pytest -k "coherence" tests/
```

### Debug with PDB

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at test start
pytest --trace
```

### Capture Warnings

```bash
# Show warnings
pytest -W default

# Treat warnings as errors
pytest -W error
```

## Resources

- **ARCHITECTURE.md**: TNFR invariants and canonical constraints
- **SECURITY.md**: Security testing guidelines
- **CONTRIBUTING.md**: General contribution workflow
- **tests/README.md**: Detailed test suite organization
- **pyproject.toml**: Test configuration and markers

---

**Last Updated**: November 2025  
**Version**: 1.0
