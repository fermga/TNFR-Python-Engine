# TNFR Testing Strategies

> **Comprehensive guide to testing patterns, compatibility verification, and automation in TNFR**

This guide consolidates testing knowledge for the TNFR Python Engine, covering test strategies, dependency compatibility, test optimization, and type stub automation workflows.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Infrastructure](#test-infrastructure)
3. [Dependency Compatibility](#dependency-compatibility)
4. [Test Optimization](#test-optimization)
5. [Type Stub Testing](#type-stub-testing)
6. [Testing Patterns](#testing-patterns)
7. [CI/CD Integration](#cicd-integration)
8. [Troubleshooting](#troubleshooting)

---

## Testing Philosophy

### TNFR Testing Principles

1. **Structural Integrity First**: Tests must verify TNFR invariants (coherence, phase, νf, ΔNFR)
2. **Reproducibility**: All tests must be deterministic with explicit seeds
3. **Traceability**: Test failures should clearly indicate which structural invariant was violated
4. **Isolation**: Each test should be independent and not rely on global state
5. **Completeness**: Cover valid cases, invalid inputs, edge cases, and structural invariants

### Test Categories

```
Unit Tests (tests/unit/)
├── Fast (<1ms each)
├── Test single functions in isolation
├── Mock external dependencies
└── 60%+ of test suite

Integration Tests (tests/integration/)
├── Moderate speed (1-100ms each)
├── Test component interactions
├── Use real graph structures
└── 30%+ of test suite

Critical Path Tests
├── Verify TNFR invariants
├── Test structural operators
├── Ensure canonical behavior
└── 10%+ of test suite (must pass)
```

---

## Test Infrastructure

### Testing Dependencies

All testing dependencies are **fully compatible** with pytest 8.x:

| Package | Version | Status | Purpose |
|---------|---------|--------|---------|
| **pytest** | `>=7,<9` | ✅ Compatible | Core testing framework |
| **pytest-cov** | `>=4,<8` | ✅ Compatible | Coverage reporting |
| **pytest-timeout** | `>=2,<3` | ✅ Compatible | Timeout functionality |
| **pytest-xdist** | `>=3,<4` | ✅ Compatible | Parallel execution |
| **pytest-benchmark** | `>=4,<6` | ✅ Compatible | Performance benchmarks |
| **hypothesis** | `>=6,<7` | ✅ Compatible | Property-based testing |
| **hypothesis-networkx** | `>=0.3,<1.0` | ✅ Compatible | Network graph generation |

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=tnfr --cov-report=html

# Run with parallelization
pytest tests/ -n auto

# Run specific category
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only

# Run with timeout protection
pytest tests/ --timeout=300

# Run benchmarks
pytest tests/ --benchmark-only
```

### Test Configuration

The test suite is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--tb=short",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "critical: Critical path tests that must pass",
    "slow: Tests that take >1s",
]
```

---

## Dependency Compatibility

### Pytest 8.x Compatibility

#### Verified Compatibility

All testing dependencies work correctly with pytest 8.x:

```bash
# Install test dependencies
pip install -e .[test-all]

# Verify pytest version
pytest --version  # Should show 8.x

# Run compatibility tests
pytest tests/ci/test_pytest_compatibility.py -v
```

#### Compatibility Test Suite

The repository includes a comprehensive compatibility test suite at `tests/ci/test_pytest_compatibility.py` that verifies:

- ✅ Pytest version is 8.x
- ✅ All plugins are available and loaded
- ✅ Plugin functionality works correctly
- ✅ No pytest deprecation warnings
- ✅ All pytest features work as expected
- ✅ Configuration from pyproject.toml loads correctly

#### Migration from Pytest 7.x to 8.x

**Good news**: No code changes required! The migration is seamless:

- All existing tests run without modification
- No API changes affecting this codebase
- All plugins work identically
- Configuration remains unchanged

### Dependency Version Strategy

TNFR uses a **forward-compatible** version pinning strategy:

```python
# pyproject.toml strategy
dependencies = [
    "pytest>=7,<9",        # Allow 7.x and 8.x
    "pytest-cov>=4,<8",    # Allow 4.x through 7.x
    "pytest-timeout>=2,<3", # Pin major version
]
```

**Benefits**:
- ✅ Pin major versions to avoid breaking changes
- ✅ Allow minor and patch updates for bug fixes
- ✅ Test with latest versions in CI
- ✅ Document compatibility explicitly

### Future Considerations

When pytest 9.x is released:

1. Review release notes for breaking changes
2. Update version constraint from `<9` to `<10`
3. Run compatibility test suite
4. Update this documentation

---

## Test Optimization

### Test Execution Speed

| Optimization | Technique | Speedup | Trade-off |
|-------------|-----------|---------|-----------|
| **Parallelization** | `pytest -n auto` | 2-8x | None (recommended) |
| **Selective Running** | `-k pattern` | N/A | Must know what changed |
| **Test Markers** | `-m unit` | 10x+ | Skips integration tests |
| **Fast Backend** | `TNFR_BACKEND=numpy` | 1.5x | vs. unoptimized Python |
| **Caching** | `--cache-clear=no` | 1.2x | May miss cache invalidations |

### Parallel Test Execution

```bash
# Automatic parallelization
pytest tests/ -n auto

# Explicit worker count
pytest tests/ -n 4

# With coverage (slower but accurate)
pytest tests/ -n auto --cov=tnfr
```

**Best practices**:
- Use `-n auto` for CI/CD pipelines
- Use specific `-n N` for local development (N = CPU cores)
- Disable parallelization for debugging: `pytest tests/ -n 0`

### Selective Test Running

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run only unit tests
pytest tests/ -m unit

# Run only critical path tests
pytest tests/ -m critical

# Run tests matching pattern
pytest tests/ -k "test_coherence"

# Run specific test file
pytest tests/unit/test_operators.py

# Run specific test function
pytest tests/unit/test_operators.py::test_coherence_operator
```

### Test Profiling

Identify slow tests:

```bash
# Show slowest 10 tests
pytest tests/ --durations=10

# Show all test durations
pytest tests/ --durations=0

# Profile with detailed timing
pytest tests/ --durations=0 --verbose
```

Example output:
```
slowest durations
==================
2.45s call     tests/integration/test_large_network.py::test_1000_nodes
1.23s call     tests/integration/test_operators.py::test_resonance_propagation
0.89s call     tests/integration/test_coherence.py::test_global_coherence
...
```

### Mocking for Speed

Use mocks to avoid slow operations in unit tests:

```python
import pytest
from unittest.mock import Mock, patch

def test_operator_with_mock_graph():
    """Test operator without real graph creation."""
    # Create mock graph
    mock_graph = Mock()
    mock_graph.nodes.return_value = ['n1', 'n2', 'n3']
    mock_graph.edges.return_value = [('n1', 'n2'), ('n2', 'n3')]
    
    # Test operator logic
    from tnfr.operators import Coherence
    op = Coherence()
    # ... test with mock_graph ...

@patch('tnfr.backends.get_backend')
def test_with_mocked_backend(mock_backend):
    """Test without loading actual backend."""
    mock_backend.return_value = Mock()
    # ... test logic ...
```

### Fixture Optimization

Share expensive fixtures across tests:

```python
import pytest
import networkx as nx

@pytest.fixture(scope="module")
def large_graph():
    """Create graph once per module."""
    G = nx.erdos_renyi_graph(1000, 0.1, seed=42)
    # Add TNFR attributes
    for node in G.nodes():
        G.nodes[node]['nu_f'] = 1.0
        G.nodes[node]['phase'] = 0.0
    return G

@pytest.fixture(scope="session")
def test_config():
    """Load config once per test session."""
    return {
        'seed': 42,
        'tolerance': 1e-9,
        'backend': 'numpy',
    }

def test_with_expensive_fixture(large_graph, test_config):
    """Test reuses pre-created graph."""
    assert len(large_graph) == 1000
```

**Fixture scopes**:
- `function` (default): New instance per test
- `class`: One instance per test class
- `module`: One instance per test file
- `session`: One instance for entire test run

---

## Type Stub Testing

### Type Stub Automation

TNFR uses automated `.pyi` stub generation to maintain type safety. Type stubs are tested at multiple levels:

#### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: check-stubs
      name: Check for missing .pyi stub files
      entry: make stubs-check
      language: system
      pass_filenames: false
      always_run: true
```

**Behavior**:
- Runs automatically before every commit
- Checks for missing `.pyi` stub files
- Prevents commits if stubs are missing
- Fast (<1s typically)

#### CI Checks

The CI pipeline includes two stub checks:

```yaml
# .github/workflows/ci.yml (type-check job)
- name: Check stub files exist
  run: python scripts/generate_stubs.py --check

- name: Check stub file synchronization
  run: python scripts/generate_stubs.py --check-sync
```

**Checks**:
1. **Missing stubs** (`--check`): Verifies all Python modules have corresponding `.pyi` files
2. **Outdated stubs** (`--check-sync`): Verifies stubs are synchronized with implementations

#### Stub Generation Commands

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

### Type Checking

Run mypy to verify type correctness:

```bash
# Type check entire codebase
mypy src/tnfr

# Type check specific module
mypy src/tnfr/operators/

# Type check with strict settings
mypy src/tnfr --strict

# Type check tests too
mypy src/tnfr tests/
```

### Stub Troubleshooting

#### Issue: CI fails with "stub files outdated"

**Solution**:
```bash
# Run locally before pushing
make stubs-check-sync

# Regenerate outdated stubs
make stubs-sync

# Commit both .py and .pyi files
git add src/tnfr/module.py src/tnfr/module.pyi
git commit -m "Update module and regenerate stub"
```

#### Issue: Pre-commit hook fails

**Solution**:
```bash
# Generate missing stubs
make stubs

# Add generated stubs
git add src/tnfr/*.pyi

# Retry commit
git commit
```

#### Issue: Stub generation fails

**Solution**:
```bash
# Ensure mypy is installed
pip install -e .[typecheck]

# Check for syntax errors in .py files
python -m py_compile src/tnfr/module.py

# Run with verbose output
python scripts/generate_stubs.py --verbose

# Try dry-run to see what would be generated
python scripts/generate_stubs.py --dry-run
```

---

## Testing Patterns

### Factory Function Testing

Every factory function should follow this test pattern:

```python
import pytest
import numpy as np
from tnfr.mathematics.operators_factory import make_coherence_operator

class TestMakeCoherenceOperator:
    """Test suite for coherence operator factory."""
    
    def test_valid_construction_defaults(self):
        """Test construction with default parameters."""
        op = make_coherence_operator(dim=5)
        assert op.shape == (5, 5)
        assert op.is_hermitian()
        assert op.is_positive_semidefinite()
    
    def test_valid_construction_custom_params(self):
        """Test construction with custom parameters."""
        spectrum = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        op = make_coherence_operator(dim=5, spectrum=spectrum)
        eigenvalues = np.sort(np.linalg.eigvals(op.to_array()))
        assert np.allclose(eigenvalues, spectrum, atol=1e-9)
    
    def test_invalid_dimension(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            make_coherence_operator(dim=0)
        with pytest.raises(ValueError, match="Dimension must be positive"):
            make_coherence_operator(dim=-5)
    
    def test_structural_invariant_hermiticity(self):
        """Test that Hermiticity is guaranteed."""
        op = make_coherence_operator(dim=10)
        matrix = op.to_array()
        assert np.allclose(matrix, matrix.conj().T, atol=1e-9)
    
    def test_structural_invariant_psd(self):
        """Test that positive semidefiniteness is guaranteed."""
        op = make_coherence_operator(dim=10)
        eigenvalues = np.linalg.eigvalsh(op.to_array())
        assert np.all(eigenvalues >= -1e-9)  # Numerical tolerance
    
    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        spec = np.linspace(0.1, 0.5, 5)
        op1 = make_coherence_operator(dim=5, spectrum=spec)
        op2 = make_coherence_operator(dim=5, spectrum=spec)
        assert np.allclose(op1.to_array(), op2.to_array())
```

**Coverage checklist for factory tests**:
- ✓ Valid construction with defaults
- ✓ Valid construction with custom parameters
- ✓ Invalid dimension handling
- ✓ Invalid parameter handling
- ✓ Structural invariants (Hermiticity, PSD, etc.)
- ✓ Reproducibility (deterministic output)
- ✓ Edge cases (boundary values)

### Structural Operator Testing

Test structural operators against TNFR invariants:

```python
import pytest
import networkx as nx
from tnfr.operators import Coherence

class TestCoherenceOperator:
    """Test Coherence structural operator."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        G = nx.Graph()
        G.add_node('n1', nu_f=1.0, phase=0.0, coherence=0.5)
        G.add_node('n2', nu_f=1.0, phase=0.5, coherence=0.5)
        G.add_node('n3', nu_f=1.0, phase=1.0, coherence=0.5)
        G.add_edge('n1', 'n2')
        G.add_edge('n2', 'n3')
        return G
    
    def test_coherence_increases_c_t(self, simple_graph):
        """Test that Coherence operator increases total coherence C(t)."""
        from tnfr.metrics import total_coherence
        
        G = simple_graph
        c_before = total_coherence(G)
        
        # Apply Coherence operator
        Coherence()(G)
        
        c_after = total_coherence(G)
        assert c_after >= c_before, "Coherence should increase C(t)"
    
    def test_coherence_preserves_phase(self, simple_graph):
        """Test that Coherence operator does not modify phase."""
        G = simple_graph
        phases_before = {n: G.nodes[n]['phase'] for n in G.nodes()}
        
        # Apply Coherence operator
        Coherence()(G)
        
        phases_after = {n: G.nodes[n]['phase'] for n in G.nodes()}
        for node in G.nodes():
            assert phases_before[node] == phases_after[node]
    
    def test_coherence_preserves_vf(self, simple_graph):
        """Test that Coherence operator does not modify νf."""
        G = simple_graph
        vf_before = {n: G.nodes[n]['nu_f'] for n in G.nodes()}
        
        # Apply Coherence operator
        Coherence()(G)
        
        vf_after = {n: G.nodes[n]['nu_f'] for n in G.nodes()}
        for node in G.nodes():
            assert vf_before[node] == vf_after[node]
    
    def test_coherence_bounded(self, simple_graph):
        """Test that coherence stays in [0, 1] range."""
        G = simple_graph
        
        # Apply Coherence operator multiple times
        for _ in range(10):
            Coherence()(G)
        
        # Verify bounds
        for node in G.nodes():
            c = G.nodes[node]['coherence']
            assert 0.0 <= c <= 1.0, f"Coherence {c} out of bounds for {node}"
```

### Property-Based Testing

Use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st
from hypothesis_networkx import graph_builder
import networkx as nx

@given(
    graph=graph_builder(
        graph_type=nx.Graph,
        min_nodes=3,
        max_nodes=20,
        edge_probability=st.floats(0.1, 0.5)
    ),
    nu_f=st.floats(0.1, 10.0),
    phase=st.floats(0.0, 2 * 3.14159)
)
def test_operator_preserves_structural_invariants(graph, nu_f, phase):
    """Test that operators preserve structural invariants for any valid graph."""
    # Initialize graph with TNFR attributes
    for node in graph.nodes():
        graph.nodes[node]['nu_f'] = nu_f
        graph.nodes[node]['phase'] = phase
        graph.nodes[node]['coherence'] = 1.0
    
    # Apply operator
    from tnfr.operators import Coherence
    Coherence()(graph)
    
    # Verify invariants
    for node in graph.nodes():
        assert graph.nodes[node]['nu_f'] >= 0, "νf must be non-negative"
        assert 0 <= graph.nodes[node]['phase'] < 2 * 3.14159, "Phase must be in [0, 2π)"
        assert 0 <= graph.nodes[node]['coherence'] <= 1, "Coherence in [0, 1]"
```

### Reproducibility Testing

Ensure deterministic behavior:

```python
import pytest
import networkx as nx
from tnfr.structural import create_nfr

def test_create_nfr_reproducibility():
    """Test that create_nfr is deterministic with same seed."""
    G1 = nx.Graph()
    G2 = nx.Graph()
    
    # Create with same seed
    G1, node1 = create_nfr(G1, nu_f=2.0, phase=0.5, seed=42)
    G2, node2 = create_nfr(G2, nu_f=2.0, phase=0.5, seed=42)
    
    # Should produce identical results
    assert node1 == node2
    assert G1.nodes[node1]['nu_f'] == G2.nodes[node2]['nu_f']
    assert G1.nodes[node1]['phase'] == G2.nodes[node2]['phase']

def test_operator_sequence_reproducibility():
    """Test that operator sequences are reproducible."""
    import numpy as np
    from tnfr.operators import Coherence, Resonance
    
    # Create two identical graphs
    np.random.seed(42)
    G1 = nx.erdos_renyi_graph(10, 0.3, seed=42)
    for node in G1.nodes():
        G1.nodes[node]['nu_f'] = 1.0
        G1.nodes[node]['phase'] = 0.0
        G1.nodes[node]['coherence'] = 1.0
    
    np.random.seed(42)
    G2 = nx.erdos_renyi_graph(10, 0.3, seed=42)
    for node in G2.nodes():
        G2.nodes[node]['nu_f'] = 1.0
        G2.nodes[node]['phase'] = 0.0
        G2.nodes[node]['coherence'] = 1.0
    
    # Apply same operations
    ops = [Coherence(), Resonance()]
    for op in ops:
        op(G1)
        op(G2)
    
    # Compare results
    for node in G1.nodes():
        assert G1.nodes[node]['coherence'] == G2.nodes[node]['coherence']
```

---

## CI/CD Integration

### GitHub Actions Workflow

The CI pipeline runs tests at multiple stages:

```yaml
# .github/workflows/ci.yml (simplified)
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e .[test-all]
      
      - name: Run tests
        run: |
          pytest tests/ -n auto --cov=tnfr --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check stub files exist
        run: make stubs-check
      
      - name: Check stub file synchronization
        run: make stubs-check-sync
      
      - name: Run mypy
        run: mypy src/tnfr
```

### Test Stages

```
┌─────────────────┐
│  Pre-commit     │  ← Local: Stub checks, formatting
└────────┬────────┘
         │
┌────────▼────────┐
│  Push to GitHub │
└────────┬────────┘
         │
┌────────▼────────┐
│  Type Check Job │  ← CI: Stub validation, mypy
└────────┬────────┘
         │
┌────────▼────────┐
│  Test Job       │  ← CI: pytest with coverage
│  (3.9-3.13)     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Coverage       │  ← Upload to Codecov
└─────────────────┘
```

### Local Pre-Push Checks

Run these checks before pushing:

```bash
# 1. Check for missing stubs
make stubs-check

# 2. Check stub synchronization
make stubs-check-sync

# 3. Run tests
pytest tests/ -n auto

# 4. Check coverage
pytest tests/ --cov=tnfr --cov-report=term-missing

# 5. Type check
mypy src/tnfr

# All-in-one command
make stubs-check && make stubs-check-sync && pytest tests/ -n auto && mypy src/tnfr
```

---

## Troubleshooting

### Common Test Failures

#### Issue: "ModuleNotFoundError: No module named 'tnfr'"

**Cause**: Package not installed in editable mode

**Solution**:
```bash
pip install -e .
# Or with test dependencies
pip install -e .[test-all]
```

#### Issue: Tests hang or timeout

**Cause**: Infinite loop or deadlock in code

**Solution**:
```bash
# Run with timeout protection
pytest tests/ --timeout=300

# Run specific test with verbose output
pytest tests/path/to/test.py::test_function -vv

# Add timeout to specific test
@pytest.mark.timeout(10)
def test_might_hang():
    ...
```

#### Issue: Flaky tests (pass/fail randomly)

**Cause**: Missing or incorrect random seeds

**Solution**:
```python
# Add explicit seeds
def test_with_seed():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    # ... test code ...

# Use fixtures for consistent state
@pytest.fixture
def seeded_rng():
    return np.random.default_rng(42)
```

#### Issue: Coverage drops unexpectedly

**Cause**: New code without tests, or test skipped

**Solution**:
```bash
# Check what's not covered
pytest tests/ --cov=tnfr --cov-report=term-missing

# Look for skipped tests
pytest tests/ -v | grep SKIPPED

# Check branch coverage
pytest tests/ --cov=tnfr --cov-branch
```

### Performance Issues

#### Issue: Tests take too long

**Diagnosis**:
```bash
# Find slow tests
pytest tests/ --durations=20
```

**Solutions**:
1. Use smaller graphs in tests
2. Mock expensive operations
3. Use module/session-scoped fixtures
4. Enable parallelization: `pytest -n auto`

#### Issue: High memory usage

**Diagnosis**:
```bash
# Profile memory
pytest tests/ --memprof
```

**Solutions**:
1. Use `del` to free large objects
2. Scope fixtures appropriately
3. Use generators instead of lists
4. Run fewer tests in parallel

---

## Best Practices Summary

### DO:
- ✅ Write tests for all factory functions
- ✅ Test structural invariants (Hermiticity, PSD, phase bounds)
- ✅ Use explicit seeds for reproducibility
- ✅ Run tests in parallel (`-n auto`)
- ✅ Keep unit tests fast (<1ms)
- ✅ Use fixtures to share expensive setup
- ✅ Generate stubs before committing
- ✅ Run pre-push checks locally

### DON'T:
- ❌ Skip testing structural invariants
- ❌ Rely on global state or random behavior
- ❌ Create massive graphs in unit tests
- ❌ Forget to add timeouts to potentially slow tests
- ❌ Commit without running stub checks
- ❌ Ignore deprecation warnings
- ❌ Mock core TNFR logic (test real behavior)

---

## See Also

- [Architecture Guide](ARCHITECTURE_GUIDE.md) - Factory patterns and dependency management
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md) - Optimization strategies
- [Development Workflow](DEVELOPMENT_WORKFLOW.md) - Contributing guidelines
- [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) - General contribution guide
- [TESTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) - Test strategy overview

---

**Last Updated**: 2025-11-06  
**Status**: Active - consolidates TESTING_*, STUB_*, TEST_* docs  
**Maintenance**: Update when testing patterns change, review quarterly
