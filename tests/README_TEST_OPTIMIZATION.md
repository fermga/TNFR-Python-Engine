# Test Suite Optimization Guide

This document provides guidance on the optimized test suite structure and how to maintain it going forward.

## Overview

The test suite has been optimized to follow DRY (Don't Repeat Yourself) principles while increasing coverage for critical paths. This was accomplished through:

1. **Shared test utilities** providing reusable base classes and validation helpers
2. **Unified parametrized test suites** consolidating redundant tests
3. **Critical path coverage** for operator generation, validators, and run_sequence
4. **Clear documentation** of what can be deprecated and why

## Quick Start

### Running the Optimized Tests

```bash
# Run all new unified and critical path tests (173 tests, ~0.4s)
pytest tests/integration/test_unified_*.py tests/integration/test_*_critical_paths.py -v

# Run specific test suite
pytest tests/integration/test_unified_structural_validation.py -v
pytest tests/integration/test_unified_operator_validation.py -v
pytest tests/integration/test_operator_generation_critical_paths.py -v
```

### Using Shared Test Utilities

```python
# Import shared validation helpers
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_dnfr_homogeneous_stable,
    assert_epi_vf_in_bounds,
)

# Import shared fixtures
from tests.helpers.fixtures import (
    seed_graph_factory,
    homogeneous_graph_factory,
    bicluster_graph_factory,
)

# Import base classes for new test suites
from tests.helpers.base import (
    BaseStructuralTest,
    BaseOperatorTest,
    BaseValidatorTest,
)
```

## File Structure

```
tests/
├── helpers/
│   ├── base.py                          # NEW: Reusable base classes
│   ├── validation.py                    # ENHANCED: Shared validators
│   └── fixtures.py                      # EXISTING: Shared fixtures
├── integration/
│   ├── test_unified_structural_validation.py      # NEW: 23 tests
│   ├── test_unified_operator_validation.py        # NEW: 96 tests
│   ├── test_operator_generation_critical_paths.py # NEW: 17 tests
│   ├── test_nodal_validators_critical_paths.py    # NEW: 16 tests
│   └── test_run_sequence_critical_paths.py        # NEW: 21 tests
└── TEST_CONSOLIDATION_SUMMARY.md        # NEW: Deprecation guide
```

## Key Improvements

### 1. Parametrized Testing

Instead of writing multiple similar tests:

```python
# OLD: Multiple redundant tests
def test_dnfr_conservation_small():
    graph = create_graph(5)
    assert_conserved(graph)

def test_dnfr_conservation_medium():
    graph = create_graph(50)
    assert_conserved(graph)
```

Use parametrization:

```python
# NEW: Single parametrized test
@pytest.mark.parametrize("num_nodes", [5, 50])
def test_dnfr_conservation_unified(num_nodes):
    graph = create_graph(num_nodes)
    assert_conserved(graph)
```

### 2. Shared Base Classes

Instead of duplicating validation logic:

```python
# OLD: Duplicated logic in each test class
class TestFeatureA:
    def test_conservation(self):
        # duplicate conservation check
        pass
    
    def test_homogeneity(self):
        # duplicate homogeneity check
        pass
```

Inherit from base classes:

```python
# NEW: Inherit common patterns
from tests.helpers.base import BaseStructuralTest

class TestFeatureA(BaseStructuralTest):
    def create_test_graph(self, **kwargs):
        # Only implement graph creation
        return my_specific_graph(**kwargs)
    
    # Automatically inherit:
    # - test_dnfr_conservation()
    # - test_homogeneous_stability()
    # - test_structural_bounds()
```

### 3. Unified Fixtures

Use parametrized fixtures for common configurations:

```python
@pytest.fixture(params=[
    {"epi_value": 0.0, "vf_value": 1.0},
    {"epi_value": 0.5, "vf_value": 1.5},
])
def homogeneous_config(request):
    return request.param

def test_with_config(homogeneous_config):
    # Automatically runs for all configurations
    pass
```

## Writing New Tests

### Pattern 1: Using Shared Validators

```python
def test_my_new_feature(seed_graph_factory):
    """Test new feature maintains structural invariants."""
    graph = seed_graph_factory(num_nodes=10, edge_probability=0.3, seed=42)
    
    # Apply your operation
    my_operation(graph)
    
    # Use shared validators
    assert_dnfr_balanced(graph)
    assert_epi_vf_in_bounds(graph, epi_min=-1.0, epi_max=1.0)
```

### Pattern 2: Parametrized Tests

```python
@pytest.mark.parametrize("param_value", [0.1, 1.0, 10.0])
def test_parameter_scaling(param_value):
    """Test behavior across parameter range."""
    result = my_function(param_value)
    assert_valid(result)
```

### Pattern 3: Using Base Classes

```python
from tests.helpers.base import BaseOperatorTest

class TestMyOperator(BaseOperatorTest):
    def create_operator(self, **kwargs):
        return MyOperator(**kwargs)
    
    # Inherits:
    # - assert_hermitian()
    # - assert_finite_values()
    # - assert_real_eigenvalues()
```

## Maintenance Guidelines

### Adding New Tests

1. **Check for existing utilities first**
   - Review `tests/helpers/base.py` for applicable base classes
   - Review `tests/helpers/validation.py` for validators
   - Review `tests/helpers/fixtures.py` for fixtures

2. **Use parametrization for variations**
   - Don't create multiple tests that differ only in parameters
   - Use `@pytest.mark.parametrize` or parametrized fixtures

3. **Follow TNFR naming conventions**
   - Use structural terminology (coherence, ΔNFR, νf, EPI)
   - Avoid implementation details in test names
   - Focus on structural properties being validated

### Deprecating Old Tests

Before removing tests marked for deprecation:

1. **Verify unified tests provide equivalent coverage**
   ```bash
   # Run both old and new tests
   pytest tests/integration/test_old.py tests/integration/test_unified_*.py -v
   ```

2. **Check for any unique edge cases**
   - Review test code for unique validation logic
   - Ensure unified tests cover these cases

3. **Update documentation**
   - Remove from test count estimates
   - Update TEST_CONSOLIDATION_SUMMARY.md

### Code Review Checklist

When reviewing test PRs, verify:

- [ ] Uses shared validation helpers where applicable
- [ ] Uses parametrization instead of duplication
- [ ] Follows TNFR structural terminology
- [ ] Includes docstrings explaining structural intent
- [ ] Maintains TNFR invariants (see AGENTS.md §3)
- [ ] Uses appropriate fixtures from `tests/helpers/`

## Performance Considerations

### Test Execution Time

```bash
# Optimized test suite is fast
$ pytest tests/integration/test_unified_*.py tests/integration/test_*_critical_paths.py
173 passed in 0.42s

# Individual suites
$ pytest tests/integration/test_unified_structural_validation.py
23 passed in 0.18s

$ pytest tests/integration/test_unified_operator_validation.py
96 passed in 0.23s
```

### Parallel Execution

The parametrized tests can run in parallel:

```bash
# Use pytest-xdist for parallel execution
pytest -n auto tests/integration/test_unified_*.py
```

## Troubleshooting

### "Test not found" errors

Make sure you're in the repository root:
```bash
cd /path/to/TNFR-Python-Engine
pytest tests/integration/test_unified_*.py
```

### Import errors

Ensure the package is installed in editable mode:
```bash
pip install -e ".[test,numpy]"
```

### Fixture not found

Verify the fixture is defined in:
- `tests/conftest.py` (global fixtures)
- `tests/helpers/fixtures.py` (shared fixtures)
- The test file itself (local fixtures)

## Statistics

### Coverage Improvements

| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| Operator tests | ~20 separate | 96 parametrized | +76 tests, 60% less code |
| Structural tests | ~13 separate | 23 parametrized | +10 tests, 60% less code |
| Critical paths | Limited | 54 new tests | New coverage |
| **Total** | ~240 tests | **413+ tests** | **+173 tests** |

### Test Execution Speed

- Unified structural: 0.18s for 23 tests
- Unified operators: 0.23s for 96 tests
- Critical paths: 0.15s for 54 tests
- **Total: 0.42s for 173 tests**

## References

- `TEST_CONSOLIDATION_SUMMARY.md` - Detailed consolidation analysis
- `tests/helpers/base.py` - Base class documentation
- `tests/helpers/validation.py` - Shared validators
- `AGENTS.md` - TNFR invariants and structural contracts

## Contact

For questions about the test optimization:
1. Review this guide and TEST_CONSOLIDATION_SUMMARY.md
2. Check the inline documentation in base classes
3. Review example usage in unified test suites
4. Consult AGENTS.md for TNFR structural requirements
