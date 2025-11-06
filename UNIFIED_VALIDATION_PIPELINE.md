# Unified Validation Pipeline Migration Guide

## Overview

Phase 4 introduces a **Unified Validation Pipeline** through the enhanced `TNFRValidator` class. This consolidates all scattered validation logic into a single, coherent API that enforces all canonical TNFR invariants.

## Why Unified Validation?

### Before (Scattered Validation)

Previously, validation logic was scattered across multiple modules:

```python
# Input validation
from tnfr.validation import validate_epi_value, validate_vf_value, validate_theta_value

# Graph validation
from tnfr.validation.graph import run_validators

# Runtime validation
from tnfr.validation.runtime import validate_canon

# Operator preconditions
from tnfr.operators.preconditions import validate_emission

# Security validation
from tnfr.security.validation import validate_structural_frequency

# Invariant checking - manual instantiation
from tnfr.validation.invariants import Invariant1_EPIOnlyThroughOperators
inv1 = Invariant1_EPIOnlyThroughOperators()
violations = inv1.validate(graph)
```

**Problems:**
- Multiple import paths to remember
- Inconsistent APIs across validation types
- No single source of truth for validation
- Difficult to ensure all validations are applied
- Code duplication and maintenance burden

### After (Unified Pipeline)

Now, everything is available through `TNFRValidator`:

```python
from tnfr.validation import TNFRValidator

validator = TNFRValidator()

# Single comprehensive validation call
result = validator.validate(
    graph=G,
    epi=0.5,
    vf=1.0,
    theta=0.0,
    include_invariants=True,
    include_graph_structure=True,
)

if not result['passed']:
    print(f"Validation failed: {result['errors']}")
```

**Benefits:**
- Single import
- Unified API
- Comprehensive validation in one call
- Consistent error handling
- Built-in caching for performance
- Extensible with custom validators

## API Reference

### TNFRValidator Class

The unified entry point for all TNFR validation operations.

#### Initialization

```python
validator = TNFRValidator(
    phase_coupling_threshold=math.pi/2,  # Optional: custom phase threshold
    enable_input_validation=True,        # Optional: enable/disable input validation
    enable_graph_validation=True,        # Optional: enable/disable graph validation
    enable_runtime_validation=True,      # Optional: enable/disable runtime validation
)
```

#### Main Methods

##### `validate()` - Comprehensive Unified Validation

The primary entry point for all validation needs:

```python
result = validator.validate(
    graph=G,                    # Optional: graph to validate
    epi=0.5,                    # Optional: EPI value to validate
    vf=1.0,                     # Optional: νf value to validate
    theta=0.0,                  # Optional: θ value to validate
    dnfr=0.0,                   # Optional: ΔNFR value to validate
    node_id='node_1',          # Optional: node ID to validate
    operator='emission',        # Optional: operator preconditions to check
    include_invariants=True,    # Include invariant validation
    include_graph_structure=True,  # Include graph structure validation
    include_runtime=False,      # Include runtime canonical validation
    raise_on_error=False,       # Whether to raise on first error
)

# Result structure
{
    'passed': bool,                      # Overall validation status
    'inputs': dict,                      # Input validation results
    'graph_structure': dict,             # Graph structure results
    'runtime': dict,                     # Runtime validation results
    'invariants': list[InvariantViolation],  # Invariant violations
    'operator_preconditions': bool,      # Operator precondition status
    'errors': list[str],                 # Any errors encountered
}
```

##### `validate_inputs()` - Input Validation

Validate structural operator inputs:

```python
result = validator.validate_inputs(
    epi=0.5,
    vf=1.0,
    theta=0.0,
    dnfr=0.0,
    node_id='node_1',
    glyph=Glyph.EMISSION,
    graph=G,
    config=G.graph,           # Optional: configuration for bounds
    raise_on_error=True,      # Whether to raise on validation failure
)

# Returns dict with validated values or error
{'epi': 0.5, 'vf': 1.0, 'theta': 0.0}
```

##### `validate_graph()` - Graph Validation

Validate graph against all TNFR invariants:

```python
violations = validator.validate_graph(
    graph=G,
    severity_filter=InvariantSeverity.ERROR,  # Optional: filter by severity
    use_cache=True,                           # Use cached results if available
    include_graph_validation=True,            # Include structure validation
    include_runtime_validation=False,         # Include runtime validation
)

# Returns list of InvariantViolation objects
```

##### `validate_operator_preconditions()` - Operator Preconditions

Validate operator preconditions before application:

```python
is_valid = validator.validate_operator_preconditions(
    graph=G,
    node='node_1',
    operator='emission',
    raise_on_error=True,
)

if is_valid:
    # Apply operator
    pass
```

##### `validate_graph_structure()` - Graph Structure

Validate graph structure and coherence:

```python
result = validator.validate_graph_structure(
    graph=G,
    raise_on_error=True,
)

# Returns dict with validation results
{'passed': True, 'message': 'Graph structure valid'}
```

##### `validate_runtime_canonical()` - Runtime Validation

Validate runtime canonical constraints:

```python
result = validator.validate_runtime_canonical(
    graph=G,
    raise_on_error=True,
)

# Returns dict with validation results
{'passed': True, 'summary': {...}, 'artifacts': {...}}
```

#### Reporting Methods

##### `generate_report()` - Human-Readable Report

```python
violations = validator.validate_graph(G)
report = validator.generate_report(violations)
print(report)
```

Output:
```
🚨 TNFR Invariant Violations Detected:

💥 CRITICAL (1):
  Invariant #1: EPI is not a finite number
    Node: node_1
    Expected: finite float
    Actual: inf
    💡 Suggestion: Check operator implementation for EPI assignment

❌ ERROR (2):
  ...
```

##### `export_to_json()` - JSON Export

```python
violations = validator.validate_graph(G)
json_report = validator.export_to_json(violations)

# Returns JSON string with structure:
{
    "total_violations": 3,
    "by_severity": {
        "critical": 1,
        "error": 2,
        "warning": 0,
        "info": 0
    },
    "violations": [...]
}
```

##### `export_to_html()` - HTML Export

```python
violations = validator.validate_graph(G)
html_report = validator.export_to_html(violations)

# Save to file
with open('validation_report.html', 'w') as f:
    f.write(html_report)
```

#### Advanced Features

##### Result Caching

```python
validator.enable_cache(True)

# First call - computes violations
violations1 = validator.validate_graph(G)

# Second call - uses cache
violations2 = validator.validate_graph(G, use_cache=True)

# Clear cache when graph changes
validator.clear_cache()
```

##### Custom Validators

```python
from tnfr.validation.invariants import TNFRInvariant, InvariantViolation

class CustomInvariant(TNFRInvariant):
    invariant_id = 100
    description = "Custom domain-specific invariant"
    
    def validate(self, graph):
        violations = []
        # Your custom validation logic
        return violations

# Add to validator
validator.add_custom_validator(CustomInvariant())
```

##### Severity Filtering

```python
# Get only critical violations
critical = validator.validate_graph(
    G,
    severity_filter=InvariantSeverity.CRITICAL
)

# Get errors and critical
errors_and_critical = validator.validate_graph(G)
filtered = [v for v in errors_and_critical 
            if v.severity in (InvariantSeverity.ERROR, InvariantSeverity.CRITICAL)]
```

## Migration Examples

### Example 1: Basic Input Validation

**Before:**
```python
from tnfr.validation.input_validation import (
    validate_epi_value,
    validate_vf_value,
    validate_theta_value,
)

try:
    epi = validate_epi_value(user_epi, config=G.graph)
    vf = validate_vf_value(user_vf, config=G.graph)
    theta = validate_theta_value(user_theta)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

**After:**
```python
from tnfr.validation import TNFRValidator

validator = TNFRValidator()

try:
    result = validator.validate_inputs(
        epi=user_epi,
        vf=user_vf,
        theta=user_theta,
        config=G.graph,
        raise_on_error=True,
    )
    epi = result['epi']
    vf = result['vf']
    theta = result['theta']
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Example 2: Graph Validation

**Before:**
```python
from tnfr.validation.graph import run_validators
from tnfr.validation.invariants import (
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    # ... import all 10 invariants
)

try:
    run_validators(G)
    
    # Check each invariant manually
    inv1 = Invariant1_EPIOnlyThroughOperators()
    violations1 = inv1.validate(G)
    
    inv2 = Invariant2_VfInHzStr()
    violations2 = inv2.validate(G)
    
    # ... repeat for all invariants
    
    all_violations = violations1 + violations2 + ...
    
    if all_violations:
        print("Violations found!")
except Exception as e:
    print(f"Validation failed: {e}")
```

**After:**
```python
from tnfr.validation import TNFRValidator

validator = TNFRValidator()

violations = validator.validate_graph(
    G,
    include_graph_validation=True,
    include_runtime_validation=False,
)

if violations:
    report = validator.generate_report(violations)
    print(report)
```

### Example 3: Operator Preconditions

**Before:**
```python
from tnfr.operators.preconditions import (
    validate_emission,
    validate_reception,
    OperatorPreconditionError,
)

try:
    if operator_name == 'emission':
        validate_emission(G, node)
    elif operator_name == 'reception':
        validate_reception(G, node)
    # ... handle all operators
    
    # Apply operator
    apply_operator(G, node, operator_name)
except OperatorPreconditionError as e:
    print(f"Precondition not met: {e}")
```

**After:**
```python
from tnfr.validation import TNFRValidator

validator = TNFRValidator()

if validator.validate_operator_preconditions(G, node, operator_name, raise_on_error=False):
    # Apply operator
    apply_operator(G, node, operator_name)
else:
    print(f"Preconditions not met for {operator_name}")
```

### Example 4: Comprehensive Validation

**Before:**
```python
from tnfr.validation.input_validation import validate_epi_value, validate_vf_value
from tnfr.validation.graph import run_validators
from tnfr.validation.runtime import validate_canon
from tnfr.operators.preconditions import validate_emission
from tnfr.validation.invariants import Invariant1_EPIOnlyThroughOperators

# Validate inputs
epi = validate_epi_value(0.5, config=G.graph)
vf = validate_vf_value(1.0, config=G.graph)

# Validate graph structure
run_validators(G)

# Validate runtime
outcome = validate_canon(G)
if not outcome.passed:
    print("Runtime validation failed")

# Check operator preconditions
validate_emission(G, node)

# Check invariants
inv1 = Invariant1_EPIOnlyThroughOperators()
violations = inv1.validate(G)
```

**After:**
```python
from tnfr.validation import TNFRValidator

validator = TNFRValidator()

result = validator.validate(
    graph=G,
    epi=0.5,
    vf=1.0,
    node_id=node,
    operator='emission',
    include_invariants=True,
    include_graph_structure=True,
    include_runtime=True,
)

if result['passed']:
    print("All validations passed!")
else:
    print(f"Validation failed: {result['errors']}")
    if result['invariants']:
        report = validator.generate_report(result['invariants'])
        print(report)
```

## Performance Considerations

### Caching

For repeated validations of the same graph:

```python
validator = TNFRValidator()
validator.enable_cache(True)

# Multiple validations use cache
for _ in range(100):
    violations = validator.validate_graph(G, use_cache=True)

# Clear cache when graph changes
G.nodes['node_1']['EPI'] = 0.7
validator.clear_cache()
```

### Selective Validation

Disable validation layers you don't need:

```python
# Only input and invariant validation, skip graph structure and runtime
validator = TNFRValidator(
    enable_input_validation=True,
    enable_graph_validation=False,    # Skip graph structure checks
    enable_runtime_validation=False,  # Skip runtime canonical checks
)

result = validator.validate(
    graph=G,
    epi=0.5,
    include_graph_structure=False,
    include_runtime=False,
)
```

### Batch Validation

For validating multiple graphs:

```python
validator = TNFRValidator()
validator.enable_cache(True)

results = {}
for graph_name, graph in graphs.items():
    violations = validator.validate_graph(graph)
    results[graph_name] = violations
    validator.clear_cache()  # Clear between graphs
```

## Best Practices

1. **Use the unified `validate()` method** for comprehensive validation in one call
2. **Enable caching** for repeated validations of the same graph
3. **Use specific methods** (`validate_inputs()`, `validate_graph()`) when you only need one validation type
4. **Filter by severity** to focus on critical issues first
5. **Export reports** in your preferred format (text, JSON, HTML)
6. **Add custom validators** for domain-specific constraints
7. **Handle validation errors gracefully** - use `raise_on_error=False` for non-critical paths

## Deprecation Timeline

- **v0.5.0**: Unified TNFRValidator introduced, old APIs still work
- **v0.5.x**: Deprecation warnings added to old APIs
- **v0.6.0**: Old APIs removed, only unified API supported

## Support

For questions or issues with the unified validation pipeline:

1. Check this migration guide
2. Review the comprehensive test suite: `tests/unit/validation/test_unified_validator.py`
3. Consult the API documentation
4. Open an issue on GitHub

## Summary

The unified validation pipeline provides:

✅ Single entry point (`TNFRValidator`)  
✅ Consistent API across all validation types  
✅ Comprehensive validation in one call  
✅ Built-in caching for performance  
✅ Flexible configuration  
✅ Extensive reporting options  
✅ Extensibility with custom validators  
✅ Complete TNFR invariant coverage  

Migrate to the unified API for a better validation experience!
