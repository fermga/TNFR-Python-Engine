# Input Validation for TNFR Structural Operators

## Overview

This document describes the input validation mechanisms implemented for TNFR structural operators to prevent injection attacks and ensure canonical TNFR invariants are preserved.

## Security Issue Addressed

**Issue #2518**: Input validation for structural operators to prevent:
- Injection attacks (XSS, template injection, command injection)
- Special float value attacks (nan, inf)
- Type confusion attacks
- Numeric overflow attacks
- Violation of TNFR canonical bounds

## Validation Module

The `tnfr.validation.input_validation` module provides comprehensive validation functions for all structural operator parameters.

### Core Validation Functions

#### `validate_epi_value(value, config=None, allow_complex=True)`

Validates EPI (Primary Information Structure) values.

**Parameters**:
- `value`: EPI value to validate (float or complex)
- `config`: Optional graph configuration for custom bounds
- `allow_complex`: Whether to allow complex EPI values

**Validation Rules**:
- Must be numeric (int, float, or complex if allowed)
- Cannot be nan or inf
- Magnitude must be within bounds (default: 0.0 to 1.0)
- Respects custom `EPI_MIN` and `EPI_MAX` from configuration

**Example**:
```python
from tnfr.validation import validate_epi_value, ValidationError

# Valid EPI
epi = validate_epi_value(0.5)

# Custom bounds
config = {"EPI_MIN": 0.1, "EPI_MAX": 0.8}
epi = validate_epi_value(0.6, config=config)

# Invalid: exceeds bounds
try:
    validate_epi_value(1.5)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### `validate_vf_value(value, config=None)`

Validates νf (structural frequency) values in Hz_str units.

**Parameters**:
- `value`: νf value to validate (must be real number)
- `config`: Optional graph configuration for custom bounds

**Validation Rules**:
- Must be numeric (int or float, not complex)
- Cannot be nan or inf
- Must be non-negative
- Must be within bounds (default: 0.0 to 1.0)
- Respects custom `VF_MIN` and `VF_MAX` from configuration

**Example**:
```python
from tnfr.validation import validate_vf_value

# Valid νf
vf = validate_vf_value(1.0)

# Invalid: negative
try:
    validate_vf_value(-0.5)
except ValidationError:
    pass  # Rejected
```

#### `validate_theta_value(value, normalize=True)`

Validates θ (phase) values.

**Parameters**:
- `value`: θ value to validate (must be real number)
- `normalize`: Whether to normalize to [-π, π]

**Validation Rules**:
- Must be numeric (int or float, not complex)
- Cannot be nan or inf
- Optionally normalized to [-π, π] range

**Example**:
```python
import math
from tnfr.validation import validate_theta_value

# Valid phase
theta = validate_theta_value(math.pi / 2)

# Normalized phase
theta = validate_theta_value(3 * math.pi, normalize=True)
# Returns: -π (wrapped around)
```

#### `validate_dnfr_value(value, config=None)`

Validates ΔNFR (reorganization operator) magnitude.

**Parameters**:
- `value`: ΔNFR value to validate (must be real number)
- `config`: Optional graph configuration for custom bounds

**Validation Rules**:
- Must be numeric (int or float, not complex)
- Cannot be nan or inf
- Absolute value must not exceed maximum (default: 1.0)
- Respects custom `DNFR_MAX` from configuration

#### `validate_node_id(value)`

Validates NodeId values for security.

**Validation Rules**:
- Must be hashable
- String NodeIds cannot contain control characters
- String NodeIds are checked for injection patterns:
  - XSS: `<script`, `javascript:`, event handlers
  - Template injection: `${`, backticks
  - Other suspicious patterns

**Example**:
```python
from tnfr.validation import validate_node_id

# Valid NodeIds
validate_node_id("node_1")
validate_node_id(42)
validate_node_id((1, 2))

# Invalid: XSS attempt
try:
    validate_node_id("<script>alert('xss')</script>")
except ValidationError:
    pass  # Rejected
```

#### `validate_glyph(value)`

Validates Glyph enumeration values.

**Validation Rules**:
- Must be a valid Glyph enumeration or valid glyph string

#### `validate_tnfr_graph(value)`

Validates TNFRGraph instances.

**Validation Rules**:
- Must be a networkx Graph instance
- Must have `graph` attribute for metadata

#### `validate_glyph_factors(factors, required_keys=None)`

Validates glyph factors dictionary.

**Validation Rules**:
- Must be a mapping
- All keys must be strings
- All values must be numeric and finite
- Can check for required keys

#### `validate_operator_parameters(parameters, config=None)`

Validates a dictionary of operator parameters.

Automatically validates known parameters (epi, vf, theta, dnfr, node, glyph, G, glyph_factors) and passes through unknown parameters unchanged.

## Integration Points

### Operator Call Sites

Validation is integrated at key operator call sites:

1. **`apply_glyph(G, n, glyph, window=None)`**
   - Validates TNFRGraph, NodeId, and Glyph before operator application
   - Located in: `src/tnfr/operators/__init__.py`

2. **`apply_glyph_obj(node, glyph, window=None)`**
   - Validates Glyph enumeration before dispatch
   - Located in: `src/tnfr/operators/__init__.py`

3. **`get_factor(gf, key, default)`**
   - Validates glyph factor values for numeric safety
   - Protects against nan/inf in operator coefficients
   - Located in: `src/tnfr/operators/__init__.py`

4. **`create_nfr(name, epi=0.0, vf=1.0, theta=0.0, ...)`**
   - Validates all node creation parameters
   - Located in: `src/tnfr/structural.py`

### Error Handling

Validation failures raise `ValidationError` with detailed information:

```python
from tnfr.validation import ValidationError

try:
    validate_epi_value(10.0)
except ValidationError as e:
    print(f"Parameter: {e.parameter}")
    print(f"Value: {e.value}")
    print(f"Constraint: {e.constraint}")
    print(f"Message: {e}")
```

## TNFR Canonical Invariants

All validation functions respect TNFR structural semantics:

1. **EPI Coherence**: EPI magnitude must remain within bounds to preserve structural stability
2. **νf Bounds**: Structural frequency must be positive and bounded (Hz_str units)
3. **Phase Normalization**: Phase values normalized to [-π, π] for canonical representation
4. **ΔNFR Constraints**: Reorganization operator magnitude bounded to prevent excessive structural changes
5. **Type Safety**: Strict type checking for TNFRGraph, NodeId, Glyph enumerations

## Security Features

### Injection Attack Prevention

**XSS Prevention**:
- NodeId strings are checked for `<script`, `<iframe`, `<img`, `<svg>` tags
- Event handler patterns (`onclick=`, `onerror=`) are blocked

**Template Injection Prevention**:
- `${...}` patterns are blocked
- Backtick characters are blocked

**Command Injection Prevention**:
- Control characters are blocked in NodeId strings

### Numeric Attack Prevention

**Special Float Values**:
- `nan` (Not a Number) is rejected in all numeric parameters
- `inf` (infinity) is rejected in all numeric parameters
- Prevents NaN propagation through calculations

**Type Confusion**:
- Strict type checking prevents string-to-number coercion
- Complex numbers only allowed where explicitly permitted (EPI)

**Numeric Overflow**:
- Bounds checking prevents extremely large values
- Magnitude checks applied to all numeric parameters

## Testing

Comprehensive test suite in `tests/unit/validation/test_input_validation.py`:

- **73 tests** covering all validation functions
- **Security-specific tests** for injection prevention
- **Boundary tests** for all numeric limits
- **Type safety tests** for all parameters

Run tests:
```bash
pytest tests/unit/validation/test_input_validation.py -v
```

## Configuration

Validation bounds can be customized through graph configuration:

```python
import networkx as nx
from tnfr.structural import create_nfr

# Create graph with custom bounds
G = nx.Graph()
G.graph["EPI_MIN"] = 0.2
G.graph["EPI_MAX"] = 0.8
G.graph["VF_MIN"] = 0.5
G.graph["VF_MAX"] = 5.0
G.graph["DNFR_MAX"] = 0.5

# Create node with validation against custom bounds
G, node = create_nfr("test", epi=0.5, vf=2.0, graph=G)
```

## Best Practices

1. **Always validate external input**: Any parameters from user input, configuration files, or external APIs should be validated
2. **Use specific validators**: Use the most specific validator for each parameter type
3. **Handle ValidationError**: Catch and handle ValidationError exceptions appropriately
4. **Configure bounds**: Set appropriate bounds in graph configuration for your use case
5. **Test edge cases**: Test boundary conditions and malicious inputs

## Future Enhancements

Potential future improvements:

- Schema-based validation for complex parameter structures
- Validation hooks for custom parameter types
- Performance optimization for high-frequency validation
- Additional injection pattern detection
- Rate limiting for validation failures

## References

- Issue #2518: SECURITY: Input validation for structural operators
- TNFR Documentation: `tnfr.pdf`
- AGENTS.md: TNFR paradigm guide for agents
- SECURITY.md: General security guidelines

## Version History

- **v1.0** (2025-11-04): Initial implementation
  - Core validation functions for EPI, νf, θ, ΔNFR
  - Security validation for NodeId, Glyph, TNFRGraph
  - Integration into operator call sites
  - Comprehensive test suite (73 tests)
