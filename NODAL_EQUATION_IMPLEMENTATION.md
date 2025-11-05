# TNFR Canonical Nodal Equation: Theory-to-Code Mapping

## Executive Summary

This document establishes the **explicit correspondence** between the canonical TNFR nodal equation and its computational implementation in the Python engine. It addresses the critical issue raised in [Issue #XX]: ensuring theoretical fidelity and transparency in the codebase.

## Canonical TNFR Equation

The fundamental equation governing nodal evolution in TNFR is:

```
∂EPI/∂t = νf · ΔNFR(t)
```

**Where:**
- **EPI**: Primary Information Structure (coherent form)
- **νf**: Structural frequency in Hz_str (reorganization rate)
- **ΔNFR**: Nodal gradient (internal reorganization operator)
- **t**: Structural time (not chronological)

### Extended Form with Network Coupling

The full implementation includes optional network coupling:

```
∂EPI/∂t = νf · ΔNFR(t) + Γi(R)
```

Where **Γi(R)** represents Kuramoto-based network synchronization.

## Implementation Mapping

### Primary Implementation: `integrators.py`

The canonical equation is implemented **explicitly** in `src/tnfr/dynamics/integrators.py`:

#### Line 321 (NumPy path):
```python
vf = collect_attr(G, nodes, ALIAS_VF, 0.0, np=np)
dnfr = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np)
# CANONICAL TNFR EQUATION: ∂EPI/∂t = νf · ΔNFR(t)
base = vf * dnfr
```

#### Line 342 (Scalar path):
```python
vf, dnfr, *_ = _node_state(nd)
# CANONICAL TNFR EQUATION: ∂EPI/∂t = νf · ΔNFR(t)
base = vf * dnfr
```

**Correspondence:**
- `vf` → **νf** (structural frequency)
- `dnfr` → **ΔNFR** (nodal gradient)
- `base` → **∂EPI/∂t** (EPI derivative)

### Canonical Reference Implementation: `canonical.py`

For explicit validation and testing, the equation is also available as a standalone function in `src/tnfr/dynamics/canonical.py`:

```python
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative

result = compute_canonical_nodal_derivative(
    nu_f=1.5,      # Structural frequency (Hz_str)
    delta_nfr=0.4, # Nodal gradient
    validate_units=True
)

print(result.derivative)  # ∂EPI/∂t = 1.5 * 0.4 = 0.6
```

This function provides:
1. **Explicit equation visibility**
2. **Unit validation** (Hz_str)
3. **Clear theory-to-code traceability**
4. **Independent testability**

## Variable Correspondence Table

| Theory Symbol | Code Variable | Type | Units | Location |
|---------------|---------------|------|-------|----------|
| **νf** | `vf`, `nu_f` | `float` | Hz_str | Node attribute `VF` |
| **ΔNFR** | `dnfr`, `delta_nfr` | `float` | dimensionless | Node attribute `DNFR` |
| **EPI** | `epi` | `float` | dimensionless | Node attribute `EPI` |
| **∂EPI/∂t** | `base`, `dEPI_dt` | `float` | Hz_str | Computed derivative |
| **θ** | `theta`, `phase` | `float` | radians | Node attribute `THETA` |
| **Γi(R)** | `gamma` | `float` | Hz_str | Gamma module |

## Structural Units: Hz_str

TNFR uses **structural hertz (Hz_str)** to distinguish structural reorganization rates from classical temporal frequencies.

### Unit Conversion

The engine provides conversion utilities in `src/tnfr/units.py`:

```python
from tnfr.units import hz_str_to_hz, hz_to_hz_str

# Convert structural to classical frequency
classical_hz = hz_str_to_hz(nu_f_structural, G)

# Convert classical to structural frequency
structural_hz = hz_to_hz_str(classical_hz, G)
```

**Bridge Factor:**
- Configured via `G.graph['HZ_STR_BRIDGE']`
- Default: 1.0 (1 Hz_str = 1 Hz)
- Must be strictly positive

### Unit Validation

The canonical module validates structural units:

```python
from tnfr.dynamics.canonical import validate_structural_frequency

# Validates that νf is non-negative and finite
validated_vf = validate_structural_frequency(nu_f)
```

## Integration Methods

The nodal equation is integrated numerically using explicit methods:

### Euler Method

```python
# Single step integration
EPI_new = EPI_old + dt * (νf · ΔNFR + Γ)
```

**Implementation:** `_integrate_euler()` in `integrators.py`

### Runge-Kutta 4th Order (RK4)

```python
# Four-stage integration
k1 = νf · ΔNFR + Γ(t)
k2 = νf · ΔNFR + Γ(t + dt/2)
k3 = νf · ΔNFR + Γ(t + dt/2)
k4 = νf · ΔNFR + Γ(t + dt)
EPI_new = EPI_old + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
```

**Implementation:** `_integrate_rk4()` in `integrators.py`

## TNFR Invariants Preserved

The implementation maintains these theoretical invariants (from AGENTS.md §3):

### 1. EPI as Coherent Form
- EPI changes **only** via structural operators
- No ad-hoc mutations allowed
- ✅ **Enforced:** Integration respects operator closure

### 2. Structural Units
- νf expressed in **Hz_str**
- ✅ **Enforced:** Unit validation available in `canonical.py`

### 3. ΔNFR Semantics
- Sign and magnitude modulate reorganization rate
- NOT a classical "error gradient"
- ✅ **Enforced:** Documented and tested in `test_canonical.py`

### 4. Operator Closure
- Composition yields valid TNFR states
- ✅ **Enforced:** Tests verify closure property

## Time Interpretation

### Structural Time vs Chronological Time

TNFR distinguishes two time concepts:

1. **Chronological time (dt)**: Integration step size
2. **Structural time**: Accumulator of reorganization events

**Implementation:**
```python
# Chronological step
dt = 0.1  # seconds or arbitrary units

# Structural time accumulation
G.graph['_t'] += dt  # Structural time advances

# Equation uses dt to integrate structural evolution
dEPI = νf · ΔNFR  # Structural rate
EPI_new = EPI_old + dt * dEPI
```

## Usage Examples

### Basic Nodal Evolution

```python
import networkx as nx
from tnfr.structural import create_nfr
from tnfr.dynamics import update_epi_via_nodal_equation

# Create node with TNFR attributes
G, node = create_nfr("test", epi=1.0, vf=1.5, theta=0.0)

# Set nodal gradient
G.nodes[node]['DNFR'] = 0.4

# Integrate canonical equation: ∂EPI/∂t = νf · ΔNFR
update_epi_via_nodal_equation(G, dt=0.1)

# EPI evolves according to canonical equation
print(G.nodes[node]['EPI'])  # Updated value
```

### Explicit Canonical Computation

```python
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative

# Direct computation of ∂EPI/∂t
result = compute_canonical_nodal_derivative(
    nu_f=1.5,      # Hz_str
    delta_nfr=0.4, # dimensionless
    validate_units=True
)

print(f"∂EPI/∂t = {result.derivative}")  # 0.6
print(f"Validated: {result.validated}")  # True
```

### With Unit Validation

```python
from tnfr.dynamics.canonical import (
    validate_structural_frequency,
    validate_nodal_gradient
)

# Validate before computation
try:
    vf = validate_structural_frequency(1.5)
    dnfr = validate_nodal_gradient(0.4)
    derivative = vf * dnfr
except ValueError as e:
    print(f"Validation failed: {e}")
```

## Testing

### Test Coverage

Comprehensive tests validate the canonical equation:

**Location:** `tests/unit/dynamics/test_canonical.py`

**Coverage:**
- ✅ Basic equation computation
- ✅ Unit validation (Hz_str)
- ✅ Edge cases (zero values, negative ΔNFR)
- ✅ Error handling (invalid inputs)
- ✅ TNFR invariant preservation
- ✅ Integration with existing code

### Running Tests

```bash
# Test canonical equation implementation
pytest tests/unit/dynamics/test_canonical.py -v

# Test integration with existing code
pytest tests/unit/dynamics/test_integrators.py -v
```

## API Reference

### `compute_canonical_nodal_derivative(nu_f, delta_nfr, *, validate_units=True, graph=None)`

Compute ∂EPI/∂t using canonical TNFR equation.

**Parameters:**
- `nu_f` (float): Structural frequency in Hz_str
- `delta_nfr` (float): Nodal gradient
- `validate_units` (bool): Enable unit validation
- `graph` (GraphLike | None): Optional graph for context

**Returns:**
- `NodalEquationResult`: Named tuple with derivative and metadata

**Raises:**
- `ValueError`: If validation fails
- `TypeError`: If inputs are non-numeric

### `validate_structural_frequency(nu_f, *, graph=None)`

Validate structural frequency constraints.

**Validates:**
- Non-negative (νf ≥ 0)
- Finite (no NaN or infinity)
- Numeric type

### `validate_nodal_gradient(delta_nfr, *, graph=None)`

Validate nodal gradient well-definedness.

**Validates:**
- Finite (no NaN or infinity)
- Numeric type
- Sign indicates direction (expansion/contraction)

## Dimensional Analysis

### Equation Units

```
[∂EPI/∂t] = [νf] × [ΔNFR]
[Hz_str] = [Hz_str] × [1]
```

**Verification:**
- νf: Hz_str (structural reorganization rate)
- ΔNFR: dimensionless (operator magnitude)
- ∂EPI/∂t: Hz_str (coherence evolution rate)

### Unit Consistency

The implementation maintains dimensional consistency:

```python
# All have units of Hz_str
base = vf * dnfr           # νf·ΔNFR term
gamma = eval_gamma(G, n)   # Γi(R) term
total = base + gamma       # Full derivative
```

## References

### Theory Documents
- **TNFR.pdf**: Canonical equation specification (Section 3.2)
- **AGENTS.md**: Invariants and operational principles (Section 3)
- **GLOSSARY.md**: Variable definitions and units

### Code Modules
- `src/tnfr/dynamics/canonical.py`: Explicit equation implementation
- `src/tnfr/dynamics/integrators.py`: Numerical integration
- `src/tnfr/units.py`: Hz_str ↔ Hz conversion
- `tests/unit/dynamics/test_canonical.py`: Validation tests

### Related Issues
- [Issue #XX]: Implementación: Discrepancia entre Teoría TNFR y Código - Ecuación Nodal

## Validation Checklist

- [x] Canonical equation implemented explicitly
- [x] Variables map clearly to theory
- [x] Units validated (Hz_str)
- [x] TNFR invariants preserved
- [x] Comprehensive test coverage
- [x] Documentation updated
- [x] Examples demonstrate usage

## Conclusion

The TNFR canonical nodal equation **is implemented explicitly** in the codebase at multiple levels:

1. **Production code**: `integrators.py` lines 321 and 342
2. **Reference implementation**: `canonical.py` module
3. **Validation**: `test_canonical.py` test suite
4. **Documentation**: This mapping document

The implementation maintains **full theoretical fidelity** to the TNFR paradigm while providing transparency and traceability from theory to code.
