# Factory Pattern Quick Reference

> **Quick guide for creating and using TNFR factory functions**
>
> For comprehensive details, see [FACTORY_PATTERNS.md](FACTORY_PATTERNS.md)

---

## Naming Conventions

| Pattern | Prefix | Purpose | Returns |
|---------|--------|---------|---------|
| **Operator Factories** | `make_*` | Create validated operator instances | Concrete objects (CoherenceOperator, FrequencyOperator) |
| **Generator Factories** | `build_*` | Construct ΔNFR generators & data structures | Raw matrices (np.ndarray) or data structures |
| **Node Factories** | `create_*` | Create TNFR nodes & higher-order factories | TNFRGraph nodes or factory functions |

---

## Quick Start Templates

### Operator Factory Template

```python
def make_operator_name(
    dim: int,
    *,
    param1: Type1 = default1,
    param2: Type2 = default2,
) -> OperatorType:
    """Return a validated operator with structural guarantees.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the operator's Hilbert space.
    param1 : Type1, optional
        Description (default: default1).
    param2 : Type2, optional
        Description (default: default2).
        
    Returns
    -------
    OperatorType
        Validated operator instance.
        
    Raises
    ------
    ValueError
        If validation fails or structural invariants violated.
    """
    # 1. Validate inputs
    dimension = _validate_dimension(dim)
    
    # 2. Get backend
    backend = get_backend()
    
    # 3. Construct operator
    operator = OperatorType(data, backend=backend)
    
    # 4. Verify structural invariants
    if not operator.is_hermitian(atol=1e-9):
        raise ValueError("Operator must be Hermitian.")
    
    return operator
```

### Generator Factory Template

```python
def build_generator_name(
    dim: int,
    *,
    nu_f: float = 1.0,
    scale: float = 1.0,
    rng: Generator | None = None,
) -> np.ndarray:
    """Construct a Hermitian generator for ΔNFR evolution.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the Hilbert space.
    nu_f : float, optional
        Structural frequency scaling (default: 1.0).
    scale : float, optional
        Additional uniform scaling (default: 1.0).
    rng : Generator | None, optional
        NumPy RNG for reproducible noise.
        
    Returns
    -------
    np.ndarray
        Hermitian generator matrix.
        
    Raises
    ------
    ValueError
        If dimension is invalid.
    """
    # 1. Validate inputs
    if dim <= 0:
        raise ValueError("Dimension must be positive.")
    
    # 2. Build base structure
    matrix = _build_base_topology(dim)
    
    # 3. Apply TNFR scaling
    matrix *= (nu_f * scale)
    
    # 4. Ensure Hermiticity
    hermitian = 0.5 * (matrix + matrix.conj().T)
    
    # 5. Backend conversion
    backend = get_backend()
    return ensure_numpy(ensure_array(hermitian, backend=backend), backend=backend)
```

---

## Validation Checklist

When creating a new factory, ensure:

- [ ] **Naming**: Uses correct prefix (`make_*`, `build_*`, `create_*`)
- [ ] **Type annotations**: Full signatures with types
- [ ] **Docstring**: NumPy-style with Parameters/Returns/Raises
- [ ] **Input validation**: Check dimensions, parameters before construction
- [ ] **Keyword-only args**: Optional params use `*, param: type = default`
- [ ] **Structural verification**: Hermiticity, PSD, trace preservation
- [ ] **Backend integration**: Uses `get_backend()`, `ensure_array()`
- [ ] **Stub file**: Corresponding `.pyi` exists and synchronized
- [ ] **Tests**: Valid construction, validation, invariants, reproducibility

---

## Common Patterns

### Pattern 1: Spectrum-Based Operators

```python
# Create operator from eigenvalue spectrum
operator = make_coherence_operator(
    dim=4,
    spectrum=np.array([0.1, 0.3, 0.5, 0.7]),
    c_min=0.1
)
```

### Pattern 2: Topology-Based Generators

```python
# Build generator from canonical topology
delta_nfr = build_delta_nfr(
    dim=10,
    topology="laplacian",  # or "adjacency"
    nu_f=1.0,
    scale=0.1,
    rng=np.random.default_rng(42)
)
```

### Pattern 3: Lindblad Superoperators

```python
# Construct trace-preserving Lindblad generator
H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
L = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)

lindblad = build_lindblad_delta_nfr(
    hamiltonian=H,
    collapse_operators=[L],
    nu_f=1.0,
    ensure_trace_preserving=True,
    ensure_contractive=True
)
```

---

## Validation Utilities

### Dimension Validation

```python
def _validate_dimension(dim: int) -> int:
    """Validate and normalize dimension parameter.
    
    Note: The int(dim) != dim check catches float values like 4.5 that aren't
    whole numbers, while allowing 4.0 (which equals 4 after int conversion).
    This ensures dimension parameters are mathematically integer-valued.
    """
    if int(dim) != dim:
        raise ValueError("Dimension must be an integer.")
    if dim <= 0:
        raise ValueError("Dimension must be strictly positive.")
    return int(dim)
```

### Hermiticity Check

```python
def _check_hermitian(matrix: np.ndarray, atol: float = 1e-9) -> None:
    """Validate matrix is Hermitian within tolerance."""
    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        raise ValueError("Matrix must be Hermitian within tolerance.")
```

### PSD Verification

```python
def _check_psd(matrix: np.ndarray, atol: float = 1e-9) -> None:
    """Validate matrix is positive semidefinite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.any(eigenvalues < -atol):
        raise ValueError("Matrix must be positive semidefinite.")
```

---

## Testing Template

```python
class TestMyFactory:
    """Tests for my_factory following TNFR factory patterns."""
    
    def test_valid_construction(self):
        """Factory creates valid objects with default parameters."""
        obj = my_factory(dim=4)
        assert obj.dimension == 4
    
    def test_invalid_dimension(self):
        """Factory rejects invalid dimensions."""
        with pytest.raises(ValueError, match="positive"):
            my_factory(dim=0)
        with pytest.raises(ValueError, match="positive"):
            my_factory(dim=-1)
    
    def test_structural_invariants(self):
        """Factory preserves TNFR structural invariants."""
        obj = my_factory(dim=4)
        assert obj.is_hermitian(atol=1e-9)
        assert obj.is_positive_semidefinite(atol=1e-9)
    
    def test_reproducibility(self):
        """Factory is deterministic with seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        obj1 = my_factory(dim=4, rng=rng1)
        obj2 = my_factory(dim=4, rng=rng2)
        np.testing.assert_array_equal(obj1.data, obj2.data)
```

---

## Stub File Generation

### Generate Missing Stubs

```bash
# Generate all missing .pyi stub files
make stubs

# Or use script directly
python scripts/generate_stubs.py
```

### Check Synchronization

```bash
# Check if any stubs are missing
make stubs-check

# Check if any stubs are outdated
make stubs-check-sync

# Regenerate outdated stubs
make stubs-sync
```

### CI/CD Integration

Stub checks run automatically in CI:
- Pre-commit hook prevents commits with missing stubs
- Type-check workflow validates stub existence and synchronization
- Failed checks block PR merges

---

## Backend Integration

### Using Backend Abstraction

```python
from tnfr.mathematics.backend import get_backend, ensure_array, ensure_numpy

# Get configured backend
backend = get_backend()

# Convert to backend-native array
array_backend = ensure_array(data, dtype=np.complex128, backend=backend)

# Perform backend operations
eigenvalues, eigenvectors = backend.eigh(array_backend)

# Convert back to numpy for return
result = ensure_numpy(array_backend, backend=backend)
```

### Backend-Agnostic Patterns

```python
def my_factory(matrix: np.ndarray) -> Operator:
    """Create operator supporting all backends."""
    backend = get_backend()
    
    # Convert input to backend
    matrix_backend = ensure_array(matrix, backend=backend)
    
    # Use backend operations
    eigenvalues_backend, _ = backend.eigh(matrix_backend)
    
    # Create operator with backend support
    operator = Operator(matrix_backend, backend=backend)
    
    return operator
```

---

## Common Errors & Solutions

### Error: Missing Stub File

```bash
Error: Python file without corresponding .pyi stub
```

**Solution:**
```bash
make stubs
```

### Error: Outdated Stub

```bash
Error: .pyi stub file older than .py implementation
```

**Solution:**
```bash
make stubs-sync
```

### Error: Non-Hermitian Operator

```python
ValueError: Operator must be Hermitian within tolerance
```

**Solution:** Ensure matrix is symmetric (or Hermitian for complex):
```python
matrix_hermitian = 0.5 * (matrix + matrix.conj().T)
```

### Error: Invalid Dimension

```python
ValueError: Dimension must be strictly positive
```

**Solution:** Validate dimension before use:
```python
if dim <= 0:
    raise ValueError("Dimension must be strictly positive.")
```

---

## Complete Example

```python
"""Example: Creating a custom operator factory."""

from __future__ import annotations

import numpy as np
from tnfr.mathematics.backend import get_backend, ensure_array, ensure_numpy
from tnfr.mathematics.operators import CustomOperator

__all__ = ["make_custom_operator"]

_ATOL = 1e-9


def _validate_dimension(dim: int) -> int:
    """Validate dimension parameter."""
    if int(dim) != dim:
        raise ValueError("Dimension must be an integer.")
    if dim <= 0:
        raise ValueError("Dimension must be strictly positive.")
    return int(dim)


def make_custom_operator(
    dim: int,
    *,
    scale: float = 1.0,
    ensure_normalized: bool = True,
) -> CustomOperator:
    """Create a validated custom operator.
    
    Parameters
    ----------
    dim : int
        Operator dimension.
    scale : float, optional
        Scaling factor (default: 1.0).
    ensure_normalized : bool, optional
        Ensure trace equals dimension (default: True).
        
    Returns
    -------
    CustomOperator
        Validated operator instance.
        
    Raises
    ------
    ValueError
        If dimension invalid or structural invariants violated.
        
    Examples
    --------
    >>> op = make_custom_operator(dim=4, scale=2.0)
    >>> op.dimension
    4
    >>> op.is_hermitian()
    True
    """
    # 1. Validate
    dimension = _validate_dimension(dim)
    if not np.isfinite(scale):
        raise ValueError("Scale must be finite.")
    
    # 2. Get backend
    backend = get_backend()
    
    # 3. Construct
    identity = np.eye(dimension, dtype=np.complex128)
    matrix = scale * identity
    matrix_backend = ensure_array(matrix, backend=backend)
    
    # 4. Create operator
    operator = CustomOperator(matrix_backend, backend=backend)
    
    # 5. Verify invariants
    if not operator.is_hermitian(atol=_ATOL):
        raise ValueError("Operator must be Hermitian.")
    
    if ensure_normalized:
        trace = ensure_numpy(backend.trace(matrix_backend), backend=backend)
        expected_trace = scale * dimension
        if not np.isclose(trace, expected_trace, atol=_ATOL):
            raise ValueError(f"Trace must equal {expected_trace}.")
    
    return operator
```

---

## Additional Resources

- **Comprehensive Guide**: [FACTORY_PATTERNS.md](FACTORY_PATTERNS.md)
- **Complete Inventory**: [FACTORY_INVENTORY_2025.md](FACTORY_INVENTORY_2025.md)
- **Audit Report**: [FACTORY_AUDIT_2025.md](FACTORY_AUDIT_2025.md)
- **Implementation Summary**: [FACTORY_HOMOGENIZATION_SUMMARY.md](FACTORY_HOMOGENIZATION_SUMMARY.md)
- **Mathematics Module**: [src/tnfr/mathematics/README.md](../src/tnfr/mathematics/README.md)
- **TNFR Paradigm**: [TNFR.pdf](../TNFR.pdf)
- **Agent Guidelines**: [AGENTS.md](../AGENTS.md)
