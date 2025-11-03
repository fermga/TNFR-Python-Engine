# Factory Pattern Guidelines for TNFR

## Overview

This document defines canonical factory patterns for TNFR operators, generators, and mathematical constructs. Following these patterns ensures self-documenting, maintainable code that preserves TNFR structural semantics.

## Core Principles

1. **TNFR Fidelity First**: Factory functions must preserve structural invariants (coherence, phase, νf, ΔNFR)
2. **Explicit Validation**: All inputs validated before construction
3. **Self-Documenting**: Clear naming conventions that reveal intent
4. **Type Safety**: Full type annotations with corresponding .pyi stubs
5. **Reproducibility**: Support for deterministic construction (seeds, explicit parameters)

## Factory Naming Conventions

### Operator Factories

Use `make_*` prefix for creating structural operators:

```python
def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = 0.1,
) -> CoherenceOperator:
    """Return a Hermitian positive semidefinite CoherenceOperator.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the operator's Hilbert space.
    spectrum : np.ndarray | None, optional
        Eigenvalue spectrum. If None, uses uniform c_min.
    c_min : float, optional
        Minimum coherence threshold (default: 0.1).
        
    Returns
    -------
    CoherenceOperator
        Validated Hermitian PSD operator.
        
    Raises
    ------
    ValueError
        If dimension is invalid or spectrum violates Hermiticity/PSD constraints.
    """
```

**Key characteristics**:
- Prefix: `make_*`
- Returns concrete operator instances
- Validates structural properties (Hermiticity, positive semi-definiteness)
- Uses keyword-only arguments for options

### Generator Factories

Use `build_*` prefix for ΔNFR and other generators:

```python
def build_delta_nfr(
    dim: int,
    *,
    topology: str = "laplacian",
    nu_f: float = 1.0,
    scale: float = 1.0,
    rng: Generator | None = None,
) -> np.ndarray:
    """Construct a Hermitian ΔNFR generator using canonical TNFR topologies.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the Hilbert space.
    topology : str, optional
        Canonical topology: "laplacian" or "adjacency" (default: "laplacian").
    nu_f : float, optional
        Structural frequency scaling (default: 1.0).
    scale : float, optional
        Additional uniform scaling (default: 1.0).
    rng : Generator | None, optional
        NumPy RNG for reproducible noise injection.
        
    Returns
    -------
    np.ndarray
        Hermitian ΔNFR generator matrix.
        
    Raises
    ------
    ValueError
        If dimension is non-positive or topology is unknown.
    """
```

**Key characteristics**:
- Prefix: `build_*`
- Returns raw matrices or data structures
- Emphasizes reproducibility (explicit seeds/RNG)
- Includes structural scaling (νf, phase)

### Higher-Order Factories

Use `create_*` or specialized naming for factories that return other factories:

```python
def build_isometry_factory(
    *,
    source_dimension: int,
    target_dimension: int,
    allow_expansion: bool = False,
) -> IsometryFactory:
    """Create a factory for constructing TNFR-aligned isometries.
    
    Returns a callable that produces concrete isometries on demand.
    
    Parameters
    ----------
    source_dimension : int
        Input space dimensionality.
    target_dimension : int
        Output space dimensionality.
    allow_expansion : bool, optional
        Allow embedding into higher dimensional space (default: False).
        
    Returns
    -------
    IsometryFactory
        Callable that creates isometric transforms.
    """
```

## Standard Factory Structure

### 1. Validation Phase

All factories must validate inputs before construction:

```python
def _validate_dimension(dim: int) -> int:
    """Validate and normalize dimension parameter."""
    if int(dim) != dim:
        raise ValueError("Operator dimension must be an integer.")
    if dim <= 0:
        raise ValueError("Operator dimension must be strictly positive.")
    return int(dim)
```

### 2. Construction Phase

Build the object with validated inputs:

```python
# Get backend for array operations
backend = get_backend()

# Convert to backend-native arrays
array_backend = ensure_array(matrix, dtype=np.complex128, backend=backend)
```

### 3. Verification Phase

Verify structural invariants are preserved:

```python
if not operator.is_hermitian(atol=_ATOL):
    raise ValueError("Coherence operator must be Hermitian.")
if not operator.is_positive_semidefinite(atol=_ATOL):
    raise ValueError("Coherence operator must be positive semidefinite.")
```

## Backend Integration

Factories should be backend-agnostic:

```python
from ..backend import ensure_array, ensure_numpy, get_backend

def make_operator(matrix: np.ndarray) -> Operator:
    """Create operator with backend-agnostic array handling."""
    backend = get_backend()
    array_backend = ensure_array(matrix, dtype=np.complex128, backend=backend)
    return Operator(array_backend, backend=backend)
```

## Common Patterns

### Pattern 1: Spectrum-Based Operators

```python
def make_operator_from_spectrum(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray | None = None,
) -> Operator:
    """Construct operator from spectral decomposition."""
    # Validate eigenvalues
    # Construct or validate eigenvectors
    # Assemble operator
    # Verify properties
    return operator
```

### Pattern 2: Topology-Based Generators

```python
def build_generator(
    dim: int,
    *,
    topology: str = "laplacian",
    **params: Any,
) -> np.ndarray:
    """Construct generator from canonical topology."""
    # Validate topology choice
    # Build base structure
    # Apply TNFR scaling (νf, etc.)
    # Ensure Hermiticity
    return generator
```

### Pattern 3: Parameterized Factories

```python
def create_parameterized_factory(
    **config: Any,
) -> Callable[..., T]:
    """Return a factory configured with specific parameters."""
    def factory(*args: Any, **kwargs: Any) -> T:
        # Merge config and kwargs
        # Validate combined parameters
        # Construct object
        return obj
    return factory
```

## Type Annotations

All factories must have complete type annotations:

```python
from typing import Protocol, Callable, TypeVar

T = TypeVar("T")

class OperatorFactory(Protocol):
    """Protocol for operator factory functions."""
    
    def __call__(
        self,
        dim: int,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Operator:
        """Create an operator with specified parameters."""
        ...
```

## Testing Requirements

Each factory must have tests covering:

1. **Valid construction**: Happy path with default and custom parameters
2. **Input validation**: Invalid dimensions, incompatible parameters
3. **Structural invariants**: Hermiticity, PSD, norm preservation
4. **Reproducibility**: Same inputs → same outputs (with seeds)
5. **Backend compatibility**: Works with numpy, jax, torch

Example test structure:

```python
def test_make_operator_valid_construction():
    """Test factory creates valid operator."""
    op = make_operator(dim=4, c_min=0.1)
    assert op.dimension == 4
    assert op.is_hermitian()
    assert op.is_positive_semidefinite()

def test_make_operator_invalid_dimension():
    """Test factory rejects invalid dimensions."""
    with pytest.raises(ValueError, match="strictly positive"):
        make_operator(dim=-1)

def test_make_operator_reproducible():
    """Test factory is deterministic with seed."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    op1 = make_operator(dim=4, rng=rng1)
    op2 = make_operator(dim=4, rng=rng2)
    np.testing.assert_array_equal(op1.matrix, op2.matrix)
```

## Migration Checklist

When updating existing factory code:

- [ ] Rename function to follow `make_*`/`build_*` convention
- [ ] Add full type annotations
- [ ] Generate/update corresponding .pyi stub
- [ ] Add input validation with descriptive errors
- [ ] Verify structural invariants after construction
- [ ] Make options keyword-only (after required positional args)
- [ ] Add comprehensive docstring with Parameters/Returns/Raises
- [ ] Add tests for valid construction and edge cases
- [ ] Document backend requirements if any
- [ ] Update references in dependent code

## Examples

### Good Factory Implementation

```python
def make_coherence_operator(
    dim: int,
    *,
    spectrum: np.ndarray | None = None,
    c_min: float = 0.1,
) -> CoherenceOperator:
    """Return a Hermitian positive semidefinite CoherenceOperator.
    
    This factory validates inputs, ensures structural invariants, and
    integrates with the TNFR backend abstraction layer.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the operator's Hilbert space. Must be positive.
    spectrum : np.ndarray | None, optional
        Custom eigenvalue spectrum. If None, uses uniform c_min values.
        Must be real-valued and match dimension.
    c_min : float, optional
        Minimum coherence threshold for default spectrum (default: 0.1).
        
    Returns
    -------
    CoherenceOperator
        Validated coherence operator with backend-native arrays.
        
    Raises
    ------
    ValueError
        If dimension is invalid, spectrum has wrong shape, or operator
        violates Hermiticity/PSD constraints.
        
    Examples
    --------
    >>> op = make_coherence_operator(dim=4)
    >>> op.dimension
    4
    >>> op.is_hermitian()
    True
    """
    dimension = _validate_dimension(dim)
    
    if not np.isfinite(c_min):
        raise ValueError("Coherence threshold must be finite.")
    
    backend = get_backend()
    
    if spectrum is None:
        eigenvalues = ensure_array(
            np.full(dimension, float(c_min), dtype=float),
            backend=backend
        )
    else:
        # Validate spectrum shape and properties
        eigenvalues = _validate_spectrum(spectrum, dimension, backend)
    
    operator = CoherenceOperator(eigenvalues, c_min=c_min, backend=backend)
    
    # Verify structural invariants
    if not operator.is_hermitian(atol=_ATOL):
        raise ValueError("Coherence operator must be Hermitian.")
    if not operator.is_positive_semidefinite(atol=_ATOL):
        raise ValueError("Coherence operator must be positive semidefinite.")
    
    return operator
```

### Poor Factory (Anti-pattern)

```python
# DON'T DO THIS
def create_operator(d, s=None, min=0.1):  # Poor naming, missing types
    """Make operator."""  # Inadequate docstring
    # No validation
    eigenvals = s if s else [min] * d
    return Operator(eigenvals)  # No invariant verification
```

## Related Documentation

- See `AGENTS.md` for TNFR structural invariants
- See `docs/api/operators.md` for operator reference
- See `src/tnfr/mathematics/README.md` for mathematics module organization
