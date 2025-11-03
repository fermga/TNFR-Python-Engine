# TNFR Mathematics Module

## Overview

The `tnfr.mathematics` module provides the mathematical foundations for TNFR structural computations. It implements backend-agnostic numerical operations, operator factories, generator construction, and transform contracts.

## Module Organization

### Backend Abstraction (`backend.py`)

Provides a unified interface for numerical operations across NumPy, JAX, and PyTorch:

```python
from tnfr.mathematics import get_backend

backend = get_backend()  # Auto-selects based on config/environment
array = backend.as_array([1, 2, 3])
eigenvalues, eigenvectors = backend.eigh(matrix)
```

**Factory Pattern**: Uses registry pattern with `register_backend()` and private `_make_*_backend()` factories.

### Operator Factories (`operators_factory.py`)

Constructs validated TNFR operators with structural guarantees:

```python
from tnfr.mathematics import make_coherence_operator, make_frequency_operator

# Create coherence operator with uniform spectrum
coherence_op = make_coherence_operator(dim=4, c_min=0.1)

# Create frequency operator from matrix
freq_op = make_frequency_operator(hamiltonian_matrix)
```

**Factory Pattern**: Uses `make_*` prefix, validates Hermiticity and PSD properties.

### Generator Construction (`generators.py`)

Builds ΔNFR generators from canonical topologies:

```python
from tnfr.mathematics import build_delta_nfr, build_lindblad_delta_nfr
import numpy as np

# Build simple ΔNFR generator
rng = np.random.default_rng(42)
delta_nfr = build_delta_nfr(
    dim=10,
    topology="laplacian",
    nu_f=1.0,
    rng=rng
)

# Build Lindblad superoperator
lindblad = build_lindblad_delta_nfr(
    hamiltonian=H,
    collapse_operators=[L1, L2],
    nu_f=1.0
)
```

**Factory Pattern**: Uses `build_*` prefix, emphasizes reproducibility with explicit RNG.

### Transform Contracts (`transforms.py`)

Defines protocols for isometric transforms and coherence verification:

```python
from tnfr.mathematics import (
    build_isometry_factory,
    validate_norm_preservation,
    ensure_coherence_monotonicity
)

# Create factory for dimension-preserving isometries
isometry_factory = build_isometry_factory(
    source_dimension=4,
    target_dimension=4,
    allow_expansion=False
)
```

**Note**: Phase 2 implementation - currently provides contracts only.

## Factory Design Patterns

All factories in this module follow the patterns documented in [FACTORY_PATTERNS.md](../../../docs/FACTORY_PATTERNS.md):

1. **Clear naming**: `make_*` for operators, `build_*` for generators
2. **Input validation**: Dimension checks, spectrum validation, topology verification
3. **Structural verification**: Hermiticity, PSD, trace preservation
4. **Backend integration**: Works with numpy/jax/torch through `get_backend()`
5. **Type safety**: Full annotations with corresponding `.pyi` stubs

## Structural Invariants

These factories preserve TNFR canonical invariants:

- **Coherence operators**: Hermitian, positive semi-definite
- **Frequency operators**: Hermitian, PSD
- **ΔNFR generators**: Hermitian (or superoperator with appropriate spectrum)
- **Lindblad generators**: Trace-preserving, contractive semigroup

## Usage Examples

### Creating a Complete Operator Set

```python
from tnfr.mathematics import (
    make_coherence_operator,
    make_frequency_operator,
    build_delta_nfr
)
import numpy as np

# Set dimension
dim = 8

# Create coherence operator
C_op = make_coherence_operator(
    dim=dim,
    spectrum=np.linspace(0.1, 1.0, dim),
    c_min=0.1
)

# Create frequency operator
H = np.random.randn(dim, dim)
H = 0.5 * (H + H.T)  # Make Hermitian
F_op = make_frequency_operator(H)

# Build ΔNFR generator
rng = np.random.default_rng(42)
delta_nfr = build_delta_nfr(
    dim=dim,
    topology="laplacian",
    nu_f=1.0,
    scale=0.1,
    rng=rng
)
```

### Backend Selection

```python
from tnfr.mathematics import get_backend, ensure_array, ensure_numpy
import numpy as np

# Use NumPy backend (default)
backend_np = get_backend("numpy")

# Use JAX backend (if installed)
try:
    backend_jax = get_backend("jax")
    array_jax = ensure_array([1, 2, 3], backend=backend_jax)
    # ... perform operations ...
    result_np = ensure_numpy(array_jax, backend=backend_jax)
except Exception:
    print("JAX backend not available")
```

## Testing

All factories have comprehensive tests covering:

- Valid construction with default and custom parameters
- Input validation (invalid dimensions, incompatible parameters)
- Structural invariants (Hermiticity, PSD, trace preservation)
- Reproducibility (deterministic with seeds)
- Backend compatibility (numpy, jax, torch where applicable)

See `tests/mathematics/` for the complete test suite.

## Related Documentation

- [Factory Patterns Guide](../../../docs/FACTORY_PATTERNS.md) - Comprehensive factory design patterns
- [TNFR Paradigm](../../../TNFR.pdf) - Theoretical foundations
- [AGENTS.md](../../../AGENTS.md) - Structural invariants and contracts
- [API Overview](../../../docs/source/api/overview.md) - Package-level documentation
