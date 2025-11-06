# TNFR Architecture Guide: Patterns and Dependencies

> **Comprehensive guide to TNFR factory patterns, type safety, and module dependencies**

This guide consolidates architectural knowledge about the TNFR Python Engine's structural organization, factory patterns, type system, and dependency management.

---

## Table of Contents

1. [Factory Patterns](#factory-patterns)
   - [Core Principles](#core-principles)
   - [Naming Conventions](#naming-conventions)
   - [Factory Templates](#factory-templates)
   - [Validation and Testing](#validation-and-testing)
2. [Type Stub Automation](#type-stub-automation)
   - [Workflow](#type-stub-workflow)
   - [Commands](#stub-commands)
   - [Troubleshooting](#stub-troubleshooting)
3. [Module Dependencies](#module-dependencies)
   - [Dependency Hierarchy](#dependency-hierarchy)
   - [API Contracts](#api-contracts)
   - [Coupling Analysis](#coupling-analysis)
4. [System Invariants](#system-invariants)
5. [Quick References](#quick-references)

---

## Factory Patterns

### Core Principles

Factory functions in TNFR follow strict principles to preserve structural coherence:

1. **TNFR Fidelity First**: Factory functions must preserve structural invariants (coherence, phase, νf, ΔNFR)
2. **Explicit Validation**: All inputs validated before construction
3. **Self-Documenting**: Clear naming conventions that reveal intent
4. **Type Safety**: Full type annotations with corresponding `.pyi` stubs
5. **Reproducibility**: Support for deterministic construction (seeds, explicit parameters)

### Naming Conventions

TNFR uses three distinct factory patterns based on what they return:

| Prefix | Returns | Purpose | Example |
|--------|---------|---------|---------|
| `make_*` | **Objects** | Create validated operator instances with structural guarantees | `make_coherence_operator` |
| `build_*` | **Arrays/Data** | Construct ΔNFR generators and raw mathematical structures | `build_delta_nfr` |
| `create_*` | **Nodes/Factories** | Create TNFR nodes or return other factory functions | `create_nfr` |

**Rationale**:
- `make_*`: Emphasizes creation of validated, ready-to-use objects
- `build_*`: Suggests construction of raw, flexible data structures
- `create_*`: Reserved for higher-order abstractions (nodes, factory generators)

### Factory Templates

#### Operator Factory Template (`make_*`)

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
    
    Examples
    --------
    >>> op = make_operator_name(dim=5, param1=custom_value)
    >>> assert op.is_hermitian()
    """
    # 1. Validate inputs
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    
    # 2. Get backend for array operations
    backend = get_backend()
    
    # 3. Construct operator
    data = backend.zeros((dim, dim), dtype=backend.complex128)
    # ... construction logic ...
    operator = OperatorType(data, backend=backend)
    
    # 4. Verify structural invariants
    if not operator.is_hermitian(atol=1e-9):
        raise ValueError("Operator must be Hermitian")
    
    if not operator.is_positive_semidefinite():
        raise ValueError("Operator must be positive semidefinite")
    
    return operator
```

**Key characteristics**:
- Keyword-only arguments (after `*`) for options
- Comprehensive input validation with descriptive errors
- Backend integration for math operations
- Post-construction verification of structural properties
- Complete NumPy-style docstring

#### Generator Factory Template (`build_*`)

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
        Structural frequency scaling (Hz_str) (default: 1.0).
    scale : float, optional
        Additional uniform scaling factor (default: 1.0).
    rng : Generator | None, optional
        NumPy RNG for reproducible noise injection.
        If None, generates deterministic structure only.
        
    Returns
    -------
    np.ndarray
        Hermitian generator matrix of shape (dim, dim).
        
    Raises
    ------
    ValueError
        If dimension is invalid or parameters out of range.
        
    Notes
    -----
    The νf parameter represents structural frequency in Hz_str units,
    modulating the rate of reorganization as per the nodal equation:
    
    .. math::
        \\frac{∂EPI}{∂t} = νf · ΔNFR(t)
    
    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> gen = build_generator_name(dim=5, nu_f=2.0, rng=rng)
    >>> assert gen.shape == (5, 5)
    >>> assert np.allclose(gen, gen.conj().T)  # Hermitian
    """
    # 1. Validate inputs
    if dim <= 0:
        raise ValueError(f"Dimension must be positive, got {dim}")
    if nu_f < 0:
        raise ValueError(f"Structural frequency must be non-negative, got {nu_f}")
    
    # 2. Get backend
    backend = get_backend()
    
    # 3. Build base structure
    matrix = backend.zeros((dim, dim), dtype=backend.complex128)
    # ... construction logic with nu_f scaling ...
    
    # 4. Add reproducible noise if requested
    if rng is not None:
        noise = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        noise = (noise + noise.conj().T) / 2  # Hermitianize
        matrix = matrix + 0.1 * scale * noise
    
    # 5. Ensure Hermiticity
    matrix = (matrix + matrix.conj().T) / 2
    
    return matrix
```

**Key characteristics**:
- Returns raw numpy array (not wrapped object)
- Includes `nu_f` parameter for structural frequency scaling
- Supports reproducible noise via explicit RNG
- Ensures Hermiticity before returning
- Documents structural semantics in Notes section

#### Node Factory Template (`create_*`)

```python
def create_nfr(
    G: TNFRGraph,
    *,
    nu_f: float = 1.0,
    phase: float = 0.0,
    epi_dim: int = 2,
    seed: int | None = None,
) -> tuple[TNFRGraph, str]:
    """Create a new resonant fractal node (NFR) in the network.
    
    Parameters
    ----------
    G : TNFRGraph
        Target network graph.
    nu_f : float, optional
        Initial structural frequency in Hz_str (default: 1.0).
    phase : float, optional
        Initial phase in radians, range [0, 2π] (default: 0.0).
    epi_dim : int, optional
        EPI dimensionality (default: 2).
    seed : int | None, optional
        Random seed for deterministic node creation.
        
    Returns
    -------
    G : TNFRGraph
        Updated graph with new node.
    node_id : str
        Identifier of created node.
        
    Raises
    ------
    ValueError
        If parameters violate structural constraints.
        
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G, node = create_nfr(G, nu_f=2.0, phase=0.5, seed=42)
    >>> assert node in G.nodes
    >>> assert G.nodes[node]['nu_f'] == 2.0
    """
    # 1. Validate inputs
    if nu_f < 0:
        raise ValueError(f"Structural frequency must be non-negative, got {nu_f}")
    if not 0 <= phase <= 2 * np.pi:
        raise ValueError(f"Phase must be in [0, 2π], got {phase}")
    if epi_dim <= 0:
        raise ValueError(f"EPI dimension must be positive, got {epi_dim}")
    
    # 2. Generate deterministic node ID
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    node_id = f"NFR_{rng.integers(0, 1000000):06d}"
    
    # 3. Initialize EPI structure
    epi = build_delta_nfr(dim=epi_dim, nu_f=nu_f, rng=rng)
    
    # 4. Add node to graph
    G.add_node(
        node_id,
        nu_f=nu_f,
        phase=phase,
        epi=epi,
        coherence=1.0,  # Initial perfect coherence
        dnfr=0.0,       # No reorganization pressure initially
    )
    
    return G, node_id
```

**Key characteristics**:
- Returns composite structures (graph + identifier)
- Creates complete TNFR nodes with all required attributes
- Uses other factories internally (`build_delta_nfr`)
- Comprehensive examples showing typical usage
- May call other factory functions

### Validation and Testing

#### Testing Template for Factory Functions

Every factory function should have tests covering:

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
    
    def test_valid_construction_custom_spectrum(self):
        """Test construction with custom eigenvalue spectrum."""
        spectrum = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        op = make_coherence_operator(dim=5, spectrum=spectrum)
        assert np.allclose(np.sort(np.linalg.eigvals(op)), spectrum, atol=1e-9)
    
    def test_invalid_dimension(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            make_coherence_operator(dim=0)
        with pytest.raises(ValueError, match="Dimension must be positive"):
            make_coherence_operator(dim=-5)
    
    def test_invalid_spectrum_shape(self):
        """Test that mismatched spectrum shape raises ValueError."""
        with pytest.raises(ValueError, match="shape"):
            make_coherence_operator(dim=5, spectrum=np.array([0.1, 0.2]))  # Wrong size
    
    def test_structural_invariant_hermiticity(self):
        """Test that Hermiticity is guaranteed."""
        op = make_coherence_operator(dim=10)
        matrix = op.to_array()
        assert np.allclose(matrix, matrix.conj().T, atol=1e-9)
    
    def test_structural_invariant_psd(self):
        """Test that positive semidefiniteness is guaranteed."""
        op = make_coherence_operator(dim=10)
        eigenvalues = np.linalg.eigvalsh(op.to_array())
        assert np.all(eigenvalues >= -1e-9)  # Allow numerical tolerance
    
    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        op1 = make_coherence_operator(dim=5, spectrum=np.linspace(0.1, 0.5, 5))
        op2 = make_coherence_operator(dim=5, spectrum=np.linspace(0.1, 0.5, 5))
        assert np.allclose(op1.to_array(), op2.to_array())
    
    def test_backend_compatibility(self):
        """Test that factory works with different backends."""
        # This test would cycle through available backends
        op = make_coherence_operator(dim=5)
        assert hasattr(op, 'backend')
        assert op.backend is not None
```

**Test coverage checklist**:
- ✓ Valid construction with defaults
- ✓ Valid construction with custom parameters
- ✓ Invalid dimension handling
- ✓ Invalid parameter handling
- ✓ Structural invariants (Hermiticity, PSD, norm preservation)
- ✓ Reproducibility (deterministic output)
- ✓ Backend compatibility
- ✓ Edge cases (boundary values)

---

## Type Stub Automation

### Type Stub Workflow

TNFR uses automated `.pyi` stub generation to maintain type safety and prevent drift between implementations and type hints.

**Why type stubs?**
- Separate type information from implementation
- Faster type checking (no need to parse implementations)
- Explicit API contracts
- IDE autocomplete support

**Automation layers**:
```
Pre-commit Hook → CI Check (missing) → CI Check (sync) → Mypy Validation
     (local)           (GitHub)             (GitHub)         (local/CI)
```

### Stub Commands

```bash
# Display all available commands
make help

# Generate missing stub files
make stubs

# Check for missing stubs (exit code 1 if any missing)
make stubs-check

# Check if stubs are synchronized (exit code 1 if outdated)
make stubs-check-sync

# Regenerate outdated stub files
make stubs-sync
```

### Stub Workflow Scenarios

#### Scenario 1: Creating a New Module

```bash
# 1. Create your Python module
touch src/tnfr/new_module.py

# 2. Implement your functions with type hints
# src/tnfr/new_module.py
def my_function(x: int, y: float) -> str:
    return f"{x} + {y}"

# 3. Generate stub file
make stubs
# Creates: src/tnfr/new_module.pyi

# 4. Verify stub was created
make stubs-check
# Output: All modules have stub files ✓

# 5. Commit both files together
git add src/tnfr/new_module.py src/tnfr/new_module.pyi
git commit -m "Add new_module with type stubs"
```

#### Scenario 2: Modifying an Existing Module

```bash
# 1. Make changes to implementation
# Edit: src/tnfr/existing_module.py

# 2. Check if stub needs update
make stubs-check-sync
# Output: Stub file outdated: src/tnfr/existing_module.pyi

# 3. Regenerate stub
make stubs-sync
# Updates: src/tnfr/existing_module.pyi

# 4. Review changes
git diff src/tnfr/existing_module.pyi

# 5. Commit both files together
git add src/tnfr/existing_module.py src/tnfr/existing_module.pyi
git commit -m "Update existing_module and regenerate stub"
```

#### Scenario 3: Pre-commit Hook Catches Missing Stub

```bash
# 1. Create new module
touch src/tnfr/another_module.py
# ... add code ...

# 2. Try to commit without generating stub
git add src/tnfr/another_module.py
git commit -m "Add another_module"

# Pre-commit hook runs and fails:
# ERROR: Missing stub files detected:
#   - src/tnfr/another_module.pyi
# Run 'make stubs' to generate them

# 3. Generate stub as instructed
make stubs

# 4. Add stub and commit
git add src/tnfr/another_module.pyi
git commit -m "Add another_module with type stub"
```

### Stub Troubleshooting

#### Issue: Stub generation fails

**Symptoms**: `make stubs` returns errors

**Solutions**:
1. Ensure mypy is installed: `pip install -e .[typecheck]`
2. Check for syntax errors in `.py` files
3. Run `python scripts/generate_stubs.py --dry-run` to see what would be generated
4. Check script output for specific module errors

#### Issue: Stub file appears outdated

**Symptoms**: `make stubs-check-sync` fails but `make stubs-sync` doesn't update

**Solutions**:
1. Check if `.py` file was actually modified
2. Manually inspect `.pyi` file for correctness
3. Delete `.pyi` file and regenerate: `rm src/tnfr/module.pyi && make stubs`
4. Verify file permissions allow writing

#### Issue: CI fails on stub check

**Symptoms**: GitHub Actions fails with "stub files outdated"

**Solutions**:
1. Run `make stubs-check-sync` locally before pushing
2. Run `make stubs-sync` to regenerate all outdated stubs
3. Commit and push both `.py` and `.pyi` files together
4. If persistent, check that CI uses same mypy version as local

---

## Module Dependencies

### Dependency Hierarchy

The TNFR codebase follows a clean layered architecture to prevent circular dependencies:

```
Layer 0 (Types & Constants):
  - types.py: Core type definitions (TNFRGraph, GraphLike, EPI)
  - constants/: Default parameters, limits, structural constants

Layer 1 (Foundation):
  - utils/init.py: Logging, lazy imports, backend loading
  - utils/numeric.py: Pure mathematical functions (clamping, angles)
  - utils/chunks.py: Parallelism utilities

Layer 2 (Data Operations):
  - utils/data.py: Type conversion, normalization
    Depends on: numeric (L1), init (L1)

Layer 3 (Graph & State):
  - utils/graph.py: Graph metadata, ΔNFR state management
    Depends on: types (L0)

Layer 4 (I/O):
  - utils/io.py: JSON/YAML parsing, file operations
    Depends on: init (L1)

Layer 5 (Caching):
  - utils/cache.py: Cache infrastructure, versioning
    Depends on: graph (L3), init (L1), io (L4)

Layer 6 (Callbacks):
  - utils/callbacks.py: Event system for observations
    Depends on: init (L1), data (L2)

Layer 7 (High-Level Modules):
  - structural.py: NFR creation (create_nfr)
  - operators/: Structural operators (Emission, Reception, etc.)
  - mathematics/: Factories for operators and generators
    Depends on: All lower layers as needed
```

**Dependency rules**:
1. **Downward only**: Modules only import from lower layers
2. **No cycles**: Circular imports are forbidden and tested in CI
3. **Minimal coupling**: Each layer has minimal dependencies on others
4. **Clear boundaries**: Layer transitions are explicit and documented

### API Contracts

#### Contract: Graph State Management (`utils/graph.py`)

**Functions**: `get_graph()`, `mark_dnfr_prep_dirty()`, `node_set_checksum()`

**Structural Operators Supported**: All operators that modify graph state

**Invariants**:
- `get_graph()` never modifies the graph
- `mark_dnfr_prep_dirty()` only sets a flag, doesn't compute ΔNFR
- Checksums are deterministic for same node set

**Pre-conditions**:
- Graph must be a valid NetworkX graph or TNFRGraph
- Node attributes must exist if accessed

**Post-conditions**:
- Graph structure unchanged (for read operations)
- Dirty flags correctly set (for write operations)
- Checksums stable across Python sessions

#### Contract: Cache Management (`utils/cache.py`)

**Classes**: `CacheManager`, `cached_node_list()`, `edge_version_cache()`

**Structural Operators Supported**: All operators (caching is transparent)

**Invariants**:
- Cache hits are bitwise identical to cache misses
- Cache invalidation maintains structural consistency
- Maximum cache sizes are respected

**Pre-conditions**:
- Cache keys are hashable
- Cached objects are pickleable (if using disk cache)

**Post-conditions**:
- Cache state consistent with graph version
- Memory usage bounded by configured limits
- No stale data returned

#### Contract: Validation (`validation/`)

**Functions**: `validate_frequency()`, `validate_phase()`, `validate_coherence()`

**Structural Operators Supported**: All operators that modify node attributes

**Invariants**:
- νf ≥ 0 (structural frequency non-negative)
- 0 ≤ phase < 2π (phase in valid range)
- 0 ≤ C(t) ≤ 1 (coherence normalized)
- ΔNFR is Hermitian (if matrix)

**Pre-conditions**:
- Input values have correct types
- Graph exists and is valid

**Post-conditions**:
- ValueError raised if validation fails
- No modifications to inputs
- Error messages clearly describe violation

### Coupling Analysis

#### Low Coupling (Desirable)

**Example**: `utils/numeric.py` → No internal TNFR dependencies
- Pure mathematical functions
- Zero coupling to graph or node state
- Easily testable and reusable

**Example**: `utils/chunks.py` → Depends only on standard library
- Computes chunk sizes for parallelism
- No TNFR-specific dependencies
- Could be extracted as standalone module

#### Moderate Coupling (Justified)

**Example**: `utils/data.py` → Depends on `utils/numeric.py`
- Needs clamping and normalization functions
- Justified: Data operations naturally need numeric utilities
- Still testable in isolation with mocked dependencies

**Example**: `utils/cache.py` → Depends on `utils/graph.py`
- Needs graph versioning to invalidate caches
- Justified: Caching must be aware of graph changes
- Interface is minimal (only `edge_version_cache` used)

#### High Coupling (Requires Justification)

**Example**: `structural.py` → Depends on many layers
- Uses operators, mathematics, utils, validation
- Justified: High-level orchestration module
- Coupling is **necessary** to coordinate subsystems
- Well-tested despite complexity

**Red flag pattern** (not present in TNFR):
- Two modules both import each other (circular)
- Utils module depending on high-level modules
- Core types importing from specific implementations

---

## System Invariants

### TNFR Structural Invariants

These invariants **must** be preserved by all factory functions and module operations:

1. **EPI as Coherent Form**
   - EPI only changes via structural operators
   - No ad-hoc mutations allowed
   - Changes are traceable and logged

2. **Structural Units**
   - νf expressed in **Hz_str** (structural hertz)
   - Do not mix with physical frequencies
   - Always use dimensionally correct scaling

3. **ΔNFR Semantics**
   - Sign and magnitude modulate reorganization rate
   - Not a classic ML "error" or "loss gradient"
   - Hermitian matrix if in operator form

4. **Operator Closure**
   - Operator composition yields valid TNFR states
   - New functions must map to existing operators or be defined as one

5. **Phase Verification**
   - No coupling valid without explicit phase check
   - Phase must be in [0, 2π] range
   - Phase synchrony computed via Kuramoto order parameter

6. **Node Birth/Collapse Conditions**
   - Birth requires: sufficient νf, coupling, reduced ΔNFR
   - Collapse causes: extreme dissonance, decoupling, frequency failure

7. **Operational Fractality**
   - EPIs can nest without losing functional identity
   - Avoid flattening that breaks recursivity
   - Sub-EPIs maintain coherence independently

8. **Controlled Determinism**
   - Simulations may be stochastic but must be reproducible
   - Use explicit seeds for RNG
   - Log all structural events with timestamps

9. **Structural Metrics**
   - Expose C(t), Si, phase, νf in telemetry
   - Avoid alien metrics that dilute TNFR semantics
   - Metrics must be computable from structural state

10. **Domain Neutrality**
    - Engine is trans-scale and trans-domain
    - No hard-wired assumptions from specific fields
    - Keep core abstractions general

### Factory-Specific Invariants

#### Operator Factories (`make_*`)

- **Hermiticity**: All operators must be Hermitian (within tolerance)
- **Positive Semidefiniteness**: Coherence and frequency operators must be PSD
- **Backend Compatibility**: Must work with all supported backends (NumPy, JAX, PyTorch)
- **Idempotence**: Same inputs always produce equivalent outputs

#### Generator Factories (`build_*`)

- **Hermiticity**: Generators must be Hermitian matrices
- **Trace Preservation**: Lindblad generators must preserve trace
- **Scaling Consistency**: νf scaling must be dimensionally correct
- **Reproducibility**: Same RNG seed must produce identical output

#### Node Factories (`create_*`)

- **Attribute Completeness**: All required node attributes must be set
- **Valid Initial State**: Initial C(t) = 1, ΔNFR = 0, 0 ≤ phase < 2π
- **Unique Identifiers**: Node IDs must be unique within graph
- **Graph Consistency**: Graph metadata must be updated correctly

---

## Quick References

### Factory Naming Cheatsheet

```python
# Create operator instances
op = make_coherence_operator(dim=5)
op = make_frequency_operator(matrix=H)
rng = make_rng(seed=42, key=1)

# Build data structures
gen = build_delta_nfr(dim=10, nu_f=2.0)
lind = build_lindblad_delta_nfr(dim=5, gamma=0.1)
iso = build_isometry_factory(input_dim=3, output_dim=5)

# Create nodes
G, node = create_nfr(G, nu_f=1.5, phase=0.5)
G, node = create_math_nfr(G, dim=10)
```

### Type Stub Commands

```bash
make help                  # Show all commands
make stubs                 # Generate missing stubs
make stubs-check           # Check for missing (CI)
make stubs-check-sync      # Check if outdated (CI)
make stubs-sync            # Regenerate outdated
```

### Module Import Rules

```python
# ✓ Good: Import from public API
from tnfr import create_nfr, Coherence
from tnfr.utils import CacheManager, clamp

# ✓ Good: Import specific backend utilities
from tnfr.backends import get_backend

# ✓ Good: Import validation functions
from tnfr.validation import validate_frequency

# ✗ Bad: Import private modules
from tnfr.utils._internal import something  # Private!

# ✗ Bad: Import from higher layers in lower layers
# (e.g., utils/ importing from operators/)

# ✗ Bad: Circular imports
# module_a.py imports module_b
# module_b.py imports module_a
```

### Validation Checklist

Before creating a PR with factory changes:

- [ ] Factory follows naming convention (`make_*`, `build_*`, `create_*`)
- [ ] All parameters have type annotations
- [ ] Keyword-only arguments for options (after `*`)
- [ ] Input validation with descriptive error messages
- [ ] Structural invariants verified (Hermiticity, PSD, etc.)
- [ ] Complete NumPy-style docstring
- [ ] Backend integration if manipulating arrays
- [ ] Tests cover: valid cases, invalid inputs, edge cases
- [ ] Reproducibility tested (if using RNG)
- [ ] Stub file generated: `make stubs`
- [ ] Stub file synchronized: `make stubs-check-sync`
- [ ] No circular import dependencies introduced
- [ ] All tests pass: `pytest tests/`

---

## References

- [AGENTS.md](../../../AGENTS.md) - TNFR paradigm fundamentals
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - General contribution guidelines
- [ARCHITECTURE.md](../../../ARCHITECTURE.md) - Overall project structure
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md) - Caching and optimization patterns
- [Testing Strategies](TESTING_STRATEGIES.md) - Test patterns and coverage requirements
- [Foundations](../foundations.md) - Mathematical foundations
- [API Reference](../api/overview.md) - Complete API documentation

---

**Last Updated**: 2025-11-06  
**Status**: Active - consolidates FACTORY_*, DEPENDENCY_*, MODULE_* docs  
**Maintenance**: Update when patterns change, review quarterly
