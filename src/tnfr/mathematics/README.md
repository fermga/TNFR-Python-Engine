# TNFR Mathematics — Canonical Hub (Single Source of Truth)

## Overview

This document is the canonical entry point for all TNFR mathematics in the codebase. It centralizes the mathematical fundamentals, links all experiments and proofs, and prevents redundancy across docs and modules.

Scope and guarantees:
- Canonical contracts for the nodal equation and operators used in code
- Canonical definitions for the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
- Cross-links to formal derivations, symbolic tools, experiments, and notebooks
- English-only documentation; historic non-English references are mapped to English

If any other document disagrees with this README on core computational mathematics, defer to this README and file an issue to reconcile the inconsistency.

Quick pointers:
- Formal theory: [docs/source/theory/mathematical_foundations.md](../../../docs/source/theory/mathematical_foundations.md)
- Symbolic suite: [src/tnfr/math](../math/README.md)
- Fields (Φ_s, |∇φ|, K_φ, ξ_C): [src/tnfr/physics/fields.py](../physics/fields.py) and docs sections below
- Number theory guide (ΔNFR prime criterion): [docs/TNFR_NUMBER_THEORY_GUIDE.md](../../../docs/TNFR_NUMBER_THEORY_GUIDE.md)
- Interactive notebook: [examples/tnfr_prime_checker.ipynb](../../../examples/tnfr_prime_checker.ipynb)

## Module Organization

### Backend Abstraction ([backend.py](backend.py))

Provides a unified interface for numerical operations across NumPy, JAX, and PyTorch:

```python
from tnfr.mathematics import get_backend

backend = get_backend()  # Auto-selects based on config/environment
array = backend.as_array([1, 2, 3])
eigenvalues, eigenvectors = backend.eigh(matrix)
```

**Factory Pattern**: Uses registry pattern with `register_backend()` and private `_make_*_backend()` factories.

### Operator Factories ([operators_factory.py](operators_factory.py))

Constructs validated TNFR operators with structural guarantees:

```python
from tnfr.mathematics import make_coherence_operator, make_frequency_operator

# Create coherence operator with uniform spectrum
coherence_op = make_coherence_operator(dim=4, c_min=0.1)

# Create frequency operator from matrix
freq_op = make_frequency_operator(hamiltonian_matrix)
```

**Factory Pattern**: Uses `make_*` prefix, validates Hermiticity and PSD properties.

### Generator Construction ([generators.py](generators.py))

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

### Transform Contracts ([transforms.py](transforms.py))

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

All factories in this module follow the patterns documented in [Architecture Guide — Factory Patterns](../../../docs/source/advanced/ARCHITECTURE_GUIDE.md#factory-patterns):

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

See [tests/mathematics](../../../tests/mathematics) for the complete test suite.

## Related Documentation

- [Architecture Guide — Factory Patterns](../../../docs/source/advanced/ARCHITECTURE_GUIDE.md#factory-patterns) — Comprehensive factory design patterns
- [TNFR Paradigm](../../../TNFR.pdf) — Theoretical foundations
- [AGENTS.md](../../../AGENTS.md) — Structural invariants and contracts
- [API Overview](../../../docs/source/api/overview.md) — Package-level documentation
- [Mathematical Foundations (theory)](../../../docs/source/theory/mathematical_foundations.md) — Complete derivations

## Canonical equations and contracts

Nodal equation (code-level contract):

∂EPI/∂t = νf · ΔNFR(t)

Inputs/outputs and units:
- EPI: Primary Information Structure (coherent form)
- νf: Structural frequency in Hz_str (must never be relabeled)
- ΔNFR: Nodal reorganization gradient (structural pressure)

Integrated evolution and boundedness (U2):
EPI(t_f) = EPI(t_0) + ∫[t_0..t_f] νf(τ) · ΔNFR(τ) dτ, with the integral required to converge under valid sequences. Destabilizers {OZ, ZHIR, VAL} must be paired with stabilizers {IL, THOL} to maintain boundedness.

Operator composition (U1–U4) in code must always map to canonical operators and preserve invariants; coupling requires phase verification (U3) with |Δφ| ≤ Δφ_max.

## Structural Field Tetrad (canonical telemetry)

These fields are canonical, read-only and do not alter dynamics; they are used for health/safety telemetry.

- Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^2 — Structural potential (global)
- |∇φ|(i) = mean_{j∈N(i)} |θ_i − θ_j| — Phase gradient (local desynchronization)
- K_φ(i) = φ_i − (1/deg(i)) Σ_{j∈N(i)} φ_j — Phase curvature (geometric confinement)
- ξ_C from C(r) ~ exp(−r/ξ_C) — Coherence length (spatial correlation scale)

Implementation: see [src/tnfr/physics/fields.py](../physics/fields.py). Safety thresholds and empirical validation are summarized in [AGENTS.md](../../../AGENTS.md) and field-specific docs.

## Symbolic analysis suite (tnfr.math)

For formal, symbolic checks and analytical tooling, use the tnfr.math package (this is the computational mathematics lab that complements the present module):

- Nodal equation display and LaTeX export
- U2 convergence checks (integral boundedness)
- U4 bifurcation risk via ∂²EPI/∂t²
- Closed-form solutions under constant parameters

See: [src/tnfr/math/README.md](../math/README.md) and [examples/math_symbolic_usage.py](../../../examples/math_symbolic_usage.py).

## Prime emergence (Arithmetic TNFR Network) ⭐

An arithmetic TNFR network demonstrates primes as structural attractors. Each integer n becomes a TNFR node with EPI (form), νf (structural frequency), and ΔNFR (factorization pressure). Primes emerge with ΔNFR = 0 (exact) under TNFR equations, providing a physics-based characterization of primality.

### Theoretical core

Numbers as TNFR nodes (n ∈ ℕ):

```
EPI_n   = 1 + α·ω(n) + β·log τ(n) + γ·(σ(n)/n − 1)
νf_n    = ν₀·(1 + δ·τ(n)/n + ε·ω(n)/log n)
ΔNFR_n  = ζ·(ω(n) − 1) + η·(τ(n) − 2) + θ·(σ(n)/n − (1 + 1/n))
```

Where:
- τ(n): number of divisors
- σ(n): sum of divisors
- ω(n): prime factor count (with multiplicity)

Prime criterion (TNFR):

```
n is prime  ⟺  ΔNFR_n = 0
```

Interpretation:
- ΔNFR = 0 → zero factorization pressure (equilibrium)
- Coherence local c_n = 1/(1+|ΔNFR_n|) = 1.0 for primes
- Composites have ΔNFR > 0 (positive structural pressure)

### Empirical validation (N up to 100,000)

- Perfect separation with ΔNFR == 0 as criterion (validated up to N=100k; AUC=1.0)
- Clear structural separation across EPI and ΔNFR telemetry
- Reproducible runs with seeded pipelines

### Quick start

```python
from tnfr.mathematics import ArithmeticTNFRNetwork

net = ArithmeticTNFRNetwork(max_number=100)

# Inspect a prime
p7 = net.get_tnfr_properties(7)
print(p7['is_prime'], p7['DELTA_NFR'])  # True, 0.0

# Detect prime candidates by low ΔNFR
candidates = net.detect_prime_candidates(delta_nfr_threshold=0.1)
print([n for n, _ in candidates][:10])
```

CLI-style quick validation:

```python
from tnfr.mathematics import run_basic_validation
run_basic_validation(max_number=100)
```

### Structural fields telemetry (Φ_s, |∇φ|, K_φ, ξ_C)

```python
# Compute phases and fields
net.compute_phase(method="spectral", store=True)
phi_grad = net.compute_phase_gradient()         # |∇φ|
k_phi    = net.compute_phase_curvature()         # K_φ
phi_s    = net.compute_structural_potential(alpha=2.0, distance_mode="arithmetic")  # Φ_s
xi       = net.estimate_coherence_length(distance_mode="topological")                # ξ_C
```

Safety/readiness metrics (from AGENTS.md):
- K_φ safety: fraction |K_φ| ≥ 3.0
- Multiscale K_φ: var(K_φ) ~ r^{-α}, expect α ≈ 2.76 (R² ≥ 0.5)

```python
net.compute_kphi_safety(threshold=3.0)
net.k_phi_multiscale_safety(distance_mode='arithmetic', alpha_hint=2.76)
```

### Performance and scaling

- Centralized caching: uses repo `@cache_tnfr_computation`
- CANONICAL fields: reuses `physics.fields` implementations when available
- Distance modes: `arithmetic` for O(n²) Φ_s on large N; `topological` for graph-aware runs
- Coherence length ξ_C: automatically skipped/approximated for very large N in benchmarks

### Benchmarks and exports

Run the provided helpers (see `benchmarks/`):

```bash
# Small (N≈200) validation with plots
python benchmarks/_run_arith_small.py

# Large (N≈5000) telemetry export (JSONL + plots)
python benchmarks/_run_arith_large.py
```

Outputs include:
- Φ_s histograms/heatmaps, K_φ multiscale fits
- JSONL per-node telemetry with EPI, νf, ΔNFR, c_i, φ, |∇φ|, K_φ
- Global metrics for reproducible analysis

### Notebook: primality check (TNFR equations only)

A ready-to-use notebook verifies a number’s primality using only the TNFR pressure equation ΔNFR (no factorization or external primality tests):

- Path: [examples/tnfr_prime_checker.ipynb](../../../examples/tnfr_prime_checker.ipynb)
- Cells: explanation, imports, `tnfr_is_prime(n)` function, interactive and batch tests

Logic: `tnfr_is_prime(n) := (ΔNFR_n == 0)` with ΔNFR_n as defined above. This complies with U1–U4 and preserves the invariants (ΔNFR as structural pressure, νf in Hz_str, no ad-hoc EPI mutations).

Notes:
- Constructive/physical approach: identifies primes as “structural fixed points” (ΔNFR=0). No factorization performed.
- For large n, build the network with `max_number ≥ n` and evaluate ΔNFR_n.

## Classical mechanics emergence (cross-reference)

For the emergence of classical mechanics from TNFR (mass m = 1/νf; force as coherence gradient), see:
- [docs/source/theory/07_emergence_classical_mechanics.md](../../../docs/source/theory/07_emergence_classical_mechanics.md)
- [docs/source/theory/08_classical_mechanics_euler_lagrange.md](../../../docs/source/theory/08_classical_mechanics_euler_lagrange.md)
- [docs/source/theory/09_classical_mechanics_numerical_validation.md](../../../docs/source/theory/09_classical_mechanics_numerical_validation.md)

This README serves as the hub; the above documents contain full derivations and validation results.
