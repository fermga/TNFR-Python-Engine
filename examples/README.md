# TNFR Examples

This directory contains Jupyter notebooks demonstrating TNFR (Resonant Fractal Nature Theory) concepts and computations.

> **New to TNFR?** Start with the [TNFR Fundamental Concepts Guide](../docs/source/getting-started/TNFR_CONCEPTS.md) to understand the paradigm before diving into examples.

## Academic Notebooks (Minimal Systems)

These notebooks provide didactic examples using minimal 2×2 systems, computing fundamental TNFR structural metrics.

### 01_unitary_minimal.ipynb

**Unitary Evolution in a 2×2 System**

Demonstrates unitary dynamics driven by a Hermitian ΔNFR generator.

**Key concepts:**
- 2×2 Hilbert space initialization
- Hermitian ΔNFR generator construction (Laplacian topology)
- **νf** (structural frequency): reorganization rate in Hz_str
- **C_min** (minimal coherence): structural stability threshold
- **d_coh** (coherence dissimilarity): measures divergence of coherence configurations
- Unitary evolution via `exp(-i·Δ·dt)`
- Visualization: state amplitudes and d_coh evolution

**Metrics computed:**
- Structural frequency: νf = 1.0 Hz_str
- Minimal coherence: C_min = 0.2
- Coherence dissimilarity trajectory over 20 time steps

**Execution:** Run all cells to generate `01_unitary_evolution.png`

---

### 02_dissipative_minimal.ipynb

**Dissipative (Contractive Semigroup) Evolution in a 2×2 System**

Demonstrates dissipative dynamics driven by a Lindblad ΔNFR generator in Liouville space.

**Key concepts:**
- 2×2 Hilbert space with Lindblad master equation
- Lindblad ΔNFR generator (Hamiltonian + collapse operators)
- **νf** (structural frequency): sets dissipation rate
- **C_min** (minimal coherence): stability threshold
- **d_coh** (coherence dissimilarity): tracks divergence under dissipation
- Contractive semigroup evolution: purity monotonically decreases
- Trace preservation verification
- Visualization: purity decay and d_coh evolution

**Metrics computed:**
- Structural frequency: νf = 1.0 Hz_str
- Minimal coherence: C_min = 0.3
- Purity evolution: initial 1.0 → final ~0.72 (dissipation signature)
- Coherence dissimilarity trajectory

**Execution:** Run all cells to generate `02_dissipative_evolution.png`

---

## API Usage

Both notebooks use **only public API functions** from the `tnfr.mathematics` module:

```python
from tnfr.mathematics import (
    HilbertSpace,
    CoherenceOperator,
    FrequencyOperator,
    MathematicalDynamicsEngine,      # Unitary
    ContractiveDynamicsEngine,       # Dissipative
    build_delta_nfr,                 # Hermitian generator
    build_lindblad_delta_nfr,        # Lindblad generator
)
from tnfr.mathematics.metrics import dcoh
```

## Requirements

```bash
pip install tnfr matplotlib jupyter scipy
```

## Running the Notebooks

### Interactive (Jupyter):
```bash
jupyter notebook
# Open and run 01_unitary_minimal.ipynb or 02_dissipative_minimal.ipynb
```

### Command-line execution:
```bash
jupyter nbconvert --to notebook --execute 01_unitary_minimal.ipynb
jupyter nbconvert --to notebook --execute 02_dissipative_minimal.ipynb
```

## Optional Dependencies Demo

### optional_dependencies_demo.py

**Demonstrates handling of optional dependencies in TNFR**

Shows how TNFR gracefully handles missing optional dependencies while providing informative error messages and type checking compatibility.

**Key concepts:**
- Compatibility layer for numpy, matplotlib, and jsonschema
- Graceful fallback behavior when packages are missing
- Type checking compatibility using stubs
- Informative error messages with installation instructions

**Requirements:**
```bash
# Minimal (core TNFR only)
pip install tnfr

# With optional dependencies
pip install tnfr[numpy,viz]
```

**Execution:**
```bash
python examples/optional_dependencies_demo.py
```

See [docs/source/getting-started/optional-dependencies.md](../docs/source/getting-started/optional-dependencies.md) for comprehensive documentation on optional dependencies.

## TNFR Paradigm References

- **νf (Hz_str)**: Structural frequency - rate of reorganization from ΔNFR
- **C_min**: Minimal eigenvalue of coherence operator - stability threshold
- **d_coh**: Coherence dissimilarity - Bures-style angle measuring configuration divergence
- **ΔNFR**: Internal reorganization operator (Hermitian or Lindblad superoperator)
- **EPI**: Primary Information Structure (coherent form)

See [TNFR.pdf](../TNFR.pdf) for theoretical foundations and [AGENTS.md](../AGENTS.md) for canonical invariants.

---

## Other Notebooks

### tnfr_visualization.ipynb

Demonstrates deterministic TNFR telemetry visualizations using `tnfr.viz.matplotlib` helpers.
