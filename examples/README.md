# TNFR Examples

> DEPRECATION NOTICE (Scope): Canonical examples are those referenced in `docs/source/examples/README.md`. Other example files in this directory are non-centralized and may be deprecated in future releases. Prefer the centralized examples and documentation paths.

This directory contains Jupyter notebooks and Python examples demonstrating TNFR (Resonant Fractal Nature Theory) concepts and computations.

> **New to TNFR?** Start with the [TNFR Fundamental Concepts Guide](../docs/source/getting-started/TNFR_CONCEPTS.md) to understand the paradigm before diving into examples.

## 🆕 Grammar 2.0 Optimization Guides

All examples have been optimized using Grammar 2.0 features for improved structural health:

- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Comprehensive guide to optimizing TNFR sequences
- **[HEALTH_BENCHMARKS.md](HEALTH_BENCHMARKS.md)** - Before/after comparisons of all optimizations

**Key improvements:**
- All SDK sequences now have health ≥ 0.70 (good structural quality)
- Balanced stabilizers and destabilizers for better structural dynamics
- Harmonic frequency transitions following Grammar 2.0 principles
- Educational comments explaining optimization patterns

## 🌱 Domain-Specific Applications

### Therapeutic Domain Examples

**[domain_applications/](domain_applications/)** - Specialized therapeutic patterns demonstrating TNFR in healing and personal transformation contexts.

**Available modules:**
- **[therapeutic_patterns.py](domain_applications/therapeutic_patterns.py)** - 5 validated therapeutic sequences (avg health: 0.800)
  - Crisis intervention, process therapy, regenerative healing, insight integration, relapse prevention
- **[therapeutic_case_studies.py](domain_applications/therapeutic_case_studies.py)** - 4 clinical case studies (avg health: 0.786)
  - Trauma recovery, addiction healing, depression emergence, relationship repair
- **[therapeutic_optimization.py](domain_applications/therapeutic_optimization.py)** - Comparative analysis and optimization
  - Speed vs. depth trade-offs, intensity calibration, strategy selection, modality comparison

**Quick start:**
```bash
# View all therapeutic patterns with validation
python examples/domain_applications/therapeutic_patterns.py

# Explore clinical case studies
python examples/domain_applications/therapeutic_case_studies.py

# Run comparative optimization analysis
python examples/domain_applications/therapeutic_optimization.py
```

See **[README_THERAPEUTIC.md](domain_applications/README_THERAPEUTIC.md)** for comprehensive documentation on therapeutic applications, including:
- Clinical interpretation of structural operators
- Pattern selection guidance
- Health metrics for therapeutic contexts
- Structural frequency considerations in healing

**Validation status:** ✅ All sequences >0.75 health, compliant with Grammar 2.0 (R1-R5 rules)

### Educational Domain Examples

**[domain_applications/](domain_applications/)** - Specialized educational patterns demonstrating TNFR in learning and cognitive development contexts.

**Available modules:**
- **[educational_patterns.py](domain_applications/educational_patterns.py)** - 5 validated educational sequences (avg health: 0.815)
  - Conceptual breakthrough, competency development, knowledge spiral, practice mastery, collaborative learning
- **[educational_case_studies.py](domain_applications/educational_case_studies.py)** - 5 pedagogical case studies (avg health: 0.854)
  - Mathematics learning, language acquisition, scientific method, skill mastery, creative writing
- **[educational_optimization.py](domain_applications/educational_optimization.py)** - Comparative analysis and optimization
  - Instruction approaches, practice structures, feedback timing, assessment strategies

**Quick start:**
```bash
# View all educational patterns with validation
python examples/domain_applications/educational_patterns.py

# Explore pedagogical case studies
python examples/domain_applications/educational_case_studies.py

# Run comparative optimization analysis
python examples/domain_applications/educational_optimization.py
```

See **[README_EDUCATIONAL.md](domain_applications/README_EDUCATIONAL.md)** for comprehensive documentation on educational applications, including:
- Pedagogical interpretation of structural operators
- Pattern selection for different learning goals
- Health metrics for educational contexts
- Structural frequency considerations in learning

**Validation status:** ✅ All sequences >0.75 health, compliant with Grammar 2.0 (R1-R5 rules)


## Quick Start Examples

### hello_world.py

**The simplest TNFR example - 3 lines of code!**

```python
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("hello_world")
network.add_nodes(10).connect_nodes(0.3, "random")
results = network.apply_sequence("basic_activation", repeat=3).measure()
```

Uses optimized "basic_activation" sequence (health: 0.79)

**Execution:**
```bash
python examples/hello_world.py
```

### sdk_example.py

**Demonstrates the TNFR SDK fluent API**

Shows how to use simplified SDK for non-experts:
- Fluent API with method chaining
- Pre-configured templates for common use cases
- Automatic validation of TNFR invariants
- Easy access to coherence metrics

All sequences use Grammar 2.0 optimizations.

**Execution:**
```bash
python examples/sdk_example.py
```

### health_analysis_demo.py

**Comprehensive health analysis demonstration**

Shows how to use the SequenceHealthAnalyzer to evaluate and optimize sequences:
- Quantitative health metrics (coherence, balance, sustainability)
- Pattern recognition and recommendations
- Before/after Grammar 2.0 optimization comparisons
- Integration with validation API

**Execution:**
```bash
python examples/health_analysis_demo.py
```

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

## Classical Mechanics Validation

**Computational validation of classical mechanics emergence from TNFR**

TNFR includes comprehensive validation that observable classical mechanics emerges naturally from the nodal equation in the low-dissonance regime (ε → 0).

**Validation Examples**:
- **`01_unitary_minimal.ipynb`** — Conservation laws (energy, momentum) from network symmetries
- **`02_dissipative_minimal.ipynb`** — Dissipative dynamics and damping

**Theoretical Framework**:
- [Classical Mechanics from TNFR](../docs/source/theory/07_emergence_classical_mechanics.md) — Derivation: `m = 1/νf`, `F = -∇U`
- [Euler-Lagrange Correspondence](../docs/source/theory/08_classical_mechanics_euler_lagrange.md) — Variational formulation
- [Numerical Validation](../docs/source/theory/09_classical_mechanics_numerical_validation.md) — Six canonical test cases

**Key Results**:
- Mass scaling `m = 1/νf` validated across harmonic, Duffing, and coupled oscillators
- Energy conservation < 0.1% error over 10³ periods in conservative systems
- Bifurcation diagrams and Lyapunov exponents match analytical predictions
- Normal modes and resonances emerge from NFR network coupling

**Run validation**:
```bash
# Interactive notebooks
jupyter notebook examples/01_unitary_minimal.ipynb
jupyter notebook examples/02_dissipative_minimal.ipynb

# See full validation suite in documentation
```

---

## Other Notebooks

### tnfr_visualization.ipynb

Demonstrates deterministic TNFR telemetry visualizations using `tnfr.viz.matplotlib` helpers.
