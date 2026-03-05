# TNFR: Resonant Fractal Nature Theory
## Computational Framework for Coherent Pattern Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17764207.svg)](https://doi.org/10.5281/zenodo.17764207)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TNFR (Resonant Fractal Nature Theory)** provides a mathematical framework for modeling coherent patterns in complex systems through resonance-based dynamics.

**Version**: 0.0.3 (March 2026)  
**Theoretical Foundation**: Universal Tetrahedral Correspondence (φ↔Φ_s, γ↔|∇φ|, π↔K_φ, e↔ξ_C)  
**Installation**: `pip install tnfr`

## Getting Started

**Installation**: `pip install tnfr`

**Essential Concepts**:
- **Nodal Equation**: `∂EPI/∂t = νf · ΔNFR(t)` - fundamental evolution law
- **Canonical Operators**: 13 structural operators for system modification
- **Unified Grammar**: U1-U6 constraint rules derived from theoretical physics
- **Structural Fields**: Tetrahedral field system (Φ_s, |∇φ|, K_φ, ξ_C)

**First Steps**:
1. Install package: `pip install tnfr`
2. Read theoretical foundation: [AGENTS.md](AGENTS.md)
3. Run basic example: `python examples/01_hello_world.py`
4. Study terminology: [theory/GLOSSARY.md](theory/GLOSSARY.md)

### Learning Path

**Beginner** (2 hours):
1. Install TNFR: `pip install tnfr`
2. Read [AGENTS.md](AGENTS.md) - complete theoretical foundation
3. Run `python examples/01_hello_world.py`

**Intermediate** (1 week):
1. Study [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md) - physics derivations
2. Work through [examples/](examples/) sequential tutorials (01-40)
3. Explore [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) - field implementations

**Advanced** (ongoing):
1. **TNFR-Riemann Program**: [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) - Complete framework connecting discrete operators to Riemann Hypothesis
2. Technical derivations: [theory/TUTORIAL_FROM_NODAL_EQUATION_TO_COSMOS.md](theory/TUTORIAL_FROM_NODAL_EQUATION_TO_COSMOS.md)
3. Contribute following [CONTRIBUTING.md](CONTRIBUTING.md)

## What is TNFR?

**Resonant Fractal Nature Theory** provides a mathematical framework for modeling coherent patterns in complex systems. The theory establishes correspondence between four universal mathematical constants (φ, γ, π, e) and four structural fields that characterize system dynamics.

**Core Principle**: Systems are modeled as coherent patterns maintained through resonant coupling rather than as discrete objects with independent properties.

**Theoretical Foundation**: [AGENTS.md](AGENTS.md) - Complete theory reference  
**Mathematical Details**: [Structural Fields and Universal Tetrahedral Correspondence](theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)  
**Theory Hub**: [theory/README.md](theory/README.md) - Comprehensive theoretical documentation  
**Implementation**: This repository provides computational tools for TNFR analysis

**Theoretical Approach**:
- **Pattern analysis** over discrete objects
- **Resonant coupling** over causal relationships
- **Dynamic processes** over static properties
- **Network coherence** over isolated systems

## Key Features

### The 13 Canonical Structural Operators

TNFR provides a complete set of 13 operators for modeling structural dynamics. All system evolution occurs through these operators according to unified grammar rules (U1-U6).

- **AL (Emission)** - Pattern creation from vacuum
- **EN (Reception)** - Information capture and integration
- **IL (Coherence)** - Stabilization through negative feedback
- **OZ (Dissonance)** - Controlled instability and exploration
- **UM (Coupling)** - Network formation via phase sync
- **RA (Resonance)** - Pattern amplification and propagation
- **SHA (Silence)** - Temporal pause, observation windows
- **VAL (Expansion)** - Structural complexity increase
- **NUL (Contraction)** - Dimensionality reduction
- **THOL (Self-organization)** - Spontaneous autopoietic structuring
- **ZHIR (Mutation)** - Phase transformation at threshold
- **NAV (Transition)** - Regime shift, state changes
- **REMESH (Recursivity)** - Multi-scale fractal operations

### Unified Grammar (U1-U6)

Physics-derived constraints for valid operator sequences:

- **U1**: Structural Initiation & Closure
- **U2**: Convergence & Boundedness
- **U3**: Resonant Coupling (phase verification)
- **U4**: Bifurcation Dynamics
- **U5**: Multi-Scale Coherence
- **U6**: Structural Potential Confinement

**Reference**: [UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md) for complete derivations

### Structural Field Tetrad (Canonical)

Four fields characterizing system state:

- **Φ_s**: Structural potential (global field)
- **|∇φ|**: Phase gradient (local desynchronization)
- **K_φ**: Phase curvature (geometric confinement)
- **ξ_C**: Coherence length (spatial correlations)

**Implementation**: `tnfr.physics.fields`  
**Documentation**: [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)

### TNFR Engines Hub

Centralized mathematical and optimization engines for advanced TNFR applications:

#### **Self-Optimization Engine** (`tnfr.engines.self_optimization`)
- **TNFRSelfOptimizingEngine**: Automatic network optimization using TNFR operators
- **Physics**: Based on ∂EPI/∂t = νf · ΔNFR(t) nodal equation
- **Capabilities**: Smart node targeting, multi-operator optimization, real-time metrics
- **Usage**: Direct engine or via SDK `auto_optimize()`

#### **Pattern Discovery Engines** (`tnfr.engines.pattern_discovery`)
- **TNFREmergentPatternEngine**: Mathematical pattern detection and emergence analysis
- **UnifiedPatternDetector**: Operator sequence pattern recognition
- **Discoveries**: Eigenmodes, spectral cascades, fractal scaling, symmetry breaking
- **Applications**: Research insights, system analysis, predictive modeling

#### **Computation Engines** (`tnfr.engines.computation`)
- **GPUEngine**: GPU-accelerated TNFR computations for high-performance analysis
- **FFTEngine**: Fast Fourier Transform processing for spectral analysis
- **Performance**: Parallel processing, optimized algorithms

#### **Integration Engines** (`tnfr.engines.integration`)
- **EmergentIntegrationEngine**: Multi-scale analysis and hierarchical coupling
- **Capabilities**: Cross-scale information flow, emergent behavior detection

**Quick Usage**:
```python
# Via SDK (recommended)
from tnfr.sdk import TNFR
net = TNFR.create(50).random(0.3).auto_optimize()

# Direct engine access
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
engine = TNFRSelfOptimizingEngine(network)
success, metrics = engine.step(node_id)
```

**Documentation**: [src/tnfr/engines/README.md](src/tnfr/engines/README.md)

### Spectral Factorization (Canonical Entry Point)

TNFR ships a canonical Paley spectral factorization workflow that reuses the
same grammar validator and self-optimization engines described above.

- Import `factorize` from `tnfr.factorization` to call the lab implementation.
- Certificates are emitted with canonical operator tokens plus optimizer
  metadata, making results auditable.
- Works with both the in-tree `factorization-lab/` checkout and the external
  `tnfr-factorization` package, so downstream tooling has a single API surface.
- Each `SpectralAnalysisResult` exposes `fft_backend` and
  `fft_capabilities` so you can see which FFT engine executed the run.

```python
from tnfr.factorization import factorize

result = factorize(2310, trace_certificates=True)
print(result.candidate_factors)
print(result.certificate.canonical_operators)
print(result.fft_backend, result.fft_capabilities)
```

See `factorization-lab/README.md` for schematics, CLI usage, and certificate
formats. The long-term scaling roadmap (partitioned graphs + distributed FFT)
is tracked in `docs/FACTORIZATION_SCALING_PLAN.md`.

### Core Metrics

Structural system telemetry:

- **C(t)**: Total coherence [0, 1]
- **Si**: Sense index (reorganization capacity)
- **ΔNFR**: Internal reorganization operator
- **νf**: Structural frequency (Hz_str)
- **φ**: Phase synchrony [0, 2π]

### Validation & Health Monitoring

Structural validation and monitoring tools:

- `run_structural_validation`: Grammar validation with field thresholds
- `compute_structural_health`: Health assessment and recommendations
- `TelemetryEmitter`: Comprehensive telemetry streaming
- `PerformanceRegistry`: Performance monitoring

Usage:

```python
from tnfr.validation.aggregator import run_structural_validation
from tnfr.validation.health import compute_structural_health
from tnfr.performance.guardrails import PerformanceRegistry

perf = PerformanceRegistry()
report = run_structural_validation(
  G,
  sequence=["AL","UM","IL","SHA"],
  perf_registry=perf,
)
health = compute_structural_health(report)
print(report.risk_level, health.recommendations)
print(perf.summary())
```

Telemetry:

```python
from tnfr.metrics.telemetry import TelemetryEmitter

with TelemetryEmitter("results/run.telemetry.jsonl", human_mirror=True) as em:
  for step, op in enumerate(["AL","UM","IL","SHA"]):
    em.record(G, step=step, operator=op, extra={"sequence_id": "demo"})
```

Risk levels:

- `low` – Grammar valid; no thresholds exceeded.
- `elevated` – Local stress: max |∇φ|, |K_φ| pocket, ξ_C watch.
- `critical` – Grammar invalid or ΔΦ_s / ξ_C critical breach.


All instrumentation preserves TNFR physics (no state mutation).

## Installation

### From PyPI (Stable)

```bash
pip install tnfr
```

### From Source (Development)

```bash
git clone https://github.com/fermga/TNFR-Python-Engine.git
cd TNFR-Python-Engine
pip install -e ".[dev-minimal]"
```

### Dependency Profiles

```bash
# Core functionality only
pip install .

# Development tools (linting, formatting, type checking)
pip install -e ".[dev-minimal]"

# Full test suite
pip install -e ".[test-all]"

# Documentation building
pip install -e ".[docs]"

# Visualization support
pip install -e ".[viz-basic]"

# Alternative backends
pip install -e ".[compute-jax]"   # JAX backend
pip install -e ".[compute-torch]"  # PyTorch backend
```

## Quick Start

### Basic Usage

```python
from tnfr.sdk import TNFRNetwork

# Create and configure network
network = TNFRNetwork("example")
network.add_nodes(10).connect_nodes(0.3, "random")
results = network.apply_sequence("basic_activation", repeat=3).measure()

print(f"Coherence: {results.coherence:.3f}")
```

### Using Operators Directly

```python
import networkx as nx
from tnfr.operators.definitions import Emission, Coherence, Resonance
from tnfr.operators.grammar import validate_sequence
from tnfr.metrics.coherence import compute_coherence

# Create network
G = nx.erdos_renyi_graph(20, 0.2)

# Apply operator sequence
sequence = ["AL", "IL", "RA", "SHA"]
result = validate_sequence(sequence)

if result.valid:
    for node in G.nodes():
        Emission().apply(G, node)
        Coherence().apply(G, node)
        Resonance().apply(G, node)
    
    # Measure
    C_t = compute_coherence(G)
    print(f"Network coherence: {C_t:.3f}")
```

### Domain Applications

The [examples/](examples/) directory contains 42 sequential tutorials covering:

- **Classical Mechanics**: `examples/12_classical_mechanics_demo.py` (Keplerian orbits)
- **Quantum Mechanics**: `examples/13_quantum_mechanics_demo.py` (emergent quantization)
- **Biology**: `examples/18_biology_emergence_demo.py` (biological emergence)
- **Neuroscience**: `examples/19_neuroscience_demo.py` (neural patterns)
- **Conservation Laws**: `examples/40_conservation_law_demo.py` (structural conservation)

See [examples/README.md](examples/README.md) for the full index.

## Documentation

**Complete Documentation**: [fermga.github.io/TNFR-Python-Engine](https://fermga.github.io/TNFR-Python-Engine/)

### Primary References

**Theoretical Foundation**:

- **[AGENTS.md](AGENTS.md)** - Primary theoretical reference and development guide
- **[theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md)** - Grammar constraint derivations (U1-U6)
- **[Structural Fields and Universal Tetrahedral Correspondence](theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)** - Mathematical foundations

**Implementation Guide**:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md)** - Field implementation specifications
- **[theory/GLOSSARY.md](theory/GLOSSARY.md)** - Technical definitions and terminology

**Development Resources**:

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development workflow and standards
- **[examples/](examples/)** - Sequential tutorials and usage examples
- **[TESTING.md](TESTING.md)** - Testing framework and requirements

## Repository Structure

```text
TNFR-Python-Engine/
├── src/tnfr/              # Core TNFR implementation (~346 files, ~104k LOC)
│   ├── operators/         # 13 canonical operators + grammar validation
│   │   ├── definitions.py        # Operator registry facade
│   │   ├── grammar.py            # Unified grammar U1-U6 validation
│   │   ├── grammar_dynamics.py   # Grammar-aware dynamic operator selection
│   │   └── grammar_application.py # Grammar-enforced operator application
│   ├── physics/           # Structural fields and conservation
│   │   ├── fields.py             # Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
│   │   ├── conservation.py       # Structural Conservation Theorem
│   │   ├── integrity.py          # Closed-loop integrity monitor (13/13 postconditions)
│   │   └── interactions.py       # Force emergence and coupling
│   ├── dynamics/          # Nodal equation integration
│   │   ├── self_optimizing_engine.py  # Autonomous structural optimization
│   │   ├── canonical.py               # Canonical dynamics
│   │   └── selectors.py               # Grammar-filtered operator selection
│   ├── engines/           # Centralized engines hub
│   │   ├── self_optimization/    # Auto-optimization engine
│   │   ├── pattern_discovery/    # Mathematical pattern detection
│   │   ├── computation/          # GPU/FFT acceleration
│   │   └── integration/          # Multi-scale integration
│   ├── metrics/           # Coherence, Si, phase sync, telemetry
│   ├── constants/         # 497+ canonical constants (φ, γ, π, e derivations)
│   ├── sdk/               # Simplified + Fluent API
│   ├── riemann/           # TNFR-Riemann operator implementation
│   ├── factorization/     # Spectral factorization workflow
│   └── mathematics/       # Nodal equation + number theory
├── tests/                 # Test suite (471 passing, 9 skipped)
├── examples/              # 42 sequential tutorials and demos
├── theory/                # Theoretical documents and derivations
├── docs/                  # Implementation specifications
├── notebooks/             # Jupyter notebooks
├── benchmarks/            # Performance validation suites
└── scripts/               # Maintenance utilities
```

## Testing

```bash
# Run all tests
pytest

# Fast smoke tests (examples + telemetry)
make smoke-tests          # Unix/Linux
.\make.cmd smoke-tests    # Windows

# Specific test suites
pytest tests/unit/mathematics/         # Math tests
pytest tests/examples/                 # Example validation
pytest tests/integration/              # Integration tests
```

## Repository Maintenance

```bash
# Clean generated artifacts
make clean                # Unix/Linux
.\make.cmd clean          # Windows

# Verify documentation references
python scripts/verify_internal_references.py

# Security audit
pip-audit
```

## Performance Characteristics

- **Sequence validation**: <1ms for typical sequences (10-20 operators)
- **Coherence computation**: O(N) for N nodes
- **Phase gradient**: O(E) for E edges
- **Memory footprint**: ~50MB for 10k-node networks

Benchmarking tools: [benchmarks/](benchmarks/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines:

- Development workflow and standards
- Testing requirements and procedures
- Documentation standards and formatting
- Pull request process and review criteria

**Theoretical Development**: Follow theoretical integrity guidelines in [AGENTS.md](AGENTS.md).

**Technical References**:

- [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) - Field implementation specifications
- [TESTING.md](TESTING.md) - Testing framework requirements

## Citation

To cite this software:

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2025},
  version = {0.0.3},
  doi = {10.5281/zenodo.17764207},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

This project is licensed under the **MIT License** - see [LICENSE.md](LICENSE.md) for details.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/fermga/TNFR-Python-Engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fermga/TNFR-Python-Engine/discussions)
- **PyPI**: [pypi.org/project/tnfr](https://pypi.org/project/tnfr/)
- **Documentation**: [fermga.github.io/TNFR-Python-Engine](https://fermga.github.io/TNFR-Python-Engine/)

## Theoretical Foundation

TNFR implements pattern-based modeling principles:

- **Pattern coherence** over discrete objects
- **Resonant coupling** over causal relationships
- **Dynamic processes** over static properties
- **Network structures** over isolated entities

Implementation follows theoretical foundations in [AGENTS.md](AGENTS.md) with adherence to canonical invariants and unified grammar constraints.
