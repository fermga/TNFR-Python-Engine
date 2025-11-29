# TNFR: Resonant Fractal Nature Theory
## Computational Framework for Coherent Pattern Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17761312.svg)](https://doi.org/10.5281/zenodo.17761312)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TNFR (Resonant Fractal Nature Theory)** provides a mathematical framework for modeling coherent patterns in complex systems through resonance-based dynamics.

**Version**: 0.0.1 (November 29, 2025)  
**Theoretical Foundation**: Universal Tetrahedral Correspondence (φ↔Φ_s, γ↔|∇φ|, π↔K_φ, e↔ξ_C)  
**Installation**: `pip install tnfr`

## Getting Started

### Getting Started

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
2. Work through [examples/](examples/) sequential tutorials (01-10)
3. Explore [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py) - field implementations

**Advanced** (ongoing):
1. **TNFR-Riemann Program**: [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) - Complete framework connecting discrete operators to Riemann Hypothesis
2. Technical derivations: [theory/TUTORIAL_FROM_NODAL_EQUATION_TO_COSMOS.md](theory/TUTORIAL_FROM_NODAL_EQUATION_TO_COSMOS.md)
3. Contribute following [CONTRIBUTING.md](CONTRIBUTING.md)

## What is TNFR?

**Resonant Fractal Nature Theory** provides a mathematical framework for modeling coherent patterns in complex systems. The theory establishes correspondence between four universal mathematical constants (φ, γ, π, e) and four structural fields that characterize system dynamics.

**Core Principle**: Systems are modeled as coherent patterns maintained through resonant coupling rather than as discrete objects with independent properties.

**Theoretical Foundation**: [AGENTS.md](AGENTS.md) - Complete theory reference  
**Mathematical Details**: [theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)  
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

**Reference**: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete derivations

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

CLI health report:

```bash
python scripts/structural_health_report.py --graph random:50:0.15 --sequence AL,UM,IL,SHA
```

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

```bash
# Therapeutic patterns (crisis, trauma, healing)
python examples/domain_applications/therapeutic_patterns.py

# Educational patterns (learning, mastery, breakthrough)
python examples/domain_applications/educational_patterns.py

# Biological systems (metabolism, evolution)
python examples/domain_applications/biological_patterns.py
```

## Documentation

**Complete Documentation**: [fermga.github.io/TNFR-Python-Engine](https://fermga.github.io/TNFR-Python-Engine/)

### Primary References

**Theoretical Foundation**:
- **[AGENTS.md](AGENTS.md)** - Primary theoretical reference and development guide
- **[theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md)** - Grammar constraint derivations (U1-U6)
- **[theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md](theory/FUNDAMENTAL_TNFR_THEORY_UNIVERSAL_TETRAHEDRAL_CORRESPONDENCE.md)** - Mathematical foundations

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
├── src/tnfr/              # Core TNFR implementation
│   ├── operators/         # Canonical operator system (immutable registry)
│   │   ├── definitions.py        # Facade (backward compatibility)
│   │   ├── definitions_base.py   # Operator base class (no dynamic metaclass)
│   │   ├── emission.py           # AL operator
│   │   ├── coherence.py          # IL operator
│   │   └── ... (13 operators)    # Individual operator modules (canonical)
│   ├── operators/grammar/ # Unified grammar constraints (Phase 1)
│   │   ├── grammar.py            # Facade (unified validation)
│   │   ├── u1_initiation_closure.py
│   │   ├── u2_convergence_boundedness.py
│   │   └── ... (8 constraint modules)
│   ├── metrics/           # Modular metrics system (Phase 1)
│   │   ├── metrics.py            # Facade (backward compatibility)
│   │   ├── coherence.py          # C(t) computation
│   │   ├── sense_index.py        # Si measurement
│   │   ├── phase_sync.py         # Phase synchronization
│   │   └── telemetry.py          # Execution tracing
│   ├── physics/           # Canonical fields (Φ_s, |∇φ|, K_φ, ξ_C)
│   ├── dynamics/          # Nodal equation integration
│   ├── sdk/               # High-level API
│   └── tutorials/         # Educational modules
├── tests/                 # Comprehensive test suite (975/976 passing)
├── examples/              # Domain applications
├── docs/                  # Documentation source
├── notebooks/             # Jupyter notebooks
├── benchmarks/            # Performance testing
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

# Check repository health
python scripts/repo_health_check.py

# Verify documentation references
python scripts/verify_internal_references.py

# Security audit
pip-audit
```

See **[REPO_OPTIMIZATION_PLAN.md](docs/REPO_OPTIMIZATION_PLAN.md)** for cleanup routines and targeted test bundles.

## Performance Characteristics

- **Sequence validation**: <1ms for typical sequences (10-20 operators)
- **Coherence computation**: O(N) for N nodes
- **Phase gradient**: O(E) for E edges
- **Memory footprint**: ~50MB for 10k-node networks

Benchmarking tools: [tools/performance/](tools/performance/)

Note on Python executable for local runs

- Windows: prefer `./test-env/Scripts/python.exe`
- macOS/Linux: prefer `./test-env/bin/python`

Using the workspace virtual environment avoids mismatches with system Pythons
that may lack the latest telemetry aliases or configuration.

### Parse precision_modes drift (benchmark_results.json)

After running `./test-env/Scripts/python.exe run_benchmark.py` (Windows) or
`./test-env/bin/python run_benchmark.py` (macOS/Linux), parse numeric drift for
the `precision_modes` track:

```python
import json

with open("benchmark_results.json", "r", encoding="utf-8") as f:
  data = json.load(f)

drift_entries = data.get("precision_modes", {}).get("drift", [])
for entry in drift_entries:
  size = entry.get("size")
  phi_s = entry.get("phi_s_max_abs")
  grad = entry.get("grad_max_abs")
  curv = entry.get("curv_max_abs")
  xi_c = entry.get("xi_c_abs")
  print(
    f"N={size:>4}  ΔΦ_s_max={phi_s:.3e}  |∇φ|_max={grad:.3e}  "
    f"K_φ_max={curv:.3e}  ξ_C_abs={xi_c if xi_c is not None else 'nan'}"
  )
```

This reports the maximum absolute difference between `standard` and `high` precision modes for the canonical fields per graph size.

PowerShell one-liners (Windows)

```powershell
# Largest ΔΦ_s drift row
Get-Content .\benchmark_results.json | ConvertFrom-Json |
  Select-Object -ExpandProperty precision_modes |
  Select-Object -ExpandProperty drift |
  Sort-Object -Property phi_s_max_abs -Descending |
  Select-Object -First 1

# Largest |∇φ| drift row
Get-Content .\benchmark_results.json | ConvertFrom-Json |
  Select-Object -ExpandProperty precision_modes |
  Select-Object -ExpandProperty drift |
  Sort-Object -Property grad_max_abs -Descending |
  Select-Object -First 1

# Largest K_φ drift row
Get-Content .\benchmark_results.json | ConvertFrom-Json |
  Select-Object -ExpandProperty precision_modes |
  Select-Object -ExpandProperty drift |
  Sort-Object -Property curv_max_abs -Descending |
  Select-Object -First 1

# Largest ξ_C drift row (skip NaNs)
Get-Content .\benchmark_results.json | ConvertFrom-Json |
  Select-Object -ExpandProperty precision_modes |
  Select-Object -ExpandProperty drift |
  Where-Object { $_.xi_c_abs -ne $null -and -not [double]::IsNaN([double]$_.xi_c_abs) } |
  Sort-Object -Property xi_c_abs -Descending |
  Select-Object -First 1
```

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
  version = {0.0.1},
  doi = {10.5281/zenodo.17761312},
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
