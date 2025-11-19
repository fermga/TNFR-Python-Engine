# TNFR Python Engine

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602861.svg)](https://doi.org/10.5281/zenodo.17602861)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Canonical computational implementation of TNFR** - A paradigm shift from modeling "things" to modeling **coherent patterns that persist through resonance**.

## What is TNFR?

**Resonant Fractal Nature Theory** proposes a radical reconceptualization of reality:

**Traditional View** → **TNFR View**:

- Objects exist independently → **Patterns exist through resonance**
- Causality (A causes B) → **Co-organization (A and B synchronize)**
- Static properties → **Dynamic reorganization**
- Isolated systems → **Coupled networks**
- Descriptive models → **Generative dynamics**

Reality is not made of "things" but of **coherence**—structures that persist in networks because they **resonate** with their environment.

## Key Features

### 🎯 The 13 Structural Operators

The complete TNFR operator set for modeling coherent structural dynamics.

Canonical Status: The operator registry is now **immutable** – exactly these
13 operators, no more. Dynamic discovery, auto-registration decorators,
metaclass telemetry and reload scripts were removed (see CHANGELOG 9.1.0).
All structural evolution MUST occur through this fixed set (grammar U1-U4).

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

### 📏 Unified Grammar (U1-U6)

Rigorous physics-derived rules ensuring structural validity:

- **U1**: Structural Initiation & Closure
- **U2**: Convergence & Boundedness
- **U3**: Resonant Coupling (phase verification)
- **U4**: Bifurcation Dynamics
- **U5**: Frequency Constraints
- **U6**: Sequential Composition

### 🔬 Four Canonical Fields

Essential structural field computations:

- **Φ_s**: Structural potential
- **|∇φ|**: Phase gradient (reorganization pressure)
- **K_φ**: Phase curvature (bifurcation predictor)
- **ξ_C**: Coherence length (network correlation scale)

### 📊 Telemetry & Metrics

Comprehensive observability:

- **C(t)**: Total coherence [0, 1]
- **Si**: Sense index (reorganization capacity)
- **ΔNFR**: Reorganization gradient
- **νf**: Structural frequency (Hz_str)
- **φ**: Phase synchrony [0, 2π]

### 🧪 Phase 3 Structural Instrumentation

Unified observability and safety layers (read-only):

- `run_structural_validation` combines grammar (U1-U4) + field thresholds.
- `compute_structural_health` converts validation output to recommendations.
- `TelemetryEmitter` streams coherence, sense index, Φ_s, |∇φ|, K_φ, ξ_C.
- `PerformanceRegistry` + `perf_guard` measure overhead (< ~8% in tests).

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

### Hello World (3 lines!)

```python
from tnfr.sdk import TNFRNetwork

network = TNFRNetwork("hello_world")
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

**📚 [Complete Documentation](https://fermga.github.io/TNFR-Python-Engine/)** - Full API reference, tutorials, and theory

**🎓 Key Resources**:

- **[Getting Started Guide](docs/source/getting-started/)** - Installation and first steps
- **[TNFR Fundamental Concepts](docs/source/getting-started/TNFR_CONCEPTS.md)** - Core theory primer
- **[API Reference](docs/source/api/)** - Complete module documentation
- **[Examples](examples/)** - Domain applications and use cases
- **[Grammar System](docs/grammar/)** - Unified grammar (U1-U6) reference
- **[AGENTS.md](AGENTS.md)** - Developer guide for contributing to TNFR
- **[Architecture](ARCHITECTURE.md)** - System design and structure

**🔬 Advanced Topics**:

- **[Unified Grammar Rules](UNIFIED_GRAMMAR_RULES.md)** - Physics derivations for U1-U6
- **[Operator Glossary](GLOSSARY.md)** - Complete operator reference
- **[Testing Strategy](TESTING.md)** - Test coverage and validation
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrading from legacy systems

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

## Performance

Grammar 2.0 optimizations deliver:

- **Sequence validation**: <1ms for typical sequences (10-20 operators)
- **Coherence computation**: O(N) for N nodes
- **Phase gradient**: O(E) for E edges
- **Memory footprint**: ~50MB for 10k-node networks

See **[tools/performance/](tools/performance/)** for benchmarking tools.

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

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for:

- Code of conduct
- Development workflow
- Testing requirements
- Documentation standards
- Pull request process

**For TNFR theory development**, consult **[AGENTS.md](AGENTS.md)** - the canonical guide for maintaining theoretical integrity.
Phase 3 adds structural validation, health assessment and guardrails; see
`docs/STRUCTURAL_HEALTH.md` for thresholds & recommendations.

## Citation

If you use TNFR in your research, please cite:

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2025},
  version = {9.0.2},
  doi = {10.5281/zenodo.17602861},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

See **[CITATION.cff](CITATION.cff)** for machine-readable citation metadata.

## License

This project is licensed under the **MIT License** - see [LICENSE.md](LICENSE.md) for details.

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/fermga/TNFR-Python-Engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fermga/TNFR-Python-Engine/discussions)
- **PyPI**: [pypi.org/project/tnfr](https://pypi.org/project/tnfr/)
- **Documentation**: [fermga.github.io/TNFR-Python-Engine](https://fermga.github.io/TNFR-Python-Engine/)

## Acknowledgments

TNFR represents a fundamental reconceptualization of modeling approaches, prioritizing **coherence over objects**, **resonance over causality**, and **structural dynamics over static properties**.

**Think in patterns, not objects. Think in dynamics, not states. Think in networks, not individuals.**

---

**Reality is not made of things—it's made of resonance. Code accordingly.**
