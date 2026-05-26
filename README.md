# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A mathematical framework for modeling coherent patterns in complex systems through resonance-based dynamics.

```bash
pip install tnfr
```

---

## Core Ideas

All systems evolve via the **nodal equation**:

$$\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)$$

Structural changes occur exclusively through **13 canonical operators** (AL, EN, IL, OZ, UM, RA, SHA, VAL, NUL, THOL, ZHIR, NAV, REMESH) governed by **unified grammar rules U1-U6**.

System state is characterized by four **structural fields** — the Universal Tetrahedral Correspondence:

| Constant | Field | Meaning |
|----------|-------|---------|
| φ | Φ_s | Structural potential (global stability) |
| γ | \|∇φ\| | Phase gradient (local stress) |
| π | K_φ | Phase curvature (geometric confinement) |
| e | ξ_C | Coherence length (spatial correlations) |

**Complete theory**: [AGENTS.md](AGENTS.md)

---

## Quick Start

```python
from tnfr.sdk import TNFR

# Create, connect, evolve
net = TNFR.create(20).ring().evolve(5)
print(net.results().summary())
# -> C=0.987, Si=0.912, N=20, E=20, rho=0.105
```

```python
# Structural Field Tetrad — four canonical fields
tetrad = net.tetrad()
print(tetrad.summary())
# -> Phi_s=0.0312, |grad_phi|=0.0841, |K_phi|=0.1523, xi_C=2.3147 (N=20)
print(tetrad.is_safe())  # canonical threshold checks
```

```python
# Conservation laws — Noether charge, Lyapunov stability
cons = net.conservation()
print(cons.summary())
# -> Q=1.2340, E=0.5678, dE/dt=-0.0012 (STABLE), quality=0.998
```

```python
# One-shot comprehensive analysis
analysis = TNFR.analyze(net)
# Returns: coherence, tetrad, conservation, tensor_invariants,
#          emergent_fields, integrity, features
```

```python
# Grammar-aware evolution (proactive U1-U6 enforcement)
net.evolve_grammar_aware(steps=10)
```

```python
# Direct operator usage
import networkx as nx
from tnfr.operators.definitions import Emission, Coherence, Silence
from tnfr.metrics.coherence import compute_coherence

G = nx.erdos_renyi_graph(20, 0.2)
for node in G.nodes():
    Emission().apply(G, node)
    Coherence().apply(G, node)
    Silence().apply(G, node)

print(f"Coherence: {compute_coherence(G):.3f}")
```

---

## Installation

```bash
pip install tnfr                       # stable release
pip install -e ".[dev-minimal]"        # development
pip install -e ".[test-all]"           # full test suite
pip install -e ".[compute-jax]"        # JAX backend
pip install -e ".[compute-torch]"      # PyTorch backend
```

---

## Project Structure

```text
src/tnfr/
├── operators/         # 13 canonical operators + grammar validation (56 modules)
├── physics/           # Structural fields, conservation, integrity (26 modules)
├── engines/           # Self-optimization, pattern discovery, GPU/FFT (7 modules)
├── dynamics/          # Nodal equation integration
├── riemann/           # TNFR-Riemann program (48 modules, P1–P49; paused at T-HP)
├── sdk/               # Simplified & Fluent API (7 modules)
│   └── simple.py      # Tetrad, conservation, grammar-aware dynamics, integrity
├── mathematics/       # Number theory, backends
├── constants/         # Canonical constants (mpmath 35-digit precision)
├── metrics/           # Coherence, Si, phase sync, telemetry
├── validation/        # Structural health monitoring
└── factorization/     # Spectral factorization workflow

examples/              # 42 sequential tutorials (01-40 + extras)
tests/                 # 1,655 tests
theory/                # Theoretical derivations
benchmarks/            # Performance validation (14 suites)
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | **Primary reference** — complete TNFR theory, operators, grammar, fields |
| [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md) | U1-U6 grammar derivations from physics |
| [theory/FUNDAMENTAL_THEORY.md](theory/FUNDAMENTAL_THEORY.md) | Universal Tetrahedral Correspondence |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) | Field implementation specifications |
| [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | TNFR-Riemann program |
| [theory/GLOSSARY.md](theory/GLOSSARY.md) | Terminology and definitions |
| [examples/](examples/) | Sequential tutorials |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development guidelines |

---

## Testing

```bash
pytest                             # all tests (1,655)
pytest tests/sdk/                  # SDK tests (tetrad, conservation, grammar)
pytest tests/unit/                 # unit tests
.\make.cmd smoke-tests             # smoke tests (Windows)
make smoke-tests                   # smoke tests (Unix)
```

---

## Citation

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2025},
  version = {0.0.3.3},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

---

## License

MIT — see [LICENSE.md](LICENSE.md).

## Links

[PyPI](https://pypi.org/project/tnfr/) · [Issues](https://github.com/fermga/TNFR-Python-Engine/issues) · [Discussions](https://github.com/fermga/TNFR-Python-Engine/discussions) · [Documentation](https://fermga.github.io/TNFR-Python-Engine/)
