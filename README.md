# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A mathematical framework for modeling coherent patterns in complex systems through resonance-based dynamics on networks.

A single **nodal equation** drives every node. From it, a complete **transport and geometric structure emerges** — measured by the engine, verified to machine precision, and anchored to classical, experimentally-established physics. The graph is only the substrate; the dynamics generates its own geometry.

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

## From One Equation, a Geometry Emerges

TNFR is more than dynamics on a graph. The graph is only the substrate; the nodal equation **generates its own geometry**, which the engine *measures* rather than postulates. Every structure below is verified to machine precision and anchored to classical, experimentally-established phenomena:

- **Transport layer** (empirically anchored) — channel by channel, the nodal equation is a graph-Laplacian diffusion. From it emerge diffusion, synchronization (Kuramoto), random walks, effective resistance (Ohm/Kirchhoff), and standing-wave modes — all textbook phenomena.
- **Emergent symplectic substrate** (TNFR-native) — the same dynamics carries a phase space with conserved charges (Noether), a Hamiltonian equal to the energy functional, complete integrability, and a polarization structure (Stokes/Poincaré).
- **Orthogonal structure** — the dissipative (transport) and conservative (symplectic) parts are the two orthogonal Helmholtz–Hodge components of one flow.

**Honest scope**: this reorganizes known mathematics and physics inside a single framework, verified in code. It is a *characterization* of structure the nodal equation already contains — not a claim of new physics.

---

## Research Status

Two clearly-separated layers:

**Solid and verified.** The engine, the tetrad, grammar U1–U6, conservation laws, and the emergent transport + symplectic geometry are implemented, anchored to experimentally-established phenomena, and covered by 1,947 tests.

**Open research programs.** TNFR is also used to probe famous open problems. These are honest, in-progress programs that **do not claim proofs**:

| Program | Done | Open |
|---------|------|------|
| TNFR–Riemann (P1–P49) | discrete operator σ_c → 1/2; ζ↔L attack surface | Riemann Hypothesis (gap G4) — **paused at T-HP** |
| TNFR–Navier–Stokes (N1–N17) | NS-G5 closed at discrete-operator level | continuum limit / Clay (NS-G1..G4) — **open** |
| TNFR–Yang–Mills (Y1–Y5) | finite U(1) structural diagnostics | non-Abelian mass gap — **open** (Branch B) |

See [AGENTS.md](AGENTS.md) and the `theory/` research notes for the full, audited status.

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
# Emergent symplectic substrate — the geometry the dynamics generates
sub = net.symplectic_substrate()
print(sub.summary())
# -> dim=80, H_sub=0.0000, U=0.0000, div(X_H)=0.00e+00 (VALID)
# dim = 4N phase space; div(X_H)=0 => Liouville (volume-preserving)
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
├── operators/         # 13 canonical operators + grammar U1–U6 (65 modules)
├── physics/           # Tetrad, conservation, emergent symplectic substrate, structural diffusion (30 modules)
├── engines/           # Self-optimization, pattern discovery, GPU/FFT (14 modules)
├── dynamics/          # Nodal equation integration
├── riemann/           # TNFR–Riemann program (62 modules, P1–P49; paused at T-HP, RH open)
├── navier_stokes/     # TNFR–Navier–Stokes program (N1–N17; NS-G5 closed at discrete level, Clay open)
├── yang_mills/        # TNFR–Yang–Mills diagnostics (Y1–Y5; Branch B, mass gap open)
├── sdk/               # Simplified & Fluent API (8 modules)
│   └── simple.py      # Tetrad, conservation, symplectic substrate, grammar-aware dynamics
├── mathematics/       # Number theory, backends
├── constants/         # Canonical constants (mpmath 35-digit precision)
├── metrics/           # Coherence, Si, phase sync, telemetry
├── validation/        # Structural health monitoring
└── factorization/     # Spectral factorization workflow

examples/              # 120 files (tutorials 01–76, 90–107; parallel Riemann/NS/Type-Hygiene series 77–89)
tests/                 # 1,947 tests
theory/                # Theoretical derivations
benchmarks/            # 50 performance & structural-validation scripts
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [AGENTS.md](AGENTS.md) | **Primary reference** — complete TNFR theory, operators, grammar, fields |
| [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md) | U1-U6 grammar derivations from physics |
| [theory/FUNDAMENTAL_THEORY.md](theory/FUNDAMENTAL_THEORY.md) | Universal Tetrahedral Correspondence |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) | Field implementation specifications |
| [docs/STRUCTURAL_INTERFACE_THEORY.md](docs/STRUCTURAL_INTERFACE_THEORY.md) | Structural-interface programme: pipelines, fair benchmarks, validated results, limitations |
| [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) | TNFR-Riemann program |
| [theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) | TNFR-Navier–Stokes program |
| [theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md](theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md) | TNFR–Yang–Mills structural gap programme (Y1–Y5; Branch B classified) |
| [theory/GLOSSARY.md](theory/GLOSSARY.md) | Terminology and definitions |
| [examples/](examples/) | Sequential tutorials |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development guidelines |

---

## Testing

```bash
pytest                             # all tests (1,947 under tests/)
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
