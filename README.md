# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TNFR is a Python framework for coherent-pattern analysis on graph-coupled
networks. Every node carries form (EPI), structural frequency (`nu_f`, in
`Hz_str`), and phase. Its evolution is organized by the nodal equation

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}(t).
$$

The repository implements 13 canonical structural operators, grammar U1-U6,
network telemetry, the structural-field tetrad, and research programs built on
those primitives. [AGENTS.md](AGENTS.md) is the canonical synthesized reference.
Mathematical scope and counterexamples are stated explicitly in the linked
theory documents.

```bash
pip install tnfr
```

## Quick start

```python
from tnfr.sdk import TNFR

net = TNFR.create(20).ring().evolve(5)
print(net.results().summary())
print(net.tetrad().summary())
print(net.tetrad().is_safe())
```

Current deterministic output for this uniform initial state:

```text
C=1.000, Si=1.000, N=20, E=20, rho=0.105
Phi_s=0.0000, |grad_phi|=0.0000, |K_phi|=0.0000, xi_C=4.5201 (N=20)
{'phi_s_safe': True, 'grad_phi_safe': True, 'k_phi_safe': True, 'xi_c_safe': True, 'overall': True}
```

The same network exposes the principal read-outs:

```python
net.conservation()
net.symplectic_substrate()
net.rhythm()
net.resonance()
net.telemetry()
net.audit_operators()
analysis = TNFR.analyze(net)
```

Grammar-aware evolution validates operator composition before applying it:

```python
net.evolve_grammar_aware(steps=10)
```

## Canonical structure

The public structural-field tetrad is `(Phi_s, |grad phi|, K_phi, xi_C)`.

| Field | Role | Exact or scoped statement |
| --- | --- | --- |
| `Phi_s` | Global pressure aggregation | General magnitude depends on pressure and graph geometry; `pi/4` and `pi/2` are selected warning policies |
| Phase-gradient magnitude (`∇φ` norm) | Local phase stress | Mean absolute wrapped phase difference across neighboring nodes; exact bound `pi`, with `pi/16` as the selected warning threshold |
| `K_phi` | Local wrapped phase curvature | Exact wrapped magnitude bound `pi`; `0.9*pi` is a warning margin |
| `xi_C` | Non-local correlation range | Spectral estimate scales as `1/sqrt(lambda_2)` under its documented hypotheses |

For small phase spread on a consistent branch and matching weight conventions,
`K_phi` agrees with the random-walk Laplacian applied to phase. The EPI channel
of `Delta NFR` is exact graph diffusion. These statements do not make every
pressure channel a linear Laplacian or make the tetrad a complete state
reconstruction theorem. See
[Structural Fields](docs/STRUCTURAL_FIELDS_TETRAD.md) and
[Minimal Structural Degrees](theory/MINIMAL_STRUCTURAL_DEGREES.md).

Operators modify four nodal channels:

- capacity `nu_f`: Silence, Expansion, Contraction;
- pressure `Delta NFR`: Coherence, Dissonance, Self-organization, Transition;
- phase: Coupling, Mutation;
- form EPI: Emission, Reception, Resonance, Recursivity.

The authoritative contracts live in
[`operator_contracts.py`](src/tnfr/operators/operator_contracts.py). Grammar
classifications are derived in
[`physics_derivation.py`](src/tnfr/config/physics_derivation.py), materialized in
[`grammar_canon.py`](src/tnfr/operators/grammar_canon.py), and exposed through
[`grammar.py`](src/tnfr/operators/grammar.py).

## Mathematical scope

TNFR provides executable structural models, diagnostics, and reproducible
experiments. Several correspondences are exact within stated finite-graph or
linearized hypotheses; others are measured diagnostics or open conjectures.
The current scope is centralized in:

- [Unified Grammar Rules](theory/UNIFIED_GRAMMAR_RULES.md)
- [Diagnostic and Grammar Scope](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
- [Minimal Structural Degrees](theory/MINIMAL_STRUCTURAL_DEGREES.md)
- [Structural Conservation Theorem](theory/STRUCTURAL_CONSERVATION_THEOREM.md)

The Riemann, Navier-Stokes, Yang-Mills, P-vs-NP, BSD, and Hodge programs remain
open research programs. They do not claim solutions to the corresponding
classical problems. Their current status is indexed in
[the theory hub](theory/README.md).

## Installation

```bash
pip install tnfr
pip install -e ".[dev-minimal]"   # local development
pip install -e ".[test-all]"      # complete test tooling
pip install -e ".[compute-jax]"   # optional JAX backend
pip install -e ".[compute-torch]" # optional Torch numerical backend
pip install -e ".[docs]"          # documentation build
```

The Torch extra provides a supported numerical backend. TNFR does not currently
ship a dedicated `TNFRGPUEngine` or promise CUDA speedups.

## Repository map

```text
src/tnfr/
├── config/          # runtime configuration and physics-derived classifications
├── constants/       # canonical and operational constants
├── operators/       # operator implementations, contracts, grammar and execution
├── dynamics/        # Delta NFR computation and nodal integration
├── physics/         # tetrad, diffusion, conservation and structural diagnostics
├── metrics/         # coherence, sense index and telemetry kernels
├── core/            # service protocols, defaults and dependency container
├── services/        # orchestration facade
├── sdk/             # simple and fluent public APIs
├── engines/         # optimization and computation services
├── mathematics/     # numerical backends and arithmetic structures
└── research areas   # riemann, navier_stokes, yang_mills and related modules
```

Executable demonstrations are grouped into ten thematic folders under
[`examples/`](examples/README.md). The full architecture and source-of-truth map
are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Development and verification

```bash
python -m pytest
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The configured default test run excludes tests marked `slow`. See
[TESTING.md](TESTING.md) for focused suites, optional backends, slow tests, and
reproducibility checks. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution
requirements.

## Documentation

| Resource | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Canonical synthesized TNFR reference and agent doctrine |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Implemented package boundaries and data flow |
| [docs/README.md](docs/README.md) | Technical documentation hub |
| [theory/README.md](theory/README.md) | Theory and research-program index |
| [docs/API_CONTRACTS.md](docs/API_CONTRACTS.md) | Operator contract reference |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md) | Field definitions and safety-policy scope |
| [examples/README.md](examples/README.md) | Executable examples |

The published site is built from these repository sources by the documentation
workflow: [TNFR documentation](https://fermga.github.io/TNFR-Python-Engine/).

## Citation

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2026},
  version = {0.0.3.5},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

MIT licensed. See [LICENSE.md](LICENSE.md).
