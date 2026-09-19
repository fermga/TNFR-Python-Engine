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

In plain text: `dEPI/dt = nu_f * DeltaNFR`.

The repository implements 13 canonical structural operators, grammar U1-U6,
network telemetry, the structural-field tetrad, and research programs built on
those primitives. [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) is the canonical synthesized reference.
Mathematical scope and counterexamples are stated explicitly in the linked
theory documents.

The graph engine's scalar EPI chart accepts a raw real value or the equivalent
uniform-real `BEPIElement` representation. Its scalar projection retains the
sign; `abs(EPI)` remains the nonnegative Banach-envelope magnitude. Genuinely
nonuniform or complex BEPI payloads keep that magnitude projection for generic
read-outs and are rejected by canonical glyphs that require a real scalar EPI
coordinate, pure-EPI diffusion and scalar certificates.

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

The nodal identity organizes the model; it does not uniquely supply the phase,
capacity, support or pressure laws. Each experiment must declare those inputs
and distinguish continuous flow from named operator jumps.

| Field | Meaning | Scope |
| --- | --- | --- |
| `Phi_s` | Nonlocal structural pressure aggregation | Source sum weighted by inverse squared graph distance; magnitude depends on graph and pressure |
| `abs(grad phi)` | Local phase desynchronization | Mean absolute wrapped neighbor phase difference; bounded by `pi` |
| `K_phi` | Circular phase curvature | Wrapped displacement from the neighbor resultant; bounded by `pi` where defined |
| `xi_C` | Static coherence correlation range | Coherence-product fit with explicitly identified spectral fallback |

Explicit edge `length` determines field path geometry; `weight` is a compatibility
fallback. Diffusion uses `weight` as conductance. The phase-wrap bounds are exact;
warning thresholds are configured policies. The xi fallback reads the first
spectral value above `1e-9`, equaling `1/sqrt(lambda_2)` only under its stated
spectral conditions. The tetrad does not reconstruct the complete nodal state.
Definitions, numerical boundaries and availability rules are centralized in
[Structural Fields](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md).

Thirteen registered operators act primarily on form, capacity, phase or pressure.
Their [shared contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/operators/operator_contracts.py) and
[U1-U6 grammar](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/UNIFIED_GRAMMAR_RULES.md) define the engine interface.
Grammar admission, live preconditions and trajectory stability are different
claims. Si is configured telemetry used by some controllers; that use is not a
derivation of spontaneous operator selection.

### Mutation temporal evidence

Mutation needs live signed EPI-change evidence and grammar context. An
instantaneous nodal-product prediction does not substitute for the observed
secant. Details belong to [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md) and
[operator semantics](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md).

### Operator-event time

Named events are declared hybrid jumps. Continuous solver spans use the shared
nodal integrator and explicit capacity/pressure; accumulated change includes
both flow and jumps. Shared all-target stages validate immutable proposals and
commit graph-owned state atomically. Their target-order scope does not establish
relabeling symmetry or future stability. Node-level Recursivity is advisory;
the separate network REMESH operation mixes delayed EPI. The execution and
certificate boundaries are specified once in [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).

## Mathematical scope

The repository contains useful conditional results and explicit counterexamples:

- Fixed reversible pure-EPI diffusion has a Dirichlet dissipation law and
  componentwise consensus under positive capacity. Directed, forced and time-varying models
  require additional hypotheses: [diffusion theorem](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIFFUSION_STABILITY_THEOREM.md).
- Projecting out nodal state can create memory. Mean closure need not preserve
  potential or coherence length: [derived memory](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/DERIVED_EPI_MEMORY.md)
  and [scale/geometry bridge](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).
- The exact P5 reflection quotient removes a discrete reflection ambiguity;
  it retains five continuous dimensions. Its hidden-form controls distinguish
  xi fits from the spectral fallback. This is a restricted model result.
- A prescribed phase motion can produce a conditional periodic form response;
  autonomous generation of that motion remains open:
  [phase/form foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md).
- The symplectic substrate, graph wave, polarization and arithmetic models are
  specified auxiliary constructions. Their identities do not establish that
  all engine trajectories obey them or that particles have emerged:
  [variational scope](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_VARIATIONAL_PRINCIPLE.md) and
  [regime comparisons](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md).

Conservation residuals are measured quantities; a nonnegative diagnostic energy
is not automatically a Lyapunov function. Passing finite tests or reproducing
arithmetic targets does not prove global stability, a Millennium conjecture or
a physical theory of emergence.

The [theory index](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) distinguishes current references,
conditional results, finite evidence and historical work. The
[research portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt) classifies the branches;
the [execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
is the sole active queue. The main objective remains a predictive generative
account of coherent patterns, with a separate reserved-data measurement bridge.

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
[`examples/`](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md). The full architecture and source-of-truth map
are documented in [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md).

## Development and verification

```bash
python -m pytest
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The configured default test run excludes tests marked `slow`. See
[TESTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) for focused suites, optional backends, slow tests, and
reproducibility checks. See [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) for contribution
requirements.

## Documentation

| Resource | Purpose |
| --- | --- |
| [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) | Canonical synthesized TNFR reference and agent doctrine |
| [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) | Implemented package boundaries and data flow |
| [docs/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) | Technical documentation hub |
| [theory/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) | Theory and research-program index |
| [docs/API_CONTRACTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md) | Operator contract reference |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md) | Field definitions and safety-policy scope |
| [examples/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) | Executable examples |

The published site is built from these repository sources by the documentation
workflow: [TNFR documentation](https://fermga.github.io/TNFR-Python-Engine/).

## Citation

The DOI below identifies the project across versions. Cite the version and its
[release tag](https://github.com/fermga/TNFR-Python-Engine/releases/tag/v0.0.3.6)
for the exact software snapshot.

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2026},
  version = {0.0.3.6},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

MIT licensed. See [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
