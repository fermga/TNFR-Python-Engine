# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TNFR is a Python research framework for coherent patterns on graph-coupled
networks. It provides network construction, 13 registered structural operators,
U1-U6 grammar, numerical evolution and diagnostics through Python and a CLI.

Each node carries form (EPI), reorganization capacity (`nu_f`) and circular
phase. The unforced nodal relation is

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}(t).
$$

In plain text: `dEPI/dt = nu_f * DeltaNFR` — form changes at a rate given by
capacity times structural pressure. A runnable model also needs explicit
pressure, phase, capacity, support and input laws. The equation alone does not
select them. Structural time requires a declared clock; comparison with
laboratory seconds requires an independent measurement bridge.

Use the engine to execute declared operator studies, observe structural fields
and examine scoped mathematical models. Autonomous persistent patterns and
their correspondence with physical entities remain research objectives.

## Installation

Python 3.10 or later is required. Install the published package:

```bash
python -m pip install tnfr
python -m tnfr --version
```

To use the current checkout instead, run `python -m pip install -e .` from the
repository root. Core dependencies include NumPy, SciPy and NetworkX. Optional
extras enable additional tools:

```bash
python -m pip install -e ".[test,docs]"     # repository tests and documentation
python -m pip install -e ".[compute-jax]"   # optional JAX backend
python -m pip install -e ".[compute-torch]" # optional Torch numerical backend
```

[Package metadata](https://github.com/fermga/TNFR-Python-Engine/blob/main/pyproject.toml)
owns versions and dependency groups. The repository can be ahead of PyPI; use
an explicit release tag when comparing results. The Torch extra supplies a
numerical backend, without promising CUDA acceleration for every engine path.

## Quick start

```python
from tnfr.sdk import TNFR

net = TNFR.create(20, seed=42).ring()
net.evolve(steps=5, sequence="basic_activation")
print(net.results().summary())
print(net.tetrad().summary())
```

Output for the declared uniform preparation in the checked environment:

```text
C=1.000, Si=1.000, N=20, E=20, rho=0.105
Phi_s=0.0000, |grad_phi|=0.0000, |K_phi|=0.0000, xi_C=4.5201 (N=20)
```

This creates a ring with supplied `EPI=0`, `nu_f=1` and phase zero, then runs
five complete operator words. It does not demonstrate spontaneous pattern
formation. Coherence and sense index are diagnostics; a high value is not a
proof of stability. The tetrad's safety flags, available separately through
`is_safe()`, apply configured policies.

For stored-pressure observations with independent field availability, use
`diagnose_network(net)` from `tnfr.sdk`. It reads a detached graph copy and
retains unavailable-field reasons and coherence-length provenance. The
[CLI and SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
explains these reports, seed ownership and advanced execution routes.

## Command line and reproducible studies

The CLI and SDK share a declared study runner:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --steps 1 --export-spec study.json --output report.json
python -m tnfr network --spec study.json --output replay-report.json
tnfr sequences basic_activation
tnfr operators emission
```

In Python, use `StudySpec`, `run_study` and `diagnose_network` from `tnfr.sdk`.
The report retains supplied inputs, finite endpoint state and diagnostic
provenance. Cycles are operator-word passes, not elapsed physical time; a report
is not a resumable checkpoint. See the [CLI and SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
for equivalent Python examples, field availability and reproducibility scope.

## Canonical structure

The scalar engine accepts signed real EPI or its uniform-real `BEPIElement`
embedding. Scalar-only operators and diffusion reject genuinely complex or
nonuniform payloads; their magnitude is not a signed form coordinate. Detailed
state and execution boundaries belong to
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).

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
warning thresholds are configured policies. The fitted coherence length and
spectral fallback have different assumptions; reports identify the estimator
used. The tetrad does not reconstruct the complete nodal state.
Definitions, numerical boundaries and availability rules are centralized in
[Structural Fields](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md).

Thirteen registered operators act primarily on form, capacity, phase or pressure.
Their [shared contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/operators/operator_contracts.py) and
[U1-U6 grammar](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/UNIFIED_GRAMMAR_RULES.md) define the engine interface.
Grammar admission, live preconditions and trajectory stability are different
claims. Si is configured telemetry used by some controllers; that use is not a
derivation of spontaneous operator selection.

Named operators can introduce declared hybrid jumps; continuous solver spans
use the shared nodal integrator. Immutable all-target proposals and atomic
graph-owned commits have their own execution scope. Live temporal evidence,
network REMESH history and rollback boundaries are specified in the
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md),
which owns these details rather than duplicating them here.

## Mathematical scope

The repository contains useful conditional results and explicit counterexamples:

- Fixed reversible pure-EPI diffusion has a Dirichlet dissipation law and
  componentwise consensus under fixed positive capacities. Directed, forced and time-varying models
  require additional hypotheses: [diffusion theorem](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIFFUSION_STABILITY_THEOREM.md).
- Projecting out nodal state can create memory. Mean closure need not preserve
  potential or coherence length: [derived memory](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/DERIVED_EPI_MEMORY.md)
  and [scale/geometry bridge](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).
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

## Repository map

```text
src/tnfr/
├── config/          # runtime configuration and declared policy classifications
├── constants/       # canonical and operational constants
├── operators/       # operator implementations, contracts, grammar and execution
├── dynamics/        # Delta NFR computation and nodal integration
├── physics/         # tetrad, diffusion, conservation and structural diagnostics
├── metrics/         # coherence, sense index and telemetry kernels
├── core/            # service protocols, defaults and dependency container
├── services/        # orchestration facade
├── sdk/             # public network APIs, study declarations and reports
├── cli/             # command adapters, including the shared SDK study runner
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
| [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) | Working definitions, invariants and agent guidance |
| [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) | Implemented package boundaries and data flow |
| [docs/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) | Technical documentation hub |
| [theory/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) | Theory and research-program index |
| [docs/CLI_AND_SDK.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md) | Shared execution, recipes, diagnostics and export |
| [docs/API_CONTRACTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md) | Operator and execution contracts |
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
