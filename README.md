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
and investigate explicit nodal models. Conditional formation, maintenance and
interaction of patterns have been established for specified laws and supplied
networks. A uniquely selected fundamental law, autonomous substrate creation
and identification with physical matter remain open.

Start with the [theory reading routes](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md)
for concepts and evidence, or the [CLI and SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
for executable interfaces. This README describes the current checkout; a
published package or archived release can have a narrower feature set.

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

For operator-word studies, the CLI and SDK share one declared runner:

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

## Execution paths and mathematical scope

Choose the declared execution path; the interfaces share engine owners but do
not all execute the same model.

| Purpose | Entry point | Contract |
| --- | --- | --- |
| Operator-word studies | `StudySpec` / `run_study`, `tnfr network` | Complete requested words, grammar and live preconditions; cycles are not seconds |
| Conditional joint form/phase evolution | `RelationalExchangeModel`, `Network.step_relational(model, dt=...)` | Explicit joint Euler step on admitted fixed support with held capacity; separate from operator-word studies |
| Observe a supplied state or pattern | `diagnose_network`, `regional_form`, `source_relative_form`, `relational_pattern` | Detached reports; stored-pressure and fresh-field observations retain their different provenance |
| Check a sufficient basin or continuous transit | `Network.relational_*capture` | Read-only, support-specific theorem admission; not a generic detector, automatic controller or replacement for execution |

The [SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
owns exact signatures, preparations, supported domains and export examples.
Reports retain availability and numerical evidence; they are not full resumable
checkpoints. Research coefficient instruments do not become production solvers
merely because their finite predictions pass.

The main results have distinct scopes:

| Topic | Established contribution | Boundary and owner |
| --- | --- | --- |
| Diffusion | Dirichlet dissipation and componentwise consensus for fixed reversible pure-EPI transport with positive held capacity | Directed, forced and time-varying laws need separate hypotheses: [diffusion](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIFFUSION_STABILITY_THEOREM.md) |
| Derived regional form | Amplitudes, angles, Gram relations and source-relative responses derived from scalar EPI | Observation closure requires its supplied law; a derived angle is not automatically primitive phase: [form geometry](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/DERIVED_FORM_PHASE.md) |
| Relational patterns | A declared joint law supports local recovery, transmission and a validated formation-to-maintenance route on supplied two-ring support | Zero-form and sign-reversed controls can reach consensus instead; general formation and physical identification do not follow: [relational model](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md) |
| Composition and memory | Exact tangent reduction, nonlinear nonclosure witnesses and a derived memory correction that improves one frozen finite prediction | Full internal state remains authoritative; matched-step numerical accuracy is not a continuous ODE error certificate: [composition](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md), [memory](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_MEMORY.md) |
| Auxiliary models and applications | Explicit Hamiltonian, graph-wave, polarization and arithmetic constructions | Their added premises and supplied inputs do not establish particle emergence or a fundamental physical theory: [comparison catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#auxiliary-models-and-physical-comparisons) |

Conservation residuals are observations; a nonnegative diagnostic energy needs
its own Lyapunov proof. Passing tests, configured grammar or agreement with
arithmetic targets does not establish unrestricted stability or physical validity.

The [theory catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md)
connects each topic to its mathematical owner, engine implementation, tests and
public interface. The [portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt)
classifies primary, supporting and parked work. The
[execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
is the sole task queue: define sufficient state, justify complete laws, check
joint consistency and make a discriminating prediction. The separate P1–P5
measurement bridge requires independent observations and reserved evaluation.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/dynamics/`, `operators/` | Declared pressure/evolution laws, named transformations, grammar and execution |
| `src/tnfr/physics/`, `mathematics/`, `metrics/` | Scoped theorem tools, shared numerical algebra, observations and telemetry |
| `src/tnfr/sdk/`, `cli/` | Public networks, study recipes, detached reports and command adapters |
| `src/tnfr/config/`, `constants/` | Model defaults, numerical settings and configured policies |
| `theory/` | Thematic mathematical catalog, detailed derivations and the research plan |
| `tests/`, `examples/`, `benchmarks/` | Contract checks, executable illustrations and scoped research instruments |
| `factorization-lab/`, `primality-test/` | Arithmetic applications with explicit input and verification boundaries |

[Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md)
owns the detailed module and dispatch map. The
[example index](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md)
and [benchmark index](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md)
identify maintained entry points and their scope.

## Development and verification

```bash
python -m pytest
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The default test run is the routine engine/API gate. Research owners must be
selected explicitly; tests marked `slow` also require explicit selection. See
[TESTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) for focused suites, optional backends, slow tests, and
reproducibility checks. See [CONTRIBUTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) for contribution
requirements. Catalog coverage and the website theory menu are checked against
`theory/README.md`. After changing its owner rows or the operator registry, run
`python scripts/check_documentation.py --write-generated` to refresh their
generated documentation views before checking them.

## Documentation

| Resource | Purpose |
| --- | --- |
| [AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) | Development rules, model boundaries and agent guidance |
| [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) | Implemented package boundaries and data flow |
| [docs/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) | Technical documentation hub |
| [theory/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) | Topic catalog, reading routes and theory-to-code/test map |
| [docs/CLI_AND_SDK.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md) | Operator studies, conditional models, observations and export |
| [docs/API_CONTRACTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md) | Operator and execution contracts |
| [docs/STRUCTURAL_FIELDS_TETRAD.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md) | Field definitions and safety-policy scope |
| [examples/README.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) | Executable examples |

The published site is built from these repository sources by the documentation
workflow: [TNFR documentation](https://fermga.github.io/TNFR-Python-Engine/).

## Citation

The DOI below identifies the project across versions. Cite the version and its
[release tag](https://github.com/fermga/TNFR-Python-Engine/releases/tag/v0.0.3.7)
for the exact software snapshot.

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2026},
  version = {0.0.3.7},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

MIT licensed. See [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
