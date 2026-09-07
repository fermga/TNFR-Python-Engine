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

The graph engine's scalar EPI chart accepts a raw real value or the equivalent
uniform-real `BEPIElement` representation. Its scalar projection retains the
sign; `abs(EPI)` remains the nonnegative Banach-envelope magnitude. Genuinely
nonuniform or complex BEPI payloads keep that magnitude projection for generic
read-outs and are rejected by certificates that require one real EPI coordinate.

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
| `grad phi` | Local phase stress | Mean absolute wrapped phase difference across neighboring nodes; exact bound `pi`, with `pi/16` as the selected warning threshold |
| `K_phi` | Local wrapped phase curvature | Exact wrapped magnitude bound `pi`; `0.9*pi` is a warning margin |
| `xi_C` | Non-local correlation range | Spectral estimate scales as `1/sqrt(lambda_2)` under its documented hypotheses |

`Phi_s` uses explicit edge `length` for path geometry when available; otherwise
it retains `weight` as a compatibility fallback. EPI diffusion always reads
`weight` as conductance, so models with distinct geometry and transport should
declare both attributes.

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

The operator registry is the canonical semantic interface for named
transformations. Declared numerical solvers may advance EPI only through the
shared nodal-equation integrator from explicit `nu_f` and `DeltaNFR`, with
provenance or a residual; ad hoc state assignment is outside the engine
contract.

SDK words preserve operator order. Each SDK Reception or Resonance stage
reads one immutable all-target snapshot and commits its validated proposals
atomically; the GPU Resonance strategy reuses that same stage. These positions
have two-phase Jacobi semantics. Other operator stages retain operator-major
Gauss-Seidel semantics, so the guarantee does not make an entire mixed word
simultaneous. No GPU Reception strategy is currently registered.

THOL's `subepi_amplitude_alignment` is a variance-based EPI-amplitude
diagnostic, not canonical `C(t)` and not U5. A concrete U5 target is evaluated
by `assess_u5_parent_child_coherence(..., alpha=...)`; the hierarchy and
nonnegative `alpha` must be supplied explicitly.

### Mutation temporal evidence

Mutation keeps three related quantities separate:

| Read-out | Definition | Scope |
| --- | --- | --- |
| `predicted_depi_dt` | instantaneous `nu_f * DeltaNFR` | Nodal-equation prediction; its crossing is exposed by the legacy SDK alias `near_bifurcation` |
| `observed_depi_dt` | signed two-sample EPI secant | Evidence used by the strict, non-disableable ZHIR threshold gate |
| `d2epi_dt2` | three-sample change between adjacent secant rates | Structural-acceleration diagnostic; timestamped or legacy unit-step, and not the ZHIR gate |

Physical evidence uses timestamped `(time, EPI)` records with finite increasing
time and a fresh final EPI endpoint. If supplied, it is authoritative and does
not fall back when invalid or stale. Legacy `epi_history` and `_epi_history`
instead retain a unit-operator-step interpretation and are explicitly not
resolved in physical time. Direct Mutation requires a valid observed rate
strictly above `ZHIR_THRESHOLD_XI`, together with active capacity and any
configured minimum capacity.

When dynamic selection cannot support a proposed ZHIR from that evidence, it
substitutes Coherence (IL) before ordinary grammar enforcement and records the
requested and applied glyphs with the reason. The SDK whole-word runner
checks all target nodes before executing a word that contains Mutation and
rejects timestamped evidence that an earlier EPI-channel operator in the word
would make stale. `ZHIR_BIFURCATION_VF_THRESHOLD = 0.5` only controls branch
proposal; it is not the Mutation gate. `MutationTriggerCertificate` is an
immutable diagnostic, and `nodal_state()` reads the same evidence without
modifying the graph. Neither evaluates the prior-IL and recent-destabilizer
context required by U4b, and neither certifies execution readiness.
`TNFRNetwork.apply_evidence_gated_mutation()` is the high-level experiment
policy: it runs the requested ZHIR word only after the same preflight;
otherwise it executes a declared Mutation-free exploration word and records
the decision in `NetworkResults.mutation_workflows`. It never synthesizes EPI
history, and malformed evidence remains an error. Direct `apply_sequence()`
calls remain strict. See
[Mutation (ZHIR)](theory/STRUCTURAL_OPERATORS.md#91-mutation-zhir).

## Mathematical scope

TNFR provides executable structural models, diagnostics, and reproducible
experiments. Several correspondences are exact within stated finite-graph or
linearized hypotheses; others are measured diagnostics or open conjectures.
The current scope is centralized in:

- [Unified Grammar Rules](theory/UNIFIED_GRAMMAR_RULES.md)
- [Diagnostic and Grammar Scope](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md)
- [Minimal Structural Degrees](theory/MINIMAL_STRUCTURAL_DEGREES.md)
- [Structural Conservation Theorem](theory/STRUCTURAL_CONSERVATION_THEOREM.md)
- [Core Dynamics Research Program](theory/CORE_RESEARCH_PROGRAM.md), with its
  [diffusion stability theorem](theory/TNFR_DIFFUSION_STABILITY_THEOREM.md) and
  [scale, geometry and bridge results](theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md)

The restricted S16 executable boundary now covers both frozen endpoints and
sampled pure-EPI trajectories. The path certificate checks every nodal update
on persistent node identifiers, the explicit-Euler modal limit and a common
switching Lyapunov metric. It separates spectral and residual tolerances and
limits cumulative positive energy variation. Its mesh comparison requires an
explicit same-dynamics declaration and keeps numerical agreement separate from
a proof of convergence.

Within that common metric, a declared affine EPI reset has a finite global
disagreement gain exactly when it preserves the consensus subspace. The engine
combines a rational Frobenius upper bound computed exactly on the represented
binary64 coefficients for each passing reset with a rationally certified lower
bound for the represented diffusion decay. The flow proof separately checks
that its materialized generator preserves the consensus subspace and requires
the stronger exact identity `A 1 = 0` so every uniform EPI field is a fixed
point. Preservation of the displayed weighted mean, `h^T A = 0`, is reported
separately. Spectral rates remain estimates. A rational log/exp enclosure
decides and bounds finite and repeated hybrid words without using caller
tolerance as a theorem gate.

For bounded time-varying capacities, the certificate constructs `W`, `D`,
`B=D-W`, its quotient gap and the Lyapunov rate rationally from the effective
binary64 conductances and declared capacity bounds. It reports the conditional
exact-real theorem, availability of a positive operational float rate, ordinary
spectral diagnostics and numerical-integration verification separately; the
last remains open because no future schedule or solver path is observed.

Local Reception (EN) and Resonance (RA) are the first two catalog operators
connected to this framework. They share one centralized unweighted-neighbour
EPI blend even when transport conductance is weighted. The RA audit keeps four
layers separate: the ideal-real convex blend, the represented binary64 affine
map, the actual two-stage binary64 proposal, and the accepted identity-gated
runtime snapshot. Only neighbours that individually pass U3 participate in
RA's EPI mean, phase mean, and frequency trigger; its configured phase limit
may tighten, but cannot exceed, the canonical `pi/2` gate.

RA permits the scalar EPI to move through convex mixing while preserving its
identity: a strict negative/positive crossing is rejected, exact zero is a
neutral boundary, and an established nonempty `epi_kind` cannot change (an
absent kind may be initialized). These sign and kind conditions are independent.
The runtime also requires `0 <= RA_epi_diff <= 1`, nonnegative
`RA_vf_amplification`, and `0 <= RA_phase_coupling <= 1` before mutation. A
local frequency boost generally changes the post-RA diffusion metric
`h_i=d_i/nu_i`; the fixed post-RA flow can still be certified, while a pre/post
switching claim abstains unless the represented metrics are exactly
proportional. Any accepted nontrivial EPI change requires pure-EPI pressure
refresh before diffusion resumes. Separate rounding, clipping, identity gates,
and multichannel effects preclude a global binary64 affinity claim. Canonical
labels do not supply gains for the remaining runtime operators.

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
