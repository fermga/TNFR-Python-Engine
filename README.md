# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Python Engine 0.0.3.8** — mathematical research, a network engine and
reproducible computational evidence.

## Summary

### The question TNFR asks

How can interacting parts form a recognizable pattern, keep its identity while
changing, and influence other patterns? TNFR studies this question through
networks and mathematical laws. Its long-term hypothesis is that properties
of physical objects might emerge from such organization.

Think of a stadium wave: a pattern moves around the stadium while each person
stays near their seat. This illustrates how organization can belong to a group
and persist while its parts change. It is an analogy for the question, not a
derivation of physical matter.

This repository contains the Python engine, mathematical definitions, proofs
with stated assumptions, and numerical experiments. It establishes results
for specified models. Identifying those models with physical constituents
remains an open research problem.

### What a TNFR network contains

A network specifies which nodes can interact. Each node has:

- **Form (`EPI`):** its structural state, represented by a signed real number
  in the scalar engine. A whole pattern depends on the arrangement of states
  and connections, not one number alone.
- **Capacity (`nu_f`):** a nonnegative factor that scales how quickly form
  responds to structural pressure.
- **Phase (`phi` or `theta`):** a circular coordinate, like a position on a
  clock face. Positions just before and after a full turn are close.

A complete model supplies **structural pressure (`DeltaNFR`)**, the term
driving form change. It may depend on differences between neighboring states
and on declared inputs. It is not automatically pressure measured in pascals.
The unforced nodal equation is

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.
$$

In words: **the rate of change of form equals capacity times structural
pressure**. Zero capacity freezes this form row even when pressure is nonzero;
it does not establish equilibrium of all variables.

To predict a trajectory, the model must also specify how pressure is calculated,
how phase, capacity and connections behave, and which clock measures change.
Holding something fixed is an explicit assumption. The nodal equation alone
does not select these laws. Calculating pressure backward from the response
being evaluated would make the equation fit without predicting that response.

### What counts as a coherent pattern

A pattern has an identity when specified relationships can be followed over
time. Its parts need not have equal values or stop moving. For example, phases
may make a full turn around a loop while internal states continue to exchange
form. A claim of persistence must say which relationship survives, under which
law, for how long and against which disturbances.

An **NFR**, or fractal-resonant node, is a modeled node or region at a chosen
scale. Assigning that description does not prove that it forms or persists.
The research asks when a larger organization can itself be described as a node
while its smaller constituents keep existing and evolving. Grouping nodes is
not enough: their interaction must remain predictable.

Two groups can have the same averages but behave differently because their
internal arrangements differ. The current mathematics therefore retains the
internal information required by the law, including cases where opposing
phases cancel and their average angle becomes undefined. A missing average
does not mean that the underlying parts have disappeared.

The engine's **operators** are named transformations of admitted state. Its
**grammar** checks words of operators and relevant live preconditions.
Numerical solvers separately evolve specified equations. These mechanisms
support experiments; an admitted word is not a guarantee of indefinite
stability or a rule that autonomously selects the next event.

### What resonance means here

Resonance asks whether a pattern responds more strongly to some input rhythms
than others. In a specified sine-based model, a response maximum at a nonzero
frequency follows from the joint form/phase equations near a stable pattern,
for a stated input and observation. The observation matters: a different
readout need not show the same peak.

A **pulse** is a different question: can internal activity continue? Exact
periodic exchange exists for a prepared isolated pair in the admitted
zero-loss sine model. This does not explain why that loss value or initial
preparation should be selected. Nor does a response peak imply perpetual
free vibration. The [resonance foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RESONANCE_FOUNDATIONS.md)
states these distinctions and their hypotheses. The named Resonance operator
has its own execution contract.

### What the research establishes

The repository contains several kinds of reusable results:

Some models use a mathematical **storage** quantity to account for exchanges
and losses. Identifying it with measured physical energy needs a separate
measurement model.

- **Relaxation and recovery:** convergence of pure-form diffusion and recovery
  of joint form/phase patterns under their stated support and law assumptions.
- **Conditional formation:** selected preparations can develop phase winding
  and reach a protected region. Native and sine models have separate results;
  finite formation or retention does not establish indefinite maintenance.
- **Identity with internal motion:** specified conservative sine families can
  preserve a collective arrangement while constituents remain active. An
  exact periodic orbit and resistance to disturbances are separate properties.
- **Interaction, scale and memory:** collective descriptions can retain the
  information needed to evolve their constituents. Eliminating hidden nodes
  can produce an interaction with memory of their initial state and inputs.
- **Obstructions and counterexamples:** equal averages, available storage or
  matching local responses need not produce the same future. Some proposed
  formations are excluded by symmetry or storage constraints.

The [theory-to-execution map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#theory-to-execution) routes models and responsibilities
to their mathematical owners, implementation and representative checks. Proofs show
what follows from premises; tests check code; finite experiments establish
evidence for their declared cases. Physical identification needs another step.

### How pulse, resonance and fractality fit together

Internal motion can change how a group responds to its surroundings. A local
signal may fade because activity has moved into neighboring parts, even when
the whole system conserves its structural storage. If those parts are hidden,
their influence can remain as memory in the observed description.

This connects to the question behind fractality: **what must a larger node
retain about its smaller constituents to inherit their dynamics?** Exact
descriptions answer parts of that question on supplied networks. They do not
yet explain the autonomous formation of every scale or a universal fractal
structure. The [connection map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#pulse-resonance-scale-connections)
identifies results whose assumptions allow them to be combined.

### The research direction

The main route is **sufficient information -> justified interaction laws ->
collective organization -> independent observation**. We first specify the
state and complete laws, check their consistency, then seek a prediction,
equivalence or obstruction that distinguishes competing explanations.

The current focus is how internal organization affects interaction: when can
two patterns look the same from outside yet exchange form differently because
of their internal state and surroundings? The
[execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the precise task, acceptance conditions and next step. The
[strategy](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md) explains the rationale; the
[portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt) classifies supporting branches.

Why a particular law, starting state or set of connections should arise on its
own remains open. Identifying the resulting patterns physically also needs
independent evidence. Conditional results help identify which assumptions
matter; they do not make those assumptions inevitable.

### How this could connect to physical reality

A physical property could depend on a whole pattern of form, capacity, phase
and connections. EPI need not equal a sensor reading such as voltage or
temperature. Any proposed correspondence still needs independently justified
preparation, measurement, clock and uncertainty models.

Physical evaluation requires separate calibration and reserved data, predictions
fixed before evaluation, and comparison with suitable alternatives. The scope
is public terrestrial data and ordinary laboratory-scale protocols accessible
with a workstation. No dataset has yet passed the complete physical admission.
Deriving particles, spin and quantum behavior from TNFR, and explaining the
initial network's origin, remain open goals. The immediate value is a framework
for testing precisely which mechanisms work, which information they require
and where they fail.

## Installation

Requires Python 3.10 or later. Install the package:

```bash
python -m pip install tnfr
python -m tnfr --version
```

For the checked-out source, run `python -m pip install -e .` from the repository
root. NumPy, SciPy and NetworkX are core dependencies. Optional tools are grouped
in [package metadata](https://github.com/fermga/TNFR-Python-Engine/blob/main/pyproject.toml):

```bash
python -m pip install -e ".[test,docs]"     # tests and documentation
python -m pip install -e ".[compute-jax]"   # optional JAX backend
python -m pip install -e ".[compute-torch]" # optional Torch numerical backend
```

Published packages describe their release; research on a checkout can include
later changes. Record the source revision and effective backend for a study.
Backend availability does not imply acceleration on every execution path.

## Quick start

```python
from tnfr.sdk import TNFR

net = TNFR.create(20, seed=42).ring()
net.evolve(steps=5, sequence="basic_activation")
print(net.results().summary())
print(net.tetrad().summary())
```

Output for this uniform preparation in the checked environment:

```text
C=1.000, Si=1.000, N=20, E=20, rho=0.105
Phi_s=0.0000, |grad_phi|=0.0000, |K_phi|=0.0000, xi_C=4.5201 (N=20)
```

This supplies a ring with `EPI=0`, `nu_f=1` and phase zero, then executes five
complete operator words. It illustrates a uniform baseline and the interface;
it does not demonstrate formation. `C` and `Si` are diagnostics. Estimator
availability and safety-policy flags have their own
[observation contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md).
`results()` also records a balance-tracker sample; use `diagnose_network()`
when a detached stored-state observation is needed.

The CLI and SDK share one operator-study runner:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --steps 1 --export-spec study.json --output report.json
python -m tnfr network --spec study.json --output replay-report.json
tnfr sequences basic_activation
tnfr operators emission
```

Use `StudySpec`, `run_study` and `diagnose_network` from `tnfr.sdk` for the
corresponding Python workflow. Cycles count operator words, not seconds.
The study runner sets topology and execution seeds; `TNFR.create(..., seed=...)`
sets the topology seed. A recipe or diagnostic report is not a complete
resumable checkpoint. The [CLI/SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md) owns full usage.

## Choose an execution path

The models below have different complete laws. Sharing variables or a storage
formula does not transfer a theorem between them.

| Task | Interface or owner | Contract |
| --- | --- | --- |
| Run operator words | `TNFR`, `StudySpec`, `run_study`, `tnfr network` | Registered transformations, grammar, live preconditions and declared hybrid events |
| Evolve native relational form and phase | `RelationalExchangeModel`, `Network.step_relational` | Resultant-direction pressure and a joint phase law; supplied support, held capacity and admitted phase domain |
| Assess the normalized-sine model | [Shared assessment owners](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md#normalized-sine-proof-adapters) and [usage guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/relational/SINE_COMPARISON_AND_INFERENCE.md) | A distinct form/phase law using neighboring sine differences; scoped observations, certificates and continuous enclosures |
| Read stored network state | `diagnose_network` | Detached diagnostics with explicit availability; stored pressure is not refreshed |
| Study regions, contacts and retained responses | [Regional workflows](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md) | Declared regions, hypothetical support changes, preparation-specific bounds and evidence |

Supplying a `reference_model` to a sine assessment does not make
`step_relational` execute the sine law. A hypothetical attachment report does
not add a live edge. Continuous theorems, numerical steps and validated
enclosures provide different guarantees. The [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
define admission, mutation scope, provenance and reporting.

## Observe the network

The **structural tetrad** is a shared set of diagnostics. Like a dashboard,
it reveals selected features without describing the complete internal state
or predicting its future by itself.

| Tetrad field | What it measures | Interpretation |
| --- | --- | --- |
| `Phi_s` | Nonlocal aggregation of structural pressure | Pressure contributions weighted by inverse squared path distance |
| `abs(grad phi)` | Local phase separation | Mean absolute wrapped difference from neighboring phases; bounded by `pi` |
| `K_phi` | Circular phase curvature | Displacement relative to a neighbor phase resultant; bounded by `pi` where defined |
| `xi_C` | Static coherence correlation range | Coherence-product fit, with an explicitly identified spectral fallback |

Field path geometry reads edge `length`, falling back to `weight`; diffusion
uses `weight` as conductance. Undefined curvature, absent temporal evidence and
unavailable estimates remain explicit. Warning thresholds are configured
policies. The [structural field guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md) owns
definitions, interpretation and numerical limits.

## Read the research

| Need | Maintained owner |
| --- | --- |
| Find a definition, proof or model | [Theory reading routes](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#choose-a-question) |
| Locate its implementation and tests | [Theory-to-execution map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#theory-to-execution) |
| Distinguish premises, results, diagnostics and hypotheses | [Glossary](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/GLOSSARY.md) |
| Understand the research rationale | [Strategy](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md) |
| Resume the active task | [Execution plan and checkpoint](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-checkpoint) |

Auxiliary Hamiltonian, graph-wave, geometric and arithmetic studies keep their
own premises. The catalog identifies their scope; their presence does not
make them part of the same generative model. Evaluated evidence, negative
results and retirement provenance remain in the [research archive](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/archive/README.md).

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/` | Shared dynamics, operators, observations, numerical tools, CLI and SDK; boundaries in [Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) |
| `theory/` | Definitions, derivations, research rationale and the single execution plan |
| `docs/` | Usage and execution contracts, organized by the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) |
| `tests/` | Contract checks and explicitly selected research tests; selection in [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) |
| `examples/`, `benchmarks/` | [Runnable illustrations](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) and [scoped instruments](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md) |
| `docs/assets/`, `artifacts/` | Published evidence and local protocols, source archives and responses; missing local evidence remains unavailable |
| [applications/](https://github.com/fermga/TNFR-Python-Engine/blob/main/applications/README.md) | Optional arithmetic applications with separate verification boundaries |

## Contribute and verify

[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) defines contributor and agent responsibilities.
[Contributing](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) and [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) own the development
workflow, dependencies and test selection. Preserve unrelated changes, reuse
shared implementations, and update a changed claim with its responsible
contract and checks.

With the test and documentation extras installed:

```bash
python -m pytest
python scripts/check_documentation.py
python scripts/verify_internal_references.py --ci
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The default tests cover the routine engine and public interfaces. Select a
research owner explicitly when changing its model or claim; the routine gate
does not replay every retained campaign. Edit maintained documents, not
generated `build/docs-source/` or `site/` files. Documentation catalogs supply
the checked menus of the [published site](https://fermga.github.io/TNFR-Python-Engine/).

## Citation and license

Cite the exact source snapshot used. [CITATION.cff](https://github.com/fermga/TNFR-Python-Engine/blob/main/CITATION.cff) owns software
citation metadata; the project DOI is
[10.5281/zenodo.17602860](https://doi.org/10.5281/zenodo.17602860).
MIT licensed; see [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
