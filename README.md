# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Summary

### The question TNFR asks

TNFR investigates how many interacting parts can produce a recognizable pattern
that forms, survives disturbances and influences other patterns. Its long-term
question is whether properties of physical objects could emerge from such
patterns, rather than having to put those properties into the model at the start.

Think of a stadium wave. The wave travels, although the spectators stay near
their seats. What persists is an organized pattern of changes across many people.
This is a useful picture of emergence: a larger structure arises from local
activity. TNFR develops mathematical rules for investigating that kind of
organization. The stadium example is an analogy, not a physical derivation.

This repository contains both the research and the Python engine used to test
it. It includes equations, proofs with stated assumptions, numerical experiments,
and tools for building and observing networks. The engine implements and tests
declared models; the broader proposal that physical constituents arise from TNFR
remains a research hypothesis.

### What a TNFR network contains

A network is a collection of nodes and connections. A connection says which
parts can influence one another. These connections do not have to represent
physical distances or wires. Their meaning belongs to the model being studied.

Each node carries three basic coordinates:

- **Form, called EPI:** its structural state. In the main scalar engine this is
  a signed number. The full pattern is described by how those values are
  arranged across the network, together with the other coordinates.
- **Capacity, written `nu_f`:** how strongly a node can change its form in
  response to a given structural pressure. It is nonnegative.
- **Phase, written `phi` or `theta`:** a circular coordinate, like a position
  on a clock face. It describes relative alignment between nodes. Phases just
  before and after the end of a turn are close, not far apart.

The model also specifies **structural pressure**, `DeltaNFR`: a driving term
calculated from the state and its relationships. For example, differences
between neighboring forms can contribute to pressure. A complete model must
say exactly how pressure is calculated and which other channels contribute.
It is not automatically mechanical pressure measured in pascals.

The central equation is

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.
$$

Read it as: **the rate of change of form equals capacity times structural
pressure**. Without an additional source term, zero capacity prevents form from
changing even when pressure is present. That alone does not mean the whole
network is in equilibrium.

The equation organizes the model, but a simulation needs more rules: how phase
and capacity evolve, whether connections change, what inputs exist and which
clock measures change. Holding a quantity fixed is also an explicit choice.
The research asks which rules can be justified from stated structural principles
and which remain independent assumptions. It does not obtain pressure by working
backward from the answer that an experiment was supposed to predict.

### What counts as a coherent pattern

A coherent pattern has organized relationships that can be followed over time.
It need not mean that every node has the same value or phase. One possible
identity is an arrangement in which phase advances around a loop. A useful
persistence claim must say what defines that identity, which disturbances it
survives and under which evolution rule.

TNFR calls a region carrying such organization a **fractal-resonant node**, or
**NFR**. The research explores whether larger patterns can be understood in
terms of interacting smaller patterns. Treating a whole region as a new node
requires checking what information is retained and whether its dynamics can
really be predicted at that larger scale. Nesting regions does not by itself
prove a universal fractal structure.

The engine offers two complementary ways to investigate change. Named
**operators** carry out specified transformations, such as adding form, coupling
nodes or changing capacity. Its **grammar** checks which sequences and live
states are admitted. Separately, numerical solvers calculate how a stated
equation changes the network over time. A valid sequence is an execution
contract; it is not a proof that the network will remain stable forever.

To observe a network, TNFR uses quantities such as pressure, phase differences
and the range of coherence. Four shared observations form the **structural
tetrad**. Think of them as instruments on a dashboard: they reveal useful
features without describing every internal detail. A high coherence score
alone does not establish that a pattern has formed or will persist.

### What the research establishes

The results connect several parts of this picture:

1. **Relaxation has a firm mathematical reference.** With only form diffusion
   active, fixed connection strengths that are nonnegative and equal in both
   directions, and positive fixed capacities, form differences decay toward
   agreement inside each group connected by positive-strength links. This gives
   a precisely understood starting point for studying richer dynamics.
2. **Form and phase can support a maintained pattern under a specified joint
   law.** In the relational model, differences between neighboring phases drive
   changes in form, while form differences affect phase. On a supplied network
   of two five-node rings joined by two connections, mathematical bounds
   verified by computation establish a route from a particular preparation
   into a protected phase pattern. A theorem then establishes its continued
   maintenance under that law. The starting network, capacities and model
   assumptions are part of this result.
3. **Patterns can recover and transmit responses in admitted settings.** There
   are local recovery results and finite tests of interaction between prepared
   regions. These study the mechanism and its limits, rather than assuming that
   every network or disturbance behaves the same way.
4. **Hidden detail matters when we simplify.** Two preparations can have the
   same observed averages and current rates yet develop differently because
   their internal arrangements differ. Eliminating hidden variables
   mathematically produces a memory term: the simplified description retains
   an effect of internal state and past evolution. One numerical test, with
   its conditions fixed before evaluating the result, found that a derived
   memory correction improved prediction over two simpler controls. That result
   concerns the declared preparation and numerical comparison; it does not
   provide a complete replacement for the full network state.
5. **A new connection can change more than its endpoints suggest.** The engine
   can compare separate components with the same components joined by a supplied
   connection. Equal values at the endpoints do not guarantee unchanged local
   rates: each endpoint's surrounding relationships also matter.
6. **Changing connections has a budget.** The joint model assigns storage to
   differences along connections. Adding a connection without changing the
   nodes cannot lower this storage. If we additionally require no supplied
   work, the endpoints must agree in form and phase. Even then, nothing in the
   budget forces a connection to appear. Exchanging an existing connection for
   another can lower storage instead. For two prepared five-node rings, a
   specified relocation preserves their phase patterns and leaves both in a
   proved recovery region. The result also admits sufficiently small changes
   to the preparation. It concerns one supplied change followed by the declared
   dynamics, not an automatically chosen sequence of changes.

Negative controls are useful parts of these results. For example, two
preparations with the same model-defined initial energy can reach different
final patterns. They show why one convenient number cannot stand in for the
complete state.
The [theory catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md)
connects each result to its assumptions, derivation, implementation and tests.

### The research direction and the next question

The main route is to specify what information a prediction needs, justify a
complete set of rules, check that the rules work together, and predict something
not used to choose those rules. Reusable results are integrated into shared tools
and the engine where their scope supports it, with explicit conditions for use.

**The immediate question is what can select a change of connections and its
timing.** We now have a conditional example in which the change respects both
the storage budget and subsequent pattern recovery. The next study examines
which choices the full nodal state, symmetries and clock actually constrain,
and which would require an additional rule. It will not infer that a permitted
event must happen or use node labels to break an unexplained tie.

The budget alone does not select an event. Keeping a loop at the instant of
a change also does not, by itself, guarantee later recovery. The research
separates these obligations so that an assumed event schedule cannot be
mistaken for a mechanism generated by the model.

This step matters to the larger objective: explaining how coherent structures
can compose and interact through their own dynamics, with fewer unexplained
choices supplied from outside. The
[execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns that next task; the
[research portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt)
explains the roles of the supporting studies.

### How this could connect to physical reality

The long-term aim is a predictive account of physical properties emerging from
interacting TNFR patterns. EPI does not have to be identified directly with a
sensor reading such as voltage or temperature. A measurable property could,
in principle, depend on a whole pattern of form, phase, capacity and connections.
Such a correspondence still needs its own justification and evidence.

The physical-testing route requires a clear observation and clock model,
separate data for calibration (choosing model settings) and evaluation, and
predictions compared with suitable alternatives without adjusting the model
after seeing the reserved answer.
Planned evidence uses accessible terrestrial observations or laboratory-scale
systems. Full physical identification is open: the results do not yet establish
particles, spin, quantum theory or the origin of the initial network.

The practical value of the project is a common place to ask these questions
precisely: which behavior follows from a model, which information it needs,
where it fails, and what observation could distinguish it from another model.

## Installation

Requires Python 3.10 or later. Install the package:

```bash
python -m pip install tnfr
python -m tnfr --version
```

For development, run `python -m pip install -e .` from the repository root.
Core dependencies include NumPy, SciPy and NetworkX. Optional tools are grouped
in [package metadata](https://github.com/fermga/TNFR-Python-Engine/blob/main/pyproject.toml):

```bash
python -m pip install -e ".[test,docs]"     # tests and documentation
python -m pip install -e ".[compute-jax]"   # optional JAX backend
python -m pip install -e ".[compute-torch]" # optional Torch numerical backend
```

Backend availability does not imply acceleration on every execution path.
Record the installed source and effective backend when comparing results.

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

This creates a ring with supplied `EPI=0`, `nu_f=1` and phase zero, then executes
five complete operator words. It illustrates the interface and a uniform
baseline; it is not the formation experiment described in the summary.
`C` and `Si` are configured diagnostics. Field availability, estimator provenance
and the separate safety-policy flags are explained in the
[CLI and SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md).

For a reproducible operator study, the CLI and SDK share one runner:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --steps 1 --export-spec study.json --output report.json
python -m tnfr network --spec study.json --output replay-report.json
tnfr sequences basic_activation
tnfr operators emission
```

Use `StudySpec`, `run_study` and `diagnose_network` from `tnfr.sdk` for the
corresponding Python workflow. Cycles count operator words, not seconds.
A recipe describes preparation and execution; a diagnostic report is not a
complete resumable checkpoint. The study runner sets both topology and execution
seeds; direct `TNFR.create(..., seed=...)` sets the topology seed.

## Choose an execution path

| Task | Public interface | Scope |
| --- | --- | --- |
| Build networks and run operator words | `TNFR`, `StudySpec`, `run_study`, `tnfr network` | Shared registered operators, grammar and live preconditions |
| Evolve joint form and phase | `RelationalExchangeModel`, `Network.step_relational` | Opt-in law with declared support, held capacity and phase-domain admission |
| Observe stored state | `diagnose_network` | Detached observations with independent availability; pressure is not refreshed |
| Observe regions and their relations | `regional_form`, `source_relative_form`, `relational_pattern` | Supplied regions and references; no automatic closed dynamics for the reduced state |
| Compare an attachment | `relational_attachment` | Fresh separate/joined fields for a hypothetical connection; does not change live support |
| Compare a bridge relocation | `relational_relocation` | Supplied atomic bridge exchange preserving internal support; passive budget and recovery are separate obligations |
| Apply a capture or transit theorem | `Network.relational_*capture` | Read-only sufficient certificates with support-specific hypotheses |

The [regional and relational guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md)
provides preparations and examples for joint dynamics. The
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
define admission, numerical behavior, atomic stages and reporting. These routes
share owners while retaining distinct mathematical contracts.

## Observe the network

| Tetrad field | What it measures | Interpretation |
| --- | --- | --- |
| `Phi_s` | Nonlocal aggregation of structural pressure | Pressure contributions weighted by inverse squared path distance |
| `abs(grad phi)` | Local phase separation | Mean absolute wrapped difference from neighboring phases; bounded by `pi` |
| `K_phi` | Circular phase curvature | Displacement relative to a neighbor phase resultant; bounded by `pi` where defined |
| `xi_C` | Static coherence correlation range | Coherence-product fit, with an explicitly identified spectral fallback |

Field path geometry reads edge `length`, falling back to `weight`; diffusion
reads `weight` as conductance. These are distinct roles. Undefined curvature,
missing temporal evidence and an unavailable estimate must remain visible.
Warning thresholds are configured policies. Definitions and interpretation
belong to the [structural field guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md).

## Read the research

| Question | Mathematical owner |
| --- | --- |
| What state and complete laws are needed? | [Foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md), [parameter definitions](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md), [pressure premises](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/PRESSURE_CONSTITUTIVE_SCOPE.md) |
| When does form diffusion relax? | [Diffusion and stability](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIFFUSION_STABILITY_THEOREM.md) |
| How can form and phase form and maintain a pattern? | [Relational law, recovery and validated formation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit) |
| What information is needed to combine regions? | [Composition and attachment](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md) |
| What constrains a change of connections? | [Event storage, passivity and selection limits](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#support-event-premise-admission) |
| Can a pattern survive a relocated connection? | [Passive relocation and recovery](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation) |
| How does hidden state produce memory? | [Derived pattern memory](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_MEMORY.md) |
| Which terms are assumptions, results or observations? | [Classified glossary](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/GLOSSARY.md) |
| How will physical claims be tested? | [Measurement and reserved evaluation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#supporting-measurement-bridge) |

The [theory catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md)
is the complete map of derivations, engine implementations and representative
tests. It also locates auxiliary Hamiltonian, graph-wave, geometric and
arithmetic studies. Each contributes within its own assumptions; their presence
does not make them interchangeable with the main generative model.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/dynamics/`, `operators/` | Pressure and evolution laws, named transformations, grammar and shared execution |
| `src/tnfr/physics/`, `mathematics/`, `metrics/` | Theorem tools, numerical algebra, observations and diagnostics |
| `src/tnfr/sdk/`, `cli/` | Public networks, study recipes, reports and command adapters |
| `src/tnfr/config/`, `constants/`, `utils/` | Shared configuration, numerical settings, validation helpers and I/O |
| `theory/` | Mathematical definitions, derivations, research strategy and the execution plan |
| `docs/` | Usage guides, execution contracts and documentation navigation |
| `tests/`, `examples/`, `benchmarks/` | Contract checks, executable illustrations and scoped measurement/research instruments |
| [applications/](https://github.com/fermga/TNFR-Python-Engine/blob/main/applications/README.md) | Optional arithmetic applications with explicit verification boundaries |

[Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md)
owns the detailed module map. Start runnable work from the
[example index](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md)
or [benchmark index](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md).

## Contribute and verify

[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)
defines contributor and agent instructions: preserve scientific scope, reuse
shared owners and verify the affected contracts. The
[contribution guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md)
and [testing guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md)
explain the development workflow.

```bash
python -m pytest
python scripts/check_documentation.py
python scripts/verify_internal_references.py --ci
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The default test selection checks the routine engine and public interfaces.
Select the relevant research owner explicitly when changing its model or claim.
Tests check implementations and finite cases; mathematical proofs and physical
evidence have separate requirements.

The [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md)
assigns each guide its responsibility. Its catalog and the theory catalog
supply the checked website menus. The
[published documentation](https://fermga.github.io/TNFR-Python-Engine/)
is built from these repository sources.

## Citation and license

For reproducibility, cite the software snapshot used. The project DOI is
[10.5281/zenodo.17602860](https://doi.org/10.5281/zenodo.17602860);
[CITATION.cff](https://github.com/fermga/TNFR-Python-Engine/blob/main/CITATION.cff)
contains the citation metadata.

```bibtex
@software{tnfr_python_engine,
  author = {Martinez Gamo, F. F.},
  title = {TNFR-Python-Engine: Resonant Fractal Nature Theory Implementation},
  year = {2026},
  version = {0.0.3.8},
  doi = {10.5281/zenodo.17602860},
  url = {https://github.com/fermga/TNFR-Python-Engine}
}
```

MIT licensed. See [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
