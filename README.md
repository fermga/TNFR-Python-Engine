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

The repository keeps three paths distinct. **Operator execution** applies
registered transformations and grammar. The **native relational law** evolves
form and phase using the direction of the neighboring phase average, with
explicit limits on where that direction is defined. A separate **normalized-sine
comparison law** uses sums of neighboring sine differences. Its smoother
equations support additional mathematical studies, but a result about that law
does not automatically apply to the native runtime or an operator sequence.

### What counts as a coherent pattern

A coherent pattern has organized relationships that can be followed over time.
It need not mean that every node has the same value or phase. One possible
identity is an arrangement in which phase advances around a loop. A useful
persistence claim must say what defines that identity, which disturbances it
survives and under which evolution rule.

The nodes and their environment evolve together under the chosen complete
law. Describing a pattern relative to one node does not hold that node still:
its motion must also be subtracted. Some models keep capacities and connections
fixed to study internal evolution. They can admit equilibria, so change at
every node at every moment is not a universal consequence of the nodal equation.

TNFR calls a region carrying such organization a **fractal-resonant node**, or
**NFR**. The research explores whether larger patterns can be understood in
terms of interacting smaller patterns. Treating a whole region as a new node
requires checking what information is retained and whether its dynamics can
really be predicted at that larger scale. Nesting regions does not by itself
prove a universal fractal structure.

A larger NFR would be a collective organization of its constituents, which
continue to exist and evolve. A description at a larger scale does not erase
them. The [scale analysis](theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md) derives
collective descriptions from specified fine-node laws. They retain internal
phase spread, form differences and the connections that distinguish the
constituents. Two groups with the same average can respond differently because
their internal arrangements differ. Only labels that the complete law treats
as interchangeable can be discarded without losing information.

One concrete example is a ring of five pairs, with each member connected to
both members of the neighboring pairs. Under a stated zero-loss sine law and
equal fixed capacities, a [persistence theorem](theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-replica-joint-persistence)
protects the collective phase arrangement while initially active pairs keep
exchanging form and phase. The moving parts need not repeat exactly the same
motion. This establishes maintenance of a prepared organization, with its
network supplied in advance. It does not establish how that organization forms.

The distinction matters when comparing laws. A specified
[alternative exchange law](theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#sine-mobility-relative-geometry)
preserves the protected relative geometry, but the same proof of repeated
return does not carry over. Likewise, an exactly prepared periodic pulse can
be unstable even when nearby collective organization stays protected.
**Formation, preserved identity, internal activity and repeated return are
different questions.** A result about one must not silently answer the others.

Resonance, pulse or vibration, and fractality are the three organizing ideas
of this research. Each needs its own mathematical statement: selective
response to an interaction, continuing internal motion, and dynamics that
can be inherited when smaller patterns are viewed as larger ones. The
program seeks to connect them through the same nodal laws; their names
alone do not make them universal properties of every admitted model.

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

### What resonance means here

Resonance asks whether a pattern responds more strongly to some rhythms than
to others. In the normalized-sine comparison law, form and phase exchange
structural storage while form differences also dissipate it. Near a stable
pattern, this produces a mathematically derived maximum response at a nonzero
frequency for a specified input and its matching structural-work readout.
The result applies even when a freely disturbed pattern returns to equilibrium
without oscillating. A phase readout can behave differently, so the chosen
measurement is part of the claim.

This [resonance foundation](theory/nodal/RESONANCE_FOUNDATIONS.md) connects
the nodal dynamics to a testable response. It does not mean that every node
must vibrate forever, that the input appears by itself, or that all admitted
TNFR laws have the same response. The Resonance operator is a separately
configured transformation. Physical interpretation still needs measurements.

A permanent pulse requires a separate result. The finite closed sine model
with positive loss approaches its equilibrium set. At its already permitted
zero-loss boundary, however, an isolated pair can exchange form and phase
periodically for indefinitely long time in the exact equations. The motion
follows from the existing rows and a nonzero initial preparation. This
[reversible pulse](theory/nodal/RESONANCE_FOUNDATIONS.md#permanent-pulse-admission)
does not yet explain why zero loss should be selected or how the initial
activity appears. The research therefore distinguishes fundamental loss from
apparent local loss caused by exchange with an unobserved environment.

That distinction has a concrete mechanism: in the conservative three-node
model, structural storage can leave one connection and enter the other.
Near consensus, the exact linearized equations carry it back again. The
[environmental-memory result](theory/nodal/RESONANCE_FOUNDATIONS.md#finite-conservative-memory)
retains this exchange when the middle node is hidden. A fading local signal
therefore need not mean that the whole network loses its activity.

Under additional hypotheses, [nonlinear recurrence](theory/nodal/RESONANCE_FOUNDATIONS.md#nonlinear-recurrence)
means that almost every state in a bounded family returns arbitrarily close
to an earlier configuration. It gives neither a waiting time nor a shared
rhythm, and does not certify every chosen state. These distinctions let the
research ask about continued organization without assuming one perfect pulse.

### What the research establishes

The results are conditional: each names its evolution law, preparation and
observation. A mathematical proof establishes what follows from those premises.
Tests check the implementation, while a numerical experiment supplies evidence
for its specified case. Neither establishes a physical identification by itself.

- **Relaxation and recovery.** Pure form diffusion has precise convergence
  results on its admitted support. Joint form/phase laws also admit protected
  phase patterns and recovery from specified disturbances. One prepared
  formation result under the native relational law combines a validated
  numerical transit with a theorem for the reached region; it is not a
  guarantee for every starting network.
- **Prepared formation and its controls.** Under the separate smooth sine
  law, supplied nonuniform form can generate phase winding and enter a
  protected region, with explicit preparation uncertainty. The
  [same-law analysis](theory/nodal/SINE_PATTERN_DYNAMICS.md) also proves
  consensus for a sufficient bounded-budget class and shows why two states
  with equal storage can have different formation outcomes. The initial
  organization, support and law are premises; they are not selected autonomously.
- **Interaction and memory.** A retained environment can transmit changes
  between patterns. Removing its coordinates from an observation produces
  memory and retains its initial state. Shared inference and forecast tools
  bound what earlier observations establish, and report unresolved information.
- **Coherent geometry with internal motion.** The sine comparison supplies
  conservative identity barriers, recurrence results and exact prepared
  periodic families. Their meanings differ: preserving a phase pattern does
  not require repeating one waveform, and an exact periodic orbit need not
  resist nearby disturbances.
- **Scale descriptions with retained constituents.** For an admitted replica
  graph, a closed description keeps both group means and internal organization.
  It removes interchangeable labels without removing the smaller nodes.
  Means alone generally lose information needed to predict the future.
- **Support changes and formation limits.** Hypothetical connections and
  disconnections have separate state, storage and recovery conditions.
  Storage barriers, symmetry and early dissipation rule out some proposed
  formations even when the target itself has an affordable storage value.
  For example, an admitted family starting with equal phases relaxes to
  consensus throughout the declared preparation budget; the
  [whole-class proof](theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-consensus-preparation-obstruction)
  does not require a trajectory search.
  Passing a budget check does not select an event or guarantee its outcome.

The [theory-to-execution map](theory/README.md#theory-to-execution) connects
these mechanisms to their proofs, shared implementation and independent checks.
Counterexamples belong to that account: equal averages, initial energy or
instantaneous rates can conceal states with different subsequent behavior.

### The research direction

The long-term objective is to determine whether coherent interacting patterns,
their internal motion and their organization across scales can yield a
predictive account of physical properties. The immediate method is to state
sufficient information, specify all evolution laws, check their joint
consistency and make a discriminating prediction or prove an obstruction.
Reusable results enter shared engine or assessment owners with their hypotheses
intact; an observation report does not install a new dynamical law.

The [foundational work](theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md) studies
how locality and a declared storage balance constrain pressure, phase evolution
and admissible states. A [conditional selection result](theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#closed-form-balance-and-source-selection)
identifies assumptions that select sine exchange: a conserved weighted total
of form, a specified local information class, a fixed storage formula and
additional capacity and phase conditions. These assumptions need justification
of their own. Two laws can share conservation and a small-disturbance response
while differing for larger changes, so those similarities alone do not choose
a fundamental law.
The [resonance and scale foundations](theory/FUNDAMENTAL_THEORY.md#resonance-fractality-foundation-audit)
connect reciprocal interaction to internal motion. Dissipation and the detailed
nonlinear response still depend on declared laws; repeating a compatible
structure across scales does not explain how the hierarchy forms.

The central generative question is how an organization appears from a declared
family of initial states, what maintains it and how it interacts with its
environment. An autonomous evolution law still needs supplied initial
conditions; "formation" must specify which property was initially absent and
later appears. Recognizing a temporary group, recovering an already prepared
pattern and entering a protected family have different requirements. Exact
periodicity is not a prerequisite for answering that formation question.

Support and preparation remain supplied. The
[relation foundations](theory/nodal/RELATION_FOUNDATIONS.md) distinguish an
existing connection, an effective interaction through other nodes and an
autonomously occurring support change. Formation of the initial substrate,
selection of the microscopic law and physical identification remain open.

The strategy's [fundamental questions](theory/NODAL_RESEARCH_STRATEGY.md#fundamental-research-dependencies)
connect sufficient state, law selection and composition. The
[portfolio](TNFR_lineas_de_investigacion.txt) classifies supporting branches.
The [execution plan](theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
is the sole task queue: it owns the current question, admission conditions,
deliverables and stopping rule. Its
[checkpoint](theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-checkpoint)
locates the reusable evidence for resuming work. This introduction describes
the scientific framework rather than maintaining a parallel progress log.

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
| Evolve native relational form and phase | `RelationalExchangeModel`, `Network.step_relational` | Argument-based law with declared support, held capacity and phase-domain admission |
| Assess the smooth sine comparison | `bound_relational_sine_exchange`, `assess_sine_replica_pulse` and related module-level reports | Separate sine law; read-only assessments do not switch the native runtime or certify arbitrary graph membership |
| Observe stored state | `diagnose_network` | Detached observations with independent availability; pressure is not refreshed |
| Observe regions and their relations | `regional_form`, `source_relative_form`, `relational_pattern` | Supplied regions and references; no automatic closed dynamics for the reduced state |
| Compare an attachment | `relational_attachment` | Fresh separate/joined fields for a hypothetical connection; does not change live support |
| Compare a bridge relocation | `relational_relocation` | Supplied atomic bridge exchange preserving internal support; passive budget and recovery are separate obligations |
| Apply a capture or transit theorem | `Network.relational_*capture` | Read-only sufficient certificates with support-specific hypotheses |
| Bound a prepared coefficient response | `bound_relational_coefficient_from_jet`, `bound_relational_coefficient_from_samples` | Module-level observers with declared uncertainty; no graph evolution or preparation authentication |

The [regional and relational guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md)
provides preparations and examples for joint dynamics. The
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
define admission, numerical behavior, atomic stages and reporting. These routes
share owners while retaining distinct mathematical contracts. Supplying a
`reference_model` to a sine assessment carries its coefficients and admission
premises; it does not make `step_relational` execute the sine law. Continuous
theorems, finite numerical steps and validated enclosures also provide different
guarantees.

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

Start with the [theory catalog's question routes](theory/README.md#choose-a-question)
to find definitions and derivations, then its
[theory-to-execution map](theory/README.md#theory-to-execution) for shared engine
owners and representative tests. The [glossary](theory/GLOSSARY.md) classifies
concepts as supplied premises, conditional results, diagnostics or open claims.

The [strategy](theory/NODAL_RESEARCH_STRATEGY.md) explains how state, law selection
and formation connect; the [execution plan](theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
specifies the next bounded question. Auxiliary Hamiltonian, graph-wave,
geometric and arithmetic studies remain in the catalog under their own
assumptions. Their presence does not make them interchangeable with the main
generative model.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/dynamics/`, `operators/` | Pressure and evolution laws, named transformations, grammar and shared execution |
| `src/tnfr/physics/`, `mathematics/`, `metrics/` | Theorem tools, numerical algebra, observations and diagnostics |
| `src/tnfr/sdk/`, `cli/` | Public networks, study recipes, reports and command adapters |
| `src/tnfr/config/`, `constants/`, `utils/` | Shared configuration, numerical settings, validation helpers and I/O |
| `src/tnfr/research/` | Reusable admission, provenance and read-only record audits; not another evolution runtime |
| `theory/` | Mathematical definitions, derivations, research strategy and the execution plan |
| `docs/` | Usage guides, execution contracts and documentation navigation |
| `tests/`, `examples/`, `benchmarks/` | Contract checks, executable illustrations and scoped measurement/research instruments |
| `docs/assets/`, `artifacts/` | Preserved published evidence and local protocols/source archives/responses; missing local evidence remains unavailable |
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
