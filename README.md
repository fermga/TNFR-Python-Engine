# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Python Engine 0.0.4** — organized mathematical objects, structural operators
and reproducible computational research.

## Summary

<a id="the-question-tnfr-asks"></a>

How can interacting parts form a recognizable pattern, keep its identity while
changing, and influence other patterns? TNFR studies this question through
**organized mathematical objects**: nodal patterns with a specified identity,
retained internal state and a declared evolution law. The repository provides
constructions, conditional proofs, a Python engine and numerical instruments
to investigate which organizations TNFR dynamics admit, what they retain when
interacting and which collective properties follow from their structure,
without prescribing the outcome.

TNFR's [constitutive premise](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md#operator-grammar-constitutive-premise)
grounds every canonical transformation in the thirteen registered
**structural operators** and their **grammar**. Continuous equations and
collective descriptions need a proved connection to those mechanisms. This is
the paradigm's research premise: the engine and auxiliary models do not yet
realize it completely, and canonical operator integration is the first
development priority.

| Status | Scope |
| --- | --- |
| Established | Exact identities, conditional theorems and finite computational evidence for declared models: preparation, recovery, interaction, memory and operator actions on organized states |
| Foundations | Four pillars (operator causality, sufficient state versus observation, persistent organizational identity and justified model selection) have [reviewed resolutions](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research_prerequisites.json) for a [prepared native organization](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md#native-foundation-scope-decision): two five-node rings joined by one supplied bridge, with evolving capacity, local recovery and transmitted response |
| Open | Autonomous occurrence of operator actions, a uniquely selected fundamental law, empirical validation and any physical identification |

**Physical mapping remains blocked** until a separate final readiness review
binds the four resolutions; the
[acceptance criteria](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#foundational-pillars)
govern any extension. The long-term
[matter-emergence hypothesis](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis)
asks whether persistent organizations under one complete dynamics could
account for matter and its observed properties. The
[shared physical targets](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md#fundamental-phenomena-synergies)
include electromagnetic, weak, strong and gravitational phenomena and quantum
matter. Electromagnetism is the highlighted route; its
[assessment](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md#electromagnetic-emergence-audit)
records the field, source and observation conditions still to be derived.

<a id="what-a-tnfr-network-contains"></a>
### From nodal state to an organized object

An **NFR**, or fractal-resonant node, is a modeled unit at a chosen scale.
A collective NFR retains the constituent and boundary information needed for
its declared dynamics; its constituents can remain distinct and internally
active, and equal group averages need not imply equal future responses. The
network represents constituent relations and supports execution; graph
construction alone does not establish an object's identity.

The scalar engine represents the underlying nodal state through:

- **Form (`EPI`):** a signed real structural coordinate.
- **Capacity (`nu_f`):** a nonnegative factor scaling the form response.
- **Phase (`phi` or `theta`):** a circular coordinate.
- **Support and consumed history:** the connections, constituent relationships
  and temporal evidence required by the selected model.

The unforced nodal equation relates form to independently specified
**structural pressure (`DeltaNFR`)**:

$$
\frac{\partial \mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.
$$

Zero capacity freezes this form row, not necessarily the whole state. A
complete model also declares pressure, phase, capacity, support, clock, inputs
and event laws; holding a quantity fixed is a premise, and the nodal identity
alone does not select these laws. The
[parameter foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md)
own their domains, units and unresolved choices.

## Installation

Requires Python 3.10 or later. Install the published release with:

```bash
python -m pip install tnfr
```

The examples on this page target the current checkout, which can be ahead of
the release. To work on it, create an isolated environment from the repository
root, activate it (`.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell) and install the checkout:

```bash
python -m venv .venv
python -m pip install -e .
python -m tnfr --version
```

NumPy, SciPy and NetworkX are core dependencies. Optional groups in the
[package metadata](https://github.com/fermga/TNFR-Python-Engine/blob/main/pyproject.toml) add tools and backends:

```bash
python -m pip install -e ".[test,docs]"     # tests and documentation
python -m pip install -e ".[compute-jax]"   # optional JAX backend
python -m pip install -e ".[compute-torch]" # optional Torch numerical backend
```

Record the source revision and effective backend for a study; backend
availability does not imply acceleration on every execution path. Each
[GitHub release](https://github.com/fermga/TNFR-Python-Engine/releases) also
ships a lightweight development bundle with onboarding and verified retrieval
of omitted evidence.

## Quick start

Start with a small organized object: five nodes whose phases make one complete
turn around a ring, with a small signed form contrast. Specify the native
relational law, assess recovery, advance form and phase, then assess the new
state.

```python
from fractions import Fraction as Q

import networkx as nx

from tnfr.sdk import Network, RelationalExchangeModel

graph = nx.cycle_graph(5)
cycle = tuple(range(5))
for node, contrast in enumerate((1, -1, 0, 0, 0)):
    graph.nodes[node].update(EPI=Q(contrast, 1024), theta=Q(5 * node, 4), nu_f=1)
nx.set_edge_attributes(graph, 1, "weight")
graph.graph["GAMMA"] = {"type": "none"}
network = Network(graph)
model = RelationalExchangeModel(
    storage_scale=1, epi_weight=0.5, phase_weight=0.5, phase_domain="acute"
)

before = network.relational_cycle_capture(model, cycle=cycle)
step = network.step_relational(model, dt=Q(1, 1024), t=0)
after = network.relational_cycle_capture(model, cycle=cycle)
print(f"Winding: {before.cycle_winding} -> {after.cycle_winding}")
print(f"Recovery: {before.status} -> {after.status}")
print(f"Form changed: {step.before.epi != step.after.epi}")
print(f"Phase changed: {step.before.phase != step.after.phase}")
print(f"Euler storage-balance defect: {float(step.energy_step_defect):.6g}")
```

Output in the checked environment:

```text
Winding: 1 -> 1
Recovery: admitted -> admitted
Form changed: True
Phase changed: True
Euler storage-balance defect: 3.18216e-11
```

Winding counts the net phase turns around the ordered cycle; here it labels
the object's recovery class, while the full state and law describe its
identity and evolution. Each certificate admits recovery under ideal
continuous evolution from its own snapshot, with held support, positive held
capacities and no forcing. The finite Euler step reports its storage-balance
defect separately: two admitted endpoints do not bound the intervening
numerical path. This example executes a conditional relational law, not
structural operators or grammar; formation and the comparison of different
laws are separate questions. Time is structural time, not laboratory seconds
or an operator count.

The [complete hello world](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/01_foundations/01_hello_world.py)
retains the network and all reports. Continue with the
[example learning route](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md)
and the [capture and memory guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md).

<a id="what-counts-as-a-coherent-pattern"></a>
<a id="what-the-research-establishes"></a>
## Mathematical objects and their status

An initialized pattern, a recovered pattern and an acquired identity establish
different claims:

| Capability | Established scope | Mathematical owner |
| --- | --- | --- |
| Native winding-pattern recovery | A prepared five-node unit ring (C5) and its ideal continuation under admitted capacity, phase and storage premises | [Native recovery](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EFFECTIVE_CONNECTIONS.md) |
| Native organization with evolving capacity | Prepared joined C5 rings retain their identity, recover locally and transmit a perturbation under one complete law | [Joint foundation account](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md#native-foundation-scope-decision) |
| Acquired native organizations and response | Formation/selection evidence and conditional regional response under a two-bridge dynamic-capacity law; quantitative entry of the actual sources into the response domains remains open | [Acquired native response](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/NATIVE_ACQUIRED_RESPONSE.md) |
| Acquisition through passive contact | Small preparations without an initial bridge acquire a stable common-form/common-phase organization with a predictive relation value, under postulated auxiliary contact laws rather than operator-driven formation | [Passive contact acquisition](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATION_FOUNDATIONS.md#passive-contact-acquired-response) |
| Formation, retention and recovery | Prepared families under specified laws, with source, stability and uncertainty conditions | [Pattern dynamics](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/SINE_PATTERN_DYNAMICS.md) |
| Collective state, interaction and memory | Exact reductions or bounded descriptions that retain required constituent and boundary information | [Emergent ontology](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md) |
| Controlled EPI simplification | Finite error bounds for omitted memory under fixed reversible dynamics; a held-input extension preserves the maintained profile and mean drift | [Memory omission and profile preservation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/DERIVED_EPI_MEMORY.md#profile-anchored-memory-omission) |
| Localized motion | Finite transport of a prepared full-state identity on supplied support | [Identity transport](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/LOCALIZED_IDENTITY_TRANSPORT.md) |
| Operator actions on organized states | Admitted actions and conditional compositions, including Resonance and Contraction maintenance with ordered return, Coupling between organizations, Emission episodes and Mutation and Recursivity participation; occurrence remains supplied | [Structural operators](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md) |

Combine results only when their state, law, support, capacity, clock and
preparation agree. Proofs establish consequences of premises, tests check
implementation, and numerical experiments describe their cases and error
controls. Empirical validation requires independent measurements;
mathematical storage is not identified with measured energy by its name.

**Current research.** The
[all-thirteen audit](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md#operator-canonical-coverage-audit)
confirms executable coverage while leaving filtering, topology, richer-form
and scale roles open, and the
[continuous contact audit](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md#continuous-operator-realization-audit)
finds no established bridge from the postulated contact laws to Coupling (UM),
Resonance (RA) or Self-organization (THOL). The
[current question](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#research-current-question)
is collective maintenance with reciprocal participant actions: complete
positive domains or obstructions under a fixed composition of existing
operators with every member participating. Supplied operator words are
interventions; autonomous occurrence remains open.

## Choose an execution path

Choose the interface for the intended model and action:

| Task | Interface or owner | Contract |
| --- | --- | --- |
| Evolve native relational form and phase | `RelationalExchangeModel`, `Network.step_relational` | Resultant-direction pressure and a joint phase law; supplied support, held capacity and admitted phase domain |
| Assess the normalized-sine model | [Shared assessment owners](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md#normalized-sine-proof-adapters) and [usage guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/relational/SINE_COMPARISON_AND_INFERENCE.md) | A distinct form/phase law using neighboring sine differences; scoped observations, certificates and continuous enclosures |
| Run operator words | `TNFR`, `StudySpec`, `run_study`, `tnfr network` | Registered transformations, grammar, live preconditions and declared hybrid events |
| Read stored network state | `diagnose_network` | Detached diagnostics with explicit availability; stored pressure is not refreshed |
| Study regions, contacts and retained responses | [Regional workflows](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md) | Declared regions, hypothetical support changes, preparation-specific bounds and evidence |

A `reference_model` in a sine assessment does not make `step_relational`
execute the sine law, and a hypothetical attachment report adds no live edge.
Continuous theorems, numerical steps and validated enclosures provide
different guarantees. The [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
define admission, mutation scope, provenance and reporting.

### Operators, grammar and evolution

The thirteen registered operators transform NFR state and relations:
Emission (AL), Reception (EN), Coherence (IL), Dissonance (OZ), Coupling (UM),
Resonance (RA), Silence (SHA), Expansion (VAL), Contraction (NUL),
Self-organization (THOL), Mutation (ZHIR), Transition (NAV) and
Recursivity (REMESH). Grammar checks composition and execution context, and
every action must satisfy its target's live preconditions; admitting a
requested action does not derive its occurrence. The
[collective readiness criterion](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md)
states what an organization must retain to support collective actions.

For requested words, the CLI and SDK share the operator-study runner:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --steps 1 --export-spec study.json --output report.json
python -m tnfr network --spec study.json --output replay-report.json
tnfr sequences basic_activation
tnfr operators emission
```

Use `StudySpec`, `run_study` and `diagnose_network` from `tnfr.sdk` for the
Python workflow. Cycles count complete words, not seconds. The runner sets
topology and execution seeds; `TNFR.create(..., seed=...)` sets only the
topology seed. Recipes and reports are not complete checkpoints. The
[CLI/SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
owns usage and directs collective studies to their scoped instruments.

<a id="observe-the-network"></a>
## Observe state and response

Structural observations are selected for the question being studied. The
**structural tetrad** is the shared four-channel diagnostic view: it reveals
selected features, while reconstructing state or predicting a response
requires a sufficient observation under the chosen complete law. The SDK also
retains admitted EPI, pressure, phase and capacity, with optional spatial
profiles, phase-resultant evidence and explicitly oriented cycle winding.

| Tetrad field | What it measures | Interpretation |
| --- | --- | --- |
| `Phi_s` | Nonlocal aggregation of structural pressure | Pressure contributions weighted by inverse squared path distance |
| `abs(grad phi)` | Local phase separation | Mean absolute wrapped difference from neighboring phases; bounded by `pi` |
| `K_phi` | Circular phase curvature | Displacement relative to a neighbor phase resultant; bounded by `pi` where defined |
| `xi_C` | Static coherence-product decay scale | Selected fit policy, with a separately identified spectral fallback |

Field path geometry reads edge `length`, falling back to `weight`; diffusion
uses `weight` as conductance. Undefined curvature, absent temporal evidence and
unavailable estimates remain explicit, and warning thresholds are configured
policies. More output channels need not mean more independent coordinates.
The [structural field guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md)
and the [observation requirements](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/MINIMAL_STRUCTURAL_DEGREES.md)
own definitions, sufficiency results, counterexamples and numerical limits.
A spatial profile's observation width is a resolution choice, not a new
fundamental TNFR parameter.

## Read the research

The [TNFR research program](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#research-objective)
keeps the objective, current state and priorities in one small entry document;
its linked chapters keep each model's premises, proofs and evidence in
separate canonical Markdown files. Browse the
[compiled research pages](https://fermga.github.io/TNFR-Python-Engine/theory/RESEARCH_PROGRAM/)
or run `python scripts/research_book.py list` in a checkout. The
[chapter workflow](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md#documentation)
explains focused extraction and editing.

| Need | Maintained owner |
| --- | --- |
| Start with the foundations | [Conceptual core, dependencies and worked examples](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/CONCEPTUAL_CORE.md) |
| Find a definition, proof or model | [Research reading route](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#reading-the-research-as-one-program) and [chapter index](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) |
| Locate its implementation and tests | [Theory-to-execution map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/THEORY_TO_EXECUTION.md) |
| Distinguish premises, results, diagnostics and hypotheses | [Glossary](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/GLOSSARY.md) |
| Assess possible physical emergence | [Hypothesis and scope](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis), [electromagnetic prerequisites](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md#electromagnetic-emergence-audit) |
| Understand priorities and select a question | [Selection criteria](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#research-selection) |
| Resume the active task | [Current task and documentation prerequisites](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#research-current-question) |

Auxiliary Hamiltonian, graph-wave, geometric and arithmetic studies keep their
own premises; the catalog identifies their scope without making them part of
the same generative model. The repository keeps what serves active research.
Secondary and superseded records are retired with their evidence and remain
recoverable from the tag
[`pre-retirement-2026-11-06`](https://github.com/fermga/TNFR-Python-Engine/tree/pre-retirement-2026-11-06)
through the [recovery index](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/archive/README.md).
The [selected raw evidence](https://github.com/fermga/TNFR-Python-Engine/blob/main/artifacts/github-consolidation-20261030/PUBLIC_EVIDENCE_README.md)
lists the published source archives, reviews and retained outcomes.
User-supplied reference books stay in the local, Git-ignored `bibliografia/`
directory; they are not needed for installation or the routine checks.

<a id="how-pulse-resonance-and-fractality-fit-together"></a>
<a id="the-research-direction"></a>
<a id="how-this-could-connect-to-physical-reality"></a>
<a id="research-direction-and-physical-scope"></a>
### Research scope

Developing TNFR means testing these constructions, their limits and the
additional premises needed for more complex organization. Fractality asks what
a larger node must retain to inherit its constituents' dynamics and admissible
actions; grouping nodes alone establishes neither that inheritance nor
autonomous formation across scales. Comparisons with independent observations
require preparation, measurement, clock and uncertainty models, separate
calibration and reserved evaluation, and suitable alternatives; EPI need not
equal a sensor reading.

### What resonance means here

Resonance is an input/output response property of a declared model. Internal
pulses, recurrence and the registered Resonance operator have separate
contracts; a response peak alone proves none of them. The
[resonance foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RESONANCE_FOUNDATIONS.md)
and [scale connection map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md)
explain when internal motion, memory and collective response can be combined.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/` | Shared dynamics, operators, observations, numerical tools, CLI and SDK; boundaries in [Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) |
| `theory/` | The [research program](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md) with the sole current question, canonical chapters and generated navigation; `research_prerequisites.json` holds the scoped foundation resolutions checked by `scripts/check_research_prerequisites.py` |
| `docs/` | Usage and execution contracts, organized by the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) |
| `tests/` | Contract checks and explicitly selected research tests; selection in [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) |
| `examples/`, `benchmarks/` | [Runnable illustrations](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) and [scoped instruments](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md) |
| `docs/assets/`, `artifacts/` | Evidence that retained claims consume: protocols, source archives and responses; missing local evidence remains unavailable |
| [applications/](https://github.com/fermga/TNFR-Python-Engine/blob/main/applications/README.md) | The basic arithmetic utility and the scope of retired applications |
| `bibliografia/` (local only) | User-supplied reference books, ignored by Git |

## Contribute and verify

[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) defines contributor and agent rules;
[Contributing](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) and [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md)
own the development workflow, dependencies and test selection. Preserve
unrelated changes, reuse shared implementations, and update a changed claim
with its responsible contract and checks.

With the test and documentation extras installed, the routine local gate is:

```bash
python scripts/check_research_prerequisites.py
python -m pytest
python scripts/check_documentation.py
python scripts/verify_internal_references.py --ci
```

The default tests cover the routine engine and public interfaces; select a
research owner explicitly when changing its model or claim. The documentation
workflow runs the strict site build (`python scripts/prepare_docs.py`, then
`python -m mkdocs build --strict`) on every Markdown change; run it locally
when navigation, catalogs, staging or rendering change. Edit maintained
documents, not generated `build/docs-source/` or `site/` files.
[Local maintenance](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md#local-disk-maintenance)
uses `python scripts/clean_repository.py --report` to show what occupies a
checkout and `--caches` to remove regenerable caches.

The prerequisite check permits only the foundation work admitted by the
program. A physical study additionally requires
`--require-physical-mapping` to pass; it currently fails, as required, because
the final readiness review is absent. Published archives and consumed
protocols are evidence, not instructions to rerun their producers.

## Citation and license

Cite the exact source snapshot used. [CITATION.cff](https://github.com/fermga/TNFR-Python-Engine/blob/main/CITATION.cff)
owns software citation metadata. The concept DOI
[10.5281/zenodo.17602860](https://doi.org/10.5281/zenodo.17602860) resolves to
the latest version; each release's notes give its version DOI.
MIT licensed; see [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
