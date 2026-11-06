# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Python Engine 0.0.4** — organized mathematical objects, nodal dynamics and
reproducible computational research.

## Summary

<a id="the-question-tnfr-asks"></a>

How can interacting parts form a recognizable pattern, keep its identity while
changing, and influence other patterns? TNFR studies this question through
**organized mathematical objects**: nodal patterns with a specified identity,
retained internal state and a declared evolution law. The repository provides
constructions, conditional proofs, a Python engine and numerical instruments
for working with these objects. Its purpose is to investigate the scope of
TNFR: which organizations its dynamics admit, what they retain when interacting,
and which collective properties follow from their structure.

TNFR's [constitutive premise](theory/FUNDAMENTAL_THEORY.md#operator-grammar-constitutive-premise)
grounds every canonical transformation in structural operators and grammar.
Continuous equations and collective descriptions need a proved connection to
those mechanisms. This is the paradigm's research premise; the present engine
and auxiliary mathematical models are not yet a complete realization of it.

The network represents constituent relations and supports execution. Object
studies test whether a state description retains the information needed for
response, which consequences follow from a law, and which operator actions
are defined on the organized target.

The long-term [physical-emergence hypothesis](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#matter-emergence-hypothesis)
asks whether different persistent organizations under one complete dynamics
can account for matter and its observed properties. The
[shared physical targets](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md#fundamental-phenomena-synergies)
include electromagnetic, weak, strong and gravitational phenomena and quantum
matter. Electromagnetism is a highlighted route; a possible double-slit
comparison has its own measurement requirements. The [physical assessment](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md#electromagnetic-emergence-audit)
connects existing organization and transport results to the field, source and
observation conditions still to be derived. Physical identification remains open.

The four foundational pillars are operator causality, sufficient state versus
observation, persistent organizational identity, and justified research/model
selection. All four have [reviewed resolutions](theory/research_prerequisites.json)
for a [prepared native organization](theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md#native-foundation-scope-decision):
two five-node rings joined by one supplied bridge, with evolving capacity,
local recovery and transmitted response under the declared complete law.
These resolutions retain that mathematical scope. **Physical mapping remains
blocked:** the separate final readiness review is absent. The
[acceptance criteria](theory/RESEARCH_PROGRAM.md#foundational-pillars) govern
any extension; earlier physical studies remain contextual material.

<a id="what-a-tnfr-network-contains"></a>
### From nodal state to an organized object

An **NFR**, or fractal-resonant node, is a modeled unit at a chosen scale.
A collective NFR retains the constituent and boundary information needed for
its declared dynamics. Its constituents can remain distinct and internally
active; equal group averages need not imply equal future responses.

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
and event laws; holding a quantity fixed is a premise. The nodal identity alone
does not select these laws. [Parameter foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md)
owns their domains, units and unresolved choices.

## Installation

Requires Python 3.10 or later. The examples on this page target the current
checkout. From its repository root, create an isolated environment if one is
not already available:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in a POSIX shell, then install that checkout:

```bash
python -m pip install -e .
python -m tnfr --version
```

NumPy, SciPy and NetworkX are core dependencies. Optional tools are grouped
in [package metadata](https://github.com/fermga/TNFR-Python-Engine/blob/main/pyproject.toml):

```bash
python -m pip install -e ".[test,docs]"     # tests and documentation
python -m pip install -e ".[compute-jax]"   # optional JAX backend
python -m pip install -e ".[compute-torch]" # optional Torch numerical backend
```

`python -m pip install tnfr` installs the published release, whose API can
precede these checkout examples. Record the source revision and effective
backend for a study.
Backend availability does not imply acceleration on every execution path.

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

Winding counts the net phase turns around the ordered cycle. Here it labels
the object's recovery class; the full state and law remain necessary to
describe its identity and evolution. Each certificate admits recovery under
ideal continuous evolution from its own snapshot, with held support,
positive held capacities and no forcing. The finite Euler step reports its
storage-balance defect separately; two admitted endpoints are not an
error bound for the intervening numerical path.

This example prepares the object and executes a conditional relational law.
It invokes no structural operators or ordered grammar; a canonical operator
realization remains to be established. Formation and the comparison of
different laws are separate research questions.
Time is structural time, not laboratory seconds or an operator count.
The [complete hello world](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/01_foundations/01_hello_world.py)
retains the network and all reports. Continue with the
[example learning route](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md)
and [capture and memory guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md).

<a id="what-counts-as-a-coherent-pattern"></a>
<a id="what-the-research-establishes"></a>
## Mathematical objects and their status

The available constructions address distinct aspects of organization.
An initialized pattern, a recovered pattern and an acquired identity establish
different claims:

| Capability | Established scope | Mathematical owner |
| --- | --- | --- |
| Native winding-pattern recovery | A prepared five-node unit ring (C5) and its ideal continuation under admitted capacity, phase and storage premises | [Native recovery](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EFFECTIVE_CONNECTIONS.md) |
| Native organization with evolving capacity | Prepared joined C5 rings retain their identity, recover locally and transmit a perturbation under one complete law | [Joint foundation account](theory/nodal/NATIVE_FOUNDATIONAL_ACCOUNT.md#native-foundation-scope-decision) |
| Acquired native organizations and response | Formation/selection evidence and conditional regional response under a two-bridge dynamic-capacity law; quantitative entry of the actual sources into the response domains remains open | [Acquired native response](theory/nodal/NATIVE_ACQUIRED_RESPONSE.md) |
| Formation, retention and recovery | Prepared families under specified laws, with source, stability and uncertainty conditions | [Pattern dynamics](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/SINE_PATTERN_DYNAMICS.md) |
| Collective state, interaction and memory | Exact reductions or bounded descriptions that retain required constituent and boundary information | [Emergent ontology](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md) |
| Controlled EPI simplification | Finite error bounds for omitted memory under fixed reversible dynamics; a held-input extension preserves the maintained profile and mean drift | [Memory omission and profile preservation](theory/DERIVED_EPI_MEMORY.md#profile-anchored-memory-omission) |
| Localized motion | Finite transport of a prepared full-state identity on supplied support | [Identity transport](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/LOCALIZED_IDENTITY_TRANSPORT.md) |
| Operator actions on organized states | Admitted actions and conditional compositions, with explicit event work, capacity and live grammar requirements | [Structural operators](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md) |

Combine results only when their state, law, support, capacity, clock and
preparation agree. Proofs establish consequences of premises, tests check
implementation, and numerical experiments describe their cases and error
controls. Empirical validation requires independent measurements; mathematical
storage is not identified with measured energy by its name.

**The first development priority is canonical implementation and integration
of structural operators**, as the mechanisms of emergence and self-organization.
The [current foundation question](theory/RESEARCH_PROGRAM.md#research-current-question)
connects each missing function to sufficient state, actual actions, grammar
and continued collective participation. The
[all-thirteen audit](theory/STRUCTURAL_OPERATORS.md#operator-canonical-coverage-audit)
confirms executable coverage and corrects concrete integration defects, while
identifying unresolved filtering, topology, richer-form and scale roles.
Registered operators are not yet a complete canonical realization. The
[operator audit](theory/STRUCTURAL_OPERATORS.md#continuous-operator-realization-audit)
finds that the recent continuous contact laws have no established bridge to
UM, RA or THOL. Closing this dependency takes priority over extending their
results as canonical TNFR dynamics.
The existing passive contact completions now have a
[conditional elementary acquisition theorem](theory/nodal/RELATION_FOUNDATIONS.md#passive-contact-acquired-response):
a relatively open family of small preparations with no initial bridge acquires
a stable common-form/common-phase organization, retains a predictive relation
value and recovers after small supplied probes. The proof controls the complete
changing-contact state through zero contact. It establishes neither prior
constituent formation nor the older amplitude-one source's destination.
The [source-to-response result](theory/nodal/RELATION_FOUNDATIONS.md#passive-contact-source-response)
now gives a controlled leading law for retained contact and its tangent
recovery scale. The same acquisition proof extends to finite collections of
components when appropriately funded contacts connect them. These results
broaden retained response geometry; acquisition of nonuniform internal
patterns remains a separate question. These are conditional results for
postulated auxiliary laws, not proofs of operator-driven formation. Exact
detached budget/source readers are not a joint weighted executor or a uniquely
selected fundamental law. The
[bibliographic synthesis](theory/BIBLIOGRAPHIC_METHODS.md#bibliographic-strategic-pivot)
connects this priority to stability, viability, symmetry, memory and scaling,
including the selected readings and their implementation scope.

The completed [two-port selection result](theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-formed-two-port-original-period-exclusion)
proves eventual absence of the original periods `(2,1,0)` for every member of
its original source family, including cut representatives. It leaves the
outgoing equilibrium, periods and a common destination unidentified. This
scoped exclusion does not rule out collective organization: the same contacts
with the [shifted-origin preparation](theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-formed-two-port-origin-decision)
have a separate positive capture certificate. The original law, source and
frozen responses remain unchanged; completing that particular trajectory is
not automatically the next research priority.

## Choose an execution path

Choose the interface for the intended model and action:

| Task | Interface or owner | Contract |
| --- | --- | --- |
| Evolve native relational form and phase | `RelationalExchangeModel`, `Network.step_relational` | Resultant-direction pressure and a joint phase law; supplied support, held capacity and admitted phase domain |
| Assess the normalized-sine model | [Shared assessment owners](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md#normalized-sine-proof-adapters) and [usage guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/relational/SINE_COMPARISON_AND_INFERENCE.md) | A distinct form/phase law using neighboring sine differences; scoped observations, certificates and continuous enclosures |
| Run operator words | `TNFR`, `StudySpec`, `run_study`, `tnfr network` | Registered transformations, grammar, live preconditions and declared hybrid events |
| Read stored network state | `diagnose_network` | Detached diagnostics with explicit availability; stored pressure is not refreshed |
| Study regions, contacts and retained responses | [Regional workflows](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md) | Declared regions, hypothetical support changes, preparation-specific bounds and evidence |

Supplying a `reference_model` to a sine assessment does not make
`step_relational` execute the sine law. A hypothetical attachment report does
not add a live edge. Continuous theorems, numerical steps and validated
enclosures provide different guarantees. The [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
define admission, mutation scope, provenance and reporting.

### Operators, grammar and evolution

The 13 registered operators specify transformations of NFR state and relations:
Emission (AL), Reception (EN), Coherence (IL) and Silence (SHA), for example.
English names lead explanations; codes abbreviate words and formulas.
Grammar checks composition and execution context, while every action must
satisfy its target's live preconditions. Admitting a requested action does not
derive its occurrence. The
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
topology and execution seeds; `TNFR.create(..., seed=...)` sets only the topology seed.
Recipes and reports are not complete checkpoints. The
[CLI/SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md)
owns usage and directs collective studies to their scoped instruments.

<a id="observe-the-network"></a>
## Observe state and response

Structural observations are selected for the question being studied. The
**structural tetrad** remains the shared four-channel diagnostic view. It
reveals selected features; reconstructing state or predicting a response
requires a sufficient observation under the chosen complete law. The SDK
also retains admitted EPI, pressure, phase and capacity, with optional spatial
profiles, phase-resultant evidence and explicitly oriented cycle winding.

| Tetrad field | What it measures | Interpretation |
| --- | --- | --- |
| `Phi_s` | Nonlocal aggregation of structural pressure | Pressure contributions weighted by inverse squared path distance |
| `abs(grad phi)` | Local phase separation | Mean absolute wrapped difference from neighboring phases; bounded by `pi` |
| `K_phi` | Circular phase curvature | Displacement relative to a neighbor phase resultant; bounded by `pi` where defined |
| `xi_C` | Static coherence-product decay scale | Selected fit policy, with a separately identified spectral fallback |

Field path geometry reads edge `length`, falling back to `weight`; diffusion
uses `weight` as conductance. Undefined curvature, absent temporal evidence and
unavailable estimates remain explicit. Warning thresholds are configured
policies. The [structural field guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md) owns
definitions, interpretation and numerical limits.

More output channels need not mean more independent structural coordinates.
The [observation requirements](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/MINIMAL_STRUCTURAL_DEGREES.md)
give constructive sufficiency results and counterexamples. A spatial profile
can also use an explicitly supplied observation width; that width is a
resolution choice, not a new fundamental TNFR parameter.

## Read the research

The [TNFR research program](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/RESEARCH_PROGRAM.md#research-objective)
keeps the objective, current state and priorities in a small entry document.
Its linked chapters preserve each model's premises, proofs and evidence in
separate canonical Markdown files. Open and preview one chapter at a time.
Use the [compiled research pages](https://fermga.github.io/TNFR-Python-Engine/theory/RESEARCH_PROGRAM/)
for chapter navigation, or `python scripts/research_book.py list` in a checkout.
The [conceptual core](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/CONCEPTUAL_CORE.md)
is the starting point for the foundations and their dependencies. The
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
own premises. The catalog identifies their scope; their presence does not
make them part of the same generative model. Evaluated evidence, negative
results and retirement provenance remain in the [research archive](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/archive/README.md).
The [selected raw evidence](https://github.com/fermga/TNFR-Python-Engine/blob/main/artifacts/github-consolidation-20261030/PUBLIC_EVIDENCE_README.md)
includes source archives, reviews and retained outcomes with an explicit
publication manifest. User-supplied reference books remain local.
The ignored `bibliografia/` directory is not included in GitHub or required
for installation and the routine checks. Bibliographic notes identify the
consulted sources and reading limits; the published evidence is a selection,
not a complete copy of local research records.

<a id="how-pulse-resonance-and-fractality-fit-together"></a>
<a id="the-research-direction"></a>
<a id="how-this-could-connect-to-physical-reality"></a>
<a id="research-direction-and-physical-scope"></a>
### Research scope

Developing TNFR means testing the constructions above, their limits and the
additional premises needed for more complex organization. Fractality asks
what a larger node must retain to inherit its constituents' dynamics and
admissible actions. Grouping nodes alone does not establish that inheritance
or autonomous formation across scales.

The research program owns the sole task queue. Comparisons with independent
observations require preparation, measurement, clock and uncertainty models, separate
calibration and reserved evaluation, and suitable alternatives. EPI need not
equal a sensor reading. The conditional results do not select a unique
fundamental law.

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
| `theory/RESEARCH_PROGRAM.md` | Research objective, sole current question and technical chapters; the theory index is generated navigation |
| `theory/research_prerequisites.json` | Scoped foundation resolutions and the separate physical-readiness review; checked by `scripts/check_research_prerequisites.py` |
| `docs/` | Usage and execution contracts, organized by the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) |
| `tests/` | Contract checks and explicitly selected research tests; selection in [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) |
| `examples/`, `benchmarks/` | [Runnable illustrations](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) and [scoped instruments](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md) |
| `docs/assets/`, `artifacts/` | Published evidence and local protocols, source archives and responses; missing local evidence remains unavailable |
| `bibliografia/` (local only) | User-supplied reference books, ignored by Git and preserved separately from publishable evidence |
| [applications/](https://github.com/fermga/TNFR-Python-Engine/blob/main/applications/README.md) | Basic arithmetic utility and retired application scope |

## Contribute and verify

[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md) defines contributor and agent responsibilities.
[Contributing](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md) and [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) own the development
workflow, dependencies and test selection. Preserve unrelated changes, reuse
shared implementations, and update a changed claim with its responsible
contract and checks.

With the test and documentation extras installed:

```bash
python scripts/check_research_prerequisites.py
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

The ordinary prerequisite check validates the current evidence associations
and permits only the foundation work admitted by the program. Before a
physical study, the additional `--require-physical-mapping` check must pass;
its failure is currently expected because the final readiness review is absent.
Published source archives and consumed protocols remain evidence, not a
standing instruction to rerun their producers.

## Citation and license

Cite the exact source snapshot used. [CITATION.cff](https://github.com/fermga/TNFR-Python-Engine/blob/main/CITATION.cff) owns software
citation metadata; the project DOI is
[10.5281/zenodo.17602860](https://doi.org/10.5281/zenodo.17602860).
MIT licensed; see [LICENSE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/LICENSE.md).
