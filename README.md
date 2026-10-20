# TNFR: Resonant Fractal Nature Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17602860.svg)](https://doi.org/10.5281/zenodo.17602860)
[![PyPI version](https://badge.fury.io/py/tnfr.svg)](https://pypi.org/project/tnfr/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Python Engine 0.0.3.9** — organized mathematical objects, nodal dynamics and
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

The network represents constituent relations and supports execution. Object
studies test whether a state description retains the information needed for
response, which consequences follow from a law, and which operator actions
are defined on the organized target.

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
checkout. From its repository root, install that source:

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

This example prepares the object. Its formation is a separate research
question, as are operator events and the comparison of different laws.
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
| Native winding-pattern recovery | A prepared five-node unit ring (C5) and its ideal continuation under admitted capacity, phase and storage premises | [Native recovery](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EFFECTIVE_CONNECTIONS.md#relational-pattern-detachment) |
| Formation, retention and recovery | Prepared families under specified laws, with source, stability and uncertainty conditions | [Pattern dynamics](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/SINE_PATTERN_DYNAMICS.md#reading-map) |
| Collective state, interaction and memory | Exact reductions or bounded descriptions that retain required constituent and boundary information | [Emergent ontology](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md) |
| Localized motion | Finite transport of a prepared full-state identity on supplied support | [Identity transport](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/LOCALIZED_IDENTITY_TRANSPORT.md) |
| Operator actions on organized states | Admitted actions and conditional compositions, with explicit event work, capacity and live grammar requirements | [Structural operators](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md#collective-grammar-descent) |

Combine results only when their state, law, support, capacity, clock and
preparation agree. Proofs establish consequences of premises, tests check
implementation, and numerical experiments describe their cases and error
controls. Empirical validation requires independent measurements; mathematical
storage is not identified with measured energy by its name.

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
[collective readiness criterion](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#collective-grammar-readiness)
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

The **structural tetrad** is a shared set of diagnostics. Like a dashboard,
it reveals selected features without describing the complete internal state
or predicting its future by itself. Use it alongside the object's identity
criterion and response records; a diagnostic score cannot replace either.

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

The execution plan linked above is the sole task queue; the
[portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt)
classifies supporting branches. Comparisons with independent observations
require preparation, measurement, clock and uncertainty models, separate
calibration and reserved evaluation, and suitable alternatives. EPI need not
equal a sensor reading. The conditional results do not select a unique
fundamental law.

### What resonance means here

Resonance is an input/output response property of a declared model. Internal
pulses, recurrence and the registered Resonance operator have separate
contracts; a response peak alone proves none of them. The
[resonance foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RESONANCE_FOUNDATIONS.md)
and [scale connection map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EMERGENT_ONTOLOGY.md#pulse-resonance-scale-connections)
explain when internal motion, memory and collective response can be combined.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/tnfr/` | Shared dynamics, operators, observations, numerical tools, CLI and SDK; boundaries in [Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) |
| `theory/` | Definitions, derivations, research rationale and the single execution plan |
| `docs/` | Usage and execution contracts, organized by the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) |
| `tests/` | Contract checks and explicitly selected research tests; selection in [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md) |
| `examples/`, `benchmarks/` | [Runnable illustrations](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md) and [scoped instruments](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md) |
| `docs/assets/`, `artifacts/` | Published evidence and local protocols, source archives and responses; missing local evidence remains unavailable |
| [applications/](https://github.com/fermga/TNFR-Python-Engine/blob/main/applications/README.md) | Basic arithmetic utility and retired application scope |

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
