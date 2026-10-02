# Documentation ownership and navigation

This is the repository-wide documentation map. `docs/` connects supported
interfaces to their execution contracts; [theory](../theory/README.md) owns
mathematical definitions, hypotheses and proofs. Tutorials show how to call an
owner, contracts specify admission and output, and theory states what follows
under which assumptions. None creates a second research queue.

Start with the [root README](../README.md) for installation and first use. For
a scientific question, use the theory index's reading routes and its map from
definitions to engine modules, tests and SDK entry points.

<!-- BEGIN DOCS CATALOG -->

## Usage guides

| Maintained owner | Use it for | Boundary |
| --- | --- | --- |
| [CLI and SDK](CLI_AND_SDK.md) | Network creation, operator studies, diagnostics, JSON and command routes | Word counts are not physical time; reports are not checkpoints |
| [Regional and relational SDK](guides/REGIONAL_AND_RELATIONAL.md) | Form observations, joint models, pattern/support reports, capture and graph-independent coefficient bounds | Preparations, references and uncertainty are inputs; method admission is distinct from a theorem's hypotheses |
| [Observational interfaces](STRUCTURAL_INTERFACE_THEORY.md) | Feature graphs, multichannel signals, comparisons and reserved forecasts | Engineering adapters do not independently identify canonical physical variables |
| [Optional Torch backend](TORCH_BACKEND.md) | Backend selection, device checks and execution limits | Requested backend, effective device and measured acceleration are different claims |

## Contracts and diagnostics

| Maintained owner | Responsibility | Primary evidence |
| --- | --- | --- |
| [API contracts](API_CONTRACTS.md) | Shared admission, nodal solvers and generated operator metadata | Actual execution owners and operator registry |
| [Relational dynamics](contracts/RELATIONAL_DYNAMICS.md) | Joint field/step, support/capture reports, coefficient uncertainty and retained acquisition audit | Admitted model and exact/finite implementations; record consistency is not experiment success or physical provenance |
| [Operator events](contracts/OPERATOR_EVENTS.md) | Schedules, jumps, atomic stages, REMESH and finite executor evidence | Shared event and history owners; no unrestricted stability guarantee |
| [Structural fields](STRUCTURAL_FIELDS_TETRAD.md) | Tetrad definitions, units, availability and estimator provenance | Shared field readers; the tetrad is not a complete state basis |

<!-- END DOCS CATALOG -->

## Repository and research owners

| Responsibility | Primary owner |
| --- | --- |
| Contributor and agent instructions | [AGENTS](../AGENTS.md), mirrored verbatim at `.github/agents/my-agent.md` |
| Package boundaries and execution paths | [Architecture](../ARCHITECTURE.md) |
| Test selection and development | [Testing](../TESTING.md), [Contributing](../CONTRIBUTING.md) |
| CI, publication and reporting | [Workflows](../.github/WORKFLOWS.md), [Security](../SECURITY.md) |
| Documentation commands and staging | [Scripts](../scripts/README.md) |
| Definitions, derivations and scientific scope | [Theory catalog](../theory/README.md); [glossary](../theory/GLOSSARY.md) for classified concept cards |
| Grammar policies and verification | [Unified grammar](../theory/UNIFIED_GRAMMAR_RULES.md#9-verification-and-reporting); rules and evidence share that owner |
| Research priorities | [Portfolio](../TNFR_lineas_de_investigacion.txt) classifies branches; [strategy](../theory/NODAL_RESEARCH_STRATEGY.md) explains them |
| Current state, resumption and active gates | [Execution checkpoint](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-checkpoint) and its [single active gate](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate), not additional overview task lists |
| Maintained runnable entry points | [Examples](../examples/README.md), [benchmarks](../benchmarks/README.md), [optional applications](../applications/README.md) |
| Historical results and supersession | [Archive](../theory/research/archive/README.md), including [reported interface observations](../theory/research/archive/REPORTED_INTERFACE_OBSERVATIONS.md) |
| Past publication statements | [Changelog](../CHANGELOG.md); historical claims are not current guarantees |

## Update rules

Change the implementation or definition and its owner together. Summaries link
to that owner; they do not copy derivations, configuration tables or delivery
histories. Keep usage examples separate from the full admission contract.
The agent mirror is the only intentionally exact prose duplicate.

The catalog above owns the technical website menu; the theory catalog owns its
own menu. `python scripts/check_documentation.py --write-generated` refreshes
both menus, the registry-derived operator table and the glossary index. The
ordinary gate checks catalog coverage and generated agreement. Link/fragment
checking and a strict site build also belong to the
[documentation checks](../scripts/README.md). Generated consistency does not
validate a theorem or a physical identification.

The glossary's declaration classes and dependency/evidence checks are specified
by its own template and `scripts/check_glossary.py`. Do not redefine concept
admission in another index or infer scientific proof from a valid link.

## Historical and generated material

`docs/assets/` holds website support and referenced frozen research evidence,
including source archives and numerical reports. Preserve those paths and bytes;
new runs belong in `artifacts/`, not over the published record. Retaining an
old result does not promote it into the current execution contract.

`artifacts/`, ignored `output/`, `outputs/` and `results/` retain local receipts
and earlier source contexts. They are not current guides or another task queue.
Archived notebooks retain original outputs and are not runnable tutorials.
`build/docs-source/` and `site/` are generated; edit their maintained inputs.

Retire a document only after moving useful material to its owner and repairing
maintained references. Preserve source-bound observations and mark their scope
and replacement. [Archive records](../theory/research/archive/README.md) explain
supersession; older manifests retain their original recovery paths and hashes.
