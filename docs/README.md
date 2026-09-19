# Documentation ownership and navigation

This is the repository-wide map of maintained documentation. A topic has one
primary owner; summaries, examples and translations point to that owner rather
than introduce their own definitions. [The theory index](../theory/README.md)
provides the detailed scientific-document inventory and status.

## One owner for each responsibility

| Responsibility | Maintained owner | Source or evidence boundary |
| --- | --- | --- |
| Installation and first use | [Root README](../README.md) | Version/dependencies in `pyproject.toml`; executable example checked by documentation gate |
| Working conventions and invariants | [AGENTS](../AGENTS.md) | Exact mirror at `.github/agents/my-agent.md`; synthesis, not a duplicate research ledger |
| Package boundaries and execution paths | [Architecture](../ARCHITECTURE.md) | Actual modules and dispatch paths |
| Test selection and local verification | [Testing](../TESTING.md) | `pyproject.toml`, test configuration and executable test paths |
| Contributor process | [Contributing](../CONTRIBUTING.md) | Uses Testing; does not define a second test/dependency matrix |
| CI and publication behavior | [Workflows](../.github/WORKFLOWS.md) | YAML owns triggers, permissions and commands; prose is descriptive |
| Security reporting | [Security](../SECURITY.md) | Reporting procedure, not a guarantee of vulnerability absence |
| Documentation commands and staging | [Scripts](../scripts/README.md) | `check_documentation.py`, `verify_internal_references.py`, `prepare_docs.py`, `mkdocs.yml` |
| Nodal definitions and mathematical types | [Fundamentals](../theory/FUNDAMENTAL_THEORY.md) | Declared chart, units, inputs and nodal row |
| Constitutive and parameter dependencies | [Parameter foundations](../theory/NODAL_PARAMETER_FOUNDATIONS.md) | Explicit model assumptions and exact/finite scope |
| Operator metadata and execution contracts | [API contracts](API_CONTRACTS.md) | Generated registry table plus execution-path boundaries |
| Operator interpretation | [Structural operators](../theory/STRUCTURAL_OPERATORS.md) | Explains the implemented contracts; no second registry |
| Grammar rules and policy premises | [Grammar](../theory/UNIFIED_GRAMMAR_RULES.md) | `grammar_canon.py`; [scope/counterexamples](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) and [verification map](grammar/PHYSICS_VERIFICATION.md) |
| Field definitions and estimator behavior | [Structural tetrad](STRUCTURAL_FIELDS_TETRAD.md) | Shared field readers, availability, units and fit/fallback provenance |
| Research branches | [Portfolio](../TNFR_lineas_de_investigacion.txt) | Main, supporting and parked lines |
| Scientific rationale | [Strategy](../theory/NODAL_RESEARCH_STRATEGY.md) | Explains choices; does not create tasks |
| Active research tasks and gates | [Execution plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md) | Sole active queue |
| Examples | [Example index](../examples/README.md) | Tutorial/model scope; not a theorem or performance inventory |
| Benchmarks and research instruments | [Benchmark index](../benchmarks/README.md) | Record input, path, seed, hardware and scope for each actual run |
| Factorization usage and configuration | [Factorization lab](../factorization-lab/README.md) | Candidate heuristics, arithmetic checks and fallback provenance |
| Arithmetic primality utility | [Primality guide](../primality-test/README.md) | Supplied divisor statistics; no physical generation or complexity theorem |
| Optional Torch backend | [Torch scope](TORCH_BACKEND.md) | Backend operations versus graph-pressure execution and measured acceleration |
| Structural application interfaces | [Interface guide](STRUCTURAL_INTERFACE_THEORY.md) | Engineering protocol; independent validation required |
| EEG correspondence report | [EEG report](EMPIRICAL_CONFRONTATION_EEG.md) | External report, not admission of the repository measurement model |
| Historical release notes | [Changelog](../CHANGELOG.md) | Statements about past versions, not current guarantees |
| Research history | [Historical archive](../theory/research/archive/README.md) | Frozen source context and explicit supersession |

## Update rules

Change the primary implementation/definition and its owner together. Update a
summary only where the result or navigation changes. Do not copy long theorem
proofs, dated delivery lists, configuration defaults or workflow matrices into
entry-point documents. The registry-generated table in API Contracts is checked
for exact agreement; use `python scripts/check_documentation.py --write-generated`
when its source metadata changes. This command does not prove mathematical claims.

The agent mirror is the only intentionally exact prose duplicate. Templates
provide structure and links; they do not define extra grammar or acceptance laws.
An educational analogy cannot override a technical contract or become evidence
of physical emergence.

## Historical and generated material

`artifacts/` contains local run receipts and pre-edit snapshots. Its preserved
paths can refer to an earlier source context; it is not another documentation
site or task queue. Archived notebooks retain their original outputs and are
not current runnable tutorials. `build/docs-source/` and `site/` are regenerated
from maintained files. Never edit generated output to change the authoritative
content.

A removed guide's useful material belongs in the corresponding owner above.
Retire dead tools only after checking callers and replacement coverage. Keep
historical observations unchanged when they carry evidence; mark their domain
and replacement explicitly rather than silently rewriting the past.

Ignored local `output/`, `outputs/` and `results/` directories also contain older
generated reports and briefs. They retain their original context and are not
current guides, published site inputs or evidence of a new validation run.

The [retirement manifest](../theory/research/archive/DOCUMENTATION_SECOND_CLEANUP_MANIFEST_2026-09-19.json)
records the second cleanup's removed paths, replacements and preserved archive hashes.
