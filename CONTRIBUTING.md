# Contributing to TNFR

Contributions should make an implemented behavior, mathematical statement or
documented limitation more accurate and useful. The nodal equation does not make
every configured algorithm a derived physical law. Read [AGENTS.md](AGENTS.md)
for current definitions and evidence boundaries.

## Getting Started

Use a separate environment and an editable checkout. Supported interpreters and
dependency groups are defined in [pyproject.toml](pyproject.toml).
[TESTING.md](TESTING.md#run-the-repository-tests) owns installation and validation
commands; [the workflow guide](.github/WORKFLOWS.md) owns CI behavior. There are
no `dev` or `all` extras: choose a defined group for the work at hand.

Before editing, inspect the affected source, tests and specialized documentation.
Search for an existing owner of the calculation or contract. Preserve unrelated
working-tree changes and keep a contribution focused enough to review.

## Development Workflow

1. State the problem and intended behavior. For a suspected defect, record a
   reproducer or counterexample when practical.
2. Reuse existing numerical kernels, readers, validation and telemetry. Add an
   abstraction when it removes real duplication or establishes a useful boundary.
3. Implement the change and run checks appropriate to its scope, following
   [TESTING.md](TESTING.md#validation-workflow-and-reporting).
4. Update the document that owns the affected definition or interface. Link to
   that owner instead of copying its derivation or inventory.
5. Describe the result, validation and remaining limitations in the pull request.

## TNFR Principles

The [working reference](AGENTS.md), [API contracts](docs/API_CONTRACTS.md) and
[operator contracts](src/tnfr/operators/operator_contracts.py) own detailed rules.

- Distinguish initial state construction, named operator maps and continuous
  nodal integration: their contracts differ.
- Declare state representation, pressure law, phase/capacity/support inputs,
  numerical domain and any held parameters.
- Separate exact identities, conditional theorems, finite runtime observations
  and hypotheses. Tests provide evidence within their asserted domain.
- Treat grammar admission and diagnostic thresholds as implemented contracts.
  Neither a high score nor an accepted sequence proves global stability or
  empirical validity.
- Reuse named configuration values and document their units and provenance.
  A configured coefficient is not automatically a uniquely derived constant.

## Code Standards

[pyproject.toml](pyproject.toml) configures Black and isort at 88 columns, flake8
and NumPy-style docstrings. Use explicit imports, meaningful names and type
annotations consistent with the surrounding API. Update maintained public stubs
when an exposed signature changes; do not generate unrelated stubs.

Tool commands live in [TESTING.md](TESTING.md#code-quality-and-documentation-checks).
Current CI distinguishes blocking checks from advisory analysis; a contributor
checklist must not silently impose a different policy.

## Testing Requirements

Use [the testing guide](TESTING.md). Meaningful tests check observable behavior
or a mathematical contract, with independent expected results where possible.
A regression should cover the failure it prevents. Small documentation changes
do not require numerical tests. The project does not configure a universal
coverage percentage or require every change to increase coherence.

Performance comparisons need a reproducible workload, environment, cache state
and measurement protocol. An optimization that changes the result needs an
explicit accuracy or behavior tradeoff.

## Documentation

Keep technical code, comments, documentation, commits and pull requests in
English. Verbatim quotations and raw data retain their original language.

Use [the theory index](theory/README.md) to find derivation owners and
[the documentation index](docs/README.md) for usage material. The
[execution plan](theory/research/FIVE_STAGE_EXECUTION_PLAN.md) owns the single
active queue; the [portfolio](TNFR_lineas_de_investigacion.txt) classifies research
lines. Do not create a competing task list in an overview. Preserve valid anchors
and historical evidence when consolidating documents.

## Pull Request Process

Use the [pull request template](.github/pull_request_template.md). Lead with the
problem and resulting behavior, then provide commands actually run, relevant
results and untested scope. Identify changes to public APIs, serialization,
dependencies or model assumptions. Link evidence instead of pasting duplicate
reports. This guide promises no review or release deadline.

Commit types are configured in [pyproject.toml](pyproject.toml). Publication
behavior belongs to [the workflow guide](.github/WORKFLOWS.md); a merge alone
does not imply a new release.

## Theoretical Contributions

State definitions and independent assumptions before the derivation. Include
countercontrols and failure domains. A proposed operator, grammar rule or
constitutive law needs its own mathematical and implementation contract; the
nodal identity alone establishes neither uniqueness nor necessity. For an
application, distinguish an observation map from a tested prediction and keep
calibration separate from evaluation. Negative results can narrow a hypothesis.

Use the [domain-extension form](.github/ISSUE_TEMPLATE/domain_extension.yml) to
propose an application. Describe its inputs, patterns, existing APIs, intended
outputs, evaluation, limitations and maintenance. Visualizations or recipes can
be included when useful; no nonexistent base class, fixed directory skeleton or
universal diagnostic-score threshold is required.

## Code of Conduct

Discuss ideas and evidence respectfully. Avoid harassment, personal attacks and
publication of private information. Report ordinary bugs and proposals through
repository issues. For vulnerabilities, follow
[SECURITY.md](SECURITY.md#reporting-a-vulnerability) instead of posting exploit
details publicly.
