# TNFR Testing Guide

This page owns local validation instructions. [pyproject.toml](pyproject.toml)
defines tools and dependencies; [the workflow guide](.github/WORKFLOWS.md)
describes CI, whose executable commands belong to the linked workflow YAML.
Expected mathematics and contracts follow [AGENTS.md](AGENTS.md)
and the source being tested. A passing test establishes its asserted behavior
and domain, not a general physical theorem.

## Run the repository tests

Run commands from the repository root with the interpreter of a separate
environment. Install the editable project and the dependency groups used by
the main CI test job:

```sh
python -m pip install -e ".[test,numpy,yaml,orjson]"
python -m pytest
```

`test` is the compatibility alias for `test-all`; NumPy is already a core
dependency. There are no `dev` or `all` extras. Smaller extras such as
`test-unit` install only their declared tools and may not support collection
of the whole repository.
Aggregate extras reference the smaller groups so their dependency bounds have
one owner; the compatibility alias does not maintain another dependency list.
The maintained benchmark instruments are standalone CLIs, so these groups do
not install `pytest-benchmark`. The `test-performance` extra remains available
as an explicit optional plugin for downstream tests; it does not define a
benchmark-directory test suite.

The default `testpaths` in [pyproject.toml](pyproject.toml) selects the routine
engine gate: nodal execution, operator/grammar contracts, numerical admission,
public diagnostics, caches, CLI and SDK. It includes the complete operator and
SDK directories and the listed production-field owners. The broad historical
research/certificate directories are not part of every routine run.
Production Hilbert-space, projector, generator and auxiliary-runtime boundary
tests are included even though their files live under `tests/mathematics`.
CI also collects the complete retained inventory on Python 3.11, catching test
module name collisions and import errors outside the routine execution gate.
Collection alone does not execute fixtures or validate their assertions.
`pythonpath = ["src"]` imports the working tree, and `addopts = "-m 'not slow'"`
excludes marked slow cases. No custom collection plugin or second test registry
is involved. The test dependency requires pytest 7.4 or newer, supporting the
[standard testpaths wildcard configuration](https://docs.pytest.org/en/7.4.x/reference/reference.html#confval-testpaths).

Use the explicit root path for the complete retained inventory, or a research
module/directory when its owner changes:

```sh
python -m pytest tests --collect-only -q
python -m pytest tests -q
python -m pytest tests/physics/test_phase_alignment_metric.py -q
python -m pytest tests/mathematics -q
```

Explicit paths take precedence over `testpaths`. Research tests remain
executable evidence for retained conditional results; they have not been
declared obsolete merely because they are outside the routine gate. Add a new
production-field regression to `testpaths` when it belongs in that gate. A
routine pass is not a claim that every mathematical campaign was replayed.

Choose affected mechanisms through the [theory-to-execution map](theory/README.md#theory-to-execution),
then inspect their source and tests. That map links shared implementations,
representative controls and theorem owners; this guide does not maintain a
second inventory of research results or individual test cases.

For a repository-wide audit, preserve the starting working tree and review
complete defect families across their actual consumers. A scalar-boundary fix
needs both direct and adapter coverage; a rollback fix needs shared views,
container nesting and restoration order. Pair invalid-domain controls with
ordinary executions, independent expected values and the model's intentional
rounding conventions. An unusual result is not a defect merely because a
different model would reject it.

Collect the full inventory before broad execution. Choose the non-slow suites
and any additional slow owners explicitly, preserving frozen-producer guards.
Finish source changes and independent review before the final regression run;
if code changes during that run, its result cannot alone validate the new
source. Record the final interpreter, selection, skipped scope and checks.
For long runs, retain verbose output and JUnit XML so failures remain identifiable
before completion; keep a failed run's evidence when validating its corrections.
Close the audit when reproduced defects in that scope are corrected and its
checks pass, without interpreting that finite coverage as proof of no further
bugs or of the physical claims.

## Shared evidence maintenance

The [evidence workflow](docs/guides/RESEARCH_EVIDENCE.md) separates maintained
artifact mechanics, model-specific audits and immutable archived programs.
For changes to their shared owners, select exact-record, archive and restoration
controls separately from scientific producers:

```sh
python -m pytest tests/research/test_artifact_io.py tests/research/test_frozen_source.py tests/scripts/test_restore_frozen_source.py tests/mathematics/test_validated_taylor_arithmetic.py tests/mathematics/test_validated_box_taylor.py tests/physics/test_sine_flow_kernel.py -q
```

Exercise duplicate/nonfinite JSON, exact tags before scalar comparison, unsafe or
duplicate archive names, expanded-byte limits, altered hashes and refusal to
replace an existing record. Restoration controls use disposable synthetic Git
repositories and read-only inspection of the committed freeze. Verify the full
pinned base, admitted supplemental files and destination rejection; do not
create a real reserved workspace or invoke an evaluator during regression.

Retained Taylor reconstruction checks primitive shape/source admission and
Horner/remainder/endpoint arithmetic. It does not prove stored derivative
enclosures or replace a consumer's source, law, event or strict Picard checks.
The shared sine field still needs independent edge-sum and jet controls for its
layouts and held coefficients. Select changed consumers and existing read-only
evidence audits in addition to these shared tests.

Exact arithmetic controls must include fractions carrying fixed-width integer
components, alongside ordinary Python integers. Normalize admitted components
before matrix products, interval rounding, comparison flows and Taylor powers;
otherwise overflow can invalidate an enclosure or a positivity certificate.
Equivalent representations must retain the same mathematical bounds without
widening each kernel's admitted scalar or container domain.

Reuse expensive parsed records and archive bytes through module-scoped fixtures;
copy a fixture before a mutation test. Pure test helpers must not import other
test modules' autouse fixtures. Keep no-execution guards scoped and reversible,
so an audit cannot leak patched producers or subprocess calls into unrelated
tests. Overrides inside an execution guard must unwind before the guard itself;
check restoration in the same process as subsequent evidence readers.
Historical report keys such as `current` refer to their recorded source. Rebuild
their summaries from retained primitive captures; a present-engine wiring test
needs a distinct protocol identity and its actual pressure path.
Shared arithmetic must not collapse a producer and its independent
mathematical expectation into the same implementation. Current-code regression
does not update frozen evidence or authorize the pending research response.

## Grammar and scoped-program integration

Select the shared owner and its public consumers together. Operator changes
need direct glyph, public class and atomic-stage coverage where supported;
word changes need initialized and uninitialized grammar, live target/history,
accepted-prefix behavior and strict rejection controls. Runtime and program
boundaries belong to [operator events](docs/contracts/OPERATOR_EVENTS.md)
and [program execution](docs/CLI_AND_SDK.md).

Exercise signed scalar and circular phase admission before diagnostics or
mutation. Test observed Mutation secants, immutable stage snapshots, all-target
preflight, graph-owned atomic commit and restoration boundaries. A valid word
does not promise increasing coherence or future stability. Optional callbacks,
monitors, activity policies and pressure refresh must match the studied path.

## Select research checks by contract

Use the [theory-to-execution map](theory/README.md#theory-to-execution) to select
the affected proof, shared implementation and test owners. Inspect those tests
before invoking them: explicit research paths override the routine testpaths,
and a producer's availability does not authorize new evaluated responses.

| Changed responsibility | Required checks |
| --- | --- |
| State, pressure, phase or capacity admission | Primitive invalid/finite domains, alias authority, boundary values, every consumed row and actual clocks; keep held and refreshed pressure distinct |
| Exact identity or theorem | Independent incidence/matrix calculations or counterexamples, all hypotheses, branch/lift choices and unavailable domains; the theorem's scope is not proved by a literal fixture |
| Numerical certificate | Outward bounds, independent high-precision controls, source inclusion, arithmetic/representation limits, residuals and explicit failure or unavailability |
| Event or collective action | Actual support and weights, same-snapshot proposal, live policy/history, complete work and successor law; continuous closure alone does not admit an event |
| Memory, reduction or observation | Hidden initialization/input, actual observation map, consumed internal correlations and finite prediction errors; equal summaries need not predict equal futures |
| Frozen evidence reader | Read-only primitive/source reconstruction, original failures and verdicts, archive integrity and no-producer guards; never regenerate evidence to make a check pass |
| CLI, SDK or export adapter | Wiring separately from costly scientific execution, strict JSON/availability, shared kernels, cold imports and source/configuration provenance |

For the organized-source word, independent passivity, joint mean/form budgets
and scalar capacity recurrence have separate owners. Include asymmetric states,
unequal positive capacities and nonzero means. Exact event work, sequential
binary64 capacity products and complete represented execution are distinct.
A scalar capacity count is not repeated-word execution or autonomous renewal.

Do not replay the retained collective pressure-handoff producer during routine
regression: its original unresolved interval-width outcome remains evidence.
Likewise, the prepared-envelope, relative-probe and maintained-successor
instruments keep their frozen protocols, budgets and response permissions.
Select their read-only evidence consumers for documentation or parser changes.
The [execution plan](theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone admits new research evaluation.

### Boundaries that need explicit regression coverage

| Boundary | Required distinction and controls |
| --- | --- |
| Primitive admission | Reject Boolean/nonfinite physical scalars, malformed support, stale state and unrepresentable arithmetic before a solver, cache or reserved sample is consumed. Preserve each owner's domain and work caps. |
| Rebuilt evidence | Alter primitive source/law fields independently of cached bounds and favorable flags. Recompute all consumed derived fields; equality of projected values is not source authentication. |
| Exact geometry | Test zero resultants, circular cuts, strict equality and unresolved denominators. Rounded pi, a small residual or overlapping intervals does not establish an exact equilibrium, symmetry or antipodal state. |
| Correlated uncertainty | Retain source balls, mean/parity correlations, phase lifts and observation budgets. Coordinate boxes and projections cannot silently replace the set consumed downstream. Include small positive errors below an interval grid. |
| Complete evolution | Check both form and phase rows, transformed clocks, source terms, conserved means and actual support weights. Derive expected rows independently from edges, including asymmetric and multiply connected cases. |
| Reduction and memory | Vary hidden/internal state at fixed collective observations. Retain initial conditions, fast transients, port association, generated discarded modes, memory and nonlinear remainder. Tangent invariance need not close the nonlinear law. |
| Events and work | Check the actual pre-event state, jump, storage work and changed mean leaf. Continuous loss cannot fund an event without a reservoir law. An admitted jump or controller does not establish autonomous occurrence. |
| Formation and maintenance | Separately check initial absence, entry, whole-window identity, boundary exchange, retention and later response. Exact invariant-family membership or local stability does not prove acquisition. |
| Partial and all-time claims | Preserve completed horizon, original requested horizon, channel-specific resolution and unavailable bounds. Finite tracking, convergence and cross-flow contraction are different claims; a failed sufficient margin is not an observed trajectory failure. |
| Controls and alternatives | Match the complete source, law, clock and information budget. Keep counterexamples to omitted premises, including low-storage nonconsensus states, phase-blind response and same-observation/different-future states. |
| Classification | Verify independent critical states, Hessian directions and domain boundaries. Numerical case enumeration does not replace a completeness proof; positive-loss results cannot be transferred to zero loss. |
| Observation and export | Preserve unavailable fields, exact fractions and source associations. Test read-only behavior, actual report types, atomic output and cold imports separately from expensive scientific execution. |

For collective formation, a common center or bounded distance on a finite
support cannot alone distinguish a composite from ordinary passage. Retain
actual relative geometry, primitive phase and internal/boundary state;
constituent shapes and trajectories may change. Require an independently
specified signal floor, an initially absent identity and a discriminating
continued response without resetting the acquired state.

A validated reduction must dominate its approximation and preparation errors
on the claimed horizon. Keep full-form and phase margins separate, check strict
first-exit or Picard conditions, and retain operator order in noncommuting
convolutions. Failed certification leaves that claim unavailable. It does not
justify changing an evaluated allowance or replaying a frozen producer.

Model-specific coordinate caps, coefficients and acceptance inequalities belong
to their mathematical/contract owners. Read those premises when selecting
coverage; the shared checklist does not broaden any theorem's domain.

## Current checks and retained evidence

Distinguish three kinds of validation when a research owner changes:

- **Current-source contracts:** shared engine, SDK or CLI behavior, including
  numerical admission, unsupported domains, read-only behavior and atomicity.
- **Mathematical controls:** independent identities, counterexamples and exact
  coefficient probes under their hypotheses. Rational probes do not replace
  the actual irrational constants in an ideal theorem.
- **Retained evidence audits:** frozen inputs, source fingerprints, saved
  intervals/checkpoints and original verdicts, without producer replay.
  Current-source regression is separate from historical execution evidence.

The [evidence workflow](docs/guides/RESEARCH_EVIDENCE.md) owns artifact mechanics
and restoration; the [benchmark guide](benchmarks/README.md#running-and-reporting)
owns declared producer invocation. Source archives and content digests associate
records with declared code; they do not authenticate chronology or independently
revalidate omitted numerical steps. Follow the particular result owner's
retained-execution premises.

For example, the [formed C9 bundle audit](tests/physics/test_sine_formed_evidence.py)
checks committed records without replaying their producers. Its shared synthetic
helper controls are selected separately:

```sh
python -m pytest tests/physics/test_sine_formed_evidence.py tests/research/test_sine_evidence_helpers.py -q
```

The [evidence helpers](tests/sine_evidence_helpers.py) supply reusable test
mechanics, not an alternative source of mathematical conclusions. Reuse parsed
records through module fixtures and isolate mutations. Synthetic intervals and
stubbed producers test consumer logic, not the execution of the historical run.

Preserve every original failure, partial result and inconclusive verdict.
An export-recovery recomputation keeps its separate source and history; a later
successful record cannot become a successful first attempt. New analytical
handoffs or tighter current bounds leave the original evaluated response intact.
Missing committed artifacts fail their audit. Optional local records may cause
an explicit skip, never an inferred pass or automatic regeneration.

Local execution is serial by default. The main CI scheduling policy is
`python -m pytest -n 2 --dist loadfile`, retaining `not slow`. Files stay on one
worker to reuse module fixtures; workers isolate Python state. Timing benchmarks
need a separate serial run.

Choose an affected directory, module or test for bounded validation:

```sh
python -m pytest tests/core_physics -q
python -m pytest tests/operators/test_u3_hard_invariant.py -q
python -m pytest tests/sdk -q
python -m pytest tests/cli -q
python -m pytest tests/sdk --collect-only -q
```

To include all retained slow tests, use `python -m pytest tests -o addopts=""`.
To select only marked slow tests, pass their paths and `-m slow`. Inspect
`python -m pytest --markers` and collection: a registered marker can select no
tests. Standalone scripts need the editable installation or an explicit
`PYTHONPATH=src`; pytest's source-path configuration does not reach them.

## Organization

[tests/](tests) contains top-level modules and subject directories. Search for
the affected API rather than maintaining another inventory of individual tests.
Core nodal behavior is in [core_physics/](tests/core_physics), operators in
[operators/](tests/operators), specialized certificates in
[physics/](tests/physics) and public network usage in [sdk/](tests/sdk).
[CLI integration](tests/cli) checks the same study recipe through Python and
the module entry point, including malformed input, diagnostic availability,
output replacement and logging isolation. Catalog checks do not execute a study.
[conftest.py](tests/conftest.py) and [utils.py](tests/utils.py) own shared helpers.
The [core scope map](tests/core_physics/README.md) identifies the engine tests
that replace retired self-contained illustrations.

Consolidation removes work, not just collected test identifiers. Test each
invalid boundary at its shared owner, then retain distinct consumer-wiring,
rollback and representation cases instead of repeating the entire Cartesian
product at every wrapper. Reuse an expensive producer only when its results
are read without mutation; keep independent numerical oracles and negative
controls. Do not turn removed parameter grids into hidden loops or replace
meaningful assertions with finiteness checks. Operator labels alone do not
justify energy-sign or conservation assertions with arbitrary tolerances.

The maintained standalone arithmetic utility has a separate test path:

```sh
python -m pytest applications/primality-test/tests -q
```

Run it separately because its local import root differs from the core suite.
The root default selection does not include it. The
[primality guide](applications/primality-test/README.md) owns setup and scope.
The [factorization lab](applications/factorization-lab/README.md) and advanced
primality adapters are archived, outside the maintained test surface and
automatic regression mappings. Their retained tests are recovery material,
not release gates. Core arithmetic/Paley proofs and tests remain maintained.
Compatibility tests exercise absent/external-package admission without
executing the retired lab.

Research producers under [benchmarks/](benchmarks/README.md) have declared entry
points and provenance requirements; a default pytest run does not implicitly
cover them. Do not regenerate retained evidence for an unrelated change.

For examples with `run_protocol` and `build_report`, reuse the real report through
a module fixture for numerical and scope checks. The shared
[example helper](tests/example_protocol_helpers.py) checks `main` separately with
a sentinel protocol and synthetic report, without rerunning a scientific producer.

Reusable independent oracles belong in test support modules rather than in
another collected test suite. The [pair oracle helper](tests/physics/_sine_pair_oracles.py)
shares fine support, exact jets and high-precision reductions without importing
production certificates or pytest fixtures. Each consuming suite retains its
own complete-law assumptions, preparation budgets and module-scoped reports.
When optimizing a shared certificate kernel, compare complete exact projections
on representative available and unavailable cases as well as these independent
controls; timing differences alone cannot establish unchanged evidence.

[Runtime facade tests](tests/physics/test_runtime_facade_imports.py) own the two
cold import orders for the P2/REMESH example families; those checks still use
fresh processes and resolve the actual public APIs.

Cold subprocesses that test this checkout explicitly request the
`source_tree_environment` fixture from [conftest.py](tests/conftest.py) and pass
it as `subprocess.run(..., env=source_tree_environment)`. It copies the process
environment and prepends pytest's configured source paths to `PYTHONPATH`,
without changing the parent's environment. Pytest's `pythonpath` setting only
updates its own interpreter; a fresh Python process can otherwise import a
different installed package. Keep the fixture opt-in: tests of installed-package
discovery or intentional import isolation retain their declared environments.
Working directories, timeouts and optimized-mode flags remain test-specific.

The [Makefile](Makefile) target `make test` uses the routine engine selection;
`make test-all` includes the retained research suites, still excluding `slow`.
`make dev-test` adds coverage over `src`, including compatibility shims. Research
producers use the individually documented entry points in the
[benchmark guide](benchmarks/README.md), not a directory-wide pytest campaign.
`make validate` checks importability, references, documentation integrity and
SDK tests.

## Structural regression evidence

Assert the contract of the changed path. Record representation, graph/support,
coefficients, primitive triad, input history, step size and operator sequence as
relevant. An operator jump and a continuous solver step need different expected
results. For a full grammar word, verify initiation, closure and contextual
admission; distinguish fragments explicitly.

Do not assume all operators increase coherence or that SHA freezes every later
EPI update. Its capacity attenuation, pure-EPI diffusion and a multichannel
trajectory have different scopes. Inspect the
[operator contracts](src/tnfr/operators/operator_contracts.py) and
[grammar scope](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) before asserting an invariant.
Initialization may set fixture state directly; test subsequent evolution through
the API under review.

When stochastic behavior changes, control the RNG and record its seed, initial
state and execution order. A seed alone does not fix timestamps, external data
or every backend. When relevant, compare available tetrad fields with estimator
provenance and unavailable-field reasons. Diagnostic scores are not acceptance
thresholds unless that specific policy is being tested.

## Backends and optional dependencies

[tests/conftest.py](tests/conftest.py) accepts `--math-backend` and
`TNFR_TEST_MATH_BACKEND`. The command-line option takes precedence; the
selected value sets `TNFR_MATH_BACKEND` and clears the backend cache before
collection.

```sh
python -m pytest tests/mathematics/test_backends.py --math-backend=numpy -q -rs
```

Install `compute-jax` or `compute-torch` before requesting those optional
adapters. Explicit per-test backend requests are not replaced by the session
setting. Review skip reasons with `-rs` and report adapters actually exercised.
NumPy is required by the shared conftest; a missing installation is not evidence
of successful NumPy-free execution.

## Code quality and documentation checks

Install tools needed for the check; options are configured in
[pyproject.toml](pyproject.toml):

```sh
python -m pip install -e ".[dev-minimal,test-quality,typecheck]"
python -m black --check src/tnfr
python -m flake8 src
python -m pydocstyle src/tnfr
python -m mypy src/tnfr
python -m pyright src/tnfr
```

Pass affected files for a focused edit where supported. Black and isort use
88 columns; pydocstyle uses NumPy conventions. CI's advisory checks are listed
in [the workflow guide](.github/WORKFLOWS.md). `make format` modifies files;
it is not a read-only validation step.

The lazy `physics` namespace has one maintained export map and a generated
typing facade. After changing that map, run
`python scripts/generate_physics_stub.py --write`; validate it with
`python scripts/generate_physics_stub.py --check`. The generator reads the
source AST without importing TNFR, and the
[facade tests](tests/physics/test_physics_facade_imports.py) check typed exports,
runtime object identity and cold import boundaries.

For shared relational report export, select
`tests/sdk/test_relational_reports.py`. Preserve every registered owner-managed
type, actual-class and subclass admission, native label validation, exact
fractions and unavailable fields. Cold-process controls must show that export
does not import unrelated research owners. Use guarded zero-horizon reports
or synthetic wiring for producer-backed types; an SDK export test must not
regenerate a selected scientific response.

Optional pre-commit setup requires installing `pre-commit` separately before
`pre-commit install`. [.pre-commit-config.yaml](.pre-commit-config.yaml)
configures Black, isort, pydocstyle and a local Bandit-command guard. That guard
uses Bash, which must be available on Windows. The hooks do not perform a code
review; CI's formatting gate checks all tracked files selected by those hooks.

For Markdown-only changes, use the relevant reference check. For site or
documentation-infrastructure changes, also run integrity and build checks:

```sh
python scripts/verify_internal_references.py --ci
python scripts/check_documentation.py
python -m pip install -e ".[docs]"
python scripts/prepare_docs.py
python -m mkdocs build --strict
```

The reference checker accepts `--dirs` followed by files or directories to
bound the check; a missing selected path fails instead of silently checking
nothing. The staging script copies repository owners into
`build/docs-source/`; MkDocs renders that generated source into `site/`.
Edit the repository owners, never either generated tree. The documentation
stager queries Git once for recursive source trees: tracked files remain
eligible even when an ignore pattern matches, and untracked nonignored work
remains eligible. Git-ignored generated artifacts are omitted. The existing
extension and directory rules still apply within the repository; names of
parent directories do not exclude maintained sources. An extracted source tree
without Git metadata uses those positive rules alone; a failed inventory in a Git
checkout stops before replacing the previous stage.

The documentation integrity check also validates the theory catalog, its
generated navigation, operator contracts, grammar roles and glossary cards/index.
Regenerate views with
`python scripts/check_documentation.py --write-generated` after changing their
source declarations, then rerun the read-only check.
It executes the Python block in the README's Quick start section and compares
its captured output with that section's declared output. Publication metadata
uses the same strict JSON reader as the engine, including duplicate-key rejection.

## Package content and size checks

Use the release extra and inspect both built package formats before publication:

```sh
python -m pip install -e ".[release]"
python -m build
python scripts/check_distribution.py dist
python -m twine check dist/*
```

The distribution check compares package modules, stubs and schemas with the
checkout, admits only the intended archive paths and rejects linked or
malformed members. Its report exposes compressed and expanded sizes. Default
review budgets are 10 MiB compressed per archive, 32 MiB expanded per archive
and 1 MiB per file. A budget is a packaging review threshold, not a mathematical
or performance limit; inspect unexpected contents before deliberately changing
the configured limits. Use `--help` for the supported overrides.

This checks the wheel and source distribution built for PyPI. Repository
source archives and the documentation site have different contents and size
requirements; see [publication payloads](.github/WORKFLOWS.md#published-payloads).
Frozen scientific evidence must retain its original bytes and dependencies.
Removing it merely to meet a package budget would not validate the package.

## Security checks

Follow [SECURITY.md](SECURITY.md) for reporting and trust boundaries:

```sh
python -m pip install -e ".[security]"
python -m pip_audit
python -m bandit -r src -c bandit.yaml
```

The dependency audit covers installed packages, including extras actually
installed; it does not automatically cover every optional dependency. Bandit's
configured exception is recorded in [bandit.yaml](bandit.yaml). A clean report
does not guarantee that dependencies or code contain no vulnerabilities.

## Validation workflow and reporting

1. Identify the affected contract and existing owners. Reproduce a suspected bug
   before the fix when feasible; distinguish a new counterexample from a measured
   baseline.
2. Add meaningful regression coverage when behavior or an important boundary
   changes. A small documentation correction need not add numerical tests.
3. Run affected checks. Broaden to dependents, optional backends or the default
   suite when changed scope or a failure justifies it. Do not rerun unrelated
   expensive research producers as a routine checklist item.
4. Report actual commands, interpreter and relevant dependency versions,
   pass/fail/skip counts, warnings and untested scope. Do not label a failure
   pre-existing without baseline evidence.

The `structural_rng` fixture supplies NumPy's generator with seed zero.
`structural_tolerances` supplies atol=1e-12 and rtol=1e-10; these are not
exact-theorem eligibility gates. Choose scale-appropriate numerical tolerances
and keep them distinct from represented exact equality. The autouse cleanup
resets selected global state; restore additional state your test changes.

Coverage is feedback, not proof of a physical invariant. With test dependencies
installed, a selected run can generate a report:

```sh
python -m pytest tests/sdk --cov=tnfr --cov-report=term-missing
```

The current project configuration does not enforce a coverage percentage.
The Python 3.11 CI job instead uses `--cov=src` to retain the whole source-directory
scope, with `-n 2 --dist loadfile`. Pytest-cov combines the workers' data and
produces terminal and XML reports. Do not wrap distributed pytest with
`coverage run`: that alone measures the controller instead of combining worker
execution. Coverage of Python subprocesses launched inside tests is a separate
configuration and is not enabled here.

## Retained C6 campaign reproduction

The union/history report tests that reconstruct the retained B54/B55 lineage
are marked `slow`, including tests whose shared fixtures perform that work.
These fixtures can be costly; inspect their declared scope and budget before
selection. Previous timings are not runtime ceilings.
Lightweight lineage and output-overwrite rejection tests remain available when
their research paths are selected; the finite-state union oracles run
independently of the retained campaign. C6 is outside the routine engine gate.

Select the expensive report checks explicitly:

```sh
python -m pytest -m slow tests/physics/test_c6_winding_union_report.py tests/physics/test_c6_winding_history_report.py -q
```

For the independent small-state controls:

```sh
python -m pytest tests/physics/test_c6_carried_return_unions.py -q
```

This classification changes selection by cost, not assertions or scientific
scope. Report the expensive campaign as unrun when omitted, and retain failures
from earlier attempts. A passing reduced suite does not certify the historical
report or close C6 global stability.
