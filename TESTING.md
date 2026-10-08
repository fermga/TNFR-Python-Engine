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

## Select research checks by contract

Select the changed mathematical owner and its consuming APIs through the
[theory-to-execution map](theory/README.md#theory-to-execution). The map owns
individual module/test links; proofs own model-specific hypotheses, constants
and frozen preparations. This guide groups the obligations needed to choose
coverage, rather than repeating each research result.

Shared admission changes can cross the routine/research selection boundary.
The default gate includes native relational execution through `tests/test_*.py`
and the listed conservation-diagnostic tests. It does **not** select
[`test_sine_admission.py`](tests/physics/test_sine_admission.py), the
`tests/physics/test_relational_sine_*.py` family, or the regional-transfer
research modules. For a change to the common stored-coefficient boundary,
exercise both native execution and its sine adapter explicitly:

```sh
python -m pytest tests/test_relational_exchange_execution.py tests/test_relational_regular_execution.py tests/physics/test_sine_admission.py -q
```

Then select the affected consumers from the theory-to-execution map. For a
chained report, cover the producer's primitive admission and the downstream
calculation: sampling smoothness to sample budgets, cycle modes to gains,
visible evidence to hidden-state/capacity inference and prior forecasts, or
captured states to regional currents and supplied increments. An SDK pass
checks its adapters and projection; it does not replace those mathematical
controls. A shared primitive-source change also needs the relative-pattern
and forecast-endpoint association controls, including unchanged held capacities
and partial validated horizons.

| Changed contract | Required independent controls |
| --- | --- |
| Complete law, state and coordinates | Differentiate both full nodal rows; retain support, degree normalization, capacity, clock and inputs. Compare native Arg, baseline sine and alternative-mobility laws only under their actual premises. Test moving references, common origins, zero coordinates and same-observation/different-response counterexamples. |
| Primitive and numerical admission | Reject malformed support, law identifiers, authoritative aliases, nonfinite values, Boolean physical scalars, invalid radii and missing consumed coordinates before arithmetic. Rebuild consumed geometry and gradients; cached flags or displayed intervals cannot widen exact admission. Preserve each consumer's zero-capacity/loss and work-limit domain. |
| Algebra, derivatives and storage | Use independent edge sums, high-precision evaluations, exact matrix identities or analytic solutions. Check signed work and loss, all field directions, endpoint events and strict boundaries. Symbolic irrational targets, rational probes and rounded graph states provide different evidence. |
| Relative state, uncertainty and memory | Retain hidden initialization, every environmental node, original port degrees, actual observation times and each member's conserved means. Distinguish unknown common origins, independent nodal residuals and correlated relative coordinates. Inference compatibility is not existence, identifiability or a replayed response. |
| Forecasts and continuous enclosures | Exercise strict Picard inclusion, Taylor remainders, whole-time tubes, endpoint chains and held-parameter semantics against separate analytic fixtures. Retain partial-horizon status and actual validated time. Samples, Euler chords and held-pressure integration cannot replace a coupled continuous enclosure. |
| Geometry, equilibrium and recovery | Check exact circular reconstruction, full sine-current cancellation, weighted gaps and Hessian inertia with independent full-node equations. Geometric feasibility is not equilibrium. Local recovery, all-sector capture and conservative trapping have different initial sets and loss premises. Failed sufficient inequalities remain unavailable. |
| Formation, preparation and budgets | Separate identity absent initially from a supplied target. Keep the original form cost, phase uncertainty and complete law in positive and negative controls. Distinguish scalar storage, directional loss, transient barrier passage, maintained endpoints and accumulated port work; an instantaneous sign is not an integrated supply. |
| Symmetry and composition | Test whole-support/state/capacity equivariance, cycle orientation, relabeling and genuine quotient reconstruction. Ordinary reflection, combined sign/reflection and set-invariant uncertainty boxes have different consequences. Retain external attachments, relative zero modes and noncommuting weighted operators. |
| Collective observations and structural events | Check all competitors, mutuality, unresolved ties and state/support admission separately. Compare quotient and inherited rates with the full fine field. Hypothetical bridges, cuts, AL proposals and resets retain their work budgets and read-only scope; conservation or eligible contact does not prove occurrence. |
| Resonance, pulse and recurrence | Retain the selected port/readout, full tangent pencil and actual coefficient domain. Distinguish gain peaks from complex poles, exact periodicity from orbital stability, and family recurrence from a chosen-state verdict. Independent quadrature or static variation controls do not prove a nonlinear infinite-time response. |
| SDK, CLI and evidence projection | Check delegation, one-source capture, immutable observations, unavailable values, supported node labels and exact fraction export. Direct report schemas and the generic SDK envelope are separate contracts. Malformed reports and source changes must not authenticate themselves through serialization. |

### Boundaries that need explicit regression coverage

- **Chained evidence is rebuilt from its premises.** Change primitive inputs
  independently of cached fields, and check that downstream bounds are
  recomputed or unsupported declarations reject. Retain valid stored
  coefficients without renormalizing them. Test stale favorable flags, zeroed
  derivative bounds and inconsistent state/support associations with independent
  expected results. Invalid source declarations must fail before a solver runs
  or independent one-shot samples are consumed; a supplied signed endpoint
  increment must not be replaced by an evaluated form rate.
- **Shared admission does not mean shared theorem domains.** Specialized
  sine-cycle recovery retains its larger support domain (including the
  51-node control); general phase geometry has separate work caps. The
  validated Taylor/comparison owner has a 24-coordinate limit, with boundary
  refusal and independent 23-coordinate controls. Its sine layout uses
  `2*n+1` coordinates; dimension admission alone certifies no response.
- **Metric uncertainty remains correlated.** Compare retained-ball propagation
  with independent analytic rotating, contracting and nonlinear flows, including
  boundary points outside coordinate axes. Check original-clock growth bounds,
  both directions, every domain margin and partial coverage. Include tiny local
  errors whose squares fall below the interval grid; normalized norm arithmetic
  must not create an artificial square-root precision floor. A coordinate
  projection is not the ball consumed by the next step, and a manufactured
  solver control is not a formation experiment.
- **Exact and represented geometry remain distinct.** A rounded `pi`, a
  small field residual or overlapping intervals cannot establish an exact
  antipodal state, equilibrium or symmetry. Test zero resultants, branch
  limits, unresolved denominators and strict equality boundaries explicitly.
  An admitted native proposal chord is not a continuous trajectory proof.
  Exact `Fraction` input to a graph may pass through binary64 capture;
  distinguish that source from an explicitly supplied rational preparation.
  An unrecognized symbolic cancellation with a zero-containing interval
  must remain unavailable, not become either equality or a counterexample.
- **Preparation and endpoint sets retain their provenance.** A tighter
  correlated source bound cannot replace a forecast's larger Cartesian
  endpoint enclosure. Partial forecasts keep their validated time and
  original requested-horizon status. Full-state capture must include remaining
  form storage, every sector face and the actual phase uncertainty; favorable
  phase geometry alone is insufficient.
- **Reduction retains information and scale.** Memory/second-order identities
  keep initial form, the weighted constant mode and moving history. Slow-phase
  controls retain the fast transient, memberwise references, original form
  remainder and preparation cost. Test zero horizon, tiny positive feedback,
  large growth and monotone exponential tails without silently clipping time.
  For collective contact observations, independently vary internal deviations
  at fixed means and verify the complete nodal response. Declared contact
  lifts can change a collective energy split while full storage is unchanged;
  test its signed remainder and exact local derivatives without treating them
  as a finite-horizon prediction. Exact-family membership, finite-width
  maintenance and acquisition into that neighborhood need separate controls.
  For reduced component ports, independently verify projection/lift identities,
  orbit multiplicities, the nonlinear bridge and its changed degree weights.
  Include odd initial errors and nonlinearly generated discarded modes; linear
  parity invariance is not an exact nonlinear quotient. Transfer the unchanged
  component law to the held-out receiver and compare total prediction error
  against the final outward full-response lower bound, retaining preparation
  and readout errors. Prove actual joined identity from the full law, not from
  the surrogate's stability or coordinate count.
  For network assembly, derive central mobility from every live contact
  degree and check exact charge/storage identities against the fine graph.
  Include a multiply connected port that rejects unchanged one-contact
  normalization, cycles with consistent component origins, and disconnected
  states whose instantaneous rows do not establish global capture. A uniform
  whole-state approximation bound must retain original preparation errors
  and generated odd modes; it is not a receiver-contrast or sensor budget.
- **All-time and symmetry verdicts have different scope.** Fixed-budget
  consensus tests need an independent mixed-Lyapunov derivative from both
  full rows, zero-budget and failed-premise controls. Reflection excludes
  winding only at nonantipodal observations; it proves neither antipodal
  avoidance nor consensus. Equal-budget sources, tiny exact asymmetries and
  combined sign/reflection provide discriminating controls.
- **Classification is not a numerical case survey.** Exact factorized
  critical sets, full-support Hessian congruences and mode counts need
  independent witnesses and invalid-domain controls. Do not replace their
  completeness proofs by enumerating thousands of trajectories or transfer
  a positive-loss convergence result to zero loss or frozen capacities.

## Current checks and retained evidence

Distinguish three kinds of validation when a research owner changes:

- **Current-source contracts:** exercise the shared engine, SDK or CLI, including
  numerical admission, unsupported domains, read-only behavior and atomicity.
- **Mathematical controls:** check independent identities, counterexamples or
  exact coefficient probes under their stated hypotheses. Rational probes do
  not replace the actual irrational constants in an ideal theorem.
- **Retained evidence audits:** validate frozen inputs, source fingerprints,
  saved intervals/checkpoints and original verdicts without rerunning a producer.
  A current-source regression is separate from authenticating historical evidence.

For example, the [composition](theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md)
and [memory](theory/nodal/RELATIONAL_PATTERN_MEMORY.md) owners identify their
static controls and retained finite responses. The
[relational admission owner](theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
identifies recovery, formation and capture obligations. A successful current
snapshot or a passing subset must not rewrite a stopped or negative historical
verdict. Check the selected fixture before claiming that a run has no evolution.

The [benchmark guide](benchmarks/README.md#running-and-reporting) owns producer
invocation, freezing and artifact lifecycle. Reuse the appropriate saved evidence
or shared fixture; do not regenerate completed responses for unrelated changes.
A new reserved prediction needs its own declared inputs and prospective protocol.

The [formed C9 bundle audit](tests/physics/test_sine_formed_evidence.py) checks
the committed artifact hashes, archived source inventories, protocol associations
and both reduced-model stopping rules without calling an assessor or producer:
the two-component error fraction uses its final outward receiver-gap lower
endpoint, while network composition uses an absolute whole-window full-state
allowance plus the exact degree/charge/storage control. Neither rule substitutes
for its saved actual-family identity and supplied-work obligations.

```sh
python -m pytest tests/physics/test_sine_formed_evidence.py -q
```

Missing committed artifacts fail this check. Historical source need not match
the current implementation; current-source regression remains separate. The
original pair/probe records still lack an evaluation-time source snapshot.
Content consistency neither authenticates chronology nor proves the mathematics.

Optional retained-record audits skip explicitly when local evidence is absent;
they must not recreate a producer or count missing evidence as a passed response.
Synthetic intervals, step records and stubbed producers test consumer logic,
not historical execution. Preserve original source archives, failures and
inconclusive verdicts; corrections require separately identified evidence.
Forecast readers recompute their consumed admission and Picard inclusion, but
do not replay the full Taylor proof or authenticate record production. Test
that boundary explicitly: partial coverage remains partial, and failure to
certify a requested duration or margin is not an exclusion of actual identity.
Retrospective channel integration from saved whole-time tubes must preserve
the original verdict and receive its own analysis provenance.

Local execution is serial by default. To use the main CI scheduling policy,
run `python -m pytest -n 2 --dist loadfile`; this keeps the same `not slow`
selection and assertions. Each file stays on one worker to reuse its module
fixtures. Workers isolate Python state and prepare session fixtures independently.
Pytest-benchmark does not collect timing measurements
under xdist; performance measurements need a separate serial run.

Choose an affected directory, module or test for bounded validation:

```sh
python -m pytest tests/core_physics -q
python -m pytest tests/operators/test_u3_hard_invariant.py -q
python -m pytest tests/sdk -q
python -m pytest tests/cli -q
python -m pytest tests/sdk --collect-only -q
```

To include all retained slow tests, use `python -m pytest tests -o addopts=""`.
To select only marked slow tests, use `python -m pytest -m slow` with those
paths. Inspect `python -m pytest --markers` and collection before treating
a marker-selected run as coverage: a registered marker can select no tests.

Standalone scripts do not inherit pytest's source-path configuration. Use the
editable installation above or explicitly set the shell's `PYTHONPATH` to
`src` so a previously installed release cannot replace the working source.

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

The separately packaged arithmetic applications have their own test paths:

```sh
python -m pytest applications/factorization-lab/tests -q
python -m pytest applications/factorization-lab/benchmarks/test_benchmark_suite.py -q
python -m pytest applications/primality-test/tests -q
```

Run these as separate invocations: their local import roots differ from the core
suite. The root default selection does not include them. See the corresponding
[factorization guide](applications/factorization-lab/README.md) and
[primality guide](applications/primality-test/README.md) for application setup and scope.

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
bound the check. The staging script copies repository owners into
`build/docs-source/`; MkDocs renders that generated source into `site/`.
Edit the repository owners, never either generated tree. The documentation
integrity check also validates the theory catalog, its generated navigation,
operator contracts, grammar roles and glossary cards/index. Regenerate views with
`python scripts/check_documentation.py --write-generated` after changing their
source declarations, then rerun the read-only check.
It executes the Python block in the README's Quick start section and compares
its captured output with that section's declared output. Publication metadata
uses the same strict JSON reader as the engine, including duplicate-key rejection.

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
During the 2026-09-19 release check, one union-report fixture consumed more
than 20 minutes of CPU time. This observed cost is not a runtime ceiling.
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
