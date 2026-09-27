# TNFR Testing Guide

This page owns local validation instructions. [pyproject.toml](pyproject.toml)
defines tools and dependencies; [the workflow guide](.github/WORKFLOWS.md) owns
CI behavior. Expected mathematics and contracts follow [AGENTS.md](AGENTS.md)
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

For the foundational F1-F4 cycle, the shared regional/source-relative form
observations and affine-closure admission are production contracts covered by
the routine gate.
The [relational execution controls](tests/test_relational_exchange_execution.py)
also belong to that gate: they exercise the shared opt-in field, atomic step,
numerical admission, zero capacities and SDK delegation. Its theorem and
countermodel checks remain explicit research selections.
The [regular-chamber controls](tests/test_relational_regular_execution.py)
and [rational cosine bounds](tests/test_phase_resultant_chamber.py) cover the
opt-in wider executor in the routine gate. They check the analytic crossing
tangent, whole-step rejection despite regular endpoints, atomicity and
independent trigonometric admission without running a formation campaign.
The [protected-capture controls](tests/test_relational_capture.py) run in the
routine gate. They check exact symmetry and energy admission, three distinct
target basins, current winding versus limiting sector, unsupported domains,
read-only SDK delegation and exact export. They neither integrate trajectories
nor replace mathematical proofs with a finite pass. The symbolic basin and
saddle controls remain in the explicit formation-admission research selection.
The same routine owner also checks the full-state local basin without exact
reflection, using affine-pi quotient distances and enclosed excess storage.
It also checks the full-state acute-sector barrier with independently admitted
edge lifts, cycle periods and energy margins, without a symmetry tolerance.
Positive heterogeneous capacities, beta/form rescaling, unavailable future
regularity bounds and strict zero-capacity exclusion are covered explicitly.
The upper-corner [response controls](tests/physics/test_relational_capture_response.py)
test protocol freezing, stopped-state accounting and retained endpoint
certificates without repeating the three-grid evolution.
The [continuous-transit controls](tests/test_relational_transit.py) exercise
the reduced/full-law correspondence, short analytic solutions, retained
central resultant, read-only execution and explicit unavailable prefixes.
Independent rational references test interval trigonometry, Taylor derivative
tails and signed-diagonal comparison. The separately selected
[proof-record controls](tests/physics/test_relational_transit_proof.py) bind
the exact archived source, original seed/verdict, every saved Picard tube and
whole-endpoint basin inequalities; they do not repeat the 256-step proof.
The [zero-form controls](tests/physics/test_relational_zero_form.py) check the
single intervention, initial phase-to-form derivatives, explicit competing
outcomes, immutable source and every retained tube without rerunning its
trajectory. They distinguish proved consensus from unavailable capture and
label transient winding at saved endpoints as a post-evaluation deduction.
Three-basin and extended arctangent derivative controls preserve the prior
positive-only default and the original small-argument arithmetic.
The [equal-storage reversal controls](tests/physics/test_relational_reversed_form.py)
compare exact initial graph energies, signed reduced coordinates and fresh
engine phase rates. They check frozen intervention admission and distinguish
different limits from unchanged or unavailable outcomes. Protocol tests do
not regenerate a trajectory; retained-evidence checks use saved interval tubes.
The [shared pattern observations](tests/test_relational_pattern_observation.py)
and [SDK report controls](tests/sdk/test_relational_reports.py) also run in the
routine gate. They check supplied real phase lifts, complete centered
coordinates, unavailable transport domains, read-only fresh pressure, retained
checkpoint agreement and exact JSON projection. They do not repeat the frozen
recovery or interaction trajectories.
The [signed-work controls](tests/test_relational_work_observation.py) check
independent graph gradients, sign reversal, stationary/uniform-form limits,
regional partitions, actual-rate defects and zero-capacity availability.
The [shared cut controls](tests/test_regional_support_cut.py) cover full support,
complements and primitive revalidation while preserving legacy transport
admission. These static controls and SDK exports add no temporal campaign.
The [regional phase-response controls](tests/test_relational_phase_response.py)
check an analytic zero-cut path response, separate capacity and metric effects,
independent edge accounting, exact covariance bounds and rate-rounding defects.
They retain zero-capacity admission, single/full regions, partitions, sign
reversal and relabeling. A static mixed-state witness shares ten coarse
coordinates and a zero cut while its regional phase response differs.
The [generic linear observation controls](tests/test_linear_observation.py)
check exact invariant-row closure, complete stabilization, resource admission
and minimal state reconstruction independently of a graph or diffusion law.
Existing affine/forced realization tests retain their original negative
generator sign, reports and counters. The explicitly selected
[local composition controls](tests/physics/test_relational_local_composition.py)
check fixed rational coefficient probes, the actual-trigonometric Jacobian
against static native differences and the nonlinear same-observation witness.
The same owner checks equal coarse state/rate with unequal acceleration, using
exact rational algebra probes and independent static directional differences
of the native field. Equal initial energy/loss does not erase that obstruction.
They do not substitute rational values for the ideal constants in a theorem
or evolve a new research trajectory.
The [joint memory controls](tests/physics/test_relational_pattern_memory.py)
check the complete quotient reconstruction, independent hidden generator,
energy split, lossless boundary and nonlinear symmetry against native fields.
An analytic prepared-even witness tests hidden generation and the leading
cubic visible residual. Static consistency does not certify a trajectory
error or supply the theorem's neighborhood/derivative bounds numerically.
The [memory coefficient controls](tests/physics/test_relational_memory_prediction.py)
check independently derived quadratic/cubic terms, generated feedback, parity,
polarization, one-step causal staging and a native static Taylor probe.
The [memory-response controls](tests/physics/test_relational_memory_response.py)
validate preparation isolation, immutable source/protocol binding and retained
endpoint errors by independent graph-energy accounting. They reuse the first
three-grid response rather than repeating its trajectories; ideal rational
means and represented matrix arithmetic have separate rounding boundaries.
The [capacity-response controls](tests/physics/test_relational_capacity_response.py)
reuse one [read-only audit](benchmarks/relational_capacity_audit.py) for retained
preparation, checkpoint and frozen-decision consistency. Adversarial mutations
check that matching the nodal product alone cannot authenticate pressure or
the other stored fields. These checks do not rerun the twelve trajectories;
current snapshot replay is separately available only when the recorded
source/runtime fingerprints match. A later incompatible environment does not
retroactively change the historical numerical verdict.
The [joint local-recovery controls](tests/physics/test_relational_local_recovery.py)
check the conditional tangent dynamics and retained two-grid C5 response, with
actual engine execution at the excluded zero-capacity boundary. The theorem
requires positive capacities and EPI dissipation; runtime admission alone
does not assert recovery. Reuse retained evidence instead of rerunning the
full finite campaign for each assertion.
The [paired-region controls](tests/physics/test_relational_region_interaction.py)
reuse that same law, plus the existing regional support-balance observer.
Independent initial-acceleration and equal-summary counterexamples check
transmission and the limits of aggregate state; retained three-grid records
and detached snapshots supply the finite evidence without repeating trajectories.
The [formation-admission controls](tests/physics/test_relational_formation_admission.py)
are an explicit research selection. Exact rational-turn geometry, symbolic
storage barriers and native-pressure/tangent checks distinguish constitutive
boundaries from the acute executor. They perform no formation trajectory or
parameter sweep and leave the frozen recovery/interaction campaigns intact.
The [bounded crossing controls](tests/physics/test_relational_formation_response.py)
separately check the frozen protocol, stop/atomicity reporting and retained
checkpoint geometry through shared observers. They do not repeat its three
grids or treat an uncompleted horizon as a final recovery verdict.
The conditional-law and frozen-response controls remain explicit research
selections at their [theoretical owner](theory/nodal/DERIVED_FORM_PHASE.md).
The [F4 tests](tests/physics/test_source_relative_form_response.py) prepare a
current-source prediction in memory and reuse one four-trajectory fixture;
they neither rewrite the retained artifacts nor certify that the current
source equals the first evaluated source. Historical fingerprint verification,
current regression execution and a new reserved prediction are separate
operations. The [benchmark guide](benchmarks/README.md#running-and-reporting)
owns their artifact lifecycle.

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

[tests/](tests/) contains top-level modules and subject directories. Search for
the affected API rather than maintaining another inventory of individual tests.
Core nodal behavior is in [core_physics/](tests/core_physics/), operators in
[operators/](tests/operators/), specialized certificates in
[physics/](tests/physics/) and public network usage in [sdk/](tests/sdk/).
[CLI integration](tests/cli/) checks the same study recipe through Python and
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
python -m pytest factorization-lab/tests -q
python -m pytest factorization-lab/benchmarks/test_benchmark_suite.py -q
python -m pytest primality-test/tests -q
```

Run these as separate invocations: their local import roots differ from the core
suite. The root default selection does not include them. See the corresponding
[factorization guide](factorization-lab/README.md) and
[primality guide](primality-test/README.md) for application setup and scope.

Research producers under [benchmarks/](benchmarks/README.md) have declared entry
points and provenance requirements; a default pytest run does not implicitly
cover them. Do not regenerate retained evidence for an unrelated change.

For examples with `run_protocol` and `build_report`, reuse the real report through
a module fixture for numerical and scope checks. The shared
[example helper](tests/example_protocol_helpers.py) checks `main` separately with
a sentinel protocol and synthetic report, without rerunning a scientific producer.
[Runtime facade tests](tests/physics/test_runtime_facade_imports.py) own the two
cold import orders for the P2/REMESH example families; those checks still use
fresh processes and resolve the actual public APIs.

The [Makefile](Makefile) target `make test` uses the routine engine selection;
`make test-all` includes the retained research suites, still excluding `slow`.
`make dev-test` adds coverage over `src`, including compatibility shims. Research
producers run through their explicit targets, such as `make riemann-benchmark`.
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
python -m flake8 src/tnfr
python -m pydocstyle src/tnfr
python -m mypy src/tnfr
python -m pyright src/tnfr
```

Pass affected files for a focused edit where supported. Black and isort use
88 columns; pydocstyle uses NumPy conventions. CI's advisory checks are listed
in [the workflow guide](.github/WORKFLOWS.md). `make format` modifies files;
it is not a read-only validation step.

Optional pre-commit setup requires installing `pre-commit` separately before
`pre-commit install`. [.pre-commit-config.yaml](.pre-commit-config.yaml)
includes local Bash hooks, so Windows needs Bash available to those hooks. The
local code-review hook prints a reminder; it does not perform a review.

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
bound the check. The staging script copies repository owners into the generated
site; edit those owners rather than generated copies.

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
