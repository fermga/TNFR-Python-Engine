# GitHub Actions Workflows

This page owns the workflow inventory. Linked YAML is authoritative for triggers,
permissions, conditions and commands. [TESTING.md](../TESTING.md) owns local
validation instructions. Configuration describes what a run attempts; it does
not establish current GitHub run health, branch protection or enabled secrets.

## Maintained workflows

| Workflow | Trigger and configured scope |
| --- | --- |
| [ci.yml](workflows/ci.yml) | Push and PR to main/master. Blocking formatting hooks, flake8 and default pytest selection on Python 3.10–3.13 with two pytest-xdist workers; combined coverage report on 3.11. Pydocstyle, mypy, pyright and vulture report advisories. |
| [tests.yml](workflows/tests.yml) | Push and PR to main. Focused SDK tests on Python 3.11; overlaps the broader CI selection. |
| [docs.yml](workflows/docs.yml) | Filtered push/PR to main and manual invocation. References, documentation integrity, staging and strict MkDocs build; publishes Pages only on pushes to main. |
| [verify-references.yml](workflows/verify-references.yml) | Filtered Markdown/notebook changes on PRs and pushes to main/develop. Independent local Markdown target and fragment check. |
| [pip-audit.yml](workflows/pip-audit.yml) | Push/PR to main/master, Monday 05:00 UTC and manual invocation. Audits the installed core, test-all and serialization environment; captures available reports and fails when the JSON audit fails. |
| [pypi-zenodo.yml](workflows/pypi-zenodo.yml) | Published GitHub release or manual invocation. Builds and checks distributions; publishes on release or when the existing manual force-publish input is true. |
| [lint-workflows.yml](workflows/lint-workflows.yml) | Filtered workflow/script changes on PRs and pushes to main. Searches for unsupported Bandit SARIF invocation; this is not a general Actions schema linter. |
| [code-review.yml](workflows/code-review.yml) | PR opened, synchronized or reopened on main. Advisory Black/mypy checks and an automated PR comment; no external code-review service is implemented. |
| [copilot-setup-steps.yml](workflows/copilot-setup-steps.yml) | Manual invocation or edits to its own workflow. Installs a Python 3.11 development/test environment and verifies importability. |

The default pytest selection excludes slow tests as configured in
[pyproject.toml](../pyproject.toml). CI does not declare a universal coverage
percentage, prove physical theorems or certify every optional backend. Audit
coverage excludes optional compute, documentation and deployment environments
unless their dependencies happen to be installed through the declared groups.

The main test matrix uses `-n 2 --dist loadfile`; worker scheduling changes
execution order, not test selection or assertions. Each file stays on one worker,
so its module fixtures are reused across its tests. Workers own separate Python
state and session fixtures. Tests that write files must retain independent
temporary paths. Python 3.11 uses pytest-cov to combine
worker coverage over `src`, preserving the previous source-directory scope.
The existing configuration does not enable coverage of separate Python
subprocesses launched by tests.

Pytest-benchmark automatically disables timing measurements under xdist, while
still running benchmarked functions and their assertions. The current `tests/`
suite has no consumers of its benchmark fixture; this matrix is not a timing
benchmark. Run future performance measurements separately without xdist.

Each Python test job attempts to retain its JUnit report for 14 days, including
when tests fail. The 3.11 job also retains its combined coverage XML when
produced. A report from an interrupted run covers only the tests reached;
its presence does not establish completion or success.

## Documentation publication

The documentation workflow validates repository sources, stages them through
[prepare_docs.py](../scripts/prepare_docs.py), builds with strict MkDocs settings
and publishes that site on eligible pushes. PRs validate without publication.
Edit source owners, not generated site copies.

## Package publication

The retained publisher is [pypi-zenodo.yml](workflows/pypi-zenodo.yml); its filename
is historical. It first checks publication metadata and, for a release event,
requires the tag to match the declared package version. It then performs build
and Twine checks and requests PyPI trusted publishing. It neither creates a
release nor runs the numerical test suite.
Release preparation must consider relevant validation evidence; the workflow
does not itself enforce a dependency on successful CI runs. A normal manual
invocation builds/checks only; its existing force-publish input enables publishing.

Publishing a GitHub release therefore also requests PyPI publication. Before
that trigger, review validation runs for the exact commit being released:
blocking CI formatting/static checks and its test matrix, focused SDK tests,
the dependency audit, and the documentation/reference checks selected by that
commit's paths. Check the workflow-invocation lint result when it is triggered.
A skipped, cancelled or absent run is not a passing result, and success on an
earlier revision does not validate the release commit. Advisory findings remain
advisory; assess their relevance without presenting them as enforced gates.

The final Zenodo step only prints a reminder. A separately configured external
integration may archive a release; this workflow does not upload an archive or
verify a DOI. No GitHub/PyPI/Zenodo settings are inferred from this file.

## Retired automation

The second documentation cleanup removed stale workflows after preserving their
original files under
`artifacts/research/documentation_second_cleanup_originals_2026_09_19/`.
The reasons are recorded here to prevent restoring broken obligations:

- The former push-to-main release workflow duplicated PyPI publication and
  depended on absent changelog configuration, fragment checker and test script.
  The missing changelog job was also removed from CI.
- The performance guard targeted an absent performance-test directory.
- The tetrad smoke workflow invoked an absent runner.
- The manual reproducibility workflow targeted an absent profiler and manifest
  comparison script.

Retained CI provides the stated checks; it is not equivalent replacement
coverage for those retired benchmark protocols. Research checks use their
maintained, explicitly documented entry points in
[benchmarks/README.md](../benchmarks/README.md). There is no automatic benchmark
or reproducibility guarantee implied by an ordinary CI pass.

## Maintenance rules

Keep dependency groups and interpreter matrices consistent with
[pyproject.toml](../pyproject.toml). Check referenced paths and distinguish
advisory steps from gates. When retiring automation, preserve useful evidence
and update this inventory. Never report a past finding count, external setting
or passing run as a permanent property of the repository.
