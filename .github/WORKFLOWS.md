# GitHub Actions Workflows

This page owns the workflow inventory. Linked YAML is authoritative for triggers,
permissions, conditions and commands. [TESTING.md](../TESTING.md) owns local
validation instructions. Configuration describes what a run attempts; it does
not establish current GitHub run health, branch protection or enabled secrets.

## Maintained workflows

| Workflow | Trigger and configured scope |
| --- | --- |
| [ci.yml](workflows/ci.yml) | Push and PR to main/master. Blocking formatting hooks, flake8 and default pytest selection on Python 3.10–3.13 with two pytest-xdist workers; combined coverage report on 3.11. Pydocstyle, mypy, pyright and vulture report advisories. |
| [docs.yml](workflows/docs.yml) | Filtered push/PR to main and manual invocation. References, documentation integrity, staging and strict MkDocs build; publishes Pages only on pushes to main. |
| [verify-references.yml](workflows/verify-references.yml) | Filtered Markdown/notebook changes on PRs and pushes to main/develop. Independent local Markdown target and fragment check. |
| [pip-audit.yml](workflows/pip-audit.yml) | Push/PR to main/master, Monday 05:00 UTC and manual invocation. Audits the installed core, test-all and serialization environment; captures available reports and fails when the JSON audit fails. |
| [pypi-zenodo.yml](workflows/pypi-zenodo.yml) | Published GitHub release or manual invocation. Builds/checks PyPI distributions and an independent lightweight development bundle. Publishes to PyPI on release or when the existing manual force-publish input is true; attaches the bundle only on release events. |
| [lint-workflows.yml](workflows/lint-workflows.yml) | Filtered workflow/script changes on PRs and pushes to main. Searches for unsupported Bandit SARIF invocation; this is not a general Actions schema linter. |
| [copilot-setup-steps.yml](workflows/copilot-setup-steps.yml) | Manual invocation or edits to its own workflow. Installs a Python 3.11 development/test environment and verifies importability. |

Black and isort hooks exclude `docs/assets/`, `build/`, the archived
`applications/factorization-lab/` and `applications/primality-test/archive/`.
These retain frozen programs or historical source outside the maintained
execution surface. Their bytes must remain available for provenance and
recovery checks; formatting maintained sources does not authorize rewriting
those copies.

The default pytest selection is the routine engine/API gate and excludes slow
tests as configured in [pyproject.toml](../pyproject.toml). The full retained
research inventory is collected on Python 3.11 to detect import and test-name
collisions, but its research fixtures are not executed by that collection.
Execution is explicitly selected with `pytest tests`; it is not replayed by
every CI job. CI does not declare a universal coverage
percentage, prove physical theorems or certify every optional backend. Audit
coverage excludes optional compute, documentation and deployment environments
unless their dependencies happen to be installed through the declared groups.

The factorization lab and advanced primality adapters are retired, not implicit
CI or self-optimization regression targets. The basic primality utility has the
separate maintained test selection in [TESTING.md](../TESTING.md).

The main test matrix uses `-n 2 --dist loadfile`; worker scheduling changes
execution order, not test selection or assertions. Each file stays on one worker,
so its module fixtures are reused across its tests. Workers own separate Python
state and session fixtures. Tests that write files must retain independent
temporary paths. Python 3.11 uses pytest-cov to combine
worker coverage over `src`.
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
Changes to either operator contracts or grammar roles also trigger the gate,
so registry edits cannot bypass checks of their generated documentation tables.
Edit source owners, not generated site copies.
The site includes retained evidence and linked source files. Recursive staging
uses the checkout inventory to omit ignored generated artifacts while retaining
tracked evidence and untracked nonignored work; source archives without Git
retain the declared positive copy rules. See the
[staging checks](../TESTING.md#code-quality-and-documentation-checks).
Pages publishes `site/` to `gh-pages` with `force_orphan: true`, replacing the
deployment history rather than accumulating a copy of every former site.

## Package publication

[pypi-zenodo.yml](workflows/pypi-zenodo.yml) installs the checkout and its declared
core dependencies with
the build tools so metadata validation can use the shared JSON reader. It then
checks publication metadata and, for a release event,
requires the tag to match the declared package version. It then performs build,
[distribution content/size](../TESTING.md#package-content-and-size-checks)
and Twine checks and publishes using the repository secret `PYPI_API_TOKEN`,
which must contain an API token scoped to the `tnfr` PyPI project.
This token-based path disables attestations and does not request an OIDC token;
it does not provide trusted-publishing attestations. It neither creates a
release nor runs the numerical test suite.
Release preparation must consider relevant validation evidence; the workflow
does not itself enforce a dependency on successful CI runs. A normal manual
invocation builds/checks only; its existing force-publish input enables publishing.

An independent `development-bundle` job uses the standard-library builder
`scripts/build_development_bundle.py` to create the mandatory
`TNFR-<version>-light-development.zip` and `.zip.sha256` from the checked-out
commit. It validates the release tag/version and verifies the extracted bundle
with its included evidence helper. It requires neither PyPI credentials nor
engine dependencies, and its output is outside `dist/`. Both manual and release
runs retain an Actions artifact. Only release events attach the ZIP and sidecar
to the existing release; only this job has `contents: write`. Existing matching
assets are accepted by size and SHA256; conflicting assets fail without
overwriting. This job's checksum/extraction checks do not replace the
[onboarding and selected execution checks](../TESTING.md#lightweight-development-bundle)
required during release preparation.

Publishing a GitHub release therefore also requests PyPI publication. Before
that trigger, review validation runs for the exact commit being released:
blocking CI formatting/static checks and its test matrix (including SDK tests),
the dependency audit, and the documentation/reference checks selected by that
commit's paths. Check the workflow-invocation lint result when it is triggered.
A skipped, cancelled or absent run is not a passing result, and success on an
earlier revision does not validate the release commit. Advisory findings remain
advisory; assess their relevance without presenting them as enforced gates.

The final Zenodo step only prints a reminder. A separately configured external
integration may archive a release; this workflow does not upload an archive or
verify a DOI. No GitHub/PyPI/Zenodo settings are inferred from this file.

### Published payloads

| Output | Contents and publication path |
| --- | --- |
| PyPI wheel | Installable `tnfr` package, typing/schema data and package metadata. The publication job uploads built distributions; it does not upload the repository or site. |
| PyPI source distribution | Maintained build inputs selected by [MANIFEST.in](https://github.com/fermga/TNFR-Python-Engine/blob/main/MANIFEST.in), including package source and small root metadata. Research evidence, tests, generated sites and optional application trees are outside this package payload. |
| Lightweight development ZIP | Exact source snapshot for agent/developer handoff: code, theory, documentation, instructions, examples and tests, with generated onboarding, a manifest and verified retrieval of omitted `docs/assets/` evidence. The publication workflow attaches it and its SHA256 sidecar to each GitHub release. It contains neither Git history nor installed dependencies. |
| GitHub source ZIP/TAR | The release tag's repository tree, including committed research evidence. GitHub's source downloads are distinct from wheel/sdist assets and do not contain Git history. |
| Zenodo archive | The complete source archive retained through the separately configured GitHub integration. The lightweight development ZIP does not replace this evidence archive or require another deposit. Verify the actual version DOI and archive after publication. |
| Documentation site | Rendered documentation plus retained downloadable source/evidence, deployed separately to `gh-pages`. Its size is not the installed package size. |
| Actions reports | JUnit, coverage and dependency-audit reports retained by their respective jobs; they are not package contents or release archives. |

Repository archives can grow as frozen responses and source snapshots are
retained. A compressed evidence archive may expand substantially, and repeated
source records can be required to reproduce different evaluated runs. Preserve
those records rather than treating all archive growth as generated waste.
Review the distribution check's contents and byte sizes to identify which
payload actually changed.

## Research verification boundary

Ordinary CI exercises the declared engine/API selection. Research campaigns
use the explicitly selected owners in [Testing](../TESTING.md) and
[benchmarks](../benchmarks/README.md). A routine pass is not an automatic
benchmark, reproducibility certificate or replay of every retained experiment.
Retirement and recovery records belong to the
[archive](../theory/research/archive/README.md), not this workflow inventory.

## Maintenance rules

Keep dependency groups and interpreter matrices consistent with
[pyproject.toml](../pyproject.toml). Check referenced paths and distinguish
advisory steps from gates. When retiring automation, preserve useful evidence
and update this inventory. Never report a past finding count, external setting
or passing run as a permanent property of the repository.
