# GitHub Actions Workflows

This page describes workflow intent. Workflow YAML is the source of truth for
triggers, matrices, permissions, and commands.

## Active validation

| Workflow | Responsibility |
| --- | --- |
| `ci.yml` | Formatting, lint, advisory type analysis, changelog policy, and tests on Python 3.10-3.13 |
| `docs.yml` | Internal paths and fragments, documentation integrity, strict MkDocs build, and Pages publication from main |
| `verify-references.yml` | Independent internal-reference validation for Markdown changes |
| `tests.yml` | Additional test execution |
| `reproducibility.yml` | Repeated seeded benchmark comparison |
| `performance-regression.yml` | Performance regression checks |
| `pip-audit.yml` | Dependency vulnerability audit |
| `release.yml` | Versioned release and publication |

## Documentation publication

There is one documentation build and deployment workflow:

1. install the project with the `docs` extra;
2. validate local Markdown targets and GitHub-style fragments;
3. run executable documentation integrity checks;
4. stage canonical repository sources with `scripts/prepare_docs.py`;
5. build MkDocs with strict mode;
6. publish the exact `site/` artifact on pushes to `main`.

Pull requests build and validate the same site without publishing it.

## Maintenance rules

- Keep supported Python versions synchronized with `pyproject.toml`.
- Keep commands in this page descriptive; copy exact command details only when
  they are stable public interfaces.
- Treat advisory jobs as advisory in both prose and YAML.
- Do not report workflow health or finding counts as permanent facts here.
- Update this page when a workflow is added, removed, or changes purpose.
