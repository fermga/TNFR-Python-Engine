# Dependency Vulnerability Auditing

The dependency audit workflow keeps the TNFR engine aligned with the security posture required to preserve structural coherence. The automation lives in `.github/workflows/pip-audit.yml` and relies on [`pip-audit`](https://pypi.org/project/pip-audit/) to inspect the Python packages that back the engine.

## When the audit runs

The workflow triggers automatically on:

- Pushes to the `main` or `master` branches.
- Pull requests that target `main` or `master`.
- A scheduled weekly execution every Monday at 05:00 UTC.

You can also start the run manually from the **Actions** tab by selecting _Dependency Vulnerability Audit_ and clicking **Run workflow**.

## What the workflow validates

1. Checks out the repository and sets up Python 3.11.
2. Installs the project with the full extra set via `pip install .[all]` to recreate a clean environment.
3. Determines the active site-packages directory and runs `pipx run pip-audit --progress-spinner off` restricted to that path.
4. Stores the JSON results (`pip-audit.json`) as the `pip-audit-report` artifact.
5. Fails the job whenever `pip-audit` reports unresolved vulnerabilities.

Because the audit step is allowed to complete even when it finds issues, the artifact is always generated for review before the workflow reports a failure.

## Interpreting the results

1. Open the failing workflow run in **Actions** and download the `pip-audit-report` artifact.
2. The archive contains `pip-audit.json` with the following schema for each dependency:
   - `name`: the audited package name.
   - `version`: the installed version under analysis.
   - `vulns`: a list of vulnerability objects with fields:
     - `id`: canonical identifier (e.g., GHSA, CVE).
     - `fix_versions`: secure versions published upstream.
     - `description`: human-readable summary supplied by the advisory feed.
3. Cross-reference multiple entries to check whether the issue stems from a direct dependency or a transitive package.
4. Prioritize remediation by severity (check the advisory linked in `id`) and by the operator it might compromise (e.g., `resonance`, `coherence`).

For quick triage you can render the JSON as columns locally with `pip-audit --progress-spinner off --format markdown --input pip-audit.json`.

## Remediation workflow

1. Validate whether an updated version already exists in the `fix_versions` list. If yes, bump the dependency in `pyproject.toml` (or its extras) and regenerate the lock/test baselines as needed.
2. If no secure release exists, evaluate temporary mitigations:
   - Vendor a patched fork with a short-lived extra.
   - Gate the vulnerable capability behind stricter runtime checks to reduce attack surface.
   - Remove or replace the dependency if it is not essential for TNFR coherence.
3. Document every mitigation or deferral in the pull request description, noting the advisory IDs and the structural rationale.
4. Run the full test suite to ensure the remediation maintains `C(t)` and preserves operator closure.
5. Re-run `pip-audit` locally (`pipx run pip-audit --progress-spinner off`) before opening the pull request to confirm the vulnerability set is clean.

## Handling exceptions

Only suppress an advisory when:

- There is no upstream fix and the affected surface cannot be removed without breaking TNFR invariants.
- You can demonstrate compensating controls (e.g., isolation, additional validation layers) that contain the risk.

In those situations, document the ignored advisory ID, justification, and planned follow-up in the PR to maintain traceability. Revisit every exception on a regular cadence until a permanent fix is deployed.
