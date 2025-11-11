# Dependabot Pull Request Flow

Dependabot keeps the TNFR engine aligned with secure dependency and workflow
baselines. This document describes how the automation is configured and how to
review the pull requests it opens without compromising the canonical TNFR
invariants.

## Automation settings

- **Ecosystems** — Python packages resolved from `pyproject.toml` via the
  `pip` ecosystem and the GitHub Actions workflows under `.github/workflows`.
- **Frequency** — Weekly on Mondays at 05:00 UTC so dependency bumps can be
  triaged during the standard maintenance window shared with the other security
  automations.
- **Routing** — Dependabot assigns the `dependencies` label and requests review
  from `@fermga` to match the maintainer responsibilities documented in
  `meta.json` and the contribution guide.

## Review checklist

1. **Confirm scope** — Ensure the diff only touches dependency manifests or the
   GitHub Actions workflow files declared above. If the PR includes unrelated
   changes, convert it to a draft and investigate before merging.
2. **Read the advisory** — Dependabot links advisories in the PR description.
   Review the impact on the structural operators affected by the dependency and
   confirm the proposed version restores or preserves coherence metrics.
3. **Run the quality gate** — Execute `./scripts/run_tests.sh` locally. The
   script exercises typing, linting, and the test suite, ensuring the update
   does not degrade `C(t)`, phase synchrony, or ΔNFR expectations.
4. **Inspect telemetry hooks** — For Python dependency bumps, check whether any
   logged metrics, cache interfaces, or serialization formats changed. Update
   downstream integration notes if the new version alters how telemetry is
   emitted.
5. **Validate workflow upgrades** — For GitHub Actions updates, review the
   upstream changelog. Confirm that permissions, caching keys, and Python
   versions remain consistent with the repository’s security posture.
6. **Document structural effects** — When approving, leave a short comment
   summarizing the expected influence on the relevant operators (e.g., improved
   `resonance` stability after a TLS library upgrade).

## Merge policy

- Merge only after the full GitHub Actions suite passes so CI logs capture the
  post-upgrade telemetry.
- Prefer squash merges so you can edit the final commit message to follow the
  `AGENT_COMMIT_TEMPLATE` before completing the merge.
- If a dependency cannot be updated immediately, open a follow-up issue that
  references the advisory IDs, the blocked operator(s), and the mitigation plan
  to track residual risk.

Adhering to this flow keeps automated dependency maintenance compatible with
the TNFR commitment to operational coherence and reproducible structural
interventions.
