# Documentation Language Policy (English Only)

All TNFR Python Engine contributions must be authored in English. This policy is canonical and non‑negotiable.

## Scope

The requirement applies to:

- Markdown documentation (README, roadmap, theory, benchmarks, proposals)
- Source code comments and docstrings
- Commit messages
- Pull request titles and descriptions
- Issue titles and bodies
- Generated textual output intended for interpretation (labels, headings, summaries)

## Limited Exceptions

Non-English text is permitted only when:

- Quoting published external sources verbatim (must cite)
- Including raw experimental artifacts whose alteration would change semantics

In all exceptions, explanatory/contextual material MUST remain in English.

## Rationale

- Ensures consistent physics and grammar terminology
- Enables global collaboration without translation friction
- Avoids semantic drift between multilingual fragments
- Simplifies automated quality and lint checks

## Enforcement

- Reviewers will request changes for non-English normative additions
- Maintainers may block merges until language compliance is achieved
- Automated scanners may be introduced for common non-English tokens

## Examples

Compliant commit message:

> Add bifurcation landscape benchmark sweeping OZ intensity grid.

Non-compliant commit message:

> Añadir benchmark de bifurcación para intensidad OZ.

Mixed-language PR descriptions are considered non-compliant.

## Contributor Agreement

By opening a pull request you affirm adherence to this English-only policy. Request translation assistance in draft PRs if needed—do not merge mixed-language content.

---

**Reality is resonance. Its documentation must be universally readable.**
