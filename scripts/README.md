# Repository scripts

This directory contains maintained entry points for documentation, reproducibility,
validation, and research workflows. Every listed command resolves to a file in the
current repository.

## Documentation and repository maintenance

| Script | Purpose |
| --- | --- |
| `check_documentation.py` | Check the agent mirror, version references, operator contracts and grammar roles, public examples, technical/theory catalogs and navigation, glossary cards/index, and documentation build inputs. |
| `check_distribution.py` | Inspect completed wheel/sdist payloads against current package source and explicit publication size budgets, without extraction. |
| `generate_physics_stub.py` | Check the typed physics facade against its sole runtime export registry without importing the engine; `--write` regenerates that facade after a registry change. |
| `check_glossary.py` | Validate the canonical glossary's concept cards and generated index; no separate CLI or engine law registry. |
| `research_book.py` | Read the modular research catalog and canonical chapters, with legacy monolithic compatibility; list, extract or atomically replace one chapter. |
| `verify_internal_references.py` | Validate repository-relative Markdown targets and GitHub-style heading fragments. |
| `prepare_docs.py` | Build the deterministic MkDocs source tree under `build/docs-source`. |
| `clean_repository.py` | Preflight declared generated directories and package metadata before deletion; reject redirected paths and retain frozen source under `build/`. |

Run `python scripts/check_distribution.py dist` after building distributions.
The [package checks](../TESTING.md#package-content-and-size-checks) define the
payload policy and configurable review budgets. `--json` prints a machine-readable
report to stdout; keep saved audit reports outside the distribution directory.

Run the complete documentation gate with:

```bash
make docs
```

The reference check covers maintained Markdown and skips ignored `artifacts/`,
`output/`, `outputs/` and `results/` captures, whose paths and assertions retain
their original context. Local
research evidence is identified explicitly instead of linked as published content.

The reference check also resolves this repository's GitHub `blob/main` and
`tree/main` links against the checkout. It resolves reference-style links and
ignores Markdown examples inside fenced code blocks.
External websites and frozen run captures are outside this local check.

The operator table in `docs/API_CONTRACTS.md` and the logical grammar chapter's
role table are generated from their shared registries. Technical navigation
reads `docs/README.md`; theory navigation and its repository index are generated
from `theory/research_catalog.json`. The glossary index reads concept cards in
`theory/GLOSSARY.md`. Edit canonical chapters directly, or use the optional
`research_book.py` extract/replace workflow; do not edit generated navigation.
Keep `theory/RESEARCH_PROGRAM.md` within its 96-KiB preview budget. After an intentional contract, chapter
metadata or concept-card change, run
`python scripts/check_documentation.py --write-generated` and review the diff.
The ordinary gate verifies these generated views and complete catalog coverage;
it does not update documentation silently or maintain a second owner registry.

The [glossary template](../theory/GLOSSARY.md#card-template-and-admission-checks)
records definition, domain, premises, dependencies, maintained owners, evidence,
implementation, tests and limits. Its validator rejects malformed cards,
circular prerequisites and unsupported classification combinations. It checks
the existence of cited evidence, not its scientific truth, and does not run the
linked research tests or promote a concept automatically.

This validates references and executable examples before running a strict MkDocs
build. Generated documentation sources and the rendered `site/` directory are not
canonical sources.
Staging preserves fenced and inline code examples verbatim while rewriting
rendered local links. Staging and cleanup share the generated-directory guard;
neither follows a redirected output directory into source or another tree.
Cleanup removes only `build/docs-source` and `build/lib` within `build/`.
Other children can contain frozen evaluators or local preparations and are
retained; the parent directory is not a disposable output.

## Reproducibility and research

| Script | Purpose |
| --- | --- |
| `rebuild_failure_manifest.py` | Recover the latest failure record per integer input from an explicitly selected artifact directory; validate inputs before replacing the index. |
| `replay/register_manifest.py` | Register replay metadata for a stored run. |
| `run_localized_mobile_response.py` | Inspect the declared mobile-pair response lifecycle; freezing and evaluation belong to its [single reserved protocol](../theory/nodal/LOCALIZED_MOBILE_RESPONSE.md), not routine verification. |
| `restore_frozen_source.py` | Inspect a supported freeze against its complete Git base; optionally restore verified supplements in a new detached worktree, without executing archived code. |
| `run_self_optimization.py` | Execute the manifest-driven self-optimization workflow. |
| `run_self_opt_validation.py` | Run current-code regression suites selected by recommendation operation type; it does not apply or evaluate the recommendations. |
| `tnfr_is_prime.py` | Compatibility entry point for the TNFR primality tool. |

Use `--help` on scripts that expose command-line options. Reproducible runs retain source/configuration, inputs, seed, node/target order,
precision/backend, law, clock, events and residuals as applicable. The
[research evidence guide](../docs/guides/RESEARCH_EVIDENCE.md) owns artifact
admission and the distinction between inspection, preparation and execution.

Frozen-source restoration follows the [research evidence workflow](../docs/guides/RESEARCH_EVIDENCE.md).
The default is read-only inspection; `--destination` must name a new location.
Source restoration does not verify installed dependencies or execute an evaluation.

Failure-manifest recovery requires `--artifacts-dir` and `--manifest`;
`--expected-count` optionally checks the number of distinct inputs. It does not
assume a historical run date, a GPU campaign or consecutive input numbers.
The resulting index compacts repeated inputs to their latest record; source
artifacts retain the complete captured history.

Manifest recovery, replay registration and recommendation discovery use the
shared JSON decoder: duplicate keys, nonfinite numbers and nonzero fractional
literals lost to represented zero reject, including in nested metadata.
These commands require the installed checkout described in [Testing](../TESTING.md).
Recovery validates every matching source artifact before writing its index.
Recommendation discovery skips well-formed JSON that is not a recommendation,
but malformed JSON stops the run before any mapped test suite executes.
Unknown valid operation names remain pending; an invalid present name cannot
fall through to another field.

Replay registration admits the selected inline/archive policy, partition list
and present telemetry before calculating its digest and ranges. Missing or null
telemetry is omitted from a range; it is not a measured zero. Range means use
the shared finite reduction. The file-list digest covers the ordered names,
not the contents of the named partition files or the truth of summary metadata.

Generated manifest, registration and self-optimization reports use shared
atomic writers. Invalid encoding or values cannot truncate an existing report.
Recovery/registration outputs and self-optimization summaries cannot replace
their input artifacts; a regression report cannot replace a discovered
recommendation payload. These path checks precede optimization or suite execution.

The self-optimization runner admits consumed partition metadata and telemetry
before dispatching that partition. Missing or null observations stay unavailable;
present malformed sequences or observations produce an explicit failed partition.
An explicitly supplied nodal-state sequence takes precedence over the metadata
sequence, including an empty list or null. Unused archive metadata is retained
without acquiring numerical meaning. Cross-partition means omit unavailable
observations; a nodal-field mean requires every supplied sample. Shared finite
reductions preserve representable means and cancellation, while unrepresentable
differences fail explicitly. CLI progress goes to stderr; stdout contains the
strict JSON summary.

Partitions supply an explicit serialized graph. A modulus and member list
alone no longer reconstruct a graph through the archived factorization lab.
The SDK wrapper uses these checkout scripts without importing that application;
an installation lacking the scripts receives an explicit checkout requirement.
Its assessment is a dry run: it does not dispatch the proposed operators and
its `promotable` mapping is empty.

Self-optimization snapshots represent an unavailable coherence-length estimate
as `telemetry.canonical.xi_c: null`, with `xi_c_available: false` and
`xi_c_provenance` from the shared estimator. Finite estimates retain their value
and fit/fallback provenance. Invalid primitive state or other invalid diagnostic
output yields `telemetry: null` with `telemetry_error`; it is not converted into
an unavailable coherence-length estimate.

Dry-run payloads use strict JSON and hash the exact stored UTF-8 bytes, including
their line endings. The payload and its `.sha256` sidecar are each written
atomically, but are not a joint filesystem transaction. Serialization failure
preserves both existing files; the digest detects changed bytes, not source
authenticity or scientific validity.

Structural-balance checks use the shared
[conservation diagnostics](../src/tnfr/physics/conservation.py) and
[contract tests](../tests/core_physics/test_conservation_laws.py). A measured
nonzero residual must remain a nonzero residual; a private phase/pressure
smoothing script cannot verify the nodal law or a general conservation theorem.
The [scope record](../theory/research/archive/README.md#foundation-reassessment-2026-09-20)
owns the retired validation entry point and its replacement.

## Related commands

- `make validate` checks imports, documentation integrity, and the SDK test area.
- `make self-optimize` and `make self-optimize-validate` run the manifest workflow.
- `pip install -e ".[security]"` installs the tools required by `make security`.

The regression report uses `validated`/`regressed` status labels
for suite exit codes and explicitly reports its `validation_scope`. A passing
mapped suite is not evidence that a recommendation was applied or improved the
network. The default operation mapping is empty; operations remain pending
until an appropriate regression suite is explicitly configured. The archived
lab's arithmetic tests are not an automatic validation target.

See [ARCHITECTURE.md](../ARCHITECTURE.md), [TESTING.md](../TESTING.md), and
[SECURITY.md](../SECURITY.md) for the governing contracts.
