# Repository scripts

This directory contains maintained entry points for documentation, reproducibility,
validation, and research workflows. Every listed command resolves to a file in the
current repository.

## Documentation and repository maintenance

| Script | Purpose |
| --- | --- |
| `check_documentation.py` | Check the agent mirror, version references, operator contracts, public examples, theory-catalog coverage and generated navigation, and documentation build inputs. |
| `verify_internal_references.py` | Validate repository-relative Markdown targets and GitHub-style heading fragments. |
| `prepare_docs.py` | Build the deterministic MkDocs source tree under `build/docs-source`. |
| `clean_repository.py` | Preflight declared generated directories, including package metadata under `src/`, before deletion; reject redirected paths. |

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

The operator table in `docs/API_CONTRACTS.md` is generated from the registry;
the theory section of `mkdocs.yml` is generated from the primary catalog's
topic headings and owner rows in `theory/README.md`. After an intentional
contract or catalog change, run
`python scripts/check_documentation.py --write-generated` and review the diff.
The ordinary gate verifies both generated views and complete catalog coverage;
it does not update documentation silently or maintain a second owner registry.

This validates references and executable examples before running a strict MkDocs
build. Generated documentation sources and the rendered `site/` directory are not
canonical sources.
Staging preserves fenced and inline code examples verbatim while rewriting
rendered local links. Staging and cleanup share the generated-directory guard;
neither follows a redirected output directory into source or another tree.

## Reproducibility and research

| Script | Purpose |
| --- | --- |
| `rebuild_failure_manifest.py` | Rebuild the failure manifest used by reproducibility investigations. |
| `replay/register_manifest.py` | Register replay metadata for a stored run. |
| `run_self_optimization.py` | Execute the manifest-driven self-optimization workflow. |
| `run_self_opt_validation.py` | Run current-code regression suites selected by recommendation operation type; it does not apply or evaluate the recommendations. |
| `tnfr_is_prime.py` | Compatibility entry point for the TNFR primality tool. |

Use `--help` on scripts that expose command-line options. Reproducible runs must
record their seed, inputs, operator sequence, and generated manifest.

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

The regression report retains historical `validated`/`regressed` status labels
for suite exit codes and explicitly reports its `validation_scope`. A passing
mapped suite is not evidence that a recommendation was applied or improved the
network; unknown operations remain pending.

See [ARCHITECTURE.md](../ARCHITECTURE.md), [TESTING.md](../TESTING.md), and
[SECURITY.md](../SECURITY.md) for the governing contracts.

## Retired profiler wrapper

The old `run_reproducible_benchmarks.py` registry referenced four absent profiler
programs and had no runnable workload. It has been removed. The remaining
`tnfr profile-si` and `tnfr profile-pipeline` compatibility commands also depend
on absent benchmark helpers and report unavailability in this checkout; they
are not maintained profiling instructions. The [benchmark guide](../benchmarks/README.md)
lists the current instruments. Their individual provenance requirements remain
necessary; a seed and checksum alone do not prove reproducibility.
