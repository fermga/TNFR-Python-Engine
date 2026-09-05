# Repository scripts

This directory contains maintained entry points for documentation, reproducibility,
validation, and research workflows. Every listed command resolves to a file in the
current repository.

## Documentation and repository maintenance

| Script | Purpose |
| --- | --- |
| `check_documentation.py` | Check the agent mirror, version references, operator contracts, public examples, and documentation build inputs. |
| `verify_internal_references.py` | Validate repository-relative Markdown targets and GitHub-style heading fragments. |
| `prepare_docs.py` | Build the deterministic MkDocs source tree under `build/docs-source`. |
| `clean_repository.py` | Remove only known generated artifacts inside the repository. |

Run the complete documentation gate with:

```bash
make docs
```

This validates references and executable examples before running a strict MkDocs
build. Generated documentation sources and the rendered `site/` directory are not
canonical sources.

## Reproducibility and research

| Script | Purpose |
| --- | --- |
| `rebuild_failure_manifest.py` | Rebuild the failure manifest used by reproducibility investigations. |
| `replay/register_manifest.py` | Register replay metadata for a stored run. |
| `run_self_optimization.py` | Execute the manifest-driven self-optimization workflow. |
| `run_self_opt_validation.py` | Validate outputs produced by the self-optimization workflow. |
| `validate_conservation_law.py` | Run the structural conservation-law validation program. |
| `tnfr_is_prime.py` | Compatibility entry point for the TNFR primality tool. |

Use `--help` on scripts that expose command-line options. Reproducible runs must
record their seed, inputs, operator sequence, and generated manifest.

## Related commands

- `make validate` checks imports, documentation integrity, and the SDK test area.
- `make self-optimize` and `make self-optimize-validate` run the manifest workflow.
- `pip install -e ".[security]"` installs the tools required by `make security`.

See [ARCHITECTURE.md](../ARCHITECTURE.md), [TESTING.md](../TESTING.md), and
[SECURITY.md](../SECURITY.md) for the governing contracts.
