# TNFR Python Engine

Canonical implementation of the Resonant Fractal Nature Theory (TNFR) for modelling structural
coherence. The engine seeds resonant nodes, applies structural operators, coordinates
ΔNFR/phase dynamics, and measures coherence metrics (C(t), Si, νf) without breaking the nodal
equation $\partial EPI/\partial t = \nu_f \cdot \Delta NFR(t)$.

## Snapshot

- **Operate:** build nodes with `tnfr.create_nfr`, execute trajectories via
  `tnfr.structural.run_sequence`, and evolve dynamics with `tnfr.dynamics.run`.
- **Observe:** register metrics/trace callbacks to capture ΔNFR, C(t), Si, and structural
  histories
  for every run.
- **Extend:** rely on the canonical operator grammar and invariants before introducing new
  utilities or telemetry.

## Quickstart

Install from PyPI (Python ≥ 3.9):

```bash
pip install tnfr
```

Then follow the [quickstart guide](docs/source/getting-started/quickstart.md) for Python and CLI
walkthroughs plus optional dependency caching helpers.

## CLI profiling helpers

Generate Sense Index and ΔNFR profiling artefacts directly from the CLI with the
``profile-pipeline`` subcommand. The helper reproduces the performance benchmark that
captures vectorised and fallback execution traces for the full pipeline:

```bash
tnfr profile-pipeline \
  --nodes 120 --edge-probability 0.28 --loops 3 \
  --si-chunk-sizes auto 48 --dnfr-chunk-sizes auto \
  --si-workers auto --dnfr-workers auto \
  --output-dir profiles/pipeline
```

The command writes ``.pstats`` and JSON summaries for each configuration/mode pair, making
it easy to inspect hot paths with :mod:`pstats`, Snakeviz, or downstream tooling.

## Documentation map

- [Documentation index](docs/source/home.md) — navigation hub for API chapters and examples.
- [API overview](docs/source/api/overview.md) — package map, invariants, and structural data flow.
- [Structural operators](docs/source/api/operators.md) — canonical grammar, key concepts, and typical
  workflows.
- [Telemetry & utilities](docs/source/api/telemetry.md) — coherence metrics, trace capture, locking,
  and helper facades.
- [Examples](docs/source/examples/README.md) — runnable scenarios, CLI artefacts, and token legend.

## Documentation build workflow

Netlify now renders the documentation with [Sphinx](https://www.sphinx-doc.org/) so MyST Markdown,
doctests, and notebooks share a single pipeline. Reproduce the hosted site locally as follows:

1. Create and activate a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).
2. Install the documentation toolchain and project extras:
   `python -m pip install -r docs/requirements.txt && python -m pip install -e .[docs]`.
3. Execute the doctest suite with `sphinx-build -b doctest docs/source docs/_build/doctest` to ensure
   structural snippets remain coherent.
4. Generate the HTML site with `make docs`, which wraps `sphinx-build -b html docs/source docs/_build/html`.

The Netlify build (`netlify.toml`) runs `python -m pip install -r docs/requirements.txt && make docs`
and publishes the resulting `docs/_build/html` directory, keeping the hosted documentation aligned with
local verification runs.

## Local development

Use the helper scripts to keep formatting aligned with the canonical configuration and to reproduce
the quality gate locally:

```bash
./scripts/format.sh           # Apply Black and isort across src/, tests/, scripts/, and benchmarks/
./scripts/format.sh --check   # Validate formatting without modifying files
./scripts/run_tests.sh        # Execute the full QA battery (type checks, tests, coverage, linting)
```

The formatting helper automatically prefers `poetry run` when a Poetry environment is available and
falls back to `python -m` invocations so local runs mirror the tooling invoked in continuous
integration.

## Additional resources

- [ARCHITECTURE.md](ARCHITECTURE.md) — orchestration layers and invariant enforcement.
- [CONTRIBUTING.md](CONTRIBUTING.md) — QA battery (`scripts/run_tests.sh`) and review
  expectations.
- [TNFR.pdf](TNFR.pdf) — theoretical background, structural operators, and paradigm glossary.
- [PRE_EXISTING_FAILURES.md](PRE_EXISTING_FAILURES.md) — documented pre-existing test failures
  requiring separate focused PRs.

## Migration notes

- **Si dispersion keys:** Ensure graph payloads and configuration files use the canonical
  ``dSi_dphase_disp`` attribute for Si dispersion sensitivity before upgrading. The runtime now
  raises :class:`ValueError` listing any unexpected sensitivity keys, and
  :func:`tnfr.metrics.sense_index.compute_Si_node` rejects unknown keyword arguments.
- Refer to the [release notes](docs/source/releases.md#1100-si-dispersion-legacy-keys-removed) for
  a migration snippet that rewrites stored graphs in place prior to running the new version.

## Licensing

Released under the [MIT License](LICENSE.md). Cite the TNFR paradigm when publishing research
or derived artefacts based on this engine.
