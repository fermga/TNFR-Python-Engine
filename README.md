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

Then follow the [quickstart guide](docs/getting-started/quickstart.md) for Python and CLI
walkthroughs plus optional dependency caching helpers.

## Documentation map

- [Documentation index](docs/index.md) — navigation hub for API chapters and examples.
- [API overview](docs/api/overview.md) — package map, invariants, and structural data flow.
- [Structural operators](docs/api/operators.md) — canonical grammar, key concepts, and typical
  workflows.
- [Telemetry & utilities](docs/api/telemetry.md) — coherence metrics, trace capture, locking,
  and helper facades.
- [Examples](docs/examples/README.md) — runnable scenarios, CLI artefacts, and token legend.

## Documentation build workflow

Netlify publishes the documentation with [MkDocs](https://www.mkdocs.org/) so the generated
site preserves the canonical TNFR structure. The same steps can be executed locally:

1. Create and activate a virtual environment (e.g. `python -m venv .venv && source .venv/bin/activate`).
2. Install the documentation toolchain: `python -m pip install -r docs/requirements.txt`.
3. Preview changes live with `mkdocs serve` or reproduce the Netlify pipeline with
   `mkdocs build`, which writes the static site to the `site/` directory.

The Netlify build (`netlify.toml`) runs `python -m pip install -r docs/requirements.txt && mkdocs build`
and publishes the resulting `site/` directory, ensuring the hosted documentation matches local builds.

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

## Migration notes

- **Si dispersion keys:** Replace any remaining ``dSi_ddisp_fase`` entries in graph payloads
  or configuration files with the English ``dSi_dphase_disp`` key before upgrading. The
  runtime now raises :class:`ValueError` listing any unexpected sensitivity keys, and
  :func:`tnfr.metrics.sense_index.compute_Si_node` rejects unknown keyword arguments.
- Refer to the [release notes](docs/releases.md#1100-si-dispersion-legacy-keys-removed) for
  a migration snippet that rewrites stored graphs in place prior to running the new version.

## Licensing

Released under the [MIT License](LICENSE.md). Cite the TNFR paradigm when publishing research
or derived artefacts based on this engine.
