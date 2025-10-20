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

## Additional resources

- [ARCHITECTURE.md](ARCHITECTURE.md) — orchestration layers and invariant enforcement.
- [CONTRIBUTING.md](CONTRIBUTING.md) — QA battery (`scripts/run_tests.sh`) and review
  expectations.
- [TNFR.pdf](TNFR.pdf) — theoretical background, structural operators, and paradigm glossary.

## Licensing

Released under the [MIT License](LICENSE.md). Cite the TNFR paradigm when publishing research
or derived artefacts based on this engine.
