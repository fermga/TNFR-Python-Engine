# Contributing

This project uses English exclusively for code identifiers and docstrings. When contributing:

- Use descriptive English names for all variables, functions, classes and modules.
- Write docstrings and comments in English.
- Update existing code to maintain this convention when modifying files.

## Testing

Run the full quality gate from the project root with:

```bash
./scripts/run_tests.sh
```

The helper sets up `PYTHONPATH` and orchestrates the tooling invoked by the
continuous integration workflow:

- `pydocstyle` for targeted docstring style checks.
- `coverage run --source=src -m pytest` to execute the test suite under
  coverage.
- `coverage report -m` to display the aggregate coverage summary.
- `vulture --min-confidence 80 src tests` to detect unused code paths.

To forward additional flags to `pytest`, append them after `--`, e.g.
`./scripts/run_tests.sh -- -k coherence`.

The [README Tests section](README.md#tests) repeats these instructions so that
contributors can find them quickly while browsing the project overview.

Make sure to honor the patterns in `.gitignore` so that dependency and build
artifacts (e.g., `node_modules/` or `dist/`) are not committed.

## Architectural conventions

- **Add new structural operators** under `src/tnfr/operators/` and register
  them with `tnfr.operators.registry.register_operator` so canonical discovery
  and validation continue to work without manual wiring.【F:src/tnfr/operators/registry.py†L12-L49】
- **Keep operator closure intact** by updating grammar/syntax rules alongside
  new operator sequences. Start with `tnfr.validation.syntax.validate_sequence`
  and `tnfr.validation.grammar.enforce_canonical_grammar`, then add any THOL
  handling you need in `tnfr.flatten`.【F:src/tnfr/validation/syntax.py†L1-L86】【F:src/tnfr/validation/grammar.py†L1-L90】【F:src/tnfr/flatten.py†L1-L120】
- **Share caches, locks, and telemetry** through the provided helpers instead
  of ad-hoc globals. Reuse `tnfr.helpers` exports, `tnfr.cache.CacheManager`,
  and `tnfr.locking.get_lock` when extending RNG, ΔNFR, or metric pipelines so
  that instrumentation stays consistent.【F:src/tnfr/helpers/__init__.py†L1-L74】【F:src/tnfr/cache.py†L1-L120】【F:src/tnfr/locking.py†L1-L36】
- **Reference the [Architecture Overview](README.md#architecture-overview)**
  for diagrams showing the structural data flow, invariant enforcement map,
  and telemetry hooks before touching the orchestration layers.
