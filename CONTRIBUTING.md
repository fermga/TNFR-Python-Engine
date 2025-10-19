# Contributing

This project uses English for code identifiers and docstrings **except** for the
canonical Spanish operator names exported by `tnfr.operators`. Those operator
identifiers (for example `Emision`, `Resonancia`, `Autoorganizacion`) are part
of the TNFR grammar and **must stay in Spanish** even when the surrounding
documentation or prose is written in English. When contributing:

- Use descriptive English names for all variables, functions, classes and modules
  aside from the canonical operator identifiers listed below.
- Write docstrings and comments in English while preserving the canonical
  operator names verbatim.
- Update existing code to maintain this convention when modifying files.

When documenting operators in English prose, pair the canonical identifier with
its English descriptor so readers can follow along without altering the API.
For example, write “apply the Emision operator (Emission)” or “the
Autoorganizacion (`Self-organization`) step”. The canonical mapping is:

| Canonical identifier | English descriptor |
| --- | --- |
| `Emision` | Emission |
| `Recepcion` | Reception |
| `Coherencia` | Coherence |
| `Disonancia` | Dissonance |
| `Acoplamiento` | Coupling |
| `Resonancia` | Resonance |
| `Silencio` | Silence |
| `Expansion` | Expansion |
| `Contraccion` | Contraction |
| `Autoorganizacion` | Self-organization |
| `Mutacion` | Mutation |
| `Transicion` | Transition |
| `Recursividad` | Recursivity |

## Commit message format

Every commit **must** follow the `AGENT_COMMIT_TEMPLATE` documented for this
repository. Copy the header exactly and fill in each field before writing the
rest of the commit body so human maintainers can audit how the change aligns
with the TNFR invariants. Refer to [AGENTS.md](AGENTS.md) for the authoritative
source of this template and any future adjustments.

```text
Intent: (which coherence is improved)
Operators involved: [Emission|Reception|...]
Affected invariants: [#1, #4, ...]
Key changes: (bullet list)
Expected risks/dissonances: (and how they’re contained)
Metrics: (C(t), Si, νf, phase) before/after expectations
Equivalence map: (if you renamed APIs)
```

Fill out each field with concise, review-ready information:

- **Intent** — Summarize the structural coherence or capability the commit
  reorganizes.
- **Operators involved** — List every structural operator touched so reviewers
  can verify closure and semantics.
- **Affected invariants** — Reference the numbered canonical invariants that
  change or require revalidation.
- **Key changes** — Provide the high-level bullet list that mirrors the diff’s
  structural impact.
- **Expected risks/dissonances** — Note possible regressions or dissonant
  dynamics and describe the containment strategy.
- **Metrics** — Describe anticipated shifts in C(t), Si, νf, or phase to align
  expectations with telemetry.
- **Equivalence map** — Document any renamed APIs or reorganized entry points
  so downstream integrations can adjust.

## Testing

Run the full quality gate from the project root with:

```bash
./scripts/run_tests.sh
```

The helper sets up `PYTHONPATH` and orchestrates the tooling invoked by the
continuous integration workflow:

- `python -m pip install ".[typecheck]"` ensures local type-checking
  dependencies such as `mypy` and `networkx-stubs` are available.
- `pydocstyle` for targeted docstring style checks.
- `python -m mypy src/tnfr` to enforce TNFR-aware typing contracts.
- `coverage run --source=src -m pytest` to execute the test suite under
  coverage.
- `coverage report -m` to display the aggregate coverage summary.
- `vulture --min-confidence 80 src tests` to detect unused code paths.

To install the tooling once for iterative local work, run
`pip install -e .[test,typecheck]`. After that, the quality gate can be run
without the bootstrap step needing to reinstall dependencies.

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
  for quick diagrams, then deep-dive in the
  [TNFR Architecture Guide](ARCHITECTURE.md) to understand orchestration,
  telemetry paths, and invariant enforcement before touching the core layers.

### Internal architecture guide

Use the [TNFR Architecture Guide](ARCHITECTURE.md#layered-responsibilities)
as the canonical reference for deep dives. The summary below highlights the
core layers, where they live, and why edits must preserve the invariant and
telemetry contracts already documented.

- **Structural grammar** — `tnfr.structural`,
  `tnfr.validation.syntax`, `tnfr.validation.grammar`, `tnfr.flatten`.
  These modules instantiate nodes, validate operator sequences, and expand
  THOL blocks so every execution path honours the canonical grammar before
  mutating EPI.【F:src/tnfr/structural.py†L39-L109】【F:src/tnfr/validation/syntax.py†L27-L121】【F:src/tnfr/validation/grammar.py†L1-L90】【F:src/tnfr/flatten.py†L1-L120】
- **Operator registry** — `tnfr.operators.definitions`,
  `tnfr.operators.registry`. They bind glyphs to implementations and enforce
  closure so the structural layer never executes unknown tokens.【F:src/tnfr/operators/definitions.py†L45-L180】【F:src/tnfr/operators/registry.py†L13-L50】
- **Dynamics and adaptation** — `tnfr.dynamics.__init__`,
  `tnfr.dynamics.dnfr`, `tnfr.dynamics.integrators`. These components mix
  ΔNFR, integrate the nodal equation, and coordinate phase/νf adjustments
  while keeping stochastic hooks reproducible.【F:src/tnfr/dynamics/__init__.py†L59-L199】【F:src/tnfr/dynamics/dnfr.py†L1958-L2020】【F:src/tnfr/dynamics/integrators.py†L420-L483】
- **Telemetry and traces** — `tnfr.metrics.common`,
  `tnfr.metrics.sense_index`, `tnfr.trace`, `tnfr.metrics.trig_cache`. They
  compute C(t), ΔNFR summaries, Si, and trace history so coherence metrics
  remain auditable.【F:src/tnfr/metrics/common.py†L32-L149】【F:src/tnfr/metrics/sense_index.py†L1-L200】【F:src/tnfr/trace.py†L169-L319】【F:src/tnfr/metrics/trig_cache.py†L1-L120】
- **Shared services** — `tnfr.helpers`, `tnfr.cache`, `tnfr.locking`,
  `tnfr.rng`. These facades provide deterministic caches, locks, and RNG
  orchestration for every layer that needs shared state.【F:src/tnfr/helpers/__init__.py†L1-L74】【F:src/tnfr/cache.py†L1-L120】【F:src/tnfr/locking.py†L1-L36】【F:src/tnfr/rng.py†L1-L88】

**Checklist before merging changes**

- Structural grammar — Re-run syntax/grammar validation, confirm THOL
  expansion leaves trace hooks intact, and verify invariants #1, #4, #5, and
  #7 remain satisfied. Consult
  [Layered responsibilities](ARCHITECTURE.md#layered-responsibilities) and
  [Structural loop orchestration](ARCHITECTURE.md#structural-loop-orchestration)
  for the expected data flow.
- Operator registry — Ensure new glyph bindings preserve invariants #3, #4,
  and #10, regenerate registry discovery tests, and cross-check the
  [Operator registration mechanics](ARCHITECTURE.md#operator-registration-mechanics)
  guidance for naming and closure rules.
- Dynamics and adaptation — Validate νf units (Hz_str), ΔNFR semantics, and
  stochastic clamp hooks against invariants #1, #2, #3, #5, and #8. Inspect
  the [ΔNFR and telemetry data paths](ARCHITECTURE.md#%CE%94nfr-and-telemetry-data-paths)
  section to confirm new flows emit required telemetry.
- Telemetry and traces — Confirm C(t), Si, ΔNFR, and phase metrics still
  reach the trace buffers, and uphold invariants #8 and #9. Follow the
  telemetry expectations in the
  [ΔNFR and telemetry data paths](ARCHITECTURE.md#%CE%94nfr-and-telemetry-data-paths)
  and [Telemetry exports](ARCHITECTURE.md#layered-responsibilities) notes.
- Shared services — Keep caches deterministic, lock scopes documented, and
  RNG seeds reproducible to satisfy invariants #8 and #9. Review the
  [Layered responsibilities](ARCHITECTURE.md#layered-responsibilities)
  summary before touching shared state helpers.
