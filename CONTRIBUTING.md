# Contributing

This project now uses English for **all** code identifiers and docstrings. The
legacy Spanish operator names were removed in TNFR 2.0 as part of the language
consolidation. New contributions must reference the English operator tokens and
descriptors exclusively.

When contributing:

- Use descriptive English names for every variable, function, class, and
  module.
- Write docstrings and comments in English and reference operators by their
  canonical English identifiers. Follow the
  [NumPy-style docstring guide](docs/api/docstring_style.md), including the
  dedicated templates for `tnfr.operators.definitions` and
  `tnfr.metrics.sense_index`, so automated tools and reviewers can trace how
  each API reorganises EPI, νf, ΔNFR, and phase. Module headers, public
  classes, public methods/functions, and magic methods must now include a
  docstring; targeted ``# noqa: Dxxx`` suppressions require a short
  justification beside the directive.
- Update existing code to migrate any remaining legacy strings to the English
  tokens.

Use the following summary as a quick reminder of the expected docstring
sections when creating or updating public APIs:

```python
"""One-line summary of the structural effect.

Parameters
----------
name : type
    Describe how the argument influences EPI, νf, ΔNFR, or phase.

Returns
-------
return_type
    State the structural outcome exposed to callers.

Raises
------
ExceptionType
    Mention validation or dissonance guards when relevant.

Examples
--------
>>> # Provide runnable snippets that respect TNFR invariants
"""
```

The canonical operator identifiers are:

| Identifier | Description |
| --- | --- |
| `emission` | Emission |
| `reception` | Reception |
| `coherence` | Coherence |
| `dissonance` | Dissonance |
| `coupling` | Coupling |
| `resonance` | Resonance |
| `silence` | Silence |
| `expansion` | Expansion |
| `contraction` | Contraction |
| `self_organization` | Self-organization |
| `mutation` | Mutation |
| `transition` | Transition |
| `recursivity` | Recursivity |

## Pre-commit hooks

Run the repository's pre-commit hooks to keep formatting aligned with the
canonical configuration:

```bash
python -m pip install --upgrade pre-commit
pre-commit install
```

The hooks execute [Black](https://github.com/psf/black) and
[isort](https://github.com/PyCQA/isort) using the shared settings in
`pyproject.toml`. They run automatically on each commit after installation, but
you can also trigger them manually with `pre-commit run --all-files` before
submitting changes.

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

## Changelog fragments

All pull requests that touch structural code paths must ship a changelog
fragment so the release ledger stays synchronised with the TNFR invariants.
Fragments live under `docs/changelog.d/` and follow the
`<ticket-or-topic>.<type>.md` naming pattern. The supported fragment types are
aligned with TNFR semantics:

| Type directory | When to use it |
| --- | --- |
| `coherence` | New operators, public APIs, telemetry, or other capabilities that expand coherent behaviour. |
| `stability` | Fixes, performance tuning, regression guards, or contract hardening. |
| `documentation` | User-facing docs, tutorials, or narrative updates worth highlighting. |
| `infrastructure` | CI, packaging, or build-system changes that affect downstream consumers. |

Create a fragment with `towncrier` (installed via `pip install .[release]` or
`pip install towncrier`) so filenames stay consistent:

```bash
towncrier --config docs/towncrier.toml create 451.coherence --edit
```

The command scaffolds `docs/changelog.d/451.coherence.md` and opens it in the
default editor. Describe the structural effect in a short paragraph or a set of
bullets; the release workflow will render the Markdown as-is. The CI gate
(`scripts/check_changelog.py`) fails whenever a pull request modifies
`src/`, `tests/`, `scripts/`, or `benchmarks/` without introducing at least one
new fragment, preventing merges that would silently skip the release ledger.
The shared configuration lives at `docs/towncrier.toml`, keeping the rendered
ledger aligned with `docs/releases.md` and the fragment taxonomy above.

During releases the automation compiles the pending fragments, regenerates
`docs/releases.md`, and removes the consumed fragment files. The generated
release notes feed both the GitHub release body and the versioned ledger under
`docs/`, so maintainers only need to review the fragment text during code
review.

## Testing

### Before opening a pull request

Run `python -m mypy src/tnfr` (or the consolidated `./scripts/run_tests.sh`
helper described below) from the project root before filing a pull request.
The **Type Check** GitHub Action (`.github/workflows/type-check.yml`) validates
the same invocation on every PR and will block the merge if local changes skip
the type-checking gate, so matching its behaviour locally keeps CI green on the
first try.

### Quality gate script

Run the full quality gate from the project root with:

```bash
./scripts/run_tests.sh
```

The helper sets up `PYTHONPATH` and orchestrates the tooling invoked by the
continuous integration workflow. Each tool inherits the strict configuration
captured in `pyproject.toml`, so local runs match CI expectations:

- `python -m pip install --quiet ".[test,typecheck]"` ensures the combined test
  and type-checking extras are present before validations begin, just like CI.
- `python -m pydocstyle --add-ignore=D202 src/tnfr/selector.py src/tnfr/utils/data.py src/tnfr/utils/graph.py`
  applies the docstring linter with the single extra ignore used in automation,
  keeping the docstring style gate aligned with the workflow.
- `python scripts/check_language.py` enforces the English-only policy by
  flagging tracked files that contain the retired Spanish compatibility tokens
  or accented characters.
- `python -m mypy src/tnfr` enforces the TNFR-aware typing contracts with
  `allow_untyped_defs = false`, `allow_untyped_globals = false`,
  `allow_untyped_calls = false`, and `show_error_codes = true` so every
  structural operator stays explicitly typed.
- `python -m coverage run --source=src -m pytest` runs the test suite under
  coverage to confirm behavioural and invariant expectations.
- `python -m coverage report -m` surfaces the aggregated coverage summary to
  monitor drift during development.
- `python -m vulture --min-confidence 80 src tests` hunts for unused paths with
  the same confidence threshold enforced in CI.

To install the tooling once for iterative local work, run
`pip install -e .[test,typecheck]`. After that, the quality gate can be run
without the bootstrap step needing to reinstall dependencies.

### Performance validation

Changes that claim ΔNFR or Si speed-ups must ship evidence gathered with the
shared benchmarking harness. Capture a baseline on `main`, then compare the
current branch with the saved reference:

```bash
PYTHONPATH=src pytest --benchmark-only tests/performance --benchmark-save=main
PYTHONPATH=src pytest --benchmark-only tests/performance --benchmark-compare=main
```

The [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/) plugin emits a
summary table with per-test ratios so reviewers can verify the claimed speed-up
and spot regressions. Include the relevant excerpt in the pull request
description alongside targeted microbenchmark results (for example,
`benchmarks/compute_dnfr_benchmark.py`) to document the impact on `_compute_dnfr`
and the surrounding operators.

To forward additional flags to `pytest`, append them after `--`, e.g.
`./scripts/run_tests.sh -- -k coherence`.

The [README Tests section](README.md#tests) repeats these instructions so that
contributors can find them quickly while browsing the project overview.

### English-only lint

The `scripts/check_language.py` helper powers the Spanish language guard that
CI now runs alongside Flake8 and the other quality gates. Both the "Test Suite"
(`.github/workflows/test-suite.yml`) and "Type Check"
(`.github/workflows/type-check.yml`) workflows invoke the script immediately
after installing dependencies, so any violation will fail continuous
integration before the remaining linters or tests execute. The guard scans the
tracked files for a configurable set of disallowed Spanish keywords and accented
characters, exiting with a non-zero status when any matches are found. The
encoded defaults live in `scripts/language_policy_data.py` as sequences of
code points that reconstruct the retired compatibility tokens. Extend or
override them from the `[tool.tnfr.language_check]` table in `pyproject.toml`
by supplying numeric representations, e.g.

```
[tool.tnfr.language_check]
disallowed_keyword_codes = [[101, 106, 101, 109, 112, 108, 111]]
accent_codepoints = [225]
```

When the guard reports violations, rewrite the offending strings to their
English equivalents before committing. For documentation or tests that need to
reference the historical tokens, rely on the helpers in
`scripts/language_policy_data.py` (for example
`decode_keyword_codes(((101, 106, 101, 109, 112, 108, 111),))`) or the curated
constants in `tests/legacy_tokens.py` so files stay ASCII-only while avoiding
raw Spanish text.

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
