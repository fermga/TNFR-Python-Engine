# TNFR Testing Guide

This is the authoritative guide to the test suite. Test expectations follow the
[nodal equation and six invariants](AGENTS.md#8-canonical-invariants),
[operator contracts](src/tnfr/operators/operator_contracts.py), and
[unified grammar](theory/UNIFIED_GRAMMAR_RULES.md). A passing test establishes
only the behavior and parameter range asserted by that test.

## Run the repository tests

Run commands from the repository root with the Python interpreter for your
active environment. Install the project and the same dependency groups used by
the main CI test job:

```sh
python -m pip install -e ".[test,numpy,yaml,orjson]"
python -m pytest
```

[pyproject.toml](pyproject.toml) sets `pythonpath = ["src"]`,
`testpaths = ["tests"]`, and `addopts = "-m 'not slow'"`. Thus pytest imports the
working source tree and excludes tests marked `slow` by default. It does not
configure `--benchmark-skip`, `--strict-markers`, or `--tb=short`.

Useful bounded runs:

```sh
python -m pytest tests/operators -q
python -m pytest tests/core_physics tests/physics -q
python -m pytest tests/mathematics/test_backends.py -q
python -m pytest tests/sdk -q
python -m pytest tests/operators/test_u3_hard_invariant.py -v
python -m pytest --collect-only -q
```

To remove the default slow exclusion, use `python -m pytest -o addopts=""`.
To select only tests actually marked slow, use `python -m pytest -m slow`.
A marker can be registered without any currently collected tests using it;
inspect collection before treating a marker-selected run as coverage.

For standalone scripts outside pytest, use an editable installation or set
`PYTHONPATH` to `src`; pytest's `pythonpath` setting does not affect ordinary
`python` invocations. Otherwise an installed release can be imported instead
of the working tree.

## Organization

| Location | Current scope |
|---|---|
| [core_physics/](tests/core_physics/) | Nodal equation, structural triad, pressure channels, conservation, backend agreement |
| [operators/](tests/operators/) | Canonical operators, contracts, grammar, U3 phase gate, selection and execution |
| [physics/](tests/physics/) | Structural fields, diffusion, symmetry, conservation, directed dynamics, cache correctness |
| [mathematics/](tests/mathematics/) | Mathematical backends, spaces, arithmetic networks, pulse, multiscale constructions |
| [sdk/](tests/sdk/) | Public network interface |
| [engines/](tests/engines/) | Self-optimization and pattern-discovery manifests |
| [parallel/](tests/parallel/) | Fractal partition manifests |
| [research/](tests/research/) | Research infrastructure |
| [scripts/](tests/scripts/) | Self-optimization command-line scripts |
| [Top-level test files](tests/) | Phase-gate interfaces, external data interfaces, replay, distributed FFT, factorization and mechanics |
| [data/](tests/data/) | Fixture data and manifests |

[tests/conftest.py](tests/conftest.py) defines shared fixtures, backend selection,
and global-state cleanup. [tests/utils.py](tests/utils.py) supplies additional
helpers. The current tree has no separate `unit`, `property`, `integration`,
`performance`, `stress`, or `grammar_operators` test directories. Research
benchmark scripts live separately in [benchmarks/](benchmarks/README.md);
use their documented entry points rather than assuming pytest collection.

## Structural regression evidence

Start with the checks relevant to the changed physical contract. These are
existing entry points, not claims of exhaustive invariant coverage:

| Behavior | Test entry points |
|---|---|
| Nodal equation and pressure computation | [test_nodal_equation.py](tests/core_physics/test_nodal_equation.py), [test_delta_nfr_computation_paths.py](tests/core_physics/test_delta_nfr_computation_paths.py), [test_dnfr_backend_consistency.py](tests/core_physics/test_dnfr_backend_consistency.py) |
| Operator channel and scale contracts | [test_operator_contracts.py](tests/operators/test_operator_contracts.py) |
| U3 rejection before mutation and wrapped phase distance | [test_u3_hard_invariant.py](tests/operators/test_u3_hard_invariant.py) |
| U1-U4 context, accepted history and per-node fallback | [test_grammar_dynamics.py](tests/operators/test_grammar_dynamics.py) |
| Grammar classification consistency | [test_grammar_canon.py](tests/operators/test_grammar_canon.py), [test_grammar_canonical_consistency.py](tests/operators/test_grammar_canonical_consistency.py) |
| Silence EPI preservation and coupling phase synchronization | [test_canonical_operators_modern.py](tests/operators/test_canonical_operators_modern.py) |
| Tetrad bounds, field readout and cache invalidation | [test_tetrad_bounds.py](tests/physics/test_tetrad_bounds.py), [test_field_readout_consistency.py](tests/physics/test_field_readout_consistency.py), [test_field_cache_invalidation.py](tests/physics/test_field_cache_invalidation.py) |
| Diffusion modes and conservation | [test_structural_diffusion.py](tests/physics/test_structural_diffusion.py), [test_dissipative_conservation.py](tests/physics/test_dissipative_conservation.py) |
| Multiscale arithmetic transport and REMESH audit | [test_crt_multiscale.py](tests/mathematics/test_crt_multiscale.py), [test_remesh_audit.py](tests/mathematics/test_remesh_audit.py) |

For new or changed dynamics, assert the actual contract: IL must not reduce
`C(t)` outside a documented dissonance test; OZ needs a handler; RA must respect
phase compatibility and preserve identity; SHA must preserve EPI over the
specified evolution interval; ZHIR must obey its threshold and context; nested
EPIs must retain identity. Check the same seeded run twice when changing
stochastic execution. Execution without an exception alone does not establish
these properties, and changing EPI alone does not verify the nodal equation.

Specify graph topology, seed, initial triad, operator sequence, time step,
tolerances and measured quantities. Keep structural frequency in `Hz_str` and
report `C(t)`, `Si`, phase, structural frequency and the tetrad when relevant.
Declare whether a sequence is a full grammar word or a fragment; full words
require initiation, closure and transformer context. Set up initial fixtures
explicitly, then exercise state changes through canonical operators.

## Backends and optional dependencies

[tests/conftest.py](tests/conftest.py) accepts `--math-backend` and
`TNFR_TEST_MATH_BACKEND`. The command-line option takes precedence over that
test-specific environment variable; the selected value sets
`TNFR_MATH_BACKEND` and clears the backend cache before test collection.

```sh
python -m pytest tests/mathematics/test_backends.py --math-backend=numpy -q
python -m pytest tests/mathematics/test_backends.py --math-backend=torch -q
```

Install optional backends before interpreting their results. The cross-backend
tests explicitly request NumPy, JAX and PyTorch and skip cases where the
requested adapter is unavailable. Setting the session backend does not replace
explicit per-test backend arguments. NumPy is required by the shared conftest;
a missing NumPy installation does not constitute a successful NumPy-free run.
Use `-rs` to review skip reasons and report which adapters were exercised.

The registered project markers are `slow`, `benchmarks`, `stress`, `val`,
`canonical`, `nodal_equation`, `fractality`, and `integration`; inspect
`python -m pytest --markers` for their definitions. Backend-specific markers such
as `requires_jax` and `numpy_only` are not registered project options.

## Validation workflow and reporting

1. Read the relevant doctrine and operator contract; search for existing helpers.
2. Run the relevant baseline before editing. Reproduce a suspected defect with
   an explicit expected result and record whether it fails before the fix.
3. Add a regression that exercises the physical or public API behavior, including
   boundary cases and cache invalidation where relevant.
4. Run the affected tests, then the full default suite before delivery. Run
   applicable slow, optional-backend or research checks explicitly when the
   changed scope requires them.
5. Report exact commands, interpreter/dependency versions, pass/fail/skip counts,
   warnings and any untested scope. Do not assume failures are pre-existing
   without baseline evidence.

The `structural_rng` fixture supplies `numpy.random.default_rng(seed=0)`.
`structural_tolerances` supplies `atol=1e-12` and `rtol=1e-10`; use tolerances
appropriate to the mathematical scale and backend precision and document any
relaxation. The autouse cleanup fixture resets selected global state; tests
that mutate additional caches or configuration must restore those explicitly.

Coverage is a measurement, not proof of a physical invariant. With the test
dependencies installed, generate a report using:

```sh
python -m pytest --cov=tnfr --cov-report=term-missing --cov-report=html
```

The current project configuration does not enforce a coverage percentage.
The [main CI workflow](.github/workflows/ci.yml) runs Python 3.10-3.13, applies
the default pytest selection and reports coverage on Python 3.11. Consult the
workflow files for the current checks rather than duplicating their matrices
or results in test reports.
