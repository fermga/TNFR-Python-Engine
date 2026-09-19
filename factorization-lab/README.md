# TNFR spectral factorization lab

This experimental lab combines supplied Jacobi-residue graphs, spectral
heuristics, arithmetic statistics and optional partition workflows. It returns
factor candidates and diagnostic records. It establishes neither autonomous
factor recovery from the nodal equation nor a complexity improvement or a
cryptographic factorization capability.

This README owns current usage and configuration. The mathematical scope is in
[TNFR number theory](../theory/TNFR_NUMBER_THEORY.md); the
[applied analysis overview](../theory/APPLIED_STRUCTURAL_ANALYSIS.md) links here.

## Results and verification

| Result | What it means |
|---|---|
| `candidate_factors` | Union of heuristic seeds, arithmetic refinements and later partition hints; provenance depends on mode. It need not be a complete prime factorization. |
| `tnfr_certified_factors` | Compatibility name for the configured structural acceptance rule. The verifier records `support_divisible` but does not require it in its final acceptance condition. |
| `tnfr_verification` | Per-candidate scores, thresholds, endorsements and a separate divisibility observation. |
| `factor_signature`, `composite_signature` | Serialized analysis/signature records; hashes check recorded content, not mathematical truth. Pure-mode composite powers and `structural_complete` are hypotheses. |
| `certificate_path`, partition paths | Optional exported analysis and workflow records; a proposed grammar-valid word is not by itself an executed operator trace. |

Check every claimed proper factor independently:

```python
verified = sorted({d for d in result.candidate_factors if 1 < d < n and n % d == 0})
```

Divisibility establishes that a candidate is a factor. It does not establish
that the factor is prime, that all multiplicities were recovered, or that
factor recovery came from a TNFR mechanism. A complete prime factorization
requires those additional checks.

## Setup and public interfaces

Use a supported Python environment for the main engine. From the repository
root, install its dependencies and make the source package available:

```bash
python -m pip install -e .
```

The local lab is a source directory, without its own `setup.py` or
`pyproject.toml`. To import its high-level interface directly, run Python from
the lab directory:

```bash
cd factorization-lab
python -m tnfr_factorization.cli 221 --max-nodes 4097 --json
```

```python
from tnfr_factorization import factorize

n = 221
result = factorize(n, pure=False, trace=False, max_nodes=4097)
verified = sorted({d for d in result.candidate_factors if 1 < d < n and n % d == 0})
print(verified)
print(result.tnfr_verification)
```

There are three related interfaces; their arguments and result types differ:

| Interface | Configuration | Return |
|---|---|---|
| `tnfr_factorization.factorize` | `pure`, `trace`, `max_nodes`, `modulus` | `FactorizationResult` |
| `SpectralPaleyFactorizer(...).analyze` | Constructor sets `max_nodes`, backend and diagnostic options; `analyze` takes `modulus`, `trace_certificates`, `certificate_dir` | `SpectralAnalysisResult` |
| `tnfr.factorization.factorize` | `modulus`, `trace_certificates`, `certificate_dir`; reuses one default factorizer and reads the environment policy | `SpectralAnalysisResult` |

The engine wrapper discovers the sibling lab in a source checkout. A deployment
without that directory needs an importable `tnfr_factorization` package.

**Node cap:** omitting the low-level constructor argument uses 4097. Passing
`None` disables the cap. The high-level wrapper and CLI currently pass `None`
when their argument is omitted.
Set an explicit positive `max_nodes` / `--max-nodes` for consistent behavior.
CLI `--max-nodes 0` also disables the cap; the Python constructor does not give
zero that special meaning.

Automatic modulus selection rounds upward to a value at least `n` and 5 that
is 1 modulo 4. It does not prove the modulus prime. For explicit overrides,
odd values 1 modulo 4 are the intended domain; the builder currently enforces
only the lower bound 5. Classical prime-modulus Paley spectral identities need
their own hypotheses. This graph is an arithmetic input construction, not an
emergent support selected by a closed nodal law.

## Candidate policy and configuration

`pure=True` sets `TNFR_PURE_MODE` temporarily and restores its prior value.
`pure=None` inherits the environment. This override is process-wide during the
call, so concurrent callers using different policies must coordinate.

The pure policy skips arithmetic-statistic seed hints and gcd refinement in
the **initial candidate stage**, and uses a periodic-confidence rule in one
partition stage. It does not make the complete pipeline arithmetic-free:

- `_compute_arithmetic_telemetry` factors `n` to obtain `Omega`, `tau` and
  `sigma` in both modes. Its cached results remain arithmetic inputs.
- Even-number hints, integer-size seeds and later partition-size divisibility
  checks remain active.
- An empty initial candidate list reaches trial division without a pure-mode
  guard. Its default bound is `isqrt(n)`.
- The structural verifier records `n % factor == 0` separately from acceptance.

| Setting | Actual role |
|---|---|
| `TNFR_PURE_MODE` | Select the partial pure policy; truthy values are `1`, `true`, `yes`, `on`. |
| `TNFR_PURE_MODE_VERIFY_DIVISIBILITY` | Filter initial pure seeds by proper divisibility. It is not a final filter over every later candidate or acceptance label. |
| `TNFR_FACTOR_FALLBACK_MAX_DIVISOR` | Positive integer cap for the empty-candidate trial fallback. Missing, invalid or nonpositive values use `isqrt(n)`. It does not cap arithmetic telemetry factorization. |
| `TNFR_DISABLE_OPTIMIZER` | Disable the optional sequence optimizer; it does not change candidate verification into a theorem. |
| `TNFR_FAILURE_TELEMETRY` | Control failure diagnostics. For isolated low-level calls use the constructor's `failure_telemetry=False`. |
| `TNFR_PARTITION_TARGET_SIZE`, `TNFR_PARTITION_OVERLAP` | Partition planner settings; defaults are 256 and 4. Planning may further adapt partition size. |
| `TNFR_PARTITION_OUTPUT_DIR` | Override partition-export destination. |

The CLI exposes backend/dispatcher choices through `--fft-backend`,
`--fft-dispatcher` and related options; consult `--help` for their spelling.
Its `--json` output is one JSON list of target results. Pure mode is selected
through the environment, not a CLI `--pure` flag. Explicit trace export uses
the Python APIs; `trace=False` does not guarantee the absence of all optional
diagnostic writes. Trace output normally uses `factorization-lab/results/`;
the low-level API accepts an explicit certificate directory.

## Telemetry and mechanism boundaries

The lab's compatibility field names are not interchangeable with canonical
nodewise fields from `tnfr.physics.fields`:

| Lab field | Computation or source |
|---|---|
| `phi_s` | Normalized edge density; no nodewise pressure or path-distance input. |
| `phase_gradient` | Selected spectral gap divided by node count. |
| `phase_curvature` | Largest supplied eigenvalue divided by node count. |
| `coherence_length` | Backend-provided value when positive, otherwise inverse selected gap. Record backend and normalization before comparison. |
| Arithmetic EPI, capacity, pressure | Static functions of already computed factor/divisor statistics. They supply no phase or autonomous evolution law. |
| Energy/Noether proxies | Algebraic combinations of lab features; their names do not prove an energy balance or conserved charge. |

The fallback spectrum on regular unit-weight graphs equals the combinatorial
Laplacian spectrum `d * spectrum(L_rw)`. Its inverse selected gap is not the
normalized EPI diffusion timescale `1 / (nu * lambda_2(L_rw))` without the
corresponding degree/time conversion. Selection skips eigenvalues at or below
`1e-9`, so it need not select the true second eigenvalue.

Partition pressure reduction includes a fixed scalar attenuation surrogate.
That calculation alone is not execution of the named TNFR operators or a
convergence theorem. Optional optimizer/workflow records must be assessed on
their actual captured execution. Lab thresholds are configured heuristic
criteria, not universal TNFR constants or a proof of U5.

## Verification and historical evidence

From the repository root, existing focused regression modules include:

```bash
python -m pytest factorization-lab/tests/test_spectral_paley.py factorization-lab/tests/test_cli.py -q
```

These exercise selected candidate, fallback, serialization and CLI cases. The
[false-positive tests](tests/test_false_positive_verifier.py) test a finite
selection of adversarial inputs; acceptance on that selection is not a general
factor certificate. Benchmarks apply only to recorded inputs, environment,
backend and cache state. Arithmetic telemetry and fallback costs belong in any
end-to-end comparison.

Run the actual verifier and configured-criteria controls directly:

```bash
python -m pytest factorization-lab/tests/test_false_positive_verifier.py factorization-lab/tests/test_verification_robustness.py -q
```

These suites read the production implementation. The copied-criteria simulation
and duplicate standalone runners have been retired. The verifier suite uses a
finite curated sample by default; `TNFR_RUN_LONG_TESTS=1` opts into its larger
generated input set. Criteria-range checks preserve a configured policy; they
do not derive the thresholds or certify general false-positive resistance.

The obsolete live `notebooks/spectral_history.ipynb` has been retired. Its
[unaltered historical copy](notebooks/archive/spectral_history_legacy_2026_09_19.ipynb)
retains saved code, outputs and metadata. It uses the removed `_fft_engine`
attribute, a synthetic spectrum and a notebook-incompatible `__file__` path;
it is not current executable guidance. Use the maintained tests above for
regressions. The archive is 37,107 bytes, SHA-256
`1b67bb2a7fdbd9b26fb9a96082f7519111a2287dab974e94436cf2f6dea6dfe5`.

The root [LICENSE.md](../LICENSE.md) is authoritative for this repository.
