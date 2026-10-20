# TNFR arithmetic primality package

This [optional arithmetic application](../README.md) contains the standalone
`tnfr-primality` 1.1.0 compatibility
package. It evaluates static arithmetic-pressure formulas from supplied divisor
and prime-factor statistics. It does not evolve a nodal network, derive phase,
or establish a cryptographic performance advantage. This README owns current
usage; [TNFR number theory](../../theory/TNFR_NUMBER_THEORY.md) owns mathematical scope.

## Mathematical criterion and runtime scope

For an integer `n >= 2`, the default pressure is

```text
DeltaNFR(n) = (Omega(n) - 1)
             + (tau(n) - 2)
             + (sigma(n)/n - (1 + 1/n))
```

`Omega` counts prime factors **with multiplicity**, `tau` counts divisors and
`sigma` sums divisors. In exact arithmetic each term is nonnegative and all
three vanish for primes. Every composite has positive pressure; for unit
weights its first two terms already sum to at least 2. Thus the exact zero
set is precisely the primes. Any positive coefficient triple preserves that
zero set, which leaves the coefficients undetermined: unit weights are the
package's selected normalization, not a uniquely derived physical constant.

The implementation first enumerates divisors and factors the input. Its
`O(sqrt(n))` arithmetic-loop scale is not a bit-complexity improvement over
established primality algorithms. Floating division and `sqrt`-based loops
are separate implementation limits; finite tests are not a proof for arbitrary
Python integers. No universal accuracy or timing guarantee follows from the
exact-real identity.

## Installation and primary API

From the repository root:

```bash
python -m pip install ./applications/primality-test
```

The basic package is independent of the engine and retains Python 3.8+ support.
Its runtime uses the standard library. The optional `dev` extra installs the
local test/build tools; documentation uses the root repository's `.[docs]`
extra and [documentation checks](../../TESTING.md#code-quality-and-documentation-checks).

```python
from tnfr_primality import (
    OptimizedTNFRPrimality,
    tnfr_component_breakdown,
    tnfr_delta_nfr,
    tnfr_is_prime,
    tnfr_structural_triad,
)

is_prime, pressure = tnfr_is_prime(997)
components = tnfr_component_breakdown(221)
summary = tnfr_structural_triad(221)
optimizer = OptimizedTNFRPrimality()
results = optimizer.batch_test([17, 21, 97])
```

| API | Return and configuration |
|---|---|
| `tnfr_delta_nfr(n, *, zeta=1, eta=1, theta=1)` | Floating pressure; `n < 2` returns positive infinity. Positive coefficients are premises of the zero-set result, not enforced by this function. |
| `tnfr_is_prime(n, *, tolerance=1e-10)` | `(abs(pressure) < tolerance, pressure)` using unit coefficients. This is a strict numerical predicate; custom tolerances can alter classification. |
| `tnfr_component_breakdown(...)` | Weighted pressure components and arithmetic statistics; the retained key `omega` means multiplicity count `Omega`. |
| `tnfr_structural_triad(...)` | Compatibility bundle with `EPI`, `vf`, `delta_nfr`, `local_coherence`, `components`. Despite its name it does not return canonical `(EPI, capacity, phase)`; phase is absent. |
| `OptimizedTNFRPrimality` | Cache of final pressure values, batch loop, timing statistics and a classical sieve followed by pressure checks. `is_prime` uses fixed strict tolerance `1e-10`. The bounded cache retains its first entries; it is not an LRU policy. |

Arithmetic EPI/capacity formulas are configured readouts of the same statistics.
Returning them does not derive their dynamics or make the integer a maintained
NFR. Neither these bundles nor arithmetic certificates constitute an executed
operator sequence or the complete structural tetrad.

## Command line

The standard installed command maps to `tnfr_primality.cli:main`:

```bash
tnfr-primality 17 97 997
tnfr-primality 221 --timing
tnfr-primality --batch --optimized 17 21 97
tnfr-primality --validate 1000
```

`python -m tnfr_primality` and `python -m tnfr_primality.cli` use this same
basic parser without importing advanced adapters. Options include `--optimized`,
`--batch`, `--timing`, `--benchmark`, `--validate`, `--compare`, `--stats`,
`--sieve` and `--json-output`. `--validate` compares finite arithmetic outputs;
it does not validate TNFR as a physical theory.

```bash
python -m tnfr_primality 2 4 1 --json-output
python -m tnfr_primality --validate 100 --json-output
```

JSON mode emits one document on stdout. Number rows contain `number`,
`is_prime`, `delta_nfr`, `delta_nfr_available`, `method` and `time_ms`.
For `n < 2`, unavailable pressure is `null` with availability `false`, rather
than a nonstandard JSON infinity. Validation uses the basic `tested`/`correct`
schema; sieve output is `{"primes": [...]}`. Benchmark and comparison modes
retain their basic reports. Human progress and requested cache statistics go
to stderr.

## Optional repository integration

The engine's `tnfr.primality` adapter can use these basic arithmetic functions.
It does not need an advanced module or a `full` extra. Core engine arithmetic
also remains independently available under `tnfr.mathematics`.

The former advanced adapters and CLI are retired. `tnfr-primality-advanced`,
`tnfr-primality-legacy` and the `full`, `benchmark` and `docs` extras are no
longer installation targets. Use `tnfr-primality` or the basic module command
for supported arithmetic output; advanced-only options are rejected.
[Archive recovery](archive/README.md) preserves their source identities and
unique working-tree examples/tests outside runtime discovery and package
payloads.

The former optional `PrimeCertificate` route classified supplied arithmetic
terms against a tolerance. It did not independently authenticate those terms
or prove primality from an untrusted report. Its retirement changes neither
the exact zero-set theorem nor the core arithmetic implementation. Package
threshold names do not redefine phase bounds, fields or operator grammar.

## Executable examples

After installing this application, run the maintained example:

```bash
python applications/primality-test/examples/basic_usage.py
```

It reports selected unit coefficients, factor multiplicity and cache timings.
It neither benchmarks arbitrary integer sizes nor validates physical TNFR.
The advanced example and former promotional benchmark belong to archive
recovery, not the current learning or performance route.

## Verification and limits

From the repository root:

```bash
python applications/primality-test/test_installation.py
```

The installation checks cover imports, selected known integers, caching and
batch behavior. The application suite checks basic CLI/module parity, strict
JSON, examples, package contents and the absence of advanced auto-imports:

```bash
python -m pytest applications/primality-test/tests -q
```

Their finite timings depend on the environment and workload;
a speedup for repeated cached inputs is not a new primality complexity result.
For benchmark comparisons retain inputs, cache state, implementation branch
and environment. The main engine's arithmetic implementation remains under
`src/tnfr/mathematics`; this compatibility package is not an additional source
of TNFR-wide physical laws. The root [LICENSE.md](../../LICENSE.md) applies.
