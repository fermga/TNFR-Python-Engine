# TNFR arithmetic primality package

This directory contains the standalone `tnfr-primality` 1.1.0 compatibility
package. It evaluates static arithmetic-pressure formulas from supplied divisor
and prime-factor statistics. It does not evolve a nodal network, derive phase,
or establish a cryptographic performance advantage. This README owns current
usage; [TNFR number theory](../theory/TNFR_NUMBER_THEORY.md) owns mathematical scope.

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
python -m pip install ./primality-test
```

Documentation is built from the repository root using the root `.[docs]` extra;
follow [the documentation checks](../TESTING.md#code-quality-and-documentation-checks).
This package's `[docs]` extra remains an empty compatibility alias and does not
install a separate documentation toolchain.

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
| `OptimizedTNFRPrimality` | Cache of final pressure values, batch loop, timing statistics and a classical sieve followed by pressure checks. `is_prime` uses fixed strict tolerance `1e-10`. |

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

The equivalent explicit module is `python -m tnfr_primality.cli`. Standard
options include `--optimized`, `--batch`, `--timing`, `--benchmark`, `--validate`,
`--compare`, `--stats` and `--sieve`. `--validate` compares finite arithmetic
outputs; it does not validate TNFR as a physical theory.

`tnfr-primality-advanced` uses the separate `advanced_cli` parser. Use its own
`--help`; its options are not automatically accepted by the standard parser.
For example:

```bash
tnfr-primality-advanced --validate 7 --json-output
tnfr-primality-advanced 2 4 --json-output
```

JSON mode emits one document on stdout and sends human progress summaries to
stderr. Numeric result fields retain the selected helper's schema: basic
validation uses `tested` and `correct`, while the optional advanced helper uses
`tested_numbers` and `correct_predictions`. Infrastructure JSON contains
`status` and `system_info`, with `system_info=null` if the optional interface is
unavailable. Its benchmark times a fixed prime-only sample bounded by the
requested maximum; it does not test all integers up to that value or measure
composite rejection.
`python -m tnfr_primality` tries the advanced entry point first, so use the
explicit `.cli` module when the standard behavior is intended.

## Optional repository integration

[advanced_core.py](tnfr_primality/advanced_core.py) tries repository arithmetic
helpers and falls back when imports or calls fail. Infrastructure availability
does not certify that every advertised cache/backend is active. The
`use_cache` parameter on `tnfr_delta_nfr_advanced` is retained but its body does
not use it; cached wrappers are separate functions.

`tnfr_is_prime_advanced(n, return_certificate=True)` can return an arithmetic
`PrimeCertificate` when the repository call succeeds. It still returns a
`(bool, pressure)` tuple for `n <= 1`, `n == 2`, absent infrastructure or a caught
certificate failure. Inspect the actual return type. The certificate concerns
supplied arithmetic statistics and a numerical zero test, not independent
proof of nodal emergence or a full structural-field evolution.

Package threshold names and legacy constants are compatibility configuration.
They do not redefine engine-wide phase bounds, coherence fields or grammar.

## Verification and limits

From the repository root:

```bash
python primality-test/test_installation.py
```

The installation checks cover imports, selected known integers, caching and
batch behavior. The focused CLI regression module checks both validation
schemas and clean JSON output across numeric and infrastructure routes:

```bash
python -m pytest primality-test/tests/test_advanced_cli.py -q
``` Their finite timings depend on the environment and workload;
a speedup for repeated cached inputs is not a new primality complexity result.
For benchmark comparisons retain inputs, cache state, implementation branch
and environment. The main engine's arithmetic implementation remains under
`src/tnfr/mathematics`; this compatibility package is not an additional source
of TNFR-wide physical laws. The root [LICENSE.md](../LICENSE.md) applies.
