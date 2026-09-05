# TNFR arithmetic primality package

This directory contains the standalone `tnfr-primality` 1.1.0 compatibility
package. It evaluates the arithmetic structural-pressure identity documented in
[TNFR number theory](../theory/TNFR_NUMBER_THEORY.md).

For an integer `n >= 2`, the package computes

```text
DeltaNFR(n) = (Omega(n) - 1)
             + (tau(n) - 2)
             + (sigma(n)/n - (1 + 1/n))
```

where `Omega` counts prime factors with multiplicity, `tau` counts divisors and
`sigma` sums divisors. Each term is nonnegative and all three vanish for a prime,
so `DeltaNFR(n) = 0` is equivalent to primality for the implemented domain. The
unit coefficients are canonical because any positive coefficients preserve this
zero set.

This is a structural characterization implemented with divisor and factor
computations. The basic implementation has trial-division scale `O(sqrt(n))`; it
is not presented as an asymptotically faster replacement for established
primality tests. Cached and optional advanced paths affect execution strategy,
not the mathematical criterion.

## Installation

From the repository root:

```bash
python -m pip install ./primality-test
```

## Python API

```python
from tnfr_primality import tnfr_component_breakdown, tnfr_is_prime

is_prime, pressure = tnfr_is_prime(997)
components = tnfr_component_breakdown(997)
```

The stable exports are:

- `tnfr_is_prime`
- `tnfr_delta_nfr`
- `tnfr_component_breakdown`
- `tnfr_structural_triad`
- `OptimizedTNFRPrimality`

## Command line

```bash
tnfr-primality 17 97 997
tnfr-primality 221 --timing
```

Use `tnfr-primality --help` for the options supported by the installed version.

## Verification

```bash
python primality-test/test_installation.py
python -m pytest primality-test -q
```

The authoritative theory remains
[theory/TNFR_NUMBER_THEORY.md](../theory/TNFR_NUMBER_THEORY.md). The main engine's
arithmetic implementation lives under `src/tnfr/mathematics`; this standalone
package must not redefine TNFR-wide constants, field bounds or grammar contracts.
