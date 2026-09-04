#!/usr/bin/env python3
"""R2 arithmetic pulse recurrence benchmark.

Prints the exact rank identity ``Hankel = Krylov = gcd(k, p-1)+1`` across primes
and powers ``k``, and emits a C5 reproducibility manifest for the claim.  Honest
scope: a recurrence-order identity, exponential in the input bit length
``log2(p)`` — not a primality or factoring speedup.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.arithmetic_pulse import (  # noqa: E402
    cyclotomic_rank,
    pointed_pulse_hankel_rank,
    pointed_pulse_krylov_dimension,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31]
POWERS = [1, 2, 3, 4, 6]


def main() -> int:
    print("R2 arithmetic pulse recurrence: Hankel = Krylov = gcd(k, p-1)+1")
    print(f"  {'p':>4} {'k':>2} {'Hankel':>7} {'Krylov':>7} {'gcd+1':>6} match")
    all_ok = True
    for p in PRIMES:
        for k in POWERS:
            h = pointed_pulse_hankel_rank(p, k)
            kr = pointed_pulse_krylov_dimension(p, k)
            c = cyclotomic_rank(p, k)
            ok = h == kr == c
            all_ok &= ok
            print(f"  {p:>4} {k:>2} {h:>7} {kr:>7} {c:>6} {ok}")

    audit = CircularityAudit()  # no factors, gcd, or phi/Omega/tau/sigma used
    manifest = ExperimentManifest(
        claim_id="NT-P02",
        git_sha="local",
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        seed=None,  # deterministic, exact rational arithmetic (no RNG)
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(PRIMES)),
        controls=("composite_controls", "point_relabel", "spectral_agreement"),
        artifacts=(),
    )
    print()
    print(f"  all primes match: {all_ok}")
    print(f"  claim status    : {ClaimStatus.DERIVED.value} + measured")
    print(f"  circularity     : {audit.verdict.value} "
          f"(discovery-claim permitted: {audit.permits_discovery_claim})")
    print(f"  input bits (max): {manifest.input_bits} (poly(p) = exp(log2 p))")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
