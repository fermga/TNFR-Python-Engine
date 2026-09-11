#!/usr/bin/env python3
"""R5 finite-field and Gaussian decomposition signatures benchmark.

Part A (structural): the trace Gauss-period count on F_q.  Prime fields reproduce
gcd(k, p-1)+1 (R2); extensions collide (count <= gcd(k, q-1)+1, strict for some).
The trace-character count and the explicit Cayley spectrum agree.

Part B (descriptive): the k=2 spectrum count of the Z[i]/(p) unit network
separates ramified (2), inert (3), split (6).  The classical type is a
ground-truth LABEL only, so this is a descriptive study; the NT-P05 claim (pulse
detects decomposition) stays CONJECTURAL and is k-sensitive.

No factoring, complexity, cryptographic, or Millennium claim is made.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.algebraic_residue_networks import (  # noqa: E402
    decomposition_signature,
    signature_separates_types,
)
from tnfr.mathematics.finite_fields import (  # noqa: E402
    FiniteField,
    cyclotomic_period_count,
    distinct_period_count,
    explicit_cayley_spectrum_count,
    prime_field_matches_cyclotomy,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

EXT = [(2, 2), (2, 3), (3, 2), (3, 3), (5, 2), (7, 2)]
GAUSS_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def main() -> int:
    print("R5A finite-field Gauss periods: prime-field regression + collisions")
    ok_prime = all(
        prime_field_matches_cyclotomy(p, k)
        for p in (5, 7, 11, 13) for k in (1, 2, 3, 4)
    )
    print(f"  prime-field matches gcd(k,p-1)+1 (all): {ok_prime}")
    print(f"  {'field':>8} {'q':>3} {'k':>2} {'pred':>4} {'count':>5} "
          f"{'explicit':>8} note")
    agree = True
    for p, f in EXT:
        F = FiniteField(p, f)
        for k in (2, 3, 4):
            pred = cyclotomic_period_count(F.q, k)
            cnt = distinct_period_count(F, k)
            exp = explicit_cayley_spectrum_count(F, k)
            agree &= (cnt == exp)
            note = "collision" if cnt < pred else "match"
            print(f"  F_{p}^{f:<5} {F.q:>3} {k:>2} {pred:>4} {cnt:>5} "
                  f"{exp:>8} {note}")

    print()
    print("R5B Z[i]/(p) decomposition signature (k=2, units):")
    for p in GAUSS_PRIMES:
        t, c = decomposition_signature(p, 2)
        print(f"    p={p:>2}  {t:>9}  count={c}")
    sep2 = signature_separates_types(GAUSS_PRIMES, 2)
    sep3 = signature_separates_types(GAUSS_PRIMES, 3)
    print(f"  separates types at k=2: {sep2}   at k=3: {sep3} (k-sensitive)")

    struct = CircularityAudit()  # part A: purely structural period counting
    descr = CircularityAudit(factors_used_only_for_scoring=True)  # part B label
    manifest = ExperimentManifest(
        claim_id="NT-P05",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(p * p for p in GAUSS_PRIMES)),
        controls=("prime_field_regression", "spectral_agreement",
                  "k_sensitivity"),
        artifacts=(),
    )
    print()
    print(f"  period/explicit spectra agree (all): {agree}")
    print(f"  finite-field claim : {ClaimStatus.DERIVED.value} (prime) + "
          f"{ClaimStatus.MEASURED.value} (extensions); {struct.verdict.value}")
    print(f"  Gaussian NT-P05    : {ClaimStatus.CONJECTURAL.value}; "
          f"{descr.verdict.value} (ground-truth label used for scoring)")
    print(f"  input bits (max)   : {manifest.input_bits}")
    return 0 if (ok_prime and agree and sep2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
