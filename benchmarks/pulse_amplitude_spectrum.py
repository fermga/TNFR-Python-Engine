#!/usr/bin/env python3
"""N12 pulse-amplitude benchmark (R2): amplitudes = normalized multiplicities.

R2 fixed the pulse ORDER (Hankel rank = #distinct eigenvalues = gcd(k,p-1)+1).
This benchmark fixes its AMPLITUDES. On the k-th power residue circulant the
pointed seed e0 has uniform Fourier weight 1/p, so the pulse
h(t) = e0* e^{-tL} e0 = sum_lambda (m_lambda/p) e^{-t lambda} carries the exact
amplitude a_lambda = m_lambda/p on each distinct eigenvalue.

Two independent confirmations: the orthogonal spectral projector gives
e0* P_lambda e0 = m_lambda/p (basis-independent under any unitary rotation of a
degenerate eigenspace), and sum_lambda (m_lambda/p) lambda^m reconstructs the exact
rational moments mu_m = e0^T L^m e0. Amplitudes are exact rationals summing to 1.

Honest scope: exponential in log2(p) (a p-node network), like the R2 rank -- not a
fast primality test. No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.pulse_amplitudes import (  # noqa: E402
    certify_pulse_amplitudes,
    pulse_amplitudes,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

CASES = [(5, 2), (7, 3), (11, 2), (13, 4)]


def main() -> int:
    print("N12 pulse amplitudes: a_lambda = m_lambda / n (normalized multiplicity)")
    header = (f"  {'(p,k)':<8} {'ndist':>5} {'rank':>5} {'sum=1':>5} "
              f"{'a=m/p':>6} {'moment_res':>11} {'basis_res':>11}")
    print(header)
    all_ok = True
    for p, k in CASES:
        c = certify_pulse_amplitudes(p, k)
        ok = (c.rank_matches_cyclotomy and c.amplitudes_sum_to_one
              and c.amplitude_equals_multiplicity
              and c.moment_reconstruction_residual < 1e-9
              and c.basis_invariance_residual < 1e-9)
        all_ok &= ok
        print(f"  ({p:2d},{k})  {c.n_distinct:>5} {gcd(k, p - 1) + 1:>5} "
              f"{str(c.amplitudes_sum_to_one):>5} "
              f"{str(c.amplitude_equals_multiplicity):>6} "
              f"{c.moment_reconstruction_residual:>11.1e} "
              f"{c.basis_invariance_residual:>11.1e}")

    print()
    print("  amplitude spectrum (p=11, k=2):")
    for lam, m, a in pulse_amplitudes(11, 2):
        print(f"    lambda={lam.real:+.3f}{lam.imag:+.3f}j  m={m}  a={a}")

    audit = CircularityAudit()  # pure spectral linear algebra, no factoring
    _ = ExperimentManifest(
        claim_id="NT-P02b",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=0,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(p for p, _ in CASES)),
        controls=("amplitude_equals_multiplicity", "exact_moment_reconstruction",
                  "basis_invariant", "rank_matches_cyclotomy"),
        artifacts=(),
    )
    print()
    print(f"  all checks pass          : {all_ok}")
    print(f"  amplitudes = m_lambda/n  : {ClaimStatus.DERIVED.value} "
          "+ MEASURED (exact reconstruction)")
    print("  claim                    : NT-P02b")
    print(f"  circularity              : {audit.verdict.value}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
