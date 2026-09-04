#!/usr/bin/env python3
"""R3 CRT multiscale composition benchmark.

Prints, for coprime moduli ``a, b``, the exact CRT Kronecker identity residual
(``L_ab = I - (I-L_a) (x) (I-L_b)``, zero over Q), the numerical child->parent
eigenvalue composition residual (``lambda + mu - lambda*mu``), and the U5
spectral-gap bound ``lambda2(ab) <= min(lambda2(a), lambda2(b))``.  The
unrestricted (non-unit) residue set is shown as the non-factorizing control.

Honest scope: this is a *structural synthesis* theorem.  It **uses the known
factors** ``a, b`` to assemble the parent from its sub-networks; it is **not** a
factoring/discovery algorithm and makes no complexity or cryptographic claim.
The circularity audit therefore flags ``graph_construction_requires_answer`` and
forbids any discovery claim (``permits_discovery_claim = False``).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.crt_multiscale import (  # noqa: E402
    crt_kronecker_residual,
    residue_set_factors,
    u5_spectral_gap_composition,
    verify_spectrum_composition,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

COPRIME_PAIRS = [(3, 5), (5, 7), (4, 9), (5, 9), (7, 8), (3, 11), (9, 11)]
POWERS = [1, 2, 3]


def main() -> int:
    print("R3 CRT multiscale: L_ab = I - (I-L_a) (x) (I-L_b) up to CRT sigma")
    header = (
        f"  {'a':>3} {'b':>3} {'k':>2} {'kron_res':>9} "
        f"{'spec_res':>9} {'gap_ab':>7} {'min_ch':>7} {'U5<=':>5} {'ctrl!':>5}"
    )
    print(header)

    all_exact = True
    all_bounded = True
    control_all_fail = True
    for a, b in COPRIME_PAIRS:
        for k in POWERS:
            kr = crt_kronecker_residual(a, b, k)
            spec = verify_spectrum_composition(a, b, k)
            gab, ga, gb, bounded = u5_spectral_gap_composition(a, b, k)
            unit_ok = residue_set_factors(a, b, k, unit=True)
            full_fail = not residue_set_factors(a, b, k, unit=False)
            all_exact &= (kr == 0) and unit_ok
            all_bounded &= bounded
            control_all_fail &= full_fail
            print(
                f"  {a:>3} {b:>3} {k:>2} {float(kr):>9.1e} {spec:>9.1e} "
                f"{gab:>7.4f} {min(ga, gb):>7.4f} {str(bounded):>5} "
                f"{str(full_fail):>5}"
            )

    audit = CircularityAudit(
        graph_construction_requires_answer=True,  # uses known factors a, b
    )
    manifest = ExperimentManifest(
        claim_id="NT-P03",
        git_sha="local",
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        seed=None,  # deterministic exact rational + eigenvalue arithmetic
        operator_sequence=(),
        uses_known_factors=True,  # structural synthesis branch
        input_bits=input_bit_length(max(a * b for a, b in COPRIME_PAIRS)),
        controls=("unrestricted_residue_set", "non_coprime_guard"),
        artifacts=(),
    )
    print()
    print(f"  Kronecker identity exact (all) : {all_exact}")
    print(f"  U5 gap bound holds (all)       : {all_bounded}")
    print(f"  control set never factors (all): {control_all_fail}")
    print(f"  claim status                   : "
          f"{ClaimStatus.DERIVED.value} + measured")
    print(f"  circularity verdict            : {audit.verdict.value}")
    print(f"  discovery-claim permitted      : {audit.permits_discovery_claim}"
          " (structural synthesis, NOT factoring)")
    ok = all_exact and all_bounded and control_all_fail
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
