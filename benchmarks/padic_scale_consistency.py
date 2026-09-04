#!/usr/bin/env python3
"""R4 projective p-adic tower scale-consistency benchmark.

Prints, for the reduction-compatible family on ``ℤ/pℤ ← ℤ/p^2ℤ ← ...``, the exact
projective consistency residuals (commutation ``R_e P_{e+1} - P_e R_e``, lift
intertwining ``P_{e+1} Lift - Lift P_e``, right-inverse ``R_e Lift - I``; all zero
over Q), the numerical surviving-spectrum containment, and how the spectral gap
scales with the exponent.

Honest scope: this establishes *projective transport consistency* only.  It is
NOT a REMESH claim: the REMESH operator contract (EPI recursion, NETWORK scale,
preserved identity, U5) is unverified, so the tower->REMESH claim (NT-P04) stays
CONJECTURAL (``realizes_remesh == False``).  No complexity/crypto/Millennium claim
is made; p is a chosen modulus base, not a discovered factor.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.padic_tower import (  # noqa: E402
    laplacian_commutation_residual,
    lift_intertwining_residual,
    lift_reduction_residual,
    padic_spectral_gaps,
    projective_commutation_residual,
    remesh_contract_audit,
    surviving_spectrum_containment,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

CASES = [
    (2, 1, frozenset({1})),
    (2, 2, frozenset({1})),
    (3, 1, frozenset({1, 2})),
    (3, 2, frozenset({1, 2})),
    (5, 1, frozenset({1, 2, 3, 4})),
    (5, 2, frozenset({1, 2, 3, 4})),
    (7, 1, frozenset({1, 2, 3, 4, 5, 6})),
]


def main() -> int:
    print("R4 projective p-adic tower: R_e P_{e+1} = P_e R_e (exact over Q)")
    print(f"  {'p':>3} {'e':>2} {'commut':>7} {'intertw':>7} {'Rlift-I':>7} "
          f"{'spec_sub':>9}")
    all_exact = True
    for p, e, base in CASES:
        c = projective_commutation_residual(p, e, base)
        lc = laplacian_commutation_residual(p, e, base)
        it = lift_intertwining_residual(p, e, base)
        lr = lift_reduction_residual(p, e)
        sub = surviving_spectrum_containment(p, e, base)
        exact = (c == 0 and lc == 0 and it == 0 and lr == 0)
        all_exact &= exact
        print(f"  {p:>3} {e:>2} {float(c):>7.0e} {float(it):>7.0e} "
              f"{float(lr):>7.0e} {sub:>9.1e}")

    print()
    print("  spectral gap lambda2(L_e) across levels (base = units):")
    for p in (2, 3, 5):
        base = frozenset(range(1, p))
        gaps = padic_spectral_gaps(p, 3 if p < 5 else 2, base)
        print(f"    p={p}: " + ", ".join(f"e={e}:{g:.4f}" for e, g in gaps))

    audit = CircularityAudit()  # p is a chosen base; no factoring involved
    remesh = remesh_contract_audit()
    manifest = ExperimentManifest(
        claim_id="NT-P04",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,  # deterministic exact rational arithmetic
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(p ** (e + 1) for p, e, _ in CASES)),
        controls=("non_uniform_fine_set", "reproducibility"),
        artifacts=(),
    )
    print()
    print(f"  projective consistency exact (all): {all_exact}")
    print(f"  transport claim  : {ClaimStatus.DERIVED.value} + measured")
    print(f"  circularity      : {audit.verdict.value}")
    print(f"  REMESH realized  : {remesh.realizes_remesh} "
          f"(NT-P04 {ClaimStatus.CONJECTURAL.value}; contract unverified)")
    print(f"  input bits (max) : {manifest.input_bits}")
    return 0 if all_exact and not remesh.realizes_remesh else 1


if __name__ == "__main__":
    raise SystemExit(main())
