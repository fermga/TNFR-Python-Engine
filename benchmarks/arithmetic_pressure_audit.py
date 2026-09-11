#!/usr/bin/env python3
"""R7 arithmetic-pressure independence and completeness audit benchmark.

Separates the three claims about the three-channel pressure
DeltaNFR(n) = (Omega(n)-1) + (tau(n)-2) + (sigma(n)/n - (1 + 1/n)):

- primality sufficiency (PROVED): each channel alone is 0 iff prime, so the set
  is REDUNDANT / non-minimal for primality (a single channel suffices);
- linear independence (MEASURED): rank 3, no affine relation, despite strong
  correlation (correlation is not dependence);
- structural completeness (OPEN): unproven; the fourth-channel gate stays closed.

Honest scope: computing the channels requires factoring n, so arithmetic pressure
is a STRUCTURAL descriptor, not a primality-discovery algorithm (circularity
verdict CIRCULAR: uses factorization in features). The "minimal and complete"
language is downgraded to explicit scope (claim NT-P07 OPEN).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.arithmetic_pressure import (  # noqa: E402
    all_channels_sufficient,
    channel_correlations,
    channel_rank,
    channels_nonnegative,
    completeness_proven,
    factor_class,
    has_linear_relation,
    is_redundant_for_primality,
    minimal_channels_for_primality,
    pressure_by_class,
    pressure_zero_iff_prime,
    prove_functional_independence,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

LO, HI = 2, 1000


def main() -> int:
    print("R7 arithmetic-pressure audit on [%d, %d]" % (LO, HI))
    print()
    print("1. Primality sufficiency (PROVED):")
    suff = all_channels_sufficient(LO, HI)
    nonneg = channels_nonnegative(LO, HI)
    zero_iff = pressure_zero_iff_prime(LO, HI)
    print(f"   each channel 0 iff prime : {suff}")
    print(f"   all channels nonnegative : {nonneg}")
    print(f"   sum 0 iff prime          : {zero_iff}")
    redundant = is_redundant_for_primality(LO, HI)
    minimal = minimal_channels_for_primality(LO, HI)
    print(f"   minimal channels for primality: {minimal} "
          f"(redundant / non-minimal: {redundant})")

    print()
    print("2. Linear independence (PROVED + MEASURED):")
    rank = channel_rank(LO, HI)
    rel = has_linear_relation(LO, HI)
    corr = channel_correlations(LO, HI)
    proof = prove_functional_independence()
    print(f"   rank[c1 c2 c3] = {rank} (3 => independent); "
          f"affine relation: {rel}")
    print(f"   exact witness proof over Q: rank {proof.rank}, "
          f"independent = {proof.independent} (a=b=c=0)")
    print(f"   correlations: c1-c2={corr[0][1]:.3f} c1-c3={corr[0][2]:.3f} "
          f"c2-c3={corr[1][2]:.3f} (correlated but independent)")

    print()
    print("3. Class-conditioned pressure (factor class):")
    for cls, s in pressure_by_class(LO, HI, factor_class).items():
        print(f"   {cls:16s} n={int(s['count']):4d} mean={s['mean']:8.3f} "
              f"range=[{s['min']:.2f}, {s['max']:.2f}]")

    print()
    print("4. Structural completeness (OPEN):")
    print(f"   completeness proven: {completeness_proven()} "
          "(no proof no fourth degree exists)")

    audit = CircularityAudit(uses_factorization_in_features=True)
    manifest = ExperimentManifest(
        claim_id="NT-P07",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,  # deterministic exact arithmetic
        operator_sequence=(),
        uses_known_factors=True,  # channels are computed FROM the factorization
        input_bits=input_bit_length(HI),
        controls=("channel_ablation", "class_conditioning",
                  "linear_dependence"),
        artifacts=(),
    )
    print()
    print(f"   NT-P07a individual sufficiency : {ClaimStatus.PROVED.value}")
    print(f"   NT-P07b functional independence: {ClaimStatus.PROVED.value} "
          "(exact witness proof over Q)")
    print(f"   NT-P07c minimality (primality) : {ClaimStatus.NEGATIVE.value} "
          "(redundant; single channel suffices)")
    print(f"   NT-P07d completeness           : "
          f"{ClaimStatus.CONJECTURAL.value} / OPEN (task-scoped)")
    print(f"   NT-P07e primality-as-algorithm : {audit.verdict.value} "
          "(uses factorisation; no algorithmic claim)")
    ok = (suff and nonneg and zero_iff and rank == 3 and redundant
          and proof.independent)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
