#!/usr/bin/env python3
"""N10 trace-collision benchmark (R5): loss of observability under the trace.

R5 reframed from a k-dependent type detector to a durable observability result:
for the k-th powers H in F_q, the trace Tr : F_q -> F_p collapses H onto the p
residues. The fiber counts N_a = #{h in H : Tr(h)=a} give the collision
structure; #{a : N_a > 0} is the number of trace values that stay observable.
The character formula (Fourier inversion on F_p) reproduces N_a exactly, and the
trace is Galois-invariant (Tr(h^p)=Tr(h)), so the histogram is a
representation-free field invariant.

Prime fields reproduce R2 (identity trace, no collisions -- H fully observable);
extensions collapse H onto fewer residues (loss of observability). This does NOT
build a k-selective type detector (NT-P05c CONJECTURAL); it publishes the
collision structure as an exact, invariant observability theory. No complexity /
crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.finite_fields import FiniteField  # noqa: E402
from tnfr.mathematics.trace_collisions import (  # noqa: E402
    certify_trace_collisions,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

CASES = [(5, 1), (7, 1), (2, 3), (3, 2), (5, 2)]


def main() -> int:
    print("N10 trace collisions: exact fiber counts vs character formula")
    header = (f"  {'F_q':<12} {'k':>2} {'|H|':>4} {'obs':>4} {'full':>5} "
              f"{'collide':>7} {'maxfib':>6} {'charres':>9} {'galois':>6}")
    print(header)
    all_exact = True
    all_galois = True
    prime_no_collision = True
    ext_collision = False
    for p, f in CASES:
        field = FiniteField(p, f)
        for k in (2, 3, 4):
            c = certify_trace_collisions(field, k)
            assert c.subset_size == (field.q - 1) // gcd(k, field.q - 1)
            all_exact &= c.character_formula_residual < 1e-9
            all_galois &= c.galois_invariant
            if f == 1:
                prime_no_collision &= not c.has_collisions
            elif c.has_collisions:
                ext_collision = True
            print(f"  F_{p}^{f}(q={field.q:<2}) {k:>2} {c.subset_size:>4} "
                  f"{c.observed_values:>4} {str(c.full_support):>5} "
                  f"{str(c.has_collisions):>7} {c.max_fiber:>6} "
                  f"{c.character_formula_residual:>9.1e} "
                  f"{str(c.galois_invariant):>6}")

    audit = CircularityAudit()  # pure field arithmetic; no factoring
    _ = ExperimentManifest(
        claim_id="NT-P05b",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(p ** f for p, f in CASES)),
        controls=("character_formula_exact", "galois_invariant",
                  "prime_field_no_collision", "extension_collapses"),
        artifacts=(),
    )
    print()
    print(f"  character formula exact (all)   : {all_exact}")
    print(f"  trace Galois-invariant (all)    : {all_galois}")
    print(f"  prime fields: no collisions (R2): {prime_no_collision}")
    print(f"  extensions collapse H           : {ext_collision}")
    print(f"  collision count / char formula  : {ClaimStatus.DERIVED.value} "
          "+ MEASURED (Fourier inversion)")
    print(f"  type detector                   : {ClaimStatus.CONJECTURAL.value}"
          " / not claimed (NT-P05c)")
    print(f"  circularity                     : {audit.verdict.value}")
    ok = (all_exact and all_galois and prime_no_collision and ext_collision)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
