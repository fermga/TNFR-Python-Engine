#!/usr/bin/env python3
"""N07 pointed-symmetry benchmark (R1): the declared break Aut(G) -> Gamma_v.

Acting at a chosen origin v reduces the symmetry from Aut(G) to the stabilizer
Gamma_v = {g in Aut(G) : g(v) = v}. This is the non-equivariant SELECTOR left open
by the word theorem (N06). Three facts make the break declared, not spontaneous:
orbit-stabilizer |Aut(G)| = |Gamma_v| * |orbit(v)|; residual sectors refine
(Fix(Aut) subset Fix(Gamma_v)); and a pointed Emission leaves Fix(Aut) but stays
in Fix(Gamma_v). Origins in one orbit are conjugate (Gamma_{g(v)} = g Gamma_v g^-1),
so no origin is privileged -- the basis of the R2 pointed residue networks.

Honest scope: group theory (orbit-stabilizer, conjugacy) applied to the canonical
operators; not new mathematics, closes no open problem. No complexity / crypto /
Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.pointed_symmetry import (  # noqa: E402
    audit_pointed_symmetry,
    conjugate_stabilizer_holds,
    pointed_symmetry_context,
)
from tnfr.physics.operator_equivariance import _seed  # noqa: E402
from tnfr.physics.symmetry_sectors import (  # noqa: E402
    automorphism_permutations,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def main() -> int:
    print("N07 pointed symmetry: declared break Aut(G) -> Gamma_v at origin v")
    header = (f"  {'case':<14} {'|Aut|':>6} {'|Gv|':>5} {'orbit':>6} "
              f"{'autOrb':>7} {'resOrb':>7} {'O-S':>5} {'break':>7} "
              f"{'stabRes':>8} {'break?':>6} {'presv?':>6}")
    print(header)
    all_os = True
    all_break = True
    all_preserve = True
    for label, ctx, res in audit_pointed_symmetry():
        all_os &= ctx.orbit_stabilizer_holds
        all_break &= res.broke_full_symmetry
        all_preserve &= res.preserves_stabilizer
        print(f"  {label:<14} {ctx.full_order:>6} {ctx.stabilizer_order:>5} "
              f"{ctx.origin_orbit_size:>6} {ctx.aut_orbit_count:>7} "
              f"{ctx.residual_orbit_count:>7} {str(ctx.orbit_stabilizer_holds):>5} "
              f"{res.break_magnitude:>7.4f} {res.stabilizer_residual:>8.1e} "
              f"{str(res.broke_full_symmetry):>6} "
              f"{str(res.preserves_stabilizer):>6}")

    # origin conjugation on the star: leaf 1 vs leaf 3 are conjugate
    import networkx as nx
    star = nx.star_graph(4)
    _seed(star, lambda n: 0.1 if n == 0 else 0.3,
          lambda n: 0.5 if n == 0 else 0.3, lambda n: 1.0)
    g = next(p for p in automorphism_permutations(star) if p[1] == 3)
    conj = conjugate_stabilizer_holds(star, 1, g)
    same_order = (pointed_symmetry_context(star, 1).stabilizer_order
                  == pointed_symmetry_context(star, 3).stabilizer_order)
    print()
    print(f"  conjugate Gamma_3 = g Gamma_1 g^-1 : {conj}")
    print(f"  orbit mates share |Gamma_v|        : {same_order}")

    audit = CircularityAudit()  # pure group theory on relabelings
    _ = ExperimentManifest(
        claim_id="NT-P01c",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=("AL",),
        uses_known_factors=False,
        input_bits=input_bit_length(6),
        controls=("orbit_stabilizer_count", "residual_sector_refinement",
                  "break_localization", "singleton_orbit_no_break",
                  "origin_conjugation"),
        artifacts=(),
    )
    print()
    print(f"  orbit-stabilizer holds (all)  : {all_os}")
    print(f"  pointed break Aut->Gamma_v    : {all_break and all_preserve}")
    print(f"  pointed reduction             : {ClaimStatus.DERIVED.value} "
          "(group theory) + MEASURED")
    print(f"  circularity                   : {audit.verdict.value}")
    ok = all_os and all_break and all_preserve and conj and same_order
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
