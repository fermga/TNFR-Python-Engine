#!/usr/bin/env python3
"""N08 structural-morphism taxonomy benchmark (R4/R8).

Shows that the morphism taxonomy is NOT an imposed category: every kind emerges
from the nodal equation. A TNFR morphism is an intertwiner M L_src = L_tgt M,
which is exactly the condition that M carries the nodal-equation flow
dEPI/dt = -L EPI from the source network to the target one
(M e^{-s L_src} = e^{-s L_tgt} M). Six kinds emerge (automorphism, relabeling,
coarse-graining, lift, conjugation-intertwiner, and the Reynolds sector
projection Q_Gamma of R1); the folding endomorphism (power map x -> x^2 mod p)
does NOT -- the R8 boundary that keeps the 13-operator catalogue intact.

The genus is the intertwiner; the species are the cells of the (dimension change)
x (rank type) grid, with AUTOMORPHISM subset RELABELING and COARSE_GRAINING
subset PROJECTION as canonical refinements. No new operator, no new grammar rule.

Honest scope: classification derived from the nodal equation; example kinds
measured. Not new mathematics; closes no open problem. No complexity / crypto /
Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.structural_morphism import (  # noqa: E402
    StructuralMorphismKind,
    audit_structural_morphisms,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def main() -> int:
    print("N08 structural morphisms: every kind emerges from the nodal equation")
    header = (f"  {'label':<16} {'kind':<16} {'dom->cod':>10} "
              f"{'intertw':>9} {'nodal_flow':>11} {'emerges':>8} {'operator':>9}")
    print(header)
    results = audit_structural_morphisms()
    all_ok = True
    n_emerge = 0
    for label, c in results:
        emerges = c.emerges_from_nodal_equation
        n_emerge += int(emerges)
        # the intertwiner defect and the nodal-flow defect must agree
        agree = (c.is_intertwiner == emerges)
        all_ok &= agree and (not c.is_operator)
        print(f"  {label:<16} {c.kind.value:<16} "
              f"{f'{c.domain_dim}->{c.codomain_dim}':>10} "
              f"{c.intertwining_residual:>9.1e} {c.nodal_flow_residual:>11.1e} "
              f"{str(emerges):>8} {str(c.is_operator):>9}")

    boundary = [c for _, c in results
                if c.kind is StructuralMorphismKind.ENDOMORPHISM]
    six_emerge = n_emerge == 6
    one_boundary = len(boundary) == 1 and not boundary[0].emerges_from_nodal_equation

    audit = CircularityAudit()  # pure linear algebra of the nodal generator
    _ = ExperimentManifest(
        claim_id="NT-P08",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(9),
        controls=("intertwiner_equals_nodal_flow", "reynolds_is_projection",
                  "folding_endomorphism_boundary", "no_operator_match",
                  "relabel_invariant"),
        artifacts=(),
    )
    print()
    print(f"  intertwiner == nodal-flow (all)  : {all_ok}")
    print(f"  six kinds emerge, one boundary   : {six_emerge and one_boundary}")
    print(f"  taxonomy from nodal equation     : {ClaimStatus.DERIVED.value} "
          "(intertwiner = nodal-flow transport)")
    print(f"  operator vs morphism boundary    : {ClaimStatus.MEASURED.value} "
          "(R8; no 14th operator)")
    print(f"  circularity                      : {audit.verdict.value}")
    ok = all_ok and six_emerge and one_boundary
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
