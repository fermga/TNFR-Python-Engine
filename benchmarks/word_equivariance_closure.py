#!/usr/bin/env python3
"""N06 word composition-closure benchmark (R1).

Shows the composition-closure theorem: a grammar word built from Gamma-equivariant
operators is itself Gamma-equivariant, so it preserves the Fix(Gamma) / Fix(Gamma)^perp
split. The inductive step is DERIVED; the length-1 base case is the per-operator
MEASURED equivariance (all 13, under cache isolation). Here the five canonical
words (Bootstrap, Bootstrap+close, Stabilize, Propagate, Explore) are measured
equivariant (residual = 0) on a vertex-transitive cycle (rotation) and a
two-orbit star (leaf swap), every prefix stays within tolerance, and a
Gamma-symmetric sweep keeps a symmetric seed orbit-constant.

Honest scope: not new mathematics; closes no open problem. The residual open
piece is the non-equivariant selector (pointed origin), a property of the
selection policy, not of the operators. No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.operators.definitions import (  # noqa: E402
    Coherence,
    Coupling,
    Emission,
    Silence,
)
from tnfr.physics.operator_equivariance import _test_cases  # noqa: E402
from tnfr.physics.word_equivariance import (  # noqa: E402
    audit_word_equivariance,
    composition_closure_holds,
    word_preserves_fix,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def main() -> int:
    print("N06 word composition closure: equivariant factors -> equivariant word")
    print(f"  {'word':<18} {'glyphs':<24} {'residual':>10} {'equiv':>6}")
    all_equiv = True
    for r in audit_word_equivariance():
        all_equiv &= r.is_equivariant
        print(f"  {r.label:<18} {'.'.join(r.glyphs):<24} {r.residual:>10.2e} "
              f"{str(r.is_equivariant):>6}")

    cycle, sigma, node = _test_cases()[0]
    word = [Emission, Coupling, Coherence, Silence]
    full, worst_prefix, holds = composition_closure_holds(word, cycle, sigma,
                                                          node)
    fixed, spread = word_preserves_fix(word, cycle)
    print()
    print(f"  closure witness (Bootstrap+close): full={full:.2e} "
          f"worst_prefix={worst_prefix:.2e} holds={holds}")
    print(f"  Fix(Gamma) preserved (symmetric sweep): fixed={fixed} "
          f"spread={spread:.2e}")

    audit = CircularityAudit()  # pure representation theory / relabeling
    _ = ExperimentManifest(
        claim_id="NT-P01b",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=("AL", "UM", "IL", "SHA"),
        uses_known_factors=False,
        input_bits=input_bit_length(6),
        controls=("cycle_rotation", "star_leaf_swap", "prefix_induction",
                  "fix_preservation_sweep"),
        artifacts=(),
    )
    print()
    print(f"  all canonical words equivariant : {all_equiv}")
    print(f"  composition closure             : {ClaimStatus.DERIVED.value} "
          "(induction) on MEASURED base case")
    print(f"  Fix(Gamma) preservation         : {ClaimStatus.DERIVED.value} "
          "(corollary)")
    print(f"  circularity                     : {audit.verdict.value}")
    ok = all_equiv and holds and fixed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
