#!/usr/bin/env python3
"""Finite operator-word relabeling and sequential-sweep probes.

Exact equivariant maps compose on compatible invariant domains and preserve
fixed inputs. These finite fixture measurements do not prove the all-state
premises of that theorem. Nonlinear equivariance alone does not preserve the
orthogonal complement of the fixed subspace.

The selected audit words are compared under cycle rotation and star leaf swap;
one word is also probed at every prefix. A separate insertion-order node sweep
reports orbit constancy and whole-graph field spread. Visiting every node is
not itself an equivariant scheduling rule. These fragments are finite probes,
not the public recipe inventory or certificates of live grammar admission.
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
    print("Finite word relabeling residuals and sequential-sweep observations")
    print(f"  {'word':<18} {'glyphs':<24} {'residual':>10} {'equiv':>6}")
    all_equiv = True
    for r in audit_word_equivariance():
        all_equiv &= r.is_equivariant
        print(
            f"  {r.label:<18} {'.'.join(r.glyphs):<24} {r.residual:>10.2e} "
            f"{str(r.is_equivariant):>6}"
        )

    cycle, sigma, node = _test_cases()[0]
    word = [Emission, Coupling, Coherence, Silence]
    full, worst_prefix, holds = composition_closure_holds(word, cycle, sigma, node)
    fixed, spread = word_preserves_fix(word, cycle)
    print()
    print(
        f"  selected prefix probe (Bootstrap+close): full={full:.2e} "
        f"worst_prefix={worst_prefix:.2e} holds={holds}"
    )
    print(
        f"  sequential-sweep orbit constancy: fixed={fixed} "
        f"whole_graph_spread={spread:.2e}"
    )

    audit = (
        CircularityAudit()
    )  # Declared input audit; this is not a theorem certificate.
    _ = ExperimentManifest(
        claim_id="NT-P01b",
        git_sha="local",
        versions={"python": platform.python_version(), "numpy": np.__version__},
        seed=None,
        operator_sequence=("AL", "UM", "IL", "SHA"),
        uses_known_factors=False,
        input_bits=input_bit_length(6),
        controls=(
            "cycle_rotation",
            "star_leaf_swap",
            "selected_prefix_residuals",
            "sequential_sweep_orbit_constancy",
        ),
        artifacts=(),
    )
    print()
    print(f"  selected word probes within tolerance : {all_equiv}")
    print(
        f"  selected prefix verdict         : {ClaimStatus.MEASURED.value} "
        f"(finite fixture): {holds}"
    )
    print(
        f"  sequential-sweep observation    : {ClaimStatus.MEASURED.value} "
        f"(selected seed): {fixed}"
    )
    print(f"  circularity                     : {audit.verdict.value}")
    print("  No universal all-state or nonlinear-complement preservation is certified.")
    ok = all_equiv and holds and fixed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
