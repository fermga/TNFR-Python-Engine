#!/usr/bin/env python3
"""N05 transient U2/U6 certificate benchmark.

Shows the R9 canonical finding: directed random-walk diffusion has NO
non-consensus transient in the Euclidean per-node energy. On the L-invariant
non-consensus subspace {y : pi^T y = 0} (orthonormal basis) the symmetric part of
the generator L_sub is positive definite, so the semigroup e^{-s L_sub} is a
genuine contraction (peak_gain = 1, Kreiss lower bound <= 1). The naive ambient
operator-norm gain > 1 is EXACTLY the oblique consensus-projection factor ||Q||
(peak at s = 0) -- a coordinate artifact, not dynamical growth. This reinforces
N03 (contraction in L2(pi)).

The U2 integral J is finite (J <= M ||LQ|| ||x0|| / omega, N04) and the U6
structural potential Phi_s(s) = -B L e^{-sL} Q x0 (canonical inverse-square B) is
confined below pi/2 over the FULL trajectory for a bounded perturbation.

Honest scope: symmetric_part_min_eig > 0 is MEASURED (2e5 random + in-hub, no
counterexample); the general positive-definiteness is CONJECTURAL. This does NOT
decide the canonical U2 metric (NT-P09b/c OPEN) and does NOT modify U2/U6. No
complexity / crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import directed_cayley_adjacency  # noqa: E402
from tnfr.physics.transient_u2 import certify_transient_u2  # noqa: E402
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


CASES = [
    ("circulant C7{1,2} (normal)", directed_cayley_adjacency(7, {1, 2}),
     _unit([1, -1, 0.5, -0.5, 0.3, -0.2, -0.1])),
    ("weighted ring+chord (SC, non-normal)",
     np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
              dtype=float), _unit([1, -1, 0.5, -0.5])),
    ("star-in cycle-out (SC, non-normal)",
     np.array([[0, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
              dtype=float), _unit([1, -1, 0.5, -0.5])),
]


def main() -> int:
    print("N05 transient U2/U6: no per-node-energy transient; ambient >1 = ||Q||")
    header = (f"  {'graph':<38} {'||Q||':>6} {'symE':>6} {'peak':>6} "
              f"{'kreiss':>7} {'ambient':>8} {'J<=b':>5} {'noAmp':>6}")
    print(header)
    all_no_amp = True
    all_artifact = True
    all_bounds = True
    for label, w, x in CASES:
        c = certify_transient_u2(w, x)
        all_no_amp &= c.no_transient_amplification
        all_bounds &= c.bounds_hold
        # ambient gain is the oblique ||Q|| factor, not dynamics
        if abs(c.ambient_oblique_gain - c.consensus_projection_norm) > 1e-3:
            all_artifact = False
        print(f"  {label:<38} {c.consensus_projection_norm:>6.4f} "
              f"{c.symmetric_part_min_eig:>6.3f} {c.peak_gain:>6.4f} "
              f"{c.kreiss_lower_bound:>7.4f} {c.ambient_oblique_gain:>8.4f} "
              f"{str(c.integrated_reorganization <= c.integrated_reorganization_bound):>5} "
              f"{str(c.no_transient_amplification):>6}")

    audit = CircularityAudit()  # pure spectral / semigroup dynamics
    _ = ExperimentManifest(
        claim_id="NT-P09d",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(len(w) for _, w, _ in CASES)),
        controls=("normal_unit_gain", "euclidean_pernode_contraction",
                  "ambient_equals_Q_norm", "kreiss_le_peak", "u2_finite",
                  "u6_confined"),
        artifacts=(),
    )
    print()
    print(f"  peak_gain == 1 (no per-node transient) : {all_no_amp}")
    print(f"  ambient gain == ||Q|| (oblique artifact): {all_artifact}")
    print(f"  U2/U6 inequality certificates hold      : {all_bounds}")
    print(f"  per-node contraction  : {ClaimStatus.DERIVED.value} form; "
          f"PSD-on-subspace {ClaimStatus.CONJECTURAL.value} (2e5 + in-hub)")
    print(f"  canonical U2 metric   : {ClaimStatus.CONJECTURAL.value} / OPEN "
          "(NT-P09d; U2/U6 unmodified)")
    print(f"  circularity           : {audit.verdict.value}")
    ok = all_no_amp and all_artifact and all_bounds
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
