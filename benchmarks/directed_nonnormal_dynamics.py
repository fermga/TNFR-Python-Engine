#!/usr/bin/env python3
"""R9 directed non-normal dynamics benchmark.

Contrasts a directed circulant residue digraph (normal: unit transient gain) with
general directed graphs (non-normal: transient amplification despite a stable
spectrum). Certifies each with normality, spectral abscissa, transient gain, the
pseudospectral / Kreiss lower bound, and the SciPy-gated Schur residual.

Honest scope: a stable spectrum does NOT preclude transient growth for non-normal
operators. The reading r_c = nu_f * lambda_2 is ASYMPTOTIC only; no generalized U2
bound is claimed until the transient contribution to the integral convergence is
derived (NT-P09 OPEN).
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import (  # noqa: E402
    certify_directed_dynamics,
    directed_cayley_adjacency,
)
from tnfr.physics.spectral_projectors import scipy_available  # noqa: E402
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

CASES = [
    ("circulant C7{1,2} (R2 residue digraph)",
     directed_cayley_adjacency(7, {1, 2})),
    ("circulant C8{1,3}", directed_cayley_adjacency(8, {1, 3})),
    ("feed-forward chain + self-loops",
     np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 1]],
              dtype=float)),
    ("asymmetric directed graph",
     np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]],
              dtype=float)),
]


def main() -> int:
    print("R9 directed non-normal dynamics (generator -L_rw)")
    print(f"  SciPy available (Schur gated): {scipy_available()}")
    print()
    header = (f"  {'graph':<38} {'normal':>7} {'abscissa':>9} "
              f"{'t_gain':>7} {'kreiss':>7} {'amplif':>7}")
    print(header)
    normal_ok = True
    nonnormal_ok = True
    for label, adjacency in CASES:
        c = certify_directed_dynamics(adjacency)
        print(f"  {label:<38} {str(c.normal):>7} {c.abscissa:>9.4f} "
              f"{c.transient_gain:>7.4f} {c.pseudospectral_bound:>7.4f} "
              f"{str(c.has_transient_amplification):>7}")
        if c.normal:
            normal_ok &= (not c.has_transient_amplification)
        else:
            # non-normal here are stable yet amplify, and Kreiss <= gain
            nonnormal_ok &= (
                c.asymptotically_stable
                and c.has_transient_amplification
                and c.pseudospectral_bound <= c.transient_gain + 1e-6
            )

    audit = CircularityAudit()  # pure spectral dynamics; no arithmetic target
    manifest = ExperimentManifest(
        claim_id="NT-P09",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__,
                  "scipy": "available" if scipy_available() else "absent"},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(8),
        controls=("normal_circulant", "kreiss_lower_bound",
                  "schur_residual", "no_eigh_on_nonsymmetric"),
        artifacts=(),
    )
    print()
    print(f"  normal circulants have unit transient gain : {normal_ok}")
    print(f"  non-normal stable yet amplify (Kreiss<=gain): {nonnormal_ok}")
    print(f"  transient-amplification claim : {ClaimStatus.DERIVED.value} "
          f"(Kreiss) + measured")
    print(f"  U2 generalization (NT-P09)    : {ClaimStatus.CONJECTURAL.value} "
          "/ OPEN (asymptotic scope only)")
    print(f"  circularity                   : {audit.verdict.value}")
    print(f"  manifest controls             : {len(manifest.controls)}")
    return 0 if (normal_ok and nonnormal_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
