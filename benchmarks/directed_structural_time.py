#!/usr/bin/env python3
"""N04 directed structural-time benchmark.

Shows the R9 structural-time theorem: for a scalar common frequency nu_f(t) >= 0,
the linear EPI transport x' = -nu_f(t) L x has the exact solution
x(t) = exp(-s(t) L) x0 with s(t) = int nu_f -- so nu_f is a CLOCK CHANGE, not a
mass. Consequences: (i) RK4 of the time-varying ODE matches the reparameterized
semigroup (clock-change residual ~1e-7); (ii) the total structural reorganization
is clock-invariant (change of variables); (iii) on the non-consensus subspace
Q = I - 1 pi^T the reorganization is finite, J <= M ||LQ|| ||x0|| / omega, with
M = 1 for normal graphs and M > 1 (the transient cost) for non-normal ones.

Honest scope: this is the EXACT scalar-nu_f, linear-EPI-channel form of U2's
integral convergence. It does NOT decide the canonical U2 metric, and does NOT
cover heterogeneous nodal nu_f (a diagonal D_vf(t), N13). AGENTS.md U2 is not
modified. No complexity / crypto / Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import (  # noqa: E402
    certify_structural_time,
    directed_cayley_adjacency,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

X0 = np.array([1.0, -1.0, 0.5, -0.5])
X0_7 = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.2, -0.1])

CASES = [
    ("circulant C7{1,2} (normal)", directed_cayley_adjacency(7, {1, 2}), X0_7),
    ("asymmetric 4-cycle (SC, non-normal)",
     np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]],
              dtype=float), X0),
    ("weighted ring+chord (SC, non-normal)",
     np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
              dtype=float), X0),
]


def _vf(t):
    return 1.0 + 0.5 * np.sin(t)


def main() -> int:
    print("N04 structural time: nu_f(t) as a clock change, x(t) = exp(-s(t)L)x0")
    tg = np.linspace(0.0, 12.0, 2000)
    header = (f"  {'graph':<38} {'M':>7} {'omega':>7} {'clock':>9} "
              f"{'invar':>9} {'J':>7} {'bound':>7} {'J<=b':>5}")
    print(header)
    normal_unit = True
    nonnormal_gt1 = True
    all_clock = True
    all_bound = True
    for label, W, x0 in CASES:
        c = certify_structural_time(W, x0, _vf, tg)
        is_normal = "normal)" in label and "non-normal" not in label
        if is_normal:
            normal_unit &= abs(c.sustained_gain - 1.0) < 1e-6
        else:
            nonnormal_gt1 &= c.sustained_gain > 1.0
        all_clock &= c.clock_change_residual < 1e-4
        all_bound &= c.bound_holds
        print(f"  {label:<38} {c.sustained_gain:>7.4f} "
              f"{c.nonconsensus_abscissa:>7.4f} {c.clock_change_residual:>9.1e} "
              f"{c.reorganization_invariance_residual:>9.1e} "
              f"{c.total_reorganization:>7.4f} "
              f"{c.total_reorganization_bound:>7.4f} "
              f"{str(c.bound_holds):>5}")

    audit = CircularityAudit()  # pure spectral / semigroup dynamics
    _ = ExperimentManifest(
        claim_id="NT-P09c",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(len(W) for _, W, _ in CASES)),
        controls=("normal_unit_gain", "nonnormal_transient_cost",
                  "clock_change_rk4", "reorganization_invariance",
                  "finite_reorganization_bound"),
        artifacts=(),
    )
    print()
    print(f"  normal M == 1            : {normal_unit}")
    print(f"  non-normal M > 1 (cost)  : {nonnormal_gt1}")
    print(f"  RK4 == reparam semigroup : {all_clock}")
    print(f"  J <= bound (all)         : {all_bound}")
    print(f"  clock-change theorem     : {ClaimStatus.DERIVED.value} + measured")
    print(f"  canonical U2 metric      : {ClaimStatus.CONJECTURAL.value} / OPEN "
          "(NT-P09c; U2 not modified)")
    print(f"  circularity              : {audit.verdict.value}")
    ok = normal_unit and nonnormal_gt1 and all_clock and all_bound
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
