#!/usr/bin/env python3
"""N03 directed U2-metric layer benchmark.

Shows that the R9 Euclidean transient gain is metric-dependent: for a
strongly-connected non-normal digraph the Euclidean gain exceeds 1, but in the
stationary-weighted norm L2(pi) the diffusion semigroup is a contraction (gain
<= 1, by Jensen). Also reports the two U2 integral readings (signed net
displacement vs total structural variation; net <= total).

Honest scope: this does NOT decide U2. Which norm / integral is the canonical U2
quantity stays OPEN (NT-P09b/c); AGENTS.md U2 is not modified until the gate is
met. No complexity/crypto/Millennium claim.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import (  # noqa: E402
    NormKind,
    directed_cayley_adjacency,
    directed_rw_laplacian,
    is_stationary_contraction,
    stationary_transient_gain,
    transient_gain_in_norm,
    u2_integral_readings,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

CASES = [
    ("circulant C7{1,2} (normal)", directed_cayley_adjacency(7, {1, 2})),
    ("asymmetric 4-cycle (SC, non-normal)",
     np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]],
              dtype=float)),
    ("weighted ring+chord (SC, non-normal)",
     np.array([[0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2], [2, 0, 1, 0]],
              dtype=float)),
]


def main() -> int:
    print("N03 directed U2 metrics: Euclidean vs stationary L2(pi) transient gain")
    print(f"  {'graph':<38} {'eucl':>7} {'stat':>7} {'contraction':>12}")
    all_contract = True
    metric_disagrees = False
    for label, W in CASES:
        L = directed_rw_laplacian(W)
        eucl = transient_gain_in_norm(-L, kind=NormKind.EUCLIDEAN)
        stat = stationary_transient_gain(W)
        contract = is_stationary_contraction(W)
        all_contract &= contract
        if eucl > 1.0 + 1e-6 and stat <= 1.0 + 1e-6:
            metric_disagrees = True
        print(f"  {label:<38} {eucl:>7.4f} {stat:>7.4f} {str(contract):>12}")

    print()
    print("  U2 integral readings (signed net vs total variation):")
    W = CASES[2][1]
    x0 = np.array([1.0, -1.0, 0.5, -0.5])
    net_le_total = True
    for kind in (NormKind.EUCLIDEAN, NormKind.STATIONARY):
        r = u2_integral_readings(W, x0, kind=kind)
        net_le_total &= r.net_le_total
        print(f"    {r.norm_kind:11s} net={r.net:.4f} total={r.total:.4f} "
              f"net<=total={r.net_le_total}")

    audit = CircularityAudit()  # pure spectral/metric dynamics
    manifest = ExperimentManifest(
        claim_id="NT-P09b",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(len(W) for _, W in CASES)),
        controls=("normal_circulant", "euclidean_vs_stationary",
                  "net_vs_total_integral"),
        artifacts=(),
    )
    print()
    print(f"  stationary contraction (all): {all_contract}")
    print(f"  Euclidean/stationary metrics disagree (non-normal): "
          f"{metric_disagrees}")
    print(f"  net <= total (all): {net_le_total}")
    print(f"  stationary contraction : {ClaimStatus.DERIVED.value} + measured "
          "(Jensen)")
    print(f"  canonical U2 metric    : {ClaimStatus.CONJECTURAL.value} / OPEN "
          "(NT-P09b; U2 not modified)")
    print(f"  circularity            : {audit.verdict.value}")
    ok = all_contract and metric_disagrees and net_le_total
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
