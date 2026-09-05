#!/usr/bin/env python3
"""N13 heterogeneous-vf benchmark (R9b): where the clock-change theorem stops.

The scalar-frequency structural-time theorem (N04) integrates xdot = -vf(t) L x
to the clock change x(t) = e^{-s(t)L} x0 because all generators vf(tau)L commute.
With a heterogeneous nodal frequency D_vf(t) = diag(vf_1,...,vf_n) the transport
xdot = -D_vf(t) L x has non-commuting generators, so there is NO clock change:
e^{-sbar(t)L} x0 is not the solution. This benchmark contrasts a common vs a
heterogeneous schedule and reports the stability facts.

Honest result: the scalar-time ansatz is exact (~0) for a common schedule and
FAILS (large residual) for a heterogeneous one -- N04 does not extend. A frozen
positive D_vf keeps -D_vf L stable (spectral abscissa <= 0); uniform time-varying
stability is only measured on the tested schedules, a general bound is OPEN
(NT-P09 heterogeneous). U2/U6 unmodified. No complexity / crypto / Millennium.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.physics.directed_diffusion import directed_cayley_adjacency  # noqa: E402
from tnfr.physics.heterogeneous_vf import certify_heterogeneous_vf  # noqa: E402
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
]


def main() -> int:
    print("N13 heterogeneous vf: the scalar clock-change theorem does not extend")
    tg = np.linspace(0.0, 10.0, 600)
    header = (f"  {'graph':<38} {'comm_sc':>8} {'comm_het':>9} {'res_sc':>8} "
              f"{'res_het':>8} {'extend':>7} {'absc':>7} {'gain':>6}")
    print(header)
    all_boundary = True
    all_stable = True
    for label, w, x in CASES:
        c = certify_heterogeneous_vf(w, x, tg)
        boundary = (c.commutator_scalar < 1e-9
                    and c.commutator_heterogeneous > 1e-3
                    and c.scalar_time_residual_common < 1e-3
                    and not c.scalar_time_theorem_extends)
        all_boundary &= boundary
        all_stable &= c.fixed_generator_stable
        print(f"  {label:<38} {c.commutator_scalar:>8.1e} "
              f"{c.commutator_heterogeneous:>9.4f} "
              f"{c.scalar_time_residual_common:>8.4f} "
              f"{c.scalar_time_residual_heterogeneous:>8.4f} "
              f"{str(c.scalar_time_theorem_extends):>7} "
              f"{c.fixed_generator_abscissa:>7.3f} "
              f"{c.heterogeneity_transient_gain:>6.3f}")

    audit = CircularityAudit()  # pure spectral / semigroup dynamics
    _ = ExperimentManifest(
        claim_id="NT-P09e",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=None,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(7),
        controls=("scalar_commutes", "heterogeneous_no_commute",
                  "scalar_time_fails_heterogeneous", "fixed_generator_stable"),
        artifacts=(),
    )
    print()
    print(f"  scalar-time theorem fails (het) : {all_boundary}")
    print(f"  frozen D_vf stable (all)        : {all_stable}")
    print(f"  scalar-time does NOT extend     : {ClaimStatus.DERIVED.value} "
          "(commutator != 0) + MEASURED")
    print(f"  uniform time-varying stability  : {ClaimStatus.CONJECTURAL.value}"
          " / OPEN (NT-P09 heterogeneous)")
    print(f"  circularity                     : {audit.verdict.value}")
    ok = all_boundary and all_stable
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
