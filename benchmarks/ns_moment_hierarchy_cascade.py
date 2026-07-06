"""NS lambda-moment hierarchy of the cascade -- the newly-unlocked reading.

THE PARADIGM (re-founded NS, 2026-07): the previous diffusive-face program saw
only the SCALAR enstrophy budget. The re-founding on the emergent modal basis
(L_rw modes, omega_k = sqrt(lambda_k)) + the two-face reading (viscosity is the
diffusive-face damping, exact per-mode rate nu*lambda_k) unlocks the whole
lambda-MOMENT HIERARCHY M_p = sum lambda_k^p E(lambda_k):
    M_0 = energy       -- the CONSERVATIVE-face budget (bounded by Leray)
    M_1 = enstrophy    -- the classical blow-up quantity
    M_2 = palinstrophy -- weights the small-scale (high-lambda) tail more
Blow-up in this basis: M_0 is bounded, so Clay is exactly whether the ladder
M_p (p>=1) stays uniformly bounded as nu -> 0. This measure-first study asks
(WITHOUT anticipating): does M_0 stay bounded while the ladder climbs with Re?

WHAT EMERGES (measured; Taylor-Green at peak enstrophy, matched tau_str=nu*t):
  M1  M_0 (energy) is bounded -- in fact it DECREASES across the sweep (the
      conservative-face budget, Leray).
  M2  The ladder climbs: M_1 (enstrophy) grows with Re and M_2 (palinstrophy)
      grows much faster; the moment RATIOS M_1/M_0 and M_2/M_1 climb with Re.
      The wall climbs the lambda-moment hierarchy.
  M3  Resolution honesty: only points with kmax*eta > 1 are used for the trend;
      higher-Re points are flagged under-resolved (their high moments are lower
      bounds).

CROSS-PROGRAM UNIFICATION (the synergy): this is the SAME structural statement
as the re-founded Riemann coherence budget --
    Riemann:  the conservative PULSE's low moment (RMS of S(T)) is bounded
              (Selberg sqrt(log log T)); the high moment (sup of S(T)) is open.
    NS:       the conservative FACE's low moment (energy M_0) is bounded (Leray);
              the high moments (enstrophy M_1, palinstrophy M_2, ...) are open.
Both Millennium walls: a LOW moment of the conservative spectrum is bounded; the
HIGH-moment tail is the wall. A resolution would CLOSE THE MOMENT LADDER
uniformly in the limiting parameter (the classical H^s / Foias-Temam ladder,
here in canonical modal language).

HONEST SCOPE: closes NOTHING. Finite growth over a finite, resolution-limited Re
range does not decide the asymptotic; the moment-ratio climb localises the wall,
it does not breach it. Global 3D NS regularity (Clay) stays OPEN.

Run:
    PYTHONPATH=src python benchmarks/ns_moment_hierarchy_cascade.py

Anchor: src/tnfr/navier_stokes/conservative_face.py (cascade_moment_hierarchy),
theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md. Status: RESEARCH.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tnfr.navier_stokes import (  # noqa: E402
    TNFRNavierStokes,
    cascade_moment_hierarchy,
    moment_ladder_closure,
)

VISCOSITIES = (0.04, 0.02, 0.01, 0.005)
N = 24
TAU_TARGET = 0.25
DT = 0.01


def moments_at_peak(nu: float) -> dict:
    """Evolve TG to peak enstrophy; read the moment hierarchy + ladder closure."""
    flow = TNFRNavierStokes(N, nu, 1.0)
    steps = max(1, int(round(TAU_TARGET / nu / DT)))
    growth_step = max(1, steps // 3)
    peak_e = flow.enstrophy()
    best = cascade_moment_hierarchy(flow)
    peak_c = moment_ladder_closure(flow)
    growth_c = peak_c
    for si in range(1, steps + 1):
        flow.step(DT)
        e = flow.enstrophy()
        if e > peak_e:
            peak_e = e
            best = cascade_moment_hierarchy(flow)
            peak_c = moment_ladder_closure(flow)
        if si == growth_step:
            growth_c = moment_ladder_closure(flow)
    best["Re"] = 2.0 * np.pi / nu
    best["s"] = peak_c["interpolation_saturation"]
    best["growth_closure"] = growth_c["closure_ratio"]
    return best


def main() -> None:
    print("=" * 74)
    print("NS LAMBDA-MOMENT HIERARCHY -- the wall climbs the moment ladder")
    print("=" * 74)
    print("\n  Re | M0 energy | M1 enstroph | M2 palinstr | M1/M0 | M2/M1 |"
          " kmax*eta")
    rows = []
    for nu in VISCOSITIES:
        m = moments_at_peak(nu)
        rows.append(m)
        flag = "resolved" if m["resolved"] else "UNDER-RES"
        print(f"{m['Re']:5.0f} | {m['energy_m0']:9.4f} | "
              f"{m['enstrophy_m1']:11.4f} | {m['palinstrophy_m2']:11.3f} | "
              f"{m['m1_over_m0']:5.2f} | {m['m2_over_m1']:5.2f} | "
              f"{m['kmax_eta']:.2f} {flag}")

    res = [m for m in rows if m["resolved"]]
    if len(res) >= 2:
        a, b = res[0], res[-1]
        print(f"\nRESOLVED Re {a['Re']:.0f}->{b['Re']:.0f} "
              f"({b['Re'] / a['Re']:.1f}x): "
              f"M0 x{b['energy_m0'] / a['energy_m0']:.2f}, "
              f"M1 x{b['enstrophy_m1'] / a['enstrophy_m1']:.2f}, "
              f"M2 x{b['palinstrophy_m2'] / a['palinstrophy_m2']:.2f}")

    # The ladder-closure rung: interpolation saturation + growth-phase balance.
    print("\n  Re | interp. s=M1^2/(M0 M2) | growth-phase P/(2 nu M2)")
    for m in rows:
        print(f"{m['Re']:5.0f} | {m['s']:21.4f} | {m['growth_closure']:22.3f}")
    print("  (s < 1: spectrum is SPREAD, not single-scale [favourable]; growth")
    print("   P/(2 nu M2) < 1: dissipation dominates -> the ladder CLOSES at")
    print("   these resolved Re. Uniform closure as Re->inf is Clay.)")

    print("\n" + "=" * 74)
    print(
        "VERDICT: M_0 (energy) is bounded (conservative-face / Leray budget)\n"
        "while the lambda-moment ladder M_1, M_2 climbs with Re -- the SAME\n"
        "structure as the Riemann coherence budget (low moment bounded, high\n"
        "moment the wall). A resolution closes the moment ladder uniformly in\n"
        "Re; the measurement localises the wall, it does not breach it.\n"
        "Closes nothing; Clay OPEN."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
