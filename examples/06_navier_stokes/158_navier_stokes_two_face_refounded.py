"""TNFR Example 158 — Navier–Stokes, re-founded on the two-face reading.

The 3D incompressible Navier–Stokes program, re-founded on the current paradigm
(the previous diffusive-face enstrophy-budget program was retired). The honest
thesis, measured here:

  M1  Faithful flow. A lean pseudo-spectral Taylor–Green integrator: energy
      decays, incompressibility holds to round-off, and the genuine 3D
      vortex-stretching production is nonzero.
  M2  The two-face reading. Incompressible NS is FIRST ORDER, so its LINEAR part
      is the diffusive (over-damped) projection of the substrate wave: mapping
      viscosity to the damping γ = 1/ν, the engine's own certificate
      (verify_diffusive_face) is VALID and recovers ν_f = ν. Linear NS carries
      NO oscillatory content — unlike EEG (under-damped). The conservative /
      inertial character is entirely the NONLINEAR stretching source.
  M3  The blow-up frontier. Therefore blow-up is a purely nonlinear K_φ cascade.
      Measured across a Reynolds sweep at matched structural time τ_str = ν·t:
      every run saturates at fixed Re (the diffusive face regularises), and the
      peak enstrophy debt GROWS with Re. Whether it stays bounded as Re → ∞ is
      exactly Clay.

HONEST SCOPE: closes NOTHING. The linear diffusive face is regular by
construction; the nonlinear cascade bound as Re → ∞ (Clay) stays OPEN. This is
the NS twin of the re-founded Riemann coherence budget: a per-instance bound that
is finite at every finite parameter, with the uniform bound over the limiting
parameter left open.

Run:
    python examples/06_navier_stokes/158_navier_stokes_two_face_refounded.py

Theory: theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md.
Status: RESEARCH.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.navier_stokes import (  # noqa: E402
    TNFRNavierStokes,
    face_of_flow,
    measure_cascade_frontier,
    verify_diffusive_face,
)


def main() -> None:
    print("=" * 72)
    print("NAVIER-STOKES, RE-FOUNDED -- the honest two-face reading")
    print("=" * 72)

    # M1 -- faithful flow
    print("\nM1 -- faithful pseudo-spectral Taylor-Green (n=16, nu=0.02):")
    flow = TNFRNavierStokes(16, 0.02, 1.0)
    e0, w0 = flow.energy(), flow.enstrophy()
    for _ in range(40):
        flow.step(0.01)
    print(f"   energy {e0:.4f} -> {flow.energy():.4f} (decays); "
          f"enstrophy {w0:.3f} -> {flow.enstrophy():.3f}")
    print(f"   incompressibility: max|div u| = {flow.divergence_sup():.1e}")
    print(f"   3D vortex-stretching production = "
          f"{flow.stretching_production():.4f} (zero in 2D)")

    # M2 -- the two-face reading
    print("\nM2 -- which face is linear NS on? (gamma = 1/nu)")
    for nu in (0.05, 0.01):
        fo = face_of_flow(nu)
        cert = verify_diffusive_face(nu)
        print(f"   nu={nu}: {fo['face']} (gamma={fo['gamma']:.0f}); "
              f"verify_diffusive_face VALID={cert.is_valid_projection}, "
              f"recovers nu_f={cert.nu_f_effective:.4f}")
    print("   => linear NS is diffusive (over-damped); the conservative /")
    print("      inertial content is the NONLINEAR stretching, not a wave.")

    # M3 -- the nonlinear K_phi cascade frontier
    print("\nM3 -- the nonlinear cascade frontier (Reynolds sweep, tau_str):")
    cert = measure_cascade_frontier(
        n=16, viscosities=(0.05, 0.02, 0.01), tau_target=0.15, dt=0.01
    )
    for r, d, s in zip(cert.reynolds, cert.peak_debt, cert.peak_stretching):
        print(f"   Re={r:6.0f}: peak enstrophy debt={d:.3f}, "
              f"peak stretching={s:.3f}")
    print(f"   {cert.summary()}")

    print("\n" + "=" * 72)
    print(
        "VERDICT: linear NS is diffusive (measured, VALID certificate); blow-up\n"
        "is a purely NONLINEAR K_phi cascade whose peak debt grows with Re but\n"
        "is bounded at every fixed Re. The Re -> inf bound = Clay, OPEN.\n"
        "Closes nothing."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
