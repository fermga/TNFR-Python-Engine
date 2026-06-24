"""TNFR-Navier-Stokes N2 demo: Taylor-Green vortex decay on a torus graph.

Validates that the discrete TNFR-NS operator on a periodic ``n x n`` grid
reproduces the analytical kinetic-energy decay rate of the 2D Taylor-Green
vortex,

    E(t) = pi^2 * U_0^2 * exp(-4 nu t),

to within finite-difference truncation error.

Success criterion
-----------------
The measured exponential decay rate ``lambda_meas`` (fitted to ``log E(t)``)
must agree with the analytical rate ``4 nu`` to within ``< 2%`` on a
``32 x 32`` grid, and the discrete divergence of the initial Taylor-Green
field must vanish to ``< 1e-10``.

Honest scope
------------
N2 covers only the viscous (linear) half of the operator. Advection is OFF.
This demo does NOT bear on the Clay Millennium statement; it certifies the
diffusion side of the discrete TNFR-NS dynamics so that N3 (energy
inequality) and N4 (discrete BKM criterion) can build on a validated base.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from tnfr.navier_stokes import TNFRNavierStokesOperator, build_torus_graph


def _fit_decay_rate(times: np.ndarray, energy: np.ndarray) -> float:
    """Least-squares fit of log E(t) = log E(0) - lambda * t."""
    log_e = np.log(energy)
    slope, _intercept = np.polyfit(times, log_e, 1)
    return -float(slope)


def main() -> int:
    resolution = 32
    viscosity = 0.05
    dt = 0.02
    steps = 50
    amplitude = 1.0

    print("=" * 72)
    print("TNFR-Navier-Stokes N2 demo: Taylor-Green vortex on 2D torus graph")
    print("=" * 72)
    print(f"Grid resolution    : {resolution} x {resolution} (periodic)")
    print(f"Spacing h          : {2 * math.pi / resolution:.6f}")
    print(f"Viscosity nu       : {viscosity}")
    print(f"Time step dt       : {dt}")
    print(f"Number of steps    : {steps}")
    print(f"Amplitude U_0      : {amplitude}")
    print()

    G = build_torus_graph(resolution)
    op = TNFRNavierStokesOperator(G, viscosity=viscosity, dimension=2)
    op.set_taylor_green(amplitude=amplitude)

    div_initial = op.divergence_residual()
    e0_measured = op.kinetic_energy()
    e0_analytic = (math.pi**2) * amplitude**2

    print("Initial diagnostics")
    print("-" * 72)
    print(f"Discrete divergence (L2)   : {div_initial:.3e}")
    print(f"Discrete E(0)              : {e0_measured:.6f}")
    print(f"Analytical E(0) = pi^2 U^2 : {e0_analytic:.6f}")
    print(
        f"Relative error             : {abs(e0_measured - e0_analytic) / e0_analytic:.3e}"
    )
    print()

    history = op.run(dt=dt, steps=steps)
    times = np.arange(steps + 1) * dt
    reference = op.taylor_green_reference(times, amplitude=amplitude)

    lambda_meas = _fit_decay_rate(times, history)
    lambda_ref = 4.0 * viscosity
    rate_error = abs(lambda_meas - lambda_ref) / lambda_ref

    print("Energy decay (selected steps)")
    print("-" * 72)
    print(f"{'t':>10s} {'E_discrete':>16s} {'E_analytical':>16s} {'rel.err.':>12s}")
    sample_idx = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    for k in sample_idx:
        rel = abs(history[k] - reference[k]) / reference[k]
        print(f"{times[k]:10.4f} {history[k]:16.8f} {reference[k]:16.8f} {rel:12.3e}")
    print()

    print("Decay-rate verification")
    print("-" * 72)
    print(f"Analytical rate  4 * nu        : {lambda_ref:.6f}")
    print(f"Measured rate    -d ln E / dt  : {lambda_meas:.6f}")
    print(f"Relative error                 : {rate_error:.3e}")
    print()

    ok_div = div_initial < 1e-10
    ok_rate = rate_error < 2e-2
    ok_initial_energy = abs(e0_measured - e0_analytic) / e0_analytic < 1e-2

    print("Success criteria (N2)")
    print("-" * 72)
    print(f"  divergence(t=0) < 1e-10          : {'PASS' if ok_div else 'FAIL'}")
    print(
        f"  |E_discrete(0) - E_analytic(0)|  : {'PASS' if ok_initial_energy else 'FAIL'}"
    )
    print(f"  |lambda_meas - 4 nu| / (4 nu)    : {'PASS' if ok_rate else 'FAIL'}")
    print()

    if ok_div and ok_rate and ok_initial_energy:
        print("[N2 PASS] Discrete TNFR-NS reproduces the analytical Taylor-Green")
        print("          decay rate on the 2D torus. Viscous half certified.")
        print()
        print("Next: N3 (discrete Leray energy inequality with advection on).")
        return 0

    print("[N2 FAIL] Numerical validation did not meet the success criteria.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
