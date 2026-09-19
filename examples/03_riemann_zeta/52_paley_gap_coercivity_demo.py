"""P25 Paley-gap coercivity demo.

Reproduces the Paley-gap philosophy of Mart\u00ednez Gamo,
*Spectral note: Paley gap via lambda_2 (residue circulants)*,
Zenodo 10.5281/zenodo.17665853 v2 (November 2025), at the
TNFR-Riemann coercivity level.

Computes three Paley-gap quantities over a real sigma-grid for two
prime-ladder Hamiltonian bundles (decoupled and weakly coupled):

    g_P12(sigma)   = |Z_P12(sigma)   - Z_classical(sigma)|
    g_P14(sigma)   = |Z_P14(sigma)   - Z_classical(sigma)|
    g_cross(sigma) = |Z_P14(sigma)   - Z_P12(sigma)|

At coupling = 0 the cross gap collapses to machine precision for
every sigma (Paley-style identity between the closed-form P12
construction and the self-adjoint P14 operator realisation).  At
coupling > 0 the cross gap quantifies coupling-induced structural
deformation of the prime-ladder spectrum.

This is a consistency diagnostic, not an RH proof.  The Zenodo
source note itself disclaims primality proof status, and P25
inherits the same scope at the coercivity level.
"""

from __future__ import annotations

import sys

import numpy as np

from tnfr.riemann.paley_gap_coercivity import sweep_paley_gap
from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_hamiltonian


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _ensure_utf8_stdout()

    n_primes = 18
    max_power = 5
    sigmas = np.linspace(1.5, 4.0, 11)
    n_max_classical = 50_000

    print("=" * 78)
    print("P25 PALEY-GAP COERCIVITY DEMO")
    print("(after Mart\u00ednez Gamo, Zenodo 17665853 v2, 2025)")
    print("=" * 78)
    print(f"n_primes        = {n_primes}")
    print(f"max_power       = {max_power}")
    print(
        f"sigma grid      = [{sigmas[0]:.3f}, {sigmas[-1]:.3f}],"
        f" {sigmas.size} points"
    )
    print(f"n_max_classical = {n_max_classical}")
    print()

    # ------------------------------------------------------------------
    # Bundle A: decoupled prime ladders (Paley-style identity expected)
    # ------------------------------------------------------------------
    bundle_decoupled = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    sweep_decoupled = sweep_paley_gap(
        bundle_decoupled,
        sigmas,
        n_max_classical=n_max_classical,
    )
    print("[A] Decoupled bundle (coupling = 0)")
    print("    " + sweep_decoupled.summary())
    print()
    print(f"    {'sigma':>8s}  {'g_P12':>14s}  " f"{'g_P14':>14s}  {'g_cross':>14s}")
    for s, g12, g14, gc in zip(
        sweep_decoupled.sigmas,
        sweep_decoupled.g_p12,
        sweep_decoupled.g_p14,
        sweep_decoupled.g_cross,
    ):
        print(f"    {s:>8.3f}  {g12:>14.6e}  {g14:>14.6e}  {gc:>14.6e}")
    print()

    # ------------------------------------------------------------------
    # Bundle B: weakly coupled prime ladders (cross gap should appear)
    # ------------------------------------------------------------------
    coupling_b = 1.0e-2
    bundle_coupled = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=coupling_b,
    )
    sweep_coupled = sweep_paley_gap(
        bundle_coupled,
        sigmas,
        n_max_classical=n_max_classical,
    )
    print(f"[B] Weakly coupled bundle (coupling = {coupling_b:.1e})")
    print("    " + sweep_coupled.summary())
    print()
    print(f"    {'sigma':>8s}  {'g_P12':>14s}  " f"{'g_P14':>14s}  {'g_cross':>14s}")
    for s, g12, g14, gc in zip(
        sweep_coupled.sigmas,
        sweep_coupled.g_p12,
        sweep_coupled.g_p14,
        sweep_coupled.g_cross,
    ):
        print(f"    {s:>8.3f}  {g12:>14.6e}  {g14:>14.6e}  {gc:>14.6e}")
    print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("-" * 78)
    print("INTERPRETATION (Paley-gap philosophy applied to coercivity)")
    print("-" * 78)
    print(
        "  - g_P12 is the truncation gap of the P12 prime-ladder vs the\n"
        "    classical sum; it shrinks as (n_primes, max_power,\n"
        "    n_max_classical) grow.\n"
        "  - g_P14 is the same fidelity check applied to the P14\n"
        "    self-adjoint Hamiltonian trace.\n"
        "  - g_cross = |Z_P14 - Z_P12| is the structural Paley-gap: at\n"
        "    coupling = 0 it must vanish to machine precision (Paley-style\n"
        "    algebraic identity), and at coupling > 0 it measures pure\n"
        "    coupling-induced deformation, free of classical truncation\n"
        "    noise.\n"
        "  - Honest scope: this is a consistency diagnostic, not a proof\n"
        "    of uniform coercivity or of the Riemann Hypothesis. Gap G4\n"
        "    (RH localisation on Re(s) = 1/2) is not addressed by P25."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
