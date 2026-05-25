"""P43 chi-twisted Paley-gap coercivity demo.

L-track analogue of the zeta-track P25 demo
(examples/52_paley_gap_coercivity_demo.py).  Transports the
Paley-gap philosophy of Mart\u00ednez Gamo, *Spectral note: Paley
gap via lambda_2 (residue circulants)*, Zenodo
10.5281/zenodo.17665853 v2 (November 2025), to the chi-twisted
TNFR-Riemann program for primitive real Dirichlet characters
chi_3, chi_4, chi_5.

Computes three chi-twisted Paley-gap quantities over a real
sigma-grid:

    g_P32(sigma)   = |Z_P32(sigma, chi)  - Z_classical(sigma, chi)|
    g_P34(sigma)   = |Z_P34(sigma, chi)  - Z_classical(sigma, chi)|
    g_cross(sigma) = |Z_P34(sigma, chi)  - Z_P32(sigma, chi)|

Two bundles per character are exercised:

    [A] Decoupled (coupling = 0) -- Paley-style identity expected:
        g_cross collapses to machine precision for every sigma and
        every chi.  This is a regression test, not a discovery.

    [B] Weakly coupled (coupling = 1e-2) -- coupling-induced cross
        gap appears.  This is the diagnostic signal: it quantifies
        how strongly inter-ladder coupling deforms the chi-twisted
        prime-ladder identity, free of classical-truncation noise.

HONEST SCOPE: this is a consistency diagnostic, not a proof of
uniform coercivity and not a proof of the Generalised Riemann
Hypothesis for L(s, chi).  The Zenodo source note itself disclaims
primality proof status; P43 inherits the same scope at the L-track
coercivity level.  Gap G4-chi (GRH localisation on Re(s) = 1/2 for
L(s, chi)) is NOT addressed by P43.
"""

from __future__ import annotations

import sys

import numpy as np

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_paley_gap_coercivity import sweep_twisted_paley_gap
from tnfr.riemann.twisted_prime_ladder_hamiltonian import (
    build_twisted_prime_ladder_hamiltonian,
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _print_sweep_table(sweep) -> None:
    print(
        f"    {'sigma':>8s}  {'g_P32':>14s}  "
        f"{'g_P34':>14s}  {'g_cross':>14s}"
    )
    for s, g32, g34, gc in zip(
        sweep.sigmas, sweep.g_p32, sweep.g_p34, sweep.g_cross,
    ):
        print(f"    {s:>8.3f}  {g32:>14.6e}  {g34:>14.6e}  {gc:>14.6e}")


def main() -> int:
    _ensure_utf8_stdout()

    n_primes = 18
    max_power = 5
    sigmas = np.linspace(1.5, 4.0, 11)
    n_max_classical = 50_000
    coupling_b = 1.0e-2

    characters = [
        ("chi_3", real_character_mod_3()),
        ("chi_4", real_character_mod_4()),
        ("chi_5", real_character_mod_5()),
    ]

    print("=" * 78)
    print("P43 CHI-TWISTED PALEY-GAP COERCIVITY DEMO")
    print("(L-track analogue of P25; after Mart\u00ednez Gamo, "
          "Zenodo 17665853 v2, 2025)")
    print("=" * 78)
    print(f"n_primes        = {n_primes}")
    print(f"max_power       = {max_power}")
    print(
        f"sigma grid      = [{sigmas[0]:.3f}, {sigmas[-1]:.3f}],"
        f" {sigmas.size} points"
    )
    print(f"n_max_classical = {n_max_classical}")
    print(f"coupling (B)    = {coupling_b:.1e}")
    print()
    print(
        "HONEST SCOPE: consistency diagnostic only.  Does NOT prove\n"
        "GRH for any L(s, chi); does NOT advance G4 = RH."
    )
    print()

    summary_rows = []

    for name, chi in characters:
        print("=" * 78)
        print(f"CHARACTER {name} (modulus q = {chi.modulus})")
        print("=" * 78)

        # --------------------------------------------------------------
        # [A] Decoupled bundle: Paley-style identity expected
        # --------------------------------------------------------------
        bundle_a = build_twisted_prime_ladder_hamiltonian(
            chi=chi,
            n_primes=n_primes,
            max_power=max_power,
            coupling=0.0,
        )
        sweep_a = sweep_twisted_paley_gap(
            bundle_a, chi, sigmas, n_max_classical=n_max_classical,
        )
        print(f"[A] {name} decoupled bundle (coupling = 0)")
        print("    " + sweep_a.summary())
        print()
        _print_sweep_table(sweep_a)
        print()

        # --------------------------------------------------------------
        # [B] Weakly coupled bundle: cross gap should appear
        # --------------------------------------------------------------
        bundle_b = build_twisted_prime_ladder_hamiltonian(
            chi=chi,
            n_primes=n_primes,
            max_power=max_power,
            coupling=coupling_b,
        )
        sweep_b = sweep_twisted_paley_gap(
            bundle_b, chi, sigmas, n_max_classical=n_max_classical,
        )
        print(f"[B] {name} weakly coupled bundle (coupling = {coupling_b:.1e})")
        print("    " + sweep_b.summary())
        print()
        _print_sweep_table(sweep_b)
        print()

        summary_rows.append(
            (
                name,
                chi.modulus,
                sweep_a.max_g_cross,
                sweep_b.max_g_cross,
                sweep_a.max_g_p32,
                sweep_b.max_g_p32,
            )
        )

    # ------------------------------------------------------------------
    # Cross-character summary
    # ------------------------------------------------------------------
    print("=" * 78)
    print("CROSS-CHARACTER SUMMARY (worst-case Paley-gaps)")
    print("=" * 78)
    print(
        f"  {'chi':>6s}  {'q':>3s}  "
        f"{'max g_cross [A]':>18s}  {'max g_cross [B]':>18s}  "
        f"{'max g_P32 [A]':>14s}  {'max g_P32 [B]':>14s}"
    )
    for name, q, gxa, gxb, gpa, gpb in summary_rows:
        print(
            f"  {name:>6s}  {q:>3d}  "
            f"{gxa:>18.3e}  {gxb:>18.3e}  "
            f"{gpa:>14.3e}  {gpb:>14.3e}"
        )
    print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("-" * 78)
    print("INTERPRETATION (chi-twisted Paley-gap philosophy)")
    print("-" * 78)
    print(
        "  - g_P32 is the truncation gap of the P32 chi-twisted\n"
        "    prime-ladder closed form vs the classical chi-twisted sum;\n"
        "    it shrinks as (n_primes, max_power, n_max_classical) grow.\n"
        "  - g_P34 is the same fidelity check applied to the P34\n"
        "    chi-twisted spectral trace.\n"
        "  - g_cross = |Z_P34 - Z_P32| is the structural Paley-gap on\n"
        "    the L-track: at coupling = 0 it must vanish to machine\n"
        "    precision (Paley-style algebraic identity between P32 and\n"
        "    P34), and at coupling > 0 it measures pure coupling-induced\n"
        "    deformation, free of classical-truncation noise.\n"
        "  - Honest scope: consistency diagnostic only.  Does NOT prove\n"
        "    GRH for any L(s, chi); does NOT advance G4 = RH.  The\n"
        "    Zenodo source disclaims primality proof status; P43\n"
        "    inherits the same scope at the L-track coercivity level."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
