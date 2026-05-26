r"""Example 76 -- chi-twisted prime-ladder oscillatory correction
(P49).

L-track analogue of Example 58 (P31 oscillatory_correction_demo).
For each primitive real Dirichlet character chi_3, chi_4, chi_5,
reconstructs the chi-twisted oscillatory remainder S_chi(T) =
pi^{-1} arg L(1/2 + iT, chi) from the canonical P34 chi-twisted
prime-ladder spectrum {(k log p, chi(p)^k log p)} via the chi-twisted
Riemann-von Mangoldt template, applies the Newton step

    gamma_n^(chi)_corr = tilde gamma_n^(chi) - d * S_chi(tilde
        gamma_n^(chi)) / bar N'_chi(tilde gamma_n^(chi))

on the canonical P46 chi-twisted smooth targets, and measures the
residual W_1 gap to the true L(s, chi) zeros honestly across the
damping sweep.

Honest scope (AGENTS.md sec. 13.2 + section 13septies T-HP):

    * Uses ONLY canonical chi-twisted TNFR ingredients (P34 spectrum,
      P46 smooth density, P34 Hilbert-Polya scaffold for the true
      chi-zeros reference).  No mpmath chi-zeros enter the
      construction side of S_chi(T).
    * Positive improvement is branch B1 evidence at the L-track
      level: the canonical chi-twisted prime-ladder spectrum suffices
      to reduce the S_chi(T) residual for the chosen character.
    * Negative or near-zero improvement corroborates branch B2 at
      the L-track level (a genuinely new canonical operator
      required).
    * Closes the final zeta/L attack-surface parity item: with P49,
      every canonical zeta-track operator from P12 through P31 has
      a matching chi-twisted L-track counterpart.
    * Does NOT close G4 = RH.  Does NOT prove GRH for any L(s, chi).
    * Sub-problems (2) (canonicity from the nodal equation) and
      (3) (positivity coincidence with the chi-twisted Weil form)
      remain open.

Usage::

    $env:PYTHONPATH=(Resolve-Path ./src).Path
    $env:PYTHONIOENCODING="utf-8"
    & ./.venv312/Scripts/python.exe examples/76_twisted_oscillatory_correction_demo.py
"""

from __future__ import annotations

import sys

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_oscillatory_correction import (
    compute_twisted_oscillatory_correction_certificate,
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def run(chi, n_targets: int, n_primes: int, max_power: int) -> None:
    cert = compute_twisted_oscillatory_correction_certificate(
        chi,
        n_targets,
        n_primes=n_primes,
        max_power=max_power,
    )
    print()
    print(
        f"=== chi = {chi.name} (mod {chi.modulus}), "
        f"N = {n_targets}, primes = {n_primes}, K = {max_power} ==="
    )
    print(cert.summary())
    print()
    print("  damping sweep (damping, W_1):")
    for d, w1 in cert.damping_sweep:
        marker = "  <-- best" if d == cert.best_damping else ""
        if w1 == float("inf"):
            print(f"    d={d:.2f}  W_1=overflow{marker}")
        else:
            print(f"    d={d:.2f}  W_1={w1:.4e}{marker}")


def main() -> int:
    _ensure_utf8_stdout()

    print(
        "P49 -- chi-twisted prime-ladder oscillatory correction"
        " (L-track lift of P31)"
    )
    print(
        "Honest scope: experimental research diagnostic at the"
        " L-track level."
    )
    print(
        "Positive improvement => branch B1 evidence (L-track)."
        "  Negative => branch B2 corroboration."
    )
    print("Neither outcome closes G4 = RH or GRH for any L(s, chi).")

    # Scaled-down parameters relative to ZETA-track demo 58 to
    # accommodate the additional cost of chi-zero enumeration.
    n_targets = 10
    n_primes = 80
    max_power = 5

    for chi in (
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ):
        run(chi, n_targets=n_targets, n_primes=n_primes,
            max_power=max_power)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
