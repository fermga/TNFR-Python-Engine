"""Finite character-twisted prime-ladder correction of smooth targets.

For supplied real characters, a truncated classical oscillatory approximation
and a chosen damping grid define candidate corrections. Reference L-function
zero ordinates are comparison inputs. Better or worse finite distances say
nothing conclusive about completeness of the native operator catalog, nor do
they prove GRH. See theory/TNFR_RIEMANN_RESEARCH_NOTES.md.
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
    print("Honest scope: experimental research diagnostic at the" " L-track level.")
    print(
        "Improvement concerns this finite candidate and grid."
        " Failure does not require a new canonical operator."
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
        run(chi, n_targets=n_targets, n_primes=n_primes, max_power=max_power)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
