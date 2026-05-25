r"""Example 75 -- chi-twisted admissible spectral-rescaling
operator (P48).

L-track analogue of Example 57 (P30 admissible_rescaling_demo).  For
each primitive real Dirichlet character chi_3, chi_4, chi_5,
constructs the canonical chi-twisted smooth rescaling operator
F^(chi)_smooth = U_P34 * diag(sqrt(tilde gamma_i^(chi) / lambda_i))
* U_P34^* lifting the P46 density-level closure to the operator
level, verifies self-adjointness preservation and exact spectrum
match against the P46 smooth targets, then measures W_1 gap to true
chi-twisted L(s, chi) zeros and sweeps the three canonical
oscillatory enrichments (phi_log, gamma_e, pi_density) honestly per
character.

Honest scope (AGENTS.md sec. 13.2 + section 13septies T-HP):

    * The smooth half of T-HP^(chi) is a constructive operator-level
      object for every primitive real chi (sub-problem (1) of T-HP,
      L-track variant).
    * The oscillatory residual S_chi(T) is NOT reachable by closed-
      form canonical constants (varphi, gamma, pi, e).
    * Negative or near-zero canonical-oscillation improvement
      mirrors branch B2 of section 13octies at the L-track level.
    * Sub-problems (2) (canonicity) and (3) (positivity coincidence
      with the chi-twisted Weil form) remain open.
    * G4 = RH AND GRH_chi BOTH remain OPEN.

Usage::

    $env:PYTHONPATH=(Resolve-Path ./src).Path
    $env:PYTHONIOENCODING="utf-8"
    & ./.venv312/Scripts/python.exe examples/75_twisted_admissible_rescaling_demo.py
"""

from __future__ import annotations

import sys

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_admissible_rescaling import (
    compute_twisted_admissible_rescaling_certificate,
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _ensure_utf8_stdout()

    n_targets = 12
    p34_n_primes = 25
    p34_max_power = 5

    characters = [
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ]

    print("=" * 78)
    print(
        "P48 -- chi-twisted admissible spectral-rescaling operator"
    )
    print(
        "L-track operator-level lift of P46 (smooth half of T-HP^(chi))"
    )
    print("=" * 78)
    print(
        f"Parameters: n_targets={n_targets}, "
        f"p34_n_primes={p34_n_primes}, p34_max_power={p34_max_power}"
    )
    print()

    summary_rows: list[tuple[str, float, float, float, str, float]] = []

    for chi in characters:
        print("-" * 78)
        print(
            f"Character: {chi.name} (modulus {chi.modulus})"
        )
        print("-" * 78)
        cert = compute_twisted_admissible_rescaling_certificate(
            chi,
            n_targets=n_targets,
            p34_n_primes=p34_n_primes,
            p34_max_power=p34_max_power,
        )
        print(cert.summary())
        print()
        summary_rows.append(
            (
                cert.character_name,
                cert.w1_p34_vs_true,
                cert.w1_smooth_vs_true,
                cert.smooth_improvement_ratio,
                cert.oscillatory_mode,
                cert.oscillatory_improvement_over_smooth * 100.0,
            )
        )

    print("=" * 78)
    print("Cross-character summary")
    print("=" * 78)
    print(
        f"{'character':<10s} {'W1(P34)':>12s} "
        f"{'W1(smooth)':>12s} {'ratio':>10s} "
        f"{'best mode':<14s} {'osc gain %':>10s}"
    )
    for name, w1_p34, w1_smooth, ratio, mode, gain in summary_rows:
        print(
            f"{name:<10s} {w1_p34:>12.4e} "
            f"{w1_smooth:>12.4e} {ratio:>9.2f}x "
            f"{mode:<14s} {gain:>+9.2f}"
        )

    print()
    print(
        "Honest scope: P48 closes sub-problem (1) of T-HP for the "
        "smooth half only,\nper primitive real character.  G4 = RH "
        "and GRH_chi BOTH remain OPEN."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
