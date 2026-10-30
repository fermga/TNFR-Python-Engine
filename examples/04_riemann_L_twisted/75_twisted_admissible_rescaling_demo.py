"""Finite character-twisted congruence rescaling to supplied smooth targets.

The P34 eigenbasis and classical smooth targets define the rescaling by
construction. Spectral agreement is a finite algebra/numerical check, not
independent emergence of L-function zeros. The configured enrichment sweep
cannot establish impossibility for all constants or require a new canonical
operator. A nodal derivation and any RH/GRH proof remain separate obligations.
Current scope: theory/TNFR_RIEMANN_RESEARCH_NOTES.md.
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
    print("P48 -- chi-twisted admissible spectral-rescaling operator")
    print("L-track operator-level lift of P46 (smooth half of T-HP^(chi))")
    print("=" * 78)
    print(
        f"Parameters: n_targets={n_targets}, "
        f"p34_n_primes={p34_n_primes}, p34_max_power={p34_max_power}"
    )
    print()

    summary_rows: list[tuple[str, float, float, float, str, float]] = []

    for chi in characters:
        print("-" * 78)
        print(f"Character: {chi.name} (modulus {chi.modulus})")
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
        "Scope: P48 rescales to supplied smooth targets for each character.\n"
        "Finite target matching neither derives those inputs from nodal "
        "dynamics nor proves RH or GRH."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
