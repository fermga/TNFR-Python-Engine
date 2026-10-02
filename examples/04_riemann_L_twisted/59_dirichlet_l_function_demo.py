"""P32 demo: chi-twisted TNFR prime-ladder reproducing Dirichlet L-functions.

Runs the verification

    Z_TNFR(s, chi) := Sum_{(mu, w) in Spec_TNFR(chi)} w exp(-s mu)
    versus
    Sum_{n <= N} chi(n) Lambda(n) n^{-s}
    converging to -L'(s, chi) / L(s, chi)  for  Re(s) > 1,

for the four canonical real characters provided by the module:
  - chi_0  mod 3   (principal)
  - chi    mod 3   (real, primitive; Legendre (n/3))
  - chi    mod 4   (real, primitive; gives Dirichlet beta)
  - chi    mod 5   (real, primitive; Legendre (n/5))

Reports the worst-case relative error against the classical truncated
twisted von Mangoldt series.  Per-prime-power correspondence
guarantees machine-precision agreement when the two truncations cover
the same set of prime powers (analogous to the P12 unit-test
invariant), and converges to the true logarithmic derivative as both
truncations grow.
"""

from __future__ import annotations

from tnfr.riemann import (
    principal_character,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    verify_dirichlet_l_reproduction,
)


def main() -> None:
    # Test on a complex s-grid with Re(s) > 1
    s_values = [
        2.0 + 0.0j,
        2.0 + 1.0j,
        3.0 + 0.0j,
        3.0 + 2.0j,
        5.0 + 0.0j,
    ]

    characters = [
        principal_character(3),
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ]

    print("=" * 78)
    print("P32 demo: chi-twisted TNFR prime-ladder vs classical")
    print("       Sum_{n<=N} chi(n) Lambda(n) n^{-s}")
    print("=" * 78)
    print()

    for chi in characters:
        result = verify_dirichlet_l_reproduction(
            chi,
            s_values,
            n_primes=200,
            max_power=12,
            n_max_classical=100_000,
        )
        print(result.summary())

        # Per-s detail
        for s_val, z_t, z_c, rel in zip(
            result.s_values,
            result.z_tnfr,
            result.z_classical,
            result.rel_error,
        ):
            print(
                f"    s = {s_val.real:+.2f}{s_val.imag:+.2f}j  "
                f"Z_TNFR = {z_t.real:+.6e}{z_t.imag:+.6e}j  "
                f"rel_err = {rel:.3e}"
            )
        print()

    print("=" * 78)
    print("Honest scope statement")
    print("=" * 78)
    print(
        "P32 is a *structural extension* of P12 (gap G5 superseded by\n"
        "P12+P13+P15). It generalises the TNFR prime-ladder representation\n"
        "from the Riemann zeta function to every Dirichlet L-function via\n"
        "chi-twisted echo weights w_{p,k} = chi(p)^k log p.\n\n"
        "It does NOT advance G4 = RH or the generalised Riemann hypothesis\n"
        "(GRH). The arithmetic obstruction documented in section 13octies\n"
        "for zeta — the residual oscillatory content S(T) = pi^-1 arg\n"
        "zeta(1/2+iT) — has an exact analogue for every L-function\n"
        "(S_chi(T) = pi^-1 arg L(1/2+iT, chi)) and remains open in the\n"
        "same RH-equivalent sense."
    )


if __name__ == "__main__":
    main()
