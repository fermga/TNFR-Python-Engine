"""Demo 68: P41 chi-twisted Hermite2-Gaussian eta-parameter sweep
for primitive real Dirichlet L-functions.

Structural analogue of the zeta-track P21 milestone (which enriches
the admissible-family bundle with the Hermite2-Gaussian profile)
extended to L(s, chi) via the P34 chi-twisted prime-ladder
Hamiltonian, the P35 chi-twisted Weil-Guinand zero-side enumerator,
and the P37 chi-twisted Weil-TNFR positivity bridge.

Whereas Demo 66 (P39) audits three admissible families with eta
fixed at 0.25 inside the Hermite2 profile, Demo 68 (P41) lifts the
audit to a sweep over the Hermite2 envelope-strength grid
DEFAULT_HERMITE2_ETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0).  eta = 0.0
recovers the pure Gaussian baseline; eta = 0.25 matches the P19
default and the P39 snapshot; eta in {0.5, 1.0, 2.0} progressively
biases the test profile toward the wings.

The demo executes:

1. **Detailed (eta, gauge) chart for chi_3 at sigma=2.0**.

2. **Aggregate sweep across chi_3, chi_4, chi_5** -- reports
   aggregate positivity flags and the [alpha_min, alpha_max] range
   for each character together with the (sigma, eta, gauge)
   coordinates of the minimum.

3. **Certificate summaries and honest scope**.

Honest scope
------------
This demo numerically probes alpha_chi(sigma; eta, g) > 0 across a
finite Gaussian-width grid, a finite Hermite2 envelope-strength
``eta`` grid, and six canonical scalar gauges DEFAULT_GAUGES (P18)
for three primitive real characters.  Positive results strengthen
the P37 / P38 / P39 / P40 numerical evidence under polynomial
envelope deformation of the admissible test profile but **do not
prove** GRH for any L(s, chi) and do NOT advance G4 = RH.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from tnfr.riemann import (
    DEFAULT_HERMITE2_ETAS,
    build_twisted_prime_ladder_hamiltonian,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    sweep_twisted_hermite2_eta,
)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

SIGMAS = (1.0, 1.5, 2.0, 2.5, 3.0)

N_PRIMES = 25
MAX_POWER = 6
COUPLING = 0.0

CHARACTERS = (
    ("chi_3", real_character_mod_3),
    ("chi_4", real_character_mod_4),
    ("chi_5", real_character_mod_5),
)


def banner(title: str, char: str = "=") -> None:
    print(char * 72)
    print(title)
    print(char * 72)


def fmt_signed(x: float, width: int = 12, prec: int = 4) -> str:
    if x != x:  # NaN
        return f"{'nan':>{width}}"
    if x == float("inf"):
        return f"{'+inf':>{width}}"
    if x == float("-inf"):
        return f"{'-inf':>{width}}"
    return f"{x:+{width}.{prec}e}"


# --------------------------------------------------------------------
# Block 1: detailed chart for chi_3
# --------------------------------------------------------------------


def block_detailed_chi3() -> None:
    banner("BLOCK 1: detailed (eta, gauge) chart for chi_3 " "across sigma grid")
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=N_PRIMES,
        max_power=MAX_POWER,
        coupling=COUPLING,
    )
    cert = sweep_twisted_hermite2_eta(chi, bundle, SIGMAS)

    print(f"  character : {cert.character_name} " f"(q={cert.character_modulus})")
    print(f"  n_sigma   : {len(cert.sigmas)}")
    print(f"  etas      : {cert.etas}")
    print(f"  gauges    : {cert.gauges}")
    print()

    print("  W_chi[sigma; eta] (gauge-independent):")
    header = "    " + f"{'eta':>8}" + " ".join(f"sigma={s:>5.2f}" for s in cert.sigmas)
    print(header)
    for i, eta_val in enumerate(cert.etas):
        row = (
            "    "
            + f"{eta_val:>8.3f}"
            + " ".join(
                fmt_signed(float(cert.weil_table[i, j]), width=11, prec=3)
                for j in range(len(cert.sigmas))
            )
        )
        print(row)
    print()

    j_target = len(cert.sigmas) // 2
    sigma_target = float(cert.sigmas[j_target])
    print(f"  alpha_chi(sigma={sigma_target:.3f}; eta, gauge):")
    header = (
        "    "
        + f"{'eta\\gauge':>10}"
        + " ".join(f"{gn[:11]:>12}" for gn in cert.gauges)
    )
    print(header)
    for i, eta_val in enumerate(cert.etas):
        row = (
            "    "
            + f"{eta_val:>10.3f}"
            + " ".join(
                fmt_signed(float(cert.alpha_table[i, k, j_target]), width=12, prec=3)
                for k in range(len(cert.gauges))
            )
        )
        print(row)
    print()


# --------------------------------------------------------------------
# Block 2: aggregate sweep
# --------------------------------------------------------------------


def block_aggregate_sweep() -> list:
    banner("BLOCK 2: aggregate sweep across chi_3, chi_4, chi_5")
    certs = []
    print(
        f"  {'character':<12} {'q':>3}  {'W>=0':>6}  {'a>0':>6}  "
        f"{'alpha_min':>14}  {'@(sigma,eta,gauge)':<42}  "
        f"{'alpha_max':>14}"
    )
    print("  " + "-" * 116)
    for name, factory in CHARACTERS:
        chi = factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi,
            n_primes=N_PRIMES,
            max_power=MAX_POWER,
            coupling=COUPLING,
        )
        cert = sweep_twisted_hermite2_eta(chi, bundle, SIGMAS)
        certs.append(cert)
        loc = (
            f"({cert.alpha_min_sigma:.3f}, "
            f"{cert.alpha_min_eta:.3f}, "
            f"{cert.alpha_min_gauge})"
        )
        print(
            f"  {cert.character_name:<12} "
            f"{cert.character_modulus:>3}  "
            f"{str(cert.weil_all_positive):>6}  "
            f"{str(cert.alpha_all_positive):>6}  "
            f"{fmt_signed(cert.alpha_min, width=14, prec=4)}  "
            f"{loc:<42}  "
            f"{fmt_signed(cert.alpha_max, width=14, prec=4)}"
        )
    print()
    return certs


# --------------------------------------------------------------------
# Block 3: summaries + scope
# --------------------------------------------------------------------


def block_summaries(certs: list) -> None:
    banner("BLOCK 3: certificate summaries")
    for cert in certs:
        print("  " + cert.summary())
    print()
    banner("HONEST SCOPE", char="-")
    print(
        "  P41 numerically verifies alpha_chi(sigma; eta, g) > 0\n"
        "  across a finite Gaussian-width grid, the Hermite2\n"
        "  envelope-strength grid DEFAULT_HERMITE2_ETAS\n"
        "  (0.0, 0.1, 0.25, 0.5, 1.0, 2.0) and six canonical\n"
        "  scalar gauges DEFAULT_GAUGES (P18) for three primitive\n"
        "  real Dirichlet characters (chi_3, chi_4, chi_5).\n"
        "  Positive results strengthen the P37 / P38 / P39 / P40\n"
        "  numerical evidence under polynomial envelope deformation\n"
        f"  of the test profile (etas tested: {DEFAULT_HERMITE2_ETAS})\n"
        "  but do NOT prove GRH for any L(s, chi) and do NOT advance\n"
        "  G4 = RH.  Like P21 / P39, this is an RH-equivalent\n"
        "  robustness audit on a finite grid, not a theorem on\n"
        "  family-uniform positivity over the full admissible\n"
        "  Schwartz-even space."
    )


def main() -> None:
    block_detailed_chi3()
    certs = block_aggregate_sweep()
    block_summaries(certs)


if __name__ == "__main__":
    main()
