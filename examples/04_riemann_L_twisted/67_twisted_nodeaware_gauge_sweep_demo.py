"""Demo 67: P40 chi-twisted node-aware gauge sweep for primitive real
Dirichlet L-functions.

Structural analogue of Demo 49 (P20 node-aware gauge sweep for zeta)
extended to L(s, chi) via the P34 chi-twisted prime-ladder
Hamiltonian, the P35 chi-twisted Weil-Guinand zero-side enumerator,
and the P37 chi-twisted Weil-TNFR positivity bridge.

Whereas Demo 66 (P39) audited the canonical scalar-h gauges
DEFAULT_GAUGES (P18), Demo 67 (P40) sweeps the **node-aware** gauges
DEFAULT_NODEAWARE_GAUGES (P20) -- gauges of the form
``(d, phi, eps) = g(h(E_n), hat nu_f(n), hat w(n))`` --
across the same three admissible Schwartz-even test families
(gaussian, gaussian_mixture, hermite2_gaussian, inherited from P19).

The demo executes:

1. **Detailed (family, node_gauge) chart for chi_3 at sigma=2.0**.

2. **Aggregate sweep across chi_3, chi_4, chi_5** -- reports
   aggregate positivity flags and the [alpha_min, alpha_max] range
   for each character together with the (sigma, family, node_gauge)
   coordinates of the minimum.

3. **Certificate summaries and honest scope**.

Honest scope
------------
This demo numerically probes alpha_chi(sigma; f, g) > 0 across three
admissible Schwartz-even test families, four canonical node-aware
gauges, and a finite Gaussian-width grid for three primitive real
characters.  Positive results strengthen the P37 / P38 / P39
numerical evidence under joint test-profile + node-aware mapping
ambiguity but **do not prove** GRH for any L(s, chi).  This demo
also does NOT advance G4 = RH or the arithmetic obstruction of GRH.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from tnfr.riemann import (
    build_twisted_prime_ladder_hamiltonian,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    sweep_twisted_nodeaware_gauge,
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
    banner(
        "BLOCK 1: detailed (family, node_gauge) chart for chi_3 "
        "across sigma grid"
    )
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=N_PRIMES,
        max_power=MAX_POWER,
        coupling=COUPLING,
    )
    cert = sweep_twisted_nodeaware_gauge(chi, bundle, SIGMAS)

    print(f"  character   : {cert.character_name} "
          f"(q={cert.character_modulus})")
    print(f"  n_sigma     : {len(cert.sigmas)}")
    print(f"  families    : {cert.families}")
    print(f"  node_gauges : {cert.node_gauges}")
    print()

    print("  W_chi[sigma; family] (gauge-independent):")
    header = "    " + f"{'family':<22}" + " ".join(
        f"sigma={s:>5.2f}" for s in cert.sigmas
    )
    print(header)
    for i, fname in enumerate(cert.families):
        row = "    " + f"{fname:<22}" + " ".join(
            fmt_signed(float(cert.weil_table[i, j]), width=11, prec=3)
            for j in range(len(cert.sigmas))
        )
        print(row)
    print()

    j_target = len(cert.sigmas) // 2
    sigma_target = float(cert.sigmas[j_target])
    print(f"  alpha_chi(sigma={sigma_target:.3f}; family, node_gauge):")
    header = "    " + f"{'family\\node_gauge':<22}" + " ".join(
        f"{gn[:13]:>14}" for gn in cert.node_gauges
    )
    print(header)
    for i, fname in enumerate(cert.families):
        row = "    " + f"{fname:<22}" + " ".join(
            fmt_signed(float(cert.alpha_table[i, k, j_target]),
                       width=14, prec=4)
            for k in range(len(cert.node_gauges))
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
        f"{'alpha_min':>14}  {'@(sigma,fam,node_gauge)':<48}  "
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
        cert = sweep_twisted_nodeaware_gauge(chi, bundle, SIGMAS)
        certs.append(cert)
        loc = (
            f"({cert.alpha_min_sigma:.3f}, "
            f"{cert.alpha_min_family}, "
            f"{cert.alpha_min_node_gauge})"
        )
        print(
            f"  {cert.character_name:<12} "
            f"{cert.character_modulus:>3}  "
            f"{str(cert.weil_all_positive):>6}  "
            f"{str(cert.alpha_all_positive):>6}  "
            f"{fmt_signed(cert.alpha_min, width=14, prec=4)}  "
            f"{loc:<48}  "
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
        "  P40 numerically verifies alpha_chi(sigma; f, g) > 0\n"
        "  across a finite Gaussian-width grid, three admissible\n"
        "  Schwartz-even test families (P19) and four canonical\n"
        "  node-aware gauges DEFAULT_NODEAWARE_GAUGES (P20) for three\n"
        "  primitive real Dirichlet characters (chi_3, chi_4, chi_5).\n"
        "  Positive results strengthen the P37 / P38 / P39 numerical\n"
        "  evidence under joint test-profile + node-aware mapping\n"
        "  ambiguity but do NOT prove GRH for any L(s, chi) and do\n"
        "  NOT advance G4 = RH.  Like P20 / P39, this is an\n"
        "  RH-equivalent robustness audit on a finite grid, not a\n"
        "  theorem on family-uniform positivity."
    )


def main() -> None:
    block_detailed_chi3()
    certs = block_aggregate_sweep()
    block_summaries(certs)


if __name__ == "__main__":
    main()
