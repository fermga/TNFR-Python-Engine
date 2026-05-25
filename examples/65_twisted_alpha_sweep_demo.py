"""Demo 65: P38 chi-twisted admissibility / gauge sweep for primitive
real Dirichlet L-functions.

Structural analogue of Demo 47 (P18 alpha-sweep for zeta) extended to
L(s, chi) via the P34 chi-twisted prime-ladder Hamiltonian, the P35
chi-twisted Weil-Guinand zero-side enumerator, and the P37 chi-twisted
Weil-TNFR positivity bridge.

The demo executes:

1. **Detailed (sigma, gauge) chart for chi_3** -- the smallest
   non-trivial primitive real character.  Prints the W_chi[sigma]
   row (gauge-independent) and the alpha_chi(sigma; g) table across
   the canonical six-gauge family DEFAULT_GAUGES (inherited from the
   zeta-track P18 stress test for canonical comparability).

2. **Aggregate sweep across chi_3, chi_4, chi_5** -- reports
   aggregate positivity flags (W_chi >= 0, alpha_chi > 0) and the
   range [alpha_min, alpha_max] for each character together with the
   coordinates of the minimum (most demanding) entry.

3. **Certificate summaries and honest scope** -- prints the frozen
   summaries and reiterates the honesty disclaimer.

Honest scope
------------
This demo numerically probes alpha_chi(sigma; g) > 0 across a finite
Gaussian grid and six structural gauges for three primitive real
characters.  Positive results strengthen the P37 numerical evidence
under canonical-mapping ambiguity but **do not prove** GRH for any
L(s, chi).  This demo also does NOT advance G4 = RH or the arithmetic
obstruction of GRH.  Negative entries would falsify the bridge *as
parameterised by the given gauge family*; they would not disprove
GRH_chi, which depends only on the gauge-independent quantity
W_chi[sigma].
"""

from __future__ import annotations

import sys

# Ensure UTF-8 on Windows cp1252 consoles (chi, gamma, sigma...).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

from tnfr.riemann import (
    DEFAULT_GAUGES,
    build_twisted_prime_ladder_hamiltonian,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    sweep_twisted_alpha,
)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# Detailed (sigma, gauge) chart for chi_3
SIGMAS_DETAILED = (1.0, 1.5, 2.0, 2.5, 3.0)

# Aggregate sweep widths
SIGMAS_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0)

# Bundle configuration
N_PRIMES = 25
MAX_POWER = 6
COUPLING = 0.0  # decoupled spectrum is exact

# Primitive real characters (small conductor)
CHARACTERS = (
    ("chi_3", 3, real_character_mod_3),
    ("chi_4", 4, real_character_mod_4),
    ("chi_5", 5, real_character_mod_5),
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
# Block 1: detailed (sigma, gauge) chart for chi_3
# --------------------------------------------------------------------


def block_detailed_chi3() -> None:
    banner("BLOCK 1: detailed (sigma, gauge) chart for chi_3")
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=N_PRIMES,
        max_power=MAX_POWER,
        coupling=COUPLING,
    )
    cert = sweep_twisted_alpha(chi, bundle, SIGMAS_DETAILED)

    gauge_names = cert.gauges
    print(f"  character  : {cert.character_name} (q={cert.character_modulus})")
    print(f"  n_sigma    : {len(cert.sigmas)}")
    print(f"  n_gauge    : {len(gauge_names)}")
    print()

    # W_chi row
    print("  W_chi[sigma] (gauge-independent):")
    print("    " + " ".join(f"sigma={s:.3f}" for s in cert.sigmas))
    print("    " + " ".join(fmt_signed(w, 11, 3) for w in cert.weil_values))
    print()

    # alpha_chi table
    name_w = max(len(n) for n in gauge_names)
    header = "  gauge".ljust(name_w + 4) + "".join(
        f"  sigma={s:.2f}".rjust(13) for s in cert.sigmas
    )
    print("  alpha_chi(sigma; g) = W_chi / E_TNFR_chi:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, g in enumerate(gauge_names):
        row_vals = "".join(
            fmt_signed(cert.alpha_table[i, j], 13, 4)
            for j in range(len(cert.sigmas))
        )
        print(f"  {g.ljust(name_w + 2)}{row_vals}")
    print()


# --------------------------------------------------------------------
# Block 2: aggregate sweep across chi_3 / chi_4 / chi_5
# --------------------------------------------------------------------


def block_sweep_all_characters() -> None:
    banner("BLOCK 2: aggregate sweep across chi_3, chi_4, chi_5")
    print(
        "  Sweeping {} sigmas x {} gauges per character...".format(
            len(SIGMAS_SWEEP), len(DEFAULT_GAUGES)
        )
    )
    print()

    header = (
        f"  {'chi':<8} {'q':>3}  "
        f"{'W>=0':>6} {'alpha>0':>9}  "
        f"{'alpha_min':>14} @ "
        f"{'(sigma, gauge)':<28}"
        f"  {'alpha_max':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    summaries = []
    for name, _q, factory in CHARACTERS:
        chi = factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi,
            n_primes=N_PRIMES,
            max_power=MAX_POWER,
            coupling=COUPLING,
        )
        cert = sweep_twisted_alpha(chi, bundle, SIGMAS_SWEEP)
        coords = (
            f"({cert.alpha_min_sigma:.3f}, '{cert.alpha_min_gauge}')"
        )
        print(
            f"  {cert.character_name:<8} {cert.character_modulus:>3d}  "
            f"{str(cert.weil_all_positive):>6} "
            f"{str(cert.alpha_all_positive):>9}  "
            f"{fmt_signed(cert.alpha_min, 14, 4)} @ "
            f"{coords:<28}"
            f"  {fmt_signed(cert.alpha_max, 13, 4)}"
        )
        summaries.append(cert)
    print()
    return summaries


# --------------------------------------------------------------------
# Block 3: certificate summaries + honest scope
# --------------------------------------------------------------------


def block_summaries_and_scope(summaries) -> None:
    banner("BLOCK 3: certificate summaries and honest scope")
    for cert in summaries:
        print("  " + cert.summary())
    print()
    print("Honest scope:")
    print("  * P38 strengthens the P37 chi-twisted positivity bridge by")
    print("    probing alpha_chi(sigma; g) across a six-gauge family")
    print("    (DEFAULT_GAUGES) inherited from the zeta-track P18.")
    print("  * Positive alpha_chi across (sigma, gauge) is a robustness")
    print("    diagnostic for the canonical-mapping ambiguity of the")
    print("    P37 bridge; it is NOT a proof of GRH for L(s, chi).")
    print("  * Negative alpha_chi entries would falsify the bridge *as")
    print("    parameterised* by the gauge family; they would not")
    print("    disprove GRH_chi, which depends only on W_chi[sigma].")
    print("  * P38 does NOT advance G4 = RH (zeta zeros on Re(s) = 1/2)")
    print("    or the arithmetic obstruction of GRH.")
    print()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main() -> int:
    banner("Demo 65: P38 chi-twisted alpha-sweep / gauge robustness")
    print()
    block_detailed_chi3()
    summaries = block_sweep_all_characters()
    block_summaries_and_scope(summaries)
    banner("Demo 65 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
