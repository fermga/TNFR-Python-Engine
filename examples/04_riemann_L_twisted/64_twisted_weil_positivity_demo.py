"""Demo 64: P37 chi-twisted Weil-TNFR positivity bridge for primitive
real Dirichlet L-functions.

Structural analogue of Demo 46 (Weil-TNFR positivity bridge for zeta)
extended to L(s, chi) via the P34 chi-twisted prime-ladder Hamiltonian
and the P35 chi-twisted Weil-Guinand explicit formula.

This demo executes three blocks:

1. **Detailed sigma chart for chi_3** -- the smallest non-trivial
   primitive real character.  Tabulates W_chi[sigma] (zero-side),
   W_chi[sigma] (explicit-formula side), the consistency residual,
   the canonical TNFR Lyapunov energy E_TNFR_chi[sigma], and the
   bridge ratio alpha_chi(sigma) = W / E across a grid of widths.

2. **Sweep across chi_3, chi_4, chi_5** -- the three smallest
   primitive real Dirichlet characters.  Reports the aggregate
   positivity flags (W_all_positive, alpha_all_positive) and the
   range [alpha_min, alpha_max] for each character.

3. **Certificate summaries and honest scope** -- prints the frozen
   summaries and reiterates the honesty disclaimer.

Honest scope
------------
This demo numerically checks W_chi[sigma] >= 0 on a finite Gaussian
grid for three primitive real characters.  Positive results are
GRH_chi-equivalent diagnostics but **do not prove** GRH for any
L(s, chi) (the inequality has to hold for an exhaustive admissible
family, which a finite grid cannot exhaust).  This demo also does
NOT advance G4 = RH or the arithmetic obstruction of GRH.
"""

from __future__ import annotations

import sys

# Ensure UTF-8 on Windows cp1252 consoles (chi, gamma, sigma...).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import numpy as np

from tnfr.riemann import (
    build_twisted_prime_ladder_hamiltonian,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    verify_twisted_weil_positivity,
    verify_twisted_weil_tnfr_bridge,
)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# Detailed chart sigmas for chi_3
SIGMAS_DETAILED = (1.0, 1.5, 2.0, 2.5, 3.0)

# Sweep sigmas for chi_3 / chi_4 / chi_5 bridge
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


def _build(builder) -> tuple:
    """Build the character and the chi-twisted P34 bundle."""
    chi = builder()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=N_PRIMES,
        max_power=MAX_POWER,
        coupling=COUPLING,
    )
    return chi, bundle


# --------------------------------------------------------------------
# Block 1: detailed chart for chi_3
# --------------------------------------------------------------------


def block_detailed_chi3() -> None:
    banner("P37 Block 1: detailed Weil-TNFR chart for chi_3 (q=3)")
    chi, bundle = _build(real_character_mod_3)
    print(
        f"Character: {chi.name}  modulus={chi.modulus}  "
        f"primes_excluded={bundle.graph.graph['primes_excluded']}  "
        f"n_nodes={bundle.graph.number_of_nodes()}"
    )
    print()
    header = (
        f"  {'sigma':>6s}  {'W_zero':>14s}  {'W_xf':>14s}  "
        f"{'residual':>10s}  {'E_TNFR':>14s}  {'alpha':>14s}  "
        f"{'pos':>4s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sigma in SIGMAS_DETAILED:
        cert = verify_twisted_weil_positivity(chi, bundle, sigma=sigma)
        from tnfr.riemann import twisted_tnfr_lyapunov_of_test_state

        E = twisted_tnfr_lyapunov_of_test_state(bundle, sigma)
        alpha = cert.weil_functional_zero_side / E if E > 0.0 else float("inf")
        print(
            f"  {sigma:>6.2f}  "
            f"{cert.weil_functional_zero_side:>+14.6e}  "
            f"{cert.weil_functional_explicit_formula:>+14.6e}  "
            f"{cert.explicit_formula_residual:>10.2e}  "
            f"{E:>+14.6e}  "
            f"{alpha:>+14.6e}  "
            f"{'YES' if cert.positive else 'no':>4s}"
        )
    print()


# --------------------------------------------------------------------
# Block 2: sweep across chi_3 / chi_4 / chi_5
# --------------------------------------------------------------------


def block_sweep() -> None:
    banner("P37 Block 2: chi-twisted Weil-TNFR bridge sweep")
    print(
        f"Gaussian widths sigma in {tuple(SIGMAS_SWEEP)}  " f"(n={len(SIGMAS_SWEEP)})"
    )
    print()
    summaries = []
    for label, q, builder in CHARACTERS:
        chi, bundle = _build(builder)
        cert = verify_twisted_weil_tnfr_bridge(chi, bundle, SIGMAS_SWEEP)
        summaries.append((label, q, cert))
        print(f"--- {label}  (q={q}, a={cert.character_parity}) ---")
        print(cert.summary())
        # Show per-sigma table
        print(
            f"  {'sigma':>6s}  {'W_chi':>14s}  {'E_TNFR':>14s}  "
            f"{'alpha':>14s}  {'W>=0':>5s}  {'a>0':>5s}"
        )
        for i, sigma in enumerate(cert.sigmas):
            print(
                f"  {float(sigma):>6.2f}  "
                f"{float(cert.weil_functional[i]):>+14.6e}  "
                f"{float(cert.tnfr_lyapunov_energy[i]):>+14.6e}  "
                f"{float(cert.alpha[i]):>+14.6e}  "
                f"{'YES' if cert.weil_positive[i] else 'no':>5s}  "
                f"{'YES' if cert.bridge_positive[i] else 'no':>5s}"
            )
        print()

    banner("Aggregate verdicts", char="-")
    n_pass = 0
    n_total = 0
    for label, q, cert in summaries:
        n_total += 1
        ok = cert.weil_positive_all and cert.bridge_positive_all
        if ok:
            n_pass += 1
        print(
            f"  {label} (q={q}): W_all_positive={cert.weil_positive_all}  "
            f"alpha_all_positive={cert.bridge_positive_all}  "
            f"alpha_range=[{cert.alpha_min:.4e}, {cert.alpha_max:.4e}]  "
            f"=> {'PASS' if ok else 'FAIL'}"
        )
    print()
    print(f"  Overall: {n_pass}/{n_total} characters pass Weil + bridge")
    print()


# --------------------------------------------------------------------
# Block 3: honest scope
# --------------------------------------------------------------------


def block_honest_scope() -> None:
    banner("P37 Honest scope")
    print(
        """
This demo numerically verifies the chi-twisted Weil positivity
inequality W_chi[sigma] >= 0 on a finite Gaussian grid for three
primitive real Dirichlet characters (chi_3, chi_4, chi_5), and
exposes the structural ratio alpha_chi(sigma) = W_chi[sigma] /
E_TNFR_chi[sigma] using a canonical TNFR test state on the P34
chi-twisted prime-ladder graph.

What this DOES establish:
  * GRH_chi-equivalent **diagnostic** on a finite Gaussian family.
  * Consistency between the zero side (P35 Hardy-Z bisection) and
    the explicit-formula side (P34 Hamiltonian prime side + P35
    archimedean/constant terms) -- the residual quantifies the
    self-consistency of P35.
  * Structural lower-bound candidate alpha_min > 0 in pure TNFR
    terms for each character considered.

What this DOES NOT establish:
  * GRH for any L(s, chi) (a finite Gaussian grid cannot exhaust
    the admissible family that makes Weil positivity equivalent to
    GRH_chi).
  * RH for the Riemann zeta (this module is about chi-twisted
    L-functions; the untwisted positivity bridge is P17).
  * Any progress on gap G4 = RH or the arithmetic obstruction of
    GRH.

The canonical TNFR test state used here is one of many possible
mappings of h_sigma to the P34 graph; the bridge ratio alpha_chi
reported is specific to this mapping.
"""
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=False)
    banner("P37 - chi-twisted Weil-TNFR positivity bridge demo")
    print(
        """Configuration:
  N_PRIMES   = {n_primes}
  MAX_POWER  = {max_power}
  COUPLING   = {coupling} (decoupled spectrum is exact)
  Sigmas (detailed) = {s_det}
  Sigmas (sweep)    = {s_swp}
""".format(
            n_primes=N_PRIMES,
            max_power=MAX_POWER,
            coupling=COUPLING,
            s_det=tuple(SIGMAS_DETAILED),
            s_swp=tuple(SIGMAS_SWEEP),
        )
    )
    block_detailed_chi3()
    block_sweep()
    block_honest_scope()


if __name__ == "__main__":
    main()
