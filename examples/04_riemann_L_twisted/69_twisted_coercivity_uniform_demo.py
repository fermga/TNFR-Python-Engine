"""Demo 69: P42 chi-twisted uniform-coercivity certificate for
primitive real Dirichlet L-functions.

Structural extension of the zeta-track P22 / P23 / P24 milestones
(empirical interval-level coercivity certificate with adaptive
refinement) to L(s, chi) via the P34 chi-twisted prime-ladder
Hamiltonian, the P35 chi-twisted Weil-Guinand zero-side enumerator,
and the P37 chi-twisted Weil-TNFR positivity bridge.

Whereas Demos 66 / 67 / 68 (P39 / P40 / P41) verify pointwise
positivity of alpha_chi(sigma; f, g) on a finite (sigma, family,
gauge) grid, Demo 69 (P42) lifts pointwise positivity to *interval*
positivity on a dense log-spaced sigma window by combining:

* a finite-difference Lipschitz proxy L_proxy, and
* three interval lower bounds (global, stratified, segment-local),
* plus optional adaptive bisection of the worst segments.

The demo executes:

1. **Sigma-window certificate for chi_3, chi_4, chi_5** without
   refinement (pure P22-style).

2. **Adaptive refinement** for the worst character with two rounds
   of bisection on the two worst segments per round (P24-style).

3. **Certificate summaries and honest scope**.

Honest scope
------------
This demo numerically probes alpha_chi > 0 across an interval-level
empirical certificate (sampled minima + Lipschitz envelope + segment
linear envelope) on three primitive real characters.  Positive
results strengthen the P37 / P38 / P39 / P40 / P41 numerical
evidence by promoting pointwise positivity to interval-level
positivity with explicit Lipschitz control, but **do not prove**
GRH for any L(s, chi) and do NOT advance G4 = RH.  Like P22 / P23
/ P24 on the zeta side, this is an empirical interval-level
diagnostic on a finite grid, not a theorem on family-uniform
positivity over the full admissible Schwartz-even space.
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
    verify_twisted_uniform_coercivity_empirical,
)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

SIGMA_MIN = 1.0
SIGMA_MAX = 3.0
N_SIGMA = 5

N_PRIMES = 15
MAX_POWER = 4
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
# Block 1: sigma-window certificate (no refinement)
# --------------------------------------------------------------------


def block_sigma_window() -> list:
    banner("BLOCK 1: chi-twisted uniform-coercivity certificate " "(no refinement)")
    print(
        f"  sigma window : [{SIGMA_MIN:.3f}, {SIGMA_MAX:.3f}]  "
        f"n_sigma = {N_SIGMA}  (log-spaced)"
    )
    print()
    print(
        f"  {'chi':<14} {'q':>3}  {'a_min':>14}  {'a_max':>14}  "
        f"{'L_proxy':>12}  {'lb_global':>14}  {'lb_strat':>14}  "
        f"{'lb_local':>14}  {'all+':>6}"
    )
    print("  " + "-" * 130)
    certs = []
    for name, factory in CHARACTERS:
        chi = factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi,
            n_primes=N_PRIMES,
            max_power=MAX_POWER,
            coupling=COUPLING,
        )
        cert = verify_twisted_uniform_coercivity_empirical(
            chi,
            bundle,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            n_sigma=N_SIGMA,
        )
        certs.append((name, cert))
        all_pos = bool(
            cert.sampled_all_positive
            and cert.interval_lower_global_positive
            and cert.interval_lower_stratified_positive
            and cert.interval_lower_local_positive
        )
        print(
            f"  {cert.character_name:<14} "
            f"{cert.character_modulus:>3}  "
            f"{fmt_signed(cert.sampled_alpha_min, width=14, prec=4)}  "
            f"{fmt_signed(cert.sampled_alpha_max, width=14, prec=4)}  "
            f"{cert.lipschitz_proxy_max:>12.4e}  "
            f"{fmt_signed(cert.interval_lower_bound_global, 14, 4)}  "
            f"{fmt_signed(cert.interval_lower_bound_stratified, 14, 4)}  "
            f"{fmt_signed(cert.interval_lower_bound_local, 14, 4)}  "
            f"{str(all_pos):>6}"
        )
    print()
    return certs


# --------------------------------------------------------------------
# Block 2: adaptive refinement on the worst character
# --------------------------------------------------------------------


def block_adaptive_refinement(certs: list) -> None:
    banner("BLOCK 2: adaptive refinement on worst-margin character " "(P24-style)")
    name, worst_cert = min(certs, key=lambda nc: nc[1].interval_lower_bound_local)
    print(
        f"  worst character     : {worst_cert.character_name} "
        f"(q={worst_cert.character_modulus})"
    )
    print(f"  pre-refinement lb   : " f"{worst_cert.interval_lower_bound_local:+.4e}")
    print(f"  pre-refinement n_sigma : {worst_cert.n_sigma}")
    print()

    factory = dict(CHARACTERS)[name]
    chi = factory()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi,
        n_primes=N_PRIMES,
        max_power=MAX_POWER,
        coupling=COUPLING,
    )
    refined = verify_twisted_uniform_coercivity_empirical(
        chi,
        bundle,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        n_sigma=N_SIGMA,
        refinement_rounds=1,
        refinement_per_round=2,
    )
    print(f"  refinement_rounds   : {refined.n_refinement_rounds}")
    print(f"  refined n_sigma     : {refined.n_sigma_refined}")
    print(f"  pre-refinement lb   : " f"{refined.interval_lower_bound_local:+.4e}")
    print(
        f"  refined lb (local)  : " f"{refined.interval_lower_bound_local_refined:+.4e}"
    )
    print(
        f"  refined lb positive : " f"{refined.interval_lower_local_refined_positive}"
    )
    print()


# --------------------------------------------------------------------
# Block 3: summaries + scope
# --------------------------------------------------------------------


def block_summaries(certs: list) -> None:
    banner("BLOCK 3: certificate summaries")
    for _, cert in certs:
        print("  " + cert.summary())
    print()
    banner("HONEST SCOPE", char="-")
    print(
        "  P42 numerically verifies alpha_chi > 0 at the interval\n"
        "  level via a sampled minimum plus a finite-difference\n"
        "  Lipschitz envelope on a finite log-spaced Gaussian-width\n"
        "  window for three primitive real Dirichlet characters\n"
        "  (chi_3, chi_4, chi_5).  Positive lower bounds strengthen\n"
        "  the P37 / P38 / P39 / P40 / P41 numerical evidence by\n"
        "  promoting pointwise positivity to interval-level\n"
        "  positivity with explicit Lipschitz control, but do NOT\n"
        "  prove GRH for any L(s, chi) and do NOT advance G4 = RH.\n"
        "  Like P22 / P23 / P24 on the zeta side, this is an\n"
        "  empirical interval-level diagnostic on a finite grid,\n"
        "  not a theorem on family-uniform positivity over the full\n"
        "  admissible Schwartz-even space."
    )


def main() -> None:
    certs = block_sigma_window()
    block_adaptive_refinement(certs)
    block_summaries(certs)


if __name__ == "__main__":
    main()
