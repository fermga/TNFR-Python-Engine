"""TNFR-Riemann P44 demo: chi-twisted Lyapunov-spectral positivity certificate.

L-track analogue of P26 demo
(:mod:`examples.53_lyapunov_spectral_positivity_demo`).  Runs
:func:`tnfr.riemann.compute_twisted_lyapunov_spectral_certificate` on
the P34 chi-twisted prime-ladder Hamiltonian for the three primitive
real Dirichlet characters used in the L-track (chi_3, chi_4, chi_5)
at the canonical decoupled limit (``coupling = 0``) and at a small
non-zero coupling (``coupling = 0.01``), and prints:

1. The full certificate per character at ``coupling = 0``.
2. A per-character coupling-sweep summary table at
   ``coupling in {0, 0.01}`` showing the empirical spectral bottom,
   the Kato-Rellich envelope, the Hilbert-Schmidt resolvent norm, the
   unitary drifts, and the boolean ``structural_positivity``.
3. An honest-scope reminder: the certificate closes operator-level
   positivity on the finite-dimensional chi-twisted prime-ladder
   Hilbert space but does **not** close gap G4-chi (GRH localisation
   on Re(s) = 1/2 for L(s, chi)).

Usage
-----
    set PYTHONPATH=src
    python examples/71_twisted_lyapunov_spectral_demo.py
"""

from __future__ import annotations

import io
import math
import sys

# Ensure UTF-8 stdout on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from tnfr.riemann import (
    build_twisted_prime_ladder_hamiltonian,
    compute_twisted_lyapunov_spectral_certificate,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


CHARACTERS = (
    ("chi_3 (mod 3)", real_character_mod_3, math.log(2.0)),
    ("chi_4 (mod 4)", real_character_mod_4, math.log(3.0)),
    ("chi_5 (mod 5)", real_character_mod_5, math.log(2.0)),
)


def main() -> None:
    n_primes = 18
    max_power = 5
    seed = 20260423

    banner(
        "HONEST SCOPE -- P44 chi-twisted Lyapunov-spectral positivity"
    )
    print(
        "  This certificate establishes self-adjointness, strict positivity\n"
        "  with explicit Kato-Rellich envelope, trace-class resolvent, and\n"
        "  unitary flow on the FINITE-DIMENSIONAL chi-twisted prime-ladder\n"
        "  Hamiltonian (P34).  It does NOT prove GRH for L(s, chi) and does\n"
        "  NOT close gap G4 = RH.  Implemented for primitive real characters\n"
        "  (chi_3, chi_4, chi_5)."
    )

    # ------------------------------------------------------------------
    # 1. Decoupled certificate per character
    # ------------------------------------------------------------------
    for char_name, char_factory, expected_gap in CHARACTERS:
        chi = char_factory()
        banner(
            f"P44 :: Decoupled certificate ({char_name}, "
            f"n_primes={n_primes}, max_power={max_power}, J0=0)"
        )
        bundle0 = build_twisted_prime_ladder_hamiltonian(
            chi, n_primes, max_power=max_power, coupling=0.0
        )
        cert0 = compute_twisted_lyapunov_spectral_certificate(
            bundle0,
            shift=1.0,
            n_unitary_states=10,
            unitary_times=(0.25, 1.0, 4.0),
            seed=seed,
        )
        print(cert0.summary())
        print()
        print(f"  character                : {cert0.character_name} "
              f"(q={cert0.character_modulus})")
        print(f"  active primes / max_pow  : {cert0.n_primes} / "
              f"{cert0.max_power}")
        print(f"  dimension                : {cert0.dimension}")
        print(f"  spectrum (min, max)      : "
              f"({cert0.spectrum_min:.6e}, {cert0.spectrum_max:.6e})")
        print(f"  spectral gap             : {cert0.spectral_gap:.6e}")
        print(f"  all_positive             : {cert0.all_positive}")
        print(f"  ||R||_1 (shift c=1)      : {cert0.schatten_1_norm:.6e}")
        print(f"  ||R||_2 (shift c=1)      : {cert0.schatten_2_norm:.6e}")
        print(f"  unperturbed gap          : {cert0.unperturbed_gap:.6e}")
        print(f"  expected gap (log p_min) : {expected_gap:.6e}")
        print(f"  perturbation bound ||V|| : {cert0.perturbation_bound:.6e}")
        print(f"  guaranteed gap           : {cert0.guaranteed_gap:.6e}")
        print(f"  perturbation_safe        : {cert0.perturbation_safe}")
        print(f"  max norm drift           : {cert0.max_norm_drift:.3e}")
        print(f"  max energy drift         : {cert0.max_energy_drift:.3e}")
        print(f"  unitary flow certified   : {cert0.unitary}")
        print(f"  STRUCTURAL POSITIVITY    : {cert0.structural_positivity}")
        # Sanity check: numerical unperturbed gap matches log(min p not | q)
        assert abs(cert0.unperturbed_gap - expected_gap) < 1.0e-12, (
            f"unperturbed gap mismatch for {char_name}: "
            f"got {cert0.unperturbed_gap}, expected {expected_gap}"
        )

    # ------------------------------------------------------------------
    # 2. Compact per-character summary at J0 in {0, 0.01}
    # ------------------------------------------------------------------
    banner(
        "P44 :: Per-character summary at coupling in {0.0, 0.01}"
    )
    print(
        f"  {'character':>15s} | {'J_0':>6s} | {'min(lambda)':>14s} | "
        f"{'guaranteed_gap':>16s} | {'pert_safe':>10s} | "
        f"{'unitary':>8s} | {'struct_pos':>11s}"
    )
    print("  " + "-" * 96)
    for char_name, char_factory, _expected_gap in CHARACTERS:
        chi = char_factory()
        for J0 in (0.0, 0.01):
            bundle = build_twisted_prime_ladder_hamiltonian(
                chi, n_primes, max_power=max_power, coupling=float(J0)
            )
            cert = compute_twisted_lyapunov_spectral_certificate(
                bundle,
                shift=1.0,
                n_unitary_states=4,
                unitary_times=(1.0,),
                seed=seed,
            )
            print(
                f"  {char_name:>15s} | {J0:>6.3f} | "
                f"{cert.spectrum_min:>14.6e} | "
                f"{cert.guaranteed_gap:>16.6e} | "
                f"{str(cert.perturbation_safe):>10s} | "
                f"{str(cert.unitary):>8s} | "
                f"{str(cert.structural_positivity):>11s}"
            )

    # ------------------------------------------------------------------
    # 3. Honest scope reminder
    # ------------------------------------------------------------------
    banner("Honest scope (do not over-claim)")
    print(
        "  - At J_0 = 0 the certificate is a finite-dimensional restatement\n"
        "    of the trivial fact diag(k log p) > 0 over the active primes\n"
        "    (those NOT dividing the conductor q); the explicit gap is\n"
        "    log(min p not dividing q): log 2 for chi_3/chi_5, log 3 for\n"
        "    chi_4.\n"
        "  - For J_0 > 0 the Kato-Rellich envelope provides a rigorous\n"
        "    quantitative interval in which positivity holds: as long as\n"
        "    ||V|| < unperturbed_gap, perturbation_safe = True.\n"
        "  - The certificate does NOT prove that non-trivial zeros of\n"
        "    L(s, chi) lie on Re(s) = 1/2 (GRH for L(s, chi)).  It is a\n"
        "    statement about positivity of the finite-dimensional chi-\n"
        "    twisted P34 operator on the chi-twisted prime-ladder Hilbert\n"
        "    space, which is a necessary but not sufficient ingredient\n"
        "    for any Hilbert-Polya-style attack on GRH or G4 = RH.\n"
        "  - The chi-twisted weight operator W^(chi) is NOT involved in\n"
        "    this certificate: positivity of H_int is independent of the\n"
        "    character (the character only enters the spectral trace\n"
        "    Z_TNFR(s, chi) and the active-prime restriction in the\n"
        "    ladder graph)."
    )


if __name__ == "__main__":
    main()
