r"""Example 72: TNFR-Riemann P45 demo - chi-twisted Hilbert-Polya scaffold.

L-track analogue of P27 (Example 47).  Builds and certifies the
explicit reference operator ``T_HP^{(chi)} = diag(gamma_n^{(chi)})``
on the truncated chi-twisted Hilbert space for each primitive real
character chi_3, chi_4, chi_5, where ``gamma_n^{(chi)}`` are positive
imaginary parts of zeros of ``L(s, chi)`` located by Hardy-Z bisection.

Each certificate checks:

* self-adjointness of ``T_HP^{(chi)}`` (real diagonal, exact);
* trace-class shifted resolvent norms;
* chi-twisted Weil-Guinand consistency with the P34 chi-twisted
  prime-ladder Hamiltonian, using a Gaussian test function with
  sigma = 2.0;
* Wasserstein-1 spectral gap against ``spec(P34 | primes_active)``
  truncated to the same length.

Honest scope (mandatory, AGENTS.md): P45 does **NOT** prove GRH for
any ``L(s, chi)``.  ``T_HP^{(chi)}`` is populated by *inputting* the
zeros from Hardy-Z bisection of the classical ``L(s, chi)``; the
operator is not derived from TNFR first principles.  P45 does NOT
contribute to closing gap G4 = RH; it is the L-track structural
mirror of the diagnostic scaffold P27 on the ζ-track.
"""

from __future__ import annotations

import sys

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_hilbert_polya import compute_twisted_hilbert_polya_certificate


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _print_certificate(cert) -> None:
    print(f"  character        = {cert.character_name}")
    print(f"  modulus q        = {cert.character_modulus}")
    print(
        f"  parity a         = {cert.character_parity}  "
        f"({'even' if cert.character_parity == 0 else 'odd'})"
    )
    print(f"  n_zeros          = {cert.n_zeros}")
    print(f"  n_primes         = {cert.n_primes}")
    print(f"  max_power        = {cert.max_power}")
    print(f"  gaussian sigma   = {cert.gaussian_sigma:.3f}")
    print(f"  resolvent shift  = {cert.resolvent_shift:.3f}")
    print()
    print("  Self-adjointness")
    print(f"    asymmetry (Frobenius) = {cert.asymmetry_frobenius:.3e}")
    print(f"    self_adjoint          = {cert.self_adjoint}")
    print()
    print("  Resolvent Schatten norms of (T_HP^2 + s^2 I)^{-1/2}")
    print(f"    ||R||_1           = {cert.schatten_1_norm:.6e}")
    print(f"    ||R||_2           = {cert.schatten_2_norm:.6e}")
    print(f"    ||R||_op          = {cert.operator_norm_inverse:.6e}")
    print(f"    trace_class       = {cert.trace_class}")
    print()
    print("  chi-twisted Weil-Guinand identity")
    print(f"    zero side  (via T_HP)         = {cert.zero_side_via_hp:+.6e}")
    print(f"    constant term g(0)log(q/pi)   = {cert.constant_term:+.6e}")
    print(f"    archimedean term              = {cert.archimedean_side:+.6e}")
    print(f"    prime side (via P34)          = {cert.prime_side_via_p34:+.6e}")
    print(f"    rhs total                     = {cert.rhs_total:+.6e}")
    print(f"    residual                      = {cert.residual:.3e}")
    print(f"    relative residual             = {cert.relative_residual:.3e}")
    print(f"    tolerance                     = {cert.weil_tolerance:.3e}")
    print(f"    weil_verified                 = {cert.weil_verified}")
    print()
    print("  Operator-level structural gap spec(P34) vs spec(T_HP^(chi))")
    print(f"    n_compared        = {cert.spectral_gap_n_compared}")
    print(f"    Wasserstein-1     = {cert.spectral_gap_wasserstein_1:.4e}")
    print(f"    growth ratio      = " f"{cert.spectral_gap_growth_ratio:.4e}")
    print()
    print(f"  scaffold_consistent = {cert.scaffold_consistent}")


def _print_comparison_table(certs) -> None:
    print(
        f"  {'chi':<18s}  {'q':>3s}  {'a':>3s}  "
        f"{'W_1(P34,HP)':>14s}  {'weil_resid':>12s}  "
        f"{'self_adj':>8s}  {'trace':>6s}  {'OK':>4s}"
    )
    for c in certs:
        print(
            f"  {c.character_name:<18s}  {c.character_modulus:>3d}  "
            f"{c.character_parity:>3d}  "
            f"{c.spectral_gap_wasserstein_1:>14.4e}  "
            f"{c.residual:>12.3e}  "
            f"{str(c.self_adjoint):>8s}  {str(c.trace_class):>6s}  "
            f"{str(c.scaffold_consistent):>4s}"
        )


def main() -> int:
    _ensure_utf8_stdout()

    n_primes = 18
    max_power = 5
    n_zeros = 25
    gaussian_sigma = 2.0
    resolvent_shift = 1.0
    weil_tolerance = 1e-2

    characters = [
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ]

    print("=" * 78)
    print("P45 CHI-TWISTED HILBERT-POLYA SCAFFOLD DEMO")
    print(
        "(L-track analogue of P27; after Mart\u00ednez Gamo, "
        "Zenodo 17665853 v2, 2025)"
    )
    print("=" * 78)
    print(f"n_primes      = {n_primes}")
    print(f"max_power     = {max_power}")
    print(f"n_zeros       = {n_zeros}")
    print(f"sigma         = {gaussian_sigma}")
    print(f"shift s       = {resolvent_shift}")
    print(f"tolerance     = {weil_tolerance:.0e}")
    print()

    # ------------------------------------------------------------------
    # Section 1 -- per-character certificates
    # ------------------------------------------------------------------

    print("-" * 78)
    print("Section 1: per-character certificates")
    print("-" * 78)

    certs = []
    for chi in characters:
        print()
        print(f"[chi = {chi.name}]")
        print()
        cert = compute_twisted_hilbert_polya_certificate(
            chi,
            n_primes=n_primes,
            max_power=max_power,
            n_zeros=n_zeros,
            gaussian_sigma=gaussian_sigma,
            resolvent_shift=resolvent_shift,
            weil_tolerance=weil_tolerance,
        )
        certs.append(cert)
        _print_certificate(cert)

    # ------------------------------------------------------------------
    # Section 2 -- comparison table
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Section 2: comparison table across primitive real characters")
    print("-" * 78)
    print()
    _print_comparison_table(certs)

    # ------------------------------------------------------------------
    # Section 3 -- honest-scope reminder
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Section 3: honest-scope reminder")
    print("-" * 78)
    notes = certs[0].notes
    for line in notes:
        print(f"  - {line}")
    print()
    print("  Open structural piece: derive T_HP^(chi) on chi-twisted TNFR")
    print("  Hilbert space from the nodal equation, conservation, and")
    print("  grammar WITHOUT inputting the L-zeros.  P45 establishes the")
    print("  operator-level slot such a derivation must fill; it does")
    print("  not fill it.  P45 is the L-track structural mirror of P27.")
    print()

    ok = all(c.scaffold_consistent for c in certs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
