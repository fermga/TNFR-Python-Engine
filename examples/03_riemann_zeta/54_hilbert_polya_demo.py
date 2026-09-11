"""TNFR-Riemann P27 demo: Hilbert-Polya scaffold.

Builds the reference operator T_HP = diag(gamma_1, ..., gamma_N) on the
truncated TNFR Hilbert space ell^2_N(N) and prints a full certificate of
its internal consistency with the rest of the TNFR-Riemann stack: P14
prime-ladder Hamiltonian, P15 Weil-Guinand explicit formula, and the
operator-level gap G4 quantified by Wasserstein-1 distance between
spec(P14) and spec(T_HP).

HONEST SCOPE
============
This demo does NOT prove the Riemann Hypothesis. T_HP is populated by
inputting the zeros computed via mpmath.zetazero. The scaffold makes
the abstract Hilbert-Polya operator explicit and shows it is internally
consistent with the TNFR stack on truncated Hilbert space. The
structural derivation of T_HP from TNFR first principles (which is what
gap G4 = RH would actually require) remains open.
"""

from __future__ import annotations

import io
import sys

import numpy as np

from tnfr.riemann.hilbert_polya import (
    HilbertPolyaCertificate,
    compute_hilbert_polya_certificate,
    fetch_zero_imaginary_parts,
    structural_gap_p14_vs_hp,
)
from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_hamiltonian

# Force UTF-8 on Windows consoles
if isinstance(sys.stdout, io.TextIOWrapper):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


def _print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_certificate(cert: HilbertPolyaCertificate) -> None:
    print(
        f"  Truncation:                  n_zeros = {cert.n_zeros}, "
        f"n_primes = {cert.n_primes}, K = {cert.max_power}"
    )
    print()
    print("  Self-adjointness:")
    print(f"    asymmetry (Frobenius)       = {cert.asymmetry_frobenius:.3e}")
    print(f"    self_adjoint                = {cert.self_adjoint}")
    print()
    print(f"  Shifted resolvent (shift s = {cert.resolvent_shift:.3f}):")
    print(f"    ||R||_1 (trace)             = {cert.schatten_1_norm:.6e}")
    print(f"    ||R||_2 (Hilbert-Schmidt)   = {cert.schatten_2_norm:.6e}")
    print(f"    ||R||_op                    = {cert.operator_norm_inverse:.6e}")
    print(f"    trace_class                 = {cert.trace_class}")
    print()
    print(f"  Weil-Guinand closure (sigma = {cert.gaussian_sigma:.3f}):")
    print(f"    zero_side (via T_HP)        = {cert.zero_side_via_hp:.10f}")
    print(f"    pole_side                   = {cert.pole_side:.10f}")
    print(f"    archimedean_side            = {cert.archimedean_side:.10f}")
    print(f"    prime_side (via P14)        = {cert.prime_side_via_p14:.10f}")
    print(f"    rhs_total                   = {cert.rhs_total:.10f}")
    print(f"    residual                    = {cert.residual:.3e}")
    print(f"    relative_residual           = {cert.relative_residual:.3e}")
    print(f"    weil_verified               = {cert.weil_verified}")
    print()
    print("  Operator-level gap G4:")
    print(f"    n_compared                  = {cert.spectral_gap_n_compared}")
    print(
        f"    Wasserstein-1(P14, T_HP)    = " f"{cert.spectral_gap_wasserstein_1:.6e}"
    )
    print(f"    asymptotic growth ratio     = " f"{cert.spectral_gap_growth_ratio:.6e}")
    print()
    print(f"  scaffold_consistent           = {cert.scaffold_consistent}")


def main() -> None:
    _print_header("Section 1: Hilbert-Polya scaffold certificate (default parameters)")
    cert = compute_hilbert_polya_certificate(
        n_primes=50,
        max_power=8,
        n_zeros=80,
        gaussian_sigma=8.0,
        resolvent_shift=1.0,
    )
    _print_certificate(cert)
    print()
    print("  Notes:")
    for line in cert.notes:
        print(f"    {line}")

    _print_header(
        "Section 2: spec(P14) vs spec(T_HP) growth diagnostic "
        "(operator-level gap G4)"
    )
    bundle = build_prime_ladder_hamiltonian(n_primes=50, max_power=8, coupling=0.0)
    gammas = fetch_zero_imaginary_parts(80, dps=30)
    gap = structural_gap_p14_vs_hp(bundle, gammas)
    print(f"  n_compared        = {gap['n_compared']}")
    print(f"  spec(P14):  min   = {gap['p14_min']:.6e}")
    print(f"              max   = {gap['p14_max']:.6e}   " f"(grows like log n)")
    print(f"  spec(T_HP): min   = {gap['hp_min']:.6e}")
    print(f"              max   = {gap['hp_max']:.6e}   " f"(grows like 2*pi*n/log n)")
    print(f"  Wasserstein-1     = {gap['wasserstein_1']:.6e}")
    print(f"  growth ratio      = {gap['asymptotic_growth_ratio']:.6e}")
    print()
    p14_pos = np.sort(np.real(bundle.hamiltonian.get_spectrum()[0]))
    p14_pos = p14_pos[p14_pos > 0.0]
    n_show = min(8, len(gammas), len(p14_pos))
    print("  Side-by-side (first 8 entries, sorted ascending):")
    print("    n    spec(P14) = k*log(p)     spec(T_HP) = gamma_n")
    for i in range(n_show):
        print(f"    {i + 1:2d}   {p14_pos[i]:.10f}             " f"{gammas[i]:.10f}")

    _print_header("Section 3: Honest scope (per AGENTS.md sec. 13.2)")
    print(
        "  * T_HP is populated by mpmath.zetazero(n) for n = 1..N.\n"
        "    The zeros are an INPUT, not a TNFR derivation.\n"
        "  * The scaffold therefore certifies INTERNAL CONSISTENCY of\n"
        "    the abstract Hilbert-Polya slot with the TNFR stack\n"
        "    (P14 + P15) on truncated Hilbert space ell^2_N(N).\n"
        "  * Gap G4 = RH remains the single OPEN gap (G1, G2, G3\n"
        "    operationally closed; G5 superseded). The genuinely open\n"
        "    piece is the structural derivation of T_HP from TNFR\n"
        "    first principles, without inputting the zeros. P27\n"
        "    quantifies that gap (Wasserstein-1 distance between\n"
        "    spec(P14) and spec(T_HP)) but does not close it.\n"
    )


if __name__ == "__main__":
    main()
