"""TNFR Von Mangoldt Zeta — Demonstration (P12).

Following the May 2026 falsification of the simple affine bridge
(Conjecture 10.1, see theory/TNFR_RIEMANN_RESEARCH_NOTES.md §7), the
priority next construction is a TNFR-native spectral object that
reproduces the logarithmic derivative of the Riemann zeta function:

    -ζ'(s)/ζ(s) = Σ_n Λ(n) n^{-s}      (Re s > 1)

This script builds the construction and verifies it numerically.

Construction (prime-ladder spectrum):
    Nodes        : the prime sequence p₁, p₂, ...
    Eigenvalues  : μ_{p,k} = k · log(p)   (REMESH echoes)
    Weights      : w_{p,k} = log(p)       (structural emission)
    TNFR ζ-deriv : Z_TNFR(s) = Σ_{p,k} w_{p,k} · exp(-s · μ_{p,k})
                              = Σ_p log(p) · p^{-s} / (1 - p^{-s})
                              = -ζ'(s)/ζ(s)            (exact identity)

TNFR interpretation:
    * Each prime acts as a node whose structural pulse has magnitude log(p).
    * REMESH (operator #13) generates the k-th harmonic at k · log(p),
      with weight log(p) — the same emission strength replicated across
      scales (operational fractality, U1a/U1b).
    * The Dirichlet sum Σ_n Λ(n) n^{-s} is recovered exactly because
      Λ(n) is supported on prime powers and equals log(p) on each.

See: theory/TNFR_RIEMANN_RESEARCH_NOTES.md §7 (gap analysis) and §8
(this construction).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure stdout can render the mathematical symbols below on any platform
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Allow running directly from the repository without installation
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.riemann import (  # noqa: E402
    build_prime_ladder_spectrum,
    classical_log_zeta_derivative,
    classical_log_zeta_derivative_matched,
    tnfr_log_zeta_derivative,
    verify_von_mangoldt_reproduction,
)


# High-precision reference values of -ζ'(s)/ζ(s), computed from
# the Euler product   Σ_p log(p) · p^{-s} / (1 - p^{-s})
# truncated at primes ≤ 10^7.  These are the limits the
# prime-ladder construction converges to.
KNOWN_VALUES = {
    2.0: 0.5699608931,
    3.0: 0.1648226822,
    4.0: 0.0636697650,
}


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    section("1. Matched-truncation invariant (exact equality)")
    print("Building prime-ladder spectrum (N=50 primes, K=15 echoes).")
    spec = build_prime_ladder_spectrum(50, max_power=15)
    print(f"  primes: {spec.n_primes}, eigenvalues: {spec.size}\n")
    print(f"  {'s':>6} {'Z_TNFR':>22} {'Z_matched':>22} {'|diff|':>10}")
    for s in (1.5, 2.0, 2.5, 3.0, 4.0):
        z_t = tnfr_log_zeta_derivative(spec, s)
        z_m = classical_log_zeta_derivative_matched(
            s, spec.primes.tolist(), spec.max_power
        )
        print(f"  {s:>6.2f} {z_t:>22.15e} {z_m:>22.15e} {abs(z_t - z_m):>10.2e}")
    print(
        "\n  Both methods evaluate exactly the same finite sum;\n"
        "  agreement to ~1e-15 confirms the TNFR construction is an\n"
        "  unambiguous reorganisation of the classical Dirichlet sum."
    )

    section("2. Convergence to -zeta'(s)/zeta(s) as N, K grow")
    print(
        "Comparing Z_TNFR(s) against the classical sieve-based sum\n"
        "  Σ_{prime powers ≤ n_max} log(p) · p^{-ks}\n"
    )
    print(f"  {'N':>5} {'K':>3} {'n_max':>8}", end="")
    for s in (2.0, 3.0, 4.0):
        print(f"   s={s:<4} rel_err", end="")
    print()
    for n_primes, max_power, n_max in [
        (50, 10, 10_000),
        (200, 15, 50_000),
        (500, 20, 200_000),
        (1000, 25, 500_000),
    ]:
        res = verify_von_mangoldt_reproduction(
            [2.0, 3.0, 4.0],
            n_primes=n_primes,
            max_power=max_power,
            n_max_classical=n_max,
        )
        row = f"  {n_primes:>5} {max_power:>3} {n_max:>8}"
        for re in res.rel_error:
            row += f"   {re:>12.2e}"
        print(row)

    section("3. Comparison to known analytic values")
    spec_big = build_prime_ladder_spectrum(2000, max_power=30)
    print(f"Spectrum: {spec_big.n_primes} primes, K={spec_big.max_power}\n")
    print(
        f"  {'s':>6} {'Z_TNFR':>18} {'known -zeta(s)/zeta(s)':>24} {'abs err':>12}"
    )
    for s, ref in sorted(KNOWN_VALUES.items()):
        z_t = tnfr_log_zeta_derivative(spec_big, s)
        print(f"  {s:>6.2f} {z_t:>18.10f} {ref:>22.10f} {abs(z_t - ref):>12.3e}")
    print(
        "\n  The TNFR sum converges to the classical analytic continuation\n"
        "  values in the half-plane Re(s) > 1, as expected from the\n"
        "  Euler-product identity\n"
        "      -ζ'(s)/ζ(s) = Σ_p log(p) · p^{-s} / (1 - p^{-s}).\n"
    )

    section("4. Summary")
    print(
        "Construction is exact at the formal-series level and converges\n"
        "numerically to known reference values.  Open extensions (see\n"
        "research notes §8.5):\n"
        "  * Build an explicit self-adjoint operator whose spectrum is\n"
        "    the prime ladder {k·log(p)} with multiplicity log(p).\n"
        "  * Analytic continuation into the critical strip 0 < Re(s) < 1.\n"
        "  * Identify the zeros of the corresponding completed function\n"
        "    with structural resonances of the TNFR operator.\n"
    )


if __name__ == "__main__":
    main()
