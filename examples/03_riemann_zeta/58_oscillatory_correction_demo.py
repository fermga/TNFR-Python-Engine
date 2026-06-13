"""P31 — Prime-Ladder Oscillatory Correction Demo (branch B1 retry).

Reproduces the canonical TNFR reconstruction of the oscillatory
remainder S(T) = pi^{-1} arg zeta(1/2 + iT) from the prime-ladder
spectrum {(k log p, log p)} (P12 / P14 canonical) and applies it as
a position-level correction to the P28 smooth targets {tilde gamma_i}.

Compares the corrected W_1 gap to the true Riemann zeros against the
P30 smooth-half baseline.  Reports honestly: positive improvement is
**branch B1 evidence** (the existing canonical operator catalog plus
prime-ladder data suffices to reduce the S(T) residual); negative
improvement corroborates **branch B2** (a new canonical operator is
required).

This demo does NOT close gap G4 = RH under any interpretation.
"""

from __future__ import annotations

from tnfr.riemann import compute_oscillatory_correction_certificate


def run(n_targets: int, n_primes: int, max_power: int) -> None:
    cert = compute_oscillatory_correction_certificate(
        n_targets,
        n_primes=n_primes,
        max_power=max_power,
    )
    print()
    print(f"=== N = {n_targets}, primes = {n_primes}, K = {max_power} ===")
    print(cert.summary())
    print()
    print("  damping sweep (damping, W_1):")
    for d, w1 in cert.damping_sweep:
        marker = "  <-- best" if d == cert.best_damping else ""
        if w1 == float("inf"):
            print(f"    d={d:.2f}  W_1=overflow{marker}")
        else:
            print(f"    d={d:.2f}  W_1={w1:.4e}{marker}")


def main() -> None:
    print(
        "P31 — Prime-Ladder Oscillatory Correction"
        " (canonical TNFR retry of branch B1)"
    )
    print(
        "Honest scope: this is an experimental research diagnostic."
    )
    print(
        "Positive improvement => branch B1 evidence."
        "  Negative => branch B2 corroboration."
    )
    print("Neither outcome closes gap G4 = RH.")

    # Match the P30 §13nonies.3 baseline grid (N=20, N=40) for a
    # direct apples-to-apples comparison.
    run(n_targets=20, n_primes=200, max_power=8)
    run(n_targets=40, n_primes=400, max_power=8)


if __name__ == "__main__":
    main()
