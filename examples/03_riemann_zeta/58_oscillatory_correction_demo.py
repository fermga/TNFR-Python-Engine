"""Finite prime-ladder correction of classical smooth spectral targets.

The supplied prime-ladder spectrum and a truncated classical oscillatory
approximation define a candidate correction. A damping sweep compares it with
reference zero ordinates. Improvement or failure concerns that finite candidate
and grid; neither proves catalog sufficiency, requires a new canonical operator,
nor proves RH. See theory/TNFR_RIEMANN_RESEARCH_NOTES.md.
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
    print("P31 — Prime-Ladder Oscillatory Correction" " (supplied finite candidate)")
    print("Honest scope: this is an experimental research diagnostic.")
    print(
        "Improvement concerns this finite candidate and grid."
        " Failure does not require a new canonical operator."
    )
    print("Neither outcome closes gap G4 = RH.")

    # Match the P30 §13nonies.3 baseline grid (N=20, N=40) for a
    # direct apples-to-apples comparison.
    run(n_targets=20, n_primes=200, max_power=8)
    run(n_targets=40, n_primes=400, max_power=8)


if __name__ == "__main__":
    main()
