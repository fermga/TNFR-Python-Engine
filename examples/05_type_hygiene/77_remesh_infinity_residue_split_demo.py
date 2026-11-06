"""P50 — finite fixed-delay Fourier split of the P31 signal.

The public ``remesh_infinity`` names are retained for compatibility. This
example evaluates the finite TNFR prime-ladder reconstruction

S_TNFR(T) = -(1/pi) sum_{(mu, w)} (w/mu) sin(T mu) exp(-mu/2)

on a finite uniform grid, then projects its DFT onto the modes fixed by
both delays. Their period is gcd(tau_l, tau_g). The lcm is used only to
align the sample-window length. Off-grid frequencies leak into multiple
DFT bins, so the reported fractions are finite-window measurements.

Pre-registered verdicts
-----------------------
* RESIDUE_IN_KER_ONLY     little energy in the selected fixed-delay bins
* RESIDUE_IN_RANGE_ONLY   little energy in their orthogonal complement
* RESIDUE_MIXED           both finite-window fractions exceed the threshold

Honest scope
------------
This diagnostic is complementary to the section 13vicies-novies
runtime REMESH, which operates on stored EPI-history snapshots. P50 does
not establish a literal tau_g -> infinity limit, identify the analytic
support of S(T), advance G4 = RH, close T-HP, or prove catalog completeness.
"""

from __future__ import annotations

from tnfr.riemann import compute_residue_split_certificate


def run(n_periods: int, n_primes: int, max_power: int) -> None:
    cert = compute_residue_split_certificate(
        n_primes=n_primes,
        max_power=max_power,
        n_periods=n_periods,
    )
    print()
    print(
        f"=== n_periods = {n_periods}  (n_samples = "
        f"{cert.n_samples}), primes = {n_primes}, K = {max_power} ==="
    )
    print(cert.summary())


def main() -> None:
    print("P50 — Fixed-Delay Fourier Split" " (finite-window diagnostic)")
    print("Honest scope: DFT-bin measurement. Does NOT advance G4 = RH.")
    print(
        "Legacy verdict RESIDUE_IN_KER_ONLY means little finite-window "
        "energy in the selected fixed-delay bins."
    )

    # Two window lengths expose the sensitivity of finite spectral leakage.
    # lcm(4, 8) = 8 is the backward-compatible sample-alignment unit;
    # gcd(4, 8) = 4 determines the common fixed modes.
    run(n_periods=64, n_primes=200, max_power=8)
    run(n_periods=256, n_primes=400, max_power=8)


if __name__ == "__main__":
    main()
