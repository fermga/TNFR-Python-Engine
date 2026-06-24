"""P50 — REMESH-infinity Residue Split of the P31 Oscillatory Correction.

Function-space diagnostic that lifts the N15 REMESH-infinity closure
(theory/REMESH_INFINITY_DERIVATION.md) into the TNFR-Riemann program.

Splits the canonical TNFR prime-ladder reconstruction
S_TNFR(T) = -(1/pi) sum_{(mu, w)} (w/mu) sin(T mu) exp(-mu/2)
into its projections on range(R_infinity) and ker(R_infinity), where
R_infinity is the orthogonal projector onto the N15-resonant Fourier
lattice {2 pi k / lcm(tau_l, tau_g)} at the canonical pair
(tau_l, tau_g) = (4, 8).

Pre-registered structural prediction
------------------------------------
The prime-ladder Fourier support is {k log p : p prime, k >= 1}.  By
Baker's theorem on linear independence of logarithms of algebraic
numbers, this set is disjoint from the rational-multiple-of-pi
lattice that defines the N15-resonant subspace.  Therefore the
canonical reconstruction S_TNFR(T) must lie asymptotically in
ker(R_infinity).

Pre-registered verdicts
-----------------------
* RESIDUE_IN_KER_ONLY     branch B2 evidence at function-space level
* RESIDUE_IN_RANGE_ONLY   refutes P31 as an oscillatory attack
* RESIDUE_MIXED           gauge leak in P30 or boundary artefact

Honest scope
------------
This diagnostic is complementary to the section 13vicies-novies
graph-iteration-matrix tests (which operate on EPI-history state
vectors); P50 operates on a function in H^2(T-axis).  It does NOT
advance G4 = RH, does NOT close T-HP, does NOT promote any new
canonical operator beyond the 13-operator catalog.
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
    print("P50 — REMESH-infinity Residue Split" " (function-space diagnostic)")
    print("Honest scope: structural-compatibility test.  Does NOT " "advance G4 = RH.")
    print(
        "Pre-registered prediction: RESIDUE_IN_KER_ONLY"
        " (Baker's theorem on log p incommensurability)."
    )

    # Two grid resolutions to test the asymptotic stability of the
    # verdict; both span many REMESH-canonical periods (lcm(4, 8) = 8
    # T-units per period).
    run(n_periods=64, n_primes=200, max_power=8)
    run(n_periods=256, n_primes=400, max_power=8)


if __name__ == "__main__":
    main()
