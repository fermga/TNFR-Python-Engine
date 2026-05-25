"""P30 — Admissible spectral-rescaling operator demo.

Closes sub-problem (1) of Conjecture T-HP for the SMOOTH half only.

Construction (canonical):
    F_smooth = U_P14 · diag(sqrt(ñ_i / λ_i)) · U_P14^*

where:
    * λ_i are the top-N positive eigenvalues of the canonical P14
      prime-ladder Hamiltonian (positive Hermitian operator built
      from the TNFR internal Hamiltonian on the prime ladder),
    * ñ_i are the P28 smooth zero positions (derived from the
      Riemann-Siegel θ function; canonical archimedean kernel),
    * U_P14 is the unitary eigenbasis of H_P14.

By construction, F_smooth · H_P14 · F_smooth^* has spectrum {ñ_i}
EXACTLY (verified at machine precision). This is the operator-level
lift of the §13sexies (P28) density-level closure of the smooth
zero distribution.

The residual W_1 distance to the true Riemann zeros {γ_n} measures
the oscillatory part S(T) — RH-equivalent and NOT canonical.

This demo:
    1. Builds the smooth half of F_cand at two resolutions.
    2. Reports max |spec - ñ_i| (spectrum match check).
    3. Reports W_1(σ(P14), {γ_n}) vs W_1({ñ_i}, {γ_n}) and the
       improvement ratio.
    4. Sweeps the three canonical oscillatory enrichment modes
       (phi_log, gamma_e, pi_density) and reports the best
       amplitude / improvement honestly. A negative or null
       improvement is the expected structural evidence for
       §13octies branch B2 (new canonical operator needed).

DISCLAIMER
----------
P30 closes sub-problem (1) of Conjecture T-HP for the smooth half
only. It does NOT close:
    * the oscillatory half of T-HP,
    * sub-problem (2) — canonicity from the nodal equation,
    * sub-problem (3) — positivity coincidence with the
      Weil quadratic form,
    * gap G4 = the Riemann Hypothesis itself.
"""

from __future__ import annotations

from tnfr.riemann.admissible_rescaling import (
    compute_admissible_rescaling_certificate,
)


def _run(label: str, n_targets: int) -> None:
    print("=" * 72)
    print(f"  {label}  (n_targets = {n_targets})")
    print("=" * 72)
    cert = compute_admissible_rescaling_certificate(
        n_targets=n_targets,
        p14_n_primes=max(40, n_targets * 2),
        p14_max_power=6,
        oscillatory_mode="phi_log",
    )
    print(cert.summary())
    print()


def _sweep_oscillatory_modes(n_targets: int = 20) -> None:
    print("=" * 72)
    print(f"  Oscillatory enrichment sweep  (n_targets = {n_targets})")
    print("=" * 72)
    for mode in ("phi_log", "gamma_e", "pi_density"):
        cert = compute_admissible_rescaling_certificate(
            n_targets=n_targets,
            p14_n_primes=max(40, n_targets * 2),
            p14_max_power=6,
            oscillatory_mode=mode,
        )
        improv = cert.oscillatory_improvement_over_smooth
        amp = cert.oscillatory_amplitude
        w1 = cert.w1_oscillatory_vs_true
        print(
            f"  mode={mode:<11}  best amp={amp:.4e}  "
            f"W1={w1:.4e}  improv={improv:+.2f}%"
        )
    print()


if __name__ == "__main__":
    _run("Resolution A (fast)", n_targets=20)
    _run("Resolution B (medium)", n_targets=40)
    _sweep_oscillatory_modes(n_targets=20)
    print(
        "Honest scope: smooth half operationally closed; "
        "oscillatory half + canonicity + positivity remain OPEN. "
        "G4 = RH NOT closed."
    )
