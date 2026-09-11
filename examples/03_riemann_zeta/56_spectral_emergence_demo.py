"""Example 56 — Spectral emergence under canonical UM+RA coupling (P29).

Runs the P29 exploratory experiment: sweeps three canonical TNFR
inter-prime coupling laws on the P14 prime-ladder Hamiltonian and
reports the Kolmogorov-Smirnov distance of the unfolded
nearest-neighbour spacing distribution to the GUE Wigner surmise
(the conjectural universality class of the Riemann zeros).

Honest scope (AGENTS.md sec. 13.2):

    * Convergence to GUE under a canonical coupling law would
      constitute structural-compatibility evidence.
    * Absence of such convergence documents a concrete computational
      obstruction.
    * Either outcome leaves gap G4 = RH OPEN.

Usage::

    $env:PYTHONPATH=(Resolve-Path ./src).Path
    python examples/03_riemann_zeta/56_spectral_emergence_demo.py
"""

from __future__ import annotations

from tnfr.riemann.spectral_emergence import (
    CANONICAL_COUPLING_LAWS,
    compute_spectral_emergence_report,
)


def main() -> None:
    n_primes = 25
    max_power = 4
    strengths = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

    print("=" * 78)
    print("TNFR-Riemann P29: Spectral emergence under canonical coupling")
    print("=" * 78)
    print(f"Prime ladder:    n_primes = {n_primes}, max_power = {max_power}")
    print(f"Hilbert dim:     N = {n_primes * max_power}")
    print(f"Strength grid:   {strengths}")
    print(f"Canonical laws:  {CANONICAL_COUPLING_LAWS}")
    print()

    reports = compute_spectral_emergence_report(
        n_primes=n_primes,
        max_power=max_power,
        laws=CANONICAL_COUPLING_LAWS,
        strengths=strengths,
    )

    for law, report in reports.items():
        print("-" * 78)
        print(f"Law: {law}")
        print("-" * 78)
        header = (
            f"{'strength':>10} {'||J||_F':>12} {'KS_GUE':>10} "
            f"{'KS_Poisson':>12} {'eig_range':>22}"
        )
        print(header)
        for i, s in enumerate(report.strengths):
            eig_lo, eig_hi = report.eigenvalue_range[i]
            print(
                f"{s:>10.4g} {report.coupling_frobenius[i]:>12.4g} "
                f"{report.ks_to_gue[i]:>10.4f} "
                f"{report.ks_to_poisson[i]:>12.4f} "
                f"[{eig_lo:>8.3g}, {eig_hi:>8.3g}]"
            )
        print()
        print(
            f"  Best KS_GUE:                {report.best_ks_to_gue:.4f} "
            f"at strength = {report.best_strength_gue:.4g}"
        )
        print(
            f"  Baseline (decoupled) KS_GUE: " f"{report.poisson_baseline_ks_gue:.4f}"
        )
        improvement = report.poisson_baseline_ks_gue - report.best_ks_to_gue
        rel = improvement / max(report.poisson_baseline_ks_gue, 1e-12)
        print(
            f"  Absolute improvement:        {improvement:+.4f} " f"({rel * 100:+.1f}%)"
        )
        for note in report.notes:
            print(f"  Note: {note}")
        print()

    print("=" * 78)
    print("Cross-law summary (best KS to GUE)")
    print("=" * 78)
    best = sorted(reports.items(), key=lambda kv: kv[1].best_ks_to_gue)
    for law, report in best:
        print(
            f"  {law:>20s}: KS_GUE_min = {report.best_ks_to_gue:.4f} "
            f"@ strength = {report.best_strength_gue:.4g}"
        )
    print()
    print("Reminder: KS_GUE -> 0 is structural-compatibility evidence only;")
    print("gap G4 = RH remains OPEN regardless of the outcome above.")


if __name__ == "__main__":
    main()
