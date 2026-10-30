"""Finite character-twisted prime-ladder spectral-spacing comparison.

For the supplied real characters chi_3, chi_4 and chi_5, configured auxiliary
coupling matrices are compared with GUE and Poisson spacing references. A
finite KS statistic proves neither asymptotic universality nor native UM/RA
execution. Failure on this grid excludes no other coupling family. RH and GRH
remain unproved. See theory/TNFR_RIEMANN_RESEARCH_NOTES.md.
"""

from __future__ import annotations

import sys

from tnfr.riemann.dirichlet_l import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
)
from tnfr.riemann.twisted_spectral_emergence import (
    TWISTED_CANONICAL_COUPLING_LAWS,
    compute_twisted_spectral_emergence_report,
)


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    _ensure_utf8_stdout()

    n_primes = 20
    max_power = 3
    strengths = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

    characters = [
        real_character_mod_3(),
        real_character_mod_4(),
        real_character_mod_5(),
    ]

    print("=" * 78)
    print("TNFR-Riemann P47: chi-twisted spectral emergence under coupling")
    print("=" * 78)
    print(f"n_primes:       {n_primes}")
    print(f"max_power:      {max_power}")
    print(f"strength grid:  {strengths}")
    print(f"laws:           {TWISTED_CANONICAL_COUPLING_LAWS}")
    print()

    # Per-character sweeps ------------------------------------------------
    all_reports: dict[str, dict] = {}
    for chi in characters:
        print("=" * 78)
        print(f"chi = {chi.name}  (q = {chi.modulus})")
        print("=" * 78)
        reports = compute_twisted_spectral_emergence_report(
            chi,
            n_primes=n_primes,
            max_power=max_power,
            strengths=strengths,
        )
        all_reports[chi.name] = reports

        for law, r in reports.items():
            print("-" * 78)
            print(f"  Law: {law}")
            print("-" * 78)
            header = (
                f"  {'strength':>10} {'||J||_F':>12} {'KS_GUE':>10} "
                f"{'KS_Poisson':>12} {'eig_range':>22}"
            )
            print(header)
            for i, s in enumerate(r.strengths):
                lo, hi = r.eigenvalue_range[i]
                print(
                    f"  {s:>10.4g} {r.coupling_frobenius[i]:>12.4g} "
                    f"{r.ks_to_gue[i]:>10.4f} "
                    f"{r.ks_to_poisson[i]:>12.4f} "
                    f"[{lo:>8.3g}, {hi:>8.3g}]"
                )
            print()
            print(
                f"    Best KS_GUE:                 {r.best_ks_to_gue:.4f} "
                f"at strength = {r.best_strength_gue:.4g}"
            )
            print(
                f"    Baseline (decoupled) KS_GUE: " f"{r.poisson_baseline_ks_gue:.4f}"
            )
            improvement = r.poisson_baseline_ks_gue - r.best_ks_to_gue
            rel = improvement / max(r.poisson_baseline_ks_gue, 1e-12)
            print(
                f"    Absolute improvement:        {improvement:+.4f} "
                f"({rel * 100:+.1f}%)"
            )
            print()

    # Cross-character cross-law summary -----------------------------------
    print("=" * 78)
    print("Cross-character / cross-law summary (best KS to GUE)")
    print("=" * 78)
    print(
        f"  {'chi':<14s} {'law':<20s} "
        f"{'KS_GUE_min':>12s} {'strength*':>12s} "
        f"{'KS_GUE@0':>12s}"
    )
    for chi_name, reports in all_reports.items():
        for law, r in sorted(reports.items(), key=lambda kv: kv[1].best_ks_to_gue):
            print(
                f"  {chi_name:<14s} {law:<20s} "
                f"{r.best_ks_to_gue:>12.4f} "
                f"{r.best_strength_gue:>12.4g} "
                f"{r.poisson_baseline_ks_gue:>12.4f}"
            )
    print()
    print("Reminder: finite KS comparisons do not prove asymptotic universality.")
    print("Gap G4 = RH and GRH for L(s, chi) remain OPEN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
