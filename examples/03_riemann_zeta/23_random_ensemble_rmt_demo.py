#!/usr/bin/env python3
r"""Example 46: P6 Random Prime-Graph Ensembles and RMT Statistics.

Demonstrates the random matrix theory (RMT) analysis of TNFR operators
on randomised prime-graph topologies.

The deterministic prime-path operator H^(k)(sigma) = L_k + V_sigma has a
fixed eigenvalue spectrum.  Comparing a single spectrum against GUE/GOE is
a category error.  The correct RMT context arises from **ensemble averages**
over randomised graph topologies:

1. Fix prime labels {p_1, ..., p_k}.
2. Randomise edges (Erdos-Renyi) or add Wigner noise.
3. Average eigenvalue spacings over the ensemble.
4. Compare with GOE/GUE/Poisson universal distributions.

This is where Tao-Vu type universality would legitimately apply.

TNFR physics basis: Dissonance (OZ operator) introduces controlled
stochastic perturbation.  Ensemble averaging tests the statistical
mechanics of TNFR networks under random perturbation, connecting to
U2 (convergence/boundedness).

Usage:
    python examples/03_riemann_zeta/23_random_ensemble_rmt_demo.py
"""

from __future__ import annotations

from tnfr.riemann.random_ensemble import (
    EnsembleConfig,
    goe_wigner_surmise,
    gue_wigner_surmise,
    poisson_spacing_pdf,
    generate_er_ensemble,
    generate_wigner_ensemble,
    compute_ensemble_spacings,
    compute_mean_spacing_ratio,
    compute_level_repulsion_exponent,
    ks_test_vs_reference,
    classify_ensemble,
    run_rmt_ensemble_analysis,
    rmt_convergence_study,
    GOE_MEAN_RATIO,
    GUE_MEAN_RATIO,
    POISSON_MEAN_RATIO,
)
from tnfr.riemann.random_ensemble import _compute_spacing_stats
from tnfr.mathematics.unified_numerical import np


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main() -> None:
    print("P6: Random Prime-Graph Ensembles and RMT Statistics")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Reference distributions
    # ------------------------------------------------------------------
    section("1. RMT Reference Distributions")
    s_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    print("  s      GOE         GUE         Poisson")
    print("  ----   ---------   ---------   ---------")
    for s in s_grid:
        g = float(goe_wigner_surmise(s))
        u = float(gue_wigner_surmise(s))
        p = float(poisson_spacing_pdf(s))
        print(f"  {s:.1f}    {g:9.5f}   {u:9.5f}   {p:9.5f}")

    print(f"\nReference <r> values:")
    print(f"  GOE:     {GOE_MEAN_RATIO:.4f}")
    print(f"  GUE:     {GUE_MEAN_RATIO:.4f}")
    print(f"  Poisson: {POISSON_MEAN_RATIO:.4f}")

    # ------------------------------------------------------------------
    # 2. Erdos-Renyi ensemble
    # ------------------------------------------------------------------
    section("2. Erdos-Renyi Ensemble (k=20, 50 samples)")
    er_samples = generate_er_ensemble(20, 50, edge_prob=0.4, seed=42)
    er_spacings = compute_ensemble_spacings(er_samples)
    er_r = compute_mean_spacing_ratio(er_spacings)
    er_beta = compute_level_repulsion_exponent(er_spacings)

    print(f"  Total spacings collected: {len(er_spacings)}")
    print(f"  Mean spacing ratio <r>:  {er_r:.4f}")
    print(f"  Level repulsion beta:    {er_beta:.3f}")

    ks_goe = ks_test_vs_reference(er_spacings, "GOE")
    ks_gue = ks_test_vs_reference(er_spacings, "GUE")
    ks_poi = ks_test_vs_reference(er_spacings, "Poisson")
    print(f"\n  KS distances:")
    print(f"    vs GOE:     {ks_goe:.4f}")
    print(f"    vs GUE:     {ks_gue:.4f}")
    print(f"    vs Poisson: {ks_poi:.4f}")

    # ------------------------------------------------------------------
    # 3. Wigner perturbation ensemble
    # ------------------------------------------------------------------
    section("3. Wigner Perturbation Ensemble (k=20, 50 samples)")
    wig_samples = generate_wigner_ensemble(20, 50, wigner_scale=0.5, seed=42)
    wig_spacings = compute_ensemble_spacings(wig_samples)
    wig_r = compute_mean_spacing_ratio(wig_spacings)
    wig_beta = compute_level_repulsion_exponent(wig_spacings)

    print(f"  Total spacings collected: {len(wig_spacings)}")
    print(f"  Mean spacing ratio <r>:  {wig_r:.4f}")
    print(f"  Level repulsion beta:    {wig_beta:.3f}")

    ks_goe_w = ks_test_vs_reference(wig_spacings, "GOE")
    ks_gue_w = ks_test_vs_reference(wig_spacings, "GUE")
    ks_poi_w = ks_test_vs_reference(wig_spacings, "Poisson")
    print(f"\n  KS distances:")
    print(f"    vs GOE:     {ks_goe_w:.4f}")
    print(f"    vs GUE:     {ks_gue_w:.4f}")
    print(f"    vs Poisson: {ks_poi_w:.4f}")

    # ------------------------------------------------------------------
    # 4. Full RMT classification
    # ------------------------------------------------------------------
    section("4. Full RMT Classification")
    analysis = run_rmt_ensemble_analysis(
        k=25, n_samples=60, ensemble_type="erdos_renyi",
        edge_prob=0.35, seed=42, compute_long_range=True,
    )
    rmt = analysis.rmt_comparison
    print(f"  Ensemble: ER, k={analysis.config.k}, "
          f"n={analysis.config.n_samples}, p={analysis.config.edge_prob}")
    print(f"\n  KS classification:    {rmt.best_match}")
    print(f"  Ratio classification: {rmt.ratio_best_match}")
    print(f"\n  KS statistics:")
    print(f"    GOE:     {rmt.ks_goe:.4f}")
    print(f"    GUE:     {rmt.ks_gue:.4f}")
    print(f"    Poisson: {rmt.ks_poisson:.4f}")
    print(f"\n  Ratio distances from reference <r>:")
    print(f"    GOE:     {rmt.ratio_distance_goe:.4f}")
    print(f"    GUE:     {rmt.ratio_distance_gue:.4f}")
    print(f"    Poisson: {rmt.ratio_distance_poisson:.4f}")

    if analysis.number_variance is not None:
        print(f"\n  Number variance (sample L values):")
        nv_L = analysis.number_variance_L
        nv = analysis.number_variance
        for i in [0, len(nv_L) // 4, len(nv_L) // 2, -1]:
            print(f"    Sigma^2(L={nv_L[i]:.2f}) = {nv[i]:.4f}")

    if analysis.spectral_rigidity is not None:
        print(f"\n  Spectral rigidity (sample L values):")
        sr_L = analysis.spectral_rigidity_L
        sr = analysis.spectral_rigidity
        for i in [0, len(sr_L) // 4, len(sr_L) // 2, -1]:
            print(f"    Delta_3(L={sr_L[i]:.2f}) = {sr[i]:.4f}")

    # ------------------------------------------------------------------
    # 5. Wigner scale sweep
    # ------------------------------------------------------------------
    section("5. Wigner Scale Sweep (k=20)")
    print("  scale     <r>      beta    KS(GOE)  KS(Poisson)")
    print("  -------  ------   ------   -------  ----------")
    for scale in [0.01, 0.1, 0.5, 1.0, 2.0]:
        w_analysis = run_rmt_ensemble_analysis(
            k=20, n_samples=40, ensemble_type="wigner",
            wigner_scale=scale, seed=42, compute_long_range=False,
        )
        sp = w_analysis.spacing_stats
        rm = w_analysis.rmt_comparison
        print(f"  {scale:7.2f}  {sp.mean_spacing_ratio:6.4f}   "
              f"{sp.level_repulsion_beta:6.3f}   "
              f"{rm.ks_goe:7.4f}  {rm.ks_poisson:10.4f}")

    # ------------------------------------------------------------------
    # 6. k-convergence study
    # ------------------------------------------------------------------
    section("6. Convergence Study (k = [8, 12, 16, 20, 25, 30])")
    k_vals = [8, 12, 16, 20, 25, 30]
    results = rmt_convergence_study(
        k_vals, n_samples=40, edge_prob=0.35, seed=42,
    )
    print("  k    n_spacings   <r>      beta    best(KS)")
    print("  ---  ----------  ------   ------   --------")
    for r in results:
        sp = r.spacing_stats
        rm = r.rmt_comparison
        print(f"  {r.config.k:3d}  {sp.n_spacings:10d}  "
              f"{sp.mean_spacing_ratio:6.4f}   "
              f"{sp.level_repulsion_beta:6.3f}   "
              f"{rm.best_match:8s}")

    # ------------------------------------------------------------------
    # 7. ER edge probability sweep
    # ------------------------------------------------------------------
    section("7. Edge Probability Sweep (k=20)")
    print("  p_edge   <r>      beta    KS(GOE)  best(KS)")
    print("  ------  ------   ------   -------  --------")
    for p_edge in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        er_analysis = run_rmt_ensemble_analysis(
            k=20, n_samples=40, ensemble_type="erdos_renyi",
            edge_prob=p_edge, seed=42, compute_long_range=False,
        )
        sp = er_analysis.spacing_stats
        rm = er_analysis.rmt_comparison
        print(f"  {p_edge:6.2f}  {sp.mean_spacing_ratio:6.4f}   "
              f"{sp.level_repulsion_beta:6.3f}   "
              f"{rm.ks_goe:7.4f}  {rm.best_match:8s}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    section("8. Summary")
    print("  P6 provides the correct RMT context for TNFR operators:")
    print("  - Deterministic operators have fixed spectra (no ensemble)")
    print("  - Randomised topologies enable proper ensemble averaging")
    print("  - Two ensemble types: Erdos-Renyi (random edges) and")
    print("    Wigner (additive GOE noise)")
    print("  - Statistics: spacing ratio <r>, repulsion beta,")
    print("    number variance, spectral rigidity, KS tests")
    print("  - Classification against GOE/GUE/Poisson universality")
    print("  - Convergence studies verify U2 (boundedness)")
    print("\n  TNFR physics: OZ (Dissonance) as stochastic graph")
    print("  perturbation; ensemble averaging = statistical mechanics")
    print("  of structural networks under controlled noise.")


if __name__ == "__main__":
    main()
