"""R∞-1a-operator: Spectrum of the REMESH iteration matrix vs {γ_n}.

Pre-registered milestone gating branch B1 (§13octies) at the *operator* level
of the TNFR-Riemann program. This benchmark closes the open question left by
R∞-1a-spectral and its robustness milestone (§13vicies-novies.6, §13vicies-
novies.7): whether the iterated REMESH operator, considered as a linear map
on the joint (EPI × temporal history) state space, has eigenvalues that
align with the Riemann γ_n.

Construction
------------
The canonical REMESH update (src/tnfr/operators/remesh.py L1212-1252) is
strictly linear and node-local:

    EPI_new(i) = (1-α)² · EPI(i, t)
               + α(1-α) · EPI(i, t-τ_l)
               + α     · EPI(i, t-τ_g)

with no edge term. The full state of node i over a delay window of length
τ_g + 1 evolves under a shift-augmented matrix M of dimension (τ_g+1)
× (τ_g+1):

    M[0, 0]    = (1-α)²
    M[0, τ_l]  = α(1-α)
    M[0, τ_g]  = α
    M[k, k-1]  = 1     for k = 1, ..., τ_g

Because there is no inter-node coupling, the full-graph iteration operator
is block-diagonal: N copies of the same matrix M. The full spectrum is
therefore the spectrum of M with multiplicity N. The graph topology and the
P14 prime-ladder initial condition do *not* enter M at all.

Pre-registration (F5)
---------------------
* H0 (refutation of operator-level B1):
      max |correlation| (Pearson or Spearman) between any natural ordering
      of the 16 non-trivial eigenvalues of M and the first 16 Riemann γ_n
      < 0.5, AND no permutation-nulled p_one_sided < 0.05.
* H1 (operator-level support):
      at least one ordering yields |correlation| ≥ 0.5 with
      p_one_sided < 0.05 under 5000-permutation null on the γ_n vector.

Outputs are deterministic up to permutation seed (default 20260526).

Result of this milestone is logged in
theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.8.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Riemann zeros via mpmath (high-precision).
import mpmath as mp

mp.mp.dps = 30


# -- canonical REMESH parameters (must match R∞-1a-spectral baseline) ----
ALPHA: float = 0.5
TAU_LOCAL: int = 4
TAU_GLOBAL: int = 16

PERM_SEED: int = 20260526
PERM_N: int = 5000

# pre-registered thresholds (F5)
CORR_THRESHOLD: float = 0.5
P_THRESHOLD: float = 0.05


def build_remesh_iteration_matrix(alpha: float, tau_l: int, tau_g: int) -> np.ndarray:
    """Construct the (τ_g+1)×(τ_g+1) shift-augmented REMESH update matrix."""
    if tau_l < 1 or tau_g < 1 or tau_l > tau_g:
        raise ValueError(f"require 1 <= tau_l <= tau_g, got tau_l={tau_l}, tau_g={tau_g}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"require 0 < alpha < 1, got alpha={alpha}")
    dim = tau_g + 1
    M = np.zeros((dim, dim), dtype=np.float64)
    M[0, 0] = (1.0 - alpha) ** 2
    M[0, tau_l] = alpha * (1.0 - alpha)
    M[0, tau_g] = alpha
    for k in range(1, dim):
        M[k, k - 1] = 1.0
    return M


def compute_riemann_zeros(n: int) -> np.ndarray:
    """First n positive imaginary parts of non-trivial zeros of ζ(s)."""
    zeros = np.array([float(mp.im(mp.zetazero(k))) for k in range(1, n + 1)])
    return zeros


@dataclass
class CorrelationTest:
    """One pre-registered correlation between an eigenvalue ordering and γ_n."""

    ordering_name: str
    statistic_name: str  # 'pearson' or 'spearman'
    r: float
    p_two_sided_analytical: float
    p_one_sided_permutation: float

    def passes_significance(self, threshold: float = CORR_THRESHOLD,
                            p_thresh: float = P_THRESHOLD) -> bool:
        return abs(self.r) >= threshold and self.p_one_sided_permutation < p_thresh


def permutation_p_one_sided(observed_r: float, x: np.ndarray, y: np.ndarray,
                            statistic: str, rng: np.random.Generator,
                            n_perm: int = PERM_N) -> float:
    """One-sided p-value: P(|r_perm| >= |r_obs|) under random reordering of y."""
    abs_obs = abs(observed_r)
    count = 0
    y_perm = y.copy()
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if statistic == "pearson":
            r_p, _ = pearsonr(x, y_perm)
        elif statistic == "spearman":
            r_p, _ = spearmanr(x, y_perm)
        else:
            raise ValueError(statistic)
        if abs(r_p) >= abs_obs:
            count += 1
    # Add-one smoothing (avoid p=0 from finite resampling).
    return (count + 1) / (n_perm + 1)


def run_correlation_battery(eigvals: np.ndarray, gammas: np.ndarray,
                            rng: np.random.Generator) -> list[CorrelationTest]:
    """Compute the pre-registered set of correlations between eigenvalue
    orderings and γ_n.

    Orderings tested (each yields a 1-D sequence aligned to γ_1..γ_n):
      * abs_desc  : |λ| sorted descending
      * abs_asc   : |λ| sorted ascending
      * arg_pos   : arg(λ) for eigenvalues with Im(λ) >= 0 (8 values), sorted
                    ascending; replicated and matched to first 8 γ_n
      * real_part : Re(λ) sorted descending
      * imag_part : Im(λ) for eigenvalues with Im(λ) >= 0, sorted ascending
    """
    tests: list[CorrelationTest] = []
    n_total = len(gammas)

    # ordering 1+2: |lambda|
    abs_eigs = np.abs(eigvals)
    abs_desc = np.sort(abs_eigs)[::-1]
    abs_asc = np.sort(abs_eigs)
    for name, seq in (("abs_desc", abs_desc), ("abs_asc", abs_asc)):
        n = min(len(seq), n_total)
        x = seq[:n].astype(np.float64)
        y = gammas[:n].astype(np.float64)
        for stat_name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
            r, p_an = fn(x, y)
            p_perm = permutation_p_one_sided(r, x, y, stat_name, rng)
            tests.append(CorrelationTest(name, stat_name, float(r),
                                         float(p_an), float(p_perm)))

    # ordering 3: arg(lambda) for upper half plane
    upper = eigvals[np.imag(eigvals) >= 1e-12]
    args = np.angle(upper)
    arg_sorted = np.sort(args)
    n = min(len(arg_sorted), n_total)
    x = arg_sorted[:n].astype(np.float64)
    y = gammas[:n].astype(np.float64)
    for stat_name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
        r, p_an = fn(x, y)
        p_perm = permutation_p_one_sided(r, x, y, stat_name, rng)
        tests.append(CorrelationTest("arg_upper_asc", stat_name, float(r),
                                     float(p_an), float(p_perm)))

    # ordering 4: Re(lambda) sorted descending
    re_desc = np.sort(np.real(eigvals))[::-1]
    n = min(len(re_desc), n_total)
    x = re_desc[:n].astype(np.float64)
    y = gammas[:n].astype(np.float64)
    for stat_name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
        r, p_an = fn(x, y)
        p_perm = permutation_p_one_sided(r, x, y, stat_name, rng)
        tests.append(CorrelationTest("real_desc", stat_name, float(r),
                                     float(p_an), float(p_perm)))

    # ordering 5: Im(lambda) for upper half plane, ascending
    im_upper = np.imag(eigvals[np.imag(eigvals) >= 1e-12])
    im_sorted = np.sort(im_upper)
    n = min(len(im_sorted), n_total)
    x = im_sorted[:n].astype(np.float64)
    y = gammas[:n].astype(np.float64)
    for stat_name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
        r, p_an = fn(x, y)
        p_perm = permutation_p_one_sided(r, x, y, stat_name, rng)
        tests.append(CorrelationTest("imag_upper_asc", stat_name, float(r),
                                     float(p_an), float(p_perm)))

    return tests


def run_sensitivity_sweep(alpha_grid: list[float], tau_l_grid: list[int],
                          tau_g: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Re-run the correlation battery on every (α, τ_l) cell. Returns one
    record per cell with the maximum |r| achieved and its associated p_perm."""
    records: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        for tau_l in tau_l_grid:
            M = build_remesh_iteration_matrix(alpha, tau_l, tau_g)
            eigvals, _ = eig(M)
            # remove the trivial λ = 1 mode (within tolerance)
            mask = np.abs(eigvals - 1.0) > 1e-9
            nontrivial = eigvals[mask]
            n_nontrivial = len(nontrivial)
            gammas = compute_riemann_zeros(n_nontrivial)
            tests = run_correlation_battery(nontrivial, gammas, rng)
            best = max(tests, key=lambda t: abs(t.r))
            records.append({
                "alpha": alpha,
                "tau_l": tau_l,
                "tau_g": tau_g,
                "n_nontrivial_eigvals": n_nontrivial,
                "best_ordering": best.ordering_name,
                "best_statistic": best.statistic_name,
                "best_r": best.r,
                "best_p_perm": best.p_one_sided_permutation,
                "best_passes": best.passes_significance(),
            })
    return records


def run_monotonicity_controls(gammas: np.ndarray,
                              rng: np.random.Generator) -> list[dict[str, Any]]:
    """Compute the same Pearson/Spearman + permutation null on control sequences
    that are monotone but carry no Riemann content. Purpose: expose the design
    flaw whereby comparing two sorted sequences yields trivially high r and a
    misleadingly low permutation p-value (since the permutation null is the
    distribution of correlations between a sorted vector and a shuffled vector,
    and shuffling almost always breaks monotonicity).

    A canonical-config "PASS" that is *matched* by every control is uninformative
    and must be reported as a kernel artefact rather than as evidence for B1.
    """
    n = len(gammas)
    controls: list[tuple[str, np.ndarray]] = [
        ("integer_ladder", np.arange(1, n + 1, dtype=np.float64)),
        ("arithmetic_decay", np.linspace(0.98, 0.94, n)),
        ("random_monotone_in_unit_disk",
            np.sort(rng.uniform(0.9, 1.0, n))),
        ("log_n_growth", np.log1p(np.arange(1, n + 1, dtype=np.float64))),
    ]
    records: list[dict[str, Any]] = []
    for name, x in controls:
        for stat_name, fn in (("pearson", pearsonr), ("spearman", spearmanr)):
            r, p_an = fn(x, gammas)
            p_perm = permutation_p_one_sided(r, x, gammas, stat_name, rng)
            records.append({
                "control_name": name,
                "statistic_name": stat_name,
                "r": float(r),
                "p_two_sided_analytical": float(p_an),
                "p_one_sided_permutation": float(p_perm),
                "passes_naive_F5": (abs(r) >= CORR_THRESHOLD
                                     and p_perm < P_THRESHOLD),
            })
    return records


def run_milestone(out_path: Path, perm_n: int = PERM_N) -> dict[str, Any]:
    """Execute R∞-1a-operator and write the JSON report."""
    rng = np.random.default_rng(PERM_SEED)

    # canonical configuration
    M = build_remesh_iteration_matrix(ALPHA, TAU_LOCAL, TAU_GLOBAL)
    eigvals, _ = eig(M)
    eigvals_sorted_by_abs = eigvals[np.argsort(-np.abs(eigvals))]

    # remove the trivial fixed-point mode
    mask = np.abs(eigvals - 1.0) > 1e-9
    nontrivial = eigvals[mask]
    n_nontrivial = len(nontrivial)

    gammas = compute_riemann_zeros(n_nontrivial)
    tests = run_correlation_battery(nontrivial, gammas, rng)
    best = max(tests, key=lambda t: abs(t.r))

    # F5 verdict (naive, pre-registered)
    any_passes = any(t.passes_significance() for t in tests)
    verdict = "INDETERMINATE_OR_SUPPORTED" if any_passes else "REFUTED"

    # Monotonicity controls: same tests on monotone non-Riemann sequences.
    # If controls also "pass", the canonical PASS is a kernel artefact.
    control_records = run_monotonicity_controls(gammas, rng)
    n_control_pass = sum(1 for c in control_records if c["passes_naive_F5"])
    n_control_total = len(control_records)
    controls_invalidate = n_control_pass >= max(1, n_control_total // 2)

    # F5_strict: a PASS counts as evidence only if controls do not also pass.
    if controls_invalidate and verdict == "INDETERMINATE_OR_SUPPORTED":
        verdict_strict = "REFUTED_BY_MONOTONICITY_ARTEFACT"
    elif verdict == "INDETERMINATE_OR_SUPPORTED":
        verdict_strict = "INDETERMINATE_OR_SUPPORTED"
    else:
        verdict_strict = "REFUTED"

    # sensitivity sweep (mirrors R∞-1a-spectral-robustness C2)
    sweep_records = run_sensitivity_sweep(
        alpha_grid=[0.25, 0.5, 0.75],
        tau_l_grid=[2, 4, 8],
        tau_g=TAU_GLOBAL,
        rng=rng,
    )
    any_sweep_passes = any(rec["best_passes"] for rec in sweep_records)

    report: dict[str, Any] = {
        "milestone": "R-inf-1a-operator",
        "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.8",
        "canonical_config": {
            "alpha": ALPHA,
            "tau_l": TAU_LOCAL,
            "tau_g": TAU_GLOBAL,
            "matrix_dim": TAU_GLOBAL + 1,
        },
        "permutation_seed": PERM_SEED,
        "permutation_n": perm_n,
        "preregistration": {
            "corr_threshold": CORR_THRESHOLD,
            "p_threshold": P_THRESHOLD,
            "H0_refute_operator_B1": (
                "no ordering achieves |r| >= 0.5 with p_perm < 0.05"
            ),
            "H1_support_operator_B1": (
                "some ordering achieves |r| >= 0.5 with p_perm < 0.05"
            ),
        },
        "structural_observations": {
            "operator_is_block_diagonal_in_nodes": True,
            "spectrum_independent_of_graph_topology": True,
            "spectrum_independent_of_prime_ladder_initial_state": True,
            "trivial_eigenvalue_lambda_1_present": True,
            "n_nontrivial_eigenvalues": int(n_nontrivial),
            "spectral_radius_excluding_unity": float(
                np.abs(nontrivial).max()
            ),
        },
        "canonical_spectrum": [
            {"real": float(z.real), "imag": float(z.imag),
             "abs": float(abs(z)), "arg": float(np.angle(z))}
            for z in eigvals_sorted_by_abs
        ],
        "first_n_riemann_gammas": [float(g) for g in gammas],
        "correlation_battery": [asdict(t) for t in tests],
        "best_test": asdict(best),
        "f5_verdict_canonical": verdict,
        "monotonicity_controls": control_records,
        "n_controls_passing_naive_F5": int(n_control_pass),
        "n_controls_total": int(n_control_total),
        "controls_invalidate_canonical_pass": bool(controls_invalidate),
        "f5_verdict_strict": verdict_strict,
        "sensitivity_sweep": sweep_records,
        "any_sweep_cell_passes": any_sweep_passes,
        "structural_b1_operator_level_refuted": True,
        "structural_b1_refutation_basis": (
            "The REMESH iteration matrix M is independent of graph topology "
            "and of the P14 prime-ladder initial state by construction. "
            "Therefore its spectrum cannot encode {gamma_n}-specific content "
            "carried by the prime ladder. Any apparent alignment is either "
            "(a) a monotonicity-induced kernel artefact (see "
            "monotonicity_controls), or (b) imposed by the choice of γ_n as "
            "the comparison target rather than discovered from the operator."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _print_summary(report: dict[str, Any]) -> None:
    obs = report["structural_observations"]
    best = report["best_test"]
    print("=" * 70)
    print("R-inf-1a-operator  (REMESH iteration matrix spectrum vs {gamma_n})")
    print("=" * 70)
    cfg = report["canonical_config"]
    print(f"Canonical config: alpha={cfg['alpha']}, tau_l={cfg['tau_l']}, "
          f"tau_g={cfg['tau_g']}, dim(M)={cfg['matrix_dim']}")
    print()
    print("Structural observations:")
    print(f"  operator block-diagonal in nodes              : "
          f"{obs['operator_is_block_diagonal_in_nodes']}")
    print(f"  spectrum independent of graph topology        : "
          f"{obs['spectrum_independent_of_graph_topology']}")
    print(f"  spectrum independent of P14 initial condition : "
          f"{obs['spectrum_independent_of_prime_ladder_initial_state']}")
    print(f"  number of non-trivial eigenvalues             : "
          f"{obs['n_nontrivial_eigenvalues']}")
    print(f"  spectral radius (excluding lambda=1)          : "
          f"{obs['spectral_radius_excluding_unity']:.6f}")
    print()
    print("Correlation battery on canonical config:")
    print(f"  {'ordering':18s} {'stat':10s} {'r':>9s} {'p_perm':>9s}  passes")
    for t in report["correlation_battery"]:
        passes = (abs(t["r"]) >= CORR_THRESHOLD
                  and t["p_one_sided_permutation"] < P_THRESHOLD)
        print(f"  {t['ordering_name']:18s} {t['statistic_name']:10s} "
              f"{t['r']:+9.4f} {t['p_one_sided_permutation']:9.4f}  "
              f"{'YES' if passes else 'no'}")
    print()
    print(f"Best test: {best['ordering_name']} ({best['statistic_name']}) "
          f"r={best['r']:+.4f}  p_perm={best['p_one_sided_permutation']:.4f}")
    print(f"F5 verdict (naive, canonical): {report['f5_verdict_canonical']}")
    print()
    print("Monotonicity controls (sorted non-Riemann sequences vs gammas):")
    print(f"  {'control':32s} {'stat':10s} {'r':>9s} {'p_perm':>9s}  passes_naive")
    for c in report["monotonicity_controls"]:
        print(f"  {c['control_name']:32s} {c['statistic_name']:10s} "
              f"{c['r']:+9.4f} {c['p_one_sided_permutation']:9.4f}  "
              f"{'YES' if c['passes_naive_F5'] else 'no'}")
    print(f"  -> {report['n_controls_passing_naive_F5']}/{report['n_controls_total']} "
          f"controls pass naive F5; controls_invalidate_canonical_pass = "
          f"{report['controls_invalidate_canonical_pass']}")
    print()
    print(f"F5 STRICT verdict: {report['f5_verdict_strict']}")
    print()
    print("Structural B1-operator verdict (independent of any statistic):")
    print(f"  refuted = {report['structural_b1_operator_level_refuted']}")
    print(f"  basis   = {report['structural_b1_refutation_basis']}")
    print()
    print(f"Sensitivity sweep ({len(report['sensitivity_sweep'])} cells): "
          f"any cell passes = {report['any_sweep_cell_passes']}")
    for rec in report["sensitivity_sweep"]:
        print(f"  alpha={rec['alpha']}, tau_l={rec['tau_l']}: "
              f"best |r|={abs(rec['best_r']):.4f} ({rec['best_ordering']}, "
              f"{rec['best_statistic']}, p={rec['best_p_perm']:.4f})  "
              f"{'PASS' if rec['best_passes'] else 'fail'}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "remesh_infinity"
                / "remesh_infinity_riemann_operator.json",
    )
    parser.add_argument("--perm-n", type=int, default=PERM_N)
    args = parser.parse_args()

    report = run_milestone(args.out, perm_n=args.perm_n)
    _print_summary(report)
    print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
