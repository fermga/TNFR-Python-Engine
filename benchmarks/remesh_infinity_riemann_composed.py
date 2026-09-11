"""R-inf-1a-composed: Spectrum of T_composed = S_IL . M_REMESH vs GUE Wigner.

Pre-registered milestone (§13vicies-novies.9) gating branch B1 at the
composed-operator level. Tests whether the canonical composition REMESH . IL
on the P14 prime-ladder graph encodes Riemann-zero content in its spectrum,
under a level-spacing discriminator (F6-A: KS distance vs the Montgomery-
Odlyzko / GUE Wigner surmise).

Construction (slot-major ordering)
----------------------------------
* Graph: canonical P14 prime-ladder
  (src/tnfr/riemann/prime_ladder_hamiltonian.py::build_prime_ladder_graph)
  with n_primes=10, max_power=4, coupling=0. N = 40 nodes = 10 disjoint P_4.
* REMESH delay window: tau_g + 1 = 17 slots.
* Joint state dim: N * (tau_g + 1) = 680.
* Index convention: index = slot * N + node. Then
      M_REMESH    = kron(M, I_N)                  shape (680, 680)
      S_IL        = block_diag((I_N - eta L_G), I_N, ..., I_N)   (slot 0 only)
      T_composed  = S_IL @ M_REMESH

with L_G the unnormalised combinatorial Laplacian of G and eta = 0.3 the
canonical IL phase-locking coefficient.

F6-A statistic (pre-registered, replaces retired F5)
-----------------------------------------------------
1. Remove trivial fixed-point cluster: |lambda - 1| < 1e-9.
2. Project to 1-D: Im(lambda) for upper-half-plane subset (Im >= 1e-12),
   sorted ascending. If projection is empty (real spectrum, e.g. GOE), fall
   back to Re(lambda) ordered ascending. Both fallbacks are documented and
   reported in the JSON.
3. Normalised consecutive spacings: delta_k = (s_{k+1} - s_k) / mean.
4. KS distance D_GUE = sup_x |F_emp(x) - F_GUE(x)| with
       P_GUE(s) = (32/pi^2) s^2 exp(-4 s^2 / pi).

Pre-registered thresholds
-------------------------
* SUPPORTED       : D_composed < 0.15 AND D_composed < D_shuffled - 0.05
* REFUTED         : D_composed > 0.30 OR  D_composed >= D_shuffled - 0.05
* INDETERMINATE   : otherwise

Controls
--------
* N1 GOE (dim 680, symmetric, real spectrum -> Re-projection fallback)
* N2 Poisson (680 uniform points -> spacings of e^{-s} distribution)
* N3 prime-ladder shuffled (primary discriminator: primes permuted across
  the 10 P_4 components; if D_shuffled ~ D_composed, prime content is not
  encoded by the operator)
* N4 REMESH-isolated (re-run of the 16-eigenvalue M matrix; reported as
  diagnostic baseline, expected to be degenerate)

Reference
---------
* D_Riemann: KS distance for the first K_ref = 100 Riemann zero imaginary
  parts via mpmath.zetazero. External anchor.

Seeds & parameters
------------------
* numpy default_rng(20260526) for all stochastic draws.
* mpmath dps = 30.
* REMESH: alpha = 0.5, tau_l = 4, tau_g = 16.
* IL coupling: eta = 0.3.
* Graph: n_primes = 10, max_power = 4, coupling = 0.

Result of this milestone is appended to
theory/TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.9 "Results" block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig, eigvalsh

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mpmath as mp

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_graph

mp.mp.dps = 30


# -- pre-registered canonical parameters ---------------------------------
ALPHA: float = 0.5
TAU_LOCAL: int = 4
TAU_GLOBAL: int = 16
ETA_IL: float = 0.3
N_PRIMES: int = 10
MAX_POWER: int = 4
PERM_SEED: int = 20260526
K_REF_RIEMANN: int = 100

# F6-A thresholds (pre-registered)
D_SUPPORTED_MAX: float = 0.15
D_REFUTED_MIN: float = 0.30
D_SHUFFLE_MARGIN: float = 0.05
TRIVIAL_TOL: float = 1e-9
IM_TOL: float = 1e-12


# -- canonical building blocks -------------------------------------------
def build_remesh_iteration_matrix(alpha: float, tau_l: int, tau_g: int) -> np.ndarray:
    """Canonical REMESH (tau_g+1)x(tau_g+1) shift-augmented matrix."""
    if tau_l < 1 or tau_g < 1 or tau_l > tau_g:
        raise ValueError(f"require 1 <= tau_l <= tau_g")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"require 0 < alpha < 1")
    dim = tau_g + 1
    M = np.zeros((dim, dim), dtype=np.float64)
    M[0, 0] = (1.0 - alpha) ** 2
    M[0, tau_l] = alpha * (1.0 - alpha)
    M[0, tau_g] = alpha
    for k in range(1, dim):
        M[k, k - 1] = 1.0
    return M


def laplacian_of_graph(G) -> np.ndarray:
    """Unnormalised combinatorial Laplacian L_G = D - A, dense, in the node
    ordering returned by `list(G.nodes())`."""
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    L = np.zeros((n, n), dtype=np.float64)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i, j] -= 1.0
        L[j, i] -= 1.0
    for i in range(n):
        L[i, i] = -L[i, :].sum()
    return L


def build_il_smoother(L: np.ndarray, eta: float) -> np.ndarray:
    """S_IL acting on the EPI field at the current time slot: (I - eta L)."""
    n = L.shape[0]
    return np.eye(n) - eta * L


def build_composed_iteration_matrix(
    M: np.ndarray, L: np.ndarray, eta: float
) -> np.ndarray:
    """T_composed = S_IL . M_REMESH in slot-major ordering.

    Index convention: index = slot * N + node. Then
        M_REMESH = kron(M, I_N)
        S_IL     = block_diag( I_N - eta L_G , I_N , ... , I_N )
                   (smoothing acts on slot 0 only; identity on history slots)
    """
    N = L.shape[0]
    dim_slot = M.shape[0]  # tau_g + 1
    I_N = np.eye(N)
    M_REMESH = np.kron(M, I_N)
    S_blocks = [np.eye(N) for _ in range(dim_slot)]
    S_blocks[0] = build_il_smoother(L, eta)
    # Build block-diagonal S_IL as a (dim_slot*N) x (dim_slot*N) matrix.
    S_IL = np.zeros((dim_slot * N, dim_slot * N), dtype=np.float64)
    for s, blk in enumerate(S_blocks):
        S_IL[s * N : (s + 1) * N, s * N : (s + 1) * N] = blk
    return S_IL @ M_REMESH


# -- F6-A: KS distance vs GUE Wigner surmise -----------------------------
def gue_wigner_pdf(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def gue_wigner_cdf(s: np.ndarray) -> np.ndarray:
    """CDF of P_GUE(s) = (32/pi^2) s^2 exp(-4 s^2 / pi). Integrated by
    50-point Gauss-Legendre over [0, s] for each s; vectorised."""
    s = np.asarray(s, dtype=np.float64)
    # Closed-form: integrate by parts twice. The CDF admits the form
    #   F(s) = 1 - (2/pi) * (2 s^2 / pi + 1) * exp(-4 s^2 / pi) * ... ?
    # Simpler: compute by quadrature with high resolution.
    grid = np.linspace(0.0, max(float(s.max()) + 1e-9, 1.0), 4001)
    pdf = gue_wigner_pdf(grid)
    cdf_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]
    )
    cdf_grid = cdf_grid / cdf_grid[-1]  # normalise (handles truncation)
    return np.interp(s, grid, cdf_grid)


def normalised_spacings(values_sorted: np.ndarray) -> np.ndarray:
    """delta_k = (s_{k+1} - s_k) / mean. Returns empty array if < 2 values."""
    if values_sorted.size < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(values_sorted)
    mean = diffs.mean()
    if mean <= 0.0:
        return np.zeros_like(diffs)
    return diffs / mean


def ks_distance_vs_gue(spacings: np.ndarray) -> float:
    """KS sup-distance between empirical CDF of spacings and GUE Wigner CDF."""
    if spacings.size == 0:
        return float("nan")
    sorted_s = np.sort(spacings)
    n = sorted_s.size
    emp_cdf_at_points = np.arange(1, n + 1) / n
    gue_at_points = gue_wigner_cdf(sorted_s)
    # KS sup distance (both upper and lower steps).
    d_plus = float(np.max(emp_cdf_at_points - gue_at_points))
    d_minus = float(np.max(gue_at_points - (np.arange(n) / n)))
    return max(d_plus, d_minus)


def project_spectrum_1d(eigvals: np.ndarray) -> tuple[np.ndarray, str]:
    """Pre-registered projection: Im(lambda) for upper-half-plane subset.
    Fallback: Re(lambda) if no complex eigenvalues are present."""
    # Remove trivial fixed-point cluster around lambda = 1.
    nontrivial = eigvals[np.abs(eigvals - 1.0) > TRIVIAL_TOL]
    upper = nontrivial[np.imag(nontrivial) >= IM_TOL]
    if upper.size >= 2:
        return np.sort(np.imag(upper).astype(np.float64)), "Im_upper"
    # Fallback for real spectra.
    return np.sort(np.real(nontrivial).astype(np.float64)), "Re_fallback"


def f6a_diagnostic(eigvals: np.ndarray, label: str) -> dict[str, Any]:
    projection, projection_kind = project_spectrum_1d(eigvals)
    spacings = normalised_spacings(projection)
    d_gue = ks_distance_vs_gue(spacings)
    return {
        "label": label,
        "projection_kind": projection_kind,
        "n_projected_values": int(projection.size),
        "n_spacings": int(spacings.size),
        "D_GUE": float(d_gue) if np.isfinite(d_gue) else None,
        "spacings_mean": float(spacings.mean()) if spacings.size else None,
        "spacings_std": float(spacings.std()) if spacings.size else None,
    }


# -- controls -------------------------------------------------------------
def control_goe(dim: int, rng: np.random.Generator) -> dict[str, Any]:
    """GOE: A = (X + X.T) / sqrt(2N) with X iid standard normal."""
    X = rng.standard_normal((dim, dim))
    A = (X + X.T) / np.sqrt(2.0 * dim)
    eigvals = eigvalsh(A).astype(np.complex128)  # real spectrum
    return f6a_diagnostic(eigvals, "N1_GOE")


def control_poisson(n_points: int, rng: np.random.Generator) -> dict[str, Any]:
    """Poisson: n_points iid uniform on [0, 1], then sort and compute spacings."""
    pts = np.sort(rng.uniform(0.0, 1.0, n_points))
    spacings = normalised_spacings(pts)
    d_gue = ks_distance_vs_gue(spacings)
    return {
        "label": "N2_Poisson",
        "projection_kind": "uniform_iid",
        "n_projected_values": int(pts.size),
        "n_spacings": int(spacings.size),
        "D_GUE": float(d_gue) if np.isfinite(d_gue) else None,
        "spacings_mean": float(spacings.mean()),
        "spacings_std": float(spacings.std()),
    }


def control_shuffled_prime(M: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Primary discriminator: rebuild P14 with the primes shuffled across the
    10 ladders, then recompute T_composed and F6-A."""
    G = build_prime_ladder_graph(N_PRIMES, max_power=MAX_POWER, coupling=0.0)
    # The P14 graph topology (10 disjoint P_4) is invariant under prime
    # relabelling. We permute the prime labels and rebuild from a permuted
    # explicit prime list to make the symmetry empirically explicit.
    from sympy import primerange

    primes = list(primerange(2, 100))[:N_PRIMES]
    perm = rng.permutation(N_PRIMES)
    primes_shuffled = [primes[i] for i in perm]
    G_shuffled = build_prime_ladder_graph(
        N_PRIMES, max_power=MAX_POWER, coupling=0.0, primes=primes_shuffled
    )
    L_shuffled = laplacian_of_graph(G_shuffled)
    T_shuffled = build_composed_iteration_matrix(M, L_shuffled, ETA_IL)
    eigvals_shuffled, _ = eig(T_shuffled)
    diag = f6a_diagnostic(eigvals_shuffled, "N3_shuffled_prime")
    diag["primes_canonical"] = primes
    diag["primes_shuffled"] = primes_shuffled
    return diag


def control_remesh_isolated(M: np.ndarray) -> dict[str, Any]:
    """N4 baseline: the 17 eigenvalues of M alone (the §13vicies-novies.8
    spectrum). Diagnostic only; spacings expected to be degenerate."""
    eigvals_M, _ = eig(M)
    return f6a_diagnostic(eigvals_M.astype(np.complex128), "N4_REMESH_isolated")


def reference_riemann_d_gue(k_ref: int) -> dict[str, Any]:
    """External anchor: D_GUE for the first k_ref Riemann zeros."""
    gammas = np.array([float(mp.im(mp.zetazero(k))) for k in range(1, k_ref + 1)])
    spacings = normalised_spacings(gammas)
    d_gue = ks_distance_vs_gue(spacings)
    return {
        "label": "Riemann_reference",
        "k_ref": k_ref,
        "n_spacings": int(spacings.size),
        "D_GUE": float(d_gue),
        "spacings_mean": float(spacings.mean()),
        "spacings_std": float(spacings.std()),
    }


# -- milestone driver -----------------------------------------------------
def run_milestone(out_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(PERM_SEED)

    # -- canonical construction
    G = build_prime_ladder_graph(N_PRIMES, max_power=MAX_POWER, coupling=0.0)
    L = laplacian_of_graph(G)
    M = build_remesh_iteration_matrix(ALPHA, TAU_LOCAL, TAU_GLOBAL)
    T = build_composed_iteration_matrix(M, L, ETA_IL)
    N = L.shape[0]
    dim_joint = T.shape[0]
    eigvals_T, _ = eig(T)

    canonical = f6a_diagnostic(eigvals_T, "canonical_REMESH_o_IL")
    canonical["dim_joint"] = int(dim_joint)
    canonical["N_nodes"] = int(N)

    # -- controls
    goe = control_goe(dim_joint, rng)
    poisson = control_poisson(dim_joint, rng)
    shuffled = control_shuffled_prime(M, rng)
    remesh_iso = control_remesh_isolated(M)
    riemann_ref = reference_riemann_d_gue(K_REF_RIEMANN)

    # -- pre-registered verdict
    d_can = canonical["D_GUE"]
    d_shuf = shuffled["D_GUE"]
    if d_can is None or d_shuf is None:
        f6a_verdict = "INDETERMINATE_INVALID_PROJECTION"
    elif d_can < D_SUPPORTED_MAX and d_can < d_shuf - D_SHUFFLE_MARGIN:
        f6a_verdict = "SUPPORTED"
    elif d_can > D_REFUTED_MIN or d_can >= d_shuf - D_SHUFFLE_MARGIN:
        f6a_verdict = "REFUTED"
    else:
        f6a_verdict = "INDETERMINATE"

    # -- structural prediction check
    # The pre-registered structural prediction: the P14 Laplacian L_G has
    # spectrum {0, 0.586, 2.0, 3.414} each with multiplicity 10 (10 disjoint
    # P_4 components). Verify this empirically.
    eig_L = np.sort(eigvalsh(L))
    distinct_L = sorted({round(float(v), 6) for v in eig_L})
    expected_L_distinct = sorted(
        {0.0, round(2.0 - np.sqrt(2.0), 6), round(2.0, 6), round(2.0 + np.sqrt(2.0), 6)}
    )
    structural_check = {
        "L_G_distinct_eigenvalues_empirical": distinct_L,
        "L_G_distinct_eigenvalues_expected_P4": expected_L_distinct,
        "L_G_full_spectrum_multiplicity_per_distinct": {
            str(round(float(v), 6)): int(np.sum(np.isclose(eig_L, v, atol=1e-8)))
            for v in distinct_L
        },
        "matches_10x_P4_prediction": (
            len(distinct_L) == 4
            and all(
                np.isclose(distinct_L[i], expected_L_distinct[i], atol=1e-8)
                for i in range(4)
            )
            and all(
                int(np.sum(np.isclose(eig_L, v, atol=1e-8))) == N_PRIMES
                for v in distinct_L
            )
        ),
    }

    # -- composite milestone verdict
    if f6a_verdict == "REFUTED" and structural_check["matches_10x_P4_prediction"]:
        milestone_verdict = "B1_COMPOSED_REFUTED_FOR_REMESH_o_IL"
    elif f6a_verdict == "SUPPORTED":
        milestone_verdict = (
            "B1_COMPOSED_POTENTIALLY_OPEN_STRUCTURAL_PREDICTION_CHALLENGED"
        )
    elif f6a_verdict == "REFUTED":
        milestone_verdict = (
            "B1_COMPOSED_REFUTED_BUT_STRUCTURAL_PREDICTION_NOT_CONFIRMED"
        )
    else:
        milestone_verdict = f"B1_COMPOSED_{f6a_verdict}"

    report: dict[str, Any] = {
        "milestone": "R-inf-1a-composed",
        "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.9",
        "canonical_config": {
            "graph": "P14 prime-ladder",
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "coupling": 0.0,
            "N_nodes": int(N),
            "alpha": ALPHA,
            "tau_l": TAU_LOCAL,
            "tau_g": TAU_GLOBAL,
            "tau_g_plus_1": TAU_GLOBAL + 1,
            "eta_IL": ETA_IL,
            "dim_joint_state": int(dim_joint),
        },
        "seed": PERM_SEED,
        "preregistration": {
            "statistic": "F6-A KS distance vs GUE Wigner surmise",
            "projection": "Im(lambda) upper-half-plane (fallback Re(lambda))",
            "thresholds": {
                "SUPPORTED": (
                    f"D_composed < {D_SUPPORTED_MAX} AND "
                    f"D_composed < D_shuffled - {D_SHUFFLE_MARGIN}"
                ),
                "REFUTED": (
                    f"D_composed > {D_REFUTED_MIN} OR "
                    f"D_composed >= D_shuffled - {D_SHUFFLE_MARGIN}"
                ),
            },
        },
        "canonical": canonical,
        "controls": {
            "N1_GOE": goe,
            "N2_Poisson": poisson,
            "N3_shuffled_prime": shuffled,
            "N4_REMESH_isolated": remesh_iso,
        },
        "riemann_reference": riemann_ref,
        "structural_prediction_check": structural_check,
        "f6a_verdict": f6a_verdict,
        "milestone_verdict": milestone_verdict,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _print_summary(report: dict[str, Any]) -> None:
    cfg = report["canonical_config"]
    print("=" * 72)
    print("R-inf-1a-composed  (T = S_IL . M_REMESH on P14 prime-ladder)")
    print("=" * 72)
    print(
        f"Graph: {cfg['graph']}  N={cfg['N_nodes']} nodes, "
        f"dim(joint)={cfg['dim_joint_state']}"
    )
    print(
        f"REMESH: alpha={cfg['alpha']}, tau_l={cfg['tau_l']}, "
        f"tau_g={cfg['tau_g']}; IL: eta={cfg['eta_IL']}"
    )
    print()
    print("Structural prediction (P14 Laplacian = 10 disjoint P_4):")
    sc = report["structural_prediction_check"]
    print(
        f"  empirical distinct eigenvalues of L_G  : {sc['L_G_distinct_eigenvalues_empirical']}"
    )
    print(
        f"  expected (4 values, 10x multiplicity)  : {sc['L_G_distinct_eigenvalues_expected_P4']}"
    )
    print(
        f"  multiplicity per distinct eigenvalue   : {sc['L_G_full_spectrum_multiplicity_per_distinct']}"
    )
    print(
        f"  matches 10 x P_4 prediction            : {sc['matches_10x_P4_prediction']}"
    )
    print()
    print("F6-A KS distance vs GUE Wigner surmise:")
    print(f"  {'label':30s} {'kind':14s} {'#spacings':>10s} {'D_GUE':>10s}")
    for diag in [
        report["canonical"],
        report["controls"]["N1_GOE"],
        report["controls"]["N2_Poisson"],
        report["controls"]["N3_shuffled_prime"],
        report["controls"]["N4_REMESH_isolated"],
        report["riemann_reference"],
    ]:
        d_str = "N/A" if diag["D_GUE"] is None else f"{diag['D_GUE']:.4f}"
        kind = diag.get("projection_kind", "iid_or_zeros")
        print(
            f"  {diag['label']:30s} {kind:14s} "
            f"{diag['n_spacings']:>10d} {d_str:>10s}"
        )
    print()
    print(f"F6-A verdict      : {report['f6a_verdict']}")
    print(f"Milestone verdict : {report['milestone_verdict']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "remesh_infinity"
        / "remesh_infinity_riemann_composed.json",
    )
    args = parser.parse_args()
    report = run_milestone(args.out)
    _print_summary(report)
    print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
