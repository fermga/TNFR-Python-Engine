"""R-inf-1b: Spectrum of T_spec = S_IL^spec . M_REMESH on the P14
internal Hilbert space {|p,k>} vs the GUE Wigner surmise.

Pre-registered milestone (§13vicies-novies.14). Tests whether the
canonical tensor-product lift of the spectral IL contraction
S_IL^spec = I_{tau_g+1} (x) exp(-eta H_P14) composed with the canonical
REMESH echo matrix M_REMESH = M (x) I_N encodes Riemann-zero content
in its iteration-matrix spectrum -- i.e. whether breaking hypothesis (i)
of the Euler-Orthogonality Lemma (§13vicies-novies.11) by moving from
edge-channel (graph Laplacian) to spectral-channel (P14 Hamiltonian)
suffices to recover Riemann level statistics.

This is a pre-registration commit. The methodology, parameters, seeds,
and decision thresholds are locked here. No data is collected at commit
time. First execution will append the Results block as
§13vicies-novies.15.

Construction
------------
* Graph: canonical P14 prime-ladder
  (src/tnfr/riemann/prime_ladder_hamiltonian.py::build_prime_ladder_graph)
  with n_primes=10, max_power=4, coupling=0. N = 40 nodes = 10 disjoint P_4.
* Hamiltonian: full canonical P14 internal Hamiltonian H_P14 = H_int
  via tnfr.operators.hamiltonian.InternalHamiltonian on the canonical
  graph; H_int = H_coh + H_freq + H_coupling with H_coupling = 0 here
  (coupling=0 in build_prime_ladder_hamiltonian), so H_freq carries the
  prime-ladder spectrum (eigenvalues k * log p_i) on its diagonal.
* REMESH delay window: tau_g + 1 = 17 slots.
* Joint state dim: N * (tau_g + 1) = 680.
* Index convention: index = slot * N + node. Then
      M_REMESH         = kron(M, I_N)
      S_IL^spec        = kron(I_{tau_g+1}, expm(-eta * H_P14))
      T_spec           = S_IL^spec @ M_REMESH

The spectral IL lift is *uniform across all slots* (not slot-0-only as
in §13vicies-novies.9 / §13vicies-novies.12), matching the canonical
spectral-space construction of §13vicies-novies.10.

F7-A statistic (pre-registered)
-------------------------------
1. Remove trivial fixed-point cluster: |lambda - 1| < 1e-9.
2. Project to 1-D: Im(lambda) for upper-half-plane subset (Im >= 1e-12),
   sorted ascending. Fallback Re(lambda) sorted ascending if projection
   is empty (real spectrum).
3. Normalised consecutive spacings: delta_k = (s_{k+1} - s_k) / mean.
4. KS distance D_GUE = sup_x |F_emp(x) - F_GUE(x)| with
       P_GUE(s) = (32/pi^2) s^2 exp(-4 s^2 / pi).

F8 structural condition (pre-registered, necessary)
---------------------------------------------------
* F8 SATISFIED: |D_canonical - D_shuffled| >= 0.01
  (spectral-space composition genuinely breaks S_n-equivariance under
  prime relabelling).
* F8 FAILED:    |D_canonical - D_shuffled| < 0.01
  (spectral equivalence persists; canonical tensor-product lift extends
  the Euler-Orthogonality obstruction to the spectral channel ->
  INDETERMINATE construction).

Pre-registered F7 verdict
-------------------------
* SUPPORTED       : D_canonical < 0.15 AND
                    D_canonical < D_shuffled - 0.05 AND
                    D_canonical < D_N5      - 0.05.
* REFUTED         : D_canonical > 0.30 OR
                    (D_canonical >= D_shuffled - 0.05 AND F8 SATISFIED).
* INDETERMINATE_DEGENERATE_CONSTRUCTION : F8 FAILED.
* INDETERMINATE_OTHER : F8 SATISFIED and neither SUPPORTED nor REFUTED.

Controls
--------
* N1 GOE (dim 680, symmetric, real spectrum -> Re-projection fallback)
* N2 Poisson (680 uniform points -> spacings of e^{-s} distribution)
* N3 prime-ladder shuffled (primary discriminator for F8: primes
  permuted across the 10 P_4 components; H_P14 re-instantiated via
  InternalHamiltonian on the relabelled graph)
* N4 REMESH-isolated (re-run of the 17-eigenvalue M matrix; reported as
  diagnostic baseline, expected to be degenerate)
* N5 random-self-adjoint-replacement (replace H_P14 with a random
  symmetric 40 x 40 matrix of the same spectral radius; tests whether
  canonical P14 spectrum structure matters vs generic self-adjoint
  operator of comparable scale)

Reference
---------
* D_Riemann: KS distance for the first K_ref = 100 Riemann zero
  imaginary parts via mpmath.zetazero. External anchor.

Seeds & parameters
------------------
* numpy default_rng(20260526) for N1/N2/N3/N5 stochastic draws
  (reused from §13vicies-novies.12 for cross-milestone reproducibility
  consistency).
* mpmath dps = 30.
* REMESH: alpha = 0.5, tau_l = 4, tau_g = 16.
* Spectral IL coupling: eta = 0.3.
* Graph: n_primes = 10, max_power = 4, coupling = 0.

Result of this milestone will be appended to
theory/TNFR_RIEMANN_RESEARCH_NOTES.md as §13vicies-novies.15.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eig, eigvalsh, expm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mpmath as mp

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_hamiltonian

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

# F7-A thresholds (pre-registered, identical to §13vicies-novies.12)
D_SUPPORTED_MAX: float = 0.15
D_REFUTED_MIN: float = 0.30
D_SHUFFLE_MARGIN: float = 0.05
D_N5_MARGIN: float = 0.05
F8_FLOOR: float = 0.01

TRIVIAL_TOL: float = 1e-9
IM_TOL: float = 1e-12
SELF_ADJOINT_TOL: float = 1e-12


# -- canonical building blocks -------------------------------------------
def build_remesh_iteration_matrix(alpha: float, tau_l: int, tau_g: int) -> np.ndarray:
    """Canonical REMESH (tau_g+1)x(tau_g+1) shift-augmented matrix.

    Mirror of remesh_infinity_riemann_composed.py and
    remesh_infinity_riemann_modified_graph.py to keep the milestones
    bit-comparable at the REMESH layer.
    """
    if tau_l < 1 or tau_g < 1 or tau_l > tau_g:
        raise ValueError("require 1 <= tau_l <= tau_g")
    if not (0.0 < alpha < 1.0):
        raise ValueError("require 0 < alpha < 1")
    dim = tau_g + 1
    M = np.zeros((dim, dim), dtype=np.float64)
    M[0, 0] = (1.0 - alpha) ** 2
    M[0, tau_l] = alpha * (1.0 - alpha)
    M[0, tau_g] = alpha
    for k in range(1, dim):
        M[k, k - 1] = 1.0
    return M


def build_p14_hamiltonian(primes: list[int] | None = None) -> np.ndarray:
    """Canonical P14 internal Hamiltonian H_int as a dense (40, 40) array.

    Uses tnfr.riemann.prime_ladder_hamiltonian.build_prime_ladder_hamiltonian
    with n_primes=10, max_power=4, coupling=0 (canonical P14 reference of
    §13quinquies). Returns the .hamiltonian.H_int matrix as float64.

    If `primes` is provided, the prime-ladder is instantiated on the
    given prime list (used by the N3 shuffled-prime control).
    """
    ph = build_prime_ladder_hamiltonian(
        N_PRIMES,
        max_power=MAX_POWER,
        coupling=0.0,
        primes=primes,
    )
    H = np.asarray(ph.hamiltonian.H_int, dtype=np.float64)
    if H.shape != (N_PRIMES * MAX_POWER, N_PRIMES * MAX_POWER):
        raise RuntimeError(
            f"unexpected H_int shape {H.shape}; "
            f"expected ({N_PRIMES * MAX_POWER}, {N_PRIMES * MAX_POWER})"
        )
    asym = float(np.max(np.abs(H - H.T)))
    if asym > SELF_ADJOINT_TOL:
        raise RuntimeError(
            f"H_P14 not self-adjoint to tolerance {SELF_ADJOINT_TOL}: "
            f"||H - H^T||_inf = {asym:.3e}"
        )
    return H


def build_spectral_il_smoother(H: np.ndarray, eta: float) -> np.ndarray:
    """S_IL^spec on H_N: exp(-eta * H_P14). Self-adjoint by construction."""
    return expm(-eta * H)


def build_spectral_iteration_matrix(
    M: np.ndarray, S_N: np.ndarray,
) -> np.ndarray:
    """T_spec = S_IL^spec . M_REMESH in slot-major ordering.

    Index convention: index = slot * N + (p,k). Then
        M_REMESH    = kron(M, I_N)
        S_IL^spec   = kron(I_{tau_g+1}, S_N)     where S_N = exp(-eta H_P14)
                      (uniform across all slots)
    """
    N = S_N.shape[0]
    dim_slot = M.shape[0]  # tau_g + 1
    I_N = np.eye(N)
    I_slot = np.eye(dim_slot)
    M_REMESH = np.kron(M, I_N)
    S_IL_spec = np.kron(I_slot, S_N)
    return S_IL_spec @ M_REMESH


# -- F7-A: KS distance vs GUE Wigner surmise -----------------------------
def gue_wigner_pdf(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def gue_wigner_cdf(s: np.ndarray) -> np.ndarray:
    """CDF by trapezoidal quadrature on a dense grid (matches the
    convention of remesh_infinity_riemann_composed.py and
    remesh_infinity_riemann_modified_graph.py)."""
    s = np.asarray(s, dtype=np.float64)
    grid = np.linspace(0.0, max(float(s.max()) + 1e-9, 1.0), 4001)
    pdf = gue_wigner_pdf(grid)
    cdf_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]
    )
    cdf_grid = cdf_grid / cdf_grid[-1]
    return np.interp(s, grid, cdf_grid)


def normalised_spacings(values_sorted: np.ndarray) -> np.ndarray:
    if values_sorted.size < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(values_sorted)
    mean = diffs.mean()
    if mean <= 0.0:
        return np.zeros_like(diffs)
    return diffs / mean


def ks_distance_vs_gue(spacings: np.ndarray) -> float:
    if spacings.size == 0:
        return float("nan")
    sorted_s = np.sort(spacings)
    n = sorted_s.size
    emp_cdf_at_points = np.arange(1, n + 1) / n
    gue_at_points = gue_wigner_cdf(sorted_s)
    d_plus = float(np.max(emp_cdf_at_points - gue_at_points))
    d_minus = float(np.max(gue_at_points - (np.arange(n) / n)))
    return max(d_plus, d_minus)


def project_spectrum_1d(eigvals: np.ndarray) -> tuple[np.ndarray, str]:
    nontrivial = eigvals[np.abs(eigvals - 1.0) > TRIVIAL_TOL]
    upper = nontrivial[np.imag(nontrivial) >= IM_TOL]
    if upper.size >= 2:
        return np.sort(np.imag(upper).astype(np.float64)), "Im_upper"
    return np.sort(np.real(nontrivial).astype(np.float64)), "Re_fallback"


def f7a_diagnostic(eigvals: np.ndarray, label: str) -> dict[str, Any]:
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
    X = rng.standard_normal((dim, dim))
    A = (X + X.T) / np.sqrt(2.0 * dim)
    eigvals = eigvalsh(A).astype(np.complex128)
    return f7a_diagnostic(eigvals, "N1_GOE")


def control_poisson(n_points: int, rng: np.random.Generator) -> dict[str, Any]:
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


def control_shuffled_prime(
    M: np.ndarray, rng: np.random.Generator,
) -> dict[str, Any]:
    """N3: rebuild H_P14 with primes permuted across the 10 ladders.

    H_P14 is re-instantiated via InternalHamiltonian on the relabelled
    graph (the prime ordering enters via H_freq diagonal entries
    k * log p_i; this is the *primary discriminator* for F8).
    """
    from sympy import primerange
    primes = list(primerange(2, 100))[:N_PRIMES]
    perm = rng.permutation(N_PRIMES)
    primes_shuffled = [primes[int(i)] for i in perm]
    H_shuffled = build_p14_hamiltonian(primes=primes_shuffled)
    S_N_shuffled = build_spectral_il_smoother(H_shuffled, ETA_IL)
    T_shuffled = build_spectral_iteration_matrix(M, S_N_shuffled)
    eigvals_shuffled, _ = eig(T_shuffled)
    diag = f7a_diagnostic(eigvals_shuffled, "N3_shuffled_prime")
    diag["primes_canonical"] = primes
    diag["primes_shuffled"] = primes_shuffled
    diag["H_shuffled_spectral_radius"] = float(np.max(np.abs(eigvalsh(H_shuffled))))
    return diag


def control_remesh_isolated(M: np.ndarray) -> dict[str, Any]:
    eigvals_M, _ = eig(M)
    return f7a_diagnostic(eigvals_M.astype(np.complex128), "N4_REMESH_isolated")


def control_random_self_adjoint(
    M: np.ndarray, target_radius: float, rng: np.random.Generator,
) -> dict[str, Any]:
    """N5: replace H_P14 with a random symmetric matrix of matching
    spectral radius.

    Tests whether the canonical P14 spectrum carries additional content
    beyond a generic self-adjoint operator of comparable scale.
    """
    N = N_PRIMES * MAX_POWER
    X = rng.standard_normal((N, N))
    H_rand = (X + X.T) / np.sqrt(2.0 * N)
    current_radius = float(np.max(np.abs(eigvalsh(H_rand))))
    if current_radius > 0:
        H_rand *= target_radius / current_radius
    S_N_rand = build_spectral_il_smoother(H_rand, ETA_IL)
    T_rand = build_spectral_iteration_matrix(M, S_N_rand)
    eigvals_rand, _ = eig(T_rand)
    diag = f7a_diagnostic(eigvals_rand, "N5_random_self_adjoint")
    diag["target_spectral_radius"] = float(target_radius)
    diag["realised_spectral_radius"] = float(np.max(np.abs(eigvalsh(H_rand))))
    return diag


def reference_riemann_d_gue(k_ref: int) -> dict[str, Any]:
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
    H_canonical = build_p14_hamiltonian()
    spectral_radius_canonical = float(np.max(np.abs(eigvalsh(H_canonical))))
    S_N_canonical = build_spectral_il_smoother(H_canonical, ETA_IL)
    M = build_remesh_iteration_matrix(ALPHA, TAU_LOCAL, TAU_GLOBAL)
    T = build_spectral_iteration_matrix(M, S_N_canonical)
    N = H_canonical.shape[0]
    dim_joint = T.shape[0]
    eigvals_T, _ = eig(T)

    canonical = f7a_diagnostic(eigvals_T, "canonical_S_IL_spec_o_M_REMESH")
    canonical["dim_joint"] = int(dim_joint)
    canonical["N_basis"] = int(N)
    canonical["H_spectral_radius"] = float(spectral_radius_canonical)

    # -- controls
    goe = control_goe(dim_joint, rng)
    poisson = control_poisson(dim_joint, rng)
    shuffled = control_shuffled_prime(M, rng)
    remesh_iso = control_remesh_isolated(M)
    random_sa = control_random_self_adjoint(M, spectral_radius_canonical, rng)
    riemann_ref = reference_riemann_d_gue(K_REF_RIEMANN)

    # -- F8 structural condition
    d_can = canonical["D_GUE"]
    d_shuf = shuffled["D_GUE"]
    d_n5 = random_sa["D_GUE"]

    if d_can is None or d_shuf is None:
        f8_satisfied = False
        f8_delta = None
    else:
        f8_delta = abs(d_can - d_shuf)
        f8_satisfied = f8_delta >= F8_FLOOR

    # -- pre-registered F7 verdict
    if d_can is None or d_shuf is None or d_n5 is None:
        f7a_verdict = "INDETERMINATE_INVALID_PROJECTION"
    elif not f8_satisfied:
        f7a_verdict = "INDETERMINATE_DEGENERATE_CONSTRUCTION"
    elif (d_can < D_SUPPORTED_MAX
          and d_can < d_shuf - D_SHUFFLE_MARGIN
          and d_can < d_n5 - D_N5_MARGIN):
        f7a_verdict = "SUPPORTED"
    elif (d_can > D_REFUTED_MIN
          or d_can >= d_shuf - D_SHUFFLE_MARGIN):
        f7a_verdict = "REFUTED"
    else:
        f7a_verdict = "INDETERMINATE_OTHER"

    # -- pre-registered milestone verdict
    if f7a_verdict == "SUPPORTED":
        milestone_verdict = (
            "B1_SPECTRAL_BASIS_POTENTIALLY_OPEN_REQUIRES_REPLICATION"
        )
    elif f7a_verdict == "REFUTED":
        milestone_verdict = (
            "B1_SPECTRAL_BASIS_REFUTED_FOR_CANONICAL_TENSOR_PRODUCT_LIFT"
        )
    elif f7a_verdict == "INDETERMINATE_DEGENERATE_CONSTRUCTION":
        milestone_verdict = (
            "B1_SPECTRAL_BASIS_INDETERMINATE_EULER_ORTHOGONALITY_EXTENDS_TO_SPECTRAL_CHANNEL"
        )
    else:
        milestone_verdict = f"B1_SPECTRAL_BASIS_{f7a_verdict}"

    report = {
        "milestone": "R-inf-1b",
        "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.14",
        "canonical_config": {
            "graph": "P14 prime-ladder (canonical, unaugmented)",
            "hamiltonian": "InternalHamiltonian.H_int via build_prime_ladder_hamiltonian",
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "coupling": 0.0,
            "N_basis": int(N),
            "alpha": ALPHA,
            "tau_l": TAU_LOCAL,
            "tau_g": TAU_GLOBAL,
            "tau_g_plus_1": TAU_GLOBAL + 1,
            "eta_IL_spec": ETA_IL,
            "dim_joint_state": int(dim_joint),
            "H_spectral_radius": float(spectral_radius_canonical),
            "lift": (
                "S_IL_spec = kron(I_{tau_g+1}, expm(-eta * H_P14)); "
                "M_REMESH = kron(M, I_N); T_spec = S_IL_spec @ M_REMESH"
            ),
        },
        "seed": PERM_SEED,
        "preregistration": {
            "statistic": "F7-A KS distance vs GUE Wigner surmise",
            "projection": "Im(lambda) upper-half-plane (fallback Re(lambda))",
            "thresholds": {
                "SUPPORTED": (
                    f"D_canonical < {D_SUPPORTED_MAX} AND "
                    f"D_canonical < D_shuffled - {D_SHUFFLE_MARGIN} AND "
                    f"D_canonical < D_N5      - {D_N5_MARGIN}"
                ),
                "REFUTED": (
                    f"D_canonical > {D_REFUTED_MIN} OR "
                    f"(D_canonical >= D_shuffled - {D_SHUFFLE_MARGIN} "
                    f"AND F8 SATISFIED)"
                ),
                "F8_SATISFIED": f"|D_canonical - D_shuffled| >= {F8_FLOOR}",
            },
            "theoretical_expectation": (
                "F8 FAILED (|D_can - D_shuf| at machine-precision floor); "
                "canonical tensor-product lift S_IL_spec = kron(I, exp(-eta H)) "
                "and M_REMESH = kron(M, I_N) commute with prime relabelling "
                "kron(I, P_sigma) up to unitary similarity, so spectra of "
                "T_spec_canonical and T_spec_shuffled coincide; would constitute "
                "a spectral-channel instantiation of the Euler-Orthogonality "
                "obstruction (§13vicies-novies.11)."
            ),
        },
        "f8_structural_condition": {
            "delta_D_can_minus_shuf_abs": (
                None if f8_delta is None else float(f8_delta)
            ),
            "floor": F8_FLOOR,
            "satisfied": bool(f8_satisfied),
        },
        "canonical": canonical,
        "controls": {
            "N1_GOE": goe,
            "N2_Poisson": poisson,
            "N3_shuffled_prime": shuffled,
            "N4_REMESH_isolated": remesh_iso,
            "N5_random_self_adjoint": random_sa,
        },
        "riemann_reference": riemann_ref,
        "f7a_verdict": f7a_verdict,
        "milestone_verdict": milestone_verdict,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _print_summary(report: dict[str, Any]) -> None:
    cfg = report["canonical_config"]
    print("=" * 72)
    print("R-inf-1b  (T_spec = S_IL^spec . M_REMESH on P14 internal Hilbert space)")
    print("=" * 72)
    print(f"Graph: {cfg['graph']}")
    print(f"Hamiltonian: {cfg['hamiltonian']}")
    print(f"  N_basis={cfg.get('N_basis', '?')}, "
          f"dim(joint)={cfg.get('dim_joint_state', '?')}")
    print(f"  H_spectral_radius = {cfg['H_spectral_radius']:.6f}")
    print(f"REMESH: alpha={cfg.get('alpha', ALPHA)}, "
          f"tau_l={cfg.get('tau_l', TAU_LOCAL)}, "
          f"tau_g={cfg.get('tau_g', TAU_GLOBAL)}; "
          f"spectral IL: eta={cfg.get('eta_IL_spec', ETA_IL)}")
    print()
    f8 = report.get("f8_structural_condition")
    if f8 is not None:
        d_str = ("N/A" if f8["delta_D_can_minus_shuf_abs"] is None
                 else f"{f8['delta_D_can_minus_shuf_abs']:.4e}")
        print(f"F8 structural condition (|D_can - D_shuf| >= {f8['floor']}):")
        print(f"  |Delta D|  = {d_str}   satisfied = {f8['satisfied']}")
        print()
    print("F7-A KS distance vs GUE Wigner surmise:")
    print(f"  {'label':36s} {'kind':14s} {'#spacings':>10s} {'D_GUE':>10s}")
    diags = [report["canonical"]]
    diags.extend([report["controls"][k] for k in (
        "N1_GOE", "N2_Poisson", "N3_shuffled_prime",
        "N4_REMESH_isolated", "N5_random_self_adjoint",
    )])
    diags.append(report["riemann_reference"])
    for diag in diags:
        d_str = ("N/A" if diag["D_GUE"] is None
                 else f"{diag['D_GUE']:.4f}")
        kind = diag.get("projection_kind", "iid_or_zeros")
        print(f"  {diag['label']:36s} {kind:14s} "
              f"{diag['n_spacings']:>10d} {d_str:>10s}")
    print()
    print(f"F7-A verdict      : {report['f7a_verdict']}")
    print(f"Milestone verdict : {report['milestone_verdict']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "remesh_infinity"
                / "remesh_infinity_riemann_spectral_basis.json",
    )
    args = parser.parse_args()
    report = run_milestone(args.out)
    _print_summary(report)
    print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
