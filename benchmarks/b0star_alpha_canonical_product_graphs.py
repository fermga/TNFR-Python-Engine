"""B0-star-alpha-Q12: Spectrum of canonical Hamiltonian on G_P14 [Cart]
G_P14 and G_P14 [tensor] G_P14 vs S_n-relabelled controls.

Pre-registered diagnostic for B0-star-alpha priority HIGH candidates
Q1 = G_P14 [Cart] G_P14 and Q2 = G_P14 [tensor] G_P14
(Sec 13sexagesima-quarta.4 of TNFR_RIEMANN_RESEARCH_NOTES.md).

Question
--------
Does the canonical Hamiltonian lifted to the Cartesian (Q1) or tensor
(Q2) square of G_P14 escape the S_n-equivariance obstruction that
closed B1 on G_P14 (CCET, Sec 13vicies-novies.16)? I.e. is

    |D_canonical - D_shuffled| >= 0.01      (F8 floor)

on the product graph, where D = KS distance vs GUE Wigner surmise on
consecutive eigenvalue spacings?

Construction
------------
* Base: H_P14 = build_p14_hamiltonian() (N x N, N=40, n_primes=10,
  max_power=4, coupling=0; canonical P14 of Sec 13quinquies).
* Q1 = G_P14 [Cart] G_P14: H_Q1 = H_P14 (x) I_N + I_N (x) H_P14
  (Kronecker sum; canonical Cartesian product Hamiltonian).
* Q2 = G_P14 [tensor] G_P14: H_Q2 = H_P14 (x) H_P14
  (Kronecker product; canonical tensor product Hamiltonian).
* Diagonal S_n action: U_sigma = P_sigma (x) P_sigma on V x V where
  P_sigma is the prime-relabelling permutation matrix on V.
* Shuffled-prime control: rebuild H_P14 with primes permuted (same
  protocol as R-inf-1b control N3); then form H_Q1_shuf, H_Q2_shuf
  the same way.

F7-A statistic (mirrors R-inf-1b)
---------------------------------
1. Remove trivial-cluster eigenvalues |lambda| < TRIVIAL_TOL.
2. Sort real eigenvalues ascending (H_Q1, H_Q2 are self-adjoint).
3. Normalised consecutive spacings delta_k = (s_{k+1} - s_k) / mean.
4. KS distance D_GUE = sup_x |F_emp(x) - F_GUE(x)| with
       P_GUE(s) = (32/pi^2) s^2 exp(-4 s^2 / pi).

F8 structural condition (necessary, pre-registered)
---------------------------------------------------
* F8 SATISFIED: |D_canonical - D_shuffled| >= 0.01
  (canonical product construction breaks S_n-equivariance under prime
  relabelling -> real opening for B0-star-alpha).
* F8 FAILED:    |D_canonical - D_shuffled| < 0.01
  (S_n-equivariance persists on the product graph; canonical
  Kronecker-sum / Kronecker-product lift extends the
  Euler-Orthogonality / CCET obstruction to the canonical-product
  channel -> INDETERMINATE_DEGENERATE_CONSTRUCTION; structural
  refutation of Q1 and/or Q2 as B0-star-alpha entry points).

Pre-registered F7 verdict (mirrors R-inf-1b thresholds)
-------------------------------------------------------
* SUPPORTED       : D_canonical < 0.15 AND
                    D_canonical < D_shuffled - 0.05 AND
                    D_canonical < D_N5      - 0.05.
* REFUTED         : D_canonical > 0.30 OR
                    (D_canonical >= D_shuffled - 0.05 AND F8 SATISFIED).
* INDETERMINATE_DEGENERATE_CONSTRUCTION : F8 FAILED.
* INDETERMINATE_OTHER : F8 SATISFIED and neither SUPPORTED nor REFUTED.

Controls
--------
* N3 (per Q_k): shuffled-prime canonical construction.
* N5 (per Q_k): random self-adjoint Hamiltonian of matched spectral
  radius, lifted by the same canonical product (rules out generic
  self-adjoint behaviour at matched scale).
* External anchor: K_REF=100 Riemann zero imaginary parts via
  mpmath.zetazero.

Pre-registered theoretical expectation
--------------------------------------
F8 FAILED on both Q1 and Q2. Sketch: diagonal S_n acts as
U_sigma = P_sigma (x) P_sigma. Then
    U_sigma H_Q1 U_sigma^T = (P_sigma H P_sigma^T) (x) I
                           + I (x) (P_sigma H P_sigma^T)
                           = H^sigma (x) I + I (x) H^sigma
and H^sigma has the same spectrum as H (just permuted), so
spec(H_Q1_canonical) = spec(H_Q1_shuffled). Identical argument for Q2.
This would constitute the canonical-product extension of CCET-G_P14
(Sec 13vicies-novies.16) and structurally close the HIGH-priority
sub-routes of B0-star-alpha on V(G_P14) x V(G_P14).

If F8 unexpectedly SATISFIED: genuine opening; B0-star-alpha active
on Q1 or Q2; further investigation of antisymmetric subspaces, swap
symmetry, and spacing statistics warranted.

Seeds & parameters
------------------
* numpy default_rng(20260527) for N3, N5 stochastic draws.
* mpmath dps = 30.
* Graph: n_primes=10, max_power=4, coupling=0 (canonical P14).
* N = 40 (base), N^2 = 1600 (product graph dimension).

Output
------
JSON report at the path given by --out (default:
results/b0star_alpha_canonical_product_graphs.json). Console summary
prints F8 deltas and final verdict per candidate.

This is an honest pre-registered diagnostic. The result -- positive or
negative -- will be appended to Sec 13sexagesima-quarta.9 (Results)
of TNFR_RIEMANN_RESEARCH_NOTES.md, regardless of which way it falls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigvalsh

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mpmath as mp

from tnfr.riemann.prime_ladder_hamiltonian import build_prime_ladder_hamiltonian

mp.mp.dps = 30


# -- pre-registered canonical parameters ---------------------------------
N_PRIMES: int = 10
MAX_POWER: int = 4
PERM_SEED: int = 20260527
K_REF_RIEMANN: int = 100

# F7-A thresholds (pre-registered, identical to R-inf-1b /
# Sec 13vicies-novies.12 / Sec 13vicies-novies.14)
D_SUPPORTED_MAX: float = 0.15
D_REFUTED_MIN: float = 0.30
D_SHUFFLE_MARGIN: float = 0.05
D_N5_MARGIN: float = 0.05
F8_FLOOR: float = 0.01

TRIVIAL_TOL: float = 1e-9
SELF_ADJOINT_TOL: float = 1e-10


# -- canonical building blocks -------------------------------------------
def build_p14_hamiltonian(primes: list[int] | None = None) -> np.ndarray:
    """Canonical P14 internal Hamiltonian H_int as a dense (N, N) array."""
    ph = build_prime_ladder_hamiltonian(
        N_PRIMES,
        max_power=MAX_POWER,
        coupling=0.0,
        primes=primes,
    )
    H = np.asarray(ph.hamiltonian.H_int, dtype=np.float64)
    expected = N_PRIMES * MAX_POWER
    if H.shape != (expected, expected):
        raise RuntimeError(f"unexpected H_int shape {H.shape}")
    asym = float(np.max(np.abs(H - H.T)))
    if asym > SELF_ADJOINT_TOL:
        raise RuntimeError(
            f"H_P14 not self-adjoint to {SELF_ADJOINT_TOL}: "
            f"||H - H^T||_inf = {asym:.3e}"
        )
    return H


def build_cartesian_product_hamiltonian(H: np.ndarray) -> np.ndarray:
    """H_Q1 = H (x) I + I (x) H on V x V (canonical Cartesian product)."""
    N = H.shape[0]
    I_N = np.eye(N)
    return np.kron(H, I_N) + np.kron(I_N, H)


def build_tensor_product_hamiltonian(H: np.ndarray) -> np.ndarray:
    """H_Q2 = H (x) H on V x V (canonical tensor / Kronecker product)."""
    return np.kron(H, H)


def build_diagonal_sn_unitary(perm: np.ndarray) -> np.ndarray:
    """U_sigma = P_sigma (x) P_sigma on V x V (diagonal S_n action).

    P_sigma is the n_primes x n_primes prime-relabelling permutation,
    embedded as (P_sigma (x) I_{max_power}) on V (lifts to ladder
    blocks; identity on the k-power axis within each ladder).
    """
    P_sigma = np.zeros((N_PRIMES, N_PRIMES), dtype=np.float64)
    for i, j in enumerate(perm):
        P_sigma[i, int(j)] = 1.0
    P_V = np.kron(P_sigma, np.eye(MAX_POWER))
    return np.kron(P_V, P_V)


# -- F7-A statistic ------------------------------------------------------
def gue_wigner_pdf(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def gue_wigner_cdf(s: np.ndarray) -> np.ndarray:
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
    emp = np.arange(1, n + 1) / n
    cdf = gue_wigner_cdf(sorted_s)
    d_plus = float(np.max(emp - cdf))
    d_minus = float(np.max(cdf - (np.arange(n) / n)))
    return max(d_plus, d_minus)


def f7a_diagnostic(eigvals_real: np.ndarray, label: str) -> dict[str, Any]:
    nontrivial = eigvals_real[np.abs(eigvals_real) > TRIVIAL_TOL]
    sorted_vals = np.sort(nontrivial.astype(np.float64))
    spacings = normalised_spacings(sorted_vals)
    d_gue = ks_distance_vs_gue(spacings)
    return {
        "label": label,
        "n_eigvals_kept": int(sorted_vals.size),
        "n_spacings": int(spacings.size),
        "D_GUE": float(d_gue) if np.isfinite(d_gue) else None,
        "spacings_mean": float(spacings.mean()) if spacings.size else None,
        "spacings_std": float(spacings.std()) if spacings.size else None,
        "spec_min": float(sorted_vals.min()) if sorted_vals.size else None,
        "spec_max": float(sorted_vals.max()) if sorted_vals.size else None,
    }


# -- verdict logic (pre-registered) --------------------------------------
def f7_verdict(
    d_can: float | None, d_shuf: float | None, d_n5: float | None,
) -> tuple[str, bool, float | None]:
    if d_can is None or d_shuf is None:
        return "INDETERMINATE_INVALID_PROJECTION", False, None
    f8_delta = abs(d_can - d_shuf)
    f8_satisfied = f8_delta >= F8_FLOOR
    if not f8_satisfied:
        return "INDETERMINATE_DEGENERATE_CONSTRUCTION", False, f8_delta
    if d_n5 is None:
        return "INDETERMINATE_OTHER", True, f8_delta
    if (d_can < D_SUPPORTED_MAX
            and d_can < d_shuf - D_SHUFFLE_MARGIN
            and d_can < d_n5 - D_N5_MARGIN):
        return "SUPPORTED", True, f8_delta
    if d_can > D_REFUTED_MIN or d_can >= d_shuf - D_SHUFFLE_MARGIN:
        return "REFUTED", True, f8_delta
    return "INDETERMINATE_OTHER", True, f8_delta


# -- per-candidate driver ------------------------------------------------
def diagnose_candidate(
    candidate_id: str,
    H_base: np.ndarray,
    H_base_shuf: np.ndarray,
    H_base_rand: np.ndarray,
    builder,
    perm: np.ndarray,
) -> dict[str, Any]:
    """Run the F7-A / F8 protocol for Q_k = builder(H_P14).

    builder: H -> H_Qk (Kronecker sum or Kronecker product).
    """
    # canonical
    H_Q = builder(H_base)
    eigs_Q = eigvalsh(H_Q)
    diag_can = f7a_diagnostic(eigs_Q, f"{candidate_id}_canonical")
    diag_can["dim"] = int(H_Q.shape[0])

    # shuffled-prime control (N3)
    H_Q_shuf = builder(H_base_shuf)
    eigs_Q_shuf = eigvalsh(H_Q_shuf)
    diag_shuf = f7a_diagnostic(eigs_Q_shuf, f"{candidate_id}_shuffled_prime")
    diag_shuf["perm"] = [int(p) for p in perm]

    # explicit S_n-similarity audit (numerical sanity check of CCET)
    U_sigma = build_diagonal_sn_unitary(perm)
    H_Q_conj = U_sigma @ H_Q @ U_sigma.T
    eigs_Q_conj = eigvalsh(H_Q_conj)
    # |spec(U_sigma H U_sigma^T) - spec(H)|_max should be ~machine eps
    spec_drift_under_diagonal_sn = float(
        np.max(np.abs(np.sort(eigs_Q_conj) - np.sort(eigs_Q)))
    )

    # random self-adjoint control (N5)
    H_Q_rand = builder(H_base_rand)
    eigs_Q_rand = eigvalsh(H_Q_rand)
    diag_rand = f7a_diagnostic(eigs_Q_rand, f"{candidate_id}_random_self_adjoint")

    # verdict
    verdict, f8_satisfied, f8_delta = f7_verdict(
        diag_can["D_GUE"], diag_shuf["D_GUE"], diag_rand["D_GUE"],
    )

    return {
        "candidate_id": candidate_id,
        "canonical": diag_can,
        "controls": {
            "N3_shuffled_prime": diag_shuf,
            "N5_random_self_adjoint": diag_rand,
        },
        "diagonal_sn_audit": {
            "spec_drift_under_diagonal_sn_max_abs": (
                spec_drift_under_diagonal_sn
            ),
            "interpretation": (
                "expected ~machine epsilon if canonical product Hamiltonian "
                "commutes with U_sigma = P_sigma (x) P_sigma; this is the "
                "explicit numerical test of CCET extension to the product "
                "graph"
            ),
        },
        "f8_structural_condition": {
            "delta_D_can_minus_shuf_abs": (
                None if f8_delta is None else float(f8_delta)
            ),
            "floor": F8_FLOOR,
            "satisfied": bool(f8_satisfied),
        },
        "f7_verdict": verdict,
    }


def reference_riemann_d_gue(k_ref: int) -> dict[str, Any]:
    gammas = np.array(
        [float(mp.im(mp.zetazero(k))) for k in range(1, k_ref + 1)]
    )
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


def run_milestone(out_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(PERM_SEED)

    # base canonical H_P14
    H_base = build_p14_hamiltonian()
    spectral_radius = float(np.max(np.abs(eigvalsh(H_base))))

    # shuffled-prime base H_P14_shuf (single permutation, reused across Q1, Q2)
    from sympy import primerange
    primes_canonical = list(primerange(2, 100))[:N_PRIMES]
    perm = rng.permutation(N_PRIMES)
    primes_shuffled = [primes_canonical[int(i)] for i in perm]
    H_base_shuf = build_p14_hamiltonian(primes=primes_shuffled)

    # random self-adjoint base of matched spectral radius (single draw,
    # reused across Q1, Q2)
    N = H_base.shape[0]
    X = rng.standard_normal((N, N))
    H_base_rand = (X + X.T) / np.sqrt(2.0 * N)
    current_radius = float(np.max(np.abs(eigvalsh(H_base_rand))))
    if current_radius > 0:
        H_base_rand *= spectral_radius / current_radius

    q1 = diagnose_candidate(
        "Q1_Cartesian", H_base, H_base_shuf, H_base_rand,
        build_cartesian_product_hamiltonian, perm,
    )
    q2 = diagnose_candidate(
        "Q2_tensor", H_base, H_base_shuf, H_base_rand,
        build_tensor_product_hamiltonian, perm,
    )

    riemann_ref = reference_riemann_d_gue(K_REF_RIEMANN)

    # per-candidate milestone verdict
    def _milestone_verdict(c: dict[str, Any]) -> str:
        v = c["f7_verdict"]
        cid = c["candidate_id"]
        if v == "SUPPORTED":
            return f"{cid}_B0_STAR_ALPHA_POTENTIALLY_OPEN_REQUIRES_REPLICATION"
        if v == "REFUTED":
            return f"{cid}_B0_STAR_ALPHA_REFUTED"
        if v == "INDETERMINATE_DEGENERATE_CONSTRUCTION":
            return (
                f"{cid}_B0_STAR_ALPHA_CCET_EXTENDS_TO_CANONICAL_PRODUCT_GRAPH"
            )
        return f"{cid}_B0_STAR_ALPHA_{v}"

    q1["milestone_verdict"] = _milestone_verdict(q1)
    q2["milestone_verdict"] = _milestone_verdict(q2)

    report = {
        "milestone": "B0-star-alpha-Q12",
        "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md Sec 13sexagesima-quarta",
        "seed": PERM_SEED,
        "canonical_config": {
            "graph_base": "P14 prime-ladder (canonical, unaugmented)",
            "hamiltonian_base": (
                "InternalHamiltonian.H_int via build_prime_ladder_hamiltonian"
            ),
            "n_primes": N_PRIMES,
            "max_power": MAX_POWER,
            "coupling": 0.0,
            "N_base": int(N),
            "primes_canonical": primes_canonical,
            "primes_shuffled": primes_shuffled,
            "perm": [int(p) for p in perm],
            "H_base_spectral_radius": spectral_radius,
            "Q1_lift": "H_Q1 = kron(H_P14, I_N) + kron(I_N, H_P14)",
            "Q2_lift": "H_Q2 = kron(H_P14, H_P14)",
            "diagonal_S_n": "U_sigma = kron(P_sigma_V, P_sigma_V)",
        },
        "preregistration": {
            "statistic": "F7-A KS distance vs GUE Wigner surmise",
            "projection": "sorted real eigenvalues (H_Qk self-adjoint)",
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
                "F8 FAILED on both Q1 and Q2: canonical Kronecker-sum and "
                "Kronecker-product Hamiltonians commute with the diagonal "
                "S_n action U_sigma = P_sigma (x) P_sigma; spec(H_Qk) is "
                "S_n-invariant; D_canonical = D_shuffled at machine "
                "precision; would constitute the canonical-product "
                "extension of CCET-G_P14 (Sec 13vicies-novies.16) and "
                "structurally close the HIGH-priority sub-routes Q1, Q2 of "
                "B0-star-alpha. If F8 unexpectedly SATISFIED: genuine "
                "opening; B0-star-alpha active on Q1 or Q2."
            ),
        },
        "candidates": {
            "Q1": q1,
            "Q2": q2,
        },
        "riemann_reference": riemann_ref,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _print_summary(report: dict[str, Any]) -> None:
    cfg = report["canonical_config"]
    print("=" * 72)
    print("B0-star-alpha-Q12  Q1 = G_P14 [Cart] G_P14   Q2 = G_P14 [tensor] G_P14")
    print("=" * 72)
    print(f"Base graph: {cfg['graph_base']}")
    print(f"  N_base = {cfg['N_base']}, "
          f"H_base_spectral_radius = {cfg['H_base_spectral_radius']:.6f}")
    print(f"  primes canonical = {cfg['primes_canonical']}")
    print(f"  primes shuffled  = {cfg['primes_shuffled']}")
    print(f"Q1 lift: {cfg['Q1_lift']}")
    print(f"Q2 lift: {cfg['Q2_lift']}")
    print()
    rr = report["riemann_reference"]
    print(f"Riemann reference  : D_GUE = {rr['D_GUE']:.6f} "
          f"({rr['n_spacings']} spacings)")
    print()
    for cid in ("Q1", "Q2"):
        c = report["candidates"][cid]
        can = c["canonical"]
        shuf = c["controls"]["N3_shuffled_prime"]
        rand = c["controls"]["N5_random_self_adjoint"]
        f8 = c["f8_structural_condition"]
        audit = c["diagonal_sn_audit"]
        delta_str = ("N/A" if f8["delta_D_can_minus_shuf_abs"] is None
                     else f"{f8['delta_D_can_minus_shuf_abs']:.6e}")
        print("-" * 72)
        print(f"{c['candidate_id']}  dim = {can['dim']}")
        print(f"  D_canonical        = {can['D_GUE']:.6f}")
        print(f"  D_shuffled_prime   = {shuf['D_GUE']:.6f}")
        print(f"  D_N5_random_SA     = {rand['D_GUE']:.6f}")
        print(f"  |D_can - D_shuf|   = {delta_str}  "
              f"(floor = {F8_FLOOR})  "
              f"F8 satisfied = {f8['satisfied']}")
        print(f"  S_n similarity audit (spec drift max abs) = "
              f"{audit['spec_drift_under_diagonal_sn_max_abs']:.3e}")
        print(f"  F7 verdict          = {c['f7_verdict']}")
        print(f"  milestone verdict   = {c['milestone_verdict']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "results" / "b0star_alpha_canonical_product_graphs.json",
    )
    args = parser.parse_args()
    report = run_milestone(args.out)
    _print_summary(report)
    print(f"\nReport written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
