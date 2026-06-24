"""R-inf-1c: Spectrum of T_aug = S_IL^aug . M_REMESH on the canonically
modified prime-ladder graph G_P14^aug vs the GUE Wigner surmise.

Pre-registered milestone (§13vicies-novies.12). Tests whether breaking
hypothesis (iii) of the Euler-Orthogonality Lemma (§13vicies-novies.11)
by adding canonical inter-prime edges to G_P14 -- derived from the
nodal-equation phase-gradient threshold gamma/pi applied to the
structural frequency nu_f = k log p -- is sufficient to encode
Riemann-zero content in the iteration-matrix spectrum.

This is a pre-registration commit. The methodology, parameters, seeds,
and decision thresholds are locked here. No data is collected at commit
time. First execution will append the Results block as
§13vicies-novies.13.

Construction
------------
* Base graph: P14 prime-ladder
  (src/tnfr/riemann/prime_ladder_hamiltonian.py::build_prime_ladder_graph)
  with n_primes=10, max_power=4, coupling=0.  N = 40 nodes = 10 disjoint P_4.
* Canonical coherence threshold:
      delta_coh = (gamma/pi) * max nu_f = (gamma/pi) * k_max * log p_max.
  For n_primes=10, max_power=4: delta_coh ~ 0.184 * 4 log 29 ~ 2.479.
* Inter-prime edges: (p_i,k) <-> (p_j,k') admissible iff p_i != p_j and
      |k log p_i - k' log p_j| <= delta_coh.
* Inter-prime edge weights (Kuramoto-U3 form, prefactor gamma/pi):
      w_ij = (gamma/pi) * exp(- |nu_f(i) - nu_f(j)|^2 / (2 delta_coh^2)).
* Intra-prime edges retain unit weight.

* REMESH delay window: tau_g + 1 = 17 slots.
* Joint state dim: N * (tau_g + 1) = 680.
* Index convention: index = slot * N + node. Then
      M_REMESH         = kron(M, I_N)
      S_IL^aug         = block_diag((I_N - eta L_{G^aug}), I_N, ..., I_N)
      T_aug            = S_IL^aug @ M_REMESH

with L_{G^aug} the weighted combinatorial Laplacian of G_P14^aug and
eta = 0.3 the canonical IL phase-locking coefficient.

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
  (Euler-Orthogonality Lemma hypothesis (iii) genuinely broken by the
  inter-prime augmentation).
* F8 FAILED:    |D_canonical - D_shuffled| < 0.01
  (label-independence persists -> implementation degeneracy or
  delta_coh too small to generate label-dependent inter-prime weights).

Pre-registered F7 verdict
-------------------------
* SUPPORTED       : D_canonical < 0.15 AND
                    D_canonical < D_shuffled - 0.05 AND
                    D_canonical < D_random   - 0.05.
* REFUTED         : D_canonical > 0.30 OR
                    (D_canonical >= D_shuffled - 0.05 AND F8 SATISFIED).
* INDETERMINATE_DEGENERATE_CONSTRUCTION : F8 FAILED.
* INDETERMINATE_OTHER : otherwise.

Controls
--------
* N1 GOE (dim 680, symmetric, real spectrum -> Re-projection fallback)
* N2 Poisson (680 uniform points -> spacings of e^{-s} distribution)
* N3 prime-ladder shuffled (primary discriminator for F8: primes
  permuted across the 10 P_4 components; inter-prime weights re-derived
  from nu_f on the shuffled labels)
* N4 REMESH-isolated (re-run of the 17-eigenvalue M matrix; reported as
  diagnostic baseline, expected to be degenerate)
* N5 random-augmentation (replace E_inter with an Erdos-Renyi random
  edge set of the same edge count, all weights set to gamma/pi; tests
  whether canonical delta_coh-derived structure matters vs generic
  random topology of the same density)

Reference
---------
* D_Riemann: KS distance for the first K_ref = 100 Riemann zero
  imaginary parts via mpmath.zetazero. External anchor.

Seeds & parameters
------------------
* numpy default_rng(20260526) for N1/N2/N3/N5 stochastic draws.
* mpmath dps = 30.
* REMESH: alpha = 0.5, tau_l = 4, tau_g = 16.
* IL coupling: eta = 0.3.
* Graph: n_primes = 10, max_power = 4, coupling = 0.
* Canonical coupling: delta_coh = (gamma/pi) * k_max * log p_max;
  w_ij = (gamma/pi) * exp(- |Delta nu_f|^2 / (2 delta_coh^2)) on E_inter.

Result of this milestone will be appended to
theory/TNFR_RIEMANN_RESEARCH_NOTES.md as §13vicies-novies.13.
"""

from __future__ import annotations

import argparse
import json
import math
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

from tnfr.constants.canonical import GAMMA, PI
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

# F7-A thresholds (pre-registered)
D_SUPPORTED_MAX: float = 0.15
D_REFUTED_MIN: float = 0.30
D_SHUFFLE_MARGIN: float = 0.05
D_RANDOM_MARGIN: float = 0.05
F8_FLOOR: float = 0.01

TRIVIAL_TOL: float = 1e-9
IM_TOL: float = 1e-12

# Canonical Kuramoto-U3 prefactor (tetrad-edge gamma/pi).
KAPPA: float = GAMMA / PI


# -- canonical building blocks -------------------------------------------
def build_remesh_iteration_matrix(alpha: float, tau_l: int, tau_g: int) -> np.ndarray:
    """Canonical REMESH (tau_g+1)x(tau_g+1) shift-augmented matrix.

    Mirror of remesh_infinity_riemann_composed.py to keep the two
    milestones bit-comparable at the REMESH layer.
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


def canonical_delta_coh(G) -> float:
    """delta_coh = (gamma/pi) * max nu_f over the graph nodes."""
    max_nu = max(float(G.nodes[n]["nu_f"]) for n in G.nodes())
    return KAPPA * max_nu


def canonical_inter_prime_weight(nu_i: float, nu_j: float, delta_coh: float) -> float:
    """Kuramoto-U3 inter-prime weight, canonical Gaussian decay."""
    dnu = nu_i - nu_j
    return KAPPA * math.exp(-(dnu * dnu) / (2.0 * delta_coh * delta_coh))


def augment_with_canonical_inter_prime_edges(
    G,
    delta_coh: float,
) -> tuple[int, list[tuple[Any, Any, float]]]:
    """Add inter-prime edges (p_i,k) <-> (p_j,k') iff
    p_i != p_j and |nu_f(i) - nu_f(j)| <= delta_coh.

    Edge weights set to the canonical Kuramoto-U3 form. Mutates G.
    Returns (n_edges_added, edges_list_with_weights).
    """
    nodes = list(G.nodes())
    added: list[tuple[Any, Any, float]] = []
    for a in range(len(nodes)):
        u = nodes[a]
        p_u = u[0]
        nu_u = float(G.nodes[u]["nu_f"])
        for b in range(a + 1, len(nodes)):
            v = nodes[b]
            p_v = v[0]
            if p_u == p_v:
                continue
            nu_v = float(G.nodes[v]["nu_f"])
            if abs(nu_u - nu_v) <= delta_coh:
                w = canonical_inter_prime_weight(nu_u, nu_v, delta_coh)
                G.add_edge(u, v, weight=w)
                added.append((u, v, w))
    return len(added), added


def weighted_laplacian(G) -> np.ndarray:
    """Unnormalised weighted combinatorial Laplacian L = D - W.

    Intra-prime edges in G_P14 have no 'weight' attribute and default
    to weight 1.0. Inter-prime edges carry the canonical Kuramoto-U3
    weight set by augment_with_canonical_inter_prime_edges.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    W = np.zeros((n, n), dtype=np.float64)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        i, j = idx[u], idx[v]
        W[i, j] += w
        W[j, i] += w
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    return L


def build_il_smoother(L: np.ndarray, eta: float) -> np.ndarray:
    """S_IL^aug acting on slot 0: (I - eta L)."""
    n = L.shape[0]
    return np.eye(n) - eta * L


def build_composed_iteration_matrix(
    M: np.ndarray,
    L: np.ndarray,
    eta: float,
) -> np.ndarray:
    """T_aug = S_IL^aug . M_REMESH in slot-major ordering.

    Same construction as remesh_infinity_riemann_composed.py with the
    only difference that L is the weighted Laplacian of G^aug.
    """
    N = L.shape[0]
    dim_slot = M.shape[0]
    I_N = np.eye(N)
    M_REMESH = np.kron(M, I_N)
    S_blocks = [np.eye(N) for _ in range(dim_slot)]
    S_blocks[0] = build_il_smoother(L, eta)
    S_IL = np.zeros((dim_slot * N, dim_slot * N), dtype=np.float64)
    for s, blk in enumerate(S_blocks):
        S_IL[s * N : (s + 1) * N, s * N : (s + 1) * N] = blk
    return S_IL @ M_REMESH


# -- F7-A: KS distance vs GUE Wigner surmise -----------------------------
def gue_wigner_pdf(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)


def gue_wigner_cdf(s: np.ndarray) -> np.ndarray:
    """CDF by trapezoidal quadrature on a dense grid (matches the
    convention used in remesh_infinity_riemann_composed.py)."""
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
    M: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], int]:
    """N3: rebuild G_P14^aug with primes permuted across the 10 ladders.

    Inter-prime weights are re-derived from nu_f on the shuffled labels
    (the augmentation is *label-dependent* by construction; this is the
    primary discriminator for F8).
    """
    from sympy import primerange

    primes = list(primerange(2, 100))[:N_PRIMES]
    perm = rng.permutation(N_PRIMES)
    primes_shuffled = [primes[i] for i in perm]
    G_shuffled = build_prime_ladder_graph(
        N_PRIMES,
        max_power=MAX_POWER,
        coupling=0.0,
        primes=primes_shuffled,
    )
    delta_coh_shuffled = canonical_delta_coh(G_shuffled)
    n_inter_shuffled, _ = augment_with_canonical_inter_prime_edges(
        G_shuffled,
        delta_coh_shuffled,
    )
    L_shuffled = weighted_laplacian(G_shuffled)
    T_shuffled = build_composed_iteration_matrix(M, L_shuffled, ETA_IL)
    eigvals_shuffled, _ = eig(T_shuffled)
    diag = f7a_diagnostic(eigvals_shuffled, "N3_shuffled_prime")
    diag["primes_canonical"] = primes
    diag["primes_shuffled"] = primes_shuffled
    diag["delta_coh_shuffled"] = float(delta_coh_shuffled)
    diag["n_inter_prime_edges_shuffled"] = int(n_inter_shuffled)
    return diag, n_inter_shuffled


def control_remesh_isolated(M: np.ndarray) -> dict[str, Any]:
    eigvals_M, _ = eig(M)
    return f7a_diagnostic(eigvals_M.astype(np.complex128), "N4_REMESH_isolated")


def control_random_augmentation(
    M: np.ndarray,
    n_inter_target: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """N5: rebuild G_P14 with an Erdos-Renyi random inter-prime edge set
    of the same edge count as the canonical augmentation, all weights set
    to the canonical prefactor gamma/pi (uniform, no nu_f dependence)."""
    G_rand = build_prime_ladder_graph(N_PRIMES, max_power=MAX_POWER, coupling=0.0)
    nodes = list(G_rand.nodes())
    # Enumerate admissible inter-prime pairs (different primes only).
    inter_pairs = [
        (a, b)
        for a in range(len(nodes))
        for b in range(a + 1, len(nodes))
        if nodes[a][0] != nodes[b][0]
    ]
    if n_inter_target > len(inter_pairs):
        raise RuntimeError(
            f"N5: n_inter_target={n_inter_target} exceeds available "
            f"inter-prime pairs {len(inter_pairs)}"
        )
    chosen_idx = rng.choice(len(inter_pairs), size=n_inter_target, replace=False)
    for k in chosen_idx:
        a, b = inter_pairs[int(k)]
        G_rand.add_edge(nodes[a], nodes[b], weight=KAPPA)
    L_rand = weighted_laplacian(G_rand)
    T_rand = build_composed_iteration_matrix(M, L_rand, ETA_IL)
    eigvals_rand, _ = eig(T_rand)
    diag = f7a_diagnostic(eigvals_rand, "N5_random_augmentation")
    diag["n_inter_prime_edges_random"] = int(n_inter_target)
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
    G = build_prime_ladder_graph(N_PRIMES, max_power=MAX_POWER, coupling=0.0)
    delta_coh = canonical_delta_coh(G)
    n_inter, inter_edges = augment_with_canonical_inter_prime_edges(G, delta_coh)

    # Degenerate-construction check (pre-registered).
    if n_inter < 2:
        report: dict[str, Any] = {
            "milestone": "R-inf-1c",
            "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.12",
            "canonical_config": {
                "n_primes": N_PRIMES,
                "max_power": MAX_POWER,
                "delta_coh": float(delta_coh),
                "n_inter_prime_edges_canonical": int(n_inter),
            },
            "seed": PERM_SEED,
            "f7a_verdict": "INDETERMINATE_DEGENERATE_CONSTRUCTION",
            "milestone_verdict": "INDETERMINATE_DEGENERATE_CONSTRUCTION",
            "reason": (
                "canonical delta_coh = (gamma/pi) * max nu_f "
                "yielded fewer than 2 inter-prime edges; "
                "Euler-orthogonality not broken; amendment required."
            ),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        return report

    L = weighted_laplacian(G)
    M = build_remesh_iteration_matrix(ALPHA, TAU_LOCAL, TAU_GLOBAL)
    T = build_composed_iteration_matrix(M, L, ETA_IL)
    N = L.shape[0]
    dim_joint = T.shape[0]
    eigvals_T, _ = eig(T)

    canonical = f7a_diagnostic(eigvals_T, "canonical_REMESH_o_IL_aug")
    canonical["dim_joint"] = int(dim_joint)
    canonical["N_nodes"] = int(N)
    canonical["delta_coh"] = float(delta_coh)
    canonical["n_inter_prime_edges_canonical"] = int(n_inter)

    # -- controls
    goe = control_goe(dim_joint, rng)
    poisson = control_poisson(dim_joint, rng)
    shuffled, n_inter_shuffled = control_shuffled_prime(M, rng)
    remesh_iso = control_remesh_isolated(M)
    random_aug = control_random_augmentation(M, n_inter, rng)
    riemann_ref = reference_riemann_d_gue(K_REF_RIEMANN)

    # -- F8 structural condition
    d_can = canonical["D_GUE"]
    d_shuf = shuffled["D_GUE"]
    d_rand = random_aug["D_GUE"]

    if d_can is None or d_shuf is None:
        f8_satisfied = False
        f8_delta = None
    else:
        f8_delta = abs(d_can - d_shuf)
        f8_satisfied = f8_delta >= F8_FLOOR

    # -- pre-registered F7 verdict
    if d_can is None or d_shuf is None or d_rand is None:
        f7a_verdict = "INDETERMINATE_INVALID_PROJECTION"
    elif not f8_satisfied:
        f7a_verdict = "INDETERMINATE_DEGENERATE_CONSTRUCTION"
    elif (
        d_can < D_SUPPORTED_MAX
        and d_can < d_shuf - D_SHUFFLE_MARGIN
        and d_can < d_rand - D_RANDOM_MARGIN
    ):
        f7a_verdict = "SUPPORTED"
    elif d_can > D_REFUTED_MIN or d_can >= d_shuf - D_SHUFFLE_MARGIN:
        f7a_verdict = "REFUTED"
    else:
        f7a_verdict = "INDETERMINATE_OTHER"

    # -- pre-registered milestone verdict
    if f7a_verdict == "SUPPORTED":
        milestone_verdict = "B1_MODIFIED_GRAPH_POTENTIALLY_OPEN_REQUIRES_REPLICATION"
    elif f7a_verdict == "REFUTED":
        milestone_verdict = (
            "B1_MODIFIED_GRAPH_REFUTED_FOR_CANONICAL_INTER_PRIME_COUPLING"
        )
    else:
        milestone_verdict = f"B1_MODIFIED_GRAPH_{f7a_verdict}"

    report = {
        "milestone": "R-inf-1c",
        "section_ref": "TNFR_RIEMANN_RESEARCH_NOTES.md §13vicies-novies.12",
        "canonical_config": {
            "graph": "P14 prime-ladder + canonical inter-prime augmentation",
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
            "delta_coh": float(delta_coh),
            "delta_coh_derivation": "(gamma/pi) * k_max * log(p_max)",
            "kappa_gamma_over_pi": float(KAPPA),
            "n_inter_prime_edges_canonical": int(n_inter),
            "n_inter_prime_edges_shuffled": int(n_inter_shuffled),
            "inter_prime_weight_law": (
                "w_ij = (gamma/pi) * exp(-|Delta nu_f|^2 / (2 delta_coh^2))"
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
                    f"D_canonical < D_random   - {D_RANDOM_MARGIN}"
                ),
                "REFUTED": (
                    f"D_canonical > {D_REFUTED_MIN} OR "
                    f"(D_canonical >= D_shuffled - {D_SHUFFLE_MARGIN} "
                    f"AND F8 SATISFIED)"
                ),
                "F8_SATISFIED": f"|D_canonical - D_shuffled| >= {F8_FLOOR}",
            },
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
            "N5_random_augmentation": random_aug,
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
    print("R-inf-1c  (T_aug = S_IL^aug . M_REMESH on G_P14 + canonical Kuramoto-U3)")
    print("=" * 72)
    print(f"Graph: {cfg['graph']}")
    print(
        f"  N={cfg.get('N_nodes', '?')} nodes, "
        f"dim(joint)={cfg.get('dim_joint_state', '?')}"
    )
    print(f"  delta_coh = {cfg['delta_coh']:.6f}  " f"({cfg['delta_coh_derivation']})")
    print(
        f"  inter-prime edges: canonical={cfg['n_inter_prime_edges_canonical']}, "
        f"shuffled={cfg.get('n_inter_prime_edges_shuffled', '?')}"
    )
    print(
        f"REMESH: alpha={cfg.get('alpha', ALPHA)}, "
        f"tau_l={cfg.get('tau_l', TAU_LOCAL)}, "
        f"tau_g={cfg.get('tau_g', TAU_GLOBAL)}; "
        f"IL: eta={cfg.get('eta_IL', ETA_IL)}"
    )
    print()
    f8 = report.get("f8_structural_condition")
    if f8 is not None:
        d_str = (
            "N/A"
            if f8["delta_D_can_minus_shuf_abs"] is None
            else f"{f8['delta_D_can_minus_shuf_abs']:.4f}"
        )
        print(f"F8 structural condition (|D_can - D_shuf| >= {f8['floor']}):")
        print(f"  |Delta D|  = {d_str}   satisfied = {f8['satisfied']}")
        print()
    print("F7-A KS distance vs GUE Wigner surmise:")
    print(f"  {'label':32s} {'kind':14s} {'#spacings':>10s} {'D_GUE':>10s}")
    diags = [report["canonical"]]
    diags.extend(
        [
            report["controls"][k]
            for k in (
                "N1_GOE",
                "N2_Poisson",
                "N3_shuffled_prime",
                "N4_REMESH_isolated",
                "N5_random_augmentation",
            )
        ]
    )
    diags.append(report["riemann_reference"])
    for diag in diags:
        d_str = "N/A" if diag["D_GUE"] is None else f"{diag['D_GUE']:.4f}"
        kind = diag.get("projection_kind", "iid_or_zeros")
        print(
            f"  {diag['label']:32s} {kind:14s} "
            f"{diag['n_spacings']:>10d} {d_str:>10s}"
        )
    print()
    print(f"F7-A verdict      : {report['f7a_verdict']}")
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
        / "remesh_infinity_riemann_modified_graph.json",
    )
    args = parser.parse_args()
    report = run_milestone(args.out)
    _print_summary(report)
    print(f"\nReport written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
