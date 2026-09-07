r"""Camino 18 - finite fixed-delay projector + Sense Index analysis.

This harness studies two separate finite constructions:

* T1-T2 verify the DFT projection onto modes fixed by both delays of the
  corrected finite cyclic REMESH surrogate. The fixed-mode period and rank
  are governed by gcd(tau_l, tau_g). The lcm is only a sample-alignment unit.
* T3-T4 measure the Sense Index response to phase dispersion and verify
  prime-relabeling invariance on a complete graph.

Tests:
  T1  The selected finite DFT mask defines an orthogonal projection
      (P^2=P, P=P^H, Parseval, rank=gcd).
  T2  Across the declared finite sweep, dim(range)=gcd and
      dim(complement)=N-gcd.
  T3  The Sense Index is a coherence-capacity functional: Si falls
      monotonically as phase dispersion (residue character) rises.
  T4  The Sense Index is S_n-degenerate: sorted Si on the complete
      prime graph is invariant under prime relabelling.

HONEST SCOPE (non-negotiable)
-----------------------------
The Fourier projection is a finite cyclic fixed-delay surrogate, not a
runtime tau_g -> infinity limit. Its complement is finite-dimensional for
every run. Finite DFT leakage does not identify the analytic support of
S(T), and graph relabeling invariance of Si does not locate S(T) in any
projector subspace. This harness proves no catalog-completeness statement
and does not advance G4 = RH, T-HP, B2, or B3.
"""

from __future__ import annotations

import logging
import math
import os
import sys

import numpy as np

# --- dual sys.path: benchmarks dir + ../src ------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
for _p in (_HERE, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# quieten the optional-orjson telemetry warning from tnfr.utils.io
logging.getLogger("tnfr").setLevel(logging.ERROR)

# --- guarded canonical imports -------------------------------------------
try:
    import networkx as nx

    from tnfr.metrics.sense_index import compute_Si
    from tnfr.riemann.remesh_infinity_residue_split import build_resonant_bin_mask
    from tnfr.riemann.von_mangoldt import build_prime_ladder_spectrum

    _HAVE_TNFR = True
except Exception as exc:  # pragma: no cover - import guard
    _HAVE_TNFR = False
    _IMPORT_ERROR = exc


# --- canonical constants -------------------------------------------------
TAU_L = 4  # local REMESH period
TAU_G = 8  # global delay; gcd(4, 8) = 4 fixes the common modes
N_MATRIX = 64  # window for the explicit projector matrix
DIM_SWEEP = (64, 128, 256, 512, 1024, 2048)
N_PRIMES = 50  # primes on the complete graph (T4)
SEED = 12345  # prime-relabelling seed (T4)
# Canonical engine SI weights (config/defaults_core.py SI_WEIGHTS) — the
# π-derived coherence-band hierarchy: alpha = π/(π+1), beta = π/(π+1)²,
# gamma = 1/(π+1)² (sums to 1 exactly).  Uniform across nodes, so the T3/T4
# dispersion and S_n-invariance arguments are weight-agnostic.
SI_WEIGHTS = {"alpha": 0.7585, "beta": 0.1832, "gamma": 0.0583}
DISP_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)  # phase-dispersion factors


# --- canonical helpers ---------------------------------------------------
def _fixed_delay_projection_matrix(
    n: int, tau_l: int = TAU_L, tau_g: int = TAU_G
) -> np.ndarray:
    """Explicit finite fixed-delay projector P = ifft . mask . fft.

    Uses the compatibility mask exposed by the legacy
    ``split_residue_by_remesh_infinity`` API.
    """
    mask = build_resonant_bin_mask(n, tau_l=tau_l, tau_g=tau_g)
    fwd = np.fft.fft(np.eye(n), axis=0)
    inv = np.fft.ifft(np.eye(n), axis=0)
    diag = np.diag(mask.astype(complex))
    return inv @ diag @ fwd


def _ring_phase_dispersion_si(n: int, spread: float) -> float:
    """Mean Si on a ring with alternating phases 0 / (spread*pi).

    nu_f and delta_nfr are uniform, so only the phase-coherence term
    of Si moves: this isolates Si's response to dispersion.  spread=0
    is full synchrony (coherent); spread=1 is maximal anti-alignment
    (residue character).
    """
    G = nx.cycle_graph(n)
    G.graph["SI_WEIGHTS"] = dict(SI_WEIGHTS)
    for i, node in enumerate(G.nodes()):
        theta = (i % 2) * spread * math.pi
        G.nodes[node].update({"nu_f": 1.0, "delta_nfr": 0.1, "phase": float(theta)})
    si = compute_Si(G, inplace=False)
    return float(np.mean(list(si.values())))


def _complete_prime_graph_si(order: np.ndarray) -> np.ndarray:
    """Sorted Si vector on K_n with prime-derived node attributes.

    ``order`` permutes which prime's attributes land on which node.
    On the complete graph K_n (invariant under all of S_n) any prime
    relabelling permutes nodes, so the *sorted* Si vector is an
    S_n-invariant of the construction.
    """
    spec = build_prime_ladder_spectrum(N_PRIMES, max_power=1)
    primes = np.asarray(spec.primes, dtype=float)
    logp = np.log(primes)
    nuf = logp / logp.max()  # normalised structural freq
    phase = primes % (2.0 * math.pi)  # bounded, prime-specific
    dnfr = 1.0 / primes  # bounded, prime-specific
    G = nx.complete_graph(N_PRIMES)
    G.graph["SI_WEIGHTS"] = dict(SI_WEIGHTS)
    for i, node in enumerate(G.nodes()):
        j = int(order[i])
        G.nodes[node].update(
            {
                "nu_f": float(nuf[j]),
                "delta_nfr": float(dnfr[j]),
                "phase": float(phase[j]),
            }
        )
    si = compute_Si(G, inplace=False)
    return np.array(sorted(si.values()))


# --- tests ---------------------------------------------------------------
def test_t1_coherence_is_orthogonal_projection() -> bool:
    """T1: the selected finite DFT mask is an orthogonal projection."""
    n = N_MATRIX
    p = _fixed_delay_projection_matrix(n)
    idem = float(np.max(np.abs(p @ p - p)))
    selfadj = float(np.max(np.abs(p - p.conj().T)))
    rank = int(round(float(np.real(np.trace(p)))))
    fixed_period = math.gcd(TAU_L, TAU_G)
    # Parseval / orthogonality on a deterministic signal
    rng = np.random.default_rng(0)
    f = rng.standard_normal(n)
    pf = np.real(p @ f)  # coherent part (range)
    qf = f - pf  # finite orthogonal-complement component
    parseval = abs((np.dot(pf, pf) + np.dot(qf, qf)) - np.dot(f, f))
    ortho = abs(float(np.dot(pf, qf)))
    ok = (
        idem < 1e-10
        and selfadj < 1e-10
        and rank == fixed_period
        and parseval < 1e-9
        and ortho < 1e-9
    )
    print("  [T1] finite fixed-delay DFT projection")
    print(f"       ||P^2 - P||  (idempotent)  : {idem:.2e}")
    print(f"       ||P - P^H||  (self-adjoint): {selfadj:.2e}")
    print(
        f"       rank = trace(P)            : {rank} "
        f"(expect gcd={fixed_period})"
    )
    print(f"       |Parseval residual|        : {parseval:.2e}")
    print(f"       |<selected, complement>|   : {ortho:.2e}")
    print("       (P^2=P and P=P^H => exact finite orthogonal projection)")
    print(f"       -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_t2_finite_complement_dimensions() -> bool:
    """T2: verify finite range/complement dimensions over the sweep."""
    fixed_period = math.gcd(TAU_L, TAU_G)
    rows = []
    ok = True
    prev_frac = 2.0
    for n in DIM_SWEEP:
        mask = build_resonant_bin_mask(n, tau_l=TAU_L, tau_g=TAU_G)
        dim_range = int(mask.sum())
        dim_ker = n - dim_range
        frac = dim_range / n
        rows.append((n, dim_range, dim_ker, frac))
        ok = ok and (dim_range == fixed_period) and (frac < prev_frac)
        prev_frac = frac
    print("  [T2] finite range/complement dimensions")
    print("       N      dim(range)  dim(ker)   coherent frac L/N")
    for n, dr, dk, fr in rows:
        print(f"       {n:<6} {dr:<11} {dk:<10} {fr:.6f}")
    print(
        f"       dim(range) = gcd = {fixed_period}: "
        f"{all(r[1] == fixed_period for r in rows)}"
    )
    print("       selected fraction decreases over this finite N sweep")
    print(f"       -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_t3_sense_index_is_coherence_functional() -> bool:
    """T3: Si falls monotonically across the declared dispersion sweep."""
    n = 60
    si_vals = [_ring_phase_dispersion_si(n, s) for s in DISP_SWEEP]
    # monotone non-increasing (tiny tolerance for round-off)
    monotone = all(si_vals[i + 1] <= si_vals[i] + 1e-9 for i in range(len(si_vals) - 1))
    margin = si_vals[0] - si_vals[-1]
    ok = monotone and (margin > 0.0)
    print("  [T3] Sense Index as a coherence-capacity functional")
    print("       phase-dispersion factor s -> mean Si (ring, N=60)")
    for s, v in zip(DISP_SWEEP, si_vals):
        print(f"       s={s:.2f}  (disp~{s:.2f})  mean Si = {v:.6f}")
    print(f"       monotone non-increasing : {monotone}")
    print(f"       Si(coherent) - Si(dispersed) = {margin:.4f} (> 0)")
    print("       (Si peaks at full synchrony s=0 and decays toward")
    print("        the dispersed endpoint over this declared sweep)")
    print(f"       -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_t4_sense_index_is_sn_degenerate() -> bool:
    """T4: sorted Si on K_n is invariant under prime relabelling ->
    Si is symmetric under the tested relabeling."""
    base = np.arange(N_PRIMES)
    shuf = base.copy()
    np.random.default_rng(SEED).shuffle(shuf)
    si_a = _complete_prime_graph_si(base)
    si_b = _complete_prime_graph_si(shuf)
    max_diff = float(np.max(np.abs(si_a - si_b)))
    spread = float(si_a.max() - si_a.min())
    ok = (max_diff < 1e-9) and (spread > 0.1)
    print("  [T4] S_n-degeneracy of the Sense Index on K_n")
    print(
        f"       sorted Si range : [{si_a.min():.4f}, "
        f"{si_a.max():.4f}] (non-trivial: {spread > 0.1})"
    )
    print(f"       max|sorted Si_canonical - Si_shuffled| : {max_diff:.2e}")
    print("       (machine-precision zero == Si is a symmetric")
    print("        functional of the prime attributes: Si in Fix(S_n),")
    print("        this statement is independent of the DFT projection)")
    print(f"       -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=" * 70)
    print("Camino 18 - Fixed-delay projector + Sense Index")
    print("Finite DFT projection and independent graph-symmetry checks")
    print("=" * 70)
    if not _HAVE_TNFR:
        print(f"SKIP: TNFR canonical imports unavailable: {_IMPORT_ERROR}")
        return 0
    print(
        f"sample alignment lcm={math.lcm(TAU_L, TAU_G)}; "
        f"fixed-mode gcd={math.gcd(TAU_L, TAU_G)}; "
        f"N_MATRIX={N_MATRIX}, N_PRIMES={N_PRIMES}"
    )
    print("-" * 70)

    results = [
        test_t1_coherence_is_orthogonal_projection(),
        test_t2_finite_complement_dimensions(),
        test_t3_sense_index_is_coherence_functional(),
        test_t4_sense_index_is_sn_degenerate(),
    ]
    n_pass = sum(1 for r in results if r)
    n_total = len(results)

    print("-" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} tests PASS")
    print("")
    print("Interpretation (honest scope):")
    print("  * The fixed-delay DFT mask is an orthogonal projection (T1).")
    print("    Its rank equals gcd(tau_l, tau_g), and its finite")
    print("    complement dimension is N - gcd over the tested sweep (T2).")
    print("  * The Sense Index Si is TNFR's node-level coherence-")
    print("    capacity metric (T3) and an S_n-symmetric functional")
    print("    (T4) on the tested complete graph. This does not identify")
    print("    any relation between Si and the analytic residue S(T).")
    print("")
    print("THESIS VERDICT: OPEN, by design. This closes nothing.")
    print("  G4 = RH OPEN. R+ and pi remain assumed.")
    print("  The finite projection is not a runtime infinity limit.")
    print("  These checks do NOT prove B2/B3/RH or catalog completeness.")
    print("=" * 70)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
