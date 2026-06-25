#!/usr/bin/env python3
"""
Example 122 — Factorization in the Phase Sector: the Complex Spectrum Completes
the Factor-Coset Recovery the Real Sector Misses
==============================================================================

Example 117 (Reading B, real sector) recovered the factor coset (i mod p) of a
semiprime n = p*q as an η²→1 Fourier mode of the EMERGENT diffusion spectrum —
but only PARTIALLY: it missed the high-frequency factor modes of 51 = 3*17 and
91 = 7*13. Example 119/120 showed the missing content lives in the PHASE: for
n ≡ 3 (mod 4) the residue graph is directed (a Paley tournament) and the
canonical emergent operator's spectrum is COMPLEX. This example uses that
complex phase sector to COMPLETE the factor-coset recovery.

The structural fact (CRT, present in both sectors)
--------------------------------------------------
For n = p*q the factor coset (i mod p) corresponds to the Fourier frequencies
k = multiples of the cofactor q. A pure Fourier mode exp(2πi k j / n) with k a
multiple of q is CONSTANT within each coset (i mod p), so it is an EXACT
eigenvector of the emergent operator (a circulant / Cayley digraph). This holds
for BOTH the undirected (real) and directed (complex) residue operator — the
factor coset is CRT/circulant structure (Z_n ≅ Z_p × Z_q), verified to machine
precision (eigenvector residual ~1e-14).

Why the real sector misses 51 and 91
-------------------------------------
The undirected residue operator is SYMMETRIC: its eigenvalues come in
degenerate pairs (λ_k = λ_{n−k}). When the factor-coset frequency lands in a
degenerate eigenspace, the eigensolver returns an arbitrary real combination
that SCRAMBLES the coset structure, so the η² test fails (51, 91). The directed
operator is NON-SYMMETRIC (a non-self-adjoint circulant): its eigenvalues are
complex Gauss sums, which are LESS
degenerate and ISOLATE the factor-coset mode — so the complex spectrum exposes
the very modes the real sector loses.

Doctrine compliance
-------------------
Everything uses the SAME canonical emergent operator —
`structural_diffusion_operator` (the literal ΔNFR EPI channel, L_rw = I − D⁻¹W)
— on the residue (di)graph. The complex eigenvectors ARE the emergent geometry
on a directed graph. Only arithmetic input: x² mod n.

Three measured results
----------------------
R1 THE FACTOR COSET IS A SHARED EIGENVECTOR. The Fourier mode at k = cofactor
   is an eigenvector of BOTH the undirected and directed emergent operator
   (residual ~1e-14). The factor structure is CRT, present in both spectra.

R2 THE COMPLEX SECTOR COMPLETES THE RECOVERY. Scanning prime candidate
   divisors d ≤ √n and reading the smallest d whose best complex-η² mode
   exceeds 0.9 recovers the smallest prime factor for 10/10 tested semiprimes —
   including 51 and 91. The real undirected sector recovers only 8/10 (it fails
   on exactly 51 and 91, the example-117 caveat).

R3 THE SIGNAL IS STRUCTURAL (shuffle control). Permuting the node labels
   collapses the factor-coset η² from 1.0 to ~0.1–0.2: the signature is the
   CRT/circulant structure of the residue digraph, not a numerical artefact.

Honest scope
------------
This re-expresses CRT / Fourier PERIOD structure (the same structure Shor's
algorithm exploits via period finding) in the canonical emergent spectrum. It
is NOT a new or fast factoring algorithm: the candidate scan is O(√n) prime
divisors — the same order as trial division, NO speedup. The complex sector's
only advantage is reduced eigenvalue degeneracy (a linear-algebra fact about
circulant / Cayley digraphs), which isolates the factor-coset mode. The result
COMPLETES example 117's partial real-sector recovery (8/10 → 10/10) via the
phase sector; it closes no open problem and provides no cryptographic threat.

References
----------
- src/tnfr/physics/structural_diffusion.py (structural_diffusion_operator)
- examples/08_emergent_geometry/117_emergent_geometry_residue_graph.py (real sector, partial)
- examples/08_emergent_geometry/119_phase_sector_directed_residue.py (the complex spectrum)
- examples/08_emergent_geometry/120_symmetry_wall_substrate_vs_spectrum.py (why arithmetic is spectral)
- theory/TNFR_NUMBER_THEORY.md §9.9 (this example; phase-sector factorization)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np

from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_graph_undirected(n):
    """Undirected residue graph (real sector): edge (i,j) iff +-(i-j) is a QR."""
    R = _qr(n)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (i - j) % n
            if d in R or (n - d) in R:
                G.add_edge(i, j)
    return G


def residue_digraph(n):
    """Directed residue graph (phase sector): edge i->j iff (j-i) mod n is QR."""
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


def _coset_eta2(vec, n, p):
    """eta^2 with COMPLEX means: is the eigenvector a function of (i mod p)?

    The factor-coset Fourier mode is constant within each coset (i mod p), so
    complex-mean eta^2 -> 1. (Magnitude is flat for a Fourier mode, so one MUST
    use complex means, not |vec|.)
    """
    labels = np.array([i % p for i in range(n)])
    v = np.asarray(vec, dtype=complex)
    grand = v.mean()
    total = float(np.sum(np.abs(v - grand) ** 2))
    if total < 1e-15:
        return 0.0
    between = 0.0
    for c in range(p):
        m = labels == c
        nc = int(m.sum())
        if nc:
            between += nc * float(np.abs(v[m].mean() - grand) ** 2)
    return float(between / total)


def _best_coset(eigvecs, n, p):
    """Max complex-eta^2 over all non-trivial emergent modes."""
    best = 0.0
    for j in range(1, eigvecs.shape[1]):
        e = _coset_eta2(eigvecs[:, j], n, p)
        if e > best:
            best = e
    return best


def _emergent_eigvecs(G):
    """Eigenvectors of the CANONICAL emergent operator, sorted by Re(lambda)."""
    _, L = structural_diffusion_operator(G)
    w, V = np.linalg.eig(L)
    return V[:, np.argsort(w.real)]


def _primes_up_to(m):
    return [d for d in range(2, m + 1) if all(d % k for k in range(2, d))]


def _recover_smallest_factor(n, eigvecs):
    """Smallest prime d<=sqrt(n) whose best emergent-eta^2 mode exceeds 0.9."""
    for d in _primes_up_to(int(math.isqrt(n)) + 1):
        e = _best_coset(eigvecs, n, d)
        if e > 0.9:
            return d, e
    return 0, 0.0


SEMIPRIMES = [
    (21, 3, 7),
    (33, 3, 11),
    (51, 3, 17),
    (57, 3, 19),
    (69, 3, 23),
    (85, 5, 17),
    (91, 7, 13),
    (93, 3, 31),
    (95, 5, 19),
    (115, 5, 23),
]


def experiment_1_shared_eigenvector():
    """R1: the factor-coset Fourier mode is an eigenvector of both operators."""
    print("=" * 74)
    print("EXPERIMENT 1: The Factor Coset Is a Shared Eigenvector (CRT)")
    print("=" * 74)
    print("The Fourier mode at k=cofactor is constant within each coset (i mod")
    print("p), so it is an EXACT eigenvector of the emergent operator on BOTH")
    print("the undirected and directed residue graph (CRT / circulant).")
    print()
    print(
        f"  {'n':>4} {'p':>3} {'q':>3} | {'undirected resid':>17} "
        f"{'directed resid':>16}"
    )
    for n, p, q in SEMIPRIMES[:6]:
        _, Lu = structural_diffusion_operator(residue_graph_undirected(n))
        _, Ld = structural_diffusion_operator(residue_digraph(n))
        j = np.arange(n)
        fm = np.exp(2j * np.pi * q * j / n)  # k = cofactor q

        def resid(L, f):
            lam = (f.conj() @ (L @ f)) / (f.conj() @ f)
            return float(np.linalg.norm(L @ f - lam * f))

        print(
            f"  {n:>4} {p:>3} {q:>3} | {resid(Lu, fm):>17.2e} "
            f"{resid(Ld, fm):>16.2e}"
        )
    print()
    print("  -> ~1e-14: the factor coset is CRT structure in both spectra.")


def experiment_2_complex_completes():
    """R2: complex sector recovers 10/10, real sector only 8/10."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: The Complex Phase Sector Completes the Recovery")
    print("=" * 74)
    print("Scan prime candidate divisors d<=sqrt(n); the smallest d whose best")
    print("emergent-eta^2 mode exceeds 0.9 is the recovered factor. Real")
    print("(undirected) vs complex (directed) emergent spectrum.")
    print()
    print(
        f"  {'n':>4} {'true p':>7} | {'real d':>7} {'ok?':>5} | "
        f"{'cplx d':>7} {'ok?':>5}"
    )
    rec_r = rec_c = 0
    for n, p, q in SEMIPRIMES:
        dr, _ = _recover_smallest_factor(
            n, _emergent_eigvecs(residue_graph_undirected(n))
        )
        dc, _ = _recover_smallest_factor(n, _emergent_eigvecs(residue_digraph(n)))
        okr, okc = (dr == p), (dc == p)
        rec_r += int(okr)
        rec_c += int(okc)
        print(f"  {n:>4} {p:>7} | {dr:>7} {str(okr):>5} | " f"{dc:>7} {str(okc):>5}")
    print()
    print(
        f"  -> real undirected sector:  {rec_r}/{len(SEMIPRIMES)} "
        f"(misses 51, 91 -- the example-117 caveat)"
    )
    print(
        f"  -> complex directed sector: {rec_c}/{len(SEMIPRIMES)} "
        f"(COMPLETE: recovers 51 and 91 via the phase)"
    )


def experiment_3_shuffle_control():
    """R3: the factor-coset signal is structural (shuffle collapses eta^2)."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: The Signal Is Structural (Shuffle Control)")
    print("=" * 74)
    print("Permuting node labels destroys the CRT/circulant structure: the")
    print("factor-coset eta^2 collapses from ~1.0 to the random baseline.")
    print()
    print(f"  {'n':>4} {'p':>3} | {'canonical eta2':>15} {'shuffled eta2':>15}")
    for n, p, q in [(51, 3, 17), (91, 7, 13), (85, 5, 17), (115, 5, 23)]:
        V = _emergent_eigvecs(residue_digraph(n))
        e_can = _best_coset(V, n, p)
        rng = np.random.default_rng(3)
        e_shuf = _best_coset(V[rng.permutation(n), :], n, p)
        print(f"  {n:>4} {p:>3} | {e_can:>15.3f} {e_shuf:>15.3f}")
    print()
    print("  -> canonical ~1.0, shuffled ~0.1-0.2: structural, not artefact.")


def main():
    print()
    print("  TNFR Example 122: Factorization in the Phase Sector")
    print("  The Complex Spectrum Completes the Factor-Coset Recovery")
    print("  =======================================================")
    print()
    experiment_1_shared_eigenvector()
    experiment_2_complex_completes()
    experiment_3_shuffle_control()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("The factor coset (i mod p) of a semiprime n=p*q is CRT/circulant")
    print("structure -- a Fourier mode at k=cofactor, an exact eigenvector of")
    print("the canonical emergent operator on BOTH the undirected and directed")
    print("residue graph. The REAL undirected operator is symmetric, so its")
    print("degenerate eigenpairs scramble the factor mode for 51 and 91 (the")
    print("example-117 caveat). The COMPLEX directed operator has less")
    print("degenerate Gauss-sum eigenvalues that ISOLATE the mode, completing")
    print("the recovery: 10/10 vs the real sector's 8/10. HONEST SCOPE: this")
    print("re-expresses CRT / Fourier PERIOD structure (the content Shor's")
    print("algorithm exploits) in the emergent spectrum; the scan is O(sqrt n)")
    print("prime divisors -- same order as trial division, NO speedup, no")
    print("cryptographic threat. It completes 117 via the phase sector; it")
    print("closes no open problem.")


if __name__ == "__main__":
    main()
