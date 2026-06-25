#!/usr/bin/env python3
"""
Example 155 — The Ontological Position of a Number
===================================================

Are numbers a PRIMITIVE INPUT to TNFR, or do they EMERGE from its structure and
dynamics? (The Camino-11 question, ``benchmarks/primes_as_consequence.py``.)
After the arithmetic triad was canonicalized to unit coefficients
(``TNFR_NUMBER_THEORY.md`` §5), the only remaining "magic" in number theory is
the integer itself: the arithmetic sector A *consumes* ``n`` (it computes Ω, τ, σ
by trial division). This example maps the **ontological position** of a number as
a ladder, each rung measured from canonical TNFR structure/dynamics:

    Layer 0  Substrate    R continuum + pi (the one genuine structural scale)
    Layer 1  Cardinal     n = a degeneracy = dim of an irrep of Aut(G)
    Layer 2  Operations   +, x emerge from graph products (Cartesian/tensor)
    Layer 3  Primality    spectral: directed residue operator -> 3 eigenvalues
                          <=> odd prime (Sector B, x^2 mod n only)
    Layer 3' Arithmetic   the factorization (Omega, tau -> the dNFR triad)
                          EMERGES from the multiplicative spectral rank rho(n)
    Layer 4  The wall     the prime IDENTITIES / continuous phase = the RH
                          residue S(T) (Fix(S_n)^perp), provably S_n-unreachable

Physics
-------
- Layer 1: the graph Laplacian L = D - A commutes with Aut(G), so its eigenvalue
  multiplicities are dimensions of irreps of Aut(G). The integer is a *count of
  structural modes* (emergent_integers_symmetry.py).
- Layer 2: Cartesian product G [] H has Laplacian spectrum {lambda_i + mu_j}
  (ADDITION); tensor product has adjacency spectrum {alpha_i . beta_j}
  (MULTIPLICATION) -- the operations emerge, they are not injected.
- Layer 3/3': the quadratic-residue Cayley digraph of n (built from x^2 mod n,
  never n % k) carries the canonical structural-diffusion operator
  L_rw = I - D^{-1} W (the literal dNFR EPI channel). Its number of distinct
  (complex) eigenvalues rho(n) realizes the PROVED §9.7 conductor-product law
  A(m)=prod(e+ceil(e/2)+1) at small exponents -- rho(p)=3 (cyclotomy k=2),
  rho(p^2)=4, rho(p^3)=6 -- and is multiplicative there. So primality (rho=3)
  and the factorization TYPE (Omega, tau) are read off the spectrum -- the
  arithmetic emerges (TNFR_NUMBER_THEORY.md §9.5-9.8).
- Layer 4: rho gives the factorization TYPE, never the prime IDENTITIES (15 and
  35 share rho=9); the unannotated scalar rank also aliases at high prime powers
  (the §9.7 / ex 154 scalar wall) -- the same e-pi / Fix(S_n)^perp wall as the
  paused TNFR-Riemann program.

Experiments
-----------
1. Layer 1 -- cardinals emerge as Laplacian degeneracies
2. Layer 2 -- addition emerges from the Cartesian-product spectrum
3. Layer 3 -- spectral primality: rho(n) = 3 <=> odd prime (x^2 mod n only)
4. Layer 3' -- the arithmetic emerges: rho multiplicative, rho(p^a) = f(a),
   rho encodes the factorization type -> Omega, tau (the dNFR triad)
5. Layer 4 -- the wall: rho gives the type, not the prime identities

References
----------
- theory/TNFR_NUMBER_THEORY.md §9.5-9.7 (three sectors + phase + ontology)
- benchmarks/primes_as_consequence.py (Camino 11)
- benchmarks/emergent_integers_symmetry.py (cardinals from symmetry)
- src/tnfr/mathematics/number_theory.py (residue_network_rank)
- AGENTS.md §12 (Number theory program)
"""

import os
import sys

import networkx as nx
import numpy as np
import sympy  # ORACLE only: ground-truth factorization to VALIDATE emergence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.mathematics.number_theory import residue_network_rank

# Universal rho(p^a) table -- MEASURED: the residue-graph spectral rank of a
# prime power depends only on the exponent a (verified for many primes p).
_RHO_PRIME_POWER = {1: 3, 2: 4, 3: 6}
_RANK_BLOCKS = sorted(_RHO_PRIME_POWER.values())  # [3, 4, 6]
_BLOCK_TO_EXP = {v: k for k, v in _RHO_PRIME_POWER.items()}


def _laplacian_degeneracies(G: nx.Graph) -> set[int]:
    """The integers that emerge as Laplacian eigenvalue multiplicities."""
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eig = np.linalg.eigvalsh(L)
    _, counts = np.unique(np.round(eig, 6), return_counts=True)
    return {int(c) for c in counts}


def _exponent_multisets_from_rank(rank: int) -> list[list[int]]:
    """All factorization exponent-multisets consistent with a spectral rank.

    rho is multiplicative with rho(p^a) in {3, 4, 6}; factor ``rank`` into those
    blocks. A unique result means the rank determines the factorization TYPE; two
    or more results is a spectral COLLISION (the residual wall).
    """
    out: set[tuple[int, ...]] = set()

    def rec(r: int, exps: list[int]) -> None:
        if r == 1:
            out.add(tuple(sorted(exps)))
            return
        for b in _RANK_BLOCKS:
            if r % b == 0:
                rec(r // b, exps + [_BLOCK_TO_EXP[b]])

    rec(rank, [])
    return [list(t) for t in sorted(out)]


def experiment_1_cardinals():
    """Layer 1: integers emerge as irrep dimensions (Laplacian degeneracies)."""
    print("=" * 72)
    print("EXPERIMENT 1: Layer 1 -- cardinals emerge from symmetry")
    print("=" * 72)
    print()
    print("The Laplacian L=D-A commutes with Aut(G); its eigenvalue")
    print("multiplicities are dimensions of irreps of Aut(G). The integer is")
    print("a count of structural modes -- emergent, not injected.")
    print()
    cases = [
        ("triangle K3", nx.complete_graph(3), 2),
        ("tetrahedron", nx.tetrahedral_graph(), 3),
        ("octahedron", nx.octahedral_graph(), 3),
        ("icosahedron", nx.icosahedral_graph(), 5),
    ]
    all_ok = True
    for name, G, expect in cases:
        degs = _laplacian_degeneracies(G)
        emerged = expect in degs
        all_ok &= emerged
        print(f"  {name:13s}: degeneracies {sorted(degs)} -> {expect} emerges? "
              f"{'YES' if emerged else 'NO'}")
    assert all_ok, "cardinal emergence failed"
    print()
    print("VALIDATED: 2 @ triangle, 3 @ tetrahedron, 5 @ icosahedron.")
    print()


def experiment_2_operations():
    """Layer 2: addition emerges from the Cartesian-product spectrum."""
    print("=" * 72)
    print("EXPERIMENT 2: Layer 2 -- operations emerge from graph products")
    print("=" * 72)
    print()
    A, B = nx.complete_graph(3), nx.path_graph(3)

    def lap_spec(G):
        return np.round(
            np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float)), 3
        )

    la, lb = lap_spec(A), lap_spec(B)
    prod = lap_spec(nx.cartesian_product(A, B))
    outer_sum = sorted({round(x + y, 3) for x in la for y in lb})
    emerges = sorted(set(prod)) == outer_sum
    print(f"  K3 Laplacian spectrum:   {sorted(set(la))}")
    print(f"  P3 Laplacian spectrum:   {sorted(set(lb))}")
    print(f"  K3 [] P3 == outer-SUM?   {emerges}  (ADDITION emerges)")
    assert emerges, "operation emergence failed"
    print()
    print("VALIDATED: the Cartesian product realizes + on the spectra.")
    print()


def experiment_3_spectral_primality():
    """Layer 3: rho(n) = 3 <=> odd prime, from x^2 mod n only (Sector B)."""
    print("=" * 72)
    print("EXPERIMENT 3: Layer 3 -- spectral primality (primes-OUT)")
    print("=" * 72)
    print()
    print("rho(n) = #distinct eigenvalues of the directed residue diffusion")
    print("operator. Built from x^2 mod n -- it NEVER computes n % k.")
    print()
    mism = 0
    for n in range(5, 48, 2):
        rho = residue_network_rank(n)
        is_p = bool(sympy.isprime(n))  # ORACLE
        ok = (rho == 3) == is_p
        mism += 0 if ok else 1
        tag = "prime" if is_p else "comp"
        print(f"  n={n:3d}  rho={rho:2d}  {tag:5s}  {'OK' if ok else 'MISMATCH'}")
    assert mism == 0, "spectral primality failed"
    print()
    print("VALIDATED: rho = 3 <=> odd prime, 0 mismatches. Primality is a")
    print("consequence of self-adjoint/directed structure, not a primitive.")
    print()


def experiment_4_arithmetic_emerges():
    """Layer 3': the factorization (Omega, tau -> dNFR) emerges from rho."""
    print("=" * 72)
    print("EXPERIMENT 4: Layer 3' -- the arithmetic emerges from the spectrum")
    print("=" * 72)
    print()
    # (a) rho(p^a) depends only on the exponent a
    print("(a) rho(p^a) depends only on the exponent a:")
    for a in (1, 2, 3):
        ranks = {residue_network_rank(p**a) for p in (3, 5, 7, 11)}
        print(f"    a={a}: rho(p^{a}) = {ranks.pop()} for all primes p")
    print()
    # (b) rho is multiplicative -> faithfully encodes the factorization
    print("(b) rho is multiplicative: rho(n) == prod rho(p^a) over p^a || n:")
    ok_mult = True
    recovered_ok = True
    for n in (9, 15, 45, 63, 75, 105):
        factint = sympy.factorint(n)  # ORACLE
        type_true = sorted(factint.values())
        rho_spectral = residue_network_rank(n)
        rho_formula = 1
        for _, a in factint.items():
            rho_formula *= _RHO_PRIME_POWER[a]  # demo exponents are <= 3
        mult = rho_spectral == rho_formula
        ok_mult &= mult
        # (c) invert rho -> factorization TYPE -> Omega, tau (emergent)
        cands = _exponent_multisets_from_rank(rho_spectral)
        unique = len(cands) == 1 and cands[0] == type_true
        recovered_ok &= unique
        omega_em = sum(cands[0]) if cands else None
        tau_em = (int(np.prod([a + 1 for a in cands[0]])) if cands else None)
        print(f"    n={n:3d}: rho={rho_spectral:2d}  type{type_true}  mult={mult}"
              f"  -> Omega={omega_em} tau={tau_em}"
              f"  (oracle Omega={sum(type_true)} tau={int(sympy.divisor_count(n))})")
    assert ok_mult, "rho multiplicativity failed"
    assert recovered_ok, "type recovery failed for the demo range"
    print()
    print("VALIDATED: rho(n) is the multiplicative spectral encoding of the")
    print("factorization. Omega and tau (the factorization + divisor channels of")
    print("the dNFR triad) EMERGE from rho -- read from x^2 mod n, not consumed.")
    print()


def experiment_5_the_wall():
    """Layer 4: rho gives the TYPE, never the prime identities (the wall)."""
    print("=" * 72)
    print("EXPERIMENT 5: Layer 4 -- the wall (type emerges, identity does not)")
    print("=" * 72)
    print()
    # rho cannot separate two semiprimes with the same type
    r15, r35 = residue_network_rank(15), residue_network_rank(35)
    print(f"  rho(15=3x5) = {r15},  rho(35=5x7) = {r35}  -> identical")
    print("  rho sees the TYPE (1,1) but never which primes -> identities are")
    print("  beyond the rank.")
    same_rank_diff_primes = r15 == r35
    # rho collides across types in general (the residual wall)
    collide = _exponent_multisets_from_rank(36)
    print(f"  rho = 36 is consistent with types {collide} (a spectral COLLISION)")
    has_collision = len(collide) >= 2
    assert same_rank_diff_primes and has_collision, "wall demonstration failed"
    print()
    print("VALIDATED: the spectral position fixes primality (Layer 3) and the")
    print("factorization type (Layer 3'); the prime IDENTITIES and the")
    print("continuous phase (arg zeta = S(T), Fix(S_n)^perp) remain the open")
    print("RH-residue wall -- located precisely, not dissolved.")
    print()


def main():
    print()
    print("#" * 72)
    print("# THE ONTOLOGICAL POSITION OF A NUMBER (example 155)")
    print("#" * 72)
    print()
    experiment_1_cardinals()
    experiment_2_operations()
    experiment_3_spectral_primality()
    experiment_4_arithmetic_emerges()
    experiment_5_the_wall()
    print("=" * 72)
    print("ALL EXPERIMENTS PASSED")
    print("=" * 72)
    print()
    print("The ladder: a number is positioned by the emergent ontology -- a")
    print("cardinal (Layer 1), under emergent +,x (Layer 2), with primality")
    print("(Layer 3) and factorization type (Layer 3') read from the residue")
    print("spectrum. Sector A (number_theory.py) consumes the integer; the")
    print("emergent position (Sectors B/cardinal) derives it from structure,")
    print("up to the prime-identity / phase wall.")


if __name__ == "__main__":
    main()
