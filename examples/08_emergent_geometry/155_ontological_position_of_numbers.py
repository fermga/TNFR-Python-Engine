#!/usr/bin/env python3
"""Example 155: finite spectral arithmetic encodings and their information loss.

Five supplied constructions are compared: eigenspace dimensions on symmetric
graphs; Cartesian-product combinatorial spectra; quadratic-residue spectral
ranks versus primality labels on odd integers 5 through 47; restricted
prime-power/type encodings using exponent blocks 1, 2 and 3; and scalar-rank
collisions. Modular arithmetic, graph families and the rank-block table are
inputs. SymPy primality and factorization provide comparison labels only.

An invariant eigenspace can contain several irreducible sectors. Its dimension
alone does not determine a symmetry group. On the selected regular graphs,
D-A and L_rw differ by the degree; that identity does not make arbitrary
equivariant operators spectrally identical.

Rank equality for 15 and 35 loses prime identities, and multiple exponent
multisets can give rank 36 within the chosen block table. Those losses concern
this observation map only. They neither identify an analytic zeta symmetry
complement nor prove an RH obstruction. Finite encoding checks do not derive
integers, their physical realization or autonomous arithmetic from the nodal
identity. See theory/TNFR_NUMBER_THEORY.md sections 9.5-9.7 and
theory/TNFR_STRUCTURAL_OBSERVABILITY.md#6-limits-beyond-linear-symmetry.
"""

import os
import sys

import networkx as nx
import numpy as np
import sympy  # ORACLE only: comparison primality and factorization labels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.mathematics.number_theory import residue_network_rank

# Restricted exponent-block table for this supplied residue-rank encoding.
# The example checks only the listed prime powers and exponent values.
_RHO_PRIME_POWER = {1: 3, 2: 4, 3: 6}
_RANK_BLOCKS = sorted(_RHO_PRIME_POWER.values())  # [3, 4, 6]
_BLOCK_TO_EXP = {v: k for k, v in _RHO_PRIME_POWER.items()}


def _laplacian_degeneracies(G: nx.Graph) -> set[int]:
    """Numerically clustered multiplicities of L_rw via its symmetric twin.

    Invariant eigenspaces can contain several irreducible sectors. For the
    supplied regular graphs L_rw and D-A differ by a scalar, but arbitrary
    equivariant operators need not have identical eigenspaces or degeneracies.
    """
    from tnfr.mathematics.spectral import get_laplacian_spectrum

    eig, _ = get_laplacian_spectrum(G, operator="symmetric")
    _, counts = np.unique(np.round(np.real(eig), 6), return_counts=True)
    return {int(c) for c in counts}


def _exponent_multisets_from_rank(rank: int) -> list[list[int]]:
    """Exponent multisets consistent with the selected rank-block table.

    Factor ``rank`` into the blocks {3, 4, 6} assigned to exponents {1, 2, 3}.
    A unique result is unique within this restricted dictionary only. Multiple
    results expose information lost by the scalar rank.
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
    """Layer 1: observe eigenspace dimensions on supplied symmetric graphs."""
    print("=" * 72)
    print("EXPERIMENT 1: eigenspace dimensions on supplied symmetric graphs")
    print("=" * 72)
    print()
    print("The supplied L_rw = I - D^-1 W commutes with the graph action; its")
    print("eigenspaces are invariant and can combine several irreducible sectors.")
    print("Their measured dimensions alone do not determine the symmetry group.")
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
        print(
            f"  {name:13s}: degeneracies {sorted(degs)} -> selected multiplicity {expect} observed? "
            f"{'YES' if emerged else 'NO'}"
        )
    assert all_ok, "selected multiplicity comparison failed"
    print()
    print("OBSERVED: 2 @ triangle, 3 @ tetrahedron, 5 @ icosahedron.")
    print()


def experiment_2_operations():
    """Compare the supplied Cartesian-product combinatorial spectrum."""
    print("=" * 72)
    print("EXPERIMENT 2: the supplied Cartesian-product spectral sum identity")
    print("=" * 72)
    print()
    A, B = nx.complete_graph(3), nx.path_graph(3)

    def lap_spec(G):
        # Cartesian-product additivity {lambda_i + mu_j} is a theorem of the
        # COMBINATORIAL graph Laplacian specifically (it fails for the
        # normalized operator), so this layer legitimately uses D - A -- the
        # graph-connectivity object, not a claim about the emergent ΔNFR L_rw.
        return np.round(
            np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float)), 3
        )

    la, lb = lap_spec(A), lap_spec(B)
    prod = lap_spec(nx.cartesian_product(A, B))
    outer_sum = sorted({round(x + y, 3) for x in la for y in lb})
    emerges = sorted(set(prod)) == outer_sum
    print(f"  K3 Laplacian spectrum:   {sorted(set(la))}")
    print(f"  P3 Laplacian spectrum:   {sorted(set(lb))}")
    print(f"  K3 [] P3 == outer-SUM?   {emerges}  (combinatorial spectral identity)")
    assert emerges, "Cartesian-product spectral sum comparison failed"
    print()
    print("OBSERVED: the supplied Cartesian product has the expected spectral sums.")
    print()


def experiment_3_spectral_primality():
    """Compare residue spectral rank with prime labels on odd n=5,...,47."""
    print("=" * 72)
    print("EXPERIMENT 3: finite residue-rank classification")
    print("=" * 72)
    print()
    print("rho(n) = #distinct eigenvalues of the directed residue diffusion")
    print("operator built from supplied modular squares; labels use SymPy.isprime.")
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
    print("OBSERVED: rho=3 matches the prime labels for odd n from 5 through 47.")
    print("This finite encoding check does not derive arithmetic from nodal dynamics.")
    print()


def experiment_4_arithmetic_emerges():
    """Compare restricted rank products and exponent-type candidates."""
    print("=" * 72)
    print("EXPERIMENT 4: restricted arithmetic encoding in residue ranks")
    print("=" * 72)
    print()
    # (a) rho(p^a) depends only on the exponent a
    print("(a) selected prime-power ranks for a in {1,2,3}:")
    for a in (1, 2, 3):
        ranks = {residue_network_rank(p**a) for p in (3, 5, 7, 11)}
        print(f"    a={a}: rho(p^{a}) = {ranks.pop()} for the listed p=3,5,7,11")
    print()
    # (b) Check the selected rank products against factorization labels.
    print("(b) selected values: rho(n) versus prod rho(p^a) over supplied factors:")
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
        # (c) Enumerate exponent types inside the restricted block dictionary.
        cands = _exponent_multisets_from_rank(rho_spectral)
        unique = len(cands) == 1 and cands[0] == type_true
        recovered_ok &= unique
        omega_em = sum(cands[0]) if cands else None
        tau_em = int(np.prod([a + 1 for a in cands[0]])) if cands else None
        print(
            f"    n={n:3d}: rho={rho_spectral:2d}  type{type_true}  mult={mult}"
            f"  -> Omega={omega_em} tau={tau_em}"
            f"  (oracle Omega={sum(type_true)} tau={int(sympy.divisor_count(n))})"
        )
    assert ok_mult, "rho multiplicativity failed"
    assert recovered_ok, "type recovery failed for the demo range"
    print()
    print("OBSERVED: the selected rank products and restricted type candidates")
    print("match their arithmetic labels. Uniqueness uses the supplied block table;")
    print("it is not a general factorization or physical emergence result.")
    print()


def experiment_5_the_wall():
    """Show lost prime identities and type ambiguity for the scalar rank."""
    print("=" * 72)
    print("EXPERIMENT 5: information lost by the scalar-rank observation")
    print("=" * 72)
    print()
    # rho cannot separate two semiprimes with the same type
    r15, r35 = residue_network_rank(15), residue_network_rank(35)
    print(f"  rho(15=3x5) = {r15},  rho(35=5x7) = {r35}  -> identical")
    print("  This rank value cannot distinguish these two supplied prime pairs.")
    same_rank_diff_primes = r15 == r35
    # The restricted dictionary also has a cross-type collision.
    collide = _exponent_multisets_from_rank(36)
    print(f"  rho = 36 is consistent with types {collide} (a spectral COLLISION)")
    has_collision = len(collide) >= 2
    assert same_rank_diff_primes and has_collision, "rank-collision control failed"
    print()
    print("OBSERVED: equal ranks lose prime identity and can also lose type.")
    print("This concerns the selected scalar observation; no analytic zeta map")
    print("or universal symmetry obstruction is established.")
    print()


def main():
    print()
    print("#" * 72)
    print("# FINITE SPECTRAL ENCODINGS AND INFORMATION LOSS (example 155)")
    print("#" * 72)
    print()
    experiment_1_cardinals()
    experiment_2_operations()
    experiment_3_spectral_primality()
    experiment_4_arithmetic_emerges()
    experiment_5_the_wall()
    print("=" * 72)
    print("ALL DECLARED FINITE COMPARISONS PASSED")
    print("=" * 72)
    print()
    print("These calculations compare supplied graph and arithmetic encodings.")
    print("The rank-collision controls delimit what their observations retain.")
    print("They select no physical model or autonomous mechanism for arithmetic.")


if __name__ == "__main__":
    main()
