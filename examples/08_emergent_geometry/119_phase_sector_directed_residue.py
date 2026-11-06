#!/usr/bin/env python3
"""Example 119: finite spectra of supplied directed residue graphs.

The graph uses the nonzero quadratic residues modulo each supplied integer n.
For primes n == 1 (mod 4), it is symmetric; for primes n == 3 (mod 4), it is a
Paley tournament. Those statements require the prime hypothesis, which is not
available for arbitrary composite inputs. The shared structural diffusion
operator supplies the random-walk Laplacian evaluated here.

Three finite comparisons are retained: the count of eigenvalues rounded to
four decimals versus SymPy prime labels for odd n in [5,119]; selected prime
powers versus their supplied factorizations; and the imaginary eigenvalue
magnitude versus sqrt(n)/(n-1) on eight chosen Paley-tournament primes.
A match in the finite scan is not a universal primality theorem, and the
factorization labels are independently supplied comparison data.

Complex eigenvalues are properties of this specified matrix. They do not
identify primitive nodal phase or a physical phase observable. The arithmetic
construction neither derives primes from nodal evolution nor establishes an
analytic Riemann-phase obstruction. The prime-power cases provide controls
against transferring the symmetric-graph criterion from example 117.

Owners and related finite controls:
- src/tnfr/physics/structural_diffusion.py
- examples/08_emergent_geometry/117_emergent_geometry_residue_graph.py
- examples/08_emergent_geometry/118_emergent_vs_classical_operator.py
- theory/TNFR_NUMBER_THEORY.md
- benchmarks/paley_bridge.py
- benchmarks/directed_paley_bridge.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np
from sympy import factorint, isprime

from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _qr(n):
    return {(x * x) % n for x in range(1, n)} - {0}


def residue_digraph(n):
    """Directed residue graph: edge i->j iff (j-i) mod n is a quadratic residue.

    For prime n ≡ 1 (mod 4) this is symmetric; for prime n ≡ 3 (mod 4) it is a
    Paley tournament (one directed edge per pair). The graph construction uses
    x² mod n.
    """
    R = _qr(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and ((j - i) % n) in R:
                G.add_edge(i, j)
    return G


def _emergent_spectrum(n):
    """Eigenvalues of the CANONICAL emergent operator on the residue digraph."""
    nodes, L = structural_diffusion_operator(residue_digraph(n))
    return np.linalg.eigvals(L)


def _n_distinct(ev, decimals=4):
    return len(np.unique(np.round(ev, decimals)))


def experiment_1_unified_primality():
    """R1: compare the rounded spectral count with prime labels on a finite range."""
    print("=" * 74)
    print("EXPERIMENT 1: Finite Complex-Spectrum Count versus Prime Labels")
    print("=" * 74)
    print()
    print("The canonical emergent operator on the DIRECTED residue graph. For")
    print("prime n=3 mod4 the spectrum is COMPLEX (Paley tournament). Compare:")
    print("'3 eigenvalues rounded to 4 decimals' against supplied prime labels.")
    print()
    print(
        f"  {'n':>4} {'mod4':>5} {'prime':>6} {'n_distinct':>11} "
        f"{'max|Im|':>9} {'3-rigid?':>9}"
    )
    for n in (7, 11, 13, 19, 23, 29, 31, 37, 15, 21, 33, 35):
        ev = _emergent_spectrum(n)
        d = _n_distinct(ev)
        mi = float(np.max(np.abs(ev.imag)))
        ok = "YES" if (d == 3) == isprime(n) else "MISS"
        print(
            f"  {n:>4} {n % 4:>5} {str(isprime(n)):>6} {d:>11} " f"{mi:>9.4f} {ok:>9}"
        )
    # full sweep
    correct = sum(
        (_n_distinct(_emergent_spectrum(n)) == 3) == isprime(n)
        for n in range(5, 120, 2)
    )
    print()
    print(
        f"  full sweep odd n in [5,119]: "
        f"'3 distinct == prime' correct = {correct}/58"
    )
    print(
        "  -> finite agreement on the declared candidates; no converse theorem follows."
    )
    print()


def experiment_2_prime_powers():
    """R2: compare selected prime-power spectra with supplied factorization labels."""
    print("=" * 74)
    print("EXPERIMENT 2: Selected Prime-Power Controls")
    print("=" * 74)
    print()
    print("Example 117's real operator made 49=7^2 rigid (3 distinct), the")
    print("symmetric-graph counterexample. Compare these directed residue spectra:")
    print()
    print(f"  {'n':>5} {'factorization':>14} {'n_distinct':>11} {'verdict':>9}")
    for n in (7, 9, 11, 25, 13, 27, 49, 121):
        d = _n_distinct(_emergent_spectrum(n))
        fac = dict(factorint(n))
        verdict = "prime" if d == 3 else ("p^k" if len(fac) == 1 else "comp")
        print(f"  {n:>5} {str(fac):>14} {d:>11} {verdict:>9}")
    print()
    print("  -> 9,25,49,121 (prime squares) give 4 distinct eigenvalues, NOT 3.")
    print("     This is a finite matrix comparison; factorint supplies the labels.")
    print()


def experiment_3_phase_encodes_sqrt_n():
    """R3: compare selected Paley-tournament spectra with their Gauss-sum formula."""
    print("=" * 74)
    print("EXPERIMENT 3: Selected Imaginary Eigenvalues versus the Prime-Case Formula")
    print("=" * 74)
    print()
    print("For n=3 mod4 primes the residue digraph is a Paley tournament with")
    print("adjacency eigenvalues (-1 +/- i*sqrt(n))/2. The diffusion operator's")
    print("max|Im(lambda)| inherits this, scaling as sqrt(n)/(n-1).")
    print()
    print(f"  {'n':>4} {'max|Im|':>10} {'sqrt(n)/(n-1)':>14} {'ratio':>8}")
    for n in (7, 11, 19, 23, 31, 43, 47, 59):
        if not isprime(n):
            continue
        mi = float(np.max(np.abs(_emergent_spectrum(n).imag)))
        ref = np.sqrt(n) / (n - 1)
        print(f"  {n:>4} {mi:>10.4f} {ref:>14.4f} {mi / ref:>8.3f}")
    print()
    print("  -> ratio ~ 1: these imaginary eigenvalues match the supplied prime-case")
    print("     sqrt(n) formula; they do not identify a primitive nodal phase.")
    print()


def main():
    print()
    print("  TNFR Example 119: Finite Directed Residue Spectra")
    print("  Selected prime labels, prime powers and prime-case spectral formulas")
    print("  ==============================================================")
    print()
    experiment_1_unified_primality()
    experiment_2_prime_powers()
    experiment_3_phase_encodes_sqrt_n()
    print("=" * 74)
    print("FINITE COMPARISON SCOPE")
    print("=" * 74)
    print("The supplied residue graphs and random-walk Laplacians give finite")
    print("spectral comparisons against supplied prime and factorization labels.")
    print("The prime-case Gauss-sum formula has its stated prime hypothesis.")
    print("Neither a universal primality criterion nor a physical or analytic")
    print("phase identification follows from these finite matrix observations.")
    print()


if __name__ == "__main__":
    main()
