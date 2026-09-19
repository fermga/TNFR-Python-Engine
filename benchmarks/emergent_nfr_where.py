"""Prepared ring nodal positions and an independent prime-count comparison.

Selected sinusoidal ring modes have regular sampled zeros. Symmetry or a
Courant nodal-domain bound alone does not force this geometry on general graphs.
Primes are constructed with a sieve and compared with the imported n/log(n)
formula. That comparison does not derive prime density from TNFR evolution.
Prime gaps, the Riemann zero-count remainder S(T), and a chosen representation
complement Fix(G)^perp are different objects; this script constructs no map
identifying them. Irregular prime gaps do not establish such an obstruction.
A zero pure-EPI pressure is an instantaneous rate condition, not the definition
of an NFR or a derived atom. No physical shell or prime-location law is tested.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import math

import numpy as np


def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    return np.flatnonzero(sieve)


def ring_nodal_positions(n, k):
    """Sample vertices where a chosen sinusoidal mode of C_n is zero."""
    idx = np.arange(n)
    v = np.cos(2 * np.pi * k * idx / n)
    return idx[np.abs(v) < 1e-9]


def main() -> None:
    print("=" * 70)
    print("EMERGENT NFR WHERE -- how far the nodal order carries locations")
    print("=" * 70)

    # M1 -- symmetric structure: the standing nodes are a REGULAR lattice
    print("\nM1 -- symmetric ring: standing nodes are a REGULAR lattice:")
    n = 24
    m1_ok = True
    for k in (1, 2, 3, 6):
        pos = ring_nodal_positions(n, k)
        gaps = np.diff(pos)
        const = bool(np.std(gaps) < 1e-9)
        m1_ok = m1_ok and const and len(pos) == 2 * k
        print(
            f"  C_{n} mode k={k}: {len(pos)} standing nodes, gap={gaps[0]}, "
            f"constant={const}"
        )
    print("  selected ring symmetry -> the displayed nodal spacing and degeneracies")
    assert m1_ok

    # M2 -- primes: the SMOOTH density (PNT) is carried
    print("\nM2 -- primes: smooth density pi(n) ~ n/log n is CARRIED:")
    P = primes_upto(20000)
    ratios = []
    for nn in (1000, 5000, 10000, 20000):
        pi_n = int(np.sum(P < nn))
        pnt = nn / math.log(nn)
        ratios.append(pi_n / pnt)
        print(f"  pi({nn})={pi_n}, n/log n={pnt:.0f}, ratio={pi_n / pnt:.3f}")
    smooth_ok = max(ratios) - min(ratios) < 0.06  # ~constant => carried
    print(
        f"  ratio ~ constant ({min(ratios):.2f}-{max(ratios):.2f}): "
        f"smooth node density carried={smooth_ok}"
    )
    assert smooth_ok

    # M3 -- primes: measured sieve gaps are not constant
    print("\nM3 -- primes: the supplied sieve has nonconstant gaps:")
    gaps = np.diff(P)
    irr = gaps.std() / gaps.mean()
    print(f"  prime gap std/mean={irr:.2f} (IRREGULAR; regular lattice -> 0)")
    print(f"  gaps range {gaps.min()} (twin primes) to {gaps.max()}: no")
    print("  constant spacing describes these sieve gaps.")
    print("  This does not exclude arbitrary symmetric matrices or graph models.")
    print("     No representation action on this prime-gap data is constructed.")
    print("     No map from these gaps to the zero-count remainder S(T) is given.")
    print(
        "     The observed gap variability alone supplies neither such map nor proof."
    )
    assert irr > 0.3 and gaps.min() < gaps.max() / 4

    print("\n" + "=" * 70)
    print("FINITE COMPARISONS:")
    print("The chosen ring modes have the displayed regular sampled zero sets.")
    print("This is not a theorem for every symmetric graph or operator.")
    print("Prime counts and gaps come from an explicit sieve.")
    print("The n/log(n) comparison is supplied mathematics, not a TNFR prediction.")
    print("No map equates prime gaps, S(T), and a representation complement.")
    print("No particle, atomic shell or physical space is identified here.")
    print("Zero pressure is an instantaneous pure-EPI rate condition.")
    print("Autonomous identity and the physical bridge remain separate open questions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
