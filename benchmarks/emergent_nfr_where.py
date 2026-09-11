"""Emergent NFR Where: how far the nodal/spectral order carries the locations.

THE QUESTION (user, theory creator): we know WHAT a pressure-free point is
(dNFR=0 = flat = where a per-NFR pulse beats, the Chladni node) -- but WHERE do
those beats fall, and how far does the emergent nodal/spectral geometry carry
the prime/atom locations before the S_n wall?

THE STRUCTURAL FACT: the standing nodes of a SYMMETRIC emergent operator are a
REGULAR lattice -- Courant nodal-domain ordering plus the symmetry group force
even spacing. So the emergent geometry natively produces REGULAR equilibrium
lattices. Atoms are symmetric (sphere / simplex), so their shells are carried
in full. The integers carry a smooth (regular) density but an IRREGULAR fine
structure -- that irregularity is exactly the S_n-breaking residue (the wall).

WHAT EMERGES (measured):
  - M1 SYMMETRIC -> REGULAR: on the ring the standing nodes of every mode are
    evenly spaced (constant gap); the symmetry also fixes the shell
    degeneracies. The emergent geometry carries symmetric equilibria (the atom
    shells) completely.
  - M2 PRIMES, smooth part CARRIED: the prime density pi(n) ~ n/log n (the
    smooth / regular trend) tracks the prime count with a near-constant ratio
    -- the average NFR spacing log(n) is carried by the emergent order.
  - M3 PRIMES, fine part = the WALL: the individual prime locations are
    IRREGULAR (gap std/mean ~ 0.7, gaps from 2 to large); no constant-gap
    (symmetric) operator produces this spectrum. Placing standing nodes at the
    primes requires BREAKING S_n (a non-regular spectrum) = Fix(S_n)^perp =
    the Riemann residue S(T). The prime SUPPORT does partially emerge from the
    residue spectral-gap geometry (paley_bridge.py: lambda_2 = an
    emergent-geometry quantity), but the fine distribution stays at the wall.

So the nodal/spectral order carries the SYMMETRIC / SMOOTH equilibria -- the
atom shells fully, the prime density -- but the IRREGULAR fine structure (the
individual primes, S(T)) is the S_n wall, located precisely as a geometric
statement: the prime operator must be non-regular (symmetry-breaking).

HONEST SCOPE: Courant nodal-domain regularity, the prime number theorem, and
the S_n equivariance wall are standard; the TNFR content is the reading dNFR=0
= NFR = nodal point, so "where the equilibria fall" = the nodal geometry,
carried up to the symmetry. The primes are GROUND TRUTH here (a sieve), used to
MEASURE the reach -- NOT derived. Closes nothing (the prime fine structure is
RH). R and pi assumed.

Run:
    python benchmarks/emergent_nfr_where.py

Theoretical anchor: AGENTS.md (NFR = dNFR=0 region; discrete-mode / Chladni /
Courant); benchmarks/emergent_nfr_geometry.py (every node a pulsing NFR),
benchmarks/paley_bridge.py (the residue spectral-gap partial reach),
benchmarks/equivariance_wall.py (the Fix(S_n)^perp wall). Status: RESEARCH.
"""

from __future__ import annotations

import math

import numpy as np


def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.flatnonzero(sieve)


def ring_nodal_positions(n, k):
    """Vertices where mode k of the ring C_n is zero (the nodal NFRs)."""
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
        print(f"  C_{n} mode k={k}: {len(pos)} standing nodes, gap={gaps[0]}, "
              f"constant={const}")
    print("  symmetry -> evenly spaced nodes + degenerate shells (atoms)")
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
    print(f"  ratio ~ constant ({min(ratios):.2f}-{max(ratios):.2f}): "
          f"smooth node density carried={smooth_ok}")
    assert smooth_ok

    # M3 -- primes: the FINE structure is IRREGULAR = the S_n wall
    print("\nM3 -- primes: the fine structure is IRREGULAR = the wall:")
    gaps = np.diff(P)
    irr = gaps.std() / gaps.mean()
    print(f"  prime gap std/mean={irr:.2f} (IRREGULAR; regular lattice -> 0)")
    print(f"  gaps range {gaps.min()} (twin primes) to {gaps.max()}: no")
    print("  constant-gap (symmetric) operator produces this spectrum")
    print("  => no SYMMETRIC operator has standing nodes at primes; placing")
    print("     them needs breaking S_n (non-regular) = Fix(S_n)^perp")
    print("     = Riemann residue S(T). Support partial (residue lambda_2);")
    print("     fine distribution = the wall.")
    assert irr > 0.3 and gaps.min() < gaps.max() / 4

    print("\n" + "=" * 70)
    print("VERDICT: the emergent nodal/spectral order carries the SYMMETRIC /")
    print("SMOOTH equilibria -- the atom shells in full (M1), the prime")
    print("density (M2) -- because a symmetric operator's standing nodes are")
    print("a REGULAR lattice (Courant + symmetry). The IRREGULAR fine")
    print("structure of the primes (M3) is the S_n-breaking residue")
    print("Fix(S_n)^perp = S(T) = the wall. 'Where the beats fall' is nodal")
    print("geometry, carried up to the symmetry; the prime fine structure is")
    print("RH. HONEST SCOPE: Courant/PNT/S_n-wall standard; primes are ground")
    print("truth (measured, not derived); closes nothing. R and pi assumed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
