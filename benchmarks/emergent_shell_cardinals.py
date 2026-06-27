"""
Emergent Shell Cardinals: magic numbers as dynamical-symmetry irrep cardinals
(the DEEP layer), not spatial-ball modes (the superficial one).
=============================================================================

DIAGNOSIS (benchmarks/emergent_shell_ordering.py was too superficial): we
mapped "atom" -> a ball of points in R^3 + k-NN -> a spatial Laplacian, and got
the spherical well 2, 8, 18, 20. That IMPORTS R^3 and the rotation group SO(3)
(we built the symmetry in geometrically) and -- worse -- the spatial radial
split BREAKS the larger dynamical degeneracies.

THE TNFR-NATIVE LAYER (emergent_integers_symmetry): an integer emerges as the
DIMENSION of an irreducible representation of the symmetry group of the
structural dynamics. So a shell and its magic number are NOT spatial counts;
they are CARDINALS of a symmetry. The periodic table's numbers are, in standard
physics, the cumulative cardinals of a chain of DYNAMICAL symmetry groups:

    SO(3)   geometric rotations     -> irrep dims 2l+1
    SO(4)   Coulomb (Runge-Lenz)    -> level degeneracy n^2 (hydrogen)
    U(3)    3D isotropic oscillator -> level deg (N+1)(N+2)/2
    SO(4,2) the "Madelung group"    -> the (n+l) period structure

THE CONCRETE RESULT (measured below): the atomic "10" (Ne closure) that the
spatial ball and the Phi_s self-consistency could NEVER produce is exactly the
SO(4) cardinal -- the n=2 Coulomb shell (2s + 2p DEGENERATE = 8, so the
cumulative is 2 + 8 = 10). The spatial ball broke SO(4) -> SO(3) (it split
2s from 2p), giving 2, 8 instead of 2, 10. So the "10" is a SYMMETRY cardinal,
not a two-body force.

HONEST SCOPE: the representation theory of SO(3)/SO(4)/U(3)/SO(4,2) is
STANDARD external mathematics -- the comparison framework, exactly as
emergent_integers_symmetry cites L-commutes-with-Aut(G). The TNFR
contribution is (i) the reading "magic number = emergent cardinal (irrep
dim)", at the same ontological level as the number-theory program, and
(ii) the diagnosis that the spatial-ball mapping sat at the BOTTOM of the
symmetry chain and broke the dynamical degeneracies. The genuinely emergent
home of these cardinals is the integrable-OSCILLATOR substrate
(symplectic_substrate.py: H_sub is a sum of decoupled oscillators with a
U(2) dynamical symmetry, action-angle / Bohr-Sommerfeld). Deriving WHICH
dynamical group the full nonlinear operator dynamics realises (and whether
it reaches SO(4,2) / Madelung) is the open deep program; this benchmark
does NOT claim it.

Run:
    python benchmarks/emergent_shell_cardinals.py

Theoretical anchor: AGENTS.md (emergent geometry; discrete-mode regime);
benchmarks/emergent_integers_symmetry.py (cardinals = irrep dims);
benchmarks/emergent_shell_ordering.py (the spatial-ball mapping it deepens).
Status: RESEARCH (falsifier / reframing).
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from emergent_shell_ordering import solid_ball_graph  # noqa: E402
from tnfr.physics.emergent_chemistry import (  # noqa: E402
    structural_eigenmodes,
)

# Observed references (post-hoc identification only)
ATOMIC_NOBLE = [2, 10, 18, 36, 54, 86]  # SO(4,2) / Madelung
NUCLEAR_MAGIC = [2, 8, 20, 28, 50, 82]  # oscillator + spin-orbit


def closures(degeneracies: list[int], spin: int = 2) -> list[int]:
    """Cumulative shell closures = running sum of (spin x level degeneracy).

    This is the magic-number rule of a dynamical symmetry: each level is an
    irrep whose dimension is the structural degeneracy; the doublet factor
    ``spin=2`` is the per-mode capacity (the +/- phase-winding pair).
    """
    out: list[int] = []
    total = 0
    for d in degeneracies:
        total += spin * d
        out.append(total)
    return out


def so3_ladder(n: int = 5) -> list[int]:
    """SO(3) angular ladder: irrep dims 2l+1 (one radial node per l)."""
    return closures([2 * ell + 1 for ell in range(n)])


def so4_coulomb(n: int = 5) -> list[int]:
    """SO(4) Coulomb: level n has degeneracy n^2 (all l of n DEGENERATE)."""
    return closures([k * k for k in range(1, n + 1)])


def u3_oscillator(n: int = 5) -> list[int]:
    """U(3) 3D isotropic oscillator: level N degeneracy (N+1)(N+2)/2."""
    return closures([(N + 1) * (N + 2) // 2 for N in range(n)])


def ball_closures(G: nx.Graph) -> list[int]:
    """Measured closures of the spatial-ball mapping (the superficial one)."""
    shells = structural_eigenmodes(G, max_modes=40, gap_factor=4.0)
    out: list[int] = []
    total = 0
    for sh in shells:
        total += 2 * sh.multiplicity
        out.append(total)
    return out


def main() -> None:
    print("=" * 70)
    print("EMERGENT SHELL CARDINALS (dynamical-symmetry irreps, not a ball)")
    print("=" * 70)

    so3 = so3_ladder()
    so4 = so4_coulomb()
    u3 = u3_oscillator()

    # -- M1: magic numbers ARE cumulative dynamical-symmetry cardinals -------
    print("\n[M1] Magic numbers as cumulative irrep cardinals (2 x dim):")
    print(f"     SO(3) angular ladder (2l+1) : {so3}")
    print(f"     SO(4) Coulomb (n^2)         : {so4}")
    print(f"     U(3) 3D oscillator          : {u3}")
    assert so3 == [2, 8, 18, 32, 50], so3
    assert so4 == [2, 10, 28, 60, 110], so4
    assert u3 == [2, 8, 20, 40, 70], u3
    print("     -> PASS: each dynamical symmetry gives an EXACT magic-number")
    print("        family as cumulative irrep dimensions.")

    # -- M2: the atomic '10' is the SO(4) cardinal the spatial ball broke ----
    ball = ball_closures(solid_ball_graph(4, 16, 8))
    print("\n[M2] The atomic '10' (Ne) is the SO(4) n=2 cardinal:")
    print(f"     SO(4) Coulomb              : {so4[:4]}  (10 = 2 + [2s+2p=8])")
    print(f"     spatial ball (spherical)   : {ball[:5]}")
    print(f"     atomic noble gases         : {ATOMIC_NOBLE}")
    assert so4[1] == 10, "SO(4) n=2 closure is not 10"
    assert ball[:2] == [2, 8] and 10 not in ball[:5], "ball kept SO(4)"
    print("     -> PASS: SO(4) (2s+2p degenerate) gives 2, 10; the ball")
    print("        broke SO(4) -> SO(3) (split 2s/2p): 2, 8 -- no '10'.")
    print("        The '10' is a SYMMETRY CARDINAL, not a two-body force.")

    # -- M3: the symmetry chain (the spatial ball sat at the bottom) ---------
    print("\n[M3] The dynamical-symmetry chain (the deep layer climbs it):")
    print(f"     SO(3) box (spatial ball)   : {ball[:4]}   least symmetry")
    print(f"     SO(4) Coulomb              : {so4[:4]}   recovers the 10")
    print(f"     SO(4,2) Madelung (atomic)  : {ATOMIC_NOBLE[:4]}   periods")
    print(f"     U(3) oscillator (nuclear)  : {u3[:3]} -> nuclear "
          f"{NUCLEAR_MAGIC[:3]}")
    assert u3[:3] == NUCLEAR_MAGIC[:3], "U(3) != lower nuclear magic"
    print("     -> PASS: U(3) reproduces the lower nuclear magic 2, 8, 20")
    print("        (spin-orbit adds 28, 50, 82). The substrate is the")
    print("        integrable-oscillator home of these cardinals.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "Magic numbers are CARDINALS of a dynamical symmetry (cumulative\n"
        "  irrep dimensions) -- the TNFR-native 'numbers are emergent'\n"
        "  layer (emergent_integers_symmetry), the SAME ontological level\n"
        "  as the number-theory program, NOT spatial counts of a ball.\n"
        "THE SUPERFICIALITY, DIAGNOSED: the spatial-ball mapping sat at\n"
        "  the BOTTOM of the symmetry chain (geometric SO(3) + a box) and\n"
        "  BROKE the dynamical degeneracies -- it split 2s from 2p, so it\n"
        "  gave 2, 8, 18, 20 and could never produce the atomic '10'.\n"
        "  That '10' is the SO(4) Coulomb cardinal (the n=2 shell, 2s+2p\n"
        "  degenerate); we hunted it as a two-body screening force when\n"
        "  it was a SYMMETRY cardinal all along.\n"
        "THE CHAIN: SO(3) box (ball) < SO(4) Coulomb (2,10,28) < SO(4,2)\n"
        "  Madelung (2,10,18,36,54,86, atomic); U(3) oscillator (2,8,20)\n"
        "  is the nuclear branch. Each level is an EXACT cardinal family.\n"
        "THE DEEP HOME: the integrable-oscillator substrate (H_sub = sum\n"
        "  of decoupled oscillators, U(2) dynamical symmetry, action-\n"
        "  angle / Bohr-Sommerfeld) is where these cardinals live -- not\n"
        "  a spatial Laplacian. OPEN: derive WHICH dynamical group the\n"
        "  full nonlinear operator dynamics realises, and whether it\n"
        "  reaches SO(4,2) / Madelung."
    )


if __name__ == "__main__":
    main()
