"""Finite comparison of externally specified shell-counting models.

The benchmark evaluates four independent inputs:

* the standard dimension sequences ``2l+1``, ``n^2`` and
  ``(N+1)(N+2)/2``, associated externally with SO(3), SO(4) and U(3) models;
* a configurable factor ``spin=2`` used here as an assumed per-mode capacity;
* numerical eigenvalue clusters of a constructed spatial-ball graph;
* published atomic and nuclear closure counts entered as reference lists.

It then compares their cumulative sums. Numerical coincidences such as the
second SO(4)-model sum ``2 + 2*4 = 10`` or the first three U(3)-model sums
``2, 8, 20`` follow from the chosen formulas and capacity. They do not show
that TNFR dynamics selected those groups, produced spin multiplicity, derived
atomic or nuclear structure, or established a hierarchy among the models.
Likewise, the spatial-ball calculation implements a different construction;
its failure to contain 10 in the sampled clusters does not demonstrate symmetry
breaking or identify the physical mechanism behind any closure.

TNFR supplies only the graph spectral-clustering read-out used for the ball
comparison. Every group, degeneracy formula, capacity and reference sequence is
external input to this script. The result is therefore a finite comparison and
a negative identification result, not an emergence derivation.

Run:
    python benchmarks/emergent_shell_cardinals.py

Status: RESEARCH benchmark; externally specified models and scoped comparisons.
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

# External reference lists used only for finite comparison.
ATOMIC_NOBLE = [2, 10, 18, 36, 54, 86]
NUCLEAR_MAGIC = [2, 8, 20, 28, 50, 82]


def closures(degeneracies: list[int], spin: int = 2) -> list[int]:
    """Cumulative sums using the explicitly supplied capacity multiplier.

    ``spin`` is retained as the historical parameter name. Its default value
    is an external model assumption; no phase-winding or TNFR derivation of the
    factor is made here.
    """
    out: list[int] = []
    total = 0
    for d in degeneracies:
        total += spin * d
        out.append(total)
    return out


def so3_ladder(n: int = 5) -> list[int]:
    """Cumulative counts from the externally chosen SO(3) formula ``2l+1``."""
    return closures([2 * ell + 1 for ell in range(n)])


def so4_coulomb(n: int = 5) -> list[int]:
    """Cumulative counts from the external SO(4) Coulomb formula ``n^2``."""
    return closures([k * k for k in range(1, n + 1)])


def u3_oscillator(n: int = 5) -> list[int]:
    """Cumulative counts from the external U(3) oscillator formula."""
    return closures([(N + 1) * (N + 2) // 2 for N in range(n)])


def ball_closures(G: nx.Graph) -> list[int]:
    """Cumulative measured cluster sizes with the same assumed factor two."""
    shells = structural_eigenmodes(G, max_modes=40, gap_factor=4.0)
    out: list[int] = []
    total = 0
    for sh in shells:
        total += 2 * sh.multiplicity
        out.append(total)
    return out


def main() -> None:
    print("=" * 70)
    print("FINITE SHELL-CARDINAL COMPARISONS")
    print("=" * 70)

    so3 = so3_ladder()
    so4 = so4_coulomb()
    u3 = u3_oscillator()

    # -- M1: arithmetic consequences of three externally chosen formulas -----
    print("\n[M1] Cumulative counts from external formulas (capacity factor 2):")
    print(f"     SO(3)-model dimensions (2l+1) : {so3}")
    print(f"     SO(4)-model dimensions (n^2)   : {so4}")
    print(f"     U(3)-model dimensions           : {u3}")
    assert so3 == [2, 8, 18, 32, 50], so3
    assert so4 == [2, 10, 28, 60, 110], so4
    assert u3 == [2, 8, 20, 40, 70], u3
    print("     -> PASS: the implementation returns the exact cumulative sums")
    print("        implied by the three supplied formulas and multiplier.")

    # -- M2: finite comparison with the independently constructed ball graph --
    ball = ball_closures(solid_ball_graph(4, 16, 8))
    print("\n[M2] SO(4)-formula sums versus one spatial-ball protocol:")
    print(f"     SO(4)-model cumulative sums : {so4[:4]}")
    print(f"     sampled spatial-ball sums   : {ball[:5]}")
    print(f"     external atomic references  : {ATOMIC_NOBLE}")
    assert so4[1] == 10, "SO(4) n=2 closure is not 10"
    assert ball[:2] == [2, 8] and 10 not in ball[:5], "unexpected ball sequence"
    print("     -> PASS: 10 is the second cumulative SO(4)-formula value and")
    print("        is absent from the first five sampled ball-cluster sums.")
    print("        This distinguishes the constructions; it does not explain the")
    print("        physical atomic reference or show that the ball broke SO(4).")

    # -- M3: side-by-side finite reference comparison ------------------------
    print("\n[M3] Side-by-side model and reference sequences:")
    print(f"     sampled spatial ball       : {ball[:4]}")
    print(f"     SO(4)-model sums            : {so4[:4]}")
    print(f"     external atomic references : {ATOMIC_NOBLE[:4]}")
    print(f"     U(3)-model sums             : {u3[:3]}")
    print(f"     external nuclear references: {NUCLEAR_MAGIC[:3]}")
    assert u3[:3] == NUCLEAR_MAGIC[:3], "U(3) != lower nuclear magic"
    print("     -> PASS: the first three U(3)-model sums equal the first three")
    print("        supplied nuclear reference values. This finite equality does")
    print("        not derive the reference sequence or its physical mechanism.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "The benchmark computes cumulative counts for three externally\n"
        "  selected representation formulas and one constructed graph.\n"
        "  The capacity factor two and both physical reference lists are\n"
        "  inputs. The observed agreements and disagreements separate the\n"
        "  finite models, but they do not identify shell counts with TNFR\n"
        "  dynamics or show that TNFR selects SO(3), SO(4), U(3), spin,\n"
        "  atomic structure, or nuclear structure."
    )


if __name__ == "__main__":
    main()
