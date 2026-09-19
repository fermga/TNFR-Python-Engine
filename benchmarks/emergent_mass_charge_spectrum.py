"""Static winding-sector energy comparisons on prescribed rings and grids.

The script constructs one-core and separated-core phase fields independently;
it does not evolve a core into two cores. A lower diagnostic energy for a
prepared separated field is not a fission trajectory or a derived force law.
The stored pressure and support are supplied. compute_energy_density is a
quadratic field diagnostic, not an established physical rest-energy functional.
The ring checks test approximate W-squared ratios at finite size and small
phase increments; the trigonometric diagnostic need not obey an exact square
law at arbitrary winding. The printed mass label is only a proxy.
Embedded lepton values are external comparisons. One phase family cannot prove
that all TNFR configurations lack an independent structural mode or label.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import winding_ring  # noqa: E402
from tnfr.physics.unified import compute_energy_density  # noqa: E402

_TWO_PI = 2.0 * math.pi
GRID = 25  # L x L open grid; odd so a single core sits on a node


def _wrap_pi(x: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    y = (x + math.pi) % _TWO_PI - math.pi
    return y - _TWO_PI if y > math.pi else y


def build_vortices(
    cores: list[tuple[float, float, int]],
    *,
    base_dnfr: float = 0.05,
    grid: int = GRID,
) -> nx.Graph:
    """Open 2D grid whose phase is the superposition of vortex cores.

    ``cores`` is a list of (cx, cy, q): a charge-q vortex has local phase
    q * atan2(y - cy, x - cx). The winding is MEASURED as an output (per face);
    the construction only sets the initial phase field.
    """
    G = nx.grid_2d_graph(grid, grid)
    for i, j in G.nodes():
        ang = 0.0
        for cx, cy, q in cores:
            ang += q * math.atan2(j - cy, i - cx)
        phi = ang % _TWO_PI
        G.nodes[(i, j)]["theta"] = float(phi)
        G.nodes[(i, j)]["phase"] = float(phi)
        G.nodes[(i, j)]["delta_nfr"] = float(base_dnfr)
        G.nodes[(i, j)]["dnfr"] = float(base_dnfr)
        G.nodes[(i, j)]["coherence"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[(i, j)]["EPI"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[(i, j)]["nu_f"] = 1.0
    return G


def loop_winding(G: nx.Graph, *, grid: int = GRID, margin: int = 2) -> int:
    """Integer winding around a large square loop = total enclosed charge.

    Walking a big boundary loop keeps every step's phase difference well below
    pi, so the wrap is unambiguous even for multiply-charged cores (a single
    |W|=2 face would sit exactly at the pi wrap boundary; a large loop does not).
    """
    theta = {n: G.nodes[n]["theta"] for n in G.nodes()}
    lo, hi = margin, grid - 1 - margin
    path: list[tuple[int, int]] = []
    path += [(x, lo) for x in range(lo, hi)]
    path += [(hi, y) for y in range(lo, hi)]
    path += [(x, hi) for x in range(hi, lo, -1)]
    path += [(lo, y) for y in range(hi, lo, -1)]
    s = sum(
        _wrap_pi(theta[path[(k + 1) % len(path)]] - theta[path[k]])
        for k in range(len(path))
    )
    return int(round(s / _TWO_PI))


def self_energy(G: nx.Graph, vacuum: float) -> float:
    """Total quadratic field diagnostic minus the supplied baseline; not physical mass."""
    ed = compute_energy_density(G)
    return float(sum(ed[n] for n in G.nodes()) - vacuum)


def vacuum_energy(*, base_dnfr: float = 0.05, grid: int = GRID) -> float:
    """Energy of the uniform-phase grid (the vacuum baseline)."""
    G = build_vortices([], base_dnfr=base_dnfr, grid=grid)
    ed = compute_energy_density(G)
    return float(sum(ed[n] for n in G.nodes()))


def ring_mass(w: int, *, n: int = 60) -> float:
    """Total diagnostic on a prepared winding ring.

    The caller subtracts the W=0 baseline; finite trigonometric terms need not
    produce exact W-squared ratios."""
    ed = compute_energy_density(winding_ring(n, w))
    return float(sum(ed.values()))


def main() -> None:
    print("=" * 74)
    print("PREPARED WINDING FIELDS AND QUADRATIC DIAGNOSTIC COMPARISONS")
    print("=" * 74)

    c = GRID // 2
    cf = c + 0.5  # half-integer face centre: no node coincides with a core
    vac = vacuum_energy()

    # -- M1: Static one-core versus two-core comparison -----------
    print("\n[M1] STATIC CORE COMPARISON: a |W|=2 core vs two separated |W|=1.")
    g_double = build_vortices([(cf, cf, 2)])
    e_double = self_energy(g_double, vac)
    w_double = loop_winding(g_double)
    sep = 8
    g_split = build_vortices([(cf - sep / 2, cf, 1), (cf + sep / 2, cf, 1)])
    e_split = self_energy(g_split, vac)
    w_split = loop_winding(g_split)
    print(f"     single |W|=2 core : winding={w_double:+d}  self-energy={e_double:.4f}")
    print(
        f"     two |W|=1 (sep={sep}) : winding={w_split:+d}  "
        f"self-energy={e_split:.4f}"
    )
    print(f"     E(W=2 core) - E(two W=1) = {e_double - e_split:+.4f}")
    assert w_double == 2 and w_split == 2, "total charge must be +2 for both"
    assert e_double > e_split, "double core not above the split -- no fission"
    print("     -> PASS: the double-winding core costs MORE than the split pair;")
    print(
        "        the supplied two-core field has lower diagnostic energy; no fission run."
    )

    # -- M2: Static energy variation across separately supplied separations ----------------
    print("\n[M2] STATIC SEPARATION COMPARISON: E(two |W|=1) vs separation.")
    print(f"     {'separation':>11} {'winding':>8} {'self-energy':>13}")
    prev = None
    decreasing = True
    for sep in (2, 4, 6, 8, 10, 12):
        g = build_vortices([(cf - sep / 2, cf, 1), (cf + sep / 2, cf, 1)])
        e = self_energy(g, vac)
        w = loop_winding(g)
        print(f"     {sep:>11} {w:>+8d} {e:>13.4f}")
        if prev is not None and e > prev + 1e-6:
            decreasing = False
        prev = e
    assert decreasing, "energy did not fall with separation -- no repulsion"
    print(
        "     -> PASS: energy falls across these separately prepared core separations"
    )
    print("        No motion, force derivative or causal fission is measured.")

    # -- M3: the finite ring-energy proxy m(W) ~ W^2 -------------------
    print("\n[M3] RING-ENERGY PROXY: self-energy of the charge-W sector.")
    print(f"     {'W':>3} {'mass (self-E)':>15} {'m(W)/m(1)':>11} {'W^2':>6}")
    vac_ring = ring_mass(0)
    m1 = ring_mass(1) - vac_ring
    masses = {}
    for w in (1, 2, 3, 4, 5):
        m = ring_mass(w) - vac_ring
        masses[w] = m
        ratio = m / m1 if m1 else float("nan")
        print(f"     {w:>3} {m:>15.4f} {ratio:>11.3f} {w * w:>6}")
    print(
        "     m(W)/m(1) = approximately the integer squares 1, 4, 9, 16, 25 (m ~ W^2)."
    )
    assert abs(masses[2] / m1 - 4.0) < 0.05, "ring mass not ~4 at W=2"
    assert abs(masses[3] / m1 - 9.0) < 0.1, "ring mass not ~9 at W=3"
    print(
        "     -> TNFR's (mass, charge) locus is m ~ q^2: this selected diagnostic depends on"
    )
    print("        prepared winding; no global mass law follows.")

    # -- M4: the honest confrontation with the real spectrum (Layer 3) ---------
    print("\n[M4] CONFRONTATION (Layer 3): the real charged leptons.")
    leptons = {"e": 0.511, "mu": 105.66, "tau": 1776.86}  # MeV; all q = -1
    m_e = leptons["e"]
    print(f"     {'lepton':>7} {'charge':>7} {'mass (MeV)':>11} {'m/m_e':>9}")
    for name, mass in leptons.items():
        print(f"     {name:>7} {-1:>+7d} {mass:>11.3f} {mass / m_e:>9.1f}")
    print(
        "     Selected ring proxy: one value per prepared winding; not all TNFR states."
    )
    print("     Nature: same charge (-1), masses in ratio 1 : 207 : 3477.")
    print("     -> The one-parameter proxy in M3 does not supply the displayed")
    print(
        "        same-charge mass ratios; no full TNFR state-space obstruction follows."
    )

    print("\n" + "=" * 74)
    print("FINITE STATIC COMPARISON:")
    print(
        "  Independently prepared one-core and two-core fields have different energies."
    )
    print("  No split trajectory, separation dynamics or physical force is executed.")
    print("  The ring diagnostic passes the stated approximate square-ratio checks.")
    print("  A diagnostic proxy is not a physical mass spectrum.")
    print("  The supplied lepton comparison is not a prediction from nodal dynamics.")
    print("  Other state channels and the physical identification remain unresolved.")
    print("=" * 74)


if __name__ == "__main__":
    main()
