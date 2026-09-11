#!/usr/bin/env python3
"""
Example 133 — Static Coordinate Winding of Psi on a Periodic Grid
=================================================================

The diagnostic complex coordinate Psi = K_phi + i*J_phi has a phase arg(Psi)
where Psi is nonzero. For a declared face with no edge difference on the wrap
branch, the sum

    w = (1/2pi) * sum_edges wrap_angle(arg Psi_j - arg Psi_i)

is an integer degree of that static circle-valued coordinate map. This example
measures that identity and its limits.

The raw arg(Psi) winding is not invariant under the auxiliary node-dependent
rephasing Psi_i -> exp(i*alpha_i) Psi_i. The bundled connection in gauge.py is
what compensates that coordinate change; raw plaquette winding alone is not a
gauge curvature. On the periodic grid, the sum over all consistently oriented
plaquettes is zero because every internal edge occurs twice with opposite
orientation. That cancellation holds independently at each snapshot and is not
a dynamical conservation theorem or a Poincare-Hopf certificate.

Four finite observations are reported:

M1 Static face winding is integer to floating precision for the sampled
   nonzero, branch-regular Psi field.
M2 The total over all periodic plaquettes is zero by oriented-edge cancellation.
M3 A seeded local rephasing changes individual face-winding classes while the
   periodic total remains zero, demonstrating coordinate dependence.
M4 The historical Q density (legacy name: compute_topological_charge) is a
   continuous bilinear snapshot and is distinct from raw arg(Psi) winding.

No engine trajectory, particle statistics, gauge flux, or new topological
theorem is inferred.

References
----------
- src/tnfr/physics/unified.py (compute_complex_geometric_field,
  compute_historical_q_density)
- src/tnfr/physics/_helpers.py (wrap_angle)
- AGENTS.md structural-field and auxiliary-substrate scope
- examples/08_emergent_geometry/132_geometric_phase_holonomy.py (phase geometry)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from collections import Counter

import networkx as nx
import numpy as np

from tnfr.alias import set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics._helpers import wrap_angle
from tnfr.physics.unified import (
    compute_complex_geometric_field,
    compute_historical_q_density,
)

L = 10


def seed_grid(rng, periodic=True, epi_range=0.3):
    """A populated grid with random phases and varied coordinate winding."""
    G = nx.grid_2d_graph(L, L, periodic=periodic)
    inject_defaults(G)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = float(rng.uniform(0, 2 * np.pi))
        set_attr(G.nodes[nd], ALIAS_EPI, float(rng.uniform(-epi_range, epi_range)))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)
    return G


def arg_psi(G):
    """Phase arg(Psi) of the canonical complex geometric field, per node."""
    psi = compute_complex_geometric_field(G)
    return {n: float(np.angle(psi[n])) for n in G.nodes()}


def faces(periodic=True):
    """Unit-square faces of the L x L grid (CCW); wraps if periodic."""
    if periodic:
        return [
            [(x, y), ((x + 1) % L, y), ((x + 1) % L, (y + 1) % L), (x, (y + 1) % L)]
            for x in range(L)
            for y in range(L)
        ]
    return [
        [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        for x in range(L - 1)
        for y in range(L - 1)
    ]


def face_winding(arg, face):
    """Integer winding number of arg around a closed face."""
    s = sum(
        wrap_angle(arg[face[(a + 1) % len(face)]] - arg[face[a]])
        for a in range(len(face))
    )
    return s / (2.0 * np.pi)


def charges(G, periodic=True):
    arg = arg_psi(G)
    return charges_from_arg(arg, periodic=periodic)


def charges_from_arg(arg, periodic=True):
    """Integer face classes for a supplied static coordinate-phase map."""
    return [round(face_winding(arg, f)) for f in faces(periodic)]


def experiment_1_integer_winding():
    """M1: static branch-regular coordinate winding is integer."""
    print("=" * 74)
    print("M1: STATIC FACE WINDING IS INTEGER ON THIS REGULAR SAMPLE")
    print("=" * 74)
    print("The winding of arg(Psi) around each face is an integer (degree of a")
    print("map S^1 -> S^1). Labels +1/-1/0 describe coordinate winding.")
    print()
    G = seed_grid(np.random.default_rng(0))
    psi = compute_complex_geometric_field(G)
    arg = arg_psi(G)
    sampled_faces = faces()
    ws = [face_winding(arg, f) for f in sampled_faces]
    min_magnitude = min(abs(value) for value in psi.values())
    min_branch_margin = min(
        np.pi
        - abs(
            wrap_angle(
                arg[face[(edge + 1) % len(face)]] - arg[face[edge]]
            )
        )
        for face in sampled_faces
        for edge in range(len(face))
    )
    if min_magnitude <= 1e-12 or min_branch_margin <= 1e-12:
        raise RuntimeError("the seeded sample is not nonzero and branch-regular")
    max_dev = max(abs(w - round(w)) for w in ws)
    cs = [round(w) for w in ws]
    print(f"  faces = {len(ws)}, max|w - round(w)| = {max_dev:.1e}")
    print(
        f"  min|Psi| = {min_magnitude:.3e}, "
        f"min branch margin = {min_branch_margin:.3e} rad"
    )
    print(f"  class histogram: {dict(sorted(Counter(cs).items()))}")
    print(
        f"  +1 faces = {cs.count(1)}, -1 faces = {cs.count(-1)}, "
        f"|class|>1 = {sum(1 for c in cs if abs(c) > 1)}"
    )
    print()
    print("  -> the sampled nonzero Psi coordinate has integer face classes;")
    print("     this is a static degree calculation, not a gauge invariant.")


def experiment_2_periodic_cancellation():
    """M2: the consistently oriented periodic face sum cancels."""
    print()
    print("=" * 74)
    print("M2: PERIODIC ORIENTED-EDGE CANCELLATION -- TOTAL = 0")
    print("=" * 74)
    print("Each periodic edge occurs in two oppositely oriented face boundaries,")
    print("so the sum of all branch-regular face degrees is zero.")
    print()
    print(
        f"  {'seed':>6} {'total class':>13} {'#+1 faces':>11} "
        f"{'#-1 faces':>14}"
    )
    n_zero = 0
    for s in range(4):
        G = seed_grid(np.random.default_rng(s))
        cs = charges(G)
        total = sum(cs)
        nv = sum(1 for c in cs if c > 0)
        na = sum(1 for c in cs if c < 0)
        n_zero += int(total == 0)
        print(f"  {s:>6} {total:>+13d} {nv:>11d} {na:>14d}")
    print()
    print(f"  -> total = 0 for {n_zero}/4 seeds by periodic edge cancellation.")
    print("     This is an identity of each static coordinate map.")


def experiment_3_local_rephasing_dependence():
    """M3: local rephasing changes raw arg(Psi) face classes."""
    print()
    print("=" * 74)
    print("M3: RAW arg(Psi) WINDING IS NOT LOCAL-U(1) INVARIANT")
    print("=" * 74)
    print("Apply a seeded node-dependent coordinate rotation alpha_i. Individual")
    print("face classes can change although the periodic total still cancels.")
    print()
    print(f"  {'seed':>6} {'net before/after':>18} {'changed faces':>14} "
          f"{'nonzero before/after':>23}")
    n_changed = 0
    for s in range(3):
        G = seed_grid(np.random.default_rng(40 + s))
        before_arg = arg_psi(G)
        rng = np.random.default_rng(900 + s)
        after_arg = {
            node: wrap_angle(before_arg[node] + float(rng.uniform(-np.pi, np.pi)))
            for node in G.nodes()
        }
        before = charges_from_arg(before_arg)
        after = charges_from_arg(after_arg)
        changed = sum(left != right for left, right in zip(before, after))
        n_changed += int(changed > 0)
        before_count = sum(value != 0 for value in before)
        after_count = sum(value != 0 for value in after)
        print(f"  {s:>6} {(str(sum(before)) + '/' + str(sum(after))):>18} "
              f"{changed:>14d} "
              f"{(str(before_count) + '/' + str(after_count)):>23}")
    print()
    print(f"  -> local rephasing changed face classes for {n_changed}/3 seeds.")
    print("     Both totals stay zero because the periodic edge sum cancels.")


def experiment_4_q_is_not_winding():
    """M4: the tensor-suite Q is a continuous density, not the integer winding."""
    print()
    print("=" * 74)
    print("M4: THE TENSOR-SUITE Q IS NOT THE INTEGER WINDING")
    print("=" * 74)
    print("Q = |grad phi|*J_phi - K_phi*J_dNFR is a continuous historical")
    print("bilinear snapshot. It does not localize the integer-winding")
    print("classes in this sample: compare mean |Q| on nonzero and zero faces.")
    print()
    print(
        f"  {'topology':>12} {'|Q| nonzero':>11} {'|Q| zero':>12} "
        f"{'ratio':>7}"
    )
    for label, periodic in [("torus", True), ("open grid", False)]:
        G = seed_grid(np.random.default_rng(0), periodic=periodic)
        arg = arg_psi(G)
        Q = compute_historical_q_density(G)
        d, r = [], []
        for f in faces(periodic):
            w = face_winding(arg, f)
            m = np.mean([abs(Q[n]) for n in f])
            (d if abs(w) > 0.5 else r).append(m)
        ratio = np.mean(d) / np.mean(r)
        print(
            f"  {label:>12} {np.mean(d):>11.4f} {np.mean(r):>12.4f} " f"{ratio:>6.2f}x"
        )
    print()
    print("  -> the ratio is ~1.0 in these two finite samples: Q does not")
    print("     distinguish their nonzero and zero raw-winding faces.")
    print("     Q is a bilinear snapshot; winding is a separate static class.")


def main():
    print()
    print("  ===============================================================")
    print("  Static Coordinate Winding of Psi = K_phi + i*J_phi")
    print("  Integer Face Classes and Their Local-U(1) Dependence")
    print("  ===============================================================")
    print()
    experiment_1_integer_winding()
    experiment_2_periodic_cancellation()
    experiment_3_local_rephasing_dependence()
    experiment_4_q_is_not_winding()
    print()
    print("=" * 74)
    print("WHAT THIS ESTABLISHES")
    print("=" * 74)
    print("For the sampled nonzero, branch-regular Psi coordinate, each face")
    print("winding is integer (M1) and periodic totals cancel (M2). A local")
    print("rephasing changes individual face classes (M3), so raw arg(Psi)")
    print("winding is not a local-U(1) invariant or gauge curvature. The")
    print("historical Q density is a separate continuous bilinear snapshot;")
    print("M4 finds no separation in its samples. No trajectory conservation")
    print("or particle physics follows.")


if __name__ == "__main__":
    main()
