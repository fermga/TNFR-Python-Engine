"""Compare two sign-reversing involutions without identifying their domains.

The benchmark verifies two standard finite statements:

* on a bipartite graph, ``Gamma = diag(+1 on X, -1 on Y)`` anticommutes
  with the adjacency matrix, hence its spectrum is symmetric under
  ``lambda -> -lambda``;
* on a declared phase loop, pointwise phase negation sends winding ``W`` to
  ``-W`` and applying it twice recovers the original phase field.

Both transformations generate groups isomorphic to Z_2 and both reverse a
signed observable. They act on different spaces, so these facts do not make
them the same transformation or identify arithmetic inverses with physical
charge conjugation. Likewise, ``W + (-W) = 0`` is an algebraic identity; this
script does not combine two defects or simulate annihilation.

CONTRAST WITH THE EQUIVARIANCE BENCHMARK:
  graph-automorphism permutation matrices commute with adjacency, whereas the
  diagonal sublattice matrix here anticommutes with it. These are distinct
  representations and should not be merged merely because both square to the
  identity.

GROUND TRUTH (standard graph and loop identities):
  - A graph is bipartite iff it is 2-colourable iff there is a diagonal sign
    matrix Gamma with Gamma A Gamma = -A (chiral / sublattice symmetry of
    bipartite tight-binding / SSH Hamiltonians). Then spec(A) = -spec(A).
  - The topological winding number W in Z is ODD under phase conjugation:
    C : phi -> -phi  =>  W -> -W (definitional: circulation reverses sign).
  - Gamma^2 = I and C^2 = id: each generates a Z_2.

TNFR scope: adjacency is a graph-coupling read-out and winding is a closed-loop
phase read-out. Their sign symmetries provide a comparison, not a derivation of
integers, particles, CPT or the Standard Model.

Run:
    python benchmarks/chiral_involution.py

Status: RESEARCH (negative identification result and finite symmetry checks).
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Robust fallback so the harness also runs without PYTHONPATH=src preset.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from composition_arithmetic import adj_spectrum, automorphism_matrices  # noqa: E402
from emergent_rationals import integer_spectrum, is_pm_symmetric  # noqa: E402

from tnfr.physics.emergent_particles import winding_number, winding_ring  # noqa: E402

TOL = 1e-9
_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
# The chiral (sublattice) involution Gamma of a bipartite graph
# --------------------------------------------------------------------------- #
def adjacency(G, nodes):
    """Coupling/adjacency matrix A in the fixed node order `nodes`."""
    return nx.to_numpy_array(G, nodelist=nodes)


def chirality_operator(G, nodes):
    """Gamma = diag(+1 on sublattice X, -1 on sublattice Y) from the bipartite
    2-colouring. Raises nx.NetworkXError if G is not bipartite (then no chiral
    involution exists -- spectral sign pairing then has no such certificate)."""
    color = nx.algorithms.bipartite.color(G)
    signs = np.array([1.0 if color[v] == 0 else -1.0 for v in nodes])
    return np.diag(signs)


def commutator_norm(M, P):
    """Frobenius norm ||M P - P M||."""
    return float(np.linalg.norm(M @ P - P @ M))


def anticommutator_norm(M, P):
    """Frobenius norm ||M P + P M||."""
    return float(np.linalg.norm(M @ P + P @ M))


def conjugate_phase_node(phi):
    """Phase negation C: phi -> -phi, re-wrapped to [0, 2pi)."""
    y = (-phi) % _TWO_PI
    return float(y)


# --------------------------------------------------------------------------- #
# (1) chiral involution -> spectral sign pairing
# --------------------------------------------------------------------------- #
def test_chiral_gives_spectral_sign_pairing():
    print("=" * 78)
    print("(1) chiral involution Gamma: Gamma A Gamma = -A  =>  spec(A) = -spec(A)")
    print("    bipartite coupling pairs each measured eigenvalue with its negative")
    print("=" * 78)
    bipartite = [
        ("C6", nx.cycle_graph(6)),
        ("K_{3,3}", nx.complete_bipartite_graph(3, 3)),
        ("Q3 (hypercube)", nx.hypercube_graph(3)),
        ("P4", nx.path_graph(4)),
    ]
    non_bipartite = [("C5", nx.cycle_graph(5)), ("K4", nx.complete_graph(4))]

    all_ok = True
    for name, G in bipartite:
        nodes = list(G.nodes())
        A = adjacency(G, nodes)
        Gamma = chirality_operator(G, nodes)
        invol = float(np.linalg.norm(Gamma @ Gamma - np.eye(len(nodes))))
        anti = anticommutator_norm(Gamma, A)  # {Gamma, A} = 0
        chiral = float(np.linalg.norm(Gamma @ A @ Gamma + A))  # Gamma A Gamma = -A
        spec = adj_spectrum(G)
        sym = is_pm_symmetric(spec)
        ok = invol < TOL and anti < TOL and chiral < TOL and sym
        all_ok &= ok
        print(
            f"  {name:<16} Gamma^2=I:{invol:.1e}  {{Gamma,A}}=0:{anti:.1e}  "
            f"GammaAGamma+A:{chiral:.1e}  spec+/-sym:{sym}  -> "
            f"{'OK' if ok else 'FAIL'}"
        )

    # non-bipartite: no 2-colouring => no chiral Gamma => spectrum not +/- symmetric
    none_chiral = True
    for name, G in non_bipartite:
        nodes = list(G.nodes())
        try:
            chirality_operator(G, nodes)
            has_gamma = True
        except nx.NetworkXError:
            has_gamma = False
        sym = is_pm_symmetric(adj_spectrum(G))
        none_chiral &= (not has_gamma) and (not sym)
        print(
            f"  {name:<16} bipartite/chiral Gamma exists? {has_gamma}  "
            f"spec +/- symmetric? {sym}   (non-bipartite contrast)"
        )

    # This particular finite graph happens to have integral paired eigenvalues.
    q3 = integer_spectrum(adj_spectrum(nx.hypercube_graph(3)))
    signed = sorted(set(int(v) for v in q3))
    z_ok = -min(signed) == max(signed)
    print(
        f"  Q3 has paired integral eigenvalues: {signed}   "
        f"({'OK' if z_ok else 'FAIL'})"
    )

    ok = all_ok and none_chiral and z_ok
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- the chiral Z_2 certifies "
        "spectral sign pairing on these bipartite graphs"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# (2) a distinct phase involution flips winding orientation
# --------------------------------------------------------------------------- #
def test_phase_negation_flips_winding():
    print("=" * 78)
    print("(2) phase negation C: phi -> -phi flips winding W -> -W")
    print("    This is a loop-orientation result on a different state space.")
    print("=" * 78)
    n = 12
    all_ok = True
    for k in (1, 2, 3):
        positive = winding_ring(n, k)
        w_positive, _ = winding_number(positive)

        # Negate every phase: the loop-orientation Z_2 on the phase field.
        negative = positive.copy()
        for v in negative.nodes():
            phi = negative.nodes[v]["phase"]
            cphi = conjugate_phase_node(phi)
            negative.nodes[v]["phase"] = cphi
            negative.nodes[v]["theta"] = cphi
        w_negative, _ = winding_number(negative)

        # winding_ring(n, -k) is the SAME field (phi -> -phi); cross-check it
        direct = winding_ring(n, -k)
        w_d, _ = winding_number(direct)
        same_field = all(
            abs(
                conjugate_phase_node(positive.nodes[v]["phase"])
                - direct.nodes[v]["phase"]
            )
            < 1e-9
            for v in positive.nodes()
        )

        # C^2 = id: negating twice recovers the initial orientation.
        back = negative.copy()
        for v in back.nodes():
            cphi = conjugate_phase_node(back.nodes[v]["phase"])
            back.nodes[v]["phase"] = cphi
            back.nodes[v]["theta"] = cphi
        w_back, _ = winding_number(back)

        ok = (
            w_positive == k
            and w_negative == -k
            and w_d == -k
            and same_field
            and w_back == k
            and w_positive > 0
            and w_negative < 0
            and abs(w_positive) == abs(w_negative)
        )
        all_ok &= ok
        print(
            f"  k={k}: W(positive)={w_positive:+d}, "
            f"C: W(negative)={w_negative:+d}, "
            f"same |W|={abs(w_positive) == abs(w_negative)}"
        )
        print(
            f"        winding_ring(n,-k)=W={w_d:+d} is the conjugate field? "
            f"{same_field};  C^2: W back to {w_back:+d}  -> "
            f"{'OK' if ok else 'FAIL'}"
        )
    print(
        f"  VERDICT: {'PASS' if all_ok else 'FAIL'} -- phase negation is a "
        "Z_2 action that reverses loop winding"
    )
    print()
    return all_ok


# --------------------------------------------------------------------------- #
# (3) compare two algebraic zero identities without identifying their domains
# --------------------------------------------------------------------------- #
def test_signed_zero_identities():
    print("=" * 78)
    print("(3) signed zero identities: n + (-n) = 0 and W + (-W) = 0")
    print("    The values coincide; their state spaces remain distinct.")
    print("=" * 78)
    # number side: emergent eigenvalue n and its chiral mirror -n sum to 0
    q3 = integer_spectrum(adj_spectrum(nx.hypercube_graph(3)))
    pos = sorted(set(int(v) for v in q3 if v > 0))
    number_ok = True
    for n in pos:
        mirror_present = (-n) in set(int(v) for v in q3)
        sums_to_zero = (n + (-n)) == 0
        number_ok &= mirror_present and sums_to_zero
        print(
            f"  number: mode n={n:+d} has chiral partner -n={-n:+d} present? "
            f"{mirror_present};  n+(-n)={n + (-n)}  (additive identity)"
        )

    # Loop side: opposite winding integers sum to zero algebraically.
    nring = 12
    winding_ok = True
    for k in (1, 2, 3):
        w_m, _ = winding_number(winding_ring(nring, k))
        w_a, _ = winding_number(winding_ring(nring, -k))
        net = w_m + w_a
        zero_winding, _ = winding_number(winding_ring(nring, 0))
        sums_to_zero = (net == 0) and (zero_winding == 0)
        winding_ok &= sums_to_zero
        print(
            f"  winding: W+={w_m:+d} + W-={w_a:+d} = {net}; "
            f"prepared W=0 sector? {zero_winding == 0}  "
            f"({'OK' if sums_to_zero else 'FAIL'})"
        )

    ok = number_ok and winding_ok
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- both signed observables "
        "obey a zero-sum identity; no cross-domain identity is inferred"
    )
    print()
    return ok


# --------------------------------------------------------------------------- #
# (4) compare two distinct Z_2 actions
# --------------------------------------------------------------------------- #
def test_two_z2_actions_contrast():
    print("=" * 78)
    print("(4) TWO Z_2 actions: an automorphism commutes; Gamma anticommutes")
    print("=" * 78)
    G = nx.cycle_graph(6)  # bipartite, Aut = D_6
    nodes = list(G.nodes())
    A = adjacency(G, nodes)
    eye = np.eye(len(nodes))

    # Camino-5 Z_2: a graph automorphism (permutation) commutes with A
    mats = automorphism_matrices(G, nodes)
    P = next(
        M
        for M in mats
        if np.linalg.norm(M - eye) > 1e-9
        and np.linalg.norm(M @ M - eye) < TOL
    )
    p_is_perm = bool(
        np.allclose(P.sum(axis=0), 1)
        and np.allclose(P.sum(axis=1), 1)
        and np.allclose(P, P.astype(bool))
    )
    p_comm = commutator_norm(A, P)  # [A, P] = 0  (P A P^T = +A)
    p_invol = float(np.linalg.norm(P @ P - eye))

    # Camino-6 Z_2: the chiral diagonal Gamma anticommutes with A
    Gamma = chirality_operator(G, nodes)
    g_is_perm = bool(
        np.allclose(Gamma.sum(axis=0), 1)
        and np.allclose(np.abs(Gamma).sum(axis=1), 1)
        and np.all(Gamma >= 0)
    )
    g_comm = commutator_norm(A, Gamma)  # [A, Gamma] != 0
    g_anti = anticommutator_norm(A, Gamma)  # {A, Gamma} = 0
    g_invol = float(np.linalg.norm(Gamma @ Gamma - eye))

    print(
        f"  automorphism P (Camino 5): permutation? {p_is_perm}  "
        f"P^2=I:{p_invol:.1e}  [A,P]={p_comm:.2e} (COMMUTES, P A P^T=+A)"
    )
    print(
        f"  chiral Gamma  (Camino 6): permutation? {g_is_perm}  "
        f"Gamma^2=I:{g_invol:.1e}  [A,Gamma]={g_comm:.2e} (NOT 0)  "
        f"{{A,Gamma}}={g_anti:.2e} (ANTICOMMUTES, Gamma A Gamma=-A)"
    )
    print("  => both are involutions (Z_2), but the COMMUTING one builds the")
    print("     equivariance wall (Camino 5) while the ANTICOMMUTING one builds")
    print("     spectral sign pairing. Two distinct")
    print("     Z_2 actions on the same bipartite graph.")

    ok = (
        p_is_perm
        and p_invol < TOL
        and p_comm < TOL
        and (not g_is_perm)
        and g_invol < TOL
        and g_comm > 1e-3
        and g_anti < TOL
    )
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- two Z_2 actions with "
        "different representations and commutation relations"
    )
    print()
    return ok


def main():
    print(__doc__)
    results = [
        (
            "(1) chiral involution -> spectral sign pairing",
            test_chiral_gives_spectral_sign_pairing(),
        ),
        (
            "(2) phase negation -> winding sign flip",
            test_phase_negation_flips_winding(),
        ),
        ("(3) separate signed zero identities", test_signed_zero_identities()),
        (
            "(4) two Z_2 actions with distinct commutation",
            test_two_z2_actions_contrast(),
        ),
    ]
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {name:<52}: {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: the sublattice sign matrix and phase negation each generate")
    print("  a Z_2 action and reverse a signed observable. Gamma anticommutes with")
    print("  adjacency, forcing spectral +/- pairing; C: phi -> -phi reverses")
    print("  loop winding. They act on different spaces and no canonical map")
    print("  between those spaces is constructed. CONTRAST: Gamma ANTICOMMUTES")
    print("  with A, unlike the COMMUTING automorphism Z_2 that builds the Camino-5")
    print("  equivariance wall -- two different Z_2 on the same graph. HONEST SCOPE:")
    print("  exact finite Z_2 group actions on a graph; a precise structural")
    print("  comparison, not a derivation of physical charge conjugation or CPT.")
    print("  R and pi remain assumed substrate; this is Z, not R;")
    print("  nothing here touches G4 = RH, Navier-Stokes, or Yang-Mills.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
