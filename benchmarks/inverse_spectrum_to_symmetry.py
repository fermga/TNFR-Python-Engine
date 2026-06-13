"""
Inverse Falsifier: Spectrum -> Symmetry Group -> Predict an UNMEASURED Degeneracy
=================================================================================

The forward harness (``emergent_integers_symmetry.py``) showed that the integers
emerging as structural-Laplacian multiplicities match the irreducible-representation
(irrep) dimensions of the manifold's symmetry group. That is necessary but weak:
a skeptic can say "you matched a template you already knew."

This harness runs the HARD test — a genuine out-of-sample prediction:

    1. OBSERVE only the low modes of a manifold (hide the rest).
    2. INFER the symmetry group from the partial multiplicity fingerprint alone.
    3. PREDICT, from pure group theory, a degeneracy that was NOT in the observed
       data (an irrep dimension the low modes never revealed).
    4. REVEAL the hidden modes (or a held-out sibling manifold) and check whether
       the predicted integer actually appears.

If the prediction lands, the integer was dictated by structure, not fitted —
the claim "TNFR explains what an integer IS (a structural invariant)" survives a
real falsification attempt. If it fails, the earlier match was a template artifact.

HEADLINE CASE (the cleanest, least circular):
  The icosahedral rotation group I has irreps of dimensions {1, 3, 3, 4, 5}
  (sum of squares 1+9+9+16+25 = 60 = |I|). But the 12-vertex icosahedron graph
  decomposes as 12 = 1+3+5+3 and NEVER exhibits a multiplicity of 4. So:

      observe icosahedron low modes [1, 3, 5]
        -> the 5 forces group = I (no smaller point group has a 5D irrep)
        -> group theory: I HAS a 4D irrep (the "G" representation)
        -> PREDICT a degeneracy of 4 must appear in this symmetry family,
           even though the icosahedron itself never shows it.
      verify on the held-out dodecahedron (dual polyhedron, same group I_h):
        20 = 1+3+5+4+4+3  -> the 4 APPEARS.  Prediction confirmed.

  The integer 4 was absent from the input and predicted from structure alone.

IRREDUCIBLE vs COMPOSITE DEGENERACY (the deeper finding):
  A naive "ceiling" prediction (no multiplicity above the max irrep dim) is FALSE:
  the truncated cube (octahedral, max irrep dim 3) shows a 5-fold degeneracy,
  because 5 = 2 + 3 is an ACCIDENTAL coincidence of a 2D and a 3D irrep. So
  emergent degeneracies split into two kinds:
    - PROTECTED  (irreducible rep): multiplicity = a single irrep dimension,
      symmetry-forced, stable.  <chi,chi> = 1.
    - ACCIDENTAL (reducible / direct sum): multiplicity = a SUM of irrep dims,
      not symmetry-forced.  <chi,chi> = number of irreps in the sum > 1.
  We verify this with the representation-theoretic inner product <chi,chi>
  computed from the graph's automorphism group: the icosahedron's 5 is
  irreducible (protected); the truncated cube's 5 is reducible (2 + 3,
  accidental). This is the rep-theory notion of irreducibility — the
  "indivisible building block" idea, realized for degeneracies (the same
  irreducible/composite intuition that underlies primes).

  SHARP EXCLUSION that DOES hold: the icosahedral group has no 2D irrep, so an
  icosahedral manifold shows NO generic 2-fold degeneracy. Verified on the
  icosahedron and the dodecahedron.

HONEST SCOPE:
  The predictive engine is L = D - A commuting with Aut(G) (a known theorem); TNFR
  supplies the physical reading L = discrete ΔNFR / phase curvature. We predict
  CARDINALS (degeneracies), not the arithmetic ring. This does not derive (+, ×)
  or primality. It does demonstrate that the emergent integers carry, and let us
  predict, structural facts we did not put in.

Run:
    python benchmarks/inverse_spectrum_to_symmetry.py

Theoretical anchor: AGENTS.md (nodal equation; discrete-mode regime; structural
Laplacian as discrete ΔNFR). Status: RESEARCH (inverse falsifier).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Emergent integers: multiplicities of the structural Laplacian L = D - A
# ---------------------------------------------------------------------------


def laplacian_multiplicities(G: nx.Graph, *, tol: float = 1e-6) -> list[tuple[float, int]]:
    """Return (eigenvalue, multiplicity) pairs of L = D - A, ascending."""
    G = nx.Graph(G)
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    D = np.diag(A.sum(axis=1))
    evals = np.sort(np.linalg.eigvalsh(D - A))
    groups: list[list[float]] = [[float(evals[0])]]
    for ev in evals[1:]:
        if abs(ev - groups[-1][-1]) <= tol:
            groups[-1].append(float(ev))
        else:
            groups.append([float(ev)])
    return [(float(np.mean(g)), len(g)) for g in groups]


# ---------------------------------------------------------------------------
# Group inference from a partial multiplicity fingerprint (rep-theory table)
# ---------------------------------------------------------------------------

# Rotation point groups relevant to the polyhedral manifolds, with the full
# multiset of irreducible-representation dimensions (independent ground truth).
IRREP_DIMS: dict[str, list[int]] = {
    "C/D (cyclic/dihedral)": [1, 1, 2],      # dims that occur: {1,2}
    "T (tetrahedral)": [1, 1, 1, 3],          # |T| = 12
    "O (octahedral)": [1, 1, 2, 3, 3],        # |O| = 24
    "I (icosahedral)": [1, 3, 3, 4, 5],       # |I| = 60
}


def allowed_dims(group: str) -> set[int]:
    return set(IRREP_DIMS[group])


def infer_group(observed_mults: set[int]) -> str:
    """Infer the minimal symmetry group consistent with the observed nontrivial
    multiplicities. The inference uses ONLY the observed integers.
    """
    nz = {m for m in observed_mults if m > 1}
    if 5 in nz or 4 in nz:
        return "I (icosahedral)"          # only icosahedral has dims 4 or 5
    if 3 in nz and 2 in nz:
        return "O (octahedral)"           # 2 and 3 together -> octahedral
    if 3 in nz:
        return "T (tetrahedral)"          # 3 without 2 -> tetrahedral
    if 2 in nz:
        return "C/D (cyclic/dihedral)"
    return "C/D (cyclic/dihedral)"


# ---------------------------------------------------------------------------
# Prediction primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    inferred_group: str
    observed: list[int]
    predicted_existing: set[int]   # irrep dims the group HAS but we have not seen
    forbidden_above: int           # no multiplicity may exceed this


def make_prediction(observed_seq: list[int]) -> Prediction:
    group = infer_group(set(observed_seq))
    dims = allowed_dims(group)
    seen = {m for m in observed_seq}
    predicted_existing = {d for d in dims if d > 1 and d not in seen}
    return Prediction(
        inferred_group=group,
        observed=observed_seq,
        predicted_existing=predicted_existing,
        forbidden_above=max(dims),
    )


def low_modes(G: nx.Graph, n_groups: int) -> list[int]:
    """Reveal only the multiplicities of the first ``n_groups`` distinct
    eigenvalues (the lowest structural modes). The rest stay hidden."""
    return [m for _ev, m in laplacian_multiplicities(G)][:n_groups]


def full_modes(G: nx.Graph) -> list[int]:
    return [m for _ev, m in laplacian_multiplicities(G)]


# ---------------------------------------------------------------------------
# Representation-theoretic irreducibility test (protected vs accidental)
# ---------------------------------------------------------------------------


def automorphism_matrices(G: nx.Graph, *, limit: int = 5000) -> list[np.ndarray]:
    """All graph automorphisms of G as permutation matrices (capped at ``limit``).

    Aut(G) is exactly the symmetry group with which L = D - A commutes; averaging
    over it gives the rep-theory inner product used to detect irreducibility.
    """
    from networkx.algorithms.isomorphism import GraphMatcher

    G = nx.Graph(G)
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    mats: list[np.ndarray] = []
    for mapping in GraphMatcher(G, G).isomorphisms_iter():
        M = np.zeros((n, n))
        for src, dst in mapping.items():
            M[idx[dst], idx[src]] = 1.0
        mats.append(M)
        if len(mats) >= limit:
            break
    return mats


def eigenspace_irreducibility(
    G: nx.Graph, *, tol: float = 1e-5
) -> list[tuple[float, int, float]]:
    """For each Laplacian eigenspace return (eigenvalue, multiplicity, <chi,chi>).

    <chi,chi> = (1/|Aut|) Σ_g |trace(P_λ · M_g)|² counts the irreducible
    representations inside the eigenspace: 1 => irreducible (symmetry-protected);
    k>1 => reducible (an accidental sum of k irreps).
    """
    G = nx.Graph(G)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    L = np.diag(A.sum(axis=1)) - A
    evals, evecs = np.linalg.eigh(L)
    mats = automorphism_matrices(G)
    order = len(mats)

    groups: list[list[int]] = [[0]]
    for i in range(1, len(evals)):
        if abs(evals[i] - evals[groups[-1][-1]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])

    out: list[tuple[float, int, float]] = []
    for grp in groups:
        U = evecs[:, grp]               # n x d, orthonormal columns
        P = U @ U.T                     # projector onto the eigenspace
        s = sum(float(np.trace(P @ M)) ** 2 for M in mats)
        out.append((float(evals[grp[0]]), len(grp), s / order))
    return out


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def headline_icosahedral_prediction() -> bool:
    """Predict the unmeasured '4' from the icosahedron's low modes; verify on
    the held-out dodecahedron AND on the icosahedron's own hidden modes."""
    _rule("HEADLINE — predict an unmeasured degeneracy (the hidden '4')")

    ico = nx.icosahedral_graph()
    observed = low_modes(ico, 3)              # reveal only [1, 3, 5]
    pred = make_prediction(observed)
    print(f"  observed (icosahedron, low modes only): {observed}")
    print(f"  inferred group (from the 5)           : {pred.inferred_group}")
    print(f"  group-theory irrep dims               : "
          f"{sorted(allowed_dims(pred.inferred_group))}")
    print(f"  PREDICTION: a degeneracy in {sorted(pred.predicted_existing)} must exist in")
    print("              this symmetry family, though unseen in the input.")
    print(f"  PREDICTION: no multiplicity will ever exceed {pred.forbidden_above}.")

    # Verify on the held-out dodecahedron (dual polyhedron, same group I_h).
    dodeca_full = full_modes(nx.dodecahedral_graph())
    dodeca_set = set(dodeca_full)
    print(f"\n  held-out dodecahedron full spectrum   : {dodeca_full}")
    four_appears = 4 in dodeca_set
    no_excess = max(dodeca_full) <= pred.forbidden_above
    print(f"  predicted 4 appears in dodecahedron   : {four_appears}")
    print(f"  no multiplicity exceeds {pred.forbidden_above}            : {no_excess}")

    # Also confirm the icosahedron itself genuinely hides the 4.
    ico_full = full_modes(ico)
    print(f"  icosahedron full spectrum             : {ico_full}  "
          f"(note: never shows a 4)")

    ok = four_appears and no_excess and (4 not in set(ico_full))
    print(f"\n  VERDICT: {'PASS — predicted an integer absent from the input' if ok else 'FAIL'}")
    return ok


def control_irreducibility() -> bool:
    """Prove the protected/accidental split with the character inner product, and
    verify the sharp exclusion (no 2-fold degeneracy in icosahedral symmetry)."""
    _rule("IRREDUCIBLE vs COMPOSITE — protected degeneracy = irreducible rep")
    print("  <chi,chi> counts irreps in an eigenspace: 1 => irreducible (protected),")
    print("  k>1 => reducible (accidental sum of k irreps). Computed over Aut(G).\n")

    ok = True

    print("  Icosahedron (group I — HAS a 5D irrep):")
    for _ev, mult, norm in eigenspace_irreducibility(nx.icosahedral_graph()):
        kind = "irreducible (protected)" if abs(norm - 1) < 0.3 else f"reducible (~{round(norm)} irreps)"
        flag = "   <- the 5 is a PROTECTED irrep" if mult == 5 else ""
        print(f"    mult={mult}  <chi,chi>={norm:4.1f}  {kind}{flag}")
        if mult == 5:
            ok = ok and abs(norm - 1) < 0.3

    tc = getattr(nx, "truncated_cube_graph", None)
    if tc is not None:
        print("\n  Truncated cube (group O — NO 5D irrep, so a 5 must be accidental):")
        for _ev, mult, norm in eigenspace_irreducibility(tc()):
            if abs(norm - 1) < 0.3:
                kind = "irreducible (protected)"
            else:
                kind = f"reducible: {mult} = sum of {round(norm)} irreps (ACCIDENTAL)"
            flag = "   <- the 5 = 2(+)3, NOT protected" if mult == 5 else ""
            print(f"    mult={mult}  <chi,chi>={norm:4.1f}  {kind}{flag}")
            if mult == 5:
                ok = ok and (round(norm) == 2)

    print("\n  Sharp exclusion (icosahedral has NO 2D irrep -> no generic 2-fold):")
    ico_m = full_modes(nx.icosahedral_graph())
    dod_m = full_modes(nx.dodecahedral_graph())
    no2 = (2 not in ico_m) and (2 not in dod_m)
    print(f"    icosahedron {ico_m}, dodecahedron {dod_m}: no 2-fold = {no2}")
    ok = ok and no2

    print(f"\n  VERDICT: {'PASS — same integer 5 is irreducible in I, reducible (2+3) in O; exclusion holds' if ok else 'FAIL'}")
    return ok


def within_manifold_prediction() -> bool:
    """Strongest non-circular form: reveal a manifold's low modes, predict its
    OWN hidden higher modes contain a structurally-required integer."""
    _rule("WITHIN-MANIFOLD — predict a manifold's own hidden modes")

    dodeca = nx.dodecahedral_graph()
    full = full_modes(dodeca)
    observed = full[:3]                       # reveal [1, 3, 5]; hide [4, 4, 3]
    hidden = full[3:]
    pred = make_prediction(observed)
    print(f"  dodecahedron — revealed low modes     : {observed}")
    print(f"  dodecahedron — hidden higher modes    : {'?' * len(hidden)} (concealed)")
    print(f"  inferred group (from the 5)           : {pred.inferred_group}")
    print(f"  PREDICTION: hidden modes must include a degeneracy in "
          f"{sorted(pred.predicted_existing)}.")

    revealed_hidden = hidden
    hit = bool(pred.predicted_existing & set(revealed_hidden))
    print(f"\n  reveal hidden modes                   : {revealed_hidden}")
    print(f"  predicted integer found in hidden set : {hit}")
    print(f"\n  VERDICT: {'PASS — hidden degeneracy predicted before revealing' if hit else 'FAIL'}")
    return hit


def main() -> None:
    print(__doc__)
    r1 = headline_icosahedral_prediction()
    r2 = within_manifold_prediction()
    r3 = control_irreducibility()

    _rule("SUMMARY")
    print(f"  headline (predict hidden 4 on sibling)   : {'PASS' if r1 else 'FAIL'}")
    print(f"  within-manifold (predict own hidden mode): {'PASS' if r2 else 'FAIL'}")
    print(f"  irreducible/composite + sharp exclusion  : {'PASS' if r3 else 'FAIL'}")
    overall = r1 and r2 and r3
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'MISMATCH'}")
    print("\n  Reading: the inverse map (partial spectrum -> group -> unseen integer)")
    print("  succeeds, and the emergent degeneracies carry an irreducible/composite")
    print("  structure: protected integers are irreducible reps, accidental ones are")
    print("  sums. This is the structural reading of 'L commutes with Aut(G)' — TNFR")
    print("  supplies L = discrete ΔNFR. It yields cardinals and their irreducibility,")
    print("  NOT the arithmetic ring; (+, ×, primality) of integers stay open.")


if __name__ == "__main__":
    main()
