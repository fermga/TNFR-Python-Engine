"""Supplied-group spectral comparisons and a multiplicity ambiguity control.

The icosahedral rotation group and its irrep dimensions are declared inputs.
The demo compares dimensions absent from a selected low-mode list with the
remaining dodecahedron spectrum and a related icosahedron spectrum. Membership
in an irrep table neither forces that irrep to occur in a given graph nor
bounds every accidental eigenspace multiplicity. These preselected examples
have no independent record of prospective preparation or held-out evaluation.

The retained truncated-cube counterexample is essential: a five-dimensional
eigenspace can split as 2+3 under octahedral symmetry, whereas the icosahedron
has an irreducible five-dimensional eigenspace. A count of five therefore does
not certify the icosahedral group. The executable result reports this ambiguity
and never infers a symmetry group from multiplicity alone.

All graphs and matrix operators are supplied. Comparisons use numerical
spectral clustering and complete finite automorphism enumeration on the stated
fixtures; no universal physical or arithmetic-emergence interpretation follows.
See theory/TNFR_STRUCTURAL_OBSERVABILITY.md#6-limits-beyond-linear-symmetry and the independent exact
counterexample in tests/physics/test_platonic_equilibrium_scope.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Numerical multiplicities of the supplied symmetric normalized Laplacian.
# On the regular fixtures it is a scalar multiple of D-A.
# ---------------------------------------------------------------------------


def laplacian_multiplicities(
    G: nx.Graph, *, tol: float = 1e-6
) -> list[tuple[float, int]]:
    """Return numerically clustered L_sym eigenvalues and multiplicities.

    The graph is supplied. The selected regular fixtures make L_sym and D-A
    scalar multiples; no equality with arbitrary equivariant operators follows.
    """
    from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian

    G = nx.Graph(G)
    _, L_sym = symmetric_normalized_laplacian(G)
    evals = np.sort(np.linalg.eigvalsh(L_sym))
    groups: list[list[float]] = [[float(evals[0])]]
    for ev in evals[1:]:
        if abs(ev - groups[-1][-1]) <= tol:
            groups[-1].append(float(ev))
        else:
            groups.append([float(ev)])
    return [(float(np.mean(g)), len(g)) for g in groups]


# ---------------------------------------------------------------------------
# Supplied finite rotation-group tables, not inferred graph symmetries
# ---------------------------------------------------------------------------

# Rotation point groups relevant to the polyhedral manifolds, with the full
# multiset of irreducible-representation dimensions (independent ground truth).
IRREP_DIMS: dict[str, list[int]] = {
    "T (tetrahedral)": [1, 1, 1, 3],  # |T| = 12
    "O (octahedral)": [1, 1, 2, 3, 3],  # |O| = 24
    "I (icosahedral)": [1, 3, 3, 4, 5],  # |I| = 60
}


def allowed_dims(group: str) -> set[int]:
    return set(IRREP_DIMS[group])


# ---------------------------------------------------------------------------
# Conditional comparison under an explicitly supplied group
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuppliedGroupComparison:
    supplied_group: str
    observed: list[int]
    unseen_irrep_dimensions: set[int]  # listed dimensions, not required occurrences
    group_inference_available: bool = False


def compare_supplied_group(
    observed_seq: list[int], *, group: str
) -> SuppliedGroupComparison:
    """Compare a coarse mode list with a supplied irrep table, without inference."""
    dims = allowed_dims(group)
    seen = {m for m in observed_seq}
    unseen = {d for d in dims if d > 1 and d not in seen}
    return SuppliedGroupComparison(
        supplied_group=group,
        observed=observed_seq,
        unseen_irrep_dimensions=unseen,
    )


def low_modes(G: nx.Graph, n_groups: int) -> list[int]:
    """Select the multiplicities of the first ``n_groups`` distinct eigenvalues."""
    return [m for _ev, m in laplacian_multiplicities(G)][:n_groups]


def full_modes(G: nx.Graph) -> list[int]:
    return [m for _ev, m in laplacian_multiplicities(G)]


# ---------------------------------------------------------------------------
# Representation-theoretic irreducibility test (protected vs accidental)
# ---------------------------------------------------------------------------


def automorphism_matrices(G: nx.Graph, *, limit: int = 5000) -> list[np.ndarray]:
    """Enumerate the complete group, rejecting if it exceeds ``limit``.

    Aut(G) is exactly the symmetry group with which L = D - A commutes; averaging
    over it gives the rep-theory inner product used to detect irreducibility.
    """
    from networkx.algorithms.isomorphism import GraphMatcher

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("automorphism limit must be a positive integer")
    G = nx.Graph(G)
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    mats: list[np.ndarray] = []
    for mapping in GraphMatcher(G, G).isomorphisms_iter():
        if len(mats) >= limit:
            raise ValueError("incomplete automorphism enumeration: limit exceeded")
        M = np.zeros((n, n))
        for src, dst in mapping.items():
            M[idx[dst], idx[src]] = 1.0
        mats.append(M)
    return mats


def eigenspace_irreducibility(
    G: nx.Graph, *, tol: float = 1e-5
) -> list[tuple[float, int, float]]:
    """For each Laplacian eigenspace return (eigenvalue, multiplicity, <chi,chi>).

    With complete enumeration, <chi,chi> is the sum of squared irrep
    multiplicities: 1 means irreducible and a value above 1 means reducible.
    The fixture groups have fewer than the enumeration cap of 5000 elements.
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
        U = evecs[:, grp]  # n x d, orthonormal columns
        P = U @ U.T  # projector onto the eigenspace
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


def compare_icosahedral_graphs() -> bool:
    """Compare two chosen graphs with the supplied icosahedral irrep table."""
    _rule("SUPPLIED ICOSAHEDRAL GROUP — compare two selected graph spectra")

    ico = nx.icosahedral_graph()
    observed = low_modes(ico, 3)
    comparison = compare_supplied_group(observed, group="I (icosahedral)")
    print(f"  observed (icosahedron, low modes only): {observed}")
    print(f"  externally supplied group           : {comparison.supplied_group}")
    print(
        f"  group inference available           : {comparison.group_inference_available}"
    )
    print(
        f"  listed but unseen irrep dimensions  : {sorted(comparison.unseen_irrep_dimensions)}"
    )
    print("  An irrep table does not require every dimension to occur in a graph.")

    # Both graph families were selected before this descriptive comparison.
    dodeca_full = full_modes(nx.dodecahedral_graph())
    dodeca_set = set(dodeca_full)
    print(f"\n  selected dodecahedron spectrum        : {dodeca_full}")
    four_appears = 4 in dodeca_set
    print(f"  listed 4 appears in dodecahedron      : {four_appears}")

    # The absent 4 also shows that an allowed irrep need not occur.
    ico_full = full_modes(ico)
    print(
        f"  icosahedron full spectrum             : {ico_full}  "
        f"(note: never shows a 4)"
    )

    ok = four_appears and (4 not in set(ico_full))
    print(f"\n  FINITE COMPARISON: {'PASS' if ok else 'FAIL'}")
    return ok


def control_irreducibility() -> bool:
    """Retain the fivefold ambiguity and finite character-inner-product controls."""
    _rule("IRREDUCIBLE vs COMPOSITE — protected degeneracy = irreducible rep")
    print("  <chi,chi> sums squared irrep multiplicities: 1 => irreducible,")
    print("  values above 1 => reducible. Computed over the complete fixture group.\n")

    ok = True
    icosahedral_five_seen = False

    print("  Icosahedron (group I — HAS a 5D irrep):")
    for _ev, mult, norm in eigenspace_irreducibility(nx.icosahedral_graph()):
        kind = (
            "irreducible (protected)"
            if abs(norm - 1) < 0.3
            else f"reducible (character norm {norm:.4f})"
        )
        flag = "   <- the 5 is a PROTECTED irrep" if mult == 5 else ""
        print(f"    mult={mult}  <chi,chi>={norm:4.1f}  {kind}{flag}")
        if mult == 5:
            icosahedral_five_seen = True
            ok = ok and abs(norm - 1) < 0.3

    octahedral_five_seen = False
    print("\n  Truncated cube (group O — selected reducible fivefold control):")
    for _ev, mult, norm in eigenspace_irreducibility(nx.truncated_cube_graph()):
        if abs(norm - 1) < 0.3:
            kind = "irreducible (protected)"
        else:
            kind = f"reducible (character norm {norm:.4f})"
        flag = "   <- fivefold eigenspace" if mult == 5 else ""
        print(f"    mult={mult}  <chi,chi>={norm:4.1f}  {kind}{flag}")
        if mult == 5:
            octahedral_five_seen = True
            ok = ok and (round(norm) == 2)
    ok = ok and icosahedral_five_seen and octahedral_five_seen

    print("\n  Finite absence check: neither selected spectrum has multiplicity two.")
    ico_m = full_modes(nx.icosahedral_graph())
    dod_m = full_modes(nx.dodecahedral_graph())
    no2 = (2 not in ico_m) and (2 not in dod_m)
    print(f"    icosahedron {ico_m}, dodecahedron {dod_m}: no 2-fold = {no2}")
    ok = ok and no2

    print(
        f"\n  VERDICT: {'PASS — fivefold multiplicity is ambiguous across the two groups' if ok else 'FAIL'}"
    )
    print(
        "  The count five alone does not certify I; the truncated cube has O symmetry."
    )
    return ok


def compare_dodecahedron_mode_subsets() -> bool:
    """Describe two selected mode subsets under the supplied group assumption."""
    _rule("DODECAHEDRON — compare low and remaining mode subsets")

    dodeca = nx.dodecahedral_graph()
    full = full_modes(dodeca)
    observed = full[:3]  # reveal [1, 3, 5]; hide [4, 4, 3]
    hidden = full[3:]
    comparison = compare_supplied_group(observed, group="I (icosahedral)")
    print(f"  dodecahedron — revealed low modes     : {observed}")
    print(f"  externally supplied group           : {comparison.supplied_group}")
    print(
        f"  listed dimensions absent below      : {sorted(comparison.unseen_irrep_dimensions)}"
    )

    revealed_hidden = hidden
    hit = bool(comparison.unseen_irrep_dimensions & set(revealed_hidden))
    print(f"\n  reveal hidden modes                   : {revealed_hidden}")
    print(f"  listed dimension occurs in remaining set: {hit}")
    print(f"\n  FINITE COMPARISON: {'PASS' if hit else 'FAIL'}")
    return hit


def main() -> int:
    print(__doc__)
    r1 = compare_icosahedral_graphs()
    r2 = compare_dodecahedron_mode_subsets()
    r3 = control_irreducibility()

    _rule("SUMMARY")
    print(f"  supplied-group graph comparison        : {'PASS' if r1 else 'FAIL'}")
    print(f"  selected dodecahedron mode subsets      : {'PASS' if r2 else 'FAIL'}")
    print(f"  fivefold multiplicity ambiguity control : {'PASS' if r3 else 'FAIL'}")
    overall = r1 and r2 and r3
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'MISMATCH'}")
    print("\n  Group inference from these coarse multiplicities: UNAVAILABLE.")
    print("  The group table and graph choices were supplied; no prospective")
    print("  validation or required occurrence of unseen irreps is claimed.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
