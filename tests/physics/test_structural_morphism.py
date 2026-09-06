r"""Tests for structural morphisms — nodal-flow transports between networks.

Every TNFR morphism is an intertwiner ``M L_src = L_tgt M``, which is exactly the
condition that ``M`` carries the nodal-equation flow ``dEPI/dt = -L EPI`` from the
source network to the target one. Six kinds emerge from the nodal equation; the
folding endomorphism does not.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics.padic_tower import (
    compatible_connection_set,
    padic_laplacian,
    padic_lift_map,
    projective_scale_map,
)
from tnfr.physics import structural_morphism as sm
from tnfr.physics.directed_diffusion import directed_rw_laplacian
from tnfr.physics.structural_morphism import (
    StructuralMorphismKind,
    audit_structural_morphisms,
    certify_morphism,
    classify_morphism,
    finite_time_intertwining_bound,
    intertwining_residual,
    is_idempotent,
    is_partition_average,
    is_permutation_matrix,
    nodal_flow_preservation_residual,
)
from tnfr.physics.symmetry_sectors import permutation_matrix, reynolds_projector


def _lap(g):
    return directed_rw_laplacian(nx.to_numpy_array(g))


def _frac(matrix):
    return np.array([[float(x) for x in row] for row in matrix], dtype=float)


# --------------------------------------------------------------------------- #
# The emergence theorem: intertwining <=> nodal-flow preservation
# --------------------------------------------------------------------------- #
def test_intertwiner_preserves_nodal_flow():
    # p-adic coarse-graining is an exact intertwiner
    base = frozenset({1, 2})
    l_hi = _frac(padic_laplacian(3, 2, compatible_connection_set(3, 2, base)))
    l_lo = _frac(padic_laplacian(3, 1, compatible_connection_set(3, 1, base)))
    r = _frac(projective_scale_map(3, 1))
    assert intertwining_residual(r, l_hi, l_lo) < 1e-9
    assert nodal_flow_preservation_residual(r, l_hi, l_lo) < 1e-9


def test_non_intertwiner_does_not_preserve_nodal_flow():
    # the folding power map x -> x^2 mod 7 fails both, in lockstep
    p = 7
    fold = np.zeros((p, p))
    for x in range(p):
        fold[(x * x) % p, x] = 1.0
    lap = _lap(nx.cycle_graph(7))
    assert intertwining_residual(fold, lap, lap) > 1e-3
    assert nodal_flow_preservation_residual(fold, lap, lap) > 1e-3


def test_flow_residual_tracks_intertwining_defect():
    # a small non-commuting perturbation lifts both residuals together
    lap = _lap(nx.cycle_graph(5))
    perturb = np.eye(5)
    perturb[0, 1] = 0.4  # breaks commutation with L
    assert intertwining_residual(perturb, lap, lap) > 1e-6
    assert nodal_flow_preservation_residual(perturb, lap, lap) > 1e-6


def test_finite_time_duhamel_bound_handles_exact_and_perturbed_transport():
    lap = _lap(nx.cycle_graph(5))
    state = np.array([1.0, -1.0, 0.5, -0.5, 0.25])
    exact, exact_bound = finite_time_intertwining_bound(
        np.eye(5), lap, lap, state, structural_time=0.7
    )
    assert exact < 1e-12
    assert exact_bound < 1e-12

    perturb = np.eye(5)
    perturb[0, 1] = 0.4
    defect, bound = finite_time_intertwining_bound(
        perturb, lap, lap, state, structural_time=0.7
    )
    assert defect > 0.0
    assert defect <= bound + 1e-12


# --------------------------------------------------------------------------- #
# Classification of each canonical kind
# --------------------------------------------------------------------------- #
def test_automorphism_vs_relabeling_split_on_generator():
    lc = _lap(nx.cycle_graph(6))
    rot = permutation_matrix({i: (i + 1) % 6 for i in range(6)}, list(range(6)))
    assert classify_morphism(rot, lc, lc) is StructuralMorphismKind.AUTOMORPHISM
    lp = _lap(nx.path_graph(4))
    q = permutation_matrix({0: 2, 1: 0, 2: 3, 3: 1}, list(range(4)))
    assert classify_morphism(q, lp, q @ lp @ q.T) is \
        StructuralMorphismKind.RELABELING


def test_padic_scale_maps_are_coarse_graining_and_lift():
    base = frozenset({1, 2})
    l_hi = _frac(padic_laplacian(3, 2, compatible_connection_set(3, 2, base)))
    l_lo = _frac(padic_laplacian(3, 1, compatible_connection_set(3, 1, base)))
    r = _frac(projective_scale_map(3, 1))
    lift = _frac(padic_lift_map(3, 1))
    assert classify_morphism(r, l_hi, l_lo) is \
        StructuralMorphismKind.COARSE_GRAINING
    assert classify_morphism(lift, l_lo, l_hi) is StructuralMorphismKind.LIFT


def test_reynolds_projector_is_projection_not_endomorphism():
    # regression: the idempotent sector projector Q_Gamma emerges (intertwines)
    # and must NOT be swallowed by ENDOMORPHISM
    star = nx.star_graph(4)
    lap = _lap(star)
    q = reynolds_projector(star, nodes=list(range(5)))
    assert is_idempotent(q)
    assert intertwining_residual(q, lap, lap) < 1e-9
    assert classify_morphism(q, lap, lap) is StructuralMorphismKind.PROJECTION


def test_folding_power_map_is_endomorphism():
    p = 7
    fold = np.zeros((p, p))
    for x in range(p):
        fold[(x * x) % p, x] = 1.0
    lap = _lap(nx.cycle_graph(7))
    assert classify_morphism(fold, lap, lap) is \
        StructuralMorphismKind.ENDOMORPHISM


def test_nonpermutation_conjugation_is_intertwiner():
    ls = _lap(nx.cycle_graph(4))
    shear = np.array(
        [[1, 0.3, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.2], [0, 0, 0, 1]], dtype=float
    )
    lt = shear @ ls @ np.linalg.inv(shear)
    assert classify_morphism(shear, ls, lt) is \
        StructuralMorphismKind.INTERTWINER


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def test_predicates():
    perm = permutation_matrix({0: 1, 1: 0}, [0, 1])
    assert is_permutation_matrix(perm)
    assert not is_permutation_matrix(np.array([[0.5, 0.5], [0.5, 0.5]]))
    r = _frac(projective_scale_map(3, 1))
    assert is_partition_average(r)                 # fiber quotient
    assert not is_partition_average(np.eye(3))     # not dimension-dropping
    assert is_idempotent(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert not is_idempotent(np.array([[0.0, 1.0], [1.0, 0.0]]))


# --------------------------------------------------------------------------- #
# Certificate + audit
# --------------------------------------------------------------------------- #
def test_certificate_marks_morphism_not_operator():
    lc = _lap(nx.cycle_graph(6))
    rot = permutation_matrix({i: (i + 1) % 6 for i in range(6)}, list(range(6)))
    cert = certify_morphism(rot, lc, lc)
    assert cert.is_operator is False           # not one of the 13 operators
    assert cert.emerges_from_nodal_equation    # but it is a nodal-flow transport
    assert cert.is_intertwiner


def test_constant_probe_cannot_certify_nonintertwining_map():
    """A consensus probe can hide a generator-level transport defect."""
    source = np.array([[1.0, -1.0], [-1.0, 1.0]])
    target = 0.5 * source
    cert = certify_morphism(
        np.eye(2), source, target, flow_probe=np.ones(2)
    )
    assert cert.nodal_flow_residual < 1e-9
    assert cert.intertwining_residual > 1e-3
    assert not cert.is_intertwiner
    assert not cert.emerges_from_nodal_equation


@pytest.mark.parametrize(
    "morphism,source,target",
    [
        (np.eye(2), np.eye(3), np.eye(2)),
        (np.eye(2), np.array([[1.0, np.nan], [0.0, 1.0]]), np.eye(2)),
        (np.eye(2, dtype=complex) * (1.0 + 1.0j), np.eye(2), np.eye(2)),
    ],
)
def test_morphism_certificate_rejects_incompatible_or_nonfinite_systems(
    morphism, source, target
):
    with pytest.raises(ValueError):
        certify_morphism(morphism, source, target)


def test_audit_six_emerge_one_boundary():
    results = audit_structural_morphisms()
    assert len(results) == 7
    emerging = [c for _, c in results if c.emerges_from_nodal_equation]
    boundary = [c for _, c in results if not c.emerges_from_nodal_equation]
    assert len(emerging) == 6                  # the intertwiners
    assert len(boundary) == 1                  # the folding endomorphism
    assert boundary[0].kind is StructuralMorphismKind.ENDOMORPHISM
    assert all(not c.is_operator for _, c in results)


def test_kinds_are_relabel_invariant():
    # consistently relabelling source and target leaves the kind unchanged
    lc = _lap(nx.cycle_graph(6))
    rot = permutation_matrix({i: (i + 1) % 6 for i in range(6)}, list(range(6)))
    sigma = permutation_matrix({i: (i + 2) % 6 for i in range(6)}, list(range(6)))
    relabelled = sigma @ rot @ sigma.T
    lc2 = sigma @ lc @ sigma.T
    assert classify_morphism(relabelled, lc2, lc2) is \
        classify_morphism(rot, lc, lc)


def test_module_exports_complete():
    expected = {
        "StructuralMorphismKind", "is_permutation_matrix", "is_partition_average",
        "is_idempotent", "intertwining_residual", "finite_time_intertwining_bound",
        "nodal_flow_preservation_residual", "classify_morphism",
        "StructuralMorphismCertificate", "certify_morphism",
        "audit_structural_morphisms",
    }
    assert expected <= set(sm.__all__)
