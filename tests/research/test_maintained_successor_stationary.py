"""Restricted-root contraction and the full closed-acute KKT obstruction."""

from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import maintained_successor_stationary as owner
from tnfr.research.maintained_parent_um_source import load_thol_successor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def certificate():
    source = load_thol_successor(
        ROOT / "docs/assets/maintained_parent_thol", repository_root=ROOT
    )
    return owner.certify_successor_stationary(source.graph)


def _certify(**overrides):
    arguments = dict(edges=owner.SELECTED_SUPPORT, shifts=owner.SELECTED_SHIFTS)
    arguments.update(overrides)
    return owner._certify_face(**arguments)


def test_checked_contraction_includes_a_restricted_root(certificate):
    face = certificate.face
    assert face.eta < Q(1, 10**12)
    assert face.q < Q(1, 100000)
    assert face.image_radius_bound < face.radius / 1000
    assert face.root_inclusion_certified
    # The independently checked map is strictly inward on every box face.
    for component, row in zip(face.preconditioned_residual, face.contraction_matrix):
        assert (
            component.abs_max + sum(value.abs_max for value in row) * face.radius
            < face.radius
        )


def test_full_boundary_witness_is_not_a_stationary_state(certificate):
    face = certificate.face
    assert Q(14, 1000) < face.multiplier.lo < face.multiplier.hi < Q(15, 1000)
    assert all(face.full_gradient_bounds[i].hi < 0 for i in (10, 11))
    assert all(face.full_gradient_bounds[i].lo > 0 for i in (14, 15))
    assert all(
        value.contains(0)
        for i, value in enumerate(face.full_gradient_bounds)
        if i not in (10, 11, 14, 15)
    )
    assert face.kkt_certified and face.closed_stationary_excluded


def test_all_twelve_face_edges_are_exact_affine_constraints(certificate):
    face = certificate.face
    active = {certificate.chamber.edges[i] for i in face.active_edge_indices}
    expected = {
        (i, j)
        for left, right in (
            ((10, 11), (18, 19)),
            ((14, 15), (16, 17)),
            ((16, 17), (18, 19)),
        )
        for i in left
        for j in right
    }
    assert active == expected
    for index in face.active_edge_indices:
        assert not any(face.edge_coefficients[index])
        assert abs(face.edge_pi_coefficients[index]) == Q(1, 2)
        assert face.acute_margins[index] == I(0)
    assert face.minimum_nonactive_margin > Q(19, 100)
    assert len(face.acute_margins) - len(face.active_edge_indices) == 41


def test_checked_source_has_positive_form_storage_for_exact_energy_path(certificate):
    # The child and parent keep different form coordinates and a positive
    # proposed conductance. Recompute their independent contribution alone.
    chamber = certificate.chamber
    index = chamber.edges.index((2, 20))
    difference = chamber.reset.epi[2] - chamber.reset.epi[20]
    assert difference != 0
    ideal_child = chamber.conductances[index] * difference**2 / 2
    conditional_child = chamber.conditional_conductances[index] * difference**2 / 2
    assert 0 < ideal_child.lo <= certificate.source_form_storage.lo
    assert 0 < conditional_child.lo <= certificate.conditional_source_form_storage.lo
    assert certificate.source_form_storage.subset_of(
        certificate.conditional_source_form_storage
    )


def test_zero_preconditioner_cannot_certify_root_or_obstruction():
    face = _certify(inverse=((0,) * 5,) * 5)
    assert face.q == 1
    assert face.image_radius_bound == face.radius
    assert not face.root_inclusion_certified
    assert not face.kkt_certified
    assert not face.closed_stationary_excluded


def test_a_finite_wrong_center_is_not_admitted_as_a_root():
    face = _certify(center=(0,) * 5)
    assert not face.root_inclusion_certified
    assert not face.closed_stationary_excluded


def test_a_wide_box_does_not_inherit_the_small_box_certificate():
    face = _certify(radius=Q(1, 4))
    assert face.q >= 1
    assert face.minimum_nonactive_margin < 0
    assert not face.kkt_certified


@pytest.mark.parametrize("radius", (True, False, float("inf"), float("nan"), 0, -1))
def test_invalid_radius_rejects_before_interval_materialization(radius):
    with pytest.raises((TypeError, ValueError)):
        _certify(radius=radius)


@pytest.mark.parametrize(
    "overrides",
    (
        {"center": (True, 0, 0, 0, 0)},
        {"center": (float("inf"), 0, 0, 0, 0)},
        {"center": (0,) * 4},
        {"inverse": ((0,) * 4,) * 5},
        {"inverse": ((0,) * 5,) * 4},
        {"inverse": ((False,) * 5,) * 5},
    ),
)
def test_invalid_witness_coordinates_reject(overrides):
    with pytest.raises((TypeError, ValueError)):
        _certify(**overrides)


@pytest.mark.parametrize(
    "case", ("missing", "reverse", "boolean", "wrong_branch", "boolean_branch")
)
def test_the_full_support_and_branch_vector_are_premises(case):
    edges, shifts = list(owner.SELECTED_SUPPORT), list(owner.SELECTED_SHIFTS)
    if case == "missing":
        edges.pop()
    elif case == "reverse":
        edges[0] = tuple(reversed(edges[0]))
    elif case == "boolean":
        edges[0] = (False, edges[0][1])
    elif case == "wrong_branch":
        shifts[0] += 1
    else:
        shifts[0] = False
    with pytest.raises((TypeError, ValueError)):
        _certify(edges=tuple(edges), shifts=tuple(shifts))


def test_interval_potential_hessian_has_independent_single_edge_controls():
    coefficients = ((1, -1, 0, 0, 0),)
    gradient, hessian = owner._gradient_and_hessian(coefficients, (I(0),))
    assert all(value.contains(0) for value in gradient)
    for i, row in enumerate(hessian):
        for j, value in enumerate(row):
            expected = coefficients[0][i] * coefficients[0][j]
            assert value.contains(expected)
    _, nonacute = owner._gradient_and_hessian(coefficients, (pi_interval(),))
    assert nonacute[0][0].hi < Q(-99, 100)
    assert nonacute[0][1].lo > Q(99, 100)
