"""Analytic controls for detached stationary root inclusion."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import repeat

import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import _organized_stationary_certificate as owner
from tnfr.physics._organized_external_interaction import (
    _derive_organized_external_interaction,
)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_frozen_execution():
    with forbid_sine_regeneration():
        yield


def _pair(**overrides):
    arguments = {
        "phase_center": (0, 0),
        "edges": ((0, 1),),
        "radius": Q(1, 100),
        "inverse": ((Q(1),),),
    }
    arguments.update(overrides)
    return owner._certify_organized_stationary(**arguments)


def test_pair_root_certificate_has_independent_exact_bounds():
    report = _pair()
    assert report.root_inclusion_certified
    assert report.relative_hessian_positive
    assert report.stable_stationary_certified
    assert report.phase_current_bounds[0].contains(0)
    assert report.phase_current_bounds[1].contains(0)
    assert report.grounded_hessian_bounds[0][0].contains(1)
    assert report.q0 < Q(1, 10**30)
    assert report.qbox == report.q0 + Q(1, 25)
    assert report.image_radius_bound == report.eta + report.qbox / 100
    assert report.image_radius_bound < report.radius
    assert Q(98, 100) < report.lower_grounded_hessian[0][0] <= Q(99, 100)
    assert report.lower_weight_bits == 32
    with pytest.raises(FrozenInstanceError):
        report.root_inclusion_certified = False


def test_nonzero_residual_can_include_the_known_pair_root():
    report = _pair(phase_center=(0, Q(1, 1000)))
    assert not report.phase_current_bounds[1].contains(0)
    assert report.phase_center[1] - report.radius < 0
    assert report.phase_center[1] + report.radius > 0
    assert report.stable_stationary_certified


def test_positive_hessian_does_not_turn_a_small_box_into_a_root():
    report = _pair(phase_center=(0, Q(1, 2)))
    assert report.contraction_certified
    assert report.relative_hessian_positive
    assert report.image_radius_bound > report.radius
    assert not report.root_inclusion_certified
    assert not report.stable_stationary_certified


def test_zero_residual_does_not_admit_a_bad_inverse():
    report = _pair(inverse=((0,),))
    assert all(value.contains(0) for value in report.phase_current_bounds)
    assert report.eta == 0
    assert report.q0 == report.qbox == 1
    assert not report.contraction_certified
    assert not report.root_inclusion_certified
    assert report.relative_hessian_positive


def test_root_inclusion_does_not_admit_a_stationary_saddle():
    report = _pair(
        phase_center=(0, pi_interval().midpoint),
        inverse=((-1,),),
    )
    assert report.root_inclusion_certified
    assert report.lower_grounded_hessian[0][0] == -1
    assert not report.relative_hessian_positive
    assert not report.stable_stationary_certified


def test_three_node_grounding_and_supplied_inverse_are_checked():
    report = owner._certify_organized_stationary(
        phase_center=(0, 0, 0),
        edges=((0, 1), (0, 2), (1, 2)),
        radius=Q(1, 1000),
        inverse=((Q(2, 3), Q(1, 3)), (Q(1, 3), Q(2, 3))),
    )
    assert report.stable_stationary_certified
    expected = ((2, -1), (-1, 2))
    for row, wanted in zip(report.grounded_hessian_bounds, expected):
        assert all(value.contains(target) for value, target in zip(row, wanted))
    assert report.inverse_norm == 1
    assert report.maximum_degree == 2
    assert report.qbox == report.q0 + Q(8, 1000)
    # Edges meeting the fixed gauge move by r; the third moves by 2*r.
    assert report.lower_edge_hessian[2] < report.lower_edge_hessian[0]
    assert report.lower_edge_hessian[0] == report.lower_edge_hessian[1]


def test_full_contact_dimension_certifies_known_constant_phase_control():
    # Constant phase is an analytic root on this connected support, with
    # Hessian exactly its unit Laplacian. This exercises all 53 free phases;
    # it neither searches for a target nor retains the winding-one identity.
    geometry = _derive_organized_external_interaction()
    edges = tuple((left, right) for left, right, _ in geometry.weighted_edges)
    assert len(geometry.node_order) == 54
    assert len(edges) == 144
    assert all(weight == 1 for _, _, weight in geometry.weighted_edges)
    laplacian = np.zeros((54, 54), dtype=np.int64)
    for left, right in edges:
        laplacian[left, left] += 1
        laplacian[right, right] += 1
        laplacian[left, right] -= 1
        laplacian[right, left] -= 1
    assert tuple(laplacian.diagonal()) == (5,) * 45 + (7,) * 9
    proposed_inverse = np.linalg.inv(laplacian[1:, 1:])
    assert np.isfinite(proposed_inverse).all()
    scale = 2**40
    inverse = tuple(
        tuple(Q(int(round(float(value) * scale)), scale) for value in row)
        for row in proposed_inverse
    )
    report = owner._certify_organized_stationary(
        phase_center=(0,) * 54,
        edges=edges,
        radius=Q(1, 10**7),
        inverse=inverse,
    )
    assert len(report.grounded_hessian_bounds) == 53
    assert all(len(row) == 53 for row in report.grounded_hessian_bounds)
    for actual, expected in zip(report.grounded_hessian_bounds, laplacian[1:, 1:]):
        assert all(bound.contains(int(value)) for bound, value in zip(actual, expected))
    assert all(bound.contains(0) for bound in report.phase_current_bounds)
    assert report.maximum_degree == 7
    assert report.qbox == report.q0 + 28 * report.radius * report.inverse_norm
    assert report.root_inclusion_certified
    assert report.relative_hessian_positive
    assert report.stable_stationary_certified


def test_a_compensated_negative_edge_can_have_positive_relative_hessian():
    report = owner._certify_organized_stationary(
        phase_center=(0, Q(9, 10), Q(-9, 10)),
        edges=((0, 1), (0, 2), (1, 2)),
        radius=Q(1, 10**5),
        inverse=((1, 0), (0, 1)),
    )
    assert report.lower_edge_hessian[2] < 0
    assert report.relative_hessian_positive
    assert not report.root_inclusion_certified
    assert not report.stable_stationary_certified


def test_extra_zero_directions_remain_unavailable_without_instability_claim():
    quarter_turn = pi_interval().midpoint / 2
    report = owner._certify_organized_stationary(
        phase_center=(0, quarter_turn, 2 * quarter_turn, 3 * quarter_turn),
        edges=((0, 1), (1, 2), (2, 3), (3, 0)),
        radius=Q(1, 10**7),
        inverse=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    # The exact pi/2-twisted C4 is stationary and has zero phase Hessian.
    assert max(value.abs_max for value in report.phase_current_bounds) < Q(1, 10**30)
    assert not report.relative_hessian_positive
    assert not report.root_inclusion_certified
    assert (
        "failed_positivity_is_unavailable_not_an_instability_certificate"
        in report.scope
    )


def test_represented_primitives_are_normalized_before_computation():
    report = _pair(phase_center=(0.0, 0.001), radius=0.01, inverse=((1.0,),))
    assert report.phase_center == (Q(0), Q.from_float(0.001))
    assert report.radius == Q.from_float(0.01)
    assert report.inverse == ((Q(1),),)
    assert report.stable_stationary_certified


def test_lower_weights_are_outward_coarsened_not_rounded_to_nearest():
    report = _pair(radius=Q(1, 7))
    raw = report.edge_hessian_bounds[0].lo - report.radius
    assert report.lower_edge_hessian[0] <= raw
    assert raw - report.lower_edge_hessian[0] < Q(1, 2**32)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), "0"])
def test_invalid_physical_scalars_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _pair(phase_center=(0, value))
    with pytest.raises((TypeError, ValueError)):
        _pair(radius=value)
    with pytest.raises((TypeError, ValueError)):
        _pair(inverse=((value,),))


@pytest.mark.parametrize("radius", [0, Q(-1), Q(-1, 10**100)])
def test_radius_must_be_strictly_positive(radius):
    with pytest.raises(ValueError, match="strictly positive"):
        _pair(radius=radius)


@pytest.mark.parametrize("center", [(0,), (0,) * 55, repeat(0)])
def test_dimension_budget_bounds_even_unending_inputs(center):
    with pytest.raises(ValueError, match="2..54"):
        _pair(phase_center=center)


def test_gauge_is_an_exact_primitive_condition():
    with pytest.raises(ValueError, match="first phase center"):
        _pair(phase_center=(Q(1, 10**100), 0))


@pytest.mark.parametrize(
    "edges",
    [(), ((0, 0),), ((0, 2),), ((-1, 1),), ((0, 1, 2),), ((0,),)],
)
def test_malformed_or_disconnected_support_is_rejected(edges):
    with pytest.raises(ValueError):
        _pair(edges=edges)


@pytest.mark.parametrize("edges", [((True, 1),), ((0, 1.0),), {(0, 1)}, "01"])
def test_support_does_not_coerce_endpoint_or_order_types(edges):
    with pytest.raises(TypeError):
        _pair(edges=edges)


def test_reversed_duplicate_edges_are_not_a_different_support():
    with pytest.raises(ValueError, match="repeat"):
        owner._certify_organized_stationary(
            phase_center=(0, 0, 0),
            edges=((0, 1), (1, 0)),
            radius=Q(1, 100),
            inverse=((1, 0), (0, 1)),
        )


def test_disconnected_ungauged_components_are_rejected():
    with pytest.raises(ValueError, match="connected"):
        owner._certify_organized_stationary(
            phase_center=(0, 0, 0),
            edges=((0, 1),),
            radius=Q(1, 100),
            inverse=((1, 0), (0, 1)),
        )


@pytest.mark.parametrize("inverse", [(), ((1,), (1,)), ((1, 1),), repeat((1,))])
def test_inverse_dimensions_are_admitted_before_products(inverse):
    with pytest.raises(ValueError):
        _pair(inverse=inverse)


def test_certificate_uses_fresh_shared_fields_and_does_not_trust_cached_verdicts(
    monkeypatch,
):
    kernel = owner._bounded_sine_edge_fields

    def changed_residual(**kwargs):
        currents, potentials, edge_currents = kernel(**kwargs)
        return (I(1), I(-1)), potentials, edge_currents

    monkeypatch.setattr(owner, "_bounded_sine_edge_fields", changed_residual)
    report = _pair()
    assert report.eta >= 1
    assert report.relative_hessian_positive
    assert not report.root_inclusion_certified
    assert not report.stable_stationary_certified


def test_certificate_keeps_positivity_failure_in_combined_result(monkeypatch):
    monkeypatch.setattr(
        owner, "exact_symmetric_semidefinite", lambda *args, **kwargs: False
    )
    report = _pair()
    assert report.root_inclusion_certified
    assert not report.relative_hessian_positive
    assert not report.stable_stationary_certified
