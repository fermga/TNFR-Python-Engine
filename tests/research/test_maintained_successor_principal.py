"""Admission and non-verdict controls for the maintained principal reduction."""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval, sin, sqrt
from tnfr.research import maintained_successor_principal as owner


def _core(coordinates=(0,) * 6, **kwargs):
    return owner._evaluate_core(
        coordinates,
        edges=owner.SELECTED_SUPPORT,
        shifts=owner.SELECTED_SHIFTS,
        **kwargs,
    )


def _four(coordinates=(1, 0, 1, 0), **kwargs):
    return owner._evaluate_reduction(
        coordinates,
        edges=owner.SELECTED_SUPPORT,
        shifts=owner.SELECTED_SHIFTS,
        **kwargs,
    )


def test_core_keeps_nonacute_chain_data_without_inventing_a_verdict():
    candidate = _core()
    # Both uncontacted chains have three gaps of 2*pi/3 and multiplicity four.
    # Every other edge has zero gap. This is not a stationary or PSD candidate.
    assert candidate.potential.contains(36)
    assert all(value.hi < 0 for value in candidate.chain_acute_margins)
    assert candidate.core_gradient[2].contains(-2 * sqrt(I(3)))
    assert candidate.core_gradient[3].contains(2 * sqrt(I(3)))
    assert candidate.core_gradient[5].contains(-2 * sqrt(I(3)))
    for index in (2, 3, 7, 8):
        assert candidate.transverse_eigenvalues[index].contains(-2)
    assert not hasattr(candidate, "stationary")
    assert not hasattr(candidate, "minimum_certified")
    assert not hasattr(candidate, "principal_minima_excluded")


def test_aligned_midpoints_have_independent_core_and_scalar_derivatives():
    result = _four()
    # At midpoint zero both middle resultants are positive real, so their
    # phases are zero. The two motifs have different middle multiplicities.
    assert all(value.contains(0) for value in result.middle_phases)
    assert all(value.lo > 3 for value in result.resultants)
    gap = (2 * pi_interval() - 2) / 3
    expected_a = 8 * sin(I(1)) - 8 * sin(gap)
    expected_b = 12 * sin(I(1)) - 8 * sin(gap)
    assert result.gradient[0].contains(expected_a)
    assert result.gradient[2].contains(expected_b)
    assert result.gradient[1].contains(0) and result.gradient[3].contains(0)
    assert result.gradient[0].hi < 0 < result.gradient[2].lo
    assert result.core.core_gradient[1].contains(0)
    assert result.core.core_gradient[4].contains(0)
    assert result.potential.hull(result.core.potential).width < Q(1, 10**30)


def test_outward_candidate_boxes_enclose_the_point_calculation():
    point = _four()
    box = _four(radius=Q(1, 1 << 20))
    assert box.potential.contains(point.potential)
    for outer, inner in zip(box.gradient, point.gradient, strict=True):
        assert outer.contains(inner)
    for outer_row, inner_row in zip(box.hessian, point.hessian, strict=True):
        for outer, inner in zip(outer_row, inner_row, strict=True):
            assert outer.contains(inner)


@pytest.mark.parametrize("radius", (True, False, -1, float("inf"), float("nan")))
def test_invalid_candidate_radius_rejects(radius):
    with pytest.raises((TypeError, ValueError)):
        _core(radius=radius)


@pytest.mark.parametrize(
    "coordinates",
    (
        (0,) * 5,
        (True,) + (0,) * 5,
        (float("nan"),) + (0,) * 5,
        (float("inf"),) + (0,) * 5,
    ),
)
def test_invalid_core_coordinates_reject(coordinates):
    with pytest.raises((TypeError, ValueError)):
        _core(coordinates)


def test_principal_cut_rejects_instead_of_rewrapping_the_identity():
    with pytest.raises(ValueError, match="principal margin"):
        _core((4, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="principal margin"):
        _core(radius=4)


@pytest.mark.parametrize(
    "case", ("missing", "boolean_node", "branch", "boolean_branch")
)
def test_complete_typed_support_and_branches_are_premises(case):
    edges, shifts = list(owner.SELECTED_SUPPORT), list(owner.SELECTED_SHIFTS)
    if case == "missing":
        edges.pop()
    elif case == "boolean_node":
        edges[0] = (False, edges[0][1])
    elif case == "branch":
        shifts[0] += 1
    else:
        shifts[0] = False
    with pytest.raises(ValueError):
        owner._evaluate_core((0,) * 6, edges=tuple(edges), shifts=tuple(shifts))


def test_four_coordinate_minimum_domain_keeps_singular_boundaries_out():
    with pytest.raises(ValueError, match="strictly acute positive chains"):
        _four((Q(1, 2), 0, 1, 0))
    with pytest.raises(ValueError, match="resultant is zero or unresolved"):
        _four((I(Q(18, 10), Q(19, 10)), 0, 1, 0))
    with pytest.raises(ValueError, match="nonpositive real ray"):
        _four((2, 0, 1, 0))


def test_four_coordinate_map_checks_all_edges_after_middle_alignment():
    # The chain gap and two resultants are regular, but an endpoint-parent
    # contact has left the retained principal branch. No wrapping repairs it.
    with pytest.raises(ValueError, match="principal margin"):
        _four((1, 3, 1, 0))


def test_public_consumers_rebuild_the_source_before_candidate_evaluation(monkeypatch):
    calls = []
    chamber = SimpleNamespace(
        edges=owner.SELECTED_SUPPORT,
        ideal=SimpleNamespace(shifts=owner.SELECTED_SHIFTS),
    )

    def rebuild(graph):
        calls.append(graph)
        return chamber

    monkeypatch.setattr(owner, "predict_successor_chamber", rebuild)
    graph = object()
    core = owner.evaluate_successor_core(graph, (0,) * 6)
    reduced = owner.evaluate_successor_reduction(graph, (1, 0, 1, 0))
    assert calls == [graph, graph]
    assert core.chamber is chamber and reduced.chamber is chamber
    assert core.candidate.potential.contains(36)
    assert reduced.reduction.gradient[0].hi < 0
