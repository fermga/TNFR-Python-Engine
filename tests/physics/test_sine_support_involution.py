"""Weighted fixed families retain bare degrees, multiplicities and fine rows."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_support import _support_comparison_from_rows
from tnfr.physics.relational_sine_symmetry import (
    _support_involution_field,
    assess_sine_support_involution,
)

PERMUTATION = (0, 2, 1, 4, 3)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (2, 3), (1, 2), (3, 4))
WEIGHTS = tuple(map(Q, (2, 2, 3, 3, 0, 0, 5, 7)))


def _source(**changes):
    arguments = dict(
        reference_model=RelationalExchangeModel(
            2, epi_weight=0, phase_weight=3, phase_domain="regular"
        ),
        nodes=tuple(range(5)),
        edges=EDGES,
        conductances=WEIGHTS,
        epi=(Q(1, 4), Q(-1, 7), Q(-1, 7), Q(2, 5), Q(2, 5)),
        phase=(Q(0), Q(1, 5), Q(1, 5), Q(-3, 10), Q(-3, 10)),
        capacity=tuple(map(Q, (1, 2, 2, 3, 3))),
    )
    return _support_comparison_from_rows(**(arguments | changes))


def _assess(source=None):
    return assess_sine_support_involution(
        _source() if source is None else source, permutation_indices=PERMUTATION
    )


def _number(value):
    return mp.mpf(value.numerator) / value.denominator


def test_exact_orbits_keep_internal_neighbors_and_zero_conductance_support():
    result = _assess()
    assert result.source_trajectory_reduction_certified
    assert result.orbits == ((0,), (1, 2), (3, 4))
    assert result.node_orbits == (0, 1, 1, 2, 2)
    assert result.multiplicities == (1, 2, 2)
    assert result.reduced_degrees == (2, 4, 3)
    assert result.bare_neighbor_counts == ((0, 2, 0), (1, 1, 2), (0, 2, 1))
    assert result.conductance_sums == ((0, 4, 0), (2, 5, 3), (0, 3, 7))
    assert result.cell_reciprocity_certified
    # Quotient adjacency alone loses both internal degree and edge multiplicity.
    assert tuple(sum(c > 0 for c in row) for row in result.bare_neighbor_counts) != (
        result.reduced_degrees
    )


def test_full_nonlinear_rows_match_independent_fine_edge_equations():
    result, flow = _support_involution_field(_assess())
    coordinates = result.source_form_coordinates + result.source_phase_coordinates
    rates = flow(tuple(map(I, coordinates)))
    source = result.source
    with mp.workdps(80):
        form, phase = tuple(map(_number, source.epi)), tuple(map(_number, source.phase))
        q, sine, degrees = [mp.mpf(0)] * 5, [mp.mpf(0)] * 5, [0] * 5
        for (i, j), weight in zip(source.edges, source.conductances):
            current = mp.sin(phase[j] - phase[i])
            gap = _number(weight) * (form[i] - form[j])
            q[i] += gap
            q[j] -= gap
            sine[i] += current
            sine[j] -= current
            degrees[i] += 1
            degrees[j] += 1
        for i in range(5):
            a = result.node_orbits[i]
            mobility = _number(source.capacity[i]) / (mp.pi * degrees[i])
            for expected, interval in (
                (mobility * sine[i], rates[a]),
                (mobility * q[i] / 2, rates[3 + a]),
            ):
                assert _number(interval.lo) <= expected <= _number(interval.hi)
    # The existing Taylor owner consumes this same field without a second law.
    jets = flow_jets(tuple(map(I, coordinates)), 2, flow)
    assert len(jets) == 6
    assert all(row[1] == rate for row, rate in zip(jets, rates))


def test_cell_storage_and_mean_weights_reconstruct_fine_quantities():
    result = _assess()
    x = result.source_form_coordinates
    m = result.multiplicities
    storage = sum(
        (
            Q(m[a]) * result.conductance_sums[a][b] * (x[a] - x[b]) ** 2 / 4
            for a in range(3)
            for b in range(3)
        ),
        Q(0),
    )
    assert storage == result.source.form_storage
    cell_weights = tuple(
        Q(count * degree) / capacity
        for count, degree, capacity in zip(
            m, result.reduced_degrees, result.reduced_capacity
        )
    )
    assert sum(w * value for w, value in zip(cell_weights, x)) == sum(
        Q(degree) * value / capacity
        for degree, value, capacity in zip(
            result.source.degrees, result.source.epi, result.source.capacity
        )
    )
    center = sum(count * value for count, value in zip(m, x)) / 5
    assert sum(count * (value - center) ** 2 for count, value in zip(m, x)) == sum(
        (value - center) ** 2 for value in result.source.epi
    )


@pytest.mark.parametrize("field", ("epi", "phase"))
def test_nonsymmetric_source_is_not_projected_into_an_invariant_family(field):
    source = _source()
    values = list(getattr(source, field))
    values[1] += Q(1, 1000)
    result = _assess(replace(source, **{field: tuple(values)}))
    assert result.family_invariance_certified
    assert not result.source_membership_certified
    assert not result.source_trajectory_reduction_certified
    assert getattr(result.source, field) == tuple(values)


@pytest.mark.parametrize("change", ("support", "weight", "capacity"))
def test_invariant_domain_rejects_changed_law_premises(change):
    source = _source()
    if change == "support":
        source = replace(source, edges=source.edges[:-1] + ((0, 3),))
    elif change == "weight":
        source = replace(source, conductances=(Q(9),) + source.conductances[1:])
    else:
        source = replace(source, capacity=(Q(1), Q(4), Q(2), Q(3), Q(3)))
    result = _assess(source)
    assert not result.family_invariance_certified
    assert not result.source_trajectory_reduction_certified
    with pytest.raises(ValueError, match="certified weighted invariant family"):
        _support_involution_field(replace(result, family_invariance_certified=True))


def test_report_consumers_rebuild_poisoned_derived_state_and_counts():
    result = _assess()
    poisoned_source = replace(
        result.source, degrees=(999,) * 5, form_gradient=(Q(999),) * 5
    )
    poisoned = replace(
        result,
        source=poisoned_source,
        bare_neighbor_counts=((999,),),
        conductance_sums=((Q(999),),),
        reduced_capacity=(Q(999),),
        family_invariance_certified=False,
    )
    rebuilt, flow = _support_involution_field(poisoned)
    assert rebuilt == result
    _, expected = _support_involution_field(result)
    coordinates = tuple(
        map(I, result.source_form_coordinates + result.source_phase_coordinates)
    )
    assert flow(coordinates) == expected(coordinates)
    assert poisoned.to_dict() == result.to_dict()


@pytest.mark.parametrize("bad", (True, float("nan"), float("inf")))
def test_invalid_primitive_cannot_hide_behind_cached_verdicts(bad):
    source = replace(_source(), capacity=(bad,) + _source().capacity[1:])
    with pytest.raises((TypeError, ValueError)):
        _assess(source)


@pytest.mark.parametrize("bad", ((True, 2, 1, 4, 3), (0, 1, 1, 3, 4), (1, 2, 0, 4, 3)))
def test_permutation_requires_an_exact_involutive_bijection(bad):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_support_involution(_source(), permutation_indices=bad)


def test_source_labels_are_associated_by_position_not_assumed_integer_nodes():
    labels = ("root", "left-a", "left-b", "right-a", "right-b")
    source = _source(
        nodes=labels, edges=tuple((labels[i], labels[j]) for i, j in EDGES)
    )
    result = _assess(source)
    assert result.source_trajectory_reduction_certified
    assert result.bare_neighbor_counts == _assess().bare_neighbor_counts


def test_field_rejects_dropped_coordinates():
    _, flow = _support_involution_field(_assess())
    with pytest.raises(ValueError, match="every reduced"):
        flow((I(0),) * 5)
