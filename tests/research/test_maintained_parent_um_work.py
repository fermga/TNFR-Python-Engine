"""Independent elementary controls for full UM work and old marked geometry."""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research.maintained_parent_um_work import (
    _bound_state,
    _old_radius,
    _storage_work,
    _widen,
)


def test_old_edge_phase_work_survives_even_when_old_form_work_is_zero():
    # Flipping one old phase costs two; adding a same-phase edge costs one
    # from its form difference. Counting only the new edge misses most work.
    pi = pi_interval()
    work = _storage_work(
        (Q(0), Q(1), Q(3)),
        (I(0), I(0), pi),
        (I(0), pi, pi),
        ((0, 1),),
        ((1, 2),),
        (I(Q(1, 2)),),
    )
    assert work.old_phase_total.contains(2)
    assert work.new_form_total == I(1)
    assert work.new_phase_total.contains(0)
    assert work.total.contains(3)
    assert work.total.lo > work.new_form_total.hi + work.new_phase_total.hi


def test_phase_relaxation_can_supply_negative_old_edge_work():
    # Edge storage is not monotone under every arbitrary phase reset.
    pi = pi_interval()
    work = _storage_work(
        (Q(1), Q(1)),
        (I(0), pi),
        (I(0), I(0)),
        ((0, 1),),
        (),
        (),
    )
    assert work.total.contains(-2)
    assert work.total.hi < 0


def test_fresh_rows_use_bare_phase_and_weighted_form_with_actual_capacity():
    pi = pi_interval()
    state = _bound_state(
        (Q(0), Q(1)),
        (I(0), pi / 2),
        (I(2), I(3)),
        ((0, 1),),
        (I(4),),
    )
    assert state.degrees == (1, 1)
    assert (state.pressure[0] * pi).contains(1)
    assert (state.pressure[1] * pi).contains(-1)
    assert (state.form_rate[0] * pi).contains(2)
    assert (state.form_rate[1] * pi).contains(-3)
    assert (state.phase_rate[0] * pi).contains(-8)
    assert (state.phase_rate[1] * pi).contains(12)
    assert state.form_mean.contains(Q(2, 5))
    assert (state.phase_mean / pi).contains(Q(1, 5))
    assert state.positive_mean_metric


def test_isolate_has_frozen_rows_and_no_positive_global_mean_metric():
    state = _bound_state(
        (Q(0), Q(1), Q(100)),
        (I(0), I(0), I(100)),
        (I(1), I(1), I(9)),
        ((0, 1),),
        (I(1),),
    )
    assert state.degrees == (1, 1, 0)
    assert state.pressure[-1] == state.form_rate[-1] == state.phase_rate[-1] == I(0)
    assert state.mean_weights[-1] == I(0)
    assert state.form_mean == I(Q(1, 2))
    assert not state.positive_mean_metric


@pytest.mark.parametrize("bad", (I(0), I(-1), I(-1, 1)))
def test_fresh_rows_reject_nonpositive_or_unresolved_capacity(bad):
    with pytest.raises(ValueError, match="positive capacity"):
        _bound_state((Q(0), Q(1)), (I(0), I(0)), (I(1), bad), ((0, 1),), (I(1),))


@pytest.mark.parametrize("bad", (I(0), I(-1), I(-1, 1)))
def test_fresh_rows_reject_nonpositive_or_unresolved_conductance(bad):
    with pytest.raises(ValueError, match="positive conductance"):
        _bound_state((Q(0), Q(1)), (I(0), I(0)), (I(1), I(1)), ((0, 1),), (bad,))


def _reference():
    return tuple((index % 10 // 2) * 2 * pi_interval() / 5 for index in range(20))


def test_marked_radius_discards_only_common_origins():
    phases = tuple(value + 7 for value in _reference())
    radius = _old_radius(tuple(Q(5) for _ in range(20)), phases)
    assert radius.contains(0)
    # Squaring rounds back to the declared 128-bit outward grid.
    assert radius.hi < Q(1, 10**30)


def test_single_marked_phase_change_has_known_centered_square():
    phases = list(_reference())
    phases[3] += 1
    radius = _old_radius(tuple(Q(5) for _ in range(20)), tuple(phases))
    # P e_3 has norm squared 1 - 1/20, independent of the common phase origin.
    assert radius.contains(Q(19, 20))
    assert radius.width < Q(1, 10**30)
    assert radius.lo > Q(1, 256)


def test_old_identity_needs_every_original_marked_coordinate():
    with pytest.raises(ValueError, match="twenty marked"):
        _old_radius((Q(0),), (I(0),))


def test_prospective_error_allowance_widens_both_sides_without_recentring():
    value, allowance = I(Q(1, 8), Q(1, 4)), Q(1, 2**40)
    result = _widen(value, allowance)
    assert result.lo == value.lo - allowance
    assert result.hi == value.hi + allowance
