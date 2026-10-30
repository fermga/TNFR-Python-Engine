"""Independent affine-output controls on a declared exact P3 nodal model.

No auxiliary generator, historical artifact, solver or phase law is supplied.
The output offsets are observations, not extra forces or state coordinates.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

from tests.physics.test_forced_epi_closure import _apply, _identity, _product
from tests.physics.test_forced_epi_realization import _assert_realization
from tnfr.physics import epi_memory as owner
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.support_transport import _from_data

A = (
    (F(1), F(-1), F(0)),
    (F(-1, 2), F(1), F(-1, 2)),
    (F(0), F(-1), F(1)),
)


def _reference(*, forcing=(1, 0, -1)):
    source = _from_data(
        (0, 1, 2),
        ((0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 1, 1)),
        ((1,), (0, 2), (1,)),
        (3, 1, -1),
        (1, 1, 1),
        (77, 77, 77),
    )
    return derive_forced_support_balance(source, epi_weight=1, forcing=forcing)


def _observe(output=((1, 0, 0),), **kwargs):
    return owner.observe_affine_nodal_realization(_reference(), output, **kwargs)


def _affine_rate(state, source):
    return tuple(b - a for b, a in zip(source, _apply(A, state), strict=True))


def _assert_all_state_identities(result):
    """Matrix equalities establish every-state behavior, not one fitted sample."""
    c, t = result.observation, result.right_inverse
    assert _product(c, t) == _identity(result.dimension)
    assert _product(c, A) == _product(result.reduced_generator, c)
    assert _product(result.output_map, c) == result.output_rows
    assert result.reduced_source == _apply(c, result.reference.forcing)
    assert result.reduced_state == _apply(c, result.epi)
    assert result.reduced_rate == _apply(
        c, _affine_rate(result.epi, result.reference.forcing)
    )
    assert result.output_state == tuple(
        observed + offset
        for observed, offset in zip(
            _apply(result.output_rows, result.epi),
            result.output_offset,
            strict=True,
        )
    )
    assert result.output_rate == _apply(
        result.output_rows, _affine_rate(result.epi, result.reference.forcing)
    )
    assert result.reconstructed_output_state == result.output_state
    assert result.reconstructed_output_rate == result.output_rate


def test_endpoint_output_needs_the_final_n_plus_one_stabilization_level():
    reference = _reference(forcing=(F(1, 4), F(-1, 2), F(3, 4)))
    result = owner.observe_affine_nodal_realization(
        reference,
        ((1, 0, 0),),
        output_offset=(7,),
        epi=(0, 1, 5),
    )
    assert result.output_rank == result.output_count == 1
    assert result.dimension == result.full_state_dimension == 3
    assert result.extra_coordinates == 2
    assert result.observation == (
        (1, 0, 0),
        (1, -1, 0),
        (F(3, 2), -2, F(1, 2)),
    )
    assert result.rank_progression == (1, 2, 3, 3)
    assert result.completed_levels == 4 and result.stabilization_power == 3
    assert result.selected_row_labels == ((0, 0), (1, 0), (2, 0))
    assert result.pivot_columns == (0, 1, 2)
    assert (
        result.level_records[-1].rank_before == result.level_records[-1].rank_after == 3
    )
    assert result.output_state == (7,) and result.output_rate == (F(5, 4),)
    _assert_all_state_identities(result)


def test_leading_zero_and_redundant_rows_never_terminate_a_krylov_level():
    result = _observe(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    assert result.output_count == 3 and result.output_rank == 1
    assert result.dimension == 3 and result.extra_coordinates == 2
    assert result.rank_progression == (1, 2, 3, 3)
    assert result.selected_row_labels == ((0, 1), (1, 1), (2, 1))
    assert all(level.candidate_rows == 3 for level in result.level_records)
    assert tuple(level.selected_row_indices for level in result.level_records) == (
        (1,),
        (1,),
        (1,),
        (),
    )
    assert result.output_state == (0, 3, 6)
    assert result.output_rate == (0, -1, -2)
    _assert_all_state_identities(result)


def test_signed_redundant_observations_keep_independent_constant_offsets():
    rows = ((1, 0, -1), (2, 0, -2), (0, 0, 0))
    result = _observe(rows, output_offset=(5, -7, 13))
    assert result.output_count == 3 and result.output_rank == result.dimension == 1
    assert result.extra_coordinates == 0
    assert result.observation == ((1, 0, -1),)
    assert result.reduced_generator == ((1,),)
    assert result.reduced_source == (2,)
    assert result.reduced_state == (4,) and result.reduced_rate == (-2,)
    assert result.output_map == ((1,), (2,), (0,))
    assert result.output_state == (9, 1, 13)
    assert result.output_rate == (-2, -4, 0)
    _assert_all_state_identities(result)


def test_offset_is_not_a_hidden_state_or_an_added_nodal_source():
    rows, offset = _identity(3), (F(5), F(-2), F(7))
    reference = _reference(forcing=(2, 3, -1))
    plain = owner.observe_affine_nodal_realization(reference, rows)
    shifted = owner.observe_affine_nodal_realization(
        reference, rows, output_offset=offset
    )
    assert shifted.dimension == 3  # No appended homogeneous coordinate.
    for field in (
        "observation",
        "right_inverse",
        "reduced_generator",
        "reduced_source",
        "reduced_state",
        "reduced_rate",
        "output_map",
        "output_rate",
        "rank_progression",
    ):
        assert getattr(shifted, field) == getattr(plain, field)
    assert (
        tuple(a - b for a, b in zip(shifted.output_state, plain.output_state)) == offset
    )
    # With y=x+offset, ydot=-A*y+(b+A*offset); its observed source cannot be b alone.
    observed_source = tuple(b + a for b, a in zip(reference.forcing, _apply(A, offset)))
    assert shifted.output_rate == tuple(
        b - a for b, a in zip(observed_source, _apply(A, shifted.output_state))
    )
    assert shifted.output_rate != _affine_rate(shifted.output_state, reference.forcing)
    _assert_all_state_identities(shifted)


def test_zero_output_rank_retains_constants_despite_nonzero_fine_source_motion():
    reference = _reference(forcing=(1, 2, 3))
    result = owner.observe_affine_nodal_realization(
        reference,
        ((0, 0, 0), (0, 0, 0)),
        output_offset=(F(7, 3), -2),
    )
    assert any(_affine_rate(result.epi, reference.forcing))
    assert result.output_count == 2 and result.output_rank == result.dimension == 0
    assert result.extra_coordinates == 0
    assert result.observation == result.reduced_generator == ()
    assert result.right_inverse == ((), (), ())
    assert result.output_map == ((), ())
    assert result.reduced_source == result.reduced_state == result.reduced_rate == ()
    assert result.output_state == result.reconstructed_output_state == (F(7, 3), -2)
    assert result.output_rate == result.reconstructed_output_rate == (0, 0)


def test_small_exact_output_independence_is_not_discarded_by_numeric_rank():
    epsilon = F(1, 2**100)
    rows = ((1, 0, 0), (1, epsilon, 0))
    result = _observe(rows)
    assert result.output_rank == 2 and result.dimension == 3
    assert result.extra_coordinates == 1
    assert result.observation[:2] == rows
    assert result.selected_row_labels[:2] == ((0, 0), (0, 1))
    _assert_all_state_identities(result)


def test_same_generic_owner_preserves_the_historical_partition_wrapper(monkeypatch):
    reference, blocks = _reference(), ((0,), (1, 2))
    core = owner._affine_nodal_realization_core
    calls = []

    def record(*args, **kwargs):
        result = core(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(owner, "_affine_nodal_realization_core", record)
    wrapped = owner.observe_forced_support_realization(reference, blocks)
    assert len(calls) == 1
    direct = calls[0]
    assert direct.output_rows == wrapped.closure.projection
    assert direct.output_offset == (0, 0)
    for field in (
        "dimension",
        "extra_coordinates",
        "observation",
        "right_inverse",
        "reduced_generator",
        "output_map",
        "reduced_source",
        "reduced_state",
        "reduced_rate",
        "rank_progression",
        "rank_calls",
        "selected_row_labels",
        "pivot_columns",
        "completed_levels",
        "stabilization_power",
    ):
        assert getattr(wrapped, field) == getattr(direct, field)
    assert wrapped.projected_state == direct.output_state
    assert wrapped.projected_rate == direct.output_rate
    public = owner.observe_affine_nodal_realization(reference, direct.output_rows)
    assert len(calls) == 2 and public == direct
    _assert_realization(wrapped, reference, blocks)


def test_rank_budget_exhaustion_rejects_partial_result_and_exact_budget_succeeds():
    complete = _observe()
    for maximum in (1, complete.rank_calls - 1):
        with pytest.raises(ValueError, match="budget"):
            _observe(max_rank_calls=maximum)
    limited = _observe(max_rank_calls=complete.rank_calls)
    assert limited.observation == complete.observation
    assert limited.rank_calls == complete.rank_calls


@pytest.mark.parametrize("maximum", (True, False, 0, -1, 1.5, "20", None, F(20)))
def test_rank_budget_requires_a_positive_nonboolean_integer(maximum):
    with pytest.raises((TypeError, ValueError)):
        _observe(max_rank_calls=maximum)


@pytest.mark.parametrize(
    "rows",
    (
        (),
        ((),),
        ((1, 0),),
        ((1, 0, 0), (1, 0)),
        ((1, 0, 0, 0),),
        ((True, 0, 0),),
        ((float("nan"), 0, 0),),
        ((float("inf"), 0, 0),),
        {"output": (1, 0, 0)},
        {(1, 0, 0)},
        (1, 0, 0),
    ),
)
def test_invalid_output_matrix_cannot_seed_a_realization(rows):
    with pytest.raises((TypeError, ValueError)):
        _observe(rows)


@pytest.mark.parametrize(
    "offset", ((), (1, 2), (True,), (float("nan"),), (float("inf"),), {1})
)
def test_invalid_output_offset_cannot_be_used_to_complete_an_observation(offset):
    with pytest.raises((TypeError, ValueError)):
        _observe(output_offset=offset)


@pytest.mark.parametrize("epi", ((0,), (True, 0, 0), (0, float("inf"), 0), {0, 1, 2}))
def test_invalid_fine_state_is_rejected_before_reporting_outputs(epi):
    with pytest.raises((TypeError, ValueError)):
        _observe(epi=epi)


def test_one_shot_observations_are_detached_and_stored_pressure_is_not_the_source():
    reference = _reference()
    rows, offset, epi = [[1, 0, -1]], [5], [3, 1, -1]
    saved = deepcopy((rows, offset, epi))
    result = owner.observe_affine_nodal_realization(
        reference,
        (iter(row) for row in rows),
        output_offset=iter(offset),
        epi=iter(epi),
    )
    assert (rows, offset, epi) == saved
    altered_pressure = replace(
        reference, source=replace(reference.source, stored_pressure=(999,) * 3)
    )
    second = owner.observe_affine_nodal_realization(
        altered_pressure, rows, output_offset=offset, epi=epi
    )
    assert second.reduced_source == result.reduced_source
    assert second.reduced_rate == result.reduced_rate
    rows[0][0], offset[0], epi[0] = 9, 9, 9
    assert result.output_rows == ((1, 0, -1),)
    assert result.output_offset == (5,) and result.epi == (3, 1, -1)
    with pytest.raises(FrozenInstanceError):
        result.output_offset = (0,)
