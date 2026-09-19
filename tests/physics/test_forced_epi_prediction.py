"""Independent exact Euler forecasts in a complete linear observation chart."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
from unittest.mock import patch

import pytest

from tests.physics.test_forced_epi_closure import _apply, _geometry, _identity
from tests.physics.test_forced_epi_realization import _reference

F = Fraction
BLOCKS = ((0, 1), (2, 3))


def _full_reference():
    return _reference(4, capacities=(F(1), F(2), F(3), F(4)))


def _fine_oracle(reference, blocks, steps, epi):
    geometry = _geometry(reference, blocks)
    a, b = geometry["A"], geometry["b"]
    identity = _identity(len(a))
    states, matrices, elapsed = [tuple(epi)], [], [F(0)]
    for step in steps:
        matrix = tuple(
            tuple(unit - step * value for unit, value in zip(row_i, row_a, strict=True))
            for row_i, row_a in zip(identity, a, strict=True)
        )
        matrices.append(matrix)
        states.append(
            tuple(
                value + step * force
                for value, force in zip(
                    _apply(matrix, states[-1]),
                    b,
                    strict=True,
                )
            )
        )
        elapsed.append(elapsed[-1] + step)
    rates = tuple(
        tuple(force - value for force, value in zip(b, _apply(a, state), strict=True))
        for state in states
    )
    return geometry, tuple(states), tuple(rates), tuple(matrices), tuple(elapsed)


def _assert_forecast(prediction, reference, blocks, steps, epi):
    geometry, states, rates, matrices, times = _fine_oracle(
        reference, blocks, steps, epi
    )
    realization = prediction.realization
    c, t, d = realization.observation, realization.right_inverse, realization.output_map
    assert realization.dimension == realization.full_state_dimension == len(epi)
    assert prediction.steps == steps
    assert prediction.fine_euler_matrices == matrices
    assert prediction.elapsed_time == sum(steps, F(0))
    assert prediction.convex_step_admissible == tuple(
        step <= reference.max_convex_step for step in steps
    )
    assert len(prediction.frames) == len(steps) + 1
    for ordinal, (frame, state, rate, time) in enumerate(
        zip(prediction.frames, states, rates, times, strict=True)
    ):
        assert frame.ordinal == ordinal and frame.time == time
        assert frame.epi == state and frame.fine_rate == rate
        assert frame.reduced_state == _apply(c, state)
        assert frame.reduced_rate == _apply(c, rate)
        assert _apply(t, frame.reduced_state) == state
        assert (
            frame.projected_epi
            == _apply(geometry["R"], state)
            == _apply(d, frame.reduced_state)
        )
        assert not any(frame.encoding_residual)
        assert not any(frame.rate_residual)
        assert not any(frame.output_residual)
        assert not any(frame.fine_euler_residual)


@pytest.mark.parametrize("steps", ((F(1, 7), F(1, 4)), (F(0), F(1, 4), F(0)), (F(2),)))
def test_full_coordinate_forecast_matches_independent_fine_affine_euler(steps):
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    epi = (F(2), F(-7), F(0), F(5))
    with patch(
        "tnfr.physics.epi_memory.matrix_exponential",
        side_effect=AssertionError("no exponential"),
    ):
        prediction = predict_forced_support_realization_euler(
            reference, BLOCKS, steps, epi=epi
        )
    _assert_forecast(prediction, reference, BLOCKS, steps, epi)
    assert prediction.realization.closure.reference == reference


def test_unstable_large_step_remains_an_exact_euler_formula_without_a_stability_claim():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    prediction = predict_forced_support_realization_euler(reference, BLOCKS, (F(2),))
    assert prediction.convex_step_admissible == (False,)
    assert any(value < 0 for row in prediction.fine_euler_matrices[0] for value in row)
    _assert_forecast(prediction, reference, BLOCKS, (F(2),), reference.source.epi)


def test_zero_duration_is_identity_with_unchanged_rates_and_no_hidden_time():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    prediction = predict_forced_support_realization_euler(
        reference, BLOCKS, (F(0), F(0))
    )
    assert prediction.elapsed_time == 0
    assert all(frame.epi == reference.source.epi for frame in prediction.frames)
    assert prediction.fine_euler_matrices == (_identity(4), _identity(4))
    assert prediction.convex_step_admissible == (True, True)
    assert prediction.frames[0].fine_rate == prediction.frames[-1].fine_rate


def test_forcing_and_stored_pressure_remain_distinct_and_no_mean_is_removed():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    steps = (F(1, 4), F(1, 4))
    prediction = predict_forced_support_realization_euler(reference, BLOCKS, steps)
    changed = replace(
        reference, source=replace(reference.source, stored_pressure=(F(-999),) * 4)
    )
    same = predict_forced_support_realization_euler(changed, BLOCKS, steps)
    assert same.frames == prediction.frames
    assert same.fine_euler_matrices == prediction.fine_euler_matrices
    unforced = predict_forced_support_realization_euler(
        replace(reference, forcing=(F(0),) * 4), BLOCKS, steps
    )
    assert unforced.fine_euler_matrices == prediction.fine_euler_matrices
    assert unforced.frames[-1].epi != prediction.frames[-1].epi
    metric = reference.metric_weights

    def mean(values):
        return sum(h * x for h, x in zip(metric, values, strict=True)) / sum(metric)

    assert (
        mean(prediction.frames[-1].epi) - mean(reference.source.epi)
        == sum(steps) * reference.mean_drift
    )
    assert mean(unforced.frames[-1].epi) == mean(reference.source.epi)


@pytest.mark.parametrize(
    "size,blocks", ((4, ((0, 3), (1, 2))), (5, ((0, 4), (1, 2, 3))))
)
def test_partial_observation_cannot_authorize_a_full_microstate_prediction(
    size, blocks
):
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    with pytest.raises(ValueError):
        predict_forced_support_realization_euler(_reference(size), blocks, (F(1, 4),))


@pytest.mark.parametrize(
    "steps",
    (
        (),
        (F(-1, 4),),
        (True,),
        (float("nan"),),
        (float("inf"),),
        ("1/4",),
        (1j,),
        {F(1, 4), F(1, 2)},
        {"first": F(1, 4)},
        "0.25",
        F(1, 4),
    ),
)
def test_step_sequence_requires_nonempty_ordered_finite_nonnegative_numbers(steps):
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    with pytest.raises((TypeError, ValueError)):
        predict_forced_support_realization_euler(_full_reference(), BLOCKS, steps)


@pytest.mark.parametrize("maximum", (False, True, 0, -1, 2.0, "2", None, F(2)))
def test_step_guard_requires_a_positive_builtin_integer(maximum):
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    with pytest.raises((TypeError, ValueError)):
        predict_forced_support_realization_euler(
            _full_reference(), BLOCKS, (F(1, 4),), max_steps=maximum
        )


def test_infinite_step_iterable_is_bounded_and_no_partial_forecast_is_returned():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    consumed = []

    def steps():
        while True:
            consumed.append(1)
            yield F(1, 4)

    with pytest.raises(ValueError):
        predict_forced_support_realization_euler(
            _full_reference(), BLOCKS, steps(), max_steps=2
        )
    assert len(consumed) <= 3
    assert len(consumed) > 0


def test_exact_step_limit_and_rank_budget_are_respected():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    prediction = predict_forced_support_realization_euler(
        reference, BLOCKS, (F(1, 4),) * 2, max_steps=2
    )
    assert prediction.max_steps == len(prediction.steps) == 2
    with pytest.raises(ValueError):
        predict_forced_support_realization_euler(
            reference, BLOCKS, (F(1, 4),) * 3, max_steps=2
        )
    with pytest.raises(ValueError, match="incomplete"):
        predict_forced_support_realization_euler(
            reference, BLOCKS, (F(1, 4),), max_rank_calls=1
        )


def test_reference_caches_are_rebuilt_and_invalid_coefficients_fail_closed():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    corrupted = replace(
        reference,
        metric_weights=(F(99),) * 4,
        strengths=(F(99),) * 4,
        relative_profile=(F(99),) * 4,
        mean_drift=F(99),
    )
    expected = predict_forced_support_realization_euler(reference, BLOCKS, (F(1, 4),))
    assert (
        predict_forced_support_realization_euler(corrupted, BLOCKS, (F(1, 4),))
        == expected
    )
    with pytest.raises(TypeError):
        predict_forced_support_realization_euler(asdict(reference), BLOCKS, (F(1, 4),))
    with pytest.raises(ValueError):
        predict_forced_support_realization_euler(
            replace(reference, epi_weight=F(0)), BLOCKS, (F(1, 4),)
        )


def test_caller_inputs_are_detached_and_exact_frames_are_immutable():
    from tnfr.physics.epi_memory import predict_forced_support_realization_euler

    reference = _full_reference()
    steps, blocks, epi = [F(1, 4), F(1, 2)], [[0, 1], [2, 3]], [F(7), F(-1), F(0), F(5)]
    before = deepcopy((asdict(reference), steps, blocks, epi))
    prediction = predict_forced_support_realization_euler(
        reference, blocks, steps, epi=epi
    )
    assert (asdict(reference), steps, blocks, epi) == before
    retained = asdict(prediction)
    steps[0] = F(0)
    epi[0] = F(999)
    blocks.reverse()
    assert asdict(prediction) == retained
    with pytest.raises(FrozenInstanceError):
        prediction.frames[0].epi = ()
