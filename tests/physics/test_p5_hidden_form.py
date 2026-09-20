"""Independent error controls for the declared fixed P5 reflection quotient.

Exact path matrices check the row constants and signed initial errors. An
independent high-precision elementary flow checks finite sampled bounds; it
does not certify runtime integration or the emergence of a maintained pattern.
"""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction as F

import pytest

from tnfr.physics.p5_hidden_form import bound_p5_hidden_form
from tnfr.physics.p5_reduction import p5_reduction_geometry, reduce_p5_state

DEGREES = (1, 2, 2, 2, 1)
LAPLACIAN = tuple(
    tuple(
        F(1) if i == j else -F(1, DEGREES[i]) if abs(i - j) == 1 else F(0)
        for j in range(5)
    )
    for i in range(5)
)
KERNEL = tuple(
    tuple(F(1, (i - j) ** 2) if i != j else F(0) for j in range(5)) for i in range(5)
)
INITIAL = (F(7, 3), F(-5, 4), F(2), F(3, 2), F(-2, 3))


def _apply(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix)


def _odd(r, s):
    return r, s, F(0), -s, -r


def _pressure(epi):
    return tuple(-value for value in _apply(LAPLACIAN, epi))


def _hidden_rows():
    pressure = tuple(zip(_pressure(_odd(F(1), F(0))), _pressure(_odd(F(0), F(1)))))
    potential = tuple(
        zip(
            _apply(KERNEL, tuple(row[0] for row in pressure)),
            _apply(KERNEL, tuple(row[1] for row in pressure)),
        )
    )
    return pressure, potential


def _decimal(value):
    value = F(value)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _hidden_flow(r, s, time, capacity):
    """Closed exponential using B**2=I/2 for -A/nu=-I+B."""
    scaled = _decimal(time) * _decimal(capacity)
    root = Decimal(2).sqrt()
    angle = scaled / root
    plus, minus = angle.exp(), (-angle).exp()
    cosh, sinh = (plus + minus) / 2, (plus - minus) / 2
    decay = (-scaled).exp()
    return (
        decay * (cosh * _decimal(r) + root * sinh * _decimal(s)),
        decay * (sinh * _decimal(r) / root + cosh * _decimal(s)),
    )


@pytest.mark.parametrize("capacity", [F(1), F(7, 4)])
def test_exact_initial_errors_and_sharp_row_constants_come_from_the_fine_path(capacity):
    bound = bound_p5_hidden_form(INITIAL, times=(0,), capacity=capacity)
    assert bound.reduced == reduce_p5_state(INITIAL)
    assert bound.geometry == p5_reduction_geometry(capacity)
    assert bound.hidden_metric == (2, 4)
    assert bound.hidden_generator == ((capacity, -capacity), (-capacity / 2, capacity))
    r = (INITIAL[0] - INITIAL[4]) / 2
    s = (INITIAL[1] - INITIAL[3]) / 2
    odd = _odd(r, s)
    even = tuple(x - z for x, z in zip(INITIAL, odd, strict=True))
    expected_pressure = tuple(
        full - retained
        for full, retained in zip(_pressure(INITIAL), _pressure(even), strict=True)
    )
    assert bound.initial_pressure_error == expected_pressure == _pressure(odd)
    assert bound.initial_potential_error == _apply(KERNEL, expected_pressure)
    assert bound.initial_hidden_energy == 2 * r * r + 4 * s * s
    assert bound.potential_geometry.kernel == KERNEL
    assert bound.potential_geometry.nodes == tuple(range(5))
    assert bound.potential_geometry.distances == tuple(
        tuple(F(abs(i - j)) for j in range(5)) for i in range(5)
    )
    pressure_rows, potential_rows = _hidden_rows()

    def dual_norms(rows):
        return tuple(a * a / 2 + b * b / 4 for a, b in rows)

    assert (
        bound.pressure_squared_factors
        == dual_norms(pressure_rows)
        == (F(3, 4), F(3, 8), 0, F(3, 8), F(3, 4))
    )
    assert (
        bound.potential_squared_factors
        == dual_norms(potential_rows)
        == (F(9809, 27648), F(2897, 3456), 0, F(2897, 3456), F(9809, 27648))
    )
    assert bound.exact_identity_checks
    assert len(set(bound.exact_identity_checks)) == len(bound.exact_identity_checks)


@pytest.mark.parametrize("kind", ["pressure", "potential"])
@pytest.mark.parametrize("node", [0, 1])
def test_each_distinct_nonzero_row_constant_is_attained_at_time_zero(kind, node):
    rows = _hidden_rows()[kind == "potential"]
    a, b = rows[node]
    # Equality in Cauchy--Schwarz uses z=H^-1 row; no fitted constant.
    bound = bound_p5_hidden_form(_odd(a / 2, b / 4), times=(0,))
    signed = getattr(bound, f"initial_{kind}_error")[node]
    upper = getattr(bound.samples[0], f"{kind}_squared_upper")[node]
    assert signed != 0
    assert signed * signed == upper


@pytest.mark.parametrize("capacity", [F(1, 2), F(7, 4)])
def test_rational_gap_is_a_positive_exact_psd_lower_bound(capacity):
    bound = bound_p5_hidden_form(INITIAL, times=(0, 1), capacity=capacity)
    rate = bound.decay_rate_lower
    assert isinstance(rate, F) and 0 < rate < capacity
    # H*A-rate*H is symmetric. Its two principal entries and determinant
    # certify the spectral lower bound without accepting a float eigensolver.
    first, second, off = 2 * (capacity - rate), 4 * (capacity - rate), -2 * capacity
    assert first > 0 and second > 0
    assert first * second - off * off >= 0
    with localcontext() as context:
        context.prec = 180
        true_gap = _decimal(capacity) * (1 - 1 / Decimal(2).sqrt())
        assert _decimal(rate) <= true_gap


@pytest.mark.parametrize("hidden", [(F(3, 2), F(-11, 8)), (F(0), F(2)), (F(2), F(0))])
@pytest.mark.parametrize("capacity", [F(1, 2), F(7, 4)])
def test_all_sampled_errors_are_enclosed_by_an_independent_high_precision_flow(
    hidden, capacity
):
    r, s = hidden
    initial = tuple(x + z for x, z in zip((1, -2, 3, -2, 1), _odd(r, s), strict=True))
    bound = bound_p5_hidden_form(initial, times=(0, F(1, 8), 2, 16), capacity=capacity)
    pressure_rows, potential_rows = _hidden_rows()
    with localcontext() as context:
        context.prec = 180
        previous = F(1)
        for sample in bound.samples:
            assert isinstance(sample.decay_factor_upper, F)
            assert 0 < sample.decay_factor_upper <= previous
            previous = sample.decay_factor_upper
            exact_factor = (
                -2 * _decimal(bound.decay_rate_lower) * _decimal(sample.time)
            ).exp()
            assert _decimal(sample.decay_factor_upper) >= exact_factor
            assert (
                sample.hidden_energy_upper
                == bound.initial_hidden_energy * sample.decay_factor_upper
            )
            rt, st = _hidden_flow(r, s, sample.time, capacity)
            energy = 2 * rt * rt + 4 * st * st
            assert energy <= _decimal(sample.hidden_energy_upper)
            for kind, rows in (
                ("pressure", pressure_rows),
                ("potential", potential_rows),
            ):
                factors = getattr(bound, f"{kind}_squared_factors")
                uppers = getattr(sample, f"{kind}_squared_upper")
                assert uppers == tuple(
                    factor * sample.hidden_energy_upper for factor in factors
                )
                for (a, b), upper in zip(rows, uppers, strict=True):
                    actual = _decimal(a) * rt + _decimal(b) * st
                    assert actual * actual <= _decimal(upper)


def test_nonzero_even_memory_contrast_has_no_discarded_reflection_error():
    initial = (1, -2, 3, -2, 1)
    bound = bound_p5_hidden_form(initial, times=(0, 1, 10))
    assert bound.reduced.memory_epi[2] == 5
    assert any(_pressure(initial))
    assert bound.initial_hidden_energy == 0
    assert bound.initial_pressure_error == bound.initial_potential_error == (0,) * 5
    for sample in bound.samples:
        assert sample.hidden_energy_upper == 0
        assert (
            sample.pressure_squared_upper == sample.potential_squared_upper == (0,) * 5
        )
        assert sample.within_tolerances(pressure=0, potential=0)


def test_capacity_time_scaling_reflection_and_translation_preserve_the_bound():
    times = (F(0), F(1, 2), F(3))
    first = bound_p5_hidden_form(INITIAL, times=times)
    fast = bound_p5_hidden_form(
        INITIAL, times=tuple(time / 3 for time in times), capacity=3
    )
    reflected = bound_p5_hidden_form(INITIAL[::-1], times=times)
    shifted = bound_p5_hidden_form(tuple(x + F(17, 9) for x in INITIAL), times=times)
    assert fast.decay_rate_lower == 3 * first.decay_rate_lower
    assert (
        fast.initial_pressure_error
        == shifted.initial_pressure_error
        == first.initial_pressure_error
    )
    assert (
        fast.initial_potential_error
        == shifted.initial_potential_error
        == first.initial_potential_error
    )
    assert reflected.initial_pressure_error == first.initial_pressure_error[::-1]
    assert reflected.initial_potential_error == first.initial_potential_error[::-1]
    assert first.samples == reflected.samples == shifted.samples
    for slow, accelerated in zip(first.samples, fast.samples, strict=True):
        assert slow.time == 3 * accelerated.time
        assert slow.decay_factor_upper == accelerated.decay_factor_upper
        assert slow.hidden_energy_upper == accelerated.hidden_energy_upper
        assert slow.pressure_squared_upper == accelerated.pressure_squared_upper
        assert slow.potential_squared_upper == accelerated.potential_squared_upper


def test_absolute_decay_does_not_make_a_pure_odd_state_relatively_reconstructed():
    bound = bound_p5_hidden_form(_odd(F(1), F(2)), times=(0, 1, 8))
    assert bound.reduced.orbit_epi == (0, 0, 0)
    assert bound.samples[-1].hidden_energy_upper < bound.initial_hidden_energy
    with localcontext() as context:
        context.prec = 180
        for sample in bound.samples:
            r, s = _hidden_flow(1, 2, sample.time, 1)
            fine = (r, s, Decimal(0), -s, -r)
            error = fine  # The exact even quotient remains identically zero.
            total = sum(degree * x * x for degree, x in zip(DEGREES, fine, strict=True))
            loss = sum(degree * x * x for degree, x in zip(DEGREES, error, strict=True))
            assert total > 0 and loss / total == 1


def test_tolerance_admission_compares_squared_bounds_inclusively_and_exactly():
    sample = bound_p5_hidden_form(_odd(F(2), F(1)), times=(0,)).samples[0]
    assert max(sample.pressure_squared_upper) == 9
    assert sample.within_tolerances(pressure=3, potential=4)
    assert not sample.within_tolerances(pressure=3 - F(1, 2**100), potential=4)
    assert not sample.within_tolerances(pressure=4, potential=3)
    with pytest.raises(TypeError):
        sample.within_tolerances(3, 4)


@pytest.mark.parametrize("name", ["pressure", "potential"])
@pytest.mark.parametrize(
    "value,error",
    [
        (-1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("1", TypeError),
        (1j, TypeError),
    ],
)
def test_tolerances_reject_malformed_values_even_for_a_zero_error(name, value, error):
    sample = bound_p5_hidden_form((0,) * 5, times=(0,)).samples[0]
    values = {"pressure": 0, "potential": 0, name: value}
    with pytest.raises(error):
        sample.within_tolerances(**values)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("initial_epi", (0,) * 4, ValueError),
        ("initial_epi", (0,) * 6, ValueError),
        ("initial_epi", (0, True, 0, 0, 0), TypeError),
        ("initial_epi", (0, float("inf"), 0, 0, 0), ValueError),
        ("initial_epi", {0, 1, 2, 3, 4}, TypeError),
        ("capacity", 0, ValueError),
        ("capacity", -1, ValueError),
        ("capacity", True, TypeError),
        ("capacity", float("nan"), ValueError),
        ("capacity", 1j, TypeError),
        ("times", (), ValueError),
        ("times", {0, 1}, TypeError),
        ("times", {0: 1}, TypeError),
        ("times", "01", TypeError),
        ("times", (True,), TypeError),
        ("times", (-1,), ValueError),
        ("times", (float("inf"),), ValueError),
        ("times", (float("nan"),), ValueError),
        ("times", (1j,), TypeError),
    ],
)
def test_invalid_model_and_sampling_inputs_fail_closed(field, value, error):
    arguments = {"initial_epi": INITIAL, "times": (0,), "capacity": 1, field: value}
    with pytest.raises(error):
        bound_p5_hidden_form(**arguments)


def test_scaled_time_resource_boundary_is_not_a_physical_capacity_ceiling():
    bound = bound_p5_hidden_form((0,) * 5, times=(0, 1), capacity=2048)
    assert bound.geometry.capacity == 2048
    assert bound.samples[1].time == 1
    with pytest.raises(ValueError, match="2048"):
        bound_p5_hidden_form(INITIAL, times=(1 + F(1, 2**100),), capacity=2048)
    with pytest.raises(ValueError, match="2048"):
        bound_p5_hidden_form(INITIAL, times=(2048 + F(1, 2**100),))


def test_ordered_one_shot_inputs_are_detached_and_results_are_frozen():
    initial, times = list(INITIAL), [0.5, 0, F(1, 3), 0.5]
    before = tuple(initial)
    bound = bound_p5_hidden_form(iter(initial), times=iter(times), capacity=1.1)
    initial[0] = 99
    times[0] = 99
    assert bound.reduced.epi == before
    assert tuple(sample.time for sample in bound.samples) == (
        F(1, 2),
        0,
        F(1, 3),
        F(1, 2),
    )
    assert bound.geometry.capacity == F.from_float(1.1)
    assert bound.samples[0] == bound.samples[-1]
    with pytest.raises(FrozenInstanceError):
        bound.initial_hidden_energy = 0
    with pytest.raises(FrozenInstanceError):
        bound.samples[0].time = 0
