"""Physics adapters preserve represented-real admission and series ownership."""

import math
from fractions import Fraction

import numpy as np
import pytest

from tnfr.physics._helpers import finite_real_scalar, finite_real_series
from tnfr.physics.life import compute_self_generation


@pytest.mark.parametrize("value", [Fraction(1, 10**400), Fraction(-1, 10**400)])
def test_scalar_and_series_reject_nonzero_materialization_loss(value):
    with pytest.raises(ValueError, match="underflows"):
        finite_real_scalar(value, "rate")
    with pytest.raises(ValueError, match="item 1 is invalid") as caught:
        finite_real_series([0.0, value], "rates")
    assert "underflows" in str(caught.value.__cause__)


@pytest.mark.parametrize("parameter", ["gamma", "epi_series"])
def test_logistic_diagnostic_cannot_turn_a_nonzero_input_into_zero(parameter):
    values = {"epi_series": [0.5], "gamma": 1.0, "epi_max": 1.0}
    values[parameter] = (
        [Fraction(1, 10**400)] if parameter == "epi_series" else Fraction(1, 10**400)
    )
    with pytest.raises(ValueError, match="finite real"):
        compute_self_generation(**values)


@pytest.mark.parametrize("zero", [0.0, -0.0, np.float64(-0.0)])
def test_scalar_and_both_series_paths_preserve_signed_zero(zero):
    assert math.copysign(1.0, finite_real_scalar(zero, "value")) == math.copysign(
        1.0, zero
    )
    for source in ([zero], np.full(16, zero)):
        result = finite_real_series(source, "values", nonnegative=True)
        assert np.all(result == 0.0)
        assert np.all(np.signbit(result) == np.signbit(zero))


@pytest.mark.parametrize(
    "value", [Fraction(1, 8), np.int64(7), np.nextafter(0.0, 1.0), np.finfo(float).max]
)
def test_representable_nonzero_scalars_survive_admission(value):
    assert finite_real_scalar(value, "value") == float(value)
    assert finite_real_series([value], "values")[0] == float(value)


@pytest.mark.parametrize(
    "invalid", [True, np.bool_(False), "1", 1 + 0j, math.inf, math.nan, 10**400]
)
def test_shared_scalar_adapter_retains_value_error_policy(invalid):
    with pytest.raises(ValueError, match="value must be a finite real scalar"):
        finite_real_scalar(invalid, "value")


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("size", [3, 16])
def test_numeric_series_preserve_strided_values_and_detached_writable_storage(
    dtype, size
):
    source = np.arange(2 * size, dtype=dtype)[::-2]
    expected = np.array([float(value) for value in source])
    source.setflags(write=False)
    result = finite_real_series(source, "values", nonnegative=True, nonempty=True)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    assert result.flags.writeable and not np.shares_memory(result, source)
    result[:] = 99
    np.testing.assert_array_equal(source, expected)


def test_numeric_series_fallback_preserves_first_invalid_index_before_sign_check():
    source = np.arange(16, dtype=float)
    source[0], source[7], source[12] = -1.0, np.inf, np.nan
    with pytest.raises(ValueError, match="item 7 is invalid"):
        finite_real_series(source, "values", nonnegative=True)


@pytest.mark.parametrize("source", [[], np.array([]), np.empty((0, 2)), [[1.0]]])
def test_series_shape_and_nonempty_checks_precede_component_admission(source):
    message = "must not be empty" if np.ndim(source) == 1 else "one-dimensional"
    with pytest.raises(ValueError, match=message):
        finite_real_series(source, "values", nonempty=True)


def test_numeric_series_reject_negative_values_after_complete_finite_admission():
    values = np.arange(16, dtype=float)
    values[-1] = -np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="nonnegative magnitudes"):
        finite_real_series(values, "values", nonnegative=True)


def test_series_subclass_retains_the_generic_array_route():
    class ArraySubclass(np.ndarray):
        pass

    source = np.arange(16, dtype=float).view(ArraySubclass)
    result = finite_real_series(source, "values")
    np.testing.assert_array_equal(result, np.arange(16, dtype=float))
    assert type(result) is np.ndarray and not np.shares_memory(result, source)


@pytest.mark.parametrize("container", [list, tuple])
def test_builtin_float_sequence_retains_range_sign_and_owned_storage(container):
    source = container(
        [0.0, -0.0, 5e-324, -5e-324, 1.7976931348623157e308, 0.125, -3.5, 7.0]
    )
    result = finite_real_series(source, "samples")
    expected = np.array([finite_real_scalar(value, "sample") for value in source])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    result[:] = 42
    assert source[4] == 1.7976931348623157e308
    assert result.flags.writeable


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_sequences_preserve_original_binary_components(container, dtype):
    source = container(
        dtype(value)
        for value in [
            0.0,
            -0.0,
            np.finfo(dtype).smallest_subnormal,
            np.finfo(dtype).max,
            0.125,
            -3.5,
            7.0,
            1.0,
        ]
    )
    result = finite_real_series(source, "samples")
    expected = np.array([finite_real_scalar(value, "sample") for value in source])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize(
    "invalid,reason",
    [
        (math.inf, "invalid"),
        (math.nan, "invalid"),
        (True, "boolean"),
        ("1.0", "textual"),
        (Fraction(1, 10**400), "invalid"),
    ],
)
def test_sequence_admission_keeps_first_bad_component_before_sign_policy(
    container, invalid, reason
):
    values = [-1.0, 2.0, invalid, 4.0, 5.0, math.nan, 7.0, 8.0]
    with pytest.raises(ValueError, match=f"item 2 is {reason}"):
        finite_real_series(container(values), "samples", nonnegative=True)


def test_float_subclass_conversion_still_uses_original_scalar_admission():
    conversions = []

    class ObservedFloat(float):
        def __float__(self):
            conversions.append(self)
            return super().__float__()

    value = ObservedFloat(0.125)
    source = [1.0] * 7 + [value]
    result = finite_real_series(source, "samples")
    assert conversions == [value]
    assert result[-1] == 0.125
