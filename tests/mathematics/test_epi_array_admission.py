"""BEPI array admission retains scalar semantics, ownership and custom hooks."""

from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics.epi import BEPIElement
from tnfr.mathematics.spaces import BanachSpaceEPI


@pytest.mark.parametrize("size", [7, 8, 33])
@pytest.mark.parametrize(
    "dtype", [np.int16, np.uint64, np.float32, np.float64, np.complex64, np.complex128]
)
def test_numeric_admission_owns_writable_components_and_grid(size, dtype):
    continuous = np.arange(size, dtype=dtype)
    if continuous.dtype.kind == "c":
        continuous.imag = np.linspace(-1, 1, size)
    discrete = continuous[::-1]
    grid = np.arange(size, dtype=float)
    expected_continuous = continuous.astype(np.complex128)
    expected_discrete = discrete.astype(np.complex128)
    expected_grid = grid.copy()

    element = BanachSpaceEPI().element(continuous, discrete, x_grid=grid)

    for result, source in (
        (element.f_continuous, continuous),
        (element.a_discrete, discrete),
        (element.x_grid, grid),
    ):
        assert result.flags.writeable
        assert not np.shares_memory(result, source)
    continuous[:] = 99
    grid[:] = -1
    np.testing.assert_array_equal(element.f_continuous, expected_continuous)
    np.testing.assert_array_equal(element.a_discrete, expected_discrete)
    np.testing.assert_array_equal(element.x_grid, expected_grid)
    element.f_continuous[0] = -3
    assert continuous[0] == 99


@pytest.mark.parametrize("size", [7, 8])
def test_numeric_arrays_preserve_canonical_zero_channels(size):
    component = np.zeros(size, dtype=np.complex128)
    component.real[:] = -0.0
    component.imag[:] = -0.0
    grid = np.arange(size, dtype=float)
    grid[0] = -0.0
    element = BEPIElement(component, component, grid)

    assert not np.signbit(element.f_continuous.real).any()
    assert not np.signbit(element.f_continuous.imag).any()
    assert not np.signbit(element.a_discrete.real).any()
    assert not np.signbit(element.a_discrete.imag).any()
    assert not np.signbit(element.x_grid).any()
    assert np.signbit(component.real).all()
    assert np.signbit(component.imag).all()
    assert np.signbit(grid[0])


@pytest.mark.parametrize("channel", ["continuous", "discrete", "grid"])
@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_numeric_nonfinite_components_retain_the_original_error(channel, bad):
    values = {
        "continuous": np.ones(8, dtype=np.complex128),
        "discrete": np.ones(8, dtype=np.complex128),
        "grid": np.arange(8, dtype=float),
    }
    values[channel][4] = bad
    message = (
        "x_grid must contain finite representable real scalars"
        if channel == "grid"
        else "Inputs must contain finite representable real or complex scalars"
    )
    with pytest.raises(TNFRValueError, match=message):
        BEPIElement(values["continuous"], values["discrete"], values["grid"])


@pytest.mark.parametrize("bad", [True, "1", Fraction(1, 10**400), Fraction(10**400)])
def test_large_object_arrays_still_admit_each_original_scalar(bad):
    values = np.ones(8, dtype=object)
    values[3] = bad
    with pytest.raises(
        TNFRValueError, match="Inputs must contain finite representable"
    ):
        BEPIElement(values, np.ones(8), np.arange(8))
    with pytest.raises(
        TNFRValueError, match="x_grid must contain finite representable"
    ):
        BEPIElement(np.ones(8), np.ones(8), values)


def test_masked_array_subclass_does_not_silently_discard_its_mask():
    values = np.ma.array(
        np.ones(8), mask=[False, False, True, False, False, False, False, False]
    )
    with pytest.raises(
        TNFRValueError, match="Inputs must contain finite representable"
    ):
        BEPIElement(values, np.ones(8), np.arange(8))


@pytest.mark.parametrize("dtype", [bool, str, complex])
def test_grid_fast_path_never_coerces_nonreal_coordinate_types(dtype):
    grid = np.arange(8).astype(dtype)
    with pytest.raises(
        TNFRValueError, match="x_grid must contain finite representable"
    ):
        BEPIElement(np.ones(8), np.ones(8), grid)


def test_reconstruction_readmits_mutated_stored_arrays():
    element = BEPIElement(np.ones(8), np.ones(8), np.arange(8))
    element.f_continuous[0] = np.nan
    with pytest.raises(TNFRValueError, match="finite representable"):
        BanachSpaceEPI().element(
            element.f_continuous, element.a_discrete, x_grid=element.x_grid
        )


class _PositiveStartSpace(BanachSpaceEPI):
    @classmethod
    def validate_domain(cls, f_continuous, a_discrete, x_grid=None):
        result = super().validate_domain(f_continuous, a_discrete, x_grid)
        if result[0][0].real <= 0:
            raise TNFRValueError("Custom domain requires a positive initial sample")
        return result


def test_subclass_domain_hook_still_runs_before_element_construction():
    continuous = np.ones(8)
    continuous[0] = -1
    BEPIElement(continuous, np.ones(8), np.arange(8))
    with pytest.raises(TNFRValueError, match="Custom domain"):
        _PositiveStartSpace().element(continuous, np.ones(8), x_grid=np.arange(8))


def test_instance_domain_override_is_not_bypassed():
    space = BanachSpaceEPI()

    def reject(*_args):
        raise TNFRValueError("Instance-specific domain")

    space.validate_domain = reject
    with pytest.raises(TNFRValueError, match="Instance-specific domain"):
        space.element(np.ones(8), np.ones(8), x_grid=np.arange(8))
