"""Primitive admission and numerical range of finite coordinate Hilbert spaces."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import runtime
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.spaces import HilbertSpace


@pytest.mark.parametrize("dimension", [True, np.bool_(True), 2.0, "2", 0, -1])
def test_dimension_requires_a_positive_integer(dimension):
    with pytest.raises(TNFRValueError, match="positive integer"):
        HilbertSpace(dimension)


def test_numpy_integral_dimension_retains_canonical_coordinate_basis():
    space = HilbertSpace(np.int64(2))
    assert type(space.dimension) is int
    np.testing.assert_array_equal(space.basis, np.eye(2))


@pytest.mark.parametrize("dtype", [bool, int, object, str, "datetime64", "unknown"])
def test_dtype_requires_floating_or_complex_storage(dtype):
    with pytest.raises(TNFRValueError, match="dtype.*floating or complex"):
        HilbertSpace(2, dtype=dtype)


@pytest.mark.parametrize(
    "container", [list, lambda values: np.array(values, dtype=object)]
)
@pytest.mark.parametrize(
    "bad",
    [True, "1", np.inf, np.nan, Fraction(1, 10**400), 10**400],
)
def test_primitive_admission_precedes_vector_materialization(container, bad):
    space = HilbertSpace(2)
    vector = container([bad, 0])
    for operation in (
        lambda: space.norm(vector),
        lambda: space.inner_product([1, 0], vector),
        lambda: space.project(vector),
        lambda: space.project([1, 0], basis=[vector]),
    ):
        with pytest.raises(TNFRValueError):
            operation()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_dtype_accepts_real_coordinates_and_rejects_imaginary_loss(dtype):
    space = HilbertSpace(2, dtype=dtype)
    source = np.array([3 + 0j, 4 + 0j])
    projected = space.project(source)
    assert projected.dtype == np.dtype(dtype)
    assert space.norm(source) == 5.0
    assert space.inner_product(source, [1, -1]) == -1.0
    np.testing.assert_array_equal(projected, [3, 4])
    for vector in ([1j, 0], np.array([1j, 0])):
        with pytest.raises(TNFRValueError, match="imaginary"):
            space.norm(vector)
        with pytest.raises(TNFRValueError, match="imaginary"):
            space.project(vector)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
@pytest.mark.parametrize("value", [1e200, 1e-200])
def test_narrow_dtype_cannot_materialize_infinity_or_erase_nonzero_values(dtype, value):
    with pytest.raises(TNFRValueError, match="dtype"):
        HilbertSpace(2, dtype=dtype).project([value, 0])


def test_narrow_complex_dtype_checks_the_imaginary_channel_independently():
    space = HilbertSpace(2, dtype=np.complex64)
    with pytest.raises(TNFRValueError, match="nonzero coordinate"):
        space.project([complex(1, 1e-200), 0])
    np.testing.assert_array_equal(space.project([1 + 2j, 0]), [1 + 2j, 0])


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200, 1e307])
def test_norm_avoids_squared_scale_and_agrees_with_runtime(scale, monkeypatch):
    space = HilbertSpace(2)
    vector = [3 * scale, 4j * scale]
    backend = get_backend("numpy")
    monkeypatch.setattr(runtime, "get_backend", lambda: backend)
    actual = space.norm(vector)
    assert actual == pytest.approx(5 * scale, rel=2e-15, abs=0)
    assert runtime.normalized(vector, space)[1] == actual


def test_norm_preserves_zero_and_smallest_represented_nonzero_value():
    space = HilbertSpace(2)
    tiny = np.nextafter(0.0, 1.0)
    assert space.norm([0, 0]) == 0
    assert space.norm([tiny, 0]) == tiny
    with pytest.raises(TNFRValueError, match="norm.*finite float"):
        space.norm([complex(1.7e308, 1.7e308), 0])


@pytest.mark.parametrize("bad", [True, "0", -1, np.nan, np.inf, Fraction(1, 10**400)])
def test_normalization_tolerance_cannot_bypass_admission(bad):
    with pytest.raises(TNFRValueError, match="atol"):
        HilbertSpace(2).is_normalized([1, 0], atol=bad)


@pytest.mark.parametrize("exponent", [-80, 80])
def test_inner_product_accumulates_narrow_storage_in_complex128(exponent):
    space = HilbertSpace(2, dtype=np.complex64)
    scale = 2.0**exponent
    # Powers of two are exact in both configured storage and accumulation.
    assert space.inner_product([scale, 1j * scale], [scale, 1j * scale]) == (
        2.0 ** (2 * exponent + 1)
    )


def test_inner_product_retains_sesquilinearity_and_exact_zero_cancellation():
    space = HilbertSpace(2)
    assert space.inner_product([1 + 2j, 3 - 1j], [2 - 1j, -1 + 4j]) == -7 + 6j
    assert space.inner_product([2, 2], [3j, -3j]) == 0j


@pytest.mark.parametrize("right", [[1e200, 1e200], [1e200, -1e200]])
def test_nonfinite_inner_product_rejects_including_unresolved_cancellation(right):
    with pytest.raises(TNFRValueError, match="inner product.*finite"):
        HilbertSpace(2).inner_product([1e200, 1e200], right)


def test_partial_complex_orthonormal_family_uses_conjugate_projection():
    space = HilbertSpace(3)
    basis = np.array([[1, 1j, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    source = np.array([2 + 1j, 3 - 2j, -4j])
    projected = space.project(source, basis)
    assert projected.shape == (2,)
    np.testing.assert_allclose(projected, [(-2j) / np.sqrt(2), -4j])
    reconstructed = projected @ basis
    assert space.inner_product(basis[0], source - reconstructed) == pytest.approx(0j)
    assert space.inner_product(basis[1], source - reconstructed) == pytest.approx(0j)


def test_projection_returns_a_detached_array_even_for_canonical_coordinates():
    source = np.array([1 + 2j, 3 + 4j])
    projected = HilbertSpace(2).project(source)
    projected[0] = 99
    np.testing.assert_array_equal(source, [1 + 2j, 3 + 4j])


def test_projection_checks_accumulation_and_final_dtype_range():
    basis = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    with pytest.raises(TNFRValueError, match="projection coefficients.*finite"):
        HilbertSpace(2).project([1.7e308, 1.7e308], basis)
    with pytest.raises(TNFRValueError, match="dtype"):
        HilbertSpace(2, dtype=np.complex64).project([3e38, 3e38], basis)
    # A bounded coefficient can still be too small for the chosen storage.
    with pytest.raises(TNFRValueError, match="nonzero coordinate"):
        HilbertSpace(2, dtype=np.complex64).project([1e-30, 0], basis=[[1e-20, 1]])


def test_nonfinite_gram_arithmetic_cannot_validate_a_basis():
    with pytest.raises(TNFRValueError, match="not orthonormal"):
        HilbertSpace(2).project([1, 0], basis=[[1e200, 0]])
