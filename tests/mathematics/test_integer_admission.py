"""Integer admission retains exact values and rejects coercive substitutes."""

from fractions import Fraction

import numpy as np
import pytest

from tnfr.mathematics import number_theory
from tnfr.mathematics._integer_admission import _integer_argument


class _IndexValue:
    def __index__(self):
        return 17


class _IntOnly:
    def __int__(self):
        return 17


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (-17, -17),
        (2**2048 + 1, 2**2048 + 1),
        (np.int64(-17), -17),
        (np.uint64(2**64 - 1), 2**64 - 1),
        (np.array(17, dtype=np.int64), 17),
        (_IndexValue(), 17),
    ],
)
def test_integer_admission_preserves_index_values(value, expected):
    admitted = _integer_argument(value, "modulus")
    assert admitted == expected
    assert type(admitted) is int


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_integer_admission_rejects_logical_values(value):
    with pytest.raises(TypeError, match="^modulus must be an integer, not boolean$"):
        _integer_argument(value, "modulus")


@pytest.mark.parametrize(
    "value",
    [17.0, 17.5, np.float64(17), Fraction(17), "17", 17 + 0j, None, _IntOnly()],
)
def test_integer_admission_rejects_truncation_and_coercion(value):
    with pytest.raises(
        TypeError, match="^modulus must be an integer, not a truncated value$"
    ):
        _integer_argument(value, "modulus")


def test_number_theory_retains_the_shared_reader_alias():
    assert number_theory._integer_argument is _integer_argument
