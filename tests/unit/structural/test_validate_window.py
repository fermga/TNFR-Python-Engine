"""Tests for validate_window parameter handling."""

import numpy as np
import pytest

from tnfr.utils import validate_window


@pytest.mark.parametrize("value", [True, False])
def test_validate_window_rejects_bool(value):
    with pytest.raises(TypeError):
        validate_window(value)


@pytest.mark.parametrize("value", [np.int32(0), np.int64(3)])
def test_validate_window_accepts_numpy_int(value):
    assert validate_window(value) == int(value)


def test_validate_window_rejects_negative_integer():
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_window(-1)


@pytest.mark.parametrize("value, raises", [(0, True), (3, False)])
def test_validate_window_positive_requires_strictly_positive(value, raises):
    if raises:
        with pytest.raises(ValueError, match="must be positive"):
            validate_window(value, positive=True)
    else:
        assert validate_window(value, positive=True) == value
