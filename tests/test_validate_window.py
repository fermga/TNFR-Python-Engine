"""Tests for validate_window parameter handling."""

import pytest
import numpy as np

from tnfr.validators import validate_window


@pytest.mark.parametrize("value", [True, False])
def test_validate_window_rejects_bool(value):
    with pytest.raises(TypeError):
        validate_window(value)


@pytest.mark.parametrize("value", [np.int32(0), np.int64(3)])
def test_validate_window_accepts_numpy_int(value):
    assert validate_window(value) == int(value)
