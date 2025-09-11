"""Tests for validate_window parameter handling."""

import pytest

from tnfr.glyph_history import validate_window


@pytest.mark.parametrize("value", [True, False])
def test_validate_window_rejects_bool(value):
    with pytest.raises(TypeError):
        validate_window(value)
