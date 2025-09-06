"""Pruebas de alias_set para conversiones nulas."""

import pytest
from tnfr.alias import alias_set


def test_alias_set_raises_value_error_on_none():
    """alias_set debe lanzar ValueError si la conversión produce None."""
    with pytest.raises(ValueError):
        alias_set({}, ["x"], lambda v: None, 123)
