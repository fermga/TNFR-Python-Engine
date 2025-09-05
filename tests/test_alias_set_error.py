"""Pruebas de alias_set para conversiones nulas."""

import pytest
from tnfr.helpers import alias_set, _validate_aliases


def test_alias_set_raises_value_error_on_none():
    """alias_set debe lanzar ValueError si la conversión produce None."""
    aliases = _validate_aliases(("x",))
    with pytest.raises(ValueError):
        alias_set({}, aliases, lambda v: None, 123)
