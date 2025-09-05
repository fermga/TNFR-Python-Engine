"""Pruebas de validación de alias como tuplas."""

import pytest
from tnfr.helpers import alias_get, alias_set


def test_alias_get_requires_tuple():
    with pytest.raises(TypeError):
        alias_get({}, ["x"], int)


def test_alias_set_requires_tuple():
    with pytest.raises(TypeError):
        alias_set({}, ["x"], int, 1)
