"""Tests for using alias helpers via functions and ``AliasAccessor``."""

import pytest

from tnfr.alias import AliasAccessor


def _func_get(d, aliases, *, default=None):
    return AliasAccessor(int).get(d, aliases, default=default)


def _func_set(d, aliases, value):
    return AliasAccessor(int).set(d, aliases, value)


_accessor = AliasAccessor(int)


@pytest.mark.parametrize("getter,setter", [(_func_get, _func_set), (_accessor.get, _accessor.set)])
def test_get_and_set_work_with_functions_and_object(getter, setter):
    d = {"a": "1"}
    assert getter(d, ("a", "b"), default=None) == 1
    setter(d, ("b", "c"), "2")
    assert getter(d, ("b", "c"), default=None) == 2
