import pytest

from tnfr.alias import AliasAccessor


def test_rejects_empty_iterable():
    with pytest.raises(ValueError):
        AliasAccessor(int).get({}, ())


def test_rejects_non_string_elements():
    with pytest.raises(TypeError):
        AliasAccessor(int).get({}, ("a", 1))


def test_accepts_tuple():
    acc = AliasAccessor(int)
    aliases, _, _ = acc._prepare(("a",), int)
    assert aliases == ("a",)


def test_get_attr_reports_all_failures():
    d = {"a": "x", "b": "y"}
    with pytest.raises(ValueError):
        AliasAccessor(int).get(d, ("a", "b"), strict=True)


def test_get_attr_includes_default_failure():
    with pytest.raises(ValueError):
        AliasAccessor(int).get({}, ("a",), default="x", strict=True)
