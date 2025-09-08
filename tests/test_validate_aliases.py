import pytest

from tnfr.alias import _validate_aliases, AliasAccessor

def test_rejects_empty_iterable():
    with pytest.raises(ValueError):
        _validate_aliases(())


def test_rejects_non_string_elements():
    with pytest.raises(TypeError):
        _validate_aliases(("a", 1))


def test_accepts_tuple():
    assert _validate_aliases(("a",)) == ("a",)


def test_get_attr_reports_all_failures():
    d = {"a": "x", "b": "y"}
    with pytest.raises(ValueError) as exc:
        AliasAccessor(int).get(d, ("a", "b"), strict=True)
    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg


def test_get_attr_includes_default_failure():
    with pytest.raises(ValueError) as exc:
        AliasAccessor(int).get({}, ("a",), default="x", strict=True)
    assert "default" in str(exc.value)
