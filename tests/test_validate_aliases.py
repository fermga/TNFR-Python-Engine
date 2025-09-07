import pytest

from tnfr.alias import _validate_aliases, alias_get


def test_rejects_string():
    with pytest.raises(TypeError):
        _validate_aliases("abc")


def test_rejects_empty_iterable():
    with pytest.raises(ValueError):
        _validate_aliases(())


def test_rejects_non_string_elements():
    with pytest.raises(TypeError):
        _validate_aliases(("a", 1))


def test_accepts_list_sequence():
    assert _validate_aliases(["a"]) == ("a",)


def test_alias_get_reports_all_failures():
    d = {"a": "x", "b": "y"}
    with pytest.raises(ValueError) as exc:
        alias_get(d, ("a", "b"), int, strict=True)
    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg


def test_alias_get_includes_default_failure():
    with pytest.raises(ValueError) as exc:
        alias_get({}, ("a",), int, default="x", strict=True)
    assert "default" in str(exc.value)
