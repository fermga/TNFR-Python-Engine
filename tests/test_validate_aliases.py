import pytest

from tnfr.alias import _validate_aliases


def test_rejects_string():
    with pytest.raises(TypeError):
        _validate_aliases("abc")


def test_rejects_empty_iterable():
    with pytest.raises(ValueError):
        _validate_aliases(())


def test_rejects_non_string_elements():
    with pytest.raises(TypeError):
        _validate_aliases(("a", 1))


def test_rejects_unhashable_sequence():
    with pytest.raises(TypeError):
        _validate_aliases(["a"])
