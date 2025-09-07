import warnings
from tnfr.alias import alias_get, alias_set

def test_alias_get_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alias_get({}, ("x",), int)
        assert any(issubclass(i.category, DeprecationWarning) for i in w)

def test_alias_set_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alias_set({}, ("x",), int, 1)
        assert any(issubclass(i.category, DeprecationWarning) for i in w)
