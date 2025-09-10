from tnfr.alias import get_attr_str


def test_get_attr_str_returns_string():
    d = {"x": 5}
    assert get_attr_str(d, ("x",), "") == "5"


def test_get_attr_str_default_used():
    d = {}
    assert get_attr_str(d, ("x",), "foo") == "foo"
