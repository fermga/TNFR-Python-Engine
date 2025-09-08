import json
import pytest

from tnfr.helpers.cache import _stable_json


def test_stable_json_set_order_deterministic():
    class Obj:
        def __init__(self, v):
            self.v = v

    s = {Obj(1), Obj(2), 3, "a"}
    res1 = _stable_json(s)
    res2 = _stable_json(s)
    assert res1 == res2
    json.loads(res1)


def test_stable_json_respects_max_depth_dict():
    obj = {"a": {"b": {"c": 1}}}
    with pytest.raises(ValueError):
        _stable_json(obj, max_depth=1)


def test_stable_json_respects_max_depth_list():
    obj = [1, [2, [3]]]
    with pytest.raises(ValueError):
        _stable_json(obj, max_depth=1)


def test_stable_json_detects_recursion():
    obj = []
    obj.append(obj)
    with pytest.raises(ValueError):
        _stable_json(obj)
