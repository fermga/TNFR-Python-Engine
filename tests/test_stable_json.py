import json

from tnfr.helpers.cache import _stable_json


def test_stable_json_dict_order_deterministic():
    obj = {"b": 1, "a": 2}
    res1 = _stable_json(obj)
    res2 = _stable_json(obj)
    assert res1 == res2
    assert json.loads(res1) == {"a": 2, "b": 1}
